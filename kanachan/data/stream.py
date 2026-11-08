"""Annotate eligible games while their source catalog is still being built."""

import argparse
import hashlib
import io
import json
import os
import shutil
import sqlite3
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from kanachan import constants
from kanachan.training.core import features

from .catalog import atomic_json, lower_priority, split_game
from .prepare import KINDS, SPLITS, assemble, build_shard, digest, json_digest


@contextmanager
def shared_reader(path):
    """Parse a snapshot after closing the underlying Windows file handle."""
    contents = Path(path).read_text(encoding="utf-8-sig")
    with io.StringIO(contents) as stream:
        yield stream


def read_json(path):
    with shared_reader(path) as stream:
        return json.load(stream)


def catalog_status(catalog, connection=None, *, allow_limited=False):
    """Observe WAL snapshots; never open the catalog writer's progress file."""
    catalog = Path(catalog).resolve()
    own_connection = connection is None
    if own_connection:
        connection = sqlite3.connect(
            (catalog / "catalog.sqlite3").as_uri() + "?mode=ro", uri=True
        )
    try:
        configuration = json.loads(
            connection.execute(
                "SELECT value FROM meta WHERE key='configuration'"
            ).fetchone()[0]
        )
        counts = dict(
            connection.execute(
                "SELECT conflicted,COUNT(*) FROM games GROUP BY conflicted"
            )
        )
        directories = dict(
            connection.execute(
                "SELECT done,COUNT(*) FROM directories GROUP BY done"
            )
        )
    finally:
        if own_connection:
            connection.close()
    status = "indexing"
    error = catalog / "error.json"
    progress = catalog / "progress.json"
    if error.exists() and (
        not progress.exists()
        or error.stat().st_mtime_ns >= progress.stat().st_mtime_ns
    ):
        status = "failed"
    elif (catalog / "pause.request").exists():
        status = "paused"
    elif (catalog / "summary.json").exists() and (
        allow_limited or (directories.get(1, 0) and not directories.get(0, 0))
    ):
        summary = read_json(catalog / "summary.json")
        if summary["configuration"] != configuration or sum(
            summary["splits"].values()
        ) != counts.get(0, 0):
            raise RuntimeError(
                "Published catalog summary differs from the committed game inventory."
            )
        if not all(
            (catalog / f"manifest.{split}.jsonl").is_file() for split in SPLITS
        ):
            raise RuntimeError(
                "Catalog summary was published before all split manifests."
            )
        status = "limited" if directories.get(0, 0) else "complete"
    return {
        "status": status,
        "configuration": configuration,
        "usable_games": counts.get(0, 0),
        "conflicted_games": counts.get(1, 0),
    }


class Ledger:
    def __init__(self, state, catalog, configuration):
        self.state = Path(state)
        self.state.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.state / "ledger.sqlite3")
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA cache_size=-32768;
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS games(uuid TEXT PRIMARY KEY,split TEXT NOT NULL,
                order_key TEXT NOT NULL,metadata TEXT NOT NULL,invalid INTEGER NOT NULL,batch_id INTEGER);
            CREATE INDEX IF NOT EXISTS pending ON games(split,order_key,uuid) WHERE batch_id IS NULL AND invalid=0;
            CREATE INDEX IF NOT EXISTS members ON games(batch_id);
            -- Keep queued-game counts off the large metadata rows.
            CREATE INDEX IF NOT EXISTS runtime_pending_count_v1 ON games(batch_id,invalid)
                WHERE batch_id IS NULL AND invalid=0;
            CREATE TABLE IF NOT EXISTS batches(id INTEGER PRIMARY KEY AUTOINCREMENT,split TEXT NOT NULL,
                records TEXT NOT NULL,status TEXT NOT NULL,receipt TEXT,
                attempted INTEGER NOT NULL DEFAULT 0,accepted INTEGER NOT NULL DEFAULT 0,rejected INTEGER NOT NULL DEFAULT 0,
                bc_rows INTEGER NOT NULL DEFAULT 0,rl_rows INTEGER NOT NULL DEFAULT 0,eor_rows INTEGER NOT NULL DEFAULT 0);
            -- Keep batch bookkeeping off the large records and receipt payloads.
            CREATE INDEX IF NOT EXISTS runtime_batch_totals_v1
                ON batches(status,attempted,accepted,rejected,bc_rows,rl_rows,eor_rows)
                WHERE status='done';
            CREATE INDEX IF NOT EXISTS runtime_assigned_batches_v1
                ON batches(id) WHERE status='assigned';
            CREATE TABLE IF NOT EXISTS final_games(uuid TEXT PRIMARY KEY,split TEXT NOT NULL,batch_id INTEGER NOT NULL);
        """)
        serialized = json.dumps(configuration, sort_keys=True)
        old = self.db.execute(
            "SELECT value FROM meta WHERE key='configuration'"
        ).fetchone()
        if old is not None and old[0] != serialized:
            self.db.close()
            raise ValueError(
                "Streaming source or implementation changed; use a new state/output directory."
            )
        self.db.execute(
            "INSERT OR IGNORE INTO meta VALUES('configuration',?)",
            (serialized,),
        )
        self.db.execute(
            "INSERT OR IGNORE INTO meta VALUES('source_cursor','0')"
        )
        self.db.commit()
        self.configuration = configuration
        self.source = sqlite3.connect(
            (Path(catalog).resolve() / "catalog.sqlite3").as_uri()
            + "?mode=ro",
            uri=True,
        )
        self.source.execute("PRAGMA query_only=ON")
        self.source.execute("PRAGMA cache_size=-32768")
        self.known_conflicts = {
            row[0]
            for row in self.db.execute(
                "SELECT uuid FROM games WHERE invalid=1"
            )
        }

    def close(self):
        self.source.close()
        self.db.close()

    def ingest(self, limit=4096):
        cursor = int(
            self.db.execute(
                "SELECT value FROM meta WHERE key='source_cursor'"
            ).fetchone()[0]
        )
        rows = self.source.execute(
            "SELECT rowid,uuid,split,semantic_sha256,metadata,conflicted FROM games WHERE rowid>? ORDER BY rowid LIMIT ?",
            (cursor, limit),
        ).fetchall()
        with self.db:
            for rowid, uuid, split, order_key, metadata, invalid in rows:
                record = json.loads(metadata)
                if (
                    record["uuid"] != uuid
                    or record["split"] != split
                    or record["semantic_sha256"] != order_key
                    or record["mode_id"] not in (12, 16)
                    or split_game(
                        uuid, self.configuration["catalog"]["split_salt"]
                    )
                    != split
                ):
                    raise ValueError(
                        "Source catalog metadata or whole-game split is inconsistent."
                    )
                self.db.execute(
                    "INSERT INTO games VALUES(?,?,?,?,?,NULL)",
                    (uuid, split, order_key, metadata, invalid),
                )
                cursor = rowid
            self.db.execute(
                "UPDATE meta SET value=? WHERE key='source_cursor'",
                (str(cursor),),
            )
        return len(rows)

    def refresh_conflicts(self):
        conflicts = {
            row[0]
            for row in self.source.execute(
                "SELECT uuid FROM games WHERE conflicted=1"
            )
        }
        newly_invalid = conflicts - self.known_conflicts
        if newly_invalid:
            with self.db:
                self.db.executemany(
                    "UPDATE games SET invalid=1 WHERE uuid=?",
                    ((uuid,) for uuid in newly_invalid),
                )
        # An invalid UUID can be discovered before its row has been ingested.
        self.known_conflicts = {
            row[0]
            for row in self.db.execute(
                "SELECT uuid FROM games WHERE invalid=1"
            )
        }

    def assign(self, split, size, *, partial=False):
        rows = self.db.execute(
            "SELECT uuid,metadata FROM games WHERE batch_id IS NULL AND invalid=0 AND split=? ORDER BY order_key,uuid LIMIT ?",
            (split, size),
        ).fetchall()
        if not rows or (len(rows) < size and not partial):
            return None
        records = [json.loads(row[1]) for row in rows]
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO batches(split,records,status) VALUES(?,?,'assigned')",
                (split, json.dumps(records, sort_keys=True)),
            )
            batch = cursor.lastrowid
            self.db.executemany(
                "UPDATE games SET batch_id=? WHERE uuid=?",
                ((batch, uuid) for uuid, _ in rows),
            )
        return batch, split, records

    def assigned(self):
        for batch, split, records in self.db.execute(
            "SELECT id,split,records FROM batches WHERE status='assigned' ORDER BY id"
        ):
            yield batch, split, json.loads(records)

    def complete(self, batch, receipt):
        with self.db:
            self.db.execute(
                "UPDATE batches SET status='done',receipt=?,attempted=?,accepted=?,rejected=?,bc_rows=?,rl_rows=?,eor_rows=? WHERE id=?",
                (
                    json.dumps(receipt, sort_keys=True),
                    receipt["attempted_games"],
                    receipt["accepted_games"],
                    receipt["rejected_games"],
                    receipt["files"]["bc"]["rows"],
                    receipt["files"]["rl"]["rows"],
                    receipt["files"]["eor"]["rows"],
                    batch,
                ),
            )

    def totals(self):
        names = (
            "batches",
            "attempted_games",
            "accepted_games",
            "rejected_games",
            "bc_rows",
            "rl_rows",
            "eor_rows",
        )
        row = self.db.execute(
            "SELECT COUNT(*),COALESCE(SUM(attempted),0),COALESCE(SUM(accepted),0),COALESCE(SUM(rejected),0),COALESCE(SUM(bc_rows),0),COALESCE(SUM(rl_rows),0),COALESCE(SUM(eor_rows),0) FROM batches WHERE status='done'"
        ).fetchone()
        return dict(zip(names, row))

    def pending(self):
        return self.db.execute(
            "SELECT COUNT(*) FROM games WHERE batch_id IS NULL AND invalid=0"
        ).fetchone()[0]

    def coverage(self):
        def signature(connection, query):
            checksum = hashlib.sha256()
            count = 0
            for row in connection.execute(query):
                checksum.update(
                    json.dumps(
                        row, ensure_ascii=False, separators=(",", ":")
                    ).encode()
                )
                checksum.update(b"\n")
                count += 1
            return {"games": count, "sha256": checksum.hexdigest()}

        source = signature(
            self.source,
            "SELECT uuid,split,semantic_sha256,metadata FROM games WHERE conflicted=0 ORDER BY uuid",
        )
        ingested = signature(
            self.db,
            "SELECT uuid,split,order_key,metadata FROM games WHERE invalid=0 ORDER BY uuid",
        )
        if source != ingested:
            raise RuntimeError(
                "Streaming ledger does not cover every usable source game exactly."
            )
        if self.pending() or next(self.assigned(), None) is not None:
            raise RuntimeError("Source games remain unprocessed.")
        return source


def clone_shard(source, target, receipt):
    target.mkdir(parents=True, exist_ok=True)
    for name in [
        *(value["name"] for value in receipt["files"].values()),
        "games.jsonl",
        "rejections.jsonl",
        "native.stderr.log",
    ]:
        temporary = target / (name + ".tmp")
        if temporary.exists():
            temporary.unlink()
        try:
            os.link(source / name, temporary)
        except OSError:
            shutil.copyfile(source / name, temporary)
        temporary.replace(target / name)


def reconcile(ledger, settings, report_progress):
    ledger.refresh_conflicts()
    coverage = ledger.coverage()
    output = Path(settings["output"])
    final = output / "final"
    final.mkdir(parents=True, exist_ok=True)
    final_settings = {**settings, "output": str(final)}
    receipts = []
    rebuilt = 0
    with ledger.db:
        ledger.db.execute("DELETE FROM final_games")
    for split in SPLITS:
        batches = ledger.db.execute(
            "SELECT id,records,receipt FROM batches WHERE status='done' AND split=?",
            (split,),
        ).fetchall()
        batches.sort(
            key=lambda row: json_digest(
                [split, row[0], settings["fingerprint"]]
            )
        )
        for index, (batch, serialized, original_receipt) in enumerate(batches):
            records = json.loads(serialized)
            invalid = {
                row[0]
                for row in ledger.db.execute(
                    "SELECT uuid FROM games WHERE batch_id=? AND invalid=1",
                    (batch,),
                )
            }
            retained = [
                record for record in records if record["uuid"] not in invalid
            ]
            target = final / "shards" / split / f"{index:06d}"
            if invalid:
                receipt = build_shard((split, index, retained), final_settings)
                rebuilt += 1
            else:
                receipt = json.loads(original_receipt)
                source = output / "shards" / split / f"{batch:06d}"
                clone_shard(source, target, receipt)
                receipt["source_shard"] = batch
                receipt["shard"] = index
                atomic_json(target / "receipt.json", receipt)
            members = 0
            expected = {record["uuid"]: record for record in retained}
            with (
                (target / "games.jsonl").open(encoding="utf-8") as stream,
                ledger.db,
            ):
                for line in stream:
                    record = json.loads(line)
                    if expected.get(record["uuid"]) != record:
                        raise RuntimeError(
                            "Final shard membership differs from its source metadata."
                        )
                    ledger.db.execute(
                        "INSERT INTO final_games VALUES(?,?,?)",
                        (record["uuid"], split, batch),
                    )
                    members += 1
            if members != receipt["accepted_games"]:
                raise RuntimeError(
                    "Final accepted-game count does not match shard provenance."
                )
            receipts.append(receipt)
            report_progress(
                "reconciling",
                reconciled_batches=len(receipts),
                rebuilt_batches=rebuilt,
            )
    accepted = sum(receipt["accepted_games"] for receipt in receipts)
    rejected = sum(receipt["rejected_games"] for receipt in receipts)
    if (
        accepted + rejected != coverage["games"]
        or ledger.db.execute("SELECT COUNT(*) FROM final_games").fetchone()[0]
        != accepted
    ):
        raise RuntimeError(
            "Final coverage, rejection counts or unique game membership differ."
        )
    if not accepted:
        raise RuntimeError("No usable games remain after reconciliation.")
    report_progress(
        "assembling", accepted_games=accepted, rejected_games=rejected
    )
    files = assemble(final, receipts, settings["reserve"])
    if any(
        not files[split][kind]["rows"] for split in SPLITS for kind in KINDS
    ):
        raise RuntimeError("A final BC/RL/EOR partition is empty.")
    return {
        "coverage": coverage,
        "accepted_games": accepted,
        "rejected_games": rejected,
        "conflicted_games": len(ledger.known_conflicts),
        "rebuilt_batches": rebuilt,
        "terminal_trajectories": sum(
            receipt["terminal_trajectories"] for receipt in receipts
        ),
        "files": files,
        "whole_game_split_disjoint": True,
    }


def run(arguments):
    if (
        not 1 <= arguments.workers <= 2
        or arguments.shard_games < 1
        or arguments.reserve_gib < 1
        or arguments.poll_seconds <= 0
        or not 0 <= arguments.max_rejection_rate < 1
    ):
        raise ValueError(
            "Invalid streaming worker, shard, polling, disk or rejection settings."
        )
    lower_priority()
    catalog, state, output, staging = [
        Path(value).resolve()
        for value in (
            arguments.catalog,
            arguments.state_dir,
            arguments.output,
            arguments.staging_root,
        )
    ]
    initial = catalog_status(
        catalog, allow_limited=arguments.allow_limited_catalog
    )
    for root in [*initial["configuration"]["roots"], str(catalog)]:
        if any(
            path == Path(root).resolve()
            or path.is_relative_to(Path(root).resolve())
            for path in (state, output, staging)
        ):
            raise ValueError(
                "Streaming outputs must be outside source and catalog directories."
            )
    for path in (state, output, staging):
        path.mkdir(parents=True, exist_ok=True)
    implementation = [
        Path(__file__),
        Path(__file__).with_name("prepare.py"),
        Path(__file__).with_name("validation.py"),
        Path(constants.__file__),
        Path(features.__file__),
    ]
    implementation += [
        arguments.repository / "bin" / "annotation" / kind / f"{kind}.py"
        for kind in ("rl", "eor")
    ]
    configuration = {
        "version": 1,
        "catalog": initial["configuration"],
        "catalog_path": str(catalog),
        "output": str(output),
        "annotator_sha256": digest(arguments.annotator),
        "implementation": {
            str(index) + ":" + path.name: digest(path)
            for index, path in enumerate(implementation)
        },
        "shard_games": arguments.shard_games,
        "max_rejection_rate": arguments.max_rejection_rate,
        "allow_limited_catalog": arguments.allow_limited_catalog,
        "python_version": sys.version,
        "bc_row_order": "deterministic_shuffle_within_shard",
        "final_shard_order": "deterministic_hash_order",
    }
    ledger = Ledger(state, catalog, configuration)
    last_report = 0

    def progress(status, **extra):
        nonlocal last_report
        if status == "reconciling" and time.monotonic() - last_report < 5:
            return
        value = {
            "status": status,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "output": str(output),
            "final_output": str(output / "final"),
            "catalog": str(catalog),
            **ledger.totals(),
            **extra,
        }
        atomic_json(state / "progress.json", value)
        if arguments.pipeline_state is not None:
            atomic_json(arguments.pipeline_state, value)
        last_report = time.monotonic()
        print(json.dumps(value), flush=True)

    settings = {
        "output": str(output),
        "staging": str(staging),
        "annotator": str(arguments.annotator.resolve()),
        "repository": str(arguments.repository.resolve()),
        "reserve": arguments.reserve_gib * 1024**3,
        "fingerprint": json_digest(configuration),
    }
    try:
        final_report = output / "final" / "dataset.json"
        if final_report.exists():
            report = read_json(final_report)
            if report["configuration"] != configuration:
                raise ValueError(
                    "A different streaming dataset already exists."
                )
            for split in SPLITS:
                for kind in KINDS:
                    file = report["files"][split][kind]
                    if digest(file["path"]) != file["sha256"]:
                        raise RuntimeError(
                            "A completed streaming output changed."
                        )
            progress(
                "complete",
                final_accepted_games=report["accepted_games"],
                final_rejected_games=report["rejected_games"],
            )
            return
        split_cursor = 0
        last_conflicts = -1
        with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
            pending = {}
            while True:
                source = catalog_status(
                    catalog,
                    ledger.source,
                    allow_limited=arguments.allow_limited_catalog,
                )
                if source["configuration"] != configuration["catalog"]:
                    raise ValueError(
                        "Source catalog implementation or inputs changed during streaming."
                    )
                if source["status"] == "failed":
                    raise RuntimeError(
                        "Source catalog failed; inspect its error.json before continuing."
                    )
                if (
                    source["status"] == "paused"
                    or (state / "pause.request").exists()
                ):
                    for future, batch in list(pending.items()):
                        ledger.complete(batch, future.result())
                    progress("paused")
                    return
                ingested = ledger.ingest()
                sealed = source["status"] == "complete" or (
                    arguments.allow_limited_catalog
                    and source["status"] == "limited"
                )
                if source.get("conflicted_games", 0) != last_conflicts:
                    ledger.refresh_conflicts()
                    last_conflicts = source.get("conflicted_games", 0)
                for future in list(pending):
                    if future.done():
                        ledger.complete(pending.pop(future), future.result())
                totals = ledger.totals()
                if (
                    totals["attempted_games"] >= 64
                    and totals["rejected_games"] / totals["attempted_games"]
                    > arguments.max_rejection_rate
                ):
                    raise RuntimeError(
                        "Streaming rejection rate exceeds the limit; inspect per-shard rejections."
                    )
                if (
                    sealed
                    and ingested == 0
                    and not pending
                    and not ledger.pending()
                    and next(ledger.assigned(), None) is None
                ):
                    break
                running = set(pending.values())
                for batch, split, records in ledger.assigned():
                    if len(pending) >= arguments.workers:
                        break
                    if batch not in running:
                        pending[
                            executor.submit(
                                build_shard, (split, batch, records), settings
                            )
                        ] = batch
                        running.add(batch)
                for _ in range(len(SPLITS)):
                    if len(pending) >= arguments.workers:
                        break
                    split = SPLITS[split_cursor % len(SPLITS)]
                    split_cursor += 1
                    job = ledger.assign(
                        split, arguments.shard_games, partial=sealed
                    )
                    if job is not None:
                        batch, split, records = job
                        pending[
                            executor.submit(
                                build_shard, (split, batch, records), settings
                            )
                        ] = batch
                if time.monotonic() - last_report >= 5:
                    progress(
                        "streaming" if pending else "waiting_for_records",
                        source_status=source["status"],
                        source_usable_games=source["usable_games"],
                        queued_games=ledger.pending(),
                    )
                if pending:
                    wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                elif ingested == 0:
                    time.sleep(arguments.poll_seconds)
        progress("reconciling")
        report = reconcile(ledger, settings, progress)
        if (
            report["rejected_games"]
            / (report["accepted_games"] + report["rejected_games"])
            > arguments.max_rejection_rate
        ):
            raise RuntimeError(
                "Final conversion rejection rate exceeds the limit."
            )
        report.update(
            configuration=configuration,
            source_status=source["status"],
            source_manifests={
                split: digest(catalog / f"manifest.{split}.jsonl")
                for split in SPLITS
            },
        )
        atomic_json(output / "final" / "dataset.json", report)
        progress(
            "complete",
            final_accepted_games=report["accepted_games"],
            final_rejected_games=report["rejected_games"],
        )
    except BaseException as error:
        progress("failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        ledger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--annotator", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--pipeline-state", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard-games", type=int, default=64)
    parser.add_argument("--reserve-gib", type=int, default=20)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--max-rejection-rate", type=float, default=0.02)
    parser.add_argument("--allow-limited-catalog", action="store_true")
    parser.add_argument("--log-file", type=Path)
    arguments = parser.parse_args()
    if arguments.log_file is None:
        run(arguments)
    else:
        arguments.log_file.parent.mkdir(parents=True, exist_ok=True)
        with (
            arguments.log_file.open(
                "a", encoding="utf-8", buffering=1
            ) as stream,
            redirect_stdout(stream),
            redirect_stderr(stream),
        ):
            try:
                run(arguments)
            except BaseException:  # noqa: BLE001 -- retain task failure diagnostics
                traceback.print_exc()
                sys.exit(1)


if __name__ == "__main__":
    main()
