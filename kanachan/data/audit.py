"""Independently audit completed streaming datasets without Torch or raw scans."""

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import time
import traceback
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path

from kanachan import constants
from kanachan.training.core import features

from . import validation
from .catalog import lower_priority

SPLITS = ("train", "validation", "test")
KINDS = ("bc", "rl", "eor")


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for attempt in range(100):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.02)


class HashedReader:
    def __init__(self, raw, aggregate=None):
        self.raw = raw
        self.checksum = hashlib.sha256()
        self.bytes = 0
        self.aggregate = aggregate

    def read(self, size=-1):
        block = self.raw.read(size)
        self.checksum.update(block)
        self.bytes += len(block)
        if self.aggregate is not None:
            self.aggregate.update(block)
        return block


def digest(path, aggregate=None):
    with Path(path).open("rb") as raw:
        reader = HashedReader(raw, aggregate)
        while reader.read(1024 * 1024):
            pass
        return reader.checksum.hexdigest()


def gzip_lines(path, expected, aggregate=None, progress=None):
    """Read every gzip member and verify CRC, compressed bytes and SHA256."""
    before = path.stat()
    rows = 0
    with path.open("rb") as raw:
        reader = HashedReader(raw, aggregate)
        with gzip.GzipFile(fileobj=reader, mode="rb") as contents:
            for line in contents:
                if not line.endswith(b"\n"):
                    raise ValueError(f"Unterminated annotation row: {path}")
                rows += 1
                if progress is not None and rows % 65536 == 0:
                    progress(current_file=str(path), current_rows=rows)
                yield line.decode("utf-8")
        if (
            reader.bytes != expected["bytes"]
            or reader.checksum.hexdigest() != expected["sha256"]
            or rows != expected["rows"]
        ):
            raise ValueError(
                f"Compressed file integrity/count mismatch: {path}"
            )
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"A file changed during the audit: {path}")


def expected_split(uuid, salt):
    value = hashlib.sha256((salt + "|" + uuid.lower()).encode()).digest()
    bucket = int.from_bytes(value[:8], "big") % 10000
    return (
        "train" if bucket < 9000 else "validation" if bucket < 9500 else "test"
    )


def read_members(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def source_configuration(catalog):
    connection = sqlite3.connect(
        (Path(catalog).resolve() / "catalog.sqlite3").as_uri() + "?mode=ro",
        uri=True,
    )
    try:
        return json.loads(
            connection.execute(
                "SELECT value FROM meta WHERE key='configuration'"
            ).fetchone()[0]
        )
    finally:
        connection.close()


def check_state_location(dataset, catalog, state, configuration):
    for root in [dataset, catalog, *map(Path, configuration["roots"])]:
        if state == root.resolve() or state.is_relative_to(root.resolve()):
            raise ValueError(
                "Audit state must stay outside data and raw input directories."
            )


def source_member(connection, uuid, split, salt):
    row = connection.execute(
        "SELECT split,conflicted,metadata FROM source.games WHERE uuid=?",
        (uuid,),
    ).fetchone()
    if (
        row is None
        or row[:2] != (split, 0)
        or expected_split(uuid, salt) != split
    ):
        raise ValueError(
            "A UUID is unknown, conflicted or in the wrong split."
        )
    metadata = json.loads(row[2])
    if (
        metadata["uuid"] != uuid
        or metadata["split"] != split
        or metadata["category"] != 2
        or metadata["mode_id"] not in (12, 16)
        or len(metadata["level_ids"]) != 4
        or len(metadata["scores"]) != 4
    ):
        raise ValueError(
            "A source game differs from the four-player room filter."
        )
    return metadata


def decision(line):
    columns = line.rstrip("\r\n").split("\t")
    return hashlib.sha256("\t".join(columns[:6]).encode()).digest()


def eor_record(line):
    columns = line.rstrip("\r\n").split("\t")
    if len(columns) != 4:
        raise ValueError("Invalid EOR column count.")
    return (columns[0], *(tuple(validation.integers(value)) for value in columns[1:]))


def expected_eor(lines):
    """Derive nonfinal round targets from one game's validated BC decisions."""
    rounds = {}
    for line in lines:
        columns = line.rstrip("\r\n").split("\t")
        if len(columns) != 8:
            raise ValueError("Invalid BC columns for EOR correspondence.")
        if "," in columns[3]:
            continue
        sparse = validation.integers(columns[1])
        numeric = validation.integers(columns[2])
        results = validation.integers(columns[7])
        if len(sparse) < 9 or not numeric or len(results) != constants.NUM_RESULTS:
            raise ValueError("Invalid BC round-start shape.")
        chang, ju, honba = sparse[7] - 75, sparse[8] - 78, numeric[0]
        key = chang, ju, honba
        if key in rounds:
            raise ValueError("BC repeated a round-start decision.")
        rounds[key] = (
            columns[0],
            tuple(sparse[:6]) + (chang + 71, ju + 74),
            tuple(results[4:10]),
            tuple(results[10:14]),
        )
    # The EOR converter deliberately excludes the final round of each game:
    # its score features would already reveal the complete final outcome.
    return Counter(rounds[key] for key in sorted(rounds)[:-1])


def audit_shard(directory, receipt, connection, salt, aggregates):
    split = receipt["split"]
    accepted, members = {}, []
    for record in read_members(directory / "games.jsonl"):
        uuid = record["uuid"]
        if (
            uuid in accepted
            or source_member(connection, uuid, split, salt) != record
        ):
            raise ValueError(
                "Accepted game provenance is duplicated or changed."
            )
        accepted[uuid] = record
        members.append((uuid, split, "accepted", receipt["shard"]))
    rejected = set()
    for record in read_members(directory / "rejections.jsonl"):
        uuid = record["uuid"]
        source_member(connection, uuid, split, salt)
        if uuid in accepted or uuid in rejected or not record.get("reason"):
            raise ValueError("Invalid conversion rejection provenance.")
        rejected.add(uuid)
        members.append((uuid, split, "rejected", receipt["shard"]))
    if (
        len(accepted) != receipt["accepted_games"]
        or len(rejected) != receipt["rejected_games"]
        or len(members) != receipt["attempted_games"]
    ):
        raise ValueError("Shard game accounting is incomplete.")
    for kind in KINDS:
        if receipt["files"][kind]["name"] != f"{kind}.tsv.gz":
            raise ValueError("Unexpected shard filename.")
    grouped = defaultdict(list)
    decisions = Counter()
    rows = {kind: 0 for kind in KINDS}
    for line in gzip_lines(
        directory / "bc.tsv.gz", receipt["files"]["bc"], aggregates["bc"]
    ):
        uuid = line.split("\t", 1)[0]
        if uuid not in accepted:
            raise ValueError("BC contains an unaccepted game.")
        grouped[uuid].append(line)
        decisions[decision(line)] += 1
        rows["bc"] += 1
    seats = {
        uuid: validation.validate_bc(grouped[uuid], record)
        for uuid, record in accepted.items()
    }
    eor_targets = Counter()
    zero_eor_games = 0
    for uuid in accepted:
        targets = expected_eor(grouped[uuid])
        eor_targets.update(targets)
        zero_eor_games += not targets
    per_game_rows = {uuid: len(lines) for uuid, lines in grouped.items()}
    del grouped
    terminals = 0
    for kind in ("rl", "eor"):
        seen = set()
        lines = gzip_lines(
            directory / f"{kind}.tsv.gz",
            receipt["files"][kind],
            aggregates[kind],
        )
        for uuid, group in groupby(
            lines, key=lambda line: line.split("\t", 1)[0]
        ):
            if uuid not in accepted or uuid in seen:
                raise ValueError("Converted data repeated or invented a game.")
            seen.add(uuid)
            game = list(group)
            if kind == "rl":
                result = validation.validate_rl(
                    game, accepted[uuid], per_game_rows[uuid], seats[uuid]
                )
                terminals += result["terminal"]
                for line in game:
                    key = decision(line)
                    decisions[key] -= 1
                    if decisions[key] < 0:
                        raise ValueError("BC/RL decision multiset mismatch.")
            else:
                validation.validate_eor(game, accepted[uuid])
                for line in game:
                    key = eor_record(line)
                    eor_targets[key] -= 1
                    if eor_targets[key] < 0:
                        raise ValueError("BC/EOR nonfinal round correspondence mismatch.")
            rows[kind] += len(game)
        if kind == "rl" and seen != set(accepted):
            raise ValueError("RL omitted an accepted game.")
    if (
        any(decisions.values())
        or terminals != receipt["terminal_trajectories"]
    ):
        raise ValueError("BC/RL decisions or terminal accounting differ.")
    if any(eor_targets.values()):
        raise ValueError("BC/EOR nonfinal round correspondence mismatch.")
    return {
        "rows": rows,
        "terminal_trajectories": terminals,
        "accepted_games": len(accepted),
        "rejected_games": len(rejected),
        "eor_zero_target_games": zero_eor_games,
        "bc_eor_nonfinal_rounds_identical": True,
    }, members


def audit(dataset, catalog, state, progress=lambda **_: None):
    dataset, catalog, state = (
        Path(value).resolve() for value in (dataset, catalog, state)
    )
    report = json.loads(
        (dataset / "dataset.json").read_text(encoding="utf-8-sig")
    )
    if report.get("source_status") != "complete":
        raise ValueError(
            "Only a complete, reconciled streaming dataset can be audited."
        )
    if Path(report["configuration"]["catalog_path"]).resolve() != catalog:
        raise ValueError("The dataset refers to a different source catalog.")
    source_config = report["configuration"]["catalog"]
    if report["configuration"].get("version") != 1:
        raise ValueError(
            "Unsupported streaming dataset configuration version."
        )
    semantic_files = {
        "2:validation.py": validation.__file__,
        "3:constants.py": constants.__file__,
        "4:features.py": features.__file__,
    }
    semantic_hashes = {
        key: digest(path) for key, path in semantic_files.items()
    }
    if any(
        report["configuration"]["implementation"].get(key) != value
        for key, value in semantic_hashes.items()
    ):
        raise ValueError(
            "Dataset feature/validation implementation differs from the auditor."
        )
    if source_configuration(catalog) != source_config:
        raise ValueError("Source catalog configuration changed.")
    if (
        source_config["category"],
        source_config["players"],
        source_config["modes"],
        source_config["whole_game_split"],
    ) != (2, 4, [12, 16], [90, 5, 5]):
        raise ValueError("Unexpected source filter or whole-game split.")
    check_state_location(dataset, catalog, state, source_config)
    state.mkdir(parents=True, exist_ok=True)
    identity = {
        "dataset": str(dataset),
        "catalog": str(catalog),
        "dataset_sha256": digest(dataset / "dataset.json"),
        "auditor_sha256": digest(__file__),
        "semantic_implementation": semantic_hashes,
    }
    connection = sqlite3.connect(state / "audit.sqlite3", uri=True)
    try:
        connection.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA cache_size=-32768;
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE IF NOT EXISTS shards(split TEXT,shard INTEGER,receipt_sha256 TEXT,result TEXT,
                PRIMARY KEY(split,shard));
            CREATE TABLE IF NOT EXISTS members(uuid TEXT PRIMARY KEY,split TEXT,outcome TEXT,shard INTEGER);
        """)
        encoded = json.dumps(identity, sort_keys=True)
        old = connection.execute(
            "SELECT value FROM meta WHERE key='identity'"
        ).fetchone()
        if old is not None and old[0] != encoded:
            raise ValueError(
                "Audit inputs or implementation changed; use a new state directory."
            )
        connection.execute(
            "INSERT OR IGNORE INTO meta VALUES('identity',?)", (encoded,)
        )
        connection.commit()
        connection.execute(
            "ATTACH DATABASE ? AS source",
            ((catalog / "catalog.sqlite3").as_uri() + "?mode=ro",),
        )
        if connection.execute("PRAGMA source.quick_check").fetchall() != [
            ("ok",)
        ]:
            raise ValueError("Source SQLite integrity check failed.")
        actual = json.loads(
            connection.execute(
                "SELECT value FROM source.meta WHERE key='configuration'"
            ).fetchone()[0]
        )
        unfinished = connection.execute(
            "SELECT COUNT(*) FROM source.directories WHERE done=0"
        ).fetchone()[0]
        directories = connection.execute(
            "SELECT COUNT(*) FROM source.directories"
        ).fetchone()[0]
        if actual != source_config or unfinished or not directories:
            raise ValueError(
                "Source catalog is unfinished or its configuration changed."
            )
        summary = json.loads(
            (catalog / "summary.json").read_text(encoding="utf-8")
        )
        if summary["configuration"] != source_config:
            raise ValueError("Source summary configuration differs.")
        totals = {split: {kind: 0 for kind in KINDS} for split in SPLITS}
        aggregates = {
            split: {kind: hashlib.sha256() for kind in KINDS}
            for split in SPLITS
        }
        terminals = completed = reused = zero_eor_games = 0
        for split in SPLITS:
            directories = sorted((dataset / "shards" / split).iterdir())
            for index, directory in enumerate(directories):
                if directory.name != f"{index:06d}" or not directory.is_dir():
                    raise ValueError(
                        "Final shard numbering is not contiguous."
                    )
                receipt_path = directory / "receipt.json"
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt["split"] != split or receipt["shard"] != index:
                    raise ValueError(
                        "Shard receipt is in a different split or index."
                    )
                checksum = digest(receipt_path)
                cached = connection.execute(
                    "SELECT receipt_sha256,result FROM shards WHERE split=? AND shard=?",
                    (split, index),
                ).fetchone()
                if cached is not None:
                    if cached[0] != checksum:
                        raise ValueError(
                            "An already audited shard receipt changed."
                        )
                    for kind in KINDS:
                        path = directory / f"{kind}.tsv.gz"
                        expected = receipt["files"][kind]
                        if (
                            path.stat().st_size != expected["bytes"]
                            or digest(path, aggregates[split][kind])
                            != expected["sha256"]
                        ):
                            raise ValueError(
                                "An already audited shard changed."
                            )
                    result = json.loads(cached[1])
                    if not result.get("bc_eor_nonfinal_rounds_identical"):
                        raise ValueError("Cached shard lacks EOR correspondence proof.")
                    reused += 1
                else:
                    result, members = audit_shard(
                        directory,
                        receipt,
                        connection,
                        source_config["split_salt"],
                        aggregates[split],
                    )
                    with connection:
                        connection.executemany(
                            "INSERT INTO members VALUES(?,?,?,?)", members
                        )
                        connection.execute(
                            "INSERT INTO shards VALUES(?,?,?,?)",
                            (split, index, checksum, json.dumps(result)),
                        )
                for kind in KINDS:
                    totals[split][kind] += result["rows"][kind]
                terminals += result["terminal_trajectories"]
                zero_eor_games += result["eor_zero_target_games"]
                completed += 1
                progress(
                    phase="shards",
                    audited_shards=completed,
                    reused_shards=reused,
                )
        counts = dict(
            connection.execute(
                "SELECT outcome,COUNT(*) FROM members GROUP BY outcome"
            )
        )
        accepted, rejected = (
            counts.get("accepted", 0),
            counts.get("rejected", 0),
        )
        missing = connection.execute(
            "SELECT COUNT(*) FROM source.games s LEFT JOIN members m ON s.uuid=m.uuid WHERE s.conflicted=0 AND m.uuid IS NULL"
        ).fetchone()[0]
        invalid = connection.execute(
            "SELECT COUNT(*) FROM members m LEFT JOIN source.games s ON m.uuid=s.uuid WHERE s.uuid IS NULL OR s.conflicted!=0 OR s.split!=m.split"
        ).fetchone()[0]
        if (
            missing
            or invalid
            or accepted != report["accepted_games"]
            or rejected != report["rejected_games"]
        ):
            raise ValueError(
                "Final UUID coverage or exclusion accounting differs from the source."
            )
        coverage = hashlib.sha256()
        games = 0
        for row in connection.execute(
            "SELECT uuid,split,semantic_sha256,metadata FROM source.games WHERE conflicted=0 ORDER BY uuid"
        ):
            coverage.update(
                json.dumps(
                    row, ensure_ascii=False, separators=(",", ":")
                ).encode()
                + b"\n"
            )
            games += 1
        conflicts = connection.execute(
            "SELECT COUNT(*) FROM source.games WHERE conflicted=1"
        ).fetchone()[0]
        split_counts = dict(
            connection.execute(
                "SELECT split,COUNT(*) FROM source.games WHERE conflicted=0 GROUP BY split"
            )
        )
        outcomes = dict(
            connection.execute(
                "SELECT outcome,COUNT(*) FROM source.files GROUP BY outcome"
            )
        )
        reasons = dict(
            connection.execute(
                "SELECT reason,COUNT(*) FROM source.files WHERE outcome='rejected' GROUP BY reason"
            )
        )
        first_end, last_end = connection.execute(
            "SELECT MIN(end_time),MAX(end_time) FROM source.games WHERE conflicted=0"
        ).fetchone()
        modes = dict(
            connection.execute(
                "SELECT mode_id,COUNT(*) FROM source.games WHERE conflicted=0 GROUP BY mode_id"
            )
        )
        if (
            outcomes != summary["file_outcomes"]
            or reasons != summary["rejection_reasons"]
            or first_end != summary["first_end_time"]
            or last_end != summary["last_end_time"]
            or {str(key): value for key, value in modes.items()}
            != summary["modes"]
        ):
            raise ValueError(
                "Catalog inventory, exclusion reasons or date range differs from its summary."
            )
        if (
            report["coverage"]
            != {"games": games, "sha256": coverage.hexdigest()}
            or conflicts != report["conflicted_games"]
            or terminals != report["terminal_trajectories"]
            or split_counts != summary["splits"]
        ):
            raise ValueError(
                "Source coverage signature, split summary or terminal totals differ."
            )
        for split in SPLITS:
            manifest = catalog / f"manifest.{split}.jsonl"
            if digest(manifest) != report["source_manifests"][split]:
                raise ValueError("A published source manifest changed.")
            for kind in KINDS:
                expected = report["files"][split][kind]
                path = dataset / f"{split}.{kind}.tsv.gz"
                if (
                    Path(expected["path"]).resolve() != path
                    or expected["rows"] <= 0
                    or totals[split][kind] != expected["rows"]
                ):
                    raise ValueError(
                        "Final file path or shard row totals differ."
                    )
                if aggregates[split][kind].hexdigest() != expected["sha256"]:
                    raise ValueError(
                        "Final file differs from the ordered audited shards."
                    )
                progress(phase="final_files", current_file=str(path))
                for _ in gzip_lines(path, expected, progress=progress):
                    pass
        result = {
            "status": "complete",
            "identity": identity,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "accepted_games": accepted,
            "rejected_games": rejected,
            "conflicted_games": conflicts,
            "audited_shards": completed,
            "reused_shards": reused,
            "rows": totals,
            "terminal_trajectories": terminals,
            "whole_game_split_disjoint": True,
            "source_uuid_coverage_complete": True,
            "bc_rl_decision_multisets_identical": True,
            "bc_eor_nonfinal_rounds_identical": True,
            "eor_zero_target_games": zero_eor_games,
            "all_nine_gzip_files_crc_sha256_rows_verified": True,
            "file_outcomes": summary["file_outcomes"],
            "source_rejection_reasons": summary["rejection_reasons"],
            "first_end_time": first_end,
            "last_end_time": last_end,
            "modes": modes,
        }
        write_json(state / "acceptance.json", result)
        return result
    finally:
        connection.close()


def run(arguments):
    lower_priority()
    state = arguments.state_dir.resolve()
    check_state_location(
        arguments.dataset.resolve(),
        arguments.catalog.resolve(),
        state,
        source_configuration(arguments.catalog),
    )
    state.mkdir(parents=True, exist_ok=True)
    last = 0

    def progress(**values):
        nonlocal last
        if time.monotonic() - last < 5 and values.get("phase") != "complete":
            return
        value = {
            "status": "auditing",
            "pid": os.getpid(),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            **values,
        }
        write_json(state / "progress.json", value)
        print(json.dumps(value), flush=True)
        last = time.monotonic()

    try:
        while not (arguments.dataset / "dataset.json").exists():
            if not arguments.wait:
                raise ValueError("The final dataset is not published yet.")
            progress(
                status="waiting_for_dataset", dataset=str(arguments.dataset)
            )
            time.sleep(10)
        result = audit(arguments.dataset, arguments.catalog, state, progress)
        progress(
            phase="complete",
            status="complete",
            accepted_games=result["accepted_games"],
            receipt=str(state / "acceptance.json"),
        )
    except BaseException as error:
        write_json(
            state / "progress.json",
            {
                "status": "failed",
                "error": str(error),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
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
            except BaseException:  # noqa: BLE001 -- retain scheduled-task diagnostics
                traceback.print_exc()
                sys.exit(1)


if __name__ == "__main__":
    main()
