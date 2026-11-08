"""Build resumable BC/RL/EOR shards from a completed raw-game catalog."""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import groupby, islice
import json
from pathlib import Path
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback

from kanachan import constants
from kanachan.training.core import features
from .catalog import atomic_json, lower_priority, split_game
from .validation import validate_bc, validate_rl, validate_eor


KINDS = ("bc", "rl", "eor")
SPLITS = ("train", "validation", "test")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()
    ).hexdigest()


def verify_copy(record, destination):
    source = Path(record["path"])
    before = source.stat()
    if (before.st_size, before.st_mtime_ns) != (
        record["bytes"],
        record["mtime_ns"],
    ):
        raise RuntimeError(f"Raw source metadata changed: {source}")
    checksum = hashlib.sha256()
    with source.open("rb") as reader, Path(destination).open("xb") as writer:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            writer.write(block)
            checksum.update(block)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ) or checksum.hexdigest() != record["sha256"]:
        raise RuntimeError(
            f"Raw source changed or was read incompletely: {source}"
        )


def chunks(catalog, shard_games, limit_games):
    remaining = limit_games or None
    salt = json.loads((catalog / "summary.json").read_text(encoding="utf-8"))[
        "configuration"
    ]["split_salt"]
    # Hash order mixes dates/rooms globally while preserving complete games.
    # SQLite spills the sort to disk rather than holding all metadata in RAM.
    connection = sqlite3.connect(
        (catalog / "catalog.sqlite3").as_uri() + "?mode=ro", uri=True
    )
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-32768")
    try:
        for split in SPLITS:
            if remaining == 0:
                break
            previous = ("", "")
            stream = connection.execute(
                "SELECT metadata FROM games WHERE conflicted=0 AND split=? ORDER BY semantic_sha256,uuid",
                (split,),
            )
            index = 0
            while remaining is None or remaining > 0:
                count = (
                    shard_games
                    if remaining is None
                    else min(shard_games, remaining)
                )
                records = [json.loads(row[0]) for row in islice(stream, count)]
                if not records:
                    break
                for record in records:
                    uuid = record["uuid"]
                    if (
                        (record["semantic_sha256"], uuid) <= previous
                        or record["split"] != split
                        or split_game(uuid, salt) != split
                        or record["mode_id"] not in (12, 16)
                    ):
                        raise ValueError(
                            "Manifest is unsorted, repeated or assigned to the wrong partition."
                        )
                    previous = (record["semantic_sha256"], uuid)
                yield split, index, records
                index += 1
                if remaining is not None:
                    remaining -= len(records)
    finally:
        connection.close()


def compress(source, destination):
    with (
        Path(source).open("rb") as reader,
        Path(destination).open("wb") as stream,
    ):
        with gzip.GzipFile(
            filename="", fileobj=stream, mode="wb", compresslevel=3, mtime=0
        ) as writer:
            shutil.copyfileobj(reader, writer, 1024 * 1024)


def build_shard(job, settings):
    split, index, records = job
    output = Path(settings["output"]) / "shards" / split / f"{index:06d}"
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "receipt.json"
    fingerprint = json_digest(
        {"configuration": settings["fingerprint"], "records": records}
    )
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["fingerprint"] != fingerprint:
            raise ValueError(f"Shard input/configuration changed: {output}")
        for value in receipt["files"].values():
            if digest(output / value["name"]) != value["sha256"]:
                raise RuntimeError(f"A completed shard changed: {output}")
        return receipt
    if (
        shutil.disk_usage(settings["staging"]).free < settings["reserve"]
        or shutil.disk_usage(settings["output"]).free < settings["reserve"]
    ):
        raise RuntimeError(
            "Insufficient free disk space; completed shards remain resumable."
        )
    with tempfile.TemporaryDirectory(
        prefix=f"{split}-{index:06d}-", dir=settings["staging"]
    ) as temporary:
        work = Path(temporary).resolve()
        if not work.is_relative_to(Path(settings["staging"]).resolve()):
            raise RuntimeError("Temporary directory escaped staging root.")
        raw = work / "raw"
        raw.mkdir()
        for record in records:
            verify_copy(record, raw / record["uuid"])
        filelist = work / "files.txt"
        filelist.write_text(
            "".join(str(raw / record["uuid"]) + "\n" for record in records),
            encoding="utf-8",
        )
        native = work / "native.tsv"
        status = work / "native-status.tsv"
        with (
            native.open("wb") as stdout,
            (work / "native.stderr.log").open("wb") as stderr,
        ):
            subprocess.run(
                [
                    settings["annotator"],
                    "--files-from",
                    str(filelist),
                    "--status-file",
                    str(status),
                    "--continue-on-error",
                ],
                stdout=stdout,
                stderr=stderr,
                check=True,
                timeout=1800,
            )
        statuses = [
            line.split("\t", 4)
            for line in status.read_text(encoding="utf-8").splitlines()
        ]
        if len(statuses) != len(records) or any(
            len(row) != 5
            or Path(row[1]).name != record["uuid"]
            or row[0] not in ("ok", "error")
            for row, record in zip(statuses, records)
        ):
            raise RuntimeError(
                "Annotation status does not match the requested games."
            )
        metadata = {record["uuid"]: record for record in records}
        expected = {
            record["uuid"]: int(row[2])
            for record, row in zip(records, statuses)
            if row[0] == "ok"
        }
        rejections = [
            {"uuid": record["uuid"], "phase": "native", "reason": row[4]}
            for record, row in zip(records, statuses)
            if row[0] == "error"
        ]
        accepted = {}
        bc = work / "bc.tsv"
        with (
            native.open(encoding="utf-8") as reader,
            bc.open("w", encoding="utf-8", newline="\n") as writer,
        ):
            for uuid, group in groupby(
                reader, key=lambda line: line.split("\t", 1)[0]
            ):
                lines = list(group)
                if uuid not in expected or len(lines) != expected.pop(uuid):
                    raise RuntimeError(
                        "Native output repeated, lost or invented a game."
                    )
                try:
                    seats = validate_bc(lines, metadata[uuid])
                except (ValueError, RuntimeError, KeyError) as error:
                    rejections.append(
                        {
                            "uuid": uuid,
                            "phase": "bc_validation",
                            "reason": str(error),
                        }
                    )
                    continue
                accepted[uuid] = {
                    "metadata": metadata[uuid],
                    "rows": len(lines),
                    "seats": seats,
                }
                writer.writelines(lines)
        if expected:
            raise RuntimeError("Native output omitted completed games.")
        files = {}
        row_counts = {
            "bc": sum(value["rows"] for value in accepted.values()),
            "rl": 0,
            "eor": 0,
        }
        terminals = 0
        for kind in ("rl", "eor"):
            command = [
                sys.executable,
                str(
                    Path(settings["repository"])
                    / "bin"
                    / "annotation"
                    / kind
                    / f"{kind}.py"
                ),
            ]
            if kind == "rl":
                command += ["--stream"]
            destination = work / f"{kind}.tsv"
            with (
                bc.open("rb") as stdin,
                destination.open("wb") as stdout,
                (work / f"{kind}.stderr.log").open("wb") as stderr,
            ):
                subprocess.run(
                    command,
                    stdin=stdin,
                    stdout=stdout,
                    stderr=stderr,
                    check=True,
                    timeout=1800,
                )
            seen = set()
            with destination.open(encoding="utf-8") as reader:
                for uuid, group in groupby(
                    reader, key=lambda line: line.split("\t", 1)[0]
                ):
                    if uuid not in accepted or uuid in seen:
                        raise RuntimeError(
                            "Converted output repeated or invented a game."
                        )
                    seen.add(uuid)
                    value = accepted[uuid]
                    if kind == "rl":
                        counts = validate_rl(
                            group,
                            value["metadata"],
                            value["rows"],
                            value["seats"],
                        )
                        row_counts[kind] += counts["rows"]
                        terminals += counts["terminal"]
                    else:
                        row_counts[kind] += validate_eor(
                            group, value["metadata"]
                        )
            if kind == "rl" and seen != set(accepted):
                raise RuntimeError("RL conversion lost complete games.")
        # BC consumes independent decisions, so mix rows within each shard.
        # RL and EOR were already converted from complete, ordered games.
        with bc.open(encoding="utf-8") as reader:
            shuffled = list(reader)
        random.Random(fingerprint).shuffle(shuffled)
        with bc.open("w", encoding="utf-8", newline="\n") as writer:
            writer.writelines(shuffled)
        del shuffled
        for kind in KINDS:
            destination = work / f"{kind}.tsv.gz"
            compress(work / f"{kind}.tsv", destination)
            files[kind] = {
                "name": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": digest(destination),
                "rows": row_counts[kind],
            }
            shutil.copyfile(destination, output / (destination.name + ".tmp"))
            (output / (destination.name + ".tmp")).replace(
                output / destination.name
            )
        for name, values in (
            ("rejections.jsonl", rejections),
            (
                "games.jsonl",
                [value["metadata"] for value in accepted.values()],
            ),
        ):
            temporary_path = output / (name + ".tmp")
            temporary_path.write_text(
                "".join(
                    json.dumps(value, ensure_ascii=False) + "\n"
                    for value in values
                ),
                encoding="utf-8",
            )
            temporary_path.replace(output / name)
        shutil.copyfile(
            work / "native.stderr.log", output / "native.stderr.log"
        )
        receipt = {
            "fingerprint": fingerprint,
            "split": split,
            "shard": index,
            "attempted_games": len(records),
            "accepted_games": len(accepted),
            "rejected_games": len(rejections),
            "terminal_trajectories": terminals,
            "files": files,
        }
        atomic_json(receipt_path, receipt)
        return receipt


def assemble(output, receipts, reserve):
    required = sum(
        value["bytes"]
        for receipt in receipts
        for value in receipt["files"].values()
    )
    if shutil.disk_usage(output).free < required + reserve:
        raise RuntimeError(
            "Insufficient room to assemble training files; shards are complete and preserved."
        )
    files = {}
    for split in SPLITS:
        selected = sorted(
            (receipt for receipt in receipts if receipt["split"] == split),
            key=lambda value: value["shard"],
        )
        files[split] = {}
        for kind in KINDS:
            target = output / f"{split}.{kind}.tsv.gz"
            temporary = target.with_name(target.name + ".tmp")
            checksum = hashlib.sha256()
            with temporary.open("wb") as writer:
                for receipt in selected:
                    source = (
                        output
                        / "shards"
                        / split
                        / f"{receipt['shard']:06d}"
                        / receipt["files"][kind]["name"]
                    )
                    member_checksum = hashlib.sha256()
                    with source.open("rb") as reader:
                        for block in iter(
                            lambda: reader.read(1024 * 1024), b""
                        ):
                            writer.write(block)
                            checksum.update(block)
                            member_checksum.update(block)
                    if (
                        member_checksum.hexdigest()
                        != receipt["files"][kind]["sha256"]
                    ):
                        raise RuntimeError(
                            f"Shard changed during assembly: {source}"
                        )
            temporary.replace(target)
            files[split][kind] = {
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": checksum.hexdigest(),
                "rows": sum(
                    receipt["files"][kind]["rows"] for receipt in selected
                ),
            }
    return files


def run(arguments):
    if (
        not 1 <= arguments.workers <= 2
        or arguments.shard_games < 1
        or arguments.limit_games < 0
        or arguments.reserve_gib < 1
        or not 0 <= arguments.max_rejection_rate < 1
    ):
        raise ValueError(
            "Invalid worker, shard, sample, disk reserve or rejection settings."
        )
    lower_priority()
    catalog, output, staging = (
        Path(value).resolve()
        for value in (
            arguments.catalog,
            arguments.output,
            arguments.staging_root,
        )
    )
    initial = json.loads(
        (catalog / "progress.json").read_text(encoding="utf-8")
    )
    for root in [*initial["configuration"]["roots"], str(catalog)]:
        if any(
            path == Path(root).resolve()
            or path.is_relative_to(Path(root).resolve())
            for path in (output, staging)
        ):
            raise ValueError(
                "Output and staging must be outside the original raw and catalog directories."
            )
    output.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)

    def progress(status, **values):
        atomic_json(
            output / "progress.json",
            {
                "status": status,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                **values,
            },
        )

    try:
        while True:
            catalog_progress = json.loads(
                (catalog / "progress.json").read_text(encoding="utf-8")
            )
            if catalog_progress["status"] == "complete" or (
                arguments.allow_limited_catalog
                and catalog_progress["status"] == "limited"
            ):
                break
            if not arguments.wait_for_catalog or catalog_progress[
                "status"
            ] in ("failed", "paused"):
                raise RuntimeError(
                    f"Catalog is not complete: {catalog_progress['status']}"
                )
            progress(
                "waiting_for_catalog",
                usable_games=catalog_progress["usable_games"],
            )
            time.sleep(30)
        for root in catalog_progress["configuration"]["roots"]:
            if any(
                path == Path(root).resolve()
                or path.is_relative_to(Path(root).resolve())
                for path in (output, staging)
            ):
                raise ValueError(
                    "Output and staging must be outside the original raw directories."
                )
        implementation = [
            Path(__file__),
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
            "catalog": catalog_progress["configuration"],
            "catalog_status": catalog_progress["status"],
            "game_order": "semantic_sha256,uuid",
            "bc_row_order": "deterministic_shuffle_within_shard",
            "python_version": sys.version,
            "manifests": {
                split: digest(catalog / f"manifest.{split}.jsonl")
                for split in SPLITS
            },
            "annotator_sha256": digest(arguments.annotator),
            "implementation": {
                path.name + ":" + str(index): digest(path)
                for index, path in enumerate(implementation)
            },
            "shard_games": arguments.shard_games,
            "limit_games": arguments.limit_games,
            "max_rejection_rate": arguments.max_rejection_rate,
        }
        configuration_path = output / "configuration.json"
        if (
            configuration_path.exists()
            and json.loads(configuration_path.read_text(encoding="utf-8"))
            != configuration
        ):
            raise ValueError(
                "Prepared output belongs to a different source, implementation or shard configuration."
            )
        atomic_json(configuration_path, configuration)
        settings = {
            "output": str(output),
            "staging": str(staging),
            "annotator": str(arguments.annotator.resolve()),
            "repository": str(arguments.repository.resolve()),
            "reserve": arguments.reserve_gib * 1024**3,
            "fingerprint": json_digest(configuration),
        }
        receipts = []
        progress("ordering_games")
        jobs = iter(
            chunks(catalog, arguments.shard_games, arguments.limit_games)
        )
        with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
            pending = set()
            exhausted = False
            while pending or not exhausted:
                while len(pending) < arguments.workers and not exhausted:
                    job = next(jobs, None)
                    if job is None:
                        exhausted = True
                    else:
                        pending.add(
                            executor.submit(build_shard, job, settings)
                        )
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    receipts.append(future.result())
                attempted = sum(value["attempted_games"] for value in receipts)
                rejected = sum(value["rejected_games"] for value in receipts)
                progress(
                    "annotating",
                    shards=len(receipts),
                    attempted_games=attempted,
                    accepted_games=attempted - rejected,
                    rejected_games=rejected,
                )
                if (
                    attempted >= min(64, arguments.limit_games or 64)
                    and rejected / attempted > arguments.max_rejection_rate
                ):
                    raise RuntimeError(
                        "Annotation rejection rate exceeds the configured limit; inspect per-shard rejections before continuing."
                    )
                if (output / "pause.request").exists():
                    for future in pending:
                        future.result()
                    progress("paused", shards=len(receipts))
                    return
        if not receipts or not sum(
            receipt["accepted_games"] for receipt in receipts
        ):
            raise RuntimeError("No usable complete games were annotated.")
        progress("assembling", shards=len(receipts))
        files = assemble(output, receipts, settings["reserve"])
        if not arguments.limit_games and any(
            not files[split][kind]["rows"]
            for split in SPLITS
            for kind in KINDS
        ):
            raise RuntimeError(
                "A full training/validation/test partition is empty."
            )
        report = {
            "configuration": configuration,
            "attempted_games": sum(
                value["attempted_games"] for value in receipts
            ),
            "accepted_games": sum(
                value["accepted_games"] for value in receipts
            ),
            "rejected_games": sum(
                value["rejected_games"] for value in receipts
            ),
            "terminal_trajectories": sum(
                value["terminal_trajectories"] for value in receipts
            ),
            "files": files,
        }
        atomic_json(output / "dataset.json", report)
        progress(
            "complete",
            accepted_games=report["accepted_games"],
            rejected_games=report["rejected_games"],
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "accepted_games": report["accepted_games"],
                    "output": str(output),
                }
            ),
            flush=True,
        )
    except BaseException as error:
        progress("failed", error_type=type(error).__name__, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--annotator", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard-games", type=int, default=64)
    parser.add_argument("--limit-games", type=int, default=0)
    parser.add_argument("--reserve-gib", type=int, default=20)
    parser.add_argument("--max-rejection-rate", type=float, default=0.02)
    parser.add_argument("--allow-limited-catalog", action="store_true")
    parser.add_argument("--wait-for-catalog", action="store_true")
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
            except BaseException:
                traceback.print_exc()
                sys.exit(1)


if __name__ == "__main__":
    main()
