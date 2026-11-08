"""Read-only raw-game inventory, semantic deduplication and whole-game splits."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time

from .records import inspect_record, is_record_name, load_schema


CATALOG_VERSION = 1
DATED_DIRECTORY = re.compile(r"(\d{4}-\d{2}-\d{2})-(\d+)")


def split_game(uuid, salt="kanachan-jade-hanchan-v1"):
    bucket = (
        int.from_bytes(
            hashlib.sha256((salt + "|" + uuid.lower()).encode()).digest()[:8],
            "big",
        )
        % 10000
    )
    return (
        "train" if bucket < 9000 else "validation" if bucket < 9500 else "test"
    )


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    for attempt in range(100):
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            if (
                os.name != "nt"
                or getattr(error, "winerror", None) not in (5, 32)
                or attempt == 99
            ):
                raise
            time.sleep(0.02)


def lower_priority():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.SetPriorityClass.restype = wintypes.BOOL
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise ctypes.WinError(ctypes.get_last_error())
    elif hasattr(os, "nice"):
        os.nice(10)


def directories(roots):
    for root in roots:
        root = Path(root).resolve()
        found_dated = False
        selected = []
        with os.scandir(root) as entries:
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                match = DATED_DIRECTORY.fullmatch(entry.name)
                if match:
                    found_dated = True
                    # Historical directory suffixes are not trustworthy:
                    # real mode-12 games also exist under suffix 16, and
                    # mode-9 games under suffix 12. Inspect each game header.
                    selected.append(Path(entry.path))
        if found_dated:
            yield root
            yield from sorted(selected)
        else:
            yield root


class Catalog:
    def __init__(self, output, configuration):
        self.output = Path(output).resolve()
        if any(
            self.output == Path(root).resolve()
            or self.output.is_relative_to(Path(root).resolve())
            for root in configuration["roots"]
        ):
            raise ValueError(
                "Catalog output must be outside every raw source directory."
            )
        self.output.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.output / "catalog.sqlite3")
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA cache_size=-32768;
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS directories (path TEXT PRIMARY KEY, mtime_ns INTEGER, done INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, bytes INTEGER, mtime_ns INTEGER, uuid TEXT,
                outcome TEXT NOT NULL, reason TEXT, sha256 TEXT, semantic_sha256 TEXT);
            CREATE TABLE IF NOT EXISTS games (uuid TEXT PRIMARY KEY, path TEXT NOT NULL, split TEXT NOT NULL,
                semantic_sha256 TEXT NOT NULL, mode_id INTEGER, start_time INTEGER, end_time INTEGER,
                conflicted INTEGER NOT NULL DEFAULT 0, metadata TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS game_split ON games(conflicted,split,uuid);
        """)
        serialized = json.dumps(configuration, sort_keys=True)
        previous = self.db.execute(
            "SELECT value FROM meta WHERE key='configuration'"
        ).fetchone()
        if previous is not None and previous[0] != serialized:
            raise ValueError(
                "Existing catalog has different inputs, schema or split settings. Use a new output directory."
            )
        self.db.execute(
            "INSERT OR IGNORE INTO meta VALUES('configuration',?)",
            (serialized,),
        )
        self.configuration = configuration
        self.counts = dict(
            self.db.execute(
                "SELECT outcome, COUNT(*) FROM files GROUP BY outcome"
            )
        )
        self.unique = self.db.execute("SELECT COUNT(*) FROM games").fetchone()[
            0
        ]
        self.conflicts = self.db.execute(
            "SELECT COUNT(*) FROM games WHERE conflicted=1"
        ).fetchone()[0]
        self.db.commit()

    def unchanged(self, path, stat):
        previous = self.db.execute(
            "SELECT bytes,mtime_ns FROM files WHERE path=?", (str(path),)
        ).fetchone()
        if previous is None:
            return False
        if previous != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError(
                f"A previously indexed raw file changed: {path}"
            )
        return True

    def add(self, record):
        outcome = record["outcome"]
        if outcome == "candidate":
            existing = self.db.execute(
                "SELECT semantic_sha256,conflicted FROM games WHERE uuid=?",
                (record["uuid"],),
            ).fetchone()
            if existing is None:
                record["split"] = split_game(
                    record["uuid"], self.configuration["split_salt"]
                )
                self.db.execute(
                    "INSERT INTO games VALUES(?,?,?,?,?,?,?,0,?)",
                    (
                        record["uuid"],
                        record["path"],
                        record["split"],
                        record["semantic_sha256"],
                        record["mode_id"],
                        record["start_time"],
                        record["end_time"],
                        json.dumps(record, sort_keys=True),
                    ),
                )
                self.unique += 1
                outcome = "selected"
            elif existing[0] == record["semantic_sha256"]:
                outcome = "duplicate"
            else:
                outcome = "conflict"
                if not existing[1]:
                    self.db.execute(
                        "UPDATE games SET conflicted=1 WHERE uuid=?",
                        (record["uuid"],),
                    )
                    self.conflicts += 1
        self.db.execute(
            "INSERT INTO files VALUES(?,?,?,?,?,?,?,?)",
            (
                record["path"],
                record.get("bytes"),
                record.get("mtime_ns"),
                record.get("uuid"),
                outcome,
                record.get("reason"),
                record.get("sha256"),
                record.get("semantic_sha256"),
            ),
        )
        self.counts[outcome] = self.counts.get(outcome, 0) + 1

    def progress(self, *, status, current=None, started=None):
        self.db.commit()
        totals = dict(
            self.db.execute(
                "SELECT done,COUNT(*) FROM directories GROUP BY done"
            )
        )
        value = {
            "status": status,
            "pid": os.getpid(),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "counts": self.counts,
            "unique_games": self.unique,
            "conflicted_games": self.conflicts,
            "usable_games": self.unique - self.conflicts,
            "directories_total": sum(totals.values()),
            "directories_done": totals.get(1, 0),
            "current_directory": current,
            "elapsed_seconds_this_session": None
            if started is None
            else time.monotonic() - started,
            "configuration": self.configuration,
        }
        atomic_json(self.output / "progress.json", value)
        return value

    def export(self):
        counts = {}
        for split in ("train", "validation", "test"):
            path = self.output / f"manifest.{split}.jsonl"
            temporary = path.with_name(path.name + ".tmp")
            count = 0
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                for (metadata,) in self.db.execute(
                    "SELECT metadata FROM games WHERE conflicted=0 AND split=? ORDER BY uuid",
                    (split,),
                ):
                    stream.write(metadata + "\n")
                    count += 1
            temporary.replace(path)
            counts[split] = count
        atomic_json(
            self.output / "summary.json",
            {
                "configuration": self.configuration,
                "file_outcomes": self.counts,
                "splits": counts,
                "conflicted_games": self.conflicts,
                "modes": dict(
                    self.db.execute(
                        "SELECT mode_id,COUNT(*) FROM games WHERE conflicted=0 GROUP BY mode_id"
                    )
                ),
                "first_end_time": self.db.execute(
                    "SELECT MIN(end_time) FROM games WHERE conflicted=0"
                ).fetchone()[0],
                "last_end_time": self.db.execute(
                    "SELECT MAX(end_time) FROM games WHERE conflicted=0"
                ).fetchone()[0],
                "rejection_reasons": dict(
                    self.db.execute(
                        "SELECT reason,COUNT(*) FROM files WHERE outcome='rejected' GROUP BY reason"
                    )
                ),
            },
        )
        return counts


def run(arguments):
    if not 1 <= arguments.workers <= 4 or arguments.batch_size < 1:
        raise ValueError("Use 1-4 workers and a positive batch size.")
    if arguments.limit < 0 or arguments.max_record_bytes <= 3:
        raise ValueError(
            "Use a nonnegative limit and a record size greater than three bytes."
        )
    lower_priority()
    schema = load_schema(arguments.schema_dir)
    implementation = hashlib.sha256()
    for name in ("catalog.py", "records.py"):
        implementation.update((Path(__file__).parent / name).read_bytes())
    configuration = {
        "version": CATALOG_VERSION,
        "roots": [str(Path(root).resolve()) for root in arguments.raw_root],
        "modes": [12, 16],
        "category": 2,
        "players": 4,
        "whole_game_split": [90, 5, 5],
        "split_salt": arguments.split_salt,
        "max_record_bytes": arguments.max_record_bytes,
        "schema_sha256": hashlib.sha256(
            (Path(arguments.schema_dir) / "mahjongsoul_pb2.py").read_bytes()
        ).hexdigest(),
        "implementation_sha256": implementation.hexdigest(),
    }
    catalog = Catalog(arguments.output, configuration)
    started = last_progress = time.monotonic()
    inspected = 0
    try:
        for directory in directories(arguments.raw_root):
            stat = directory.stat()
            previous = catalog.db.execute(
                "SELECT mtime_ns FROM directories WHERE path=?",
                (str(directory),),
            ).fetchone()
            if previous is not None and previous[0] != stat.st_mtime_ns:
                raise RuntimeError(
                    f"An indexed source directory changed: {directory}"
                )
            catalog.db.execute(
                "INSERT OR IGNORE INTO directories(path,mtime_ns) VALUES(?,?)",
                (str(directory), stat.st_mtime_ns),
            )
        catalog.db.commit()
        pending = [
            Path(row[0])
            for row in catalog.db.execute(
                "SELECT path FROM directories WHERE done=0 ORDER BY path"
            )
        ]
        with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
            for directory in pending:
                batch = []

                def flush():
                    nonlocal inspected, last_progress
                    for record in executor.map(
                        lambda item: inspect_record(
                            item[0],
                            schema,
                            max_bytes=arguments.max_record_bytes,
                            source_stat=item[1],
                        ),
                        batch,
                    ):
                        catalog.add(record)
                        inspected += 1
                    batch.clear()
                    if time.monotonic() - last_progress >= 5:
                        value = catalog.progress(
                            status="indexing",
                            current=str(directory),
                            started=started,
                        )
                        print(
                            json.dumps(
                                {
                                    key: value[key]
                                    for key in (
                                        "status",
                                        "counts",
                                        "usable_games",
                                        "directories_done",
                                        "directories_total",
                                    )
                                }
                            ),
                            flush=True,
                        )
                        last_progress = time.monotonic()

                with os.scandir(directory) as entries:
                    for entry in entries:
                        if not entry.is_file(
                            follow_symlinks=False
                        ) or not is_record_name(entry.name):
                            continue
                        path = Path(entry.path)
                        source_stat = entry.stat(follow_symlinks=False)
                        if catalog.unchanged(path, source_stat):
                            continue
                        batch.append((path, source_stat))
                        if len(batch) >= arguments.batch_size:
                            flush()
                        if (Path(arguments.output) / "pause.request").exists():
                            if batch:
                                flush()
                            catalog.progress(
                                status="paused",
                                current=str(directory),
                                started=started,
                            )
                            return
                        if (
                            arguments.limit
                            and inspected + len(batch) >= arguments.limit
                        ):
                            if batch:
                                flush()
                            catalog.export()
                            catalog.progress(
                                status="limited",
                                current=str(directory),
                                started=started,
                            )
                            return
                if batch:
                    flush()
                catalog.db.execute(
                    "UPDATE directories SET done=1 WHERE path=?",
                    (str(directory),),
                )
                catalog.progress(
                    status="indexing", current=str(directory), started=started
                )
        counts = catalog.export()
        catalog.progress(status="complete", started=started)
        print(json.dumps({"status": "complete", "splits": counts}), flush=True)
    except BaseException as error:
        catalog.progress(status="failed", started=started)
        atomic_json(
            Path(arguments.output) / "error.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        catalog.db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root", action="append", type=Path, required=True
    )
    parser.add_argument("--schema-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--max-record-bytes", type=int, default=32 * 1024 * 1024
    )
    parser.add_argument("--split-salt", default="kanachan-jade-hanchan-v1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-file", type=Path)
    arguments = parser.parse_args()
    if arguments.log_file is None:
        run(arguments)
    else:
        arguments.log_file.parent.mkdir(parents=True, exist_ok=True)
        with arguments.log_file.open(
            "a", encoding="utf-8", buffering=1
        ) as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                try:
                    run(arguments)
                except BaseException:
                    import traceback

                    traceback.print_exc()
                    sys.exit(1)


if __name__ == "__main__":
    main()
