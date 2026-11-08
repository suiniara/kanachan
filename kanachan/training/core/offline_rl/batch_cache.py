"""Bounded, single-file cache of parsed sequential RL batches.

It stores a fixed window until full, so a long sequential scan cannot evict
the same useful window before the next scan reaches it. Cache misses always
fall through to the original reader; sampling and rewards are unaffected.
"""

import hashlib
import io
import json
import logging
from pathlib import Path
import sqlite3

import torch

import kanachan.constants as constants
from kanachan.training.core import features
from . import data_iterator
from kanachan.training.online.state import file_digest


class CachedBatches:
    def __init__(self, *, path, source, start, batch_size, byte_budget,
                 rewrite_rooms, rewrite_grades, first_reader, reader_factory):
        self.path, self.source = Path(path).resolve(), Path(source).resolve()
        if self.path == self.source or not self.path.parent.is_dir():
            raise ValueError("Batch cache needs its own file in an existing directory.")
        if type(byte_budget) is not int or byte_budget < 65536:
            raise ValueError("Batch cache byte budget must be at least 65536.")
        self.cursor = self.reader_cursor = start
        self.reader, self.factory = first_reader, reader_factory
        self.batch_size, self.budget = batch_size, byte_budget
        self.stats = {"hits": 0, "misses": 0, "writes": 0, "payload_bytes": 0}
        self.signature = self._signature()
        identity = {
            "format_version": 1, "source": str(self.source),
            "source_sha256": file_digest(self.source), "signature": self.signature,
            "batch_size": batch_size, "byte_budget": byte_budget,
            "rewrite_rooms": rewrite_rooms, "rewrite_grades": rewrite_grades,
            "reader": {
                module.__name__: file_digest(module.__file__)
                for module in (constants, features, data_iterator)
            },
        }
        if self._signature() != self.signature:
            raise ValueError("Source changed while identifying the batch cache.")
        serialized = json.dumps(identity, sort_keys=True)
        existed = self.path.exists()
        self.connection = sqlite3.connect(self.path, timeout=30)
        try:
            if existed:
                tables = {row[0] for row in self.connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )}
                if tables != {"meta", "blocks"}:
                    raise ValueError("Refusing to alter an unrecognized cache database.")
                saved = self.connection.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
                if saved is None or saved[0] != serialized:
                    raise ValueError("Batch cache source, reader or configuration differs.")
            else:
                with self.connection:
                    self.connection.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                    self.connection.execute("CREATE TABLE blocks(start INTEGER PRIMARY KEY, rows INTEGER NOT NULL, sha256 TEXT NOT NULL, payload BLOB NOT NULL)")
                    self.connection.executemany("INSERT INTO meta VALUES (?,?)", (
                        ("identity", serialized), ("full", "0"),
                    ))
            self.page_size = self.connection.execute("PRAGMA page_size").fetchone()[0]
            maximum = byte_budget // self.page_size
            actual = self.connection.execute(f"PRAGMA max_page_count={maximum}").fetchone()[0]
            if actual > maximum:
                raise ValueError("Existing batch cache exceeds its configured byte budget.")
            self.full = self.connection.execute("SELECT value FROM meta WHERE key='full'").fetchone()[0] == "1"
        except BaseException:
            self.connection.close()
            self.connection = None
            raise

    def _signature(self):
        stat = self.source.stat()
        return [stat.st_size, stat.st_mtime_ns]

    def __iter__(self):
        return self

    def _store(self, start, data):
        if self.full:
            return
        buffer = io.BytesIO()
        torch.save(data, buffer)
        payload = buffer.getvalue()
        # Reserve B-tree/metadata pages as well as payload pages. The SQLite
        # page limit is an additional hard bound on the database file itself.
        pages = self.connection.execute("PRAGMA page_count").fetchone()[0]
        needed = (len(payload) + self.page_size - 1) // self.page_size + 8
        if (pages + needed) * self.page_size > self.budget:
            with self.connection:
                self.connection.execute("UPDATE meta SET value='1' WHERE key='full'")
            self.full = True
            return
        with self.connection:
            self.connection.execute("INSERT INTO blocks VALUES (?,?,?,?)", (
                start, 0 if data is None else len(data[0]),
                hashlib.sha256(payload).hexdigest(), payload,
            ))
        self.stats["writes"] += 1
        self.stats["payload_bytes"] += len(payload)

    def __next__(self):
        try:
            return self._next()
        except BaseException:
            self.close()
            raise

    def _next(self):
        if self.connection is None:
            raise StopIteration
        if self._signature() != self.signature:
            raise ValueError("Source metadata changed while using the batch cache.")
        cached = self.connection.execute(
            "SELECT rows,sha256,payload FROM blocks WHERE start=?", (self.cursor,)
        ).fetchone()
        if cached is not None:
            count, checksum, payload = cached
            if hashlib.sha256(payload).hexdigest() != checksum:
                raise ValueError("Parsed batch cache checksum differs.")
            data = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
            self.stats["hits"] += 1
            if data is not None and (
                not isinstance(data, list) or len(data) != 13
                or not all(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                           and len(value) == count for value in data)
                or not 0 < count <= self.batch_size
            ):
                raise ValueError("Invalid parsed batch cache schema.")
            if data is None and count != 0:
                raise ValueError("Invalid parsed batch cache end marker.")
        else:
            self.stats["misses"] += 1
            if self.reader_cursor != self.cursor:
                # A cache hit can jump over raw parsing. Reopening a zero-worker
                # iterator at a later raw cursor must not consume learner RNG.
                with torch.random.fork_rng(devices=[]):
                    self.reader = self.factory(self.cursor)
                self.reader_cursor = self.cursor
            try:
                data = next(self.reader)
                self.reader_cursor += len(data[0])
            except StopIteration:
                data = None
            self._store(self.cursor, data)
        if data is None:
            self.close()
            raise StopIteration
        self.cursor += len(data[0])
        return data

    def close(self):
        if getattr(self, "connection", None) is not None:
            self.connection.close()
            self.connection = None
            self.reader = None
            logging.getLogger(__name__).info("Parsed batch cache: %s", self.stats)

    def __del__(self):
        self.close()
