"""Incremental catalog consumption, recovery and late duplicate conflicts."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from kanachan.data.catalog import split_game
from kanachan.data.stream import Ledger, catalog_status, shared_reader


class StreamingLedgerTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.catalog = self.root / "catalog"
        self.catalog.mkdir()
        self.source = sqlite3.connect(self.catalog / "catalog.sqlite3")
        self.addCleanup(self.source.close)
        self.source.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE games(uuid TEXT PRIMARY KEY, split TEXT,
                semantic_sha256 TEXT, metadata TEXT, conflicted INTEGER);
            CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE directories(path TEXT PRIMARY KEY, done INTEGER);
            INSERT INTO directories VALUES('raw',0);
        """)
        self.configuration = {"catalog": {"split_salt": "stream-test"}}
        self.source.execute(
            "INSERT INTO meta VALUES('configuration',?)",
            (json.dumps(self.configuration["catalog"]),),
        )
        self.source.commit()
        self.ledger = Ledger(
            self.root / "state", self.catalog, self.configuration
        )
        self.addCleanup(lambda: self.ledger.close())

    def add(self, number):
        uuid = f"200401-00000000-0000-0000-0000-{number:012d}"
        record = {
            "uuid": uuid,
            "split": split_game(uuid, "stream-test"),
            "semantic_sha256": f"{number:064x}",
            "mode_id": 12,
        }
        self.source.execute(
            "INSERT INTO games VALUES(?,?,?,?,0)",
            (
                uuid,
                record["split"],
                record["semantic_sha256"],
                json.dumps(record),
            ),
        )
        self.source.commit()
        return record

    def complete(self, job):
        self.ledger.complete(
            job[0],
            {
                "attempted_games": len(job[2]),
                "accepted_games": len(job[2]),
                "rejected_games": 0,
                "files": {
                    kind: {"rows": len(job[2])} for kind in ("bc", "rl", "eor")
                },
            },
        )

    def test_processes_and_resumes_before_source_is_sealed(self):
        record = self.add(1)
        self.assertEqual(self.ledger.ingest(limit=1), 1)
        job = self.ledger.assign(record["split"], 1)
        self.assertEqual(job[2], [record])
        self.ledger.close()
        self.ledger = Ledger(
            self.root / "state", self.catalog, self.configuration
        )
        self.assertEqual(list(self.ledger.assigned()), [job])
        self.assertEqual(self.ledger.ingest(), 0)
        self.complete(job)
        second = self.add(2)
        self.assertEqual(self.ledger.ingest(), 1)
        next_job = self.ledger.assign(second["split"], 1)
        self.assertNotEqual(job[0], next_job[0])
        self.complete(next_job)
        self.assertEqual(self.ledger.coverage()["games"], 2)

    def test_late_conflict_removes_previously_converted_uuid(self):
        record = self.add(1)
        self.ledger.ingest()
        job = self.ledger.assign(record["split"], 1)
        self.complete(job)
        self.source.execute(
            "UPDATE games SET conflicted=1 WHERE uuid=?", (record["uuid"],)
        )
        self.source.commit()
        self.ledger.refresh_conflicts()
        self.assertIn(record["uuid"], self.ledger.known_conflicts)
        self.assertEqual(self.ledger.coverage()["games"], 0)
        self.assertEqual(
            self.ledger.db.execute(
                "SELECT invalid,batch_id FROM games"
            ).fetchone(),
            (1, job[0]),
        )

    def test_pending_count_stays_covered_after_assignment_and_conflict(self):
        records = [self.add(number) for number in range(1, 9)]
        self.assertEqual(self.ledger.ingest(), 8)
        query = "SELECT COUNT(*) FROM games WHERE batch_id IS NULL AND invalid=0"
        plan = " ".join(
            row[-1]
            for row in self.ledger.db.execute("EXPLAIN QUERY PLAN " + query)
        )
        self.assertIn("USING COVERING INDEX", plan)
        self.assertEqual(self.ledger.pending(), 8)
        job = self.ledger.assign(records[0]["split"], 1)
        self.assertEqual(job[2], [records[0]])
        self.assertEqual(self.ledger.pending(), 7)
        self.source.execute(
            "UPDATE games SET conflicted=1 WHERE uuid=?",
            (records[1]["uuid"],),
        )
        self.source.commit()
        self.ledger.refresh_conflicts()
        self.assertEqual(self.ledger.pending(), 6)
        self.ledger.close()
        self.ledger = Ledger(
            self.root / "state", self.catalog, self.configuration
        )
        self.assertEqual(self.ledger.pending(), 6)
        self.assertEqual(list(self.ledger.assigned()), [job])
        plan = " ".join(
            row[-1]
            for row in self.ledger.db.execute("EXPLAIN QUERY PLAN " + query)
        )
        self.assertIn("USING COVERING INDEX", plan)

    def test_wrong_split_and_changed_configuration_fail_closed(self):
        record = self.add(1)
        wrong = "test" if record["split"] != "test" else "train"
        self.source.execute("UPDATE games SET split=?", (wrong,))
        self.source.commit()
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.ledger.ingest()
        with self.assertRaisesRegex(ValueError, "changed"):
            Ledger(
                self.root / "state",
                self.catalog,
                {"catalog": {"split_salt": "changed"}},
            )

    def test_reader_allows_atomic_progress_replacement(self):
        path = self.root / "progress.json"
        path.write_text('{"version":1}', encoding="utf-8")
        with shared_reader(path) as reader:
            replacement = self.root / "replacement.json"
            replacement.write_text('{"version":2}', encoding="utf-8")
            replacement.replace(path)
            self.assertEqual(json.load(reader), {"version": 1})
        self.assertEqual(json.loads(path.read_text()), {"version": 2})

    def test_completion_requires_published_manifests_and_finished_directories(
        self,
    ):
        record = self.add(1)
        self.assertEqual(catalog_status(self.catalog)["status"], "indexing")
        summary = {
            "configuration": self.configuration["catalog"],
            "splits": {record["split"]: 1},
        }
        (self.catalog / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        self.assertEqual(catalog_status(self.catalog)["status"], "indexing")
        self.source.execute("UPDATE directories SET done=1")
        self.source.commit()
        with self.assertRaisesRegex(RuntimeError, "manifests"):
            catalog_status(self.catalog)
        for split in ("train", "validation", "test"):
            (self.catalog / f"manifest.{split}.jsonl").touch()
        self.assertEqual(catalog_status(self.catalog)["status"], "complete")


if __name__ == "__main__":
    unittest.main()
