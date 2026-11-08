"""Independent checks for coverage, BC/RL identity, gzip corruption and resume."""

import gzip
import hashlib
import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from kanachan import constants
from kanachan.data import validation
from kanachan.data.audit import audit, digest, expected_eor
from kanachan.data.catalog import Catalog, split_game
from kanachan.training.core import features


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.catalog = self.root / "catalog"
        self.output = self.root / "dataset"
        self.output.mkdir()
        self.state = self.root / "audit-state"
        self.config = {
            "version": 1,
            "roots": [str(self.raw)],
            "category": 2,
            "players": 4,
            "modes": [12, 16],
            "whole_game_split": [90, 5, 5],
            "split_salt": "audit-test",
        }
        source = Catalog(self.catalog, self.config)
        self.records = {}
        index = 1
        while len(self.records) < 3:
            record = self.record(index)
            split = split_game(record["uuid"], self.config["split_salt"])
            if split not in self.records:
                source.add(record)
                self.records[split] = record
            index += 1
        source.db.execute(
            "INSERT INTO directories VALUES(?,0,1)", (str(self.raw),)
        )
        source.export()
        source.progress(status="complete")
        checksum = hashlib.sha256()
        for row in source.db.execute(
            "SELECT uuid,split,semantic_sha256,metadata FROM games WHERE conflicted=0 ORDER BY uuid"
        ):
            checksum.update(
                json.dumps(
                    row, ensure_ascii=False, separators=(",", ":")
                ).encode()
                + b"\n"
            )
        source.db.close()
        self.report = {
            "configuration": {
                "version": 1,
                "catalog": self.config,
                "catalog_path": str(self.catalog),
                "implementation": {
                    "2:validation.py": digest(validation.__file__),
                    "3:constants.py": digest(constants.__file__),
                    "4:features.py": digest(features.__file__),
                },
            },
            "source_status": "complete",
            "coverage": {"games": 3, "sha256": checksum.hexdigest()},
            "accepted_games": 3,
            "rejected_games": 0,
            "conflicted_games": 0,
            "terminal_trajectories": 3,
            "source_manifests": {},
            "files": {},
        }
        self.receipts = {}
        self.bc_columns = {}
        for split, record in self.records.items():
            directory = self.output / "shards" / split / "000000"
            directory.mkdir(parents=True)
            (directory / "games.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            (directory / "rejections.jsonl").write_text("", encoding="utf-8")
            columns = [
                record["uuid"],
                "3,6,16,32,48,64,71,75,78,151,152",
                "0,0,25000,25000,25000,25000",
                "0",
                "0,4",
                "1",
                "0",
                "0,0,0,0,0,0,25000,25000,25000,25000,25000,25000,25000,25000",
            ]
            following = columns.copy()
            sparse = following[1].split(",")
            sparse[8] = "79"
            following[1] = ",".join(sparse)
            self.bc_columns[split] = [columns, following]
            bc = "\t".join(columns) + "\n" + "\t".join(following) + "\n"
            transition = columns[:6] + following[1:5] + [
                columns[6], ",".join(columns[7].split(",")[:10])
            ]
            rl = "\t".join(transition) + "\n" + "\t".join(following) + "\n"
            eor = (
                "\t".join(
                    [
                        record["uuid"],
                        "3,6,16,32,48,64,71,74",
                        columns[2],
                        "25000,25000,25000,25000",
                    ]
                )
                + "\n"
            )
            receipt = {
                "split": split,
                "shard": 0,
                "accepted_games": 1,
                "rejected_games": 0,
                "attempted_games": 1,
                "terminal_trajectories": 1,
                "files": {},
            }
            self.receipts[split] = receipt
            self.report["files"][split] = {}
            for kind, content in (("bc", bc), ("rl", rl), ("eor", eor)):
                self.update_file(split, kind, content)
            self.report["source_manifests"][split] = digest(
                self.catalog / f"manifest.{split}.jsonl"
            )
        self.save_report()

    def record(self, number):
        uuid = f"200401-00000000-0000-0000-0000-{number:012d}"
        return {
            "uuid": uuid,
            "path": str(self.raw / uuid),
            "bytes": 1,
            "mtime_ns": 1,
            "sha256": "0" * 64,
            "semantic_sha256": hashlib.sha256(uuid.encode()).hexdigest(),
            "category": 2,
            "mode_id": 12,
            "start_time": 1,
            "end_time": 2,
            "level_ids": [10401] * 4,
            "scores": [25000] * 4,
            "outcome": "candidate",
        }

    def save_report(self):
        (self.output / "dataset.json").write_text(
            json.dumps(self.report), encoding="utf-8"
        )

    def update_file(self, split, kind, content):
        directory = self.output / "shards" / split / "000000"
        path = directory / f"{kind}.tsv.gz"
        path.write_bytes(gzip.compress(content.encode(), mtime=0))
        target = self.output / f"{split}.{kind}.tsv.gz"
        shutil.copyfile(path, target)
        values = {
            "bytes": path.stat().st_size,
            "rows": content.count("\n"),
            "sha256": digest(path),
        }
        self.receipts[split]["files"][kind] = {**values, "name": path.name}
        self.report["files"][split][kind] = {**values, "path": str(target)}
        (directory / "receipt.json").write_text(
            json.dumps(self.receipts[split]), encoding="utf-8"
        )

    def test_complete_audit_and_cached_resume(self):
        result = audit(self.output, self.catalog, self.state)
        self.assertEqual(result["accepted_games"], 3)
        self.assertTrue(result["all_nine_gzip_files_crc_sha256_rows_verified"])
        self.assertTrue(result["bc_eor_nonfinal_rounds_identical"])
        self.assertEqual(result["eor_zero_target_games"], 0)
        self.assertEqual(
            audit(self.output, self.catalog, self.state)["reused_shards"], 3
        )

    def test_valid_but_different_rl_action_is_rejected(self):
        path = self.output / "train.rl.tsv.gz"
        with gzip.open(path, "rt") as stream:
            lines = list(stream)
        columns = lines[0].rstrip("\n").split("\t")
        columns[5] = "0"
        lines[0] = "\t".join(columns) + "\n"
        self.update_file("train", "rl", "".join(lines))
        self.save_report()
        with self.assertRaisesRegex(ValueError, "decision multiset"):
            audit(self.output, self.catalog, self.state)

    def test_eor_round_scores_must_match_bc(self):
        with gzip.open(self.output / "train.eor.tsv.gz", "rt") as stream:
            columns = stream.read().rstrip("\n").split("\t")
        numeric = columns[2].split(",")
        numeric[2], numeric[3] = "25100", "24900"
        columns[2] = ",".join(numeric)
        self.update_file("train", "eor", "\t".join(columns) + "\n")
        self.save_report()
        with self.assertRaisesRegex(ValueError, "BC/EOR"):
            audit(self.output, self.catalog, self.state)

    def test_eor_round_cannot_be_duplicated(self):
        with gzip.open(self.output / "train.eor.tsv.gz", "rt") as stream:
            content = stream.read()
        self.update_file("train", "eor", content * 2)
        self.save_report()
        with self.assertRaisesRegex(ValueError, "BC/EOR"):
            audit(self.output, self.catalog, self.state)

    def test_missing_nonfinal_eor_round_is_rejected(self):
        self.update_file("train", "eor", "")
        self.save_report()
        with self.assertRaisesRegex(ValueError, "BC/EOR"):
            audit(self.output, self.catalog, self.state)

    def test_final_round_cannot_be_included_as_eor(self):
        with gzip.open(self.output / "train.eor.tsv.gz", "rt") as stream:
            content = stream.read()
        columns = content.rstrip("\n").split("\t")
        sparse = columns[1].split(",")
        sparse[-1] = "75"
        columns[1] = ",".join(sparse)
        self.update_file("train", "eor", content + "\t".join(columns) + "\n")
        self.save_report()
        with self.assertRaisesRegex(ValueError, "BC/EOR"):
            audit(self.output, self.catalog, self.state)

    def test_one_round_game_has_no_eor_target(self):
        columns = self.bc_columns["train"][0]
        self.assertFalse(expected_eor(["\t".join(columns) + "\n"]))

    def test_one_round_game_cannot_invent_an_eor_target(self):
        content = "\t".join(self.bc_columns["train"][0]) + "\n"
        self.update_file("train", "bc", content)
        self.update_file("train", "rl", content)
        self.save_report()
        with self.assertRaisesRegex(ValueError, "BC/EOR"):
            audit(self.output, self.catalog, self.state)

    def test_bc_shuffle_preserves_eor_targets(self):
        lines = ["\t".join(columns) + "\n" for columns in self.bc_columns["train"]]
        self.assertEqual(expected_eor(lines), expected_eor(reversed(lines)))

    def test_duplicate_bc_round_start_is_rejected(self):
        line = "\t".join(self.bc_columns["train"][0]) + "\n"
        with self.assertRaisesRegex(ValueError, "round-start"):
            expected_eor([line, line])

    def test_unprocessed_source_game_is_rejected(self):
        source = Catalog(self.catalog, self.config)
        source.add(self.record(999999))
        source.db.commit()
        source.db.close()
        with self.assertRaisesRegex(ValueError, "UUID coverage"):
            audit(self.output, self.catalog, self.state)

    def test_cached_shards_are_rehashed(self):
        audit(self.output, self.catalog, self.state)
        path = self.output / "shards/train/000000/bc.tsv.gz"
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "audited shard changed"):
            audit(self.output, self.catalog, self.state)

    def test_final_gzip_is_read_even_on_resume(self):
        audit(self.output, self.catalog, self.state)
        path = self.output / "test.eor.tsv.gz"
        path.write_bytes(path.read_bytes()[:-4])
        with self.assertRaises((EOFError, ValueError)):
            audit(self.output, self.catalog, self.state)

    def test_unfinished_catalog_cannot_pass(self):
        connection = sqlite3.connect(self.catalog / "catalog.sqlite3")
        try:
            connection.execute("UPDATE directories SET done=0")
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(ValueError, "unfinished"):
            audit(self.output, self.catalog, self.state)

    def test_state_cannot_write_into_raw_inputs(self):
        state = self.raw / "audit"
        with self.assertRaisesRegex(ValueError, "outside"):
            audit(self.output, self.catalog, state)
        self.assertFalse(state.exists())


if __name__ == "__main__":
    unittest.main()
