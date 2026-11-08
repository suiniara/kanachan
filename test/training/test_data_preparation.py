"""Small data-integrity checks and optional real-game annotation checks."""

import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from kanachan.data.prepare import assemble, digest, verify_copy
from kanachan.data.records import load_schema
from kanachan.data.validation import validate_bc, validate_rl, validate_eor


class DataPreparationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metadata = {
            "uuid": "200401-00000000-0000-0000-0000-000000000001",
            "mode_id": 12,
            "level_ids": [10401] * 4,
            "scores": [25000] * 4,
        }

    def columns(self, progression="0"):
        return [
            self.metadata["uuid"],
            "3,6,16,32,48,64,71,75,78,151,152",
            "0,0,25000,25000,25000,25000",
            progression,
            "0,4",
            "1",
            "0",
            "0,0,0,0,0,0,25000,25000,25000,25000,25000,25000,25000,25000",
        ]

    def test_valid_short_game_can_have_only_one_acting_seat(self):
        line = "\t".join(self.columns()) + "\n"
        self.assertEqual(validate_bc([line], self.metadata), {0})
        self.assertEqual(
            validate_rl([line], self.metadata, 1, {0}),
            {"rows": 1, "terminal": 1},
        )

    def test_wrong_grades_final_scores_and_action_are_rejected(self):
        for position, value in (
            (1, "3,6,17,32,48,64,71,75,78,151,152"),
            (5, "2"),
            (7, "0,0,0,0,0,0,25000,25000,25000,25000,24000,25000,25000,26000"),
        ):
            with self.subTest(position=position):
                columns = self.columns()
                columns[position] = value
                with self.assertRaises(ValueError):
                    validate_bc(["\t".join(columns)], self.metadata)

    def test_rl_links_match_actual_next_decision(self):
        first, second = self.columns("0"), self.columns("0,1")
        lines = ["\t".join(first[:6] + second[1:5]), "\t".join(second)]
        self.assertEqual(
            validate_rl(lines, self.metadata, 2, {0}),
            {"rows": 2, "terminal": 1},
        )
        corrupt = self.columns("0,2")
        with self.assertRaises(ValueError):
            validate_rl([lines[0], "\t".join(corrupt)], self.metadata, 2, {0})
        with self.assertRaises(ValueError):
            validate_rl(
                ["\t".join(first), "\t".join(second)], self.metadata, 2, {0}
            )

    def test_eor_shape_and_raw_scores(self):
        line = (
            self.metadata["uuid"]
            + "\t3,6,16,32,48,64,71,74\t0,0,25000,25000,25000,25000\t25000,25000,25000,25000"
        )
        self.assertEqual(validate_eor([line], self.metadata), 1)
        with self.assertRaises((ValueError, RuntimeError)):
            validate_eor([line.replace("71,74", "75,74")], self.metadata)

    def test_source_copy_requires_the_catalog_hash(self):
        path = self.root / "raw"
        path.write_bytes(b"original game bytes")
        stat = path.stat()
        record = {
            "path": str(path),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest(path),
        }
        verify_copy(record, self.root / "verified")
        self.assertEqual(
            (self.root / "verified").read_bytes(), path.read_bytes()
        )
        record["sha256"] = "0" * 64
        with self.assertRaises(RuntimeError):
            verify_copy(record, self.root / "unverified")
        self.assertEqual(path.read_bytes(), b"original game bytes")

    def test_gzip_shards_assemble_without_splitting_games(self):
        receipts = []
        for index in range(2):
            directory = self.root / "shards" / "train" / f"{index:06d}"
            directory.mkdir(parents=True)
            receipt = {"split": "train", "shard": index, "files": {}}
            for kind in ("bc", "rl", "eor"):
                path = directory / f"{kind}.tsv.gz"
                with gzip.open(path, "wt", encoding="utf-8") as stream:
                    stream.write(f"game-{index}\n")
                receipt["files"][kind] = {
                    "name": path.name,
                    "sha256": digest(path),
                    "bytes": path.stat().st_size,
                    "rows": 1,
                }
            receipts.append(receipt)
        files = assemble(self.root, receipts, 0)
        with gzip.open(files["train"]["bc"]["path"], "rt") as reader:
            self.assertEqual(reader.read(), "game-0\ngame-1\n")
        self.assertEqual(files["train"]["bc"]["rows"], 2)
        (self.root / "shards" / "train" / "000000" / "bc.tsv.gz").write_bytes(
            b"corrupted"
        )
        with self.assertRaises(RuntimeError):
            assemble(self.root, receipts, 0)


@unittest.skipUnless(
    os.environ.get("KANACHAN_ANNOTATOR")
    and os.environ.get("KANACHAN_SAMPLE_MANIFEST")
    and os.environ.get("KANACHAN_SCHEMA_DIR"),
    "real annotation fixture not configured",
)
class NativeAnnotationIntegrityTest(unittest.TestCase):
    def test_reordered_accounts_and_late_failure_are_atomic(self):
        schema = load_schema(os.environ["KANACHAN_SCHEMA_DIR"])
        with Path(os.environ["KANACHAN_SAMPLE_MANIFEST"]).open(
            encoding="utf-8"
        ) as stream:
            metadata = json.loads(next(stream))
        source = Path(metadata["path"])
        before = digest(source)
        content = source.read_bytes()
        envelope = schema.Wrapper.FromString(content[3:])
        response = schema.ResGameRecord.FromString(envelope.data)

        def encode():
            envelope.data = response.SerializeToString()
            return content[:3] + envelope.SerializeToString()

        with tempfile.TemporaryDirectory(
            prefix="kanachan-native-integrity-"
        ) as temporary:
            work = Path(temporary)
            original = work / "original" / metadata["uuid"]
            original.parent.mkdir()
            original.write_bytes(content)
            accounts = [
                account.SerializeToString()
                for account in response.head.accounts
            ]
            response.head.ClearField("accounts")
            for account in reversed(accounts):
                response.head.accounts.add().ParseFromString(account)
            reordered = work / "reordered" / metadata["uuid"]
            reordered.parent.mkdir()
            reordered.write_bytes(encode())

            def annotate(name, paths, continue_on_error=False):
                filelist = work / (name + ".txt")
                status = work / (name + ".status")
                filelist.write_text(
                    "".join(str(path) + "\n" for path in paths),
                    encoding="utf-8",
                )
                command = [
                    os.environ["KANACHAN_ANNOTATOR"],
                    "--files-from",
                    str(filelist),
                    "--status-file",
                    str(status),
                ]
                if continue_on_error:
                    command.append("--continue-on-error")
                result = subprocess.run(
                    command, capture_output=True, check=True, timeout=60
                )
                return result.stdout, status.read_text(encoding="utf-8")

            expected, _ = annotate("expected", [original])
            actual, _ = annotate("reordered", [reordered])
            self.assertEqual(actual, expected)
            detail_wrapper = schema.Wrapper.FromString(response.data)
            detail = schema.GameDetailRecords.FromString(detail_wrapper.data)
            malformed = schema.Wrapper(
                name=".lq.RecordHule", data=b"\xff"
            ).SerializeToString()
            if detail.version == 0:
                detail.records[-1] = malformed
            else:
                next(
                    action
                    for action in reversed(detail.actions)
                    if action.result
                ).result = malformed
            detail_wrapper.data = detail.SerializeToString()
            response.data = detail_wrapper.SerializeToString()
            broken = work / "broken" / metadata["uuid"]
            broken.parent.mkdir()
            broken.write_bytes(encode())
            actual, statuses = annotate("continue", [broken, original], True)
            self.assertEqual(actual, expected)
            self.assertEqual(
                [line.split("\t")[0] for line in statuses.splitlines()],
                ["error", "ok"],
            )
        self.assertEqual(digest(source), before)


if __name__ == "__main__":
    unittest.main()
