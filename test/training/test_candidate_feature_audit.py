"""Reject empty or underfilled audit inputs and exercise the real CLI gate."""

import bz2
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.audit_candidate_features import audit


class AuditCoverageTests(unittest.TestCase):
    def test_empty_plain_gzip_and_bzip2_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix, opener in (
                (".tsv", open),
                (".gz", gzip.open),
                (".bz2", bz2.open),
            ):
                with self.subTest(suffix=suffix):
                    path = Path(directory) / ("empty" + suffix)
                    with opener(path, "wt", encoding="utf-8"):
                        pass
                    with self.assertRaisesRegex(
                        ValueError, "checked 0 rows; require at least 1"
                    ):
                        audit(path, 4, 2)

    def test_minimum_coverage_must_be_positive_and_fit_limit(self):
        for minimum in (0, -1, 5):
            with (
                self.subTest(min_rows=minimum),
                self.assertRaisesRegex(ValueError, "Minimum audit rows"),
            ):
                audit(Path("does-not-exist"), 4, 2, min_rows=minimum)

    def test_empty_cli_exits_nonzero_without_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, output = root / "empty.tsv", root / "audit.json"
            path.touch()
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(
                        Path(__file__).resolve().parents[2]
                        / "tools/audit_candidate_features.py"
                    ),
                    "--input",
                    str(path),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Insufficient audit coverage", result.stderr)
            self.assertNotIn('"status": "passed"', result.stdout)
            self.assertFalse(output.exists())


@unittest.skipUnless(
    importlib.util.find_spec("kanachan._candidate_features"),
    "Build native candidate features for populated audit checks",
)
class NativeAuditCoverageTests(unittest.TestCase):
    def setUp(self):
        from test_candidate_features import training_line

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "three.tsv.gz"
        with gzip.open(self.path, "wt", encoding="utf-8") as stream:
            stream.writelines(training_line(index) for index in range(3))

    def test_exact_minimum_and_partial_final_batch_preserve_input(self):
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        result = audit(self.path, 4, 2, min_rows=3)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(
            (result["rows"], result["min_rows"], result["row_limit"]),
            (3, 3, 4),
        )
        self.assertEqual(result["discard_candidates"], 6)
        self.assertEqual(
            before, hashlib.sha256(self.path.read_bytes()).hexdigest()
        )

    def test_nonempty_but_insufficient_coverage_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "checked 3 rows; require at least 4"
        ):
            audit(self.path, 4, 2, min_rows=4)

    def test_cli_release_gate_records_required_coverage(self):
        output = self.root / "audit.json"
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(
                    Path(__file__).resolve().parents[2]
                    / "tools/audit_candidate_features.py"
                ),
                "--input",
                str(self.path),
                "--limit",
                "2",
                "--min-rows",
                "2",
                "--batch-size",
                "1",
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(
            (report["rows"], report["min_rows"], report["row_limit"]),
            (2, 2, 2),
        )


if __name__ == "__main__":
    unittest.main()
