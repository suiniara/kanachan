import os
import subprocess
import sys
import unittest
from pathlib import Path


@unittest.skipUnless(
    os.environ.get("KANACHAN_NATIVE_FIXTURES"),
    "Set KANACHAN_NATIVE_FIXTURES to complete generated game fixtures",
)
class NativeReplayTests(unittest.TestCase):
    def test_annotation_and_simulator_match_complete_games(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(root / "test/annotation_vs_simulation/run.py"),
                os.environ["KANACHAN_NATIVE_FIXTURES"],
            ],
            cwd=root,
            env=dict(
                os.environ, PYTHONPATH=str(root), TORCHDYNAMO_DISABLE="1"
            ),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS", result.stdout)
