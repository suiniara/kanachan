import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from test_training_regressions import bc_line

from kanachan.constants import SEAT_INDEX, SEAT_OFFSET
from kanachan.training.audit_data import audit


class AnnotationConversionTests(unittest.TestCase):
    def setUp(self):
        source = (
            Path(__file__).resolve().parents[2] / "bin/annotation/rl/rl.py"
        )
        spec = importlib.util.spec_from_file_location("annotation_rl", source)
        self.converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.converter)

    def test_streaming_conversion_matches_full_conversion_and_preserves_terminals(
        self,
    ):
        rows = [bc_line(0), bc_line(1), bc_line(0).replace("game-0", "game-1")]
        full, streamed = io.StringIO(), io.StringIO()
        with redirect_stdout(full):
            self.converter._parse(rows, None, None)
        with redirect_stdout(streamed):
            self.converter._parse_stream(iter(rows), None, None)
        self.assertEqual(full.getvalue(), streamed.getvalue())
        self.assertEqual(
            [len(line.split("\t")) for line in full.getvalue().splitlines()],
            [10, 8, 8],
        )

    def test_streaming_rejects_a_game_that_reappears(self):
        rows = [bc_line(), bc_line().replace("game-0", "game-1"), bc_line()]
        with (
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "reappeared"),
        ):
            self.converter._parse_stream(rows, None, None)

    def test_audit_flags_bc_rows_used_as_terminal_only_rl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.tsv"
            path.write_text(bc_line() * 3)
            result = audit(path, "rl")
            self.assertEqual(result["end_of_game_fraction"], 1.0)
            self.assertTrue(result["warnings"])
            self.assertFalse(audit(path, "bc", 2)["warnings"])
            converted = io.StringIO()
            with redirect_stdout(converted):
                self.converter._parse(
                    path.read_text().splitlines(), None, None
                )
            path.write_text(converted.getvalue())
            self.assertFalse(audit(path, "rl")["warnings"])

    def test_public_seat_constants_match_each_current_annotation_seat(self):
        fields = [int(value) for value in bc_line().split("\t")[1].split(",")]
        for seat in range(4):
            fields[6] = 71 + seat
            self.assertEqual(fields[SEAT_INDEX] - SEAT_OFFSET, seat)

    def test_legacy_prefix_is_rejected_even_if_column_count_was_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.tsv"
            columns = bc_line().split("\t")
            fields = columns[1].split(",")
            fields[3] = "11"
            columns[1] = ",".join(fields)
            path.write_text("\t".join(columns))
            with self.assertRaisesRegex(
                RuntimeError, "Invalid v2 sparse prefix"
            ):
                audit(path, "bc")
