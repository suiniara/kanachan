"""CPU-only checks; set KANACHAN_SCHEMA_DIR to generated protobuf sources."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from kanachan.data.catalog import Catalog, directories, run, split_game
from kanachan.data.records import inspect_record, load_schema
from kanachan.data.migrate_cache import LEGACY_IMPLEMENTATION, migrate


@unittest.skipUnless(
    os.environ.get("KANACHAN_SCHEMA_DIR"), "protobuf schema not configured"
)
class RecordCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = load_schema(os.environ["KANACHAN_SCHEMA_DIR"])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.uuid = "200401-00000000-0000-0000-0000-000000000001"

    def record(
        self,
        *,
        uuid=None,
        mode=12,
        seats=(0, 1, 2, 3),
        prefix=b"\x02\x01\x00",
        nickname="original",
        score=25000,
        terminal=".lq.RecordHule",
        version=0,
    ):
        schema = self.schema
        response = schema.ResGameRecord()
        head = response.head
        head.uuid = uuid or self.uuid
        head.start_time = 1585735200
        head.end_time = 1585738800
        head.config.category = 2
        head.config.meta.mode_id = mode
        for seat in seats:
            account = head.accounts.add(seat=seat, nickname=nickname)
            account.level.id = 10401 + seat
        for seat in range(4):
            head.result.players.add(
                seat=seat,
                part_point_1=score,
                total_point=40000 - seat * 10000,
                grading_score=seat,
            )
        details = schema.GameDetailRecords(version=version)
        final = schema.Wrapper(name=terminal).SerializeToString()
        if version == 0:
            details.records.append(final)
        else:
            details.actions.add(type=1, result=final)
        response.data = schema.Wrapper(
            name=".lq.GameDetailRecords", data=details.SerializeToString()
        ).SerializeToString()
        return (
            prefix
            + schema.Wrapper(
                data=response.SerializeToString()
            ).SerializeToString()
        )

    def write(self, directory=None, **kwargs):
        directory = directory or self.raw
        directory.mkdir(exist_ok=True)
        path = directory / kwargs.get("uuid", self.uuid)
        path.write_bytes(self.record(**kwargs))
        return path

    def inspect(self, path):
        return inspect_record(path, self.schema)

    def test_dedup_ignores_transport_id_and_nickname(self):
        first = self.inspect(self.write())
        second = self.inspect(
            self.write(
                self.root / "copy", prefix=b"\x02\x02\x00", nickname="renamed"
            )
        )
        self.assertEqual(first["outcome"], "candidate")
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["semantic_sha256"], second["semantic_sha256"])

    def test_seats_are_explicit_not_repeated_field_order(self):
        result = self.inspect(self.write(seats=(3, 0, 2, 1)))
        self.assertEqual(result["account_order"], [3, 0, 2, 1])
        self.assertEqual(result["level_ids"], [10401, 10402, 10403, 10404])
        self.assertEqual(
            result["semantic_sha256"],
            self.inspect(self.write())["semantic_sha256"],
        )
        self.assertEqual(
            self.inspect(self.write(seats=(0, 0, 2, 3)))["reason"],
            "invalid_four_player_accounts",
        )

    def test_reject_wrong_room_incomplete_and_wrong_uuid(self):
        self.assertEqual(
            self.inspect(self.write(mode=11))["reason"],
            "not_selected_room_or_style",
        )
        self.assertEqual(
            self.inspect(self.write(terminal=".lq.RecordDiscardTile"))[
                "reason"
            ],
            "missing_terminal_round_record",
        )
        path = self.write()
        path.write_bytes(
            self.record(uuid="200401-00000000-0000-0000-0000-000000000002")
        )
        self.assertEqual(
            self.inspect(path)["reason"], "uuid_filename_mismatch"
        )
        path.write_bytes(b"\x02\x01\x00\xff")
        self.assertEqual(self.inspect(path)["reason"], "protobuf_decode_error")

    def test_both_record_versions(self):
        for version in (0, 210715):
            with self.subTest(version=version):
                result = self.inspect(self.write(version=version))
                self.assertEqual(result["record_version"], version)
                self.assertEqual(result["record_count"], 1)
                self.assertEqual(result["outcome"], "candidate")

    def test_io_failure_stops_instead_of_rejecting_data(self):
        with self.assertRaises(OSError):
            self.inspect(self.raw / self.uuid)

    def test_directory_metadata_fast_path_matches_full_inspection(self):
        path = self.write()
        expected = self.inspect(path)
        with os.scandir(self.raw) as entries:
            entry = next(entries)
            source_stat = entry.stat(follow_symlinks=False)
        actual = inspect_record(path, self.schema, source_stat=source_stat)
        self.assertEqual(actual, expected)
        path.write_bytes(b"changed")
        with self.assertRaises(RuntimeError):
            inspect_record(path, self.schema, source_stat=source_stat)

    def test_conflicting_copy_excludes_the_entire_uuid(self):
        catalog = Catalog(
            self.root / "catalog",
            {"roots": [str(self.raw)], "split_salt": "test"},
        )
        self.addCleanup(catalog.db.close)
        catalog.add(self.inspect(self.write()))
        catalog.add(self.inspect(self.write(self.root / "copy")))
        catalog.add(
            self.inspect(self.write(self.root / "conflict", score=25001))
        )
        self.assertEqual(
            catalog.counts, {"selected": 1, "duplicate": 1, "conflict": 1}
        )
        self.assertEqual(
            catalog.export(), {"train": 0, "validation": 0, "test": 0}
        )

    def test_cache_import_preserves_every_metadata_row(self):
        source = self.root / "old-catalog"
        configuration = {
            "version": 1,
            "roots": [str(self.raw)],
            "split_salt": "test",
            "implementation_sha256": LEGACY_IMPLEMENTATION,
            "schema_sha256": hashlib.sha256(
                (
                    Path(os.environ["KANACHAN_SCHEMA_DIR"])
                    / "mahjongsoul_pb2.py"
                ).read_bytes()
            ).hexdigest(),
        }
        catalog = Catalog(source, configuration)
        catalog.add(self.inspect(self.write()))
        catalog.add(self.inspect(self.write(self.root / "copy")))
        catalog.progress(status="paused")
        catalog.db.close()
        output = self.root / "new-catalog"
        receipt = migrate(source, output, os.environ["KANACHAN_SCHEMA_DIR"])
        self.assertEqual(receipt["tables"]["files"]["rows"], 2)
        self.assertEqual(receipt["tables"]["games"]["rows"], 1)
        imported = Catalog(output, receipt["configuration"])
        self.addCleanup(imported.db.close)
        self.assertEqual(imported.counts, {"selected": 1, "duplicate": 1})
        self.assertNotEqual(
            receipt["configuration"]["implementation_sha256"],
            LEGACY_IMPLEMENTATION,
        )
        with self.assertRaises(ValueError):
            migrate(source, output, os.environ["KANACHAN_SCHEMA_DIR"])

    def test_resume_and_whole_game_manifest(self):
        first = self.write()
        second = self.write(
            uuid="200401-00000000-0000-0000-0000-000000000002", mode=16
        )
        source_hashes = [
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (first, second)
        ]
        output = self.root / "catalog"
        arguments = argparse.Namespace(
            workers=2,
            batch_size=1,
            limit=1,
            max_record_bytes=1024 * 1024,
            schema_dir=Path(os.environ["KANACHAN_SCHEMA_DIR"]),
            raw_root=[self.raw],
            output=output,
            split_salt="test",
        )
        run(arguments)
        self.assertEqual(
            json.loads((output / "progress.json").read_text())["status"],
            "limited",
        )
        arguments.limit = 0
        run(arguments)
        self.assertEqual(
            json.loads((output / "progress.json").read_text())["status"],
            "complete",
        )
        manifest = [
            json.loads(line)
            for path in output.glob("manifest.*.jsonl")
            for line in path.read_text().splitlines()
        ]
        self.assertEqual(len(manifest), 2)
        self.assertEqual(len({entry["uuid"] for entry in manifest}), 2)
        self.assertTrue(
            all(
                entry["split"] == split_game(entry["uuid"], "test")
                for entry in manifest
            )
        )
        run(arguments)
        self.assertEqual(
            json.loads((output / "summary.json").read_text())["file_outcomes"],
            {"selected": 2},
        )
        self.assertEqual(
            source_hashes,
            [
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (first, second)
            ],
        )

    def test_modes_and_source_output_boundary(self):
        for name in ("2020-04-01-11", "2020-04-01-12", "2020-04-01-16"):
            (self.raw / name).mkdir()
        self.assertEqual(
            [path.name for path in directories([self.raw])],
            ["raw", "2020-04-01-11", "2020-04-01-12", "2020-04-01-16"],
        )
        with self.assertRaises(ValueError):
            Catalog(self.raw / "output", {"roots": [str(self.raw)]})

    def test_split_is_stable_and_covers_all_partitions(self):
        self.assertEqual(split_game(self.uuid), split_game(self.uuid.upper()))
        splits = {
            split_game(f"200401-{index:08x}-0000-0000-0000-000000000001")
            for index in range(1000)
        }
        self.assertEqual(splits, {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
