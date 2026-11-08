"""Inspect raw Mahjong Soul responses using the project's protobuf schema."""

import hashlib
import importlib
import json
from pathlib import Path
import re
import sys


UUID = re.compile(
    r"[0-9]{6}-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.I
)
FINAL_RECORDS = {".lq.RecordHule", ".lq.RecordNoTile", ".lq.RecordLiuJu"}


class Rejected(ValueError):
    pass


def load_schema(directory):
    directory = Path(directory).resolve()
    sys.path.insert(0, str(directory))
    schema = importlib.import_module("mahjongsoul_pb2")
    if Path(schema.__file__).resolve().parent != directory:
        raise ValueError("An unexpected mahjongsoul_pb2 module was imported.")
    return schema


def is_record_name(name):
    return UUID.fullmatch(name) is not None


def inspect_record(
    path,
    schema,
    *,
    modes=(12, 16),
    max_bytes=32 * 1024 * 1024,
    source_stat=None,
):
    path = Path(path)
    # Catalog discovery has already resolved the parent and excluded links.
    # Reuse FindFirstFile/DirEntry metadata instead of opening each Windows
    # file several extra times just to canonicalize its path and stat it.
    if source_stat is None:
        resolved = path.resolve()
    elif not path.is_absolute():
        raise ValueError("A catalog entry must have an absolute path.")
    else:
        resolved = path
    result = {
        "path": str(resolved),
        "outcome": "rejected",
        "reason": None,
    }
    try:
        before = path.stat() if source_stat is None else source_stat
        result.update(bytes=before.st_size, mtime_ns=before.st_mtime_ns)
        if not 3 < before.st_size <= max_bytes:
            raise Rejected("invalid_size")
        with path.open("rb") as stream:
            contents = stream.read(max_bytes + 1)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"Source changed while being read: {path}")
        if len(contents) != before.st_size:
            raise RuntimeError(f"Incomplete source read: {path}")
        result["sha256"] = hashlib.sha256(contents).hexdigest()
        envelope = schema.Wrapper()
        envelope.ParseFromString(contents[3:])
        if envelope.name or not envelope.data:
            raise Rejected("invalid_response_wrapper")
        response = schema.ResGameRecord()
        response.ParseFromString(envelope.data)
        if response.error.code:
            raise Rejected("api_error")
        head = response.head
        result["uuid"] = head.uuid
        if (
            not UUID.fullmatch(head.uuid)
            or head.uuid.lower() != path.name.lower()
        ):
            raise Rejected("uuid_filename_mismatch")
        result["uuid"] = head.uuid.lower()
        result["mode_id"] = head.config.meta.mode_id
        if head.config.category != 2:
            raise Rejected("not_ranked")
        if head.config.meta.mode_id not in modes:
            raise Rejected("not_selected_room_or_style")
        if len(head.accounts) != 4 or sorted(
            account.seat for account in head.accounts
        ) != list(range(4)):
            raise Rejected("invalid_four_player_accounts")
        players = sorted(head.result.players, key=lambda player: player.seat)
        if len(players) != 4 or [player.seat for player in players] != list(
            range(4)
        ):
            raise Rejected("missing_final_results")
        if not head.start_time or head.end_time < head.start_time:
            raise Rejected("missing_or_invalid_game_times")
        if not response.data:
            raise Rejected("missing_record_payload")
        detail_wrapper = schema.Wrapper()
        detail_wrapper.ParseFromString(response.data)
        if detail_wrapper.name != ".lq.GameDetailRecords":
            raise Rejected("invalid_detail_wrapper")
        details = schema.GameDetailRecords()
        details.ParseFromString(detail_wrapper.data)
        if details.version == 0:
            if details.actions:
                raise Rejected("mixed_record_versions")
            records = details.records
        elif details.version == 210715:
            if details.records:
                raise Rejected("mixed_record_versions")
            records = [
                action.result for action in details.actions if action.result
            ]
        else:
            raise Rejected("unsupported_record_version")
        if not records:
            raise Rejected("empty_game_records")
        final = schema.Wrapper()
        final.ParseFromString(records[-1])
        if final.name not in FINAL_RECORDS:
            raise Rejected("missing_terminal_round_record")
        accounts = sorted(head.accounts, key=lambda account: account.seat)
        training_header = {
            "uuid": head.uuid.lower(),
            "category": head.config.category,
            "mode_id": head.config.meta.mode_id,
            "start_time": head.start_time,
            "end_time": head.end_time,
            "level_ids": [account.level.id for account in accounts],
            "scores": [player.part_point_1 for player in players],
            "total_points": [player.total_point for player in players],
            "grading_points": [player.grading_score for player in players],
        }
        if any(level == 0 for level in training_header["level_ids"]):
            raise Rejected("missing_player_level")
        semantic = hashlib.sha256(
            json.dumps(
                training_header, sort_keys=True, separators=(",", ":")
            ).encode()
        )
        semantic.update(response.data)
        result.update(training_header)
        result.update(
            outcome="candidate",
            semantic_sha256=semantic.hexdigest(),
            record_version=details.version,
            record_count=len(records),
            account_order=[account.seat for account in head.accounts],
        )
    except Rejected as error:
        result["reason"] = str(error)
    except ValueError as error:
        result["reason"] = type(error).__name__ + ": " + str(error)[:500]
    except Exception as error:
        # DecodeError is intentionally handled without importing protobuf at
        # module import time; no Torch/runtime dependency is needed here.
        if type(error).__name__ == "DecodeError":
            result["reason"] = "protobuf_decode_error"
        else:
            raise
    return result
