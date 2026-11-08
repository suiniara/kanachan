"""Import an explicitly compatible, paused catalog into a fresh directory."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from .catalog import Catalog, atomic_json, lower_priority


LEGACY_IMPLEMENTATION = (
    "dfcdb9e335d64952ac11e6d21d3522a011accb8cf1d839f82abb18d1846589c2"
)
TABLES = ("directories", "files", "games")


def table_digest(connection, table):
    checksum = hashlib.sha256()
    count = 0
    for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid"):
        checksum.update(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode()
        )
        checksum.update(b"\n")
        count += 1
    return {"rows": count, "sha256": checksum.hexdigest()}


def migrate(source, output, schema_dir):
    source, output, schema_dir = (
        Path(value).resolve() for value in (source, output, schema_dir)
    )
    if (
        source == output
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise ValueError("Use a separate, fresh output directory.")
    if (output / "catalog.sqlite3").exists():
        raise ValueError("A destination catalog already exists.")
    progress = json.loads(
        (source / "progress.json").read_text(encoding="utf-8")
    )
    if progress["status"] not in ("paused", "limited", "complete"):
        raise ValueError(
            "Checkpoint the source catalog before importing its cache."
        )
    donor = sqlite3.connect(
        (source / "catalog.sqlite3").as_uri() + "?mode=ro", uri=True
    )
    recipient = None
    try:
        configuration = json.loads(
            donor.execute(
                "SELECT value FROM meta WHERE key='configuration'"
            ).fetchone()[0]
        )
        if (
            configuration["implementation_sha256"] != LEGACY_IMPLEMENTATION
            or configuration["version"] != 1
        ):
            raise ValueError(
                "This source implementation has not been verified as cache-compatible."
            )
        if (
            hashlib.sha256(
                (schema_dir / "mahjongsoul_pb2.py").read_bytes()
            ).hexdigest()
            != configuration["schema_sha256"]
        ):
            raise ValueError(
                "The source catalog used a different protobuf schema."
            )
        implementation = hashlib.sha256()
        for name in ("catalog.py", "records.py"):
            implementation.update((Path(__file__).parent / name).read_bytes())
        old_configuration = dict(configuration)
        configuration["implementation_sha256"] = implementation.hexdigest()
        recipient = Catalog(output, configuration)
        donor.execute("BEGIN")
        tables = {}
        for table in TABLES:
            old_columns = donor.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
            new_columns = recipient.db.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
            if old_columns != new_columns:
                raise ValueError(f"Cache schema differs: {table}")
            placeholders = ",".join("?" for _ in old_columns)
            recipient.db.executemany(
                f"INSERT INTO {table} VALUES({placeholders})",
                donor.execute(f"SELECT * FROM {table} ORDER BY rowid"),
            )
            old_digest = table_digest(donor, table)
            new_digest = table_digest(recipient.db, table)
            if old_digest != new_digest:
                raise RuntimeError(f"Imported cache differs: {table}")
            tables[table] = old_digest
        if recipient.db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise RuntimeError(
                "Imported catalog failed SQLite integrity checks."
            )
        receipt = {
            "source": str(source),
            "output": str(output),
            "source_configuration": old_configuration,
            "configuration": configuration,
            "tables": tables,
            "importer_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }
        recipient.db.execute(
            "INSERT INTO meta VALUES('cache_import',?)",
            (json.dumps(receipt, sort_keys=True),),
        )
        recipient.db.commit()
        atomic_json(output / "cache-import.json", receipt)
        recipient.counts = dict(
            recipient.db.execute(
                "SELECT outcome,COUNT(*) FROM files GROUP BY outcome"
            )
        )
        recipient.unique = recipient.db.execute(
            "SELECT COUNT(*) FROM games"
        ).fetchone()[0]
        recipient.conflicts = recipient.db.execute(
            "SELECT COUNT(*) FROM games WHERE conflicted=1"
        ).fetchone()[0]
        recipient.progress(status="ready")
        return receipt
    finally:
        if recipient is not None:
            recipient.db.close()
        donor.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema-dir", type=Path, required=True)
    arguments = parser.parse_args()
    lower_priority()
    receipt = migrate(arguments.source, arguments.output, arguments.schema_dir)
    print(
        json.dumps({"status": "imported", "tables": receipt["tables"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
