from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from evernest_assistant.config import CRM_DIR, DATA_DIR, DB_PATH

TABLES = (
    "branches",
    "agents",
    "listings",
    "listing_collaborators",
    "contacts",
    "activities",
    "offers",
)


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _load_csv(con: sqlite3.Connection, path: Path) -> None:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    table = path.stem
    con.execute(f'DROP TABLE IF EXISTS "{table}"')
    column_sql = ", ".join(f'"{name}" TEXT' for name in columns)
    con.execute(f'CREATE TABLE "{table}" ({column_sql})')
    placeholders = ", ".join("?" for _ in columns)
    con.executemany(
        f'INSERT INTO "{table}" VALUES ({placeholders})',
        [tuple(row[name] for name in columns) for row in rows],
    )


def load_crm(con: sqlite3.Connection) -> None:
    for name in TABLES:
        _load_csv(con, CRM_DIR / f"{name}.csv")
    con.commit()


def one(con: sqlite3.Connection, sql: str, params: tuple = ()) -> sqlite3.Row | None:
    return con.execute(sql, params).fetchone()


def all_rows(con: sqlite3.Connection, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    return con.execute(sql, params).fetchall()
