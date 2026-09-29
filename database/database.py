"""SQLite persistence for evaluation metadata and result summaries."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluations (
    session_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    concept TEXT NOT NULL,
    transcription TEXT NOT NULL,
    similarity_score REAL NOT NULL,
    understanding_score REAL NOT NULL,
    fluency_score REAL NOT NULL,
    communication_score REAL NOT NULL,
    overall_score REAL NOT NULL,
    audio_duration REAL NOT NULL,
    filler_count INTEGER NOT NULL,
    pause_ratio REAL NOT NULL,
    result_json TEXT NOT NULL
)
"""


def _connect(database_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(database_path or settings.database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute(SCHEMA)
    return connection


def initialize_database(database_path: Path | str | None = None) -> None:
    try:
        with _connect(database_path) as connection:
            connection.execute(SCHEMA)
    except sqlite3.Error:
        logger.exception("Could not initialize evaluation database")
        raise


def save_evaluation(result: dict, database_path: Path | str | None = None) -> str:
    import uuid

    session_id = str(result.get("session_id") or uuid.uuid4())
    timestamp = result.get("timestamp") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    scores = result["scores"]
    semantic = result["semantic"]
    audio = result["audio"]
    fluency = result["fluency"]
    values = (
        session_id, timestamp, result["concept"], result["transcription"],
        semantic["similarity"], scores["understanding_score"], scores["fluency_score"],
        scores["communication_score"], scores["overall_score"], audio["duration_seconds"],
        fluency["filler_count"], audio["pause_ratio"],
        json.dumps(result, ensure_ascii=False),
    )
    try:
        with _connect(database_path) as connection:
            connection.execute(
                "INSERT INTO evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values
            )
    except (sqlite3.Error, OSError):
        logger.exception("Could not save evaluation session")
        raise
    return session_id


def list_evaluations(database_path: Path | str | None = None) -> list[dict]:
    try:
        with _connect(database_path) as connection:
            rows = connection.execute(
                "SELECT * FROM evaluations ORDER BY timestamp DESC"
            ).fetchall()
        return [dict(row) for row in rows]
    except (sqlite3.Error, OSError):
        logger.exception("Could not load evaluation history")
        raise


def get_evaluation(session_id: str, database_path: Path | str | None = None) -> dict | None:
    try:
        with _connect(database_path) as connection:
            row = connection.execute(
                "SELECT result_json FROM evaluations WHERE session_id = ?", (session_id,)
            ).fetchone()
        return json.loads(row["result_json"]) if row else None
    except (sqlite3.Error, OSError, json.JSONDecodeError):
        logger.exception("Could not load evaluation result")
        raise


def delete_evaluation(session_id: str, database_path: Path | str | None = None) -> bool:
    try:
        with _connect(database_path) as connection:
            cursor = connection.execute("DELETE FROM evaluations WHERE session_id = ?", (session_id,))
        return cursor.rowcount > 0
    except (sqlite3.Error, OSError):
        logger.exception("Could not delete evaluation session")
        raise


def delete_all_evaluations(database_path: Path | str | None = None) -> int:
    try:
        with _connect(database_path) as connection:
            cursor = connection.execute("DELETE FROM evaluations")
        return cursor.rowcount
    except (sqlite3.Error, OSError):
        logger.exception("Could not clear evaluation history")
        raise