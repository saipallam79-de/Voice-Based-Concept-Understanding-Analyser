"""Central application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    whisper_model: str = os.getenv(
        "WHISPER_MODEL", os.getenv("VBCUA_WHISPER_MODEL", "base")
    )
    semantic_model: str = os.getenv(
        "VBCUA_SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    max_upload_mb: int = _int_env("VBCUA_MAX_UPLOAD_MB", 25)
    max_audio_seconds: int = _int_env("VBCUA_MAX_AUDIO_SECONDS", 600)
    database_path: Path = Path(os.getenv("VBCUA_DATABASE_PATH", ROOT_DIR / "data" / "vbcua.sqlite3"))
    audio_dir: Path = ROOT_DIR / "data" / "audio"
    understanding_weight: float = _float_env("VBCUA_UNDERSTANDING_WEIGHT", 0.60)
    fluency_weight: float = _float_env("VBCUA_FLUENCY_WEIGHT", 0.25)
    communication_weight: float = _float_env("VBCUA_COMMUNICATION_WEIGHT", 0.15)
    semantic_strong_threshold: float = _float_env("VBCUA_SEMANTIC_STRONG", 0.70)
    semantic_moderate_threshold: float = _float_env("VBCUA_SEMANTIC_MODERATE", 0.50)
    coverage_threshold: float = _float_env("VBCUA_COVERAGE_THRESHOLD", 0.48)
    weak_coverage_threshold: float = _float_env("VBCUA_WEAK_COVERAGE_THRESHOLD", 0.34)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")


settings = Settings()