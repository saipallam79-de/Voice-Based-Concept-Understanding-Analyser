"""Input validation helpers for uploaded audio."""

from __future__ import annotations

import io
import wave
from pathlib import Path

from utils.constants import SUPPORTED_AUDIO_FORMATS


class ValidationError(ValueError):
    """Raised when an uploaded audio payload is not safe to analyze."""


def validate_audio_extension(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_AUDIO_FORMATS:
        allowed = ", ".join(f".{item}" for item in SUPPORTED_AUDIO_FORMATS)
        raise ValidationError(f"Unsupported audio format. Use one of: {allowed}.")
    return suffix


def validate_audio_payload(
    payload: bytes,
    filename: str,
    max_size_mb: int = 25,
    max_duration_seconds: int = 600,
) -> float:
    """Validate extension, size, and decodability; return duration in seconds."""
    suffix = validate_audio_extension(filename)
    if not payload:
        raise ValidationError("The uploaded audio file is empty.")
    if len(payload) > max_size_mb * 1024 * 1024:
        raise ValidationError(f"Audio files must be smaller than {max_size_mb} MB.")

    try:
        if suffix == "wav":
            with wave.open(io.BytesIO(payload), "rb") as audio:
                frames = audio.getnframes()
                rate = audio.getframerate()
                if frames < 1 or rate < 1:
                    raise ValidationError("The audio file contains no playable samples.")
                duration = frames / rate
        else:
            import librosa

            duration = float(librosa.get_duration(path=io.BytesIO(payload)))
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(
            "The audio file could not be decoded. Check that it is a valid audio file."
        ) from exc

    if duration <= 0.01:
        raise ValidationError("The audio recording is too short to analyze.")
    if duration > max_duration_seconds:
        raise ValidationError(
            f"Audio must be {max_duration_seconds // 60} minutes or shorter."
        )
    return duration