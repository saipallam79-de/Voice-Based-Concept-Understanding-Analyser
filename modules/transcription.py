"""Speech transcription and transcript statistics."""

from __future__ import annotations

import logging

from config.settings import settings
from models.whisper_model import (
    WhisperModelError,
    _format_missing_requirements,
    check_whisper_dependencies,
    load_whisper_model,
)
from utils.helpers import text_metrics

logger = logging.getLogger(__name__)


def transcribe_audio(audio_path: str, model_name: str | None = None) -> dict:
    """Transcribe audio with Whisper and return text plus basic counts."""
    diagnostics = check_whisper_dependencies(model_name or settings.whisper_model)
    if not diagnostics["ready"]:
        raise WhisperModelError(
            "Whisper prerequisites are not ready. "
            + _format_missing_requirements(diagnostics)
        )
    try:
        model = load_whisper_model(model_name or settings.whisper_model)
        result = model.transcribe(audio_path, fp16=False)
        text = str(result.get("text", "")).strip()
        return {"text": text, **text_metrics(text), "language": result.get("language")}
    except WhisperModelError:
        raise
    except Exception as exc:
        logger.exception("Whisper transcription failed")
        raise RuntimeError(
            f"OpenAI Whisper transcription failed: {type(exc).__name__}: {exc}"
        ) from exc