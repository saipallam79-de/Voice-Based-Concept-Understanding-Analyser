"""Shared small helpers for safe storage and text metrics."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from nltk.tokenize import PunktSentenceTokenizer, wordpunct_tokenize


def save_uploaded_audio(payload: bytes, extension: str, directory: Path) -> Path:
    """Write validated bytes to a unique application-owned filename."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{uuid.uuid4().hex}.{extension.lower()}"
    destination.write_bytes(payload)
    return destination


def text_metrics(text: str) -> dict[str, int]:
    words = [token for token in wordpunct_tokenize(text or "") if re.search(r"\w", token)]
    sentences = PunktSentenceTokenizer().tokenize(text or "")
    return {"word_count": len(words), "sentence_count": len(sentences)}