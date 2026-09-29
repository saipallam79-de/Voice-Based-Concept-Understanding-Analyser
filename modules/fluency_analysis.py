"""Transcript-level fluency indicators; score calculation lives in scoring.py."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


def analyze_fluency(transcription: str, duration_seconds: float, filler_words: Iterable[str]) -> dict:
    tokens = re.findall(r"\b[\w']+\b", (transcription or "").lower())
    labels_by_normalized = {str(word).strip().lower(): str(word).strip() for word in filler_words}
    normalized_fillers = sorted(labels_by_normalized, key=len, reverse=True)
    counts: Counter[str] = Counter()
    for filler in normalized_fillers:
        phrase = re.findall(r"\b[\w']+\b", filler)
        if not phrase:
            continue
        pattern = r"\b" + r"\s+".join(re.escape(word) for word in phrase) + r"\b"
        counts[labels_by_normalized[filler]] = len(re.findall(pattern, (transcription or "").lower()))
    total = sum(counts.values())
    minutes = max(float(duration_seconds) / 60.0, 1 / 60)
    return {
        "filler_counts": {word: count for word, count in counts.items() if count},
        "filler_count": total,
        "filler_rate_per_minute": total / minutes,
        "word_count": len(tokens),
        "words_per_minute": len(tokens) / minutes if duration_seconds > 0 else 0.0,
    }