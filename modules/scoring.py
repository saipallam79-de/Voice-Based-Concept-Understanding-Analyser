"""Centralized, configurable evaluation scores and performance bands."""

from __future__ import annotations

import math
from collections.abc import Mapping

from config.settings import settings

PERFORMANCE_BANDS = (
    (90, "Excellent"),
    (75, "Strong"),
    (60, "Moderate"),
    (40, "Needs Improvement"),
    (0, "Requires Significant Improvement"),
)


def _score(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError(f"{name} must be a finite number between 0 and 100.")
    return value


def classify_performance(score: float) -> str:
    score = _score(score, "score")
    return next(label for minimum, label in PERFORMANCE_BANDS if score >= minimum)


def calculate_scores(
    semantic: Mapping,
    fluency: Mapping,
    audio: Mapping,
    transcript: Mapping,
) -> dict:
    """Derive component scores and weighted overall score from measured inputs."""
    similarity = max(0.0, min(1.0, float(semantic.get("similarity", 0.0))))
    concept_scores = semantic.get("concept_scores", {})
    coverage = (
        sum(float(value) >= settings.coverage_threshold for value in concept_scores.values())
        / len(concept_scores)
        if concept_scores else similarity
    )
    understanding = _score((similarity * 0.65 + coverage * 0.35) * 100, "understanding score")

    filler_rate = max(0.0, float(fluency.get("filler_rate_per_minute", 0.0)))
    pause_ratio = max(0.0, min(1.0, float(audio.get("pause_ratio", 0.0))))
    activity = max(0.0, min(1.0, float(audio.get("speech_activity_ratio", 0.0))))
    transcript_words = max(0, int(fluency.get("word_count", transcript.get("word_count", 0))))
    fluency_score = (
        _score(min(100.0, max(0.0,
            100 - min(35, filler_rate * 3.0) - min(45, pause_ratio * 75)
            + min(12, activity * 12))), "fluency score")
        if transcript_words else 0.0
    )

    words = max(0, int(transcript.get("word_count", fluency.get("word_count", 0))))
    sentences = max(0, int(transcript.get("sentence_count", 0)))
    sentence_structure = min(1.0, sentences / max(1.0, words / 18.0)) if words else 0.0
    word_volume = min(1.0, words / 45.0)
    communication = (
        _score(min(100.0, max(0.0,
            100 - min(25, filler_rate * 2.0) - min(30, pause_ratio * 50)
            + activity * 15 + sentence_structure * 10 + word_volume * 10)), "communication score")
        if words else 0.0
    )

    weights = (
        settings.understanding_weight,
        settings.fluency_weight,
        settings.communication_weight,
    )
    if any(weight < 0 for weight in weights) or not math.isclose(sum(weights), 1.0, abs_tol=0.001):
        raise ValueError("Configured score weights must be non-negative and sum to 1.0.")
    overall = _score(
        understanding * weights[0] + fluency_score * weights[1] + communication * weights[2],
        "overall score",
    )
    return {
        "understanding_score": round(understanding, 1),
        "fluency_score": round(fluency_score, 1),
        "communication_score": round(communication, 1),
        "overall_score": round(overall, 1),
        "performance_level": classify_performance(overall),
        "scoring_explanation": (
            "Understanding combines semantic similarity (65%) and key-point coverage (35%). "
            "Fluency and communication use filler rate, pauses, speech activity, transcript length, "
            "and sentence structure. These are delivery indicators, not measures of confidence."
        ),
    }