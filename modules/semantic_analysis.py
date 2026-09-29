"""Semantic similarity and per-key-point coverage analysis."""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from config.settings import settings


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if left.shape != right.shape or denominator == 0:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def interpret_similarity(similarity: float) -> str:
    if similarity >= settings.semantic_strong_threshold:
        return "Strong Understanding"
    if similarity >= settings.semantic_moderate_threshold:
        return "Moderate Understanding"
    return "Needs Improvement"


def split_reference_points(reference: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", reference) if sentence.strip()]


def analyze_semantics(
    reference: str,
    transcription: str,
    key_concepts: Sequence[str] | None = None,
    model_name: str | None = None,
) -> dict:
    """Run Sentence-BERT cosine comparison, including independent key-point checks."""
    from models.semantic_model import load_semantic_model

    if not reference.strip() or not transcription.strip():
        points = list(key_concepts or split_reference_points(reference))
        return {
            "similarity": 0.0,
            "similarity_percent": 0.0,
            "interpretation": "Needs Improvement",
            "covered_concepts": [],
            "weak_concepts": [],
            "missing_concepts": points,
            "concept_scores": {point: 0.0 for point in points},
        }

    model = load_semantic_model(model_name)
    points = list(key_concepts or split_reference_points(reference))
    texts = [reference, transcription, *points]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    similarity = cosine_similarity(embeddings[0], embeddings[1])
    concept_scores = {
        point: cosine_similarity(embeddings[index + 2], embeddings[1])
        for index, point in enumerate(points)
    }
    covered = [point for point, score in concept_scores.items() if score >= settings.coverage_threshold]
    weak = [
        point for point, score in concept_scores.items()
        if settings.weak_coverage_threshold <= score < settings.coverage_threshold
    ]
    missing = [point for point, score in concept_scores.items() if score < settings.weak_coverage_threshold]
    return {
        "similarity": similarity,
        "similarity_percent": max(0.0, similarity) * 100,
        "interpretation": interpret_similarity(similarity),
        "covered_concepts": covered,
        "weak_concepts": weak,
        "missing_concepts": missing,
        "concept_scores": concept_scores,
    }