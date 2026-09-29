import pytest
import numpy as np

from modules.semantic_analysis import analyze_semantics, cosine_similarity, interpret_similarity


def test_cosine_similarity_for_aligned_vectors():
    assert cosine_similarity([1, 0], [2, 0]) == pytest.approx(1.0)


def test_cosine_similarity_handles_zero_and_shape_mismatch():
    assert cosine_similarity([0, 0], [1, 0]) == 0
    assert cosine_similarity([1], [1, 2]) == 0


def test_similarity_threshold_interpretation_is_configured():
    assert interpret_similarity(0.8) == "Strong Understanding"
    assert interpret_similarity(0.6) == "Moderate Understanding"
    assert interpret_similarity(0.2) == "Needs Improvement"


def test_semantic_analysis_classifies_key_concept_coverage(monkeypatch):
    class FakeModel:
        def encode(self, texts, **kwargs):
            return np.asarray([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=float)

    monkeypatch.setattr("models.semantic_model.load_semantic_model", lambda _: FakeModel())
    result = analyze_semantics("Reference", "Transcript", ["covered", "missing"])
    assert result["similarity"] == pytest.approx(1)
    assert result["covered_concepts"] == ["covered"]
    assert result["missing_concepts"] == ["missing"]