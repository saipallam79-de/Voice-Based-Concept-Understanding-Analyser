import pytest

from modules.scoring import calculate_scores, classify_performance


def test_weighted_overall_uses_default_weights():
    semantic = {"similarity": 0.8, "concept_scores": {"a": 0.8}}
    result = calculate_scores(
        semantic,
        {"filler_rate_per_minute": 0},
        {"pause_ratio": 0, "speech_activity_ratio": 0},
        {"word_count": 50, "sentence_count": 3},
    )
    expected = result["understanding_score"] * 0.60 + result["fluency_score"] * 0.25 + result["communication_score"] * 0.15
    assert result["overall_score"] == round(expected, 1)


@pytest.mark.parametrize(("score", "label"), [(100, "Excellent"), (90, "Excellent"), (89.9, "Strong"), (75, "Strong"), (60, "Moderate"), (40, "Needs Improvement"), (0, "Requires Significant Improvement")])
def test_performance_boundaries(score, label):
    assert classify_performance(score) == label


@pytest.mark.parametrize("score", [-1, 101, float("nan"), float("inf")])
def test_invalid_performance_scores_are_rejected(score):
    with pytest.raises(ValueError):
        classify_performance(score)


def test_empty_transcription_does_not_crash():
    result = calculate_scores(
        {"similarity": 0, "concept_scores": {}},
        {"filler_rate_per_minute": 0, "word_count": 0},
        {"pause_ratio": 1, "speech_activity_ratio": 0},
        {"word_count": 0, "sentence_count": 0},
    )
    assert result["understanding_score"] == 0
    assert result["fluency_score"] == 0
    assert result["communication_score"] == 0
    assert 0 <= result["overall_score"] <= 100