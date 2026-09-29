from modules.fluency_analysis import analyze_fluency


def test_counts_single_and_multiword_fillers():
    result = analyze_fluency("Um, I mean, the model learns. Like, from data.", 60, ["um", "I mean", "like"])
    assert result["filler_count"] == 3
    assert result["filler_counts"] == {"um": 1, "I mean": 1, "like": 1}
    assert result["filler_rate_per_minute"] == 3


def test_empty_transcript_has_zero_words_and_fillers():
    result = analyze_fluency("", 0, ["um"])
    assert result["word_count"] == 0
    assert result["filler_count"] == 0
    assert result["words_per_minute"] == 0