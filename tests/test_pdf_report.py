import pytest


def test_pdf_report_is_generated_in_memory():
    pytest.importorskip("reportlab")
    from reports.pdf_report import generate_pdf_report

    result = {
        "session_id": "test-session", "concept": "Machine Learning", "reference": "Learn from data.",
        "transcription": "It learns patterns from data.",
        "semantic": {"similarity": 0.8, "similarity_percent": 80, "interpretation": "Strong Understanding", "covered_concepts": ["Learn from data"], "weak_concepts": [], "missing_concepts": []},
        "audio": {"duration_seconds": 12, "speech_duration_seconds": 9, "pause_count": 1, "pause_ratio": 0.1, "rms_energy": 0.04, "speech_activity_ratio": 0.8},
        "fluency": {"filler_count": 1, "filler_rate_per_minute": 5},
        "scores": {"overall_score": 80, "understanding_score": 82, "fluency_score": 76, "communication_score": 80, "performance_level": "Strong"},
        "feedback": {"summary": "Clear summary.", "strengths": ["Good coverage."], "improvement_suggestions": ["Add an example."]},
    }
    output = generate_pdf_report(result)
    assert output.startswith(b"%PDF")
    assert len(output) > 1000