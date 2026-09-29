from database.database import delete_evaluation, get_evaluation, list_evaluations, save_evaluation


def sample_result():
    return {
        "session_id": "session-1", "timestamp": "2026-09-29T12:00:00+00:00",
        "concept": "Machine Learning", "transcription": "Learns from data.",
        "semantic": {"similarity": 0.8},
        "scores": {"understanding_score": 80, "fluency_score": 70, "communication_score": 90, "overall_score": 80},
        "audio": {"duration_seconds": 30, "pause_ratio": 0.1},
        "fluency": {"filler_count": 2},
    }


def test_database_round_trip_and_delete(tmp_path):
    path = tmp_path / "test.sqlite3"
    assert save_evaluation(sample_result(), path) == "session-1"
    assert get_evaluation("session-1", path)["concept"] == "Machine Learning"
    assert len(list_evaluations(path)) == 1
    assert delete_evaluation("session-1", path)
    assert get_evaluation("session-1", path) is None