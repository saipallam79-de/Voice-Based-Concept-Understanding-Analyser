from models.whisper_model import check_whisper_dependencies


def test_dependency_diagnostic_reports_missing_prerequisites(monkeypatch):
    packages = {
        name: {"available": True, "version": "1.0", "error": None}
        for name in ("whisper", "torch", "numpy", "tqdm", "numba", "more-itertools", "tiktoken", "requests")
    }
    packages["whisper"] = {
        "available": False,
        "version": None,
        "error": "ModuleNotFoundError: No module named 'whisper'",
    }
    monkeypatch.setattr("models.whisper_model._check_packages", lambda: packages)
    monkeypatch.setattr(
        "models.whisper_model._check_ffmpeg",
        lambda: {"available": False, "version": None, "error": "FFmpeg missing from PATH."},
    )

    result = check_whisper_dependencies("base")

    assert result["whisper"]["available"] is False
    assert result["torch"]["available"] is True
    assert result["ffmpeg"]["available"] is False
    assert result["model"]["status"] == "Failed"
    assert "No module named 'whisper'" in result["model"]["error"]
    assert "FFmpeg missing from PATH" in result["model"]["error"]
    assert result["ready"] is False