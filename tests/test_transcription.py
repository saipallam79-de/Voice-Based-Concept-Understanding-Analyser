from modules.transcription import transcribe_audio


def ready_diagnostics(*args, **kwargs):
    return {
        "ready": True,
        "packages": {},
        "ffmpeg": {"available": True, "error": None},
    }


class FakeWhisper:
    def transcribe(self, path, fp16=False):
        assert path == "sample.wav"
        assert fp16 is False
        return {"text": "  Speech has clear words. ", "language": "en"}


def test_transcription_returns_text_metrics(monkeypatch):
    monkeypatch.setattr("modules.transcription.check_whisper_dependencies", ready_diagnostics)
    monkeypatch.setattr("modules.transcription.load_whisper_model", lambda _: FakeWhisper())
    result = transcribe_audio("sample.wav", "tiny")
    assert result["text"] == "Speech has clear words."
    assert result["word_count"] == 4
    assert result["sentence_count"] == 1
    assert result["language"] == "en"


def test_transcription_wraps_model_errors(monkeypatch):
    monkeypatch.setattr("modules.transcription.check_whisper_dependencies", ready_diagnostics)
    monkeypatch.setattr("modules.transcription.load_whisper_model", lambda _: (_ for _ in ()).throw(RuntimeError("load")))
    try:
        transcribe_audio("sample.wav")
    except RuntimeError as exc:
        assert "OpenAI Whisper transcription failed" in str(exc)
    else:
        raise AssertionError("expected transcription failure")


def test_missing_prerequisite_stops_before_loading_model(monkeypatch):
    monkeypatch.setattr(
        "modules.transcription.check_whisper_dependencies",
        lambda *args, **kwargs: {
            "ready": False,
            "packages": {"whisper": {"available": False, "error": "ModuleNotFoundError: whisper"}},
            "ffmpeg": {"available": False, "error": "FFmpeg missing from PATH."},
        },
    )
    monkeypatch.setattr(
        "modules.transcription.load_whisper_model",
        lambda _: (_ for _ in ()).throw(AssertionError("model must not load")),
    )
    try:
        transcribe_audio("sample.wav")
    except RuntimeError as exc:
        assert "ModuleNotFoundError: whisper" in str(exc)
        assert "FFmpeg missing from PATH" in str(exc)
    else:
        raise AssertionError("expected missing-prerequisite failure")