import numpy as np
import pytest


def _write_wave(path, samples, sample_rate=16000):
    import soundfile as sf

    sf.write(path, samples, sample_rate)


def test_audio_analysis_handles_silence_and_reports_metrics(tmp_path):
    pytest.importorskip("librosa")
    pytest.importorskip("soundfile")
    from modules.audio_analysis import analyze_audio

    path = tmp_path / "silence.wav"
    _write_wave(path, np.zeros(4000, dtype=np.float32))
    result = analyze_audio(str(path))
    assert result["duration_seconds"] == pytest.approx(0.25, abs=0.02)
    assert result["speech_duration_seconds"] == pytest.approx(0)
    assert result["rms_energy"] == pytest.approx(0)
    assert result["pause_ratio"] == 1


def test_audio_analysis_rejects_undecodable_file(tmp_path):
    pytest.importorskip("librosa")
    path = tmp_path / "broken.wav"
    path.write_bytes(b"broken audio")
    from modules.audio_analysis import analyze_audio

    with pytest.raises(ValueError, match="could not be decoded"):
        analyze_audio(str(path))