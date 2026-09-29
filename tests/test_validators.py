import io
import wave

import pytest

from utils.validators import ValidationError, validate_audio_extension, validate_audio_payload


def wav_bytes(duration=0.25, sample_rate=8000):
    buffer = io.BytesIO()
    frames = b"\0\0" * int(duration * sample_rate)
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(frames)
    return buffer.getvalue()


def test_audio_extension_is_case_insensitive():
    assert validate_audio_extension("lesson.WAV") == "wav"


@pytest.mark.parametrize("filename", ["recording.exe", "no-extension", "voice.ogg"])
def test_rejects_unsupported_audio(filename):
    with pytest.raises(ValidationError):
        validate_audio_extension(filename)


def test_rejects_empty_and_corrupt_audio():
    with pytest.raises(ValidationError, match="empty"):
        validate_audio_payload(b"", "voice.wav")
    with pytest.raises(ValidationError, match="could not be decoded"):
        validate_audio_payload(b"not a wave", "voice.wav")


def test_validates_wav_and_returns_duration():
    assert validate_audio_payload(wav_bytes(), "voice.wav") == pytest.approx(0.25)


def test_rejects_upload_over_size_limit():
    with pytest.raises(ValidationError, match="smaller"):
        validate_audio_payload(wav_bytes(), "voice.wav", max_size_mb=0)