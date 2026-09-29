"""Robust speech-delivery signal metrics computed with librosa."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def analyze_audio(audio_path: str, sample_rate: int = 22050) -> dict:
    """Return duration, energy, activity, and contiguous silence regions."""
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("Audio analysis requires Librosa. Install the project requirements and restart the app.") from exc

    try:
        signal, actual_rate = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception as exc:
        logger.exception("Audio decoding failed during signal analysis")
        raise ValueError("Audio could not be decoded for speech analysis.") from exc

    if signal.size == 0:
        raise ValueError("The audio recording contains no samples.")
    duration = float(signal.size / actual_rate)
    frame_length = min(2048, max(256, 2 ** int(np.floor(np.log2(signal.size)))))
    hop_length = max(1, min(512, frame_length // 4))
    rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
    threshold = max(0.008, float(rms.max(initial=0.0)) * 0.08)
    active = rms > threshold
    frame_seconds = hop_length / actual_rate
    pause_regions: list[dict[str, float]] = []
    start = None
    for index, is_active in enumerate(active):
        if not is_active and start is None:
            start = index
        elif is_active and start is not None:
            end = index * frame_seconds
            begin = start * frame_seconds
            if end - begin >= 0.25:
                pause_regions.append({"start": begin, "end": min(end, duration), "duration": end - begin})
            start = None
    if start is not None:
        begin = start * frame_seconds
        if duration - begin >= 0.25:
            pause_regions.append({"start": begin, "end": duration, "duration": duration - begin})

    pause_duration = sum(region["duration"] for region in pause_regions)
    activity = float(np.mean(active)) if active.size else 0.0
    return {
        "duration_seconds": duration,
        "speech_duration_seconds": activity * duration,
        "rms_energy": float(np.mean(rms)) if rms.size else 0.0,
        "rms_peak": float(np.max(rms)) if rms.size else 0.0,
        "zero_crossing_rate": float(np.mean(zcr)) if zcr.size else 0.0,
        "pause_regions": pause_regions,
        "pause_count": len(pause_regions),
        "pause_duration_seconds": min(pause_duration, duration),
        "pause_ratio": min(1.0, pause_duration / duration) if duration else 0.0,
        "speech_activity_ratio": activity,
    }