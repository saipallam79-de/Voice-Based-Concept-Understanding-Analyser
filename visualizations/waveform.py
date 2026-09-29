"""Memory-conscious waveform image generation."""

from __future__ import annotations

import io


def waveform_png(audio_path: str) -> bytes:
    import librosa
    import matplotlib.pyplot as plt

    signal, sample_rate = librosa.load(audio_path, sr=22050, mono=True)
    figure, axis = plt.subplots(figsize=(10, 2.5), constrained_layout=True)
    try:
        stride = max(1, signal.size // 100_000)
        indices = range(0, signal.size, stride)
        times = librosa.samples_to_time(indices, sr=sample_rate)
        axis.plot(times, signal[::stride], color="#187f72", linewidth=0.45)
        axis.set(xlabel="Time (seconds)", ylabel="Amplitude", title="Speech waveform")
        axis.grid(alpha=0.18)
        output = io.BytesIO()
        figure.savefig(output, format="png", dpi=140, facecolor="white")
        return output.getvalue()
    finally:
        plt.close(figure)