"""Data shape for a persisted evaluation session."""

from dataclasses import asdict, dataclass


@dataclass
class EvaluationSession:
    session_id: str
    timestamp: str
    concept: str
    transcription: str
    similarity_score: float
    understanding_score: float
    fluency_score: float
    communication_score: float
    overall_score: float
    audio_duration: float
    filler_count: int
    pause_ratio: float
    result_json: str

    def as_dict(self) -> dict:
        return asdict(self)