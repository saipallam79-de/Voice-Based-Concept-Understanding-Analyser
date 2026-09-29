"""Evidence-based local feedback with optional Gemini enrichment."""

from __future__ import annotations

import logging

from config.settings import settings

logger = logging.getLogger(__name__)


def generate_feedback(result: dict, use_gemini: bool = False) -> dict:
    scores = result["scores"]
    semantic = result["semantic"]
    fluency = result["fluency"]
    audio = result["audio"]
    strengths: list[str] = []
    suggestions: list[str] = []

    if semantic["similarity"] >= settings.semantic_strong_threshold:
        strengths.append("The explanation closely matches the reference meaning.")
    if semantic["covered_concepts"]:
        strengths.append(f"Covered {len(semantic['covered_concepts'])} key concept(s).")
    if fluency["filler_count"] <= 2 and fluency["word_count"]:
        strengths.append("Few filler words were detected in the transcript.")
    if audio["pause_ratio"] < 0.2 and audio["speech_activity_ratio"] > 0.55:
        strengths.append("Speech activity was relatively continuous in this recording.")
    if not strengths:
        strengths.append("A complete recording and transcript are available for review.")

    missing = semantic["missing_concepts"]
    weak = semantic["weak_concepts"]
    if missing:
        suggestions.append("Add a brief explanation of: " + "; ".join(missing[:3]) + ".")
    if weak:
        suggestions.append("Clarify these points with a definition or example: " + "; ".join(weak[:2]) + ".")
    if fluency["filler_count"] >= 3:
        suggestions.append("Try replacing repeated fillers with a short pause before the next idea.")
    if audio["pause_ratio"] > 0.35:
        suggestions.append("Review the longer silent gaps and try grouping related ideas into complete phrases.")
    if fluency["word_count"] < 30:
        suggestions.append("Develop the explanation with a definition, a mechanism, and one concrete example.")
    if not suggestions:
        suggestions.append("Keep the explanation focused, and support the key points with a concrete example.")

    summary = (
        f"Your overall result is {scores['overall_score']:.1f}/100 ({scores['performance_level']}). "
        f"Concept understanding was {scores['understanding_score']:.1f}/100; delivery indicators are reported separately."
    )
    feedback = {
        "summary": summary,
        "strengths": strengths,
        "missing_concepts": missing,
        "improvement_suggestions": suggestions,
        "source": "local",
    }
    if use_gemini and settings.gemini_api_key:
        feedback = _enrich_with_gemini(result, feedback)
    return feedback


def _enrich_with_gemini(result: dict, fallback: dict) -> dict:
    try:
        from google import genai

        client = genai.Client(api_key=settings.gemini_api_key)
        evidence = {
            "concept": result["concept"],
            "reference": result["reference"],
            "transcription": result["transcription"],
            "covered_concepts": result["semantic"]["covered_concepts"],
            "weak_concepts": result["semantic"]["weak_concepts"],
            "missing_concepts": result["semantic"]["missing_concepts"],
            "similarity_percent": result["semantic"]["similarity_percent"],
            "scores": result["scores"],
        }
        prompt = (
            "Give concise, respectful, evidence-based feedback for this student's explanation. "
            "Do not invent evidence or concepts. Treat scores as indicators, not psychological traits. "
            f"Analysis JSON: {evidence}"
        )
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        text = (response.text or "").strip()
        if text:
            return {**fallback, "summary": text, "source": "gemini"}
    except Exception:
        logger.exception("Optional Gemini feedback failed; using deterministic feedback")
    return fallback