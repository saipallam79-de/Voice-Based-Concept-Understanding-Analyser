"""Generate a downloadable PDF report in memory."""

from __future__ import annotations

import io
import logging
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

logger = logging.getLogger(__name__)


def generate_pdf_report(result: dict, waveform: bytes | None = None) -> bytes:
    """Render the evaluation, transcript, coverage, and recommendations to PDF bytes."""
    output = io.BytesIO()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="VBCUA_Title", parent=styles["Title"], textColor=colors.HexColor("#12695f"), alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=11))

    def paragraph(text: str, style: str = "BodyText") -> Paragraph:
        return Paragraph(escape(str(text)).replace("\n", "<br/>"), styles[style])

    try:
        document = SimpleDocTemplate(
            output, pagesize=letter, rightMargin=0.65 * inch,
            leftMargin=0.65 * inch, topMargin=0.55 * inch, bottomMargin=0.55 * inch,
            title="VBCUA Evaluation Report",
        )
        scores, semantic = result["scores"], result["semantic"]
        audio, fluency = result["audio"], result["fluency"]
        story = [
            paragraph("VBCUA", "VBCUA_Title"),
            paragraph("Voice-Based Concept Understanding Analyser", "Heading2"),
            paragraph(f"Evaluation date: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}"),
            paragraph(f"Session: {result.get('session_id', 'Local evaluation')}"),
            Spacer(1, 12),
            paragraph(f"Concept: {result['concept']}", "Heading2"),
            paragraph("Reference explanation", "Heading3"), paragraph(result["reference"]),
            Spacer(1, 10),
        ]
        score_data = [
            ["Overall", "Understanding", "Fluency", "Communication"],
            [f"{scores['overall_score']:.1f}/100", f"{scores['understanding_score']:.1f}/100", f"{scores['fluency_score']:.1f}/100", f"{scores['communication_score']:.1f}/100"],
        ]
        table = Table(score_data, colWidths=[1.7 * inch] * 4)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12695f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d7e2df")),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.extend([table, Spacer(1, 10), paragraph(f"Performance level: {scores['performance_level']}", "Heading3")])
        story.extend([
            paragraph("Speech analysis", "Heading2"),
            paragraph(f"Semantic similarity: {semantic['similarity_percent']:.1f}% ({semantic['interpretation']})"),
            paragraph(f"Audio duration: {audio['duration_seconds']:.1f} seconds"),
            paragraph(f"Estimated speech activity: {audio['speech_duration_seconds']:.1f} seconds"),
            paragraph(f"Filler words: {fluency['filler_count']} ({fluency['filler_rate_per_minute']:.1f} per minute)"),
            paragraph(f"Pauses: {audio['pause_count']} detected; pause ratio {audio['pause_ratio']:.1%}"),
            paragraph(f"RMS energy: {audio['rms_energy']:.4f}; speech activity: {audio['speech_activity_ratio']:.1%}"),
            paragraph("Concept coverage", "Heading3"),
        ])
        for label, key in (("Covered", "covered_concepts"), ("Needs clarification", "weak_concepts"), ("Missing", "missing_concepts")):
            items = semantic[key]
            story.append(paragraph(f"{label}: " + ("; ".join(items) if items else "None identified.")))
        if waveform:
            story.extend([Spacer(1, 8), Image(io.BytesIO(waveform), width=7.0 * inch, height=1.75 * inch)])
        story.extend([
            PageBreak(), paragraph("Student transcription", "Heading2"),
            paragraph(result.get("transcription") or "No speech was transcribed."),
            Spacer(1, 12), paragraph("Feedback", "Heading2"),
        ])
        feedback = result["feedback"]
        story.append(paragraph(feedback["summary"]))
        for heading, key in (("Strengths", "strengths"), ("Recommendations", "improvement_suggestions")):
            story.append(paragraph(heading, "Heading3"))
            for item in feedback[key]:
                story.append(paragraph("- " + item, "Small"))
        document.build(story)
        return output.getvalue()
    except Exception:
        logger.exception("PDF report generation failed")
        raise RuntimeError("The evaluation PDF could not be generated.") from None