"""Streamlit interface for the Voice-Based Concept Understanding Analyser."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import streamlit as st

from config.settings import settings
from database.database import (
    delete_evaluation,
    get_evaluation,
    initialize_database,
    list_evaluations,
    save_evaluation,
)
from modules.audio_analysis import analyze_audio
from modules.feedback import generate_feedback
from modules.fluency_analysis import analyze_fluency
from modules.scoring import calculate_scores
from modules.semantic_analysis import analyze_semantics
from modules.transcription import transcribe_audio
from models.whisper_model import check_whisper_dependencies
from reports.pdf_report import generate_pdf_report
from utils.constants import CONCEPTS, FILLER_WORDS
from utils.helpers import save_uploaded_audio
from utils.validators import ValidationError, validate_audio_payload
from visualizations.waveform import waveform_png

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="VBCUA | Concept Analysis", page_icon="🎙️", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
    :root { --ink:#17312e; --muted:#607470; --teal:#14796d; --mint:#dff2ed; --line:#dce7e3; --coral:#d87559; }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; color:var(--ink); }
    h1, h2, h3 { font-family:'Manrope', sans-serif; letter-spacing:0; }
    .stApp { background:linear-gradient(135deg,#f7faf8 0%,#f1f7f5 58%,#f9f7f1 100%); }
    [data-testid="stSidebar"] { background:#edf5f2; border-right:1px solid var(--line); }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:8px; padding:16px 18px; }
    [data-testid="stMetricLabel"] { color:var(--muted); }
    .hero { padding:30px 34px; border-radius:10px; color:#f7fffc; background:linear-gradient(112deg,#12695f,#238d7c 58%,#78a990); margin:8px 0 22px; }
    .hero h1 { color:white; font-size:38px; margin:0 0 8px; }
    .hero p { color:#e0f5ee; font-size:17px; margin:0; }
    .feature { background:#fff; border:1px solid var(--line); border-top:3px solid var(--teal); border-radius:7px; padding:18px; min-height:126px; }
    .feature b { display:block; font-size:17px; margin:9px 0 5px; }
    .feature span { color:var(--muted); font-size:14px; }
    .eyebrow { color:var(--teal); font-weight:700; text-transform:uppercase; font-size:12px; letter-spacing:1px; }
    .stButton > button[kind="primary"] { background:#14796d; border-color:#14796d; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _navigate_to(page: str) -> None:
    st.session_state.current_page = page


def go_to_analyze() -> None:
    _navigate_to("Analyze Concept")


def go_to_results() -> None:
    _navigate_to("Results")


def go_to_history() -> None:
    _navigate_to("History")


def _sync_page_from_sidebar() -> None:
    st.session_state.current_page = st.session_state.navigation


def _open_saved_evaluation(session_id: str) -> None:
    try:
        st.session_state.current_result = get_evaluation(session_id)
        st.session_state.current_waveform = None
        st.session_state.current_report_bytes = None
        st.session_state.current_page = "Results"
    except Exception:
        logger.exception("Could not open saved evaluation")
        st.session_state.history_error = "This saved evaluation could not be opened."


def _init_db() -> bool:
    try:
        initialize_database()
        return True
    except Exception:
        logger.exception("Database startup failed")
        return False


def _render_dashboard(db_ready: bool) -> None:
    st.markdown(
        '<section class="hero"><div class="eyebrow" style="color:#c6eadf">VBCUA / LEARNING ANALYTICS</div>'
        '<h1>Voice-Based Concept<br>Understanding Analyser</h1>'
        '<p>Evaluate your conceptual understanding through spoken explanations.</p></section>',
        unsafe_allow_html=True,
    )
    features = [
        ("🧠", "Concept understanding", "Compare your explanation with key ideas in a reference."),
        ("🎙️", "Speech analysis", "Review transcription, energy, pauses, and speech activity."),
        ("📊", "Fluency evaluation", "See measurable delivery indicators and filler-word counts."),
        ("📄", "Evaluation reports", "Download a clear PDF summary with actionable feedback."),
    ]
    columns = st.columns(4)
    for column, (icon, title, description) in zip(columns, features):
        with column:
            st.markdown(
                f'<div class="feature"><span>{icon}</span><b>{title}</b><span>{description}</span></div>',
                unsafe_allow_html=True,
            )
    st.write("")
    actions = st.columns(4)
    actions[0].button("Start Analysis", type="primary", use_container_width=True, on_click=go_to_analyze)
    actions[1].button("Analyze Concept", use_container_width=True, on_click=go_to_analyze)
    actions[2].button("View Results", use_container_width=True, on_click=go_to_results)
    actions[3].button("History", use_container_width=True, on_click=go_to_history)
    if not db_ready:
        st.warning("Local history is unavailable. Analysis can continue, but this run may not be saved.")
    else:
        try:
            history = list_evaluations()
            st.caption(f"{len(history)} saved evaluation{'s' if len(history) != 1 else ''} · Audio recordings are not retained in history.")
        except Exception:
            st.warning("Could not read the local evaluation history.")


def _render_analysis() -> None:
    st.title("Analyze a concept")
    st.caption("Choose a concept, upload a short explanation, and review evidence-based results.")
    options = [*CONCEPTS, "Custom Concept"]
    selected = st.selectbox("Concept", options)
    if selected == "Custom Concept":
        concept = st.text_input("Concept title", max_chars=100).strip()
        reference = st.text_area("Reference explanation", height=130, max_chars=5000).strip()
        key_concepts = []
        if reference:
            key_concepts = [sentence.strip() for sentence in reference.replace("?", ".").replace("!", ".").split(".") if sentence.strip()]
    else:
        concept = selected
        reference = CONCEPTS[selected]["reference"]
        key_concepts = CONCEPTS[selected]["key_concepts"]
        with st.expander("Reference explanation"):
            st.write(reference)
            st.caption(f"{len(key_concepts)} key concepts are checked independently.")

    uploaded = st.file_uploader("Upload your explanation", type=list(settings_audio_types()))
    use_gemini = False
    if settings.gemini_api_key:
        use_gemini = st.checkbox(
            "Send transcript and concept analysis to Google Gemini for feedback",
            value=False,
        )
    audio_bytes = uploaded.getvalue() if uploaded else None
    if uploaded:
        st.audio(audio_bytes, format=f"audio/{uploaded.name.rsplit('.', 1)[-1].lower()}")
        st.caption(f"{uploaded.name} · {len(audio_bytes) / (1024 * 1024):.2f} MB")
        try:
            duration = validate_audio_payload(
                audio_bytes, uploaded.name, settings.max_upload_mb, settings.max_audio_seconds
            )
            st.caption(f"Approximate duration: {duration:.1f} seconds")
            if duration < 2:
                st.info("Very short recordings may provide limited speech and concept evidence.")
        except ValidationError as exc:
            st.error(str(exc))

    run_analysis = st.button("Analyze explanation", type="primary", disabled=not (uploaded and concept and reference))
    if not run_analysis:
        return
    try:
        duration = validate_audio_payload(
            audio_bytes, uploaded.name, settings.max_upload_mb, settings.max_audio_seconds
        )
    except ValidationError as exc:
        st.error(str(exc))
        return

    saved_path = save_uploaded_audio(audio_bytes, uploaded.name.rsplit(".", 1)[-1], settings.audio_dir)
    progress = st.progress(0, text="Preparing audio")
    try:
        with st.status("Analyzing your explanation", expanded=True) as status:
            st.write("Step 1 / 7 · Audio validated")
            progress.progress(10, text="Transcribing speech")
            st.write("Step 2 / 7 · Transcribing speech with Whisper")
            transcript = transcribe_audio(str(saved_path))
            if not transcript["text"].strip():
                raise RuntimeError(
                    "Whisper returned an empty transcription. Semantic analysis was skipped; "
                    "check that the recording contains audible speech and retry."
                )
            progress.progress(34, text="Analyzing concept understanding")
            st.write("Step 3 / 7 · Comparing meaning and key concepts")
            semantic = analyze_semantics(reference, transcript["text"], key_concepts)
            progress.progress(55, text="Analyzing speech delivery")
            st.write("Step 4 / 7 · Measuring audio delivery indicators")
            audio = analyze_audio(str(saved_path))
            waveform = waveform_png(str(saved_path))
            fluency = analyze_fluency(transcript["text"], audio["duration_seconds"], FILLER_WORDS)
            progress.progress(73, text="Calculating evaluation scores")
            st.write("Step 5 / 7 · Calculating scores")
            scores = calculate_scores(semantic, fluency, audio, transcript)
            result = {
                "session_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "concept": concept,
                "reference": reference,
                "transcription": transcript["text"],
                "transcript_metrics": transcript,
                "semantic": semantic,
                "audio": audio,
                "fluency": fluency,
                "scores": scores,
            }
            progress.progress(86, text="Generating feedback")
            st.write("Step 6 / 7 · Preparing personalized feedback")
            result["feedback"] = generate_feedback(result, use_gemini=use_gemini)
            progress.progress(94, text="Saving evaluation")
            st.write("Step 7 / 7 · Preparing session report")
            try:
                report_bytes = generate_pdf_report(result, waveform)
            except RuntimeError as exc:
                result["pdf_error"] = str(exc)
                report_bytes = None
            try:
                result["session_id"] = save_evaluation(result)
            except Exception:
                logger.exception("Evaluation completed but could not be persisted")
                st.warning("Analysis succeeded, but the result could not be saved to History.")
            st.session_state.current_result = result
            st.session_state.current_waveform = waveform
            st.session_state.current_report_bytes = report_bytes
            status.update(label="Analysis complete", state="complete", expanded=False)
        progress.progress(100, text="Complete")
        st.success("Your evaluation is ready.")
        st.session_state.current_page = "Results"
        st.rerun()
    except Exception as exc:
        logger.exception("Analysis pipeline failed")
        st.error(str(exc) if isinstance(exc, (RuntimeError, ValueError)) else "Analysis could not be completed. Check the setup and try again.")
    finally:
        saved_path.unlink(missing_ok=True)


def settings_audio_types() -> tuple[str, ...]:
    from utils.constants import SUPPORTED_AUDIO_FORMATS

    return SUPPORTED_AUDIO_FORMATS


def _coverage_list(title: str, concepts: list[str], empty_text: str) -> None:
    st.subheader(title)
    if concepts:
        for item in concepts:
            st.markdown(f"- {item}")
    else:
        st.caption(empty_text)


def _render_results(result: dict | None, waveform: bytes | None = None, report_bytes: bytes | None = None) -> None:
    st.title("Evaluation results")
    if not result:
        st.info("No evaluation is open. Run an analysis or select a saved session from History.")
        return
    scores = result["scores"]
    st.caption(f"{result['concept']} · {result.get('timestamp', '')[:19].replace('T', ' ')} · {scores['performance_level']}")
    metrics = st.columns(4)
    for column, label, key in zip(
        metrics, ("Understanding", "Fluency", "Communication", "Overall"),
        ("understanding_score", "fluency_score", "communication_score", "overall_score"),
    ):
        column.metric(label, f"{scores[key]:.1f} / 100")
    st.progress(int(round(scores["overall_score"])), text=f"Performance: {scores['performance_level']}")
    if report_bytes:
        st.download_button(
            "Download PDF report", data=report_bytes, file_name=f"vbcua-{result['session_id'][:8]}.pdf",
            mime="application/pdf", type="primary",
        )
    elif result.get("pdf_error"):
        st.warning(result["pdf_error"])
    else:
        try:
            report = generate_pdf_report(result, waveform)
            st.download_button("Download PDF report", report, f"vbcua-{result['session_id'][:8]}.pdf", "application/pdf")
        except RuntimeError as exc:
            st.warning(str(exc))

    overview, understanding, speech, feedback_tab, transcript = st.tabs(
        ["Overview", "Understanding", "Speech Analysis", "Feedback", "Transcript"]
    )
    with overview:
        left, right = st.columns([1, 1])
        left.metric("Semantic similarity", f"{result['semantic']['similarity_percent']:.1f}%")
        right.metric("Audio duration", f"{result['audio']['duration_seconds']:.1f} sec")
        st.info(scores["scoring_explanation"])
        if waveform:
            st.image(waveform, caption="Speech waveform", use_container_width=True)
    with understanding:
        st.write(f"**Similarity interpretation:** {result['semantic']['interpretation']}")
        st.caption("Each key point is compared with the transcript independently using Sentence-BERT cosine similarity.")
        _coverage_list("Covered concepts", result["semantic"]["covered_concepts"], "No reference key points crossed the configured coverage threshold.")
        _coverage_list("Potentially weak concepts", result["semantic"]["weak_concepts"], "No weak concepts identified.")
        _coverage_list("Missing concepts", result["semantic"]["missing_concepts"], "No missing concepts identified.")
    with speech:
        audio, fluency = result["audio"], result["fluency"]
        cols = st.columns(4)
        cols[0].metric("Filler words", fluency["filler_count"])
        cols[1].metric("Fillers / minute", f"{fluency['filler_rate_per_minute']:.1f}")
        cols[2].metric("Pauses", audio["pause_count"])
        cols[3].metric("Pause ratio", f"{audio['pause_ratio']:.1%}")
        cols = st.columns(3)
        cols[0].metric("Speech activity", f"{audio['speech_activity_ratio']:.1%}")
        cols[1].metric("Estimated speech", f"{audio['speech_duration_seconds']:.1f} sec")
        cols[2].metric("Mean RMS energy", f"{audio['rms_energy']:.4f}")
        st.caption(f"Zero crossing rate: {audio['zero_crossing_rate']:.4f}")
        if fluency["filler_counts"]:
            st.write("**Detected fillers:** " + ", ".join(f"{word} ({count})" for word, count in fluency["filler_counts"].items()))
        if waveform:
            st.image(waveform, caption="Speech waveform", use_container_width=True)
        st.caption("These speech delivery indicators describe signal and transcript patterns, not confidence or personality.")
    with feedback_tab:
        st.subheader("Summary")
        st.write(result["feedback"]["summary"])
        st.subheader("Strengths")
        for item in result["feedback"]["strengths"]:
            st.markdown(f"- {item}")
        st.subheader("Suggestions")
        for item in result["feedback"]["improvement_suggestions"]:
            st.markdown(f"- {item}")
    with transcript:
        st.caption(f"{result['transcript_metrics']['word_count']} words · {result['transcript_metrics']['sentence_count']} sentences")
        st.text_area("Transcription", value=result["transcription"] or "No speech was transcribed.", height=280, disabled=True)
        with st.expander("Reference explanation"):
            st.write(result["reference"])


def _render_history() -> None:
    st.title("Evaluation history")
    if error := st.session_state.pop("history_error", None):
        st.error(error)
    try:
        rows = list_evaluations()
    except Exception:
        st.error("Stored history could not be loaded. Check local database access.")
        return
    if not rows:
        st.info("No saved evaluations yet.")
        return
    st.dataframe(
        [{
            "Date": row["timestamp"][:19].replace("T", " "), "Concept": row["concept"],
            "Understanding": row["understanding_score"], "Fluency": row["fluency_score"],
            "Communication": row["communication_score"], "Overall": row["overall_score"],
        } for row in rows],
        use_container_width=True, hide_index=True,
    )
    selected_id = st.selectbox("Open saved evaluation", [row["session_id"] for row in rows], format_func=lambda session_id: next(
        f"{row['concept']} · {row['timestamp'][:10]} · {session_id[:8]}" for row in rows if row["session_id"] == session_id
    ))
    open_col, delete_col = st.columns([1, 1])
    with open_col:
        st.button(
            "Open result", type="primary", on_click=_open_saved_evaluation,
            args=(selected_id,),
        )
    with delete_col:
        if st.button("Delete selected evaluation"):
            try:
                delete_evaluation(selected_id)
                if (st.session_state.get("current_result") or {}).get("session_id") == selected_id:
                    st.session_state.current_result = None
                    st.session_state.current_waveform = None
                    st.session_state.current_report_bytes = None
                st.success("Evaluation deleted.")
                st.rerun()
            except Exception:
                st.error("The evaluation could not be deleted.")


def _render_about() -> None:
    st.title("About VBCUA")
    st.write("VBCUA compares a spoken explanation with a reference and reports semantic, transcript, and acoustic indicators.")
    st.subheader("How results are produced")
    st.markdown(
        "1. Whisper transcribes the uploaded audio.\n"
        "2. Sentence-BERT compares the transcript and reference, including key concepts.\n"
        "3. Librosa measures energy, pauses, zero-crossing rate, and speech activity.\n"
        "4. Rule-based scoring and feedback summarize those measured inputs.\n"
        "5. SQLite keeps evaluation summaries locally; the uploaded audio is removed after analysis."
    )
    st.warning("Scores are learning aids, not a substitute for instructor judgment. Accent, recording quality, and transcription errors can affect results.")
    st.caption(f"Whisper model: {settings.whisper_model} · Sentence-Transformer: {settings.semantic_model}")
    st.subheader("Speech recognition diagnostics")
    diagnostics = check_whisper_dependencies()
    columns = st.columns(4)
    columns[0].metric("Whisper", "Installed" if diagnostics["whisper"]["available"] else "Missing")
    columns[1].metric("PyTorch", "Installed" if diagnostics["torch"]["available"] else "Missing")
    columns[2].metric("FFmpeg", "Available" if diagnostics["ffmpeg"]["available"] else "Missing")
    columns[3].metric("Model", diagnostics["model"]["status"])
    for package_name, package in diagnostics["packages"].items():
        if not package["available"]:
            st.error(f"{package_name}: {package['error']}")
    if not diagnostics["ffmpeg"]["available"]:
        st.error(diagnostics["ffmpeg"]["error"])
    if diagnostics["model"]["error"]:
        st.error(f"Model loading detail: {diagnostics['model']['error']}")
    if st.button("Test Whisper model loading", help="Loads the configured model now; the first run may download model weights."):
        with st.spinner(f"Loading Whisper '{settings.whisper_model}' for the diagnostic..."):
            diagnostics = check_whisper_dependencies(load_model=True)
        st.metric("Model diagnostic", diagnostics["model"]["status"])
        if diagnostics["model"]["error"]:
            st.error(diagnostics["model"]["error"])
        elif diagnostics["model"]["loaded"]:
            st.success(f"Whisper '{diagnostics['model']['name']}' loaded successfully.")


def main() -> None:
    logger.info("VBCUA application startup")
    db_ready = _init_db()
    if "current_page" not in st.session_state:
        st.session_state.current_page = st.session_state.get("navigation", "Dashboard")
    # Synchronize callback-driven routes before the keyed radio is instantiated.
    if st.session_state.get("navigation") != st.session_state.current_page:
        st.session_state.navigation = st.session_state.current_page
    with st.sidebar:
        st.markdown("## VBCUA")
        st.caption("VOICE-BASED LEARNING ANALYSIS")
        st.radio(
            "Navigation", ["Dashboard", "Analyze Concept", "Results", "History", "About"],
            key="navigation", on_change=_sync_page_from_sidebar,
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"Whisper · {settings.whisper_model}")
        st.caption("Audio is processed locally; saved history excludes recordings.")
    page = st.session_state.current_page
    if page == "Dashboard":
        _render_dashboard(db_ready)
    elif page == "Analyze Concept":
        _render_analysis()
    elif page == "Results":
        _render_results(
            st.session_state.get("current_result"),
            st.session_state.get("current_waveform"),
            st.session_state.get("current_report_bytes"),
        )
    elif page == "History":
        _render_history()
    else:
        _render_about()


if __name__ == "__main__":
    main()