# VBCUA

**Voice-Based Concept Understanding Analyser** evaluates an explanation by comparing its transcript with a reference concept and measuring observable speech-delivery indicators.

## Features

- OpenAI Whisper transcription, loaded only when an analysis is started.
- Sentence-Transformers semantic similarity and independent key-concept coverage.
- Librosa duration, RMS energy, zero-crossing rate, estimated activity, and pause analysis.
- Configurable filler-word counts and centralized understanding, fluency, communication, and overall scores.
- Evidence-based local feedback, with optional Google Gemini enrichment.
- PDF reports with transcript, concept coverage, scores, recommendations, and waveform.
- SQLite history containing evaluation summaries; uploaded audio is removed after each run.
- Custom reference concepts, safe upload validation, and unit tests.

## Architecture

`app.py` is the Streamlit presentation and orchestration layer. Independently testable business logic lives under `modules/`; cached loaders are in `models/`; persistence, visualization, and PDF generation are separated into `database/`, `visualizations/`, and `reports/`. Configuration and sample concepts live under `config/` and `utils/`.

```text
app.py
  -> validators.py -> transcription.py -> semantic_analysis.py
  -> audio_analysis.py + fluency_analysis.py -> scoring.py
  -> feedback.py -> pdf_report.py -> database.py
```

## Technology

Python 3.10+, Streamlit, OpenAI Whisper, Sentence-Transformers / PyTorch, NLTK, Librosa / SoundFile, NumPy, Matplotlib, ReportLab, SQLite, and Pytest. Gemini is optional.

## Project Structure

```text
VBCUA/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── config/              # Environment-driven settings
├── models/              # Cached Whisper and Sentence-Transformer loaders
├── modules/             # Transcription, semantic/audio analysis, scoring, feedback
├── reports/             # In-memory PDF reports
├── visualizations/      # Waveform rendering
├── database/            # SQLite session persistence
├── utils/               # Concepts, validation, storage, text helpers
├── data/audio/          # Temporary uploads (removed after processing)
├── data/results/        # Reserved for local generated artifacts
├── tests/               # Pytest unit and integration tests
└── assets/
```

## Installation

Use Python 3.10 or newer. A CPU-capable PyTorch installation works, though inference will be slower. From PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

PyTorch wheels differ by operating system and accelerator. If the default dependency resolver does not select the desired CPU/CUDA build, install the appropriate PyTorch build from the official PyTorch selector, then install the remaining requirements.

## Run

```powershell
streamlit run app.py
```

The first analysis downloads/loads the configured Whisper and Sentence-Transformer model if they are not cached locally. Select a sample concept or enter a custom title and reference, upload a WAV, MP3, M4A, or FLAC recording, then choose **Analyze explanation**. Review the overview, understanding, speech, feedback, and transcript tabs, and download the PDF report. History lets you reopen or delete saved evaluations.

Whisper requires the FFmpeg executable on `PATH`. On Windows, one installation option is `winget install --id Gyan.FFmpeg.Shared --exact`; restart the terminal and Streamlit after installation. The About page checks package imports and FFmpeg, and includes an explicit button to test loading the configured model. Model weights are not downloaded when the app first opens.

## Configuration

Copy `.env.example` to `.env`. Supported options include:

| Variable | Default | Purpose |
| --- | --- | --- |
| `WHISPER_MODEL` | `base` | Whisper model name (`tiny`, `base`, `small`, `medium`) |
| `VBCUA_SEMANTIC_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence-Transformer model identifier |
| `VBCUA_MAX_UPLOAD_MB` | `25` | Maximum upload size |
| `VBCUA_MAX_AUDIO_SECONDS` | `600` | Maximum recording duration |
| `VBCUA_UNDERSTANDING_WEIGHT` | `0.60` | Overall score weight |
| `VBCUA_FLUENCY_WEIGHT` | `0.25` | Overall score weight |
| `VBCUA_COMMUNICATION_WEIGHT` | `0.15` | Overall score weight |
| `VBCUA_SEMANTIC_STRONG` | `0.70` | Strong semantic interpretation boundary |
| `VBCUA_SEMANTIC_MODERATE` | `0.50` | Moderate semantic interpretation boundary |
| `VBCUA_COVERAGE_THRESHOLD` | `0.48` | Minimum key-point cosine score for covered |
| `VBCUA_WEAK_COVERAGE_THRESHOLD` | `0.34` | Lower bound for potentially weak coverage |
| `VBCUA_DATABASE_PATH` | `data/vbcua.sqlite3` | Optional local SQLite path |
| `GEMINI_API_KEY` | empty | Enables optional Gemini feedback when configured |

Overall score weights must be non-negative and sum to 1.0. Similarity and coverage cutoffs are configurable heuristics, not universal educational standards. Gemini is never needed for the normal workflow; without a key or successful API call, feedback remains deterministic and local.

## Resource Notes

Whisper models are downloaded on first use and cached in memory for the Streamlit process. Approximate model weights are about 75 MB (`tiny`), 150 MB (`base`), 500 MB (`small`), 1.5 GB (`medium`), and 3 GB (`large`); inference needs additional RAM and varies with PyTorch, precision, and audio length. `small` is the default and is the practical starting point for a laptop with several GB of free memory. `all-MiniLM-L6-v2` has roughly 90 MB of model weights. CPU inference is supported but may take several minutes for longer clips.

## Tests

```powershell
python -m pytest -q
```

Audio integration tests skip when Librosa or SoundFile is not installed. Model-loading and real inference tests require the dependencies, model download, and (for Whisper) compatible FFmpeg availability; tests do not download large model weights by default.

## Configuration and Privacy

Uploads are checked for extension, size, and decodability before use, saved under a generated UUID filename while processing, then deleted in a `finally` block. SQLite retains the transcript, concept, timestamp, measured scores, duration, filler count, and pause ratio, but not the audio. Use **Delete selected evaluation** in History or remove the local `data/vbcua.sqlite3` file to clear history. Avoid entering sensitive personal information in custom concepts or spoken recordings.

## Screenshots

_Placeholder: add screenshots of Dashboard, Analysis, Results, and History after launch._

## Limitations and Future Work

- Whisper transcription can vary with language, accent, noise, and recording quality; transcript errors affect semantic and filler results.
- Key-concept scoring compares whole key-point phrases against the full transcript; nuanced paraphrases may be under- or over-estimated.
- Silence/activity thresholds are signal heuristics. They do not measure confidence, effort, or psychological state.
- Local models need disk space, initial network access, and substantial memory. Gemini feedback requires a user-provided API key and network access.
- Future improvements could add multilingual reference sets, human-calibrated rubrics, configurable concept editing, anonymized export, and longitudinal instructor views.

## Project Information

VBCUA is an educational analysis tool scaffold intended for local use and further institutional review. Before deployment with learners, establish consent, retention, accessibility, and data-governance policies appropriate to the setting.