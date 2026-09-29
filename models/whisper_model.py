"""Lazy OpenAI Whisper loading and explicit environment diagnostics."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)
SUPPORTED_WHISPER_MODELS = {"tiny", "base", "small", "medium"}
REQUIRED_PACKAGES = {
    "whisper": ("openai-whisper", "whisper"),
    "torch": ("torch", "torch"),
    "numpy": ("numpy", "numpy"),
    "tqdm": ("tqdm", "tqdm"),
    "numba": ("numba", "numba"),
    "more-itertools": ("more-itertools", "more_itertools"),
    "tiktoken": ("tiktoken", "tiktoken"),
    "requests": ("requests", "requests"),
}
_loaded_models: dict[str, Any] = {}
_model_errors: dict[str, str] = {}

try:
    import streamlit as st

    _cache_resource = st.cache_resource(show_spinner=False)
except ImportError:
    _cache_resource = lru_cache(maxsize=4)


class WhisperModelError(RuntimeError):
    """Whisper setup, checkpoint download, or model-loading failure."""


def _model_name(model_name: str | None = None) -> str:
    name = (model_name or settings.whisper_model).strip().lower()
    if name not in SUPPORTED_WHISPER_MODELS:
        supported = ", ".join(sorted(SUPPORTED_WHISPER_MODELS))
        raise ValueError(f"Unsupported Whisper model '{name}'. Choose from: {supported}.")
    return name


def _check_ffmpeg() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=True
        )
        first_line = result.stdout.splitlines()[0] if result.stdout else "FFmpeg is available."
        return {"available": True, "version": first_line, "error": None}
    except FileNotFoundError:
        return {
            "available": False,
            "version": None,
            "error": (
                "FFmpeg executable was not found. Install FFmpeg (for example, "
                "`winget install --id Gyan.FFmpeg.Shared --exact`), add its bin folder "
                "to PATH, then restart the terminal and Streamlit."
            ),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        return {"available": False, "version": None, "error": f"FFmpeg check failed: {detail}"}


def _check_packages() -> dict[str, dict[str, Any]]:
    modules = [module_name for _, module_name in REQUIRED_PACKAGES.values()]
    import_script = (
        "import importlib, sys\n"
        f"for name in {modules!r}:\n"
        " print('CHECKING\\t' + name, flush=True)\n"
        " try:\n"
        "  module = importlib.import_module(name)\n"
        "  version = str(getattr(module, '__version__', 'unknown')).replace('\\n', ' ')\n"
        "  print('RESULT\\t' + name + '\\tOK\\t' + version, flush=True)\n"
        " except BaseException as exc:\n"
        "  detail = (type(exc).__name__ + ': ' + str(exc)).replace('\\n', ' ')\n"
        "  print('RESULT\\t' + name + '\\tFAIL\\t' + detail, flush=True)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", import_script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout.splitlines()
        checked = {line.split("\t", 3)[1]: line.split("\t", 3) for line in output if line.startswith("RESULT\t")}
        last_check = next((line.split("\t", 1)[1] for line in reversed(output) if line.startswith("CHECKING\t")), None)
        packages = {}
        for display_name, (distribution, module_name) in REQUIRED_PACKAGES.items():
            item = checked.get(module_name)
            if item and item[2] == "OK":
                packages[display_name] = {"available": True, "version": item[3], "error": None}
            elif item:
                packages[display_name] = {"available": False, "version": None, "error": item[3]}
            elif last_check == module_name and result.returncode:
                packages[display_name] = {
                    "available": False,
                    "version": None,
                    "error": f"Python import process exited unexpectedly with code {result.returncode}.",
                }
            else:
                packages[display_name] = {
                    "available": False,
                    "version": None,
                    "error": "Import diagnostic did not complete for this package.",
                }
    except subprocess.TimeoutExpired:
        packages = {
            display_name: {
                "available": False,
                "version": None,
                "error": "Package import check timed out after 60 seconds.",
            }
            for display_name in REQUIRED_PACKAGES
        }
    return packages


def check_whisper_dependencies(
    model_name: str | None = None, *, load_model: bool = False
) -> dict[str, Any]:
    """Report Whisper dependencies, FFmpeg, checkpoint, and optional model load status.

    Model weights are loaded only when ``load_model=True`` or transcription is requested.
    """
    name = _model_name(model_name)
    packages = _check_packages()
    ffmpeg = _check_ffmpeg()
    cache_dir = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "whisper"
    checkpoint = cache_dir / f"{name}.pt"
    ready = all(item["available"] for item in packages.values()) and ffmpeg["available"]
    model: dict[str, Any] = {
        "name": name,
        "loaded": name in _loaded_models,
        "checkpoint_cached": checkpoint.is_file(),
        "status": "Loaded" if name in _loaded_models else "Not tested",
        "error": _model_errors.get(name),
    }

    if load_model:
        unavailable = [key for key, item in packages.items() if not item["available"]]
        if unavailable:
            model["status"] = "Failed"
            model["error"] = _format_missing_requirements(
                {"packages": packages, "ffmpeg": ffmpeg}
            )
        elif not ffmpeg["available"]:
            model["status"] = "Failed"
            model["error"] = ffmpeg["error"]
        else:
            try:
                load_whisper_model(name)
                model.update(loaded=True, status="Loaded", error=None)
            except Exception as exc:
                model.update(
                    loaded=False,
                    status="Failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
    elif model["error"]:
        model["status"] = "Failed"
    elif not ready:
        model["status"] = "Failed"
        model["error"] = _format_missing_requirements(
            {"packages": packages, "ffmpeg": ffmpeg}
        )

    return {
        "packages": packages,
        "whisper": packages["whisper"],
        "torch": packages["torch"],
        "ffmpeg": ffmpeg,
        "model": model,
        "ready": ready,
    }


def _format_missing_requirements(diagnostics: dict[str, Any]) -> str:
    problems = [
        f"{name}: {info['error']}"
        for name, info in diagnostics["packages"].items()
        if not info["available"]
    ]
    if not diagnostics["ffmpeg"]["available"]:
        problems.append(diagnostics["ffmpeg"]["error"])
    return " ".join(problems)


@_cache_resource
def load_whisper_model(model_name: str | None = None):
    """Load OpenAI Whisper once per model name, preserving the original failure detail."""
    name = _model_name(model_name)
    logger.info("Loading OpenAI Whisper model '%s'", name)
    try:
        import whisper

        model = whisper.load_model(name)
        _loaded_models[name] = model
        _model_errors.pop(name, None)
        return model
    except Exception as exc:
        _model_errors[name] = f"{type(exc).__name__}: {exc}"
        logger.exception("OpenAI Whisper model '%s' failed to load", name)
        raise WhisperModelError(
            f"OpenAI Whisper model '{name}' failed to load: {type(exc).__name__}: {exc}"
        ) from exc
