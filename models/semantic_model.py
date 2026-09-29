"""Lazy, cached Sentence-Transformers model loader."""

from __future__ import annotations

import logging
from functools import lru_cache

from config.settings import settings

logger = logging.getLogger(__name__)

try:
    import streamlit as st

    cache_model = st.cache_resource(show_spinner=False)
except ImportError:
    cache_model = lru_cache(maxsize=2)


@cache_model
def load_semantic_model(model_name: str | None = None):
    name = model_name or settings.semantic_model
    logger.info("Loading Sentence-Transformer model '%s'", name)
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(name)
    except Exception as exc:
        logger.exception("Sentence-Transformer model loading failed")
        raise RuntimeError(
            "Semantic analysis could not be started. Install the dependencies and verify model access."
        ) from exc