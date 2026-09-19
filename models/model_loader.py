"""
models/model_loader.py

Loads and caches the Hugging Face transformer model for AI text detection.
Uses @st.cache_resource to avoid reloading on every Streamlit rerun.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def load_detector_model(model_name: str) -> Optional[tuple]:
    """
    Load a Hugging Face sequence classification model and tokenizer.

    Args:
        model_name: Hugging Face model ID (e.g. 'roberta-base-openai-detector').

    Returns:
        (tokenizer, model) tuple, or (None, None) if loading fails.
    """
    if not model_name or model_name.strip() == "":
        logger.info("No MODEL_NAME configured. Transformer signal will be unavailable.")
        return None, None

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        logger.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info(f"Loading model: {model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        logger.info(f"Model loaded on {device}: {model_name}")
        return tokenizer, model

    except ImportError:
        logger.error("transformers or torch not installed.")
        return None, None
    except OSError as e:
        logger.error(f"Failed to load model '{model_name}': {e}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error loading model '{model_name}': {e}")
        return None, None
