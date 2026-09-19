"""
config/settings.py

Central configuration module for PDF AI Authorship Analyzer.
Loads values from .env if present, otherwise uses sensible defaults.
All settings are accessible as module-level constants and via the Settings dataclass.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists (silently skipped if absent)
load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "roberta-base-openai-detector"

MODEL_NAME: str = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME).strip()

# ---------------------------------------------------------------------------
# Text thresholds
# ---------------------------------------------------------------------------
MIN_WORD_COUNT: int = int(os.getenv("MIN_WORD_COUNT", "300"))
MAX_SEGMENT_TOKENS: int = int(os.getenv("MAX_SEGMENT_TOKENS", "512"))
SEGMENT_OVERLAP_TOKENS: int = 50  # token overlap between segments

# Minimum characters to consider a PDF as "having extractable text"
MIN_EXTRACTABLE_CHARS: int = 50

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------
TRANSFORMER_WEIGHT: float = float(os.getenv("TRANSFORMER_WEIGHT", "0.40"))
STYLOMETRIC_WEIGHT: float = float(os.getenv("STYLOMETRIC_WEIGHT", "0.20"))
STATISTICAL_WEIGHT: float = float(os.getenv("STATISTICAL_WEIGHT", "0.15"))
REPETITION_WEIGHT: float = float(os.getenv("REPETITION_WEIGHT", "0.10"))
SENTENCE_UNIFORMITY_WEIGHT: float = float(
    os.getenv("SENTENCE_UNIFORMITY_WEIGHT", "0.15")
)

# ---------------------------------------------------------------------------
# Score classification thresholds
# ---------------------------------------------------------------------------
SCORE_THRESHOLDS = {
    "high": 81,          # 81–100: High AI-authorship indicators
    "elevated": 61,      # 61–80: Elevated AI-authorship indicators
    "uncertain": 41,     # 41–60: Uncertain / mixed signals
    "relatively_low": 21, # 21–40: Relatively low AI-authorship indicators
    # 0–20: Low AI-authorship indicators
}

CLASSIFICATION_LABELS = {
    "high": "High AI-authorship indicators",
    "elevated": "Elevated AI-authorship indicators",
    "uncertain": "Uncertain / mixed signals",
    "relatively_low": "Relatively low AI-authorship indicators",
    "low": "Low AI-authorship indicators",
    "insufficient": "Insufficient text",
    "no_text": "No extractable text",
    "error": "Error",
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "2"))

# ---------------------------------------------------------------------------
# Dataclass for passing settings around
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    model_name: str = MODEL_NAME
    min_word_count: int = MIN_WORD_COUNT
    max_segment_tokens: int = MAX_SEGMENT_TOKENS
    transformer_weight: float = TRANSFORMER_WEIGHT
    stylometric_weight: float = STYLOMETRIC_WEIGHT
    statistical_weight: float = STATISTICAL_WEIGHT
    repetition_weight: float = REPETITION_WEIGHT
    sentence_uniformity_weight: float = SENTENCE_UNIFORMITY_WEIGHT
    max_workers: int = MAX_WORKERS

    def get_weights(self) -> dict:
        return {
            "transformer": self.transformer_weight,
            "stylometric": self.stylometric_weight,
            "statistical": self.statistical_weight,
            "repetition": self.repetition_weight,
            "sentence_uniformity": self.sentence_uniformity_weight,
        }

    def weights_sum(self) -> float:
        return sum(self.get_weights().values())

    def validate_weights(self) -> bool:
        """Check that weights sum to approximately 1.0."""
        total = self.weights_sum()
        if abs(total - 1.0) > 0.01:
            logger.warning(
                f"Scoring weights sum to {total:.3f}, not 1.0. "
                "Scores will be normalized automatically."
            )
            return False
        return True


def get_default_settings() -> Settings:
    """Return a Settings instance with current environment values."""
    return Settings()


def classify_score(score: float) -> str:
    """Map a numeric score 0–100 to a classification label key."""
    if score >= SCORE_THRESHOLDS["high"]:
        return "high"
    elif score >= SCORE_THRESHOLDS["elevated"]:
        return "elevated"
    elif score >= SCORE_THRESHOLDS["uncertain"]:
        return "uncertain"
    elif score >= SCORE_THRESHOLDS["relatively_low"]:
        return "relatively_low"
    else:
        return "low"
