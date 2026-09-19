"""
core/scoring.py

Hybrid scoring engine that combines multiple independent signals into
a final AI Authorship Risk Score (0–100).

Signals:
  1. Transformer classifier (40%)
  2. Stylometric analysis (20%)
  3. Statistical analysis (15%)
  4. Repetition analysis (10%)
  5. Sentence uniformity (15%)

IMPORTANT: These weights are a configurable baseline, not scientifically validated.
The score represents signal strength, NOT a probability of AI authorship.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.feature_extractor import TextFeatures
from core.ai_detector import TransformerDetectionResult
from config.settings import (
    CLASSIFICATION_LABELS,
    classify_score,
    MIN_WORD_COUNT,
)
from utils.helpers import clamp, safe_divide

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal scorers
# ---------------------------------------------------------------------------

def _score_transformer(transformer: TransformerDetectionResult) -> float:
    """
    Convert transformer result to a 0–100 signal.
    Uses weighted combination of mean and pct_flagged.
    Returns 0.0 if transformer is unavailable (handled in caller).
    """
    if not transformer.available:
        return 0.0

    # Weighted blend: mean score has more weight than pct_flagged
    mean_component = transformer.mean_score            # already 0–100
    flagged_component = transformer.pct_segments_flagged  # 0–100

    score = 0.7 * mean_component + 0.3 * flagged_component
    return clamp(score)


def _score_stylometric(features: TextFeatures) -> float:
    """
    Compute a stylometric signal from 0–100.

    AI-generated text tends to:
    - Have lower Yule's K (more uniform vocabulary usage)
    - Have lower character n-gram entropy (more predictable character patterns)
    - Have higher function word ratio (tendency toward formal structure)

    This is a heuristic; individual signals are weak.
    """
    if features.word_count < 50:
        return 0.0

    score_components = []

    # Yule's K: AI tends to produce text with lower K (more repetitive vocab)
    # Typical human text: K ~ 50–200; very uniform text: K < 30
    # Scale: K=0 → 90, K=100 → 40, K=300+ → 10
    if features.vocabulary_richness > 0:
        k = features.vocabulary_richness
        k_signal = clamp(90.0 - k * 0.3, 0.0, 90.0)
    else:
        k_signal = 50.0  # unknown
    score_components.append(k_signal)

    # Char n-gram entropy: lower entropy = more predictable = more AI-like
    # Typical range: 3–5 bits per trigram; lower = more uniform
    char_ent = features.char_ngram_entropy
    # Scale: entropy=2 → 80, entropy=4 → 40, entropy=6 → 10
    ent_signal = clamp(90.0 - char_ent * 15.0, 5.0, 90.0)
    score_components.append(ent_signal)

    # Type-token ratio: AI tends to have moderate TTR (not too high, not too low)
    # Very high TTR (>0.8) is unusual in long texts; low TTR (<0.3) is repetitive
    ttr = features.type_token_ratio
    # AI tends to cluster around 0.4–0.6 for medium documents
    # We reward deviation from human patterns (very high or very low TTR)
    ttr_distance_from_ai_zone = abs(ttr - 0.50)
    ttr_signal = clamp(70.0 - ttr_distance_from_ai_zone * 120.0, 0.0, 80.0)
    score_components.append(ttr_signal)

    return clamp(float(np.mean(score_components)))


def _score_statistical(features: TextFeatures) -> float:
    """
    Statistical signal based on sentence length distribution shape.

    AI-generated text often has:
    - Narrow sentence length distribution (concentrated around mean)
    - Higher kurtosis (more peaky distribution)
    - Consistent paragraph structure
    """
    if not features.sentence_lengths or len(features.sentence_lengths) < 5:
        return 0.0

    arr = np.array(features.sentence_lengths, dtype=float)

    score_components = []

    # Std dev signal: very low std suggests uniform AI-like writing
    # Typical human: std 7–15; AI: std 3–8
    std = features.std_sentence_length
    std_signal = clamp(80.0 - std * 4.0, 0.0, 85.0)
    score_components.append(std_signal)

    # Skewness: human text is often right-skewed (occasional long sentences)
    # AI text tends toward symmetry
    if len(arr) >= 8:
        try:
            from scipy.stats import kurtosis, skew
            kurt = kurtosis(arr)
            sk = abs(skew(arr))
            # High kurtosis (peaky) → more AI-like
            kurt_signal = clamp(50.0 + kurt * 5.0, 0.0, 90.0)
            # Low skewness → more symmetric → more AI-like
            skew_signal = clamp(70.0 - sk * 15.0, 0.0, 80.0)
            score_components.extend([kurt_signal, skew_signal])
        except Exception:
            pass

    # Paragraph uniformity: AI tends to write consistently-sized paragraphs
    para_uni_signal = features.paragraph_length_uniformity * 80.0
    score_components.append(para_uni_signal)

    return clamp(float(np.mean(score_components))) if score_components else 0.0


def _score_repetition(features: TextFeatures) -> float:
    """
    Repetition signal based on n-gram repetition ratios.

    AI text can have elevated phrase repetition — especially at the trigram level.
    """
    if features.word_count < 50:
        return 0.0

    bigram_signal = clamp(features.bigram_repetition_ratio * 150.0, 0.0, 90.0)
    trigram_signal = clamp(features.trigram_repetition_ratio * 200.0, 0.0, 90.0)

    # Repeated word ratio (AI tends toward moderate repetition)
    repeated_word_signal = clamp(features.repeated_word_ratio * 100.0, 0.0, 80.0)

    score = float(np.mean([bigram_signal, trigram_signal, repeated_word_signal]))
    return clamp(score)


def _score_sentence_uniformity(features: TextFeatures) -> float:
    """
    Sentence uniformity signal.

    High uniformity → AI-like.
    This is one of the clearest signals of AI text — sentences of similar length.
    """
    if not features.sentence_lengths or len(features.sentence_lengths) < 3:
        return 0.0

    # uniformity is already [0, 1]; scale to [0, 100]
    uniformity_signal = features.sentence_length_uniformity * 90.0

    # Add the avg sentence length component:
    # AI tends to write sentences of 18–28 words consistently
    avg_len = features.avg_sentence_length
    in_ai_range = 15.0 <= avg_len <= 30.0
    length_signal = 60.0 if in_ai_range else 30.0

    return clamp(float(np.mean([uniformity_signal, length_signal])))


# ---------------------------------------------------------------------------
# Confidence calculation
# ---------------------------------------------------------------------------

def _calculate_confidence(
    word_count: int,
    transformer_available: bool,
    signal_scores: dict[str, float],
    transformer_std: float,
) -> str:
    """
    Determine confidence level (High / Medium / Low).

    Factors:
    - Document word count (more text → more reliable)
    - Whether transformer model was available
    - Agreement between signals (low std → high agreement)
    - Transformer's own segment variation
    """
    # Start with a base score
    confidence_points = 0

    # Word count contribution
    if word_count >= 1000:
        confidence_points += 3
    elif word_count >= 500:
        confidence_points += 2
    elif word_count >= 300:
        confidence_points += 1

    # Transformer availability
    if transformer_available:
        confidence_points += 2

    # Signal agreement
    available_scores = [v for v in signal_scores.values() if v is not None]
    if len(available_scores) >= 3:
        signal_std = float(np.std(available_scores))
        if signal_std < 10:
            confidence_points += 2  # high agreement
        elif signal_std < 25:
            confidence_points += 1
        # else: low agreement → no points

    # Transformer segment variation (low std → stable prediction)
    if transformer_available and transformer_std < 15:
        confidence_points += 1

    # Map points to confidence level
    if confidence_points >= 6:
        return "High"
    elif confidence_points >= 3:
        return "Medium"
    else:
        return "Low"


# ---------------------------------------------------------------------------
# Detected patterns (human-readable)
# ---------------------------------------------------------------------------

def _detect_patterns(
    features: TextFeatures,
    signals: dict[str, float],
    transformer: TransformerDetectionResult,
) -> list[str]:
    """
    Generate a list of human-readable detected pattern descriptions.
    Uses cautious language — never claims definitive AI authorship.
    """
    patterns = []

    if signals.get("sentence_uniformity", 0) >= 60:
        patterns.append("Elevated sentence-length uniformity detected")

    if transformer.available and transformer.mean_score >= 60:
        patterns.append("Transformer classifier indicates elevated AI-authorship signals")

    if features.trigram_repetition_ratio >= 0.4:
        patterns.append("Above-average trigram phrase repetition observed")

    if features.bigram_repetition_ratio >= 0.45:
        patterns.append("Elevated bigram repetition ratio detected")

    if features.sentence_length_uniformity >= 0.75:
        patterns.append("Highly consistent sentence lengths across document")

    if features.vocabulary_richness < 30 and features.word_count > 200:
        patterns.append("Lower-than-typical lexical richness (Yule's K)")

    if features.paragraph_length_uniformity >= 0.80:
        patterns.append("Consistent paragraph structure and length")

    if features.std_sentence_length < 5 and features.sentence_count > 5:
        patterns.append("Low variance in sentence length distribution")

    if features.function_word_ratio > 0.55:
        patterns.append("High function-word ratio observed")

    if signals.get("statistical", 0) >= 65:
        patterns.append("Statistical analysis indicates sentence distribution consistent with AI text patterns")

    if not patterns:
        patterns.append("No strongly elevated AI-authorship indicators detected")

    return patterns


# ---------------------------------------------------------------------------
# Main scoring dataclass and engine
# ---------------------------------------------------------------------------

@dataclass
class ScoringResult:
    """Final scoring output for a single document."""

    filename: str = ""

    # Final score and classification
    ai_risk_score: Optional[float] = None   # None if insufficient text
    classification: str = ""
    confidence: str = ""

    # Individual signals (0–100 each)
    transformer_signal: Optional[float] = None
    stylometric_signal: float = 0.0
    statistical_signal: float = 0.0
    repetition_signal: float = 0.0
    sentence_uniformity_signal: float = 0.0

    # Detected patterns
    detected_patterns: list[str] = field(default_factory=list)

    # Metadata
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    vocabulary_size: int = 0
    avg_sentence_length: float = 0.0
    std_sentence_length: float = 0.0
    type_token_ratio: float = 0.0
    transformer_segment_count: int = 0
    transformer_pct_flagged: float = 0.0

    status: str = "completed"
    error: Optional[str] = None


class ScoringEngine:
    """
    Combines multiple detection signals into a final AI Authorship Risk Score.

    Weights are configurable. They are normalized if they don't sum to 1.0.
    """

    def __init__(self, weights: Optional[dict[str, float]] = None):
        """
        Args:
            weights: Dict with keys:
                transformer, stylometric, statistical, repetition, sentence_uniformity
        """
        default_weights = {
            "transformer": 0.40,
            "stylometric": 0.20,
            "statistical": 0.15,
            "repetition": 0.10,
            "sentence_uniformity": 0.15,
        }
        self.weights = weights if weights else default_weights
        self._normalize_weights()

    def _normalize_weights(self):
        """Ensure weights sum to 1.0."""
        total = sum(self.weights.values())
        if total > 0 and abs(total - 1.0) > 0.001:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def score(
        self,
        filename: str,
        features: TextFeatures,
        transformer: TransformerDetectionResult,
        min_word_count: int = MIN_WORD_COUNT,
    ) -> ScoringResult:
        """
        Generate the full scoring result for a document.

        Args:
            filename: PDF filename for identification.
            features: Extracted linguistic features.
            transformer: Transformer detection result.
            min_word_count: Minimum words required for scoring.

        Returns:
            ScoringResult with all signals and final score.
        """
        result = ScoringResult(filename=filename)

        # Populate metadata from features
        result.word_count = features.word_count
        result.sentence_count = features.sentence_count
        result.paragraph_count = features.paragraph_count
        result.vocabulary_size = features.vocabulary_size
        result.avg_sentence_length = features.avg_sentence_length
        result.std_sentence_length = features.std_sentence_length
        result.type_token_ratio = features.type_token_ratio

        if transformer.available:
            result.transformer_segment_count = transformer.segment_count
            result.transformer_pct_flagged = transformer.pct_segments_flagged

        # Check minimum text requirement
        if features.word_count < min_word_count:
            result.ai_risk_score = None
            result.classification = CLASSIFICATION_LABELS["insufficient"]
            result.confidence = "Low"
            result.detected_patterns = [
                f"Document has only {features.word_count} words "
                f"(minimum {min_word_count} required for reliable analysis)."
            ]
            result.status = "insufficient_text"
            return result

        # Compute individual signals
        transformer_signal = _score_transformer(transformer)
        stylometric_signal = _score_stylometric(features)
        statistical_signal = _score_statistical(features)
        repetition_signal = _score_repetition(features)
        uniformity_signal = _score_sentence_uniformity(features)

        result.stylometric_signal = round(stylometric_signal, 1)
        result.statistical_signal = round(statistical_signal, 1)
        result.repetition_signal = round(repetition_signal, 1)
        result.sentence_uniformity_signal = round(uniformity_signal, 1)

        # Compute weighted final score
        if transformer.available:
            result.transformer_signal = round(transformer_signal, 1)
            total_score = (
                transformer_signal * self.weights["transformer"]
                + stylometric_signal * self.weights["stylometric"]
                + statistical_signal * self.weights["statistical"]
                + repetition_signal * self.weights["repetition"]
                + uniformity_signal * self.weights["sentence_uniformity"]
            )
        else:
            # Redistribute transformer weight proportionally to other signals
            result.transformer_signal = None
            other_total = (
                self.weights["stylometric"]
                + self.weights["statistical"]
                + self.weights["repetition"]
                + self.weights["sentence_uniformity"]
            )
            if other_total > 0:
                w_sty = self.weights["stylometric"] / other_total
                w_sta = self.weights["statistical"] / other_total
                w_rep = self.weights["repetition"] / other_total
                w_uni = self.weights["sentence_uniformity"] / other_total
            else:
                w_sty = w_sta = w_rep = w_uni = 0.25

            total_score = (
                stylometric_signal * w_sty
                + statistical_signal * w_sta
                + repetition_signal * w_rep
                + uniformity_signal * w_uni
            )

        result.ai_risk_score = round(clamp(total_score), 1)

        # Classification
        label_key = classify_score(result.ai_risk_score)
        result.classification = CLASSIFICATION_LABELS[label_key]

        # Confidence
        signal_scores = {
            "stylometric": stylometric_signal,
            "statistical": statistical_signal,
            "repetition": repetition_signal,
            "sentence_uniformity": uniformity_signal,
        }
        if transformer.available:
            signal_scores["transformer"] = transformer_signal

        result.confidence = _calculate_confidence(
            word_count=features.word_count,
            transformer_available=transformer.available,
            signal_scores=signal_scores,
            transformer_std=transformer.std_score if transformer.available else 0.0,
        )

        # Detected patterns
        result.detected_patterns = _detect_patterns(features, signal_scores, transformer)

        result.status = "completed"

        logger.info(
            f"[{filename}] Score={result.ai_risk_score:.0f} | "
            f"{result.classification} | Confidence={result.confidence}"
        )

        return result
