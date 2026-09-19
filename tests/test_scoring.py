"""
tests/test_scoring.py

Unit tests for core/scoring.py

NOTE: These tests validate that the scoring machinery works correctly —
not that the scores are scientifically accurate AI-authorship predictions.
No labeled ground-truth is available for scientific validation.
"""

import pytest
from core.text_preprocessor import preprocess
from core.feature_extractor import extract_features
from core.ai_detector import TransformerDetectionResult
from core.scoring import ScoringEngine, ScoringResult
from config.settings import CLASSIFICATION_LABELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_no_transformer() -> TransformerDetectionResult:
    """Simulate a missing/unavailable transformer model."""
    return TransformerDetectionResult(available=False, error="No model configured.")


def _make_transformer(mean: float, pct_flagged: float = 0.0, std: float = 5.0) -> TransformerDetectionResult:
    """Simulate a transformer result."""
    return TransformerDetectionResult(
        available=True,
        mean_score=mean,
        median_score=mean,
        std_score=std,
        min_score=max(0, mean - 10),
        max_score=min(100, mean + 10),
        pct_segments_flagged=pct_flagged,
        segment_count=5,
    )


MEDIUM_TEXT = (
    "This research paper investigates the relationship between machine learning "
    "and natural language processing. The methodology employs a variety of techniques "
    "including transformer models and statistical analysis. Results indicate significant "
    "improvements in detection accuracy. The findings suggest that hybrid approaches "
    "outperform single-method baselines. Future work will explore multilingual settings. " * 10
)

SHORT_TEXT = "Hello."


class TestScoringEngine:

    def test_returns_scoring_result(self):
        engine = ScoringEngine()
        preprocessed = preprocess(MEDIUM_TEXT)
        features = extract_features(preprocessed)
        result = engine.score("test.pdf", features, _make_no_transformer())
        assert isinstance(result, ScoringResult)

    def test_score_in_range_0_100(self):
        engine = ScoringEngine()
        preprocessed = preprocess(MEDIUM_TEXT)
        features = extract_features(preprocessed)
        result = engine.score("test.pdf", features, _make_no_transformer())
        if result.ai_risk_score is not None:
            assert 0 <= result.ai_risk_score <= 100

    def test_short_text_returns_insufficient(self):
        """Text with fewer than min_word_count words should return insufficient status."""
        engine = ScoringEngine()
        preprocessed = preprocess(SHORT_TEXT)
        features = extract_features(preprocessed)
        result = engine.score("short.pdf", features, _make_no_transformer(), min_word_count=300)
        assert result.ai_risk_score is None
        assert result.status == "insufficient_text"
        assert "insufficient" in result.classification.lower() or "insufficient" in result.status

    def test_confidence_is_valid_string(self):
        engine = ScoringEngine()
        preprocessed = preprocess(MEDIUM_TEXT)
        features = extract_features(preprocessed)
        result = engine.score("test.pdf", features, _make_no_transformer())
        assert result.confidence in ("High", "Medium", "Low")

    def test_classification_string_non_empty(self):
        engine = ScoringEngine()
        preprocessed = preprocess(MEDIUM_TEXT)
        features = extract_features(preprocessed)
        result = engine.score("test.pdf", features, _make_no_transformer())
        assert isinstance(result.classification, str)
        assert len(result.classification) > 0

    def test_transformer_signal_included_when_available(self):
        engine = ScoringEngine()
        preprocessed = preprocess(MEDIUM_TEXT)
        features = extract_features(preprocessed)
        transformer = _make_transformer(mean=80, pct_flagged=75)
        result = engine.score("test.pdf", features, transformer)
        assert result.transformer_signal is not None
        assert 0 <= result.transformer_signal <= 100

    def test_transformer_signal_none_when_unavailable(self):
        engine = ScoringEngine()
        preprocessed = preprocess(MEDIUM_TEXT)
        features = extract_features(preprocessed)
        result = engine.score("test.pdf", features, _make_no_transformer())
        assert result.transformer_signal is None

    def test_weights_normalized(self):
        """Engine with non-summing weights should still produce a valid score."""
        bad_weights = {
            "transformer": 2.0,
            "stylometric": 2.0,
            "statistical": 2.0,
            "repetition": 2.0,
            "sentence_uniformity": 2.0,
        }
        engine = ScoringEngine(weights=bad_weights)
        # Weights should have been normalized
        total = sum(engine.weights.values())
        assert abs(total - 1.0) < 0.01

    def test_detected_patterns_non_empty(self):
        engine = ScoringEngine()
        preprocessed = preprocess(MEDIUM_TEXT)
        features = extract_features(preprocessed)
        result = engine.score("test.pdf", features, _make_no_transformer())
        assert isinstance(result.detected_patterns, list)
        assert len(result.detected_patterns) > 0

    def test_filename_preserved(self):
        engine = ScoringEngine()
        preprocessed = preprocess(MEDIUM_TEXT)
        features = extract_features(preprocessed)
        result = engine.score("my_file.pdf", features, _make_no_transformer())
        assert result.filename == "my_file.pdf"

    def test_score_normalization(self):
        """Score should be clamped to 0–100."""
        engine = ScoringEngine()
        preprocessed = preprocess(MEDIUM_TEXT)
        features = extract_features(preprocessed)
        # Inject very high transformer signal
        transformer = _make_transformer(mean=100.0, pct_flagged=100.0)
        result = engine.score("high.pdf", features, transformer)
        if result.ai_risk_score is not None:
            assert result.ai_risk_score <= 100.0
            assert result.ai_risk_score >= 0.0

    def test_no_fake_random_scores(self):
        """Running twice on the same input should give the same score (deterministic)."""
        engine = ScoringEngine()
        preprocessed = preprocess(MEDIUM_TEXT)
        features = extract_features(preprocessed)
        r1 = engine.score("doc.pdf", features, _make_no_transformer())
        r2 = engine.score("doc.pdf", features, _make_no_transformer())
        assert r1.ai_risk_score == r2.ai_risk_score
