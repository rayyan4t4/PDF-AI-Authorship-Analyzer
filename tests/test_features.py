"""
tests/test_features.py

Unit tests for core/feature_extractor.py
"""

import pytest
from core.text_preprocessor import preprocess
from core.feature_extractor import extract_features, TextFeatures


# ---------------------------------------------------------------------------
# Synthetic test texts
# ---------------------------------------------------------------------------

# Deliberately uniform: all sentences ~15 words (AI-like)
UNIFORM_TEXT = " ".join([
    "The results of this study indicate a significant improvement in performance metrics. " * 20
])

# Deliberately varied: mix of short and long sentences (human-like)
VARIED_TEXT = """
This works.
The comprehensive analysis of the methodology reveals that the approach taken by researchers
was fundamentally sound and yielded results consistent with prior literature.
OK.
Furthermore, the implications of these findings extend across multiple domains and suggest
that practitioners should consider revising their current frameworks to accommodate these insights.
Yes.
This is a short one.
The extended theoretical framework proposed here challenges conventional assumptions and opens
new avenues for empirical investigation in the field.
"""

SHORT_TEXT = "Hello world."

RICH_VOCAB_TEXT = " ".join([
    "An extraordinary amalgamation of peculiar and esoteric nomenclature demonstrates "
    "remarkable vocabularic breadth, encompassing multitudinous lexical constructs. " * 5
])


class TestFeatureExtractor:

    def test_returns_text_features(self):
        preprocessed = preprocess(UNIFORM_TEXT)
        features = extract_features(preprocessed)
        assert isinstance(features, TextFeatures)

    def test_word_count_positive(self):
        preprocessed = preprocess(UNIFORM_TEXT)
        features = extract_features(preprocessed)
        assert features.word_count > 0

    def test_sentence_count_positive(self):
        preprocessed = preprocess(UNIFORM_TEXT)
        features = extract_features(preprocessed)
        assert features.sentence_count > 0

    def test_avg_sentence_length_positive(self):
        preprocessed = preprocess(UNIFORM_TEXT)
        features = extract_features(preprocessed)
        assert features.avg_sentence_length > 0

    def test_type_token_ratio_in_range(self):
        """TTR must be in [0, 1]."""
        preprocessed = preprocess(UNIFORM_TEXT)
        features = extract_features(preprocessed)
        assert 0.0 <= features.type_token_ratio <= 1.0

    def test_sentence_uniformity_in_range(self):
        """Uniformity must be in [0, 1]."""
        preprocessed = preprocess(UNIFORM_TEXT)
        features = extract_features(preprocessed)
        assert 0.0 <= features.sentence_length_uniformity <= 1.0

    def test_uniform_text_has_high_uniformity(self):
        """Highly uniform text should score higher uniformity than varied text."""
        uni_features = extract_features(preprocess(UNIFORM_TEXT))
        var_features = extract_features(preprocess(VARIED_TEXT))
        assert uni_features.sentence_length_uniformity > var_features.sentence_length_uniformity

    def test_punctuation_frequencies_non_negative(self):
        preprocessed = preprocess(UNIFORM_TEXT)
        features = extract_features(preprocessed)
        assert features.comma_freq >= 0
        assert features.period_freq >= 0
        assert features.semicolon_freq >= 0

    def test_vocabulary_size_positive(self):
        preprocessed = preprocess(UNIFORM_TEXT)
        features = extract_features(preprocessed)
        assert features.vocabulary_size > 0

    def test_bigram_repetition_uniform_text(self):
        """Repetitive text should have non-zero bigram repetition."""
        preprocessed = preprocess(UNIFORM_TEXT)
        features = extract_features(preprocessed)
        assert features.bigram_repetition_ratio > 0

    def test_short_text_handles_gracefully(self):
        """Short text should not raise errors."""
        preprocessed = preprocess(SHORT_TEXT)
        features = extract_features(preprocessed)
        assert features.word_count >= 0

    def test_char_ngram_entropy_positive(self):
        preprocessed = preprocess(RICH_VOCAB_TEXT)
        features = extract_features(preprocessed)
        assert features.char_ngram_entropy > 0

    def test_function_word_ratio_in_range(self):
        preprocessed = preprocess(UNIFORM_TEXT)
        features = extract_features(preprocessed)
        assert 0.0 <= features.function_word_ratio <= 1.0

    def test_repeated_word_ratio_in_range(self):
        preprocessed = preprocess(UNIFORM_TEXT)
        features = extract_features(preprocessed)
        assert 0.0 <= features.repeated_word_ratio <= 1.0

    def test_yules_k_non_negative(self):
        preprocessed = preprocess(UNIFORM_TEXT)
        features = extract_features(preprocessed)
        assert features.vocabulary_richness >= 0
