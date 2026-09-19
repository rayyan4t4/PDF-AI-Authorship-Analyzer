"""
core/feature_extractor.py

Extracts linguistic and stylometric features from preprocessed text.
These features form independent signals for the hybrid AI detection system.
No single feature proves AI authorship; they are used collectively.
"""

import logging
import re
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats

from core.text_preprocessor import PreprocessedText
from utils.helpers import safe_divide

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Common English function words
# ---------------------------------------------------------------------------
FUNCTION_WORDS = frozenset([
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "can", "could", "must", "that",
    "this", "these", "those", "it", "its", "i", "you", "he", "she", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "our",
    "their", "not", "no", "nor", "so", "yet", "both", "either", "neither",
    "each", "every", "any", "all", "as", "if", "when", "while", "though",
    "because", "since", "until", "than", "then", "there", "here", "now",
    "just", "also", "even", "more", "most", "very", "too", "about", "up",
    "out", "what", "which", "who", "how", "into", "through", "during",
    "after", "before", "between", "such", "other", "than",
])


@dataclass
class TextFeatures:
    """All extracted features for a single document."""

    # Basic statistics
    word_count: int = 0
    char_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    avg_sentence_length: float = 0.0
    median_sentence_length: float = 0.0
    std_sentence_length: float = 0.0
    sentence_length_variance: float = 0.0
    avg_word_length: float = 0.0

    # Lexical features
    vocabulary_size: int = 0
    type_token_ratio: float = 0.0
    vocabulary_richness: float = 0.0     # Yule's K (lexical richness measure)
    repeated_word_ratio: float = 0.0
    unique_word_ratio: float = 0.0
    function_word_ratio: float = 0.0

    # N-gram repetition
    bigram_repetition_ratio: float = 0.0
    trigram_repetition_ratio: float = 0.0

    # Punctuation frequencies (per 100 words)
    comma_freq: float = 0.0
    period_freq: float = 0.0
    semicolon_freq: float = 0.0
    colon_freq: float = 0.0
    parenthesis_freq: float = 0.0
    quote_freq: float = 0.0
    exclamation_freq: float = 0.0
    question_freq: float = 0.0
    dash_freq: float = 0.0

    # Sentence uniformity
    sentence_length_uniformity: float = 0.0   # 0=varied, 1=uniform
    sentence_variation_coefficient: float = 0.0  # CV of sentence lengths

    # Paragraph features
    avg_paragraph_length: float = 0.0
    paragraph_length_uniformity: float = 0.0

    # Stylometric signals (aggregated)
    char_ngram_entropy: float = 0.0
    word_ngram_entropy: float = 0.0

    # Raw lists for internal use (not exported to Excel directly)
    sentence_lengths: list[int] = field(default_factory=list)
    paragraph_lengths: list[int] = field(default_factory=list)


def _sentence_lengths(sentences: list[str]) -> list[int]:
    """Return list of word counts per sentence."""
    return [len(s.split()) for s in sentences if s.strip()]


def _paragraph_word_lengths(paragraphs: list[str]) -> list[int]:
    """Return list of word counts per paragraph."""
    return [len(p.split()) for p in paragraphs if p.strip()]


def _ngram_repetition_ratio(tokens: list[str], n: int) -> float:
    """
    Fraction of n-grams that appear more than once.
    High ratio → repetitive text (AI signal).
    """
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(counts)


def _yules_k(tokens: list[str]) -> float:
    """
    Yule's K measure of lexical richness.
    Lower K → richer vocabulary. Higher K → more repetitive.
    Returns value in [0, ~1000] range; normalized later.
    """
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    n = len(tokens)
    m2 = sum(v * v for v in freq.values())
    if n == 0:
        return 0.0
    k = 10_000 * (m2 - n) / (n * n) if n > 1 else 0.0
    return max(0.0, k)


def _entropy_of_ngrams(tokens: list[str], n: int) -> float:
    """Shannon entropy of n-gram distribution (higher = more varied)."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    counts = Counter(ngrams)
    total = len(ngrams)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def _char_ngram_entropy(text: str, n: int = 3) -> float:
    """Character-level n-gram entropy."""
    text = text.lower()
    ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    total = len(ngrams)
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def _uniformity_score(lengths: list[int]) -> tuple[float, float]:
    """
    Compute uniformity and coefficient of variation for a list of lengths.

    uniformity: 1 - (std / mean), clamped to [0, 1].
    cv: std / mean (coefficient of variation).
    """
    if not lengths or len(lengths) < 2:
        return 0.0, 0.0
    arr = np.array(lengths, dtype=float)
    mean = arr.mean()
    std = arr.std()
    if mean == 0:
        return 0.0, 0.0
    cv = std / mean
    uniformity = max(0.0, min(1.0, 1.0 - cv))
    return uniformity, cv


def _punct_freq(text: str, word_count: int) -> dict[str, float]:
    """Count punctuation marks, normalized per 100 words."""
    scale = safe_divide(100.0, word_count, 0.0)
    return {
        "comma":       text.count(",") * scale,
        "period":      text.count(".") * scale,
        "semicolon":   text.count(";") * scale,
        "colon":       text.count(":") * scale,
        "parenthesis": (text.count("(") + text.count(")")) * scale,
        "quote":       (text.count('"') + text.count("'")) * scale,
        "exclamation": text.count("!") * scale,
        "question":    text.count("?") * scale,
        "dash":        (text.count("-") + text.count("—")) * scale,
    }


def extract_features(preprocessed: PreprocessedText) -> TextFeatures:
    """
    Extract all linguistic and stylometric features from preprocessed text.

    Args:
        preprocessed: Output of text_preprocessor.preprocess().

    Returns:
        TextFeatures populated with all computed features.
    """
    f = TextFeatures()

    tokens = preprocessed.tokens          # lowercase words only
    sentences = preprocessed.sentences
    paragraphs = preprocessed.paragraphs
    text = preprocessed.processed_text

    word_count = len(tokens)
    char_count = len(text)

    f.word_count = word_count
    f.char_count = char_count
    f.sentence_count = len(sentences)
    f.paragraph_count = len(paragraphs)

    # ------------------------------------------------------------------
    # Sentence length statistics
    # ------------------------------------------------------------------
    sent_lens = _sentence_lengths(sentences)
    f.sentence_lengths = sent_lens

    if sent_lens:
        arr = np.array(sent_lens, dtype=float)
        f.avg_sentence_length = float(arr.mean())
        f.median_sentence_length = float(np.median(arr))
        f.std_sentence_length = float(arr.std()) if len(arr) > 1 else 0.0
        f.sentence_length_variance = float(arr.var()) if len(arr) > 1 else 0.0
        f.sentence_length_uniformity, f.sentence_variation_coefficient = (
            _uniformity_score(sent_lens)
        )

    # ------------------------------------------------------------------
    # Paragraph length statistics
    # ------------------------------------------------------------------
    para_lens = _paragraph_word_lengths(paragraphs)
    f.paragraph_lengths = para_lens

    if para_lens:
        f.avg_paragraph_length = float(np.mean(para_lens))
        f.paragraph_length_uniformity, _ = _uniformity_score(para_lens)

    # ------------------------------------------------------------------
    # Average word length
    # ------------------------------------------------------------------
    if word_count > 0:
        f.avg_word_length = safe_divide(
            sum(len(w) for w in tokens), word_count
        )

    # ------------------------------------------------------------------
    # Lexical features
    # ------------------------------------------------------------------
    word_counter = Counter(tokens)
    f.vocabulary_size = len(word_counter)
    f.type_token_ratio = safe_divide(f.vocabulary_size, word_count)
    f.unique_word_ratio = safe_divide(
        sum(1 for c in word_counter.values() if c == 1), word_count
    )
    f.repeated_word_ratio = safe_divide(
        sum(1 for c in word_counter.values() if c > 1), word_count
    )
    f.function_word_ratio = safe_divide(
        sum(1 for t in tokens if t in FUNCTION_WORDS), word_count
    )
    f.vocabulary_richness = _yules_k(tokens)

    # ------------------------------------------------------------------
    # N-gram repetition
    # ------------------------------------------------------------------
    f.bigram_repetition_ratio = _ngram_repetition_ratio(tokens, 2)
    f.trigram_repetition_ratio = _ngram_repetition_ratio(tokens, 3)

    # ------------------------------------------------------------------
    # Punctuation frequencies
    # ------------------------------------------------------------------
    punct = _punct_freq(text, word_count)
    f.comma_freq = punct["comma"]
    f.period_freq = punct["period"]
    f.semicolon_freq = punct["semicolon"]
    f.colon_freq = punct["colon"]
    f.parenthesis_freq = punct["parenthesis"]
    f.quote_freq = punct["quote"]
    f.exclamation_freq = punct["exclamation"]
    f.question_freq = punct["question"]
    f.dash_freq = punct["dash"]

    # ------------------------------------------------------------------
    # Stylometric entropy
    # ------------------------------------------------------------------
    f.char_ngram_entropy = _char_ngram_entropy(text, n=3)
    f.word_ngram_entropy = _entropy_of_ngrams(tokens, n=2)

    logger.debug(
        f"Features extracted: vocab={f.vocabulary_size}, "
        f"TTR={f.type_token_ratio:.3f}, "
        f"sent_uniformity={f.sentence_length_uniformity:.3f}"
    )

    return f
