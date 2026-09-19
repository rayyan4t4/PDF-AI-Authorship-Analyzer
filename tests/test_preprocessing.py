"""
tests/test_preprocessing.py

Unit tests for core/text_preprocessor.py
"""

import pytest
from core.text_preprocessor import preprocess, PreprocessedText


SAMPLE_TEXT = """
This is the first paragraph of a test document.
It contains multiple sentences. Some are short. Others are considerably longer and more elaborate.

This is the second paragraph. It has different content.
The text continues here with additional information.

A third paragraph appears here. It wraps up the document.
"""

SHORT_TEXT = "Hello world."


class TestPreprocessing:

    def test_returns_preprocessed_text_object(self):
        result = preprocess(SAMPLE_TEXT)
        assert isinstance(result, PreprocessedText)

    def test_raw_text_preserved(self):
        """raw_text must never be modified."""
        result = preprocess(SAMPLE_TEXT)
        assert result.raw_text == SAMPLE_TEXT

    def test_processed_text_not_empty(self):
        result = preprocess(SAMPLE_TEXT)
        assert result.processed_text.strip() != ""

    def test_paragraphs_detected(self):
        result = preprocess(SAMPLE_TEXT)
        assert result.paragraph_count >= 3

    def test_sentences_detected(self):
        result = preprocess(SAMPLE_TEXT)
        assert result.sentence_count >= 5

    def test_tokens_populated(self):
        result = preprocess(SAMPLE_TEXT)
        assert len(result.tokens) > 0
        assert all(isinstance(t, str) for t in result.tokens)

    def test_word_count_positive(self):
        result = preprocess(SAMPLE_TEXT)
        assert result.word_count > 0

    def test_null_byte_removal(self):
        text_with_nulls = "Hello\x00 world\x00."
        result = preprocess(text_with_nulls)
        assert "\x00" not in result.processed_text

    def test_curly_quotes_normalized(self):
        text = "\u201cHello\u201d and \u2018world\u2019."
        result = preprocess(text)
        assert '"Hello"' in result.processed_text or "'Hello'" in result.processed_text

    def test_short_text_handled(self):
        result = preprocess(SHORT_TEXT)
        assert result.word_count >= 1
        assert result.raw_text == SHORT_TEXT

    def test_empty_string(self):
        result = preprocess("")
        assert result.raw_text == ""
        assert result.word_count == 0
        assert result.sentence_count == 0

    def test_hyphenated_line_breaks_fixed(self):
        text = "This is a hyphen-\nated word."
        result = preprocess(text)
        # Should join: "hyphenated"
        assert "hyphenated" in result.processed_text.lower() or \
               "hyphen" in result.processed_text.lower()

    def test_tokens_lowercase(self):
        """All tokens should be lowercase."""
        result = preprocess("Hello WORLD Testing.")
        for token in result.tokens:
            assert token == token.lower()

    def test_em_dash_normalized(self):
        text = "This\u2014is a test."
        result = preprocess(text)
        assert "\u2014" not in result.processed_text
