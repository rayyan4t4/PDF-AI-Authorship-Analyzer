"""
core/ai_detector.py

AITextDetector: Transformer-based AI text detection with document segmentation.

Design principles:
- Does NOT hard-code an arbitrary or nonexistent model.
- Falls back gracefully if no model is configured.
- Segments long documents — does NOT simply truncate to 512 tokens.
- Reports per-segment scores and aggregate statistics.
- Clearly indicates when the transformer signal is unavailable.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SegmentResult:
    """Prediction result for a single text segment."""
    text_preview: str = ""
    ai_score: float = 0.0       # 0–100
    raw_logit_ai: float = 0.0
    token_count: int = 0


@dataclass
class TransformerDetectionResult:
    """Aggregated transformer detection result for a full document."""
    available: bool = False        # True only if model was loaded and ran
    mean_score: float = 0.0        # 0–100
    median_score: float = 0.0
    std_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    pct_segments_flagged: float = 0.0   # % segments with score ≥ 60
    segment_count: int = 0
    segments: list[SegmentResult] = field(default_factory=list)
    error: Optional[str] = None


class AITextDetector:
    """
    Transformer-based AI text detector with document segmentation.

    Usage:
        detector = AITextDetector(tokenizer, model, max_tokens=512)
        result = detector.predict_document(text)
    """

    FLAG_THRESHOLD = 60.0   # score >= this considered "flagged"

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer (or None).
            model:     HuggingFace sequence classification model (or None).
            max_tokens: Maximum tokens per segment.
            overlap_tokens: Token overlap between consecutive segments.
        """
        self.tokenizer = tokenizer
        self.model = model
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.available = tokenizer is not None and model is not None

        if self.available:
            # Determine which label index corresponds to AI/machine
            try:
                labels = model.config.id2label
                self._ai_label_idx = next(
                    (i for i, v in labels.items()
                     if "fake" in v.lower() or "machine" in v.lower()
                        or "ai" in v.lower() or "generated" in v.lower()),
                    1  # default: label 1 = AI
                )
            except Exception:
                self._ai_label_idx = 1

            logger.info(
                f"AITextDetector ready. AI label index: {self._ai_label_idx}, "
                f"max_tokens: {max_tokens}"
            )
        else:
            logger.info("AITextDetector running WITHOUT a model (no transformer signal).")

    def _get_device(self):
        """Return the device the model is on."""
        try:
            return next(self.model.parameters()).device
        except Exception:
            return "cpu"

    def _segment_text(self, text: str) -> list[str]:
        """
        Split text into overlapping segments that fit within max_tokens.

        Strategy: tokenize the full text, then chunk by token indices.
        """
        if not self.available:
            return [text]

        import torch

        try:
            encoding = self.tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False,
            )
        except Exception as e:
            logger.warning(f"Tokenization failed, using paragraph split: {e}")
            return self._fallback_segment(text)

        token_ids = encoding["input_ids"]
        offsets = encoding.get("offset_mapping", None)

        effective_max = self.max_tokens - 2  # reserve for [CLS]/[SEP]
        step = max(1, effective_max - self.overlap_tokens)

        segments = []
        for start in range(0, len(token_ids), step):
            chunk_ids = token_ids[start: start + effective_max]
            if offsets is not None:
                chunk_offsets = offsets[start: start + effective_max]
                if chunk_offsets:
                    char_start = chunk_offsets[0][0]
                    char_end = chunk_offsets[-1][1]
                    segment = text[char_start:char_end].strip()
                else:
                    segment = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            else:
                segment = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)

            if segment.strip():
                segments.append(segment)

        return segments if segments else [text[:2000]]

    def _fallback_segment(self, text: str) -> list[str]:
        """Paragraph-based fallback when tokenization offset mapping is unavailable."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            # Character-based chunking
            chunk_size = 1500
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        return paragraphs

    def _predict_segment(self, text: str) -> SegmentResult:
        """Run the model on a single segment and return a SegmentResult."""
        import torch

        result = SegmentResult(text_preview=text[:100])

        try:
            device = self._get_device()
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_tokens,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            result.token_count = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0]
                probs = torch.softmax(logits, dim=-1)

            ai_prob = float(probs[self._ai_label_idx].cpu())
            result.ai_score = ai_prob * 100.0
            result.raw_logit_ai = float(logits[self._ai_label_idx].cpu())

        except Exception as e:
            logger.warning(f"Segment prediction failed: {e}")
            result.ai_score = 0.0
            result.error = str(e) if hasattr(result, 'error') else None

        return result

    def predict_document(self, text: str) -> TransformerDetectionResult:
        """
        Analyze a full document, segmenting as needed.

        Args:
            text: The (preprocessed) document text.

        Returns:
            TransformerDetectionResult with per-segment and aggregate scores.
        """
        result = TransformerDetectionResult()

        if not self.available:
            result.available = False
            result.error = "No transformer model configured."
            return result

        if not text or not text.strip():
            result.available = False
            result.error = "Empty text."
            return result

        try:
            segments = self._segment_text(text)
            seg_results = [self._predict_segment(seg) for seg in segments]

            result.available = True
            result.segments = seg_results
            result.segment_count = len(seg_results)

            scores = np.array([s.ai_score for s in seg_results])
            result.mean_score = float(scores.mean())
            result.median_score = float(np.median(scores))
            result.std_score = float(scores.std()) if len(scores) > 1 else 0.0
            result.min_score = float(scores.min())
            result.max_score = float(scores.max())
            result.pct_segments_flagged = float(
                np.mean(scores >= self.FLAG_THRESHOLD) * 100.0
            )

            logger.debug(
                f"Transformer: {len(seg_results)} segments, "
                f"mean={result.mean_score:.1f}, "
                f"flagged={result.pct_segments_flagged:.0f}%"
            )

        except Exception as e:
            result.available = False
            result.error = str(e)
            logger.error(f"Transformer detection failed: {e}")

        return result
