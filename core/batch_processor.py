"""
core/batch_processor.py

Orchestrates the full analysis pipeline for a batch of PDFs.
Uses concurrent.futures for parallelism.
One failed PDF does NOT stop the batch.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional, Any

from core.pdf_extractor import PDFExtractionResult, extract_pdf
from core.text_preprocessor import preprocess
from core.feature_extractor import extract_features, TextFeatures
from core.ai_detector import AITextDetector, TransformerDetectionResult
from core.scoring import ScoringEngine, ScoringResult
from config.settings import MIN_WORD_COUNT, CLASSIFICATION_LABELS

logger = logging.getLogger(__name__)


@dataclass
class DocumentResult:
    """Complete result for a single PDF document."""

    # Identity
    filename: str = ""

    # Extraction
    page_count: int = 0
    word_count: int = 0
    char_count: int = 0
    extraction_status: str = "pending"

    # Scoring
    ai_risk_score: Optional[float] = None
    classification: str = ""
    confidence: str = ""

    # Signals
    transformer_signal: Optional[float] = None
    stylometric_signal: float = 0.0
    statistical_signal: float = 0.0
    repetition_signal: float = 0.0
    sentence_uniformity_signal: float = 0.0

    # Patterns
    detected_patterns: list[str] = field(default_factory=list)

    # Extended stats
    sentence_count: int = 0
    paragraph_count: int = 0
    vocabulary_size: int = 0
    avg_sentence_length: float = 0.0
    std_sentence_length: float = 0.0
    type_token_ratio: float = 0.0
    transformer_segment_count: int = 0
    transformer_pct_flagged: float = 0.0

    # Overall status
    status: str = "pending"   # completed | error | no_text | encrypted | insufficient_text
    error: Optional[str] = None

    # Timing
    processing_time_s: float = 0.0


@dataclass
class BatchResult:
    """Summary of a complete batch analysis run."""
    documents: list[DocumentResult] = field(default_factory=list)
    total: int = 0
    analyzed: int = 0
    insufficient_text: int = 0
    no_text: int = 0
    failed: int = 0
    high_risk: int = 0
    elevated_risk: int = 0
    uncertain: int = 0
    low_risk: int = 0
    average_score: Optional[float] = None
    total_time_s: float = 0.0


def _process_single(
    pdf_bytes: bytes,
    filename: str,
    detector: AITextDetector,
    scoring_engine: ScoringEngine,
    min_word_count: int,
) -> DocumentResult:
    """
    Full pipeline for a single PDF.

    Steps:
    1. Extract text (PyMuPDF)
    2. Preprocess
    3. Extract features
    4. Run transformer detection
    5. Score

    Errors at any step are caught and recorded.
    """
    t0 = time.perf_counter()
    doc = DocumentResult(filename=filename)

    # Step 1: Extract
    try:
        extraction: PDFExtractionResult = extract_pdf(pdf_bytes, filename)
        doc.page_count = extraction.page_count
        doc.char_count = extraction.char_count
        doc.extraction_status = extraction.status

        if extraction.status in ("no_text", "encrypted", "error"):
            doc.status = extraction.status
            doc.error = extraction.error
            doc.classification = CLASSIFICATION_LABELS.get(
                "no_text" if extraction.status == "no_text" else "error", "Error"
            )
            doc.confidence = "Low"
            doc.processing_time_s = time.perf_counter() - t0
            return doc

    except Exception as e:
        doc.status = "error"
        doc.error = f"Extraction failed: {e}"
        doc.classification = CLASSIFICATION_LABELS["error"]
        doc.processing_time_s = time.perf_counter() - t0
        logger.error(f"[{filename}] Extraction exception: {e}")
        return doc

    # Step 2: Preprocess
    try:
        preprocessed = preprocess(extraction.raw_text)
        doc.word_count = preprocessed.word_count
    except Exception as e:
        doc.status = "error"
        doc.error = f"Preprocessing failed: {e}"
        doc.classification = CLASSIFICATION_LABELS["error"]
        doc.processing_time_s = time.perf_counter() - t0
        logger.error(f"[{filename}] Preprocessing exception: {e}")
        return doc

    # Step 3: Extract features
    try:
        features: TextFeatures = extract_features(preprocessed)
    except Exception as e:
        doc.status = "error"
        doc.error = f"Feature extraction failed: {e}"
        doc.classification = CLASSIFICATION_LABELS["error"]
        doc.processing_time_s = time.perf_counter() - t0
        logger.error(f"[{filename}] Feature extraction exception: {e}")
        return doc

    # Step 4: Transformer detection
    try:
        transformer_result: TransformerDetectionResult = detector.predict_document(
            preprocessed.processed_text
        )
    except Exception as e:
        logger.warning(f"[{filename}] Transformer detection failed: {e}")
        from core.ai_detector import TransformerDetectionResult
        transformer_result = TransformerDetectionResult(
            available=False, error=str(e)
        )

    # Step 5: Score
    try:
        score_result: ScoringResult = scoring_engine.score(
            filename=filename,
            features=features,
            transformer=transformer_result,
            min_word_count=min_word_count,
        )
    except Exception as e:
        doc.status = "error"
        doc.error = f"Scoring failed: {e}"
        doc.classification = CLASSIFICATION_LABELS["error"]
        doc.processing_time_s = time.perf_counter() - t0
        logger.error(f"[{filename}] Scoring exception: {e}")
        return doc

    # Assemble final DocumentResult
    doc.ai_risk_score = score_result.ai_risk_score
    doc.classification = score_result.classification
    doc.confidence = score_result.confidence
    doc.transformer_signal = score_result.transformer_signal
    doc.stylometric_signal = score_result.stylometric_signal
    doc.statistical_signal = score_result.statistical_signal
    doc.repetition_signal = score_result.repetition_signal
    doc.sentence_uniformity_signal = score_result.sentence_uniformity_signal
    doc.detected_patterns = score_result.detected_patterns
    doc.sentence_count = score_result.sentence_count
    doc.paragraph_count = score_result.paragraph_count
    doc.vocabulary_size = score_result.vocabulary_size
    doc.avg_sentence_length = score_result.avg_sentence_length
    doc.std_sentence_length = score_result.std_sentence_length
    doc.type_token_ratio = score_result.type_token_ratio
    doc.transformer_segment_count = score_result.transformer_segment_count
    doc.transformer_pct_flagged = score_result.transformer_pct_flagged
    doc.status = score_result.status
    doc.error = score_result.error
    doc.processing_time_s = time.perf_counter() - t0

    return doc


def _summarize_batch(documents: list[DocumentResult], total_time: float) -> BatchResult:
    """Compute batch-level summary statistics."""
    batch = BatchResult(documents=documents, total=len(documents), total_time_s=total_time)

    scores = []
    for doc in documents:
        if doc.status == "completed":
            batch.analyzed += 1
            if doc.ai_risk_score is not None:
                scores.append(doc.ai_risk_score)
                # Classification bucketing
                score = doc.ai_risk_score
                if score >= 81:
                    batch.high_risk += 1
                elif score >= 61:
                    batch.elevated_risk += 1
                elif score >= 41:
                    batch.uncertain += 1
                else:
                    batch.low_risk += 1
        elif doc.status == "insufficient_text":
            batch.insufficient_text += 1
        elif doc.status in ("no_text", "encrypted"):
            batch.no_text += 1
        else:
            batch.failed += 1

    if scores:
        batch.average_score = round(sum(scores) / len(scores), 1)

    return batch


def process_batch(
    uploaded_files: list[tuple[str, bytes]],  # (filename, bytes) pairs
    detector: AITextDetector,
    scoring_engine: ScoringEngine,
    min_word_count: int = MIN_WORD_COUNT,
    max_workers: int = 2,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> BatchResult:
    """
    Process a batch of PDFs with concurrent execution.

    Args:
        uploaded_files: List of (filename, bytes) tuples.
        detector: Initialized AITextDetector.
        scoring_engine: Initialized ScoringEngine.
        min_word_count: Minimum words for full analysis.
        max_workers: Thread pool size.
        progress_callback: Optional fn(completed, total, current_filename).

    Returns:
        BatchResult with all document results and summary statistics.
    """
    t0 = time.perf_counter()
    total = len(uploaded_files)
    results: list[DocumentResult] = [None] * total  # preserve order

    logger.info(f"Starting batch: {total} PDFs, max_workers={max_workers}")

    # NOTE: Transformer models are NOT thread-safe for GPU inference.
    # We use max_workers=1 when transformer is available to avoid CUDA race conditions.
    # Text extraction and feature computation are safe to parallelize.
    effective_workers = 1 if (detector.available and max_workers > 1) else max_workers

    completed_count = 0

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        # Submit all jobs, preserving index for ordering
        future_to_idx = {
            executor.submit(
                _process_single,
                pdf_bytes,
                filename,
                detector,
                scoring_engine,
                min_word_count,
            ): idx
            for idx, (filename, pdf_bytes) in enumerate(uploaded_files)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            filename = uploaded_files[idx][0]

            try:
                doc_result = future.result()
            except Exception as e:
                logger.error(f"[{filename}] Unexpected future error: {e}")
                doc_result = DocumentResult(
                    filename=filename,
                    status="error",
                    error=str(e),
                    classification=CLASSIFICATION_LABELS["error"],
                )

            results[idx] = doc_result
            completed_count += 1

            if progress_callback:
                try:
                    progress_callback(completed_count, total, filename)
                except Exception:
                    pass

    # Filter out None (shouldn't happen, but defensive)
    results = [r for r in results if r is not None]

    total_time = time.perf_counter() - t0
    logger.info(f"Batch complete: {total} PDFs in {total_time:.1f}s")

    return _summarize_batch(results, total_time)
