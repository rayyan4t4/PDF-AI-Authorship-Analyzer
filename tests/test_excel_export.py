"""
tests/test_excel_export.py

Unit tests for core/excel_exporter.py
"""

import io
import pytest
from openpyxl import load_workbook

from core.batch_processor import DocumentResult, BatchResult
from core.excel_exporter import export_to_excel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch_result(n_docs: int = 3) -> BatchResult:
    """Create a synthetic BatchResult with n_docs documents."""
    docs = []
    for i in range(n_docs):
        doc = DocumentResult(
            filename=f"document_{i+1:02d}.pdf",
            page_count=i + 1,
            word_count=500 + i * 100,
            char_count=3000 + i * 500,
            extraction_status="success",
            ai_risk_score=20.0 + i * 25.0,
            classification="Elevated AI-authorship indicators" if i % 2 == 0 else "Low AI-authorship indicators",
            confidence="High" if i > 0 else "Low",
            transformer_signal=60.0 + i * 5,
            stylometric_signal=40.0 + i * 3,
            statistical_signal=55.0,
            repetition_signal=50.0,
            sentence_uniformity_signal=65.0,
            detected_patterns=["Elevated sentence uniformity", "High repetition ratio"],
            sentence_count=30 + i * 5,
            paragraph_count=5 + i,
            vocabulary_size=200 + i * 20,
            avg_sentence_length=18.5,
            std_sentence_length=3.2,
            type_token_ratio=0.45,
            transformer_segment_count=4,
            transformer_pct_flagged=60.0,
            status="completed",
        )
        docs.append(doc)

    # Add one error document
    error_doc = DocumentResult(
        filename="corrupted.pdf",
        status="error",
        error="PDF could not be opened: invalid structure.",
        classification="Error",
        confidence="",
    )
    docs.append(error_doc)

    batch = BatchResult(
        documents=docs,
        total=len(docs),
        analyzed=n_docs,
        failed=1,
        high_risk=1,
        elevated_risk=1,
        uncertain=0,
        low_risk=1,
        average_score=55.0,
        total_time_s=4.2,
    )
    return batch


class TestExcelExport:

    def test_returns_bytes(self):
        batch = _make_batch_result()
        result = export_to_excel(batch)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_valid_xlsx(self):
        """Output should be a valid Excel file that openpyxl can open."""
        batch = _make_batch_result()
        excel_bytes = export_to_excel(batch)
        wb = load_workbook(io.BytesIO(excel_bytes))
        assert wb is not None

    def test_has_three_sheets(self):
        """Excel file must contain exactly 3 sheets."""
        batch = _make_batch_result()
        excel_bytes = export_to_excel(batch)
        wb = load_workbook(io.BytesIO(excel_bytes))
        assert len(wb.sheetnames) == 3

    def test_sheet_names(self):
        """Sheet names must be Summary, Detailed Analysis, Errors."""
        batch = _make_batch_result()
        excel_bytes = export_to_excel(batch)
        wb = load_workbook(io.BytesIO(excel_bytes))
        expected = {"Summary", "Detailed Analysis", "Errors"}
        assert set(wb.sheetnames) == expected

    def test_summary_sheet_has_header(self):
        """Summary sheet row 1 must contain 'PDF Name'."""
        batch = _make_batch_result()
        excel_bytes = export_to_excel(batch)
        wb = load_workbook(io.BytesIO(excel_bytes))
        ws = wb["Summary"]
        headers = [cell.value for cell in ws[1]]
        assert "PDF Name" in headers

    def test_detailed_sheet_has_signals(self):
        """Detailed Analysis sheet must include transformer/stylometric columns."""
        batch = _make_batch_result()
        excel_bytes = export_to_excel(batch)
        wb = load_workbook(io.BytesIO(excel_bytes))
        ws = wb["Detailed Analysis"]
        headers = [str(cell.value) for cell in ws[1]]
        assert any("Transformer" in h for h in headers)
        assert any("Stylometric" in h for h in headers)

    def test_errors_sheet_contains_error(self):
        """Errors sheet must list the corrupted.pdf error."""
        batch = _make_batch_result()
        excel_bytes = export_to_excel(batch)
        wb = load_workbook(io.BytesIO(excel_bytes))
        ws = wb["Errors"]
        all_values = [str(cell.value) for row in ws.iter_rows() for cell in row]
        assert any("corrupted.pdf" in v for v in all_values)

    def test_summary_row_count(self):
        """Summary sheet should have 1 header row + n_docs+1 data rows."""
        n = 3
        batch = _make_batch_result(n_docs=n)
        excel_bytes = export_to_excel(batch)
        wb = load_workbook(io.BytesIO(excel_bytes))
        ws = wb["Summary"]
        # header row + n docs + 1 error doc
        assert ws.max_row == 1 + n + 1

    def test_freeze_panes_set(self):
        """Summary sheet should have freeze pane at A2."""
        batch = _make_batch_result()
        excel_bytes = export_to_excel(batch)
        wb = load_workbook(io.BytesIO(excel_bytes))
        ws = wb["Summary"]
        assert ws.freeze_panes is not None

    def test_autofilter_set(self):
        """Summary sheet should have autofilter."""
        batch = _make_batch_result()
        excel_bytes = export_to_excel(batch)
        wb = load_workbook(io.BytesIO(excel_bytes))
        ws = wb["Summary"]
        assert ws.auto_filter.ref is not None

    def test_empty_batch_no_crash(self):
        """Empty batch should produce a valid (empty) Excel file."""
        batch = BatchResult(documents=[], total=0)
        excel_bytes = export_to_excel(batch)
        wb = load_workbook(io.BytesIO(excel_bytes))
        assert wb is not None
