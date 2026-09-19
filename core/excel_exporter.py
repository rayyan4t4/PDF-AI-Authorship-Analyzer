"""
core/excel_exporter.py

Exports analysis results to a formatted Excel .xlsx file with three sheets:
  1. Summary
  2. Detailed Analysis
  3. Errors

Uses pandas + openpyxl with formatting: bold headers, autofilter,
freeze row 1, column widths, and conditional formatting on AI Risk Score.
"""

import io
import logging
from typing import Optional

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side, numbers
)
from openpyxl.styles.numbers import FORMAT_NUMBER_00
from openpyxl.formatting.rule import ColorScaleRule, CellIsRule
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

from core.batch_processor import DocumentResult, BatchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color constants for conditional formatting
# ---------------------------------------------------------------------------
COLOR_HIGH_RISK   = "FF4444"   # Red
COLOR_ELEVATED    = "FF8800"   # Orange
COLOR_UNCERTAIN   = "FFCC00"   # Yellow
COLOR_LOW_RISK    = "22CC44"   # Green
COLOR_HEADER_BG   = "1A3A5C"  # Dark blue
COLOR_HEADER_FG   = "FFFFFF"  # White

THIN = Side(style="thin", color="CCCCCC")
THIN_BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def _style_header_row(ws, row_idx: int = 1):
    """Apply bold, colored header style to a worksheet row."""
    for cell in ws[row_idx]:
        cell.font = Font(bold=True, color=COLOR_HEADER_FG, name="Calibri", size=11)
        cell.fill = PatternFill("solid", fgColor=COLOR_HEADER_BG)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = THIN_BORDER


def _auto_column_widths(ws, min_width: int = 10, max_width: int = 45):
    """Set column widths based on content."""
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = max(
            (len(str(cell.value)) if cell.value is not None else 0 for cell in col),
            default=10,
        )
        ws.column_dimensions[col_letter].width = max(min_width, min(max_width, max_len + 2))


def _apply_score_conditional_formatting(ws, score_col_letter: str, start_row: int, end_row: int):
    """
    Apply conditional formatting to AI Risk Score column:
    - >= 81: Red
    - 61–80: Orange
    - 41–60: Yellow
    - < 41: Green
    """
    score_range = f"{score_col_letter}{start_row}:{score_col_letter}{end_row}"

    rules = [
        (81, 100, COLOR_HIGH_RISK),
        (61, 80,  COLOR_ELEVATED),
        (41, 60,  COLOR_UNCERTAIN),
        (0,  40,  COLOR_LOW_RISK),
    ]

    for lo, hi, color in rules:
        fill = PatternFill("solid", fgColor=color)
        ws.conditional_formatting.add(
            score_range,
            CellIsRule(
                operator="between",
                formula=[str(lo), str(hi)],
                fill=fill,
            ),
        )


def _format_score(val) -> str:
    """Display score as integer string or N/A."""
    if val is None or (isinstance(val, float) and val != val):  # NaN check
        return "N/A"
    try:
        return str(int(round(float(val))))
    except (TypeError, ValueError):
        return "N/A"


def _format_signal(val) -> str:
    """Display signal as integer string or N/A."""
    if val is None:
        return "N/A"
    try:
        return str(int(round(float(val))))
    except (TypeError, ValueError):
        return "N/A"


def _build_summary_df(documents: list[DocumentResult]) -> pd.DataFrame:
    rows = []
    for doc in documents:
        rows.append({
            "PDF Name": doc.filename,
            "AI Risk Score": _format_score(doc.ai_risk_score),
            "Classification": doc.classification,
            "Confidence": doc.confidence,
            "Word Count": doc.word_count,
            "Page Count": doc.page_count,
            "Status": doc.status.replace("_", " ").title(),
        })
    return pd.DataFrame(rows)


def _build_detailed_df(documents: list[DocumentResult]) -> pd.DataFrame:
    rows = []
    for doc in documents:
        rows.append({
            "PDF Name": doc.filename,
            "AI Risk Score": _format_score(doc.ai_risk_score),
            "Transformer Signal": _format_signal(doc.transformer_signal),
            "Stylometric Signal": _format_signal(doc.stylometric_signal),
            "Statistical Signal": _format_signal(doc.statistical_signal),
            "Repetition Signal": _format_signal(doc.repetition_signal),
            "Sentence Uniformity Signal": _format_signal(doc.sentence_uniformity_signal),
            "Word Count": doc.word_count,
            "Sentence Count": doc.sentence_count,
            "Paragraph Count": doc.paragraph_count,
            "Vocabulary Size": doc.vocabulary_size,
            "Average Sentence Length": round(doc.avg_sentence_length, 1),
            "Sentence Length Std": round(doc.std_sentence_length, 1),
            "Type Token Ratio": round(doc.type_token_ratio, 3),
            "Transformer Segments": doc.transformer_segment_count,
            "Segments Flagged (%)": round(doc.transformer_pct_flagged, 1),
            "Detected Patterns": "; ".join(doc.detected_patterns),
        })
    return pd.DataFrame(rows)


def _build_errors_df(documents: list[DocumentResult]) -> pd.DataFrame:
    rows = []
    for doc in documents:
        if doc.error:
            rows.append({
                "PDF Name": doc.filename,
                "Error Type": doc.status.replace("_", " ").title(),
                "Error Message": doc.error or "",
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["PDF Name", "Error Type", "Error Message"]
    )


def _write_df_to_sheet(ws, df: pd.DataFrame, sheet_title: str):
    """Write a DataFrame to a worksheet with styling."""
    if df.empty:
        ws.append(["No data available."])
        return

    # Write header
    ws.append(list(df.columns))
    _style_header_row(ws, row_idx=1)

    # Write data rows
    for row in dataframe_to_rows(df, index=False, header=False):
        ws.append(row)
        # Light border on data cells
        for cell in ws[ws.max_row]:
            cell.border = THIN_BORDER
            cell.alignment = Alignment(horizontal="left", vertical="center")

    # Freeze header row
    ws.freeze_panes = "A2"

    # Autofilter
    if ws.max_row > 1:
        ws.auto_filter.ref = ws.dimensions

    # Column widths
    _auto_column_widths(ws)


def export_to_excel(batch_result: BatchResult) -> bytes:
    """
    Generate a formatted Excel file from batch results.

    Returns:
        Excel file as bytes (suitable for Streamlit download_button).
    """
    wb = Workbook()

    # Remove default sheet
    wb.remove(wb.active)

    documents = batch_result.documents

    # ------------------------------------------------------------------
    # Sheet 1: Summary
    # ------------------------------------------------------------------
    ws_summary = wb.create_sheet("Summary")
    summary_df = _build_summary_df(documents)
    _write_df_to_sheet(ws_summary, summary_df, "Summary")

    # Conditional formatting on AI Risk Score column (col B = index 2)
    if not summary_df.empty and ws_summary.max_row > 1:
        _apply_score_conditional_formatting(ws_summary, "B", 2, ws_summary.max_row)

    # ------------------------------------------------------------------
    # Sheet 2: Detailed Analysis
    # ------------------------------------------------------------------
    ws_detail = wb.create_sheet("Detailed Analysis")
    detail_df = _build_detailed_df(documents)
    _write_df_to_sheet(ws_detail, detail_df, "Detailed Analysis")

    if not detail_df.empty and ws_detail.max_row > 1:
        _apply_score_conditional_formatting(ws_detail, "B", 2, ws_detail.max_row)

    # ------------------------------------------------------------------
    # Sheet 3: Errors
    # ------------------------------------------------------------------
    ws_errors = wb.create_sheet("Errors")
    errors_df = _build_errors_df(documents)
    _write_df_to_sheet(ws_errors, errors_df, "Errors")

    # ------------------------------------------------------------------
    # Write to bytes
    # ------------------------------------------------------------------
    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    content = buffer.read()

    logger.info(
        f"Excel export: {len(documents)} documents, "
        f"{len(content)//1024} KB"
    )

    return content
