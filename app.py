"""
app.py — PDF AI Authorship Analyzer

Main Streamlit application entry point.
All detection and processing logic lives in the core/ and models/ packages.

Run with:
    streamlit run app.py
"""

import logging
import time
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.logging_config import setup_logging
from config.settings import (
    Settings,
    get_default_settings,
    CLASSIFICATION_LABELS,
    MODEL_NAME,
    MIN_WORD_COUNT,
    MAX_SEGMENT_TOKENS,
    TRANSFORMER_WEIGHT,
    STYLOMETRIC_WEIGHT,
    STATISTICAL_WEIGHT,
    REPETITION_WEIGHT,
    SENTENCE_UNIFORMITY_WEIGHT,
    MAX_WORKERS,
)
from utils.validators import validate_uploaded_file, validate_weights
from utils.helpers import format_score, seconds_to_human

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PDF AI Authorship Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "PDF AI Authorship Analyzer — V1\n\nAnalyzes text-based PDFs for AI-authorship indicators.",
    },
)

# ---------------------------------------------------------------------------
# Custom CSS (light, clean, professional)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */
    .app-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #E8EDF2;
        margin-bottom: 2rem;
    }
    .app-header h1 {
        font-size: 1.9rem;
        font-weight: 700;
        color: #1A3A5C;
        margin: 0;
    }
    .app-header p {
        color: #6B7280;
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    /* Metric cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1A3A5C;
        line-height: 1.1;
    }
    .metric-card .metric-label {
        font-size: 0.78rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 0.25rem;
    }

    /* Risk badge colors */
    .badge-high     { background:#FFEAEA; color:#CC0000; border:1px solid #FFAAAA; }
    .badge-elevated { background:#FFF3E0; color:#C04800; border:1px solid #FFCC80; }
    .badge-uncertain{ background:#FFFDE7; color:#806600; border:1px solid #FFE082; }
    .badge-low      { background:#E8F5E9; color:#1B5E20; border:1px solid #A5D6A7; }
    .badge-insuf    { background:#F3F4F6; color:#4B5563; border:1px solid #D1D5DB; }

    .risk-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1A3A5C;
        border-left: 4px solid #1A3A5C;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Signal bar */
    .signal-row {
        display: flex;
        align-items: center;
        margin: 0.4rem 0;
        gap: 0.8rem;
    }
    .signal-label {
        min-width: 200px;
        font-size: 0.88rem;
        color: #374151;
    }
    .signal-bar-bg {
        flex: 1;
        height: 8px;
        background: #E5E7EB;
        border-radius: 4px;
        overflow: hidden;
    }
    .signal-bar-fill {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #1A3A5C, #2D6A9F);
    }
    .signal-value {
        min-width: 40px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #1A3A5C;
        text-align: right;
    }

    /* Disclaimer box */
    .disclaimer {
        background: #F0F4F8;
        border-left: 4px solid #90A4AE;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        color: #546E7A;
        margin: 1rem 0;
    }

    /* Streamlit overrides */
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
    .stDataFrame { border: 1px solid #E5E7EB; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI detection model...")
def load_model(model_name: str):
    """Load and cache the transformer model. Called once per session."""
    from models.model_loader import load_detector_model
    tokenizer, model = load_detector_model(model_name)
    return tokenizer, model


# ---------------------------------------------------------------------------
# Helper: get detector instance
# ---------------------------------------------------------------------------
def get_detector(model_name: str, max_tokens: int):
    from core.ai_detector import AITextDetector
    tokenizer, model = load_model(model_name)
    return AITextDetector(tokenizer, model, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Sidebar settings
# ---------------------------------------------------------------------------
def render_sidebar() -> Settings:
    st.sidebar.markdown("## ⚙️ Settings")

    model_name = st.sidebar.text_input(
        "Detection Model",
        value=MODEL_NAME,
        help="Hugging Face model ID. See models/README.md for recommendations.",
        key="sidebar_model_name",
    )

    min_word_count = st.sidebar.number_input(
        "Minimum Word Count",
        min_value=50,
        max_value=2000,
        value=MIN_WORD_COUNT,
        step=50,
        help="Documents below this word count are marked as 'Insufficient text'.",
        key="sidebar_min_words",
    )

    with st.sidebar.expander("Advanced Scoring Weights", expanded=False):
        st.caption("Weights control how much each signal contributes to the final score. They are normalized automatically.")

        w_trans = st.slider("Transformer Weight", 0.0, 1.0, TRANSFORMER_WEIGHT, 0.05, key="w_trans")
        w_sty   = st.slider("Stylometric Weight", 0.0, 1.0, STYLOMETRIC_WEIGHT, 0.05, key="w_sty")
        w_sta   = st.slider("Statistical Weight", 0.0, 1.0, STATISTICAL_WEIGHT, 0.05, key="w_sta")
        w_rep   = st.slider("Repetition Weight", 0.0, 1.0, REPETITION_WEIGHT, 0.05, key="w_rep")
        w_uni   = st.slider("Sentence Uniformity Weight", 0.0, 1.0, SENTENCE_UNIFORMITY_WEIGHT, 0.05, key="w_uni")

        total_w = w_trans + w_sty + w_sta + w_rep + w_uni
        if total_w > 0:
            st.caption(f"Weight sum: **{total_w:.2f}** (auto-normalized to 1.0)")
        else:
            st.warning("All weights are zero.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size:0.75rem; color:#9CA3AF;">
    <b>Privacy:</b> PDFs are processed locally.<br>
    No text is sent to external APIs.<br>
    Data is not stored between sessions.
    </div>
    """, unsafe_allow_html=True)

    return Settings(
        model_name=model_name.strip(),
        min_word_count=int(min_word_count),
        max_segment_tokens=MAX_SEGMENT_TOKENS,
        transformer_weight=w_trans,
        stylometric_weight=w_sty,
        statistical_weight=w_sta,
        repetition_weight=w_rep,
        sentence_uniformity_weight=w_uni,
        max_workers=MAX_WORKERS,
    )


# ---------------------------------------------------------------------------
# Score badge HTML
# ---------------------------------------------------------------------------
def _score_badge(score, classification: str) -> str:
    score_str = format_score(score)

    if score is None or classification in (
        CLASSIFICATION_LABELS["insufficient"],
        CLASSIFICATION_LABELS["no_text"],
        CLASSIFICATION_LABELS["error"],
    ):
        cls = "badge-insuf"
    elif score >= 81:
        cls = "badge-high"
    elif score >= 61:
        cls = "badge-elevated"
    elif score >= 41:
        cls = "badge-uncertain"
    else:
        cls = "badge-low"

    return (
        f'<span class="risk-badge {cls}">{score_str}</span>'
    )


# ---------------------------------------------------------------------------
# Dashboard metrics
# ---------------------------------------------------------------------------
def render_dashboard(batch_result):
    from core.batch_processor import BatchResult

    st.markdown('<div class="section-header">📊 Analysis Summary</div>', unsafe_allow_html=True)

    cols = st.columns(9)
    metrics = [
        ("Total PDFs",           batch_result.total,            "#1A3A5C"),
        ("Analyzed",             batch_result.analyzed,          "#2563EB"),
        ("Insuf. Text",          batch_result.insufficient_text, "#6B7280"),
        ("No Text / Error",      batch_result.no_text + batch_result.failed, "#6B7280"),
        ("🔴 High Risk",         batch_result.high_risk,         "#DC2626"),
        ("🟠 Elevated Risk",     batch_result.elevated_risk,     "#D97706"),
        ("🟡 Uncertain",         batch_result.uncertain,         "#CA8A04"),
        ("🟢 Low Risk",          batch_result.low_risk,          "#16A34A"),
        ("Avg Risk Score",
         f"{batch_result.average_score:.0f}" if batch_result.average_score is not None else "N/A",
         "#1A3A5C"),
    ]

    for col, (label, value, color) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color}">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def render_charts(batch_result):
    scored_docs = [
        d for d in batch_result.documents
        if d.ai_risk_score is not None
    ]

    if not scored_docs:
        st.info("No scored documents to visualize.")
        return

    scores = [d.ai_risk_score for d in scored_docs]
    names  = [d.filename for d in scored_docs]

    st.markdown('<div class="section-header">📈 Score Distribution</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Histogram
    with col1:
        fig_hist = px.histogram(
            x=scores,
            nbins=min(20, len(scores)),
            labels={"x": "AI Risk Score", "y": "Count"},
            title="AI Risk Score Distribution",
            color_discrete_sequence=["#1A3A5C"],
        )
        fig_hist.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_family="Inter",
            title_font_size=14,
            showlegend=False,
            xaxis=dict(range=[0, 100], gridcolor="#F3F4F6"),
            yaxis=dict(gridcolor="#F3F4F6"),
            bargap=0.1,
        )
        # Add reference lines
        for x_val, color, label in [
            (20, "#22CC44", "Low"),
            (40, "#FFCC00", "Rel. Low"),
            (60, "#FF8800", "Elevated"),
            (80, "#FF4444", "High"),
        ]:
            fig_hist.add_vline(x=x_val, line_dash="dash", line_color=color,
                               annotation_text=label, annotation_position="top",
                               line_width=1)
        st.plotly_chart(fig_hist, width='stretch')

    # Bar chart per file
    with col2:
        # Sort by score descending
        sorted_pairs = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
        sorted_names, sorted_scores = zip(*sorted_pairs) if sorted_pairs else ([], [])

        # Color bars by risk level
        bar_colors = []
        for s in sorted_scores:
            if s >= 81:   bar_colors.append("#FF4444")
            elif s >= 61: bar_colors.append("#FF8800")
            elif s >= 41: bar_colors.append("#FFCC00")
            else:         bar_colors.append("#22CC44")

        fig_bar = go.Figure(go.Bar(
            x=list(sorted_names),
            y=list(sorted_scores),
            marker_color=bar_colors,
            text=[f"{s:.0f}" for s in sorted_scores],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="PDF Name vs AI Risk Score",
            xaxis_title="Document",
            yaxis_title="AI Risk Score",
            yaxis=dict(range=[0, 110], gridcolor="#F3F4F6"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_family="Inter",
            title_font_size=14,
            showlegend=False,
            xaxis_tickangle=-35,
        )
        st.plotly_chart(fig_bar, width='stretch')


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
def render_results_table(batch_result) -> Optional[str]:
    """
    Render the main results table. Returns the filename selected for inspection,
    or None.
    """
    st.markdown('<div class="section-header">📋 Results Table</div>', unsafe_allow_html=True)

    rows = []
    for doc in batch_result.documents:
        rows.append({
            "PDF Name": doc.filename,
            "AI Risk Score": format_score(doc.ai_risk_score),
            "Classification": doc.classification,
            "Confidence": doc.confidence,
            "Word Count": doc.word_count,
            "Page Count": doc.page_count,
            "Transformer": format_score(doc.transformer_signal),
            "Stylometric": format_score(doc.stylometric_signal),
            "Statistical": format_score(doc.statistical_signal),
            "Repetition": format_score(doc.repetition_signal),
            "Sent. Uniformity": format_score(doc.sentence_uniformity_signal),
            "Status": doc.status.replace("_", " ").title(),
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        width="stretch",
        height=min(600, 60 + 35 * len(rows)),
        hide_index=True,
    )

    return None


# ---------------------------------------------------------------------------
# Individual document inspector
# ---------------------------------------------------------------------------
def render_individual_inspector(batch_result):
    st.markdown('<div class="section-header">🔎 Individual Document Inspection</div>', unsafe_allow_html=True)

    filenames = [d.filename for d in batch_result.documents]
    selected = st.selectbox("Select a document to inspect", filenames, key="inspector_select")

    if not selected:
        return

    doc = next((d for d in batch_result.documents if d.filename == selected), None)
    if doc is None:
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Document Information**")
        st.markdown(f"- **Filename:** {doc.filename}")
        st.markdown(
            f"- **AI Authorship Risk:** {format_score(doc.ai_risk_score)} / 100"
            if doc.ai_risk_score is not None
            else "- **AI Authorship Risk:** N/A"
        )
        st.markdown(f"- **Classification:** {doc.classification}")
        st.markdown(f"- **Confidence:** {doc.confidence}")
        st.markdown(f"- **Word Count:** {doc.word_count:,}")
        st.markdown(f"- **Page Count:** {doc.page_count}")
        st.markdown(f"- **Status:** {doc.status.replace('_', ' ').title()}")
        if doc.error:
            st.markdown(f"- **Error:** {doc.error}")

    with col2:
        st.markdown("**Detection Signals**")

        signals = [
            ("Transformer Classifier",  doc.transformer_signal),
            ("Stylometric Analysis",    doc.stylometric_signal),
            ("Statistical Analysis",    doc.statistical_signal),
            ("Repetition Analysis",     doc.repetition_signal),
            ("Sentence Uniformity",     doc.sentence_uniformity_signal),
        ]

        for label, val in signals:
            if val is None:
                val_str = "N/A (model not configured)"
                bar_width = 0
            else:
                val_str = f"{val:.0f}"
                bar_width = int(val)

            st.markdown(
                f"""
                <div class="signal-row">
                    <span class="signal-label">{label}</span>
                    <div class="signal-bar-bg">
                        <div class="signal-bar-fill" style="width:{bar_width}%"></div>
                    </div>
                    <span class="signal-value">{val_str}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Detected patterns
    if doc.detected_patterns:
        st.markdown("**Detected Indicators**")
        st.markdown(
            '<div class="disclaimer">'
            + "".join(f"<p style='margin:0.2rem 0'>• {p}</p>" for p in doc.detected_patterns)
            + "</div>",
            unsafe_allow_html=True,
        )

    # Linguistic stats
    with st.expander("Linguistic Statistics", expanded=False):
        stats_cols = st.columns(3)
        with stats_cols[0]:
            st.metric("Sentences", doc.sentence_count)
            st.metric("Paragraphs", doc.paragraph_count)
            st.metric("Vocabulary Size", doc.vocabulary_size)
        with stats_cols[1]:
            st.metric("Avg Sentence Length", f"{doc.avg_sentence_length:.1f} words")
            st.metric("Sentence Length Std", f"{doc.std_sentence_length:.1f}")
        with stats_cols[2]:
            st.metric("Type-Token Ratio", f"{doc.type_token_ratio:.3f}")
            if doc.transformer_signal is not None:
                st.metric("Transformer Segments", doc.transformer_segment_count)
                st.metric("Segments Flagged", f"{doc.transformer_pct_flagged:.0f}%")


# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------
def render_disclaimer():
    st.markdown("""
    <div class="disclaimer">
    <b>Important:</b> AI Authorship Risk Scores are screening signals, not definitive proof of AI authorship.
    This system can produce false positives and false negatives. Short documents are especially unreliable.
    AI-generated text can be edited to evade detection, and some human writing styles can resemble AI output.
    Use these results as one input in a broader review process — not as a final determination.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # Header
    st.markdown("""
    <div class="app-header">
        <h1>🔍 PDF AI Authorship Analyzer</h1>
        <p>Batch text-based AI authorship analysis · V1 · Text extraction only (no OCR)</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar settings
    settings = render_sidebar()

    # ---------------------------------------------------------------------------
    # Upload section
    # ---------------------------------------------------------------------------
    st.markdown('<div class="section-header">📁 Upload PDF Documents</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag and drop PDF files here, or click to browse",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
        help="Text-based PDFs only. Scanned PDFs without selectable text will be marked as 'No extractable text'.",
        label_visibility="collapsed",
    )

    if uploaded:
        st.markdown(f"**{len(uploaded)} document{'s' if len(uploaded) != 1 else ''} selected**")

        # Validate files
        invalid_files = []
        for f in uploaded:
            ok, msg = validate_uploaded_file(f.name, f.size)
            if not ok:
                invalid_files.append(msg)

        if invalid_files:
            for msg in invalid_files:
                st.error(msg)

        # Show file list
        with st.expander(f"Show file list ({len(uploaded)} files)", expanded=False):
            for f in uploaded:
                st.markdown(f"- `{f.name}` ({f.size / 1024:.1f} KB)")

    # ---------------------------------------------------------------------------
    # Analyze button
    # ---------------------------------------------------------------------------
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        analyze_clicked = st.button(
            "🔍 Analyze Documents",
            type="primary",
            disabled=not uploaded,
            key="btn_analyze",
        )

    # ---------------------------------------------------------------------------
    # Run analysis
    # ---------------------------------------------------------------------------
    if analyze_clicked and uploaded:
        from core.batch_processor import process_batch
        from core.scoring import ScoringEngine

        # Validate weights
        weights = settings.get_weights()
        weights_valid, weights_msg = validate_weights(weights)
        if not weights_valid:
            st.warning(f"⚠️ {weights_msg} — weights will be normalized.")

        # Load model (cached)
        with st.spinner("Initializing detection model..."):
            detector = get_detector(settings.model_name, settings.max_segment_tokens)

        if not detector.available:
            st.warning(
                "⚠️ Transformer model is not available. "
                "The Transformer signal will be excluded from scoring. "
                f"Check MODEL_NAME in settings (currently: `{settings.model_name}`)."
            )

        scoring_engine = ScoringEngine(weights=settings.get_weights())

        # Prepare files
        file_pairs = [(f.name, f.read()) for f in uploaded]

        # Progress display
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        def update_progress(completed: int, total: int, current_filename: str):
            pct = int(completed / total * 100)
            progress_bar.progress(pct)
            progress_placeholder.markdown(
                f"**Analyzing documents...** {completed} / {total} completed"
            )
            status_placeholder.caption(f"Processing: `{current_filename}`")

        # Run batch
        t0 = time.time()
        batch_result = process_batch(
            uploaded_files=file_pairs,
            detector=detector,
            scoring_engine=scoring_engine,
            min_word_count=settings.min_word_count,
            max_workers=settings.max_workers,
            progress_callback=update_progress,
        )
        elapsed = time.time() - t0

        # Clear progress UI
        progress_bar.empty()
        progress_placeholder.empty()
        status_placeholder.empty()

        st.success(f"✅ Analysis complete — {len(batch_result.documents)} documents processed in {seconds_to_human(elapsed)}.")

        # Store in session state
        st.session_state["batch_result"] = batch_result

    # ---------------------------------------------------------------------------
    # Display results
    # ---------------------------------------------------------------------------
    if "batch_result" in st.session_state:
        batch_result = st.session_state["batch_result"]

        st.markdown("---")

        # Dashboard
        render_dashboard(batch_result)

        # Charts
        render_charts(batch_result)

        # Results table
        render_results_table(batch_result)

        # Individual inspector
        render_individual_inspector(batch_result)

        # Excel export
        st.markdown('<div class="section-header">📥 Export Results</div>', unsafe_allow_html=True)

        from core.excel_exporter import export_to_excel

        try:
            excel_bytes = export_to_excel(batch_result)
            st.download_button(
                label="⬇️ Export Results to Excel",
                data=excel_bytes,
                file_name="AI_Authorship_Analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="btn_export_excel",
                type="secondary",
            )
        except Exception as e:
            st.error(f"Excel export failed: {e}")
            logger.error(f"Excel export error: {e}", exc_info=True)

        render_disclaimer()


if __name__ == "__main__":
    main()
