# PDF AI Authorship Analyzer — V1

A Python Streamlit application that analyzes text-based PDF documents for indicators associated with AI-generated writing, assigns an **AI Authorship Risk Score (0–100)**, and exports results to a formatted Excel file.

> **Important**: This tool provides screening signals — not definitive proof of AI authorship. Scores represent the strength of detected statistical patterns, not a guaranteed probability. See [Limitations](#limitations) below.

---

## Features

- Upload multiple PDF documents in one batch
- Text extraction via PyMuPDF — **no OCR, no image processing**
- **Hybrid detection** combining 5 independent signals:
  - Transformer classifier (configurable, uses `roberta-base-openai-detector` by default)
  - Stylometric analysis (vocabulary richness, character n-gram entropy, type-token ratio)
  - Statistical analysis (sentence length distribution, kurtosis, skewness)
  - Repetition analysis (bigram/trigram repetition ratios)
  - Sentence uniformity (coefficient of variation, length consistency)
- AI Authorship Risk Score: 0–100
- Classification: High / Elevated / Uncertain / Relatively Low / Low
- Confidence level: High / Medium / Low
- Summary dashboard with metric cards
- Score distribution charts (histogram + per-document bar chart)
- Sortable results table
- Individual document inspector with detected patterns
- **Export to Excel** (.xlsx) with 3 sheets: Summary, Detailed Analysis, Errors
- Privacy-first: all processing is local; no text is sent to external APIs

---

## Architecture

```
pdf-ai-analyzer/
├── app.py                     # Streamlit UI (no business logic)
├── requirements.txt
├── .env.example               # Configuration template
│
├── config/
│   └── settings.py            # All settings, thresholds, weights
│
├── core/
│   ├── pdf_extractor.py       # PyMuPDF text extraction
│   ├── text_preprocessor.py   # Cleaning, segmentation, tokenization
│   ├── feature_extractor.py   # Linguistic & stylometric features
│   ├── ai_detector.py         # AITextDetector class (transformer)
│   ├── scoring.py             # Hybrid scoring engine
│   ├── batch_processor.py     # Concurrent batch pipeline
│   └── excel_exporter.py      # Excel export with formatting
│
├── models/
│   ├── model_loader.py        # Cached HuggingFace model loading
│   └── README.md              # Model documentation
│
├── utils/
│   ├── logging_config.py
│   ├── validators.py
│   └── helpers.py
│
├── data/
│   ├── results/               # Session-only; not stored permanently
│   └── cache/
│
└── tests/
    ├── test_pdf_extraction.py
    ├── test_preprocessing.py
    ├── test_features.py
    ├── test_scoring.py
    └── test_excel_export.py
```

---

## Installation

### Requirements

- **Python 3.12** (recommended; 3.10+ should work)
- pip

### Step 1 — Clone / navigate to the project

```bash
cd "d:\EEAAO\Projects\AI Detection\pdf-ai-analyzer"
```

### Step 2 — Create and activate a virtual environment

```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (CMD)
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyTorch**: `requirements.txt` installs the CPU version by default (via pip). For GPU support, install the correct CUDA build of PyTorch from https://pytorch.org/get-started/locally/ **before** running `pip install -r requirements.txt`.

### Step 4 — Configure the model

Copy `.env.example` to `.env`:

```bash
copy .env.example .env
```

Edit `.env` and set `MODEL_NAME`:

```env
MODEL_NAME=roberta-base-openai-detector
```

The first time you run the app, the model (~500 MB) will be downloaded automatically from Hugging Face and cached locally.

---

## Model Configuration

| Setting | Default | Description |
|---|---|---|
| `MODEL_NAME` | `roberta-base-openai-detector` | Hugging Face model ID |
| `MIN_WORD_COUNT` | `300` | Minimum words for full analysis |
| `MAX_SEGMENT_TOKENS` | `512` | Max tokens per transformer segment |
| `TRANSFORMER_WEIGHT` | `0.40` | Weight of transformer signal |
| `STYLOMETRIC_WEIGHT` | `0.20` | Weight of stylometric signal |
| `STATISTICAL_WEIGHT` | `0.15` | Weight of statistical signal |
| `REPETITION_WEIGHT` | `0.10` | Weight of repetition signal |
| `SENTENCE_UNIFORMITY_WEIGHT` | `0.15` | Weight of uniformity signal |

All weights are configurable in the Streamlit sidebar as well (no restart required).

### Recommended Models

| Model | Notes |
|---|---|
| `roberta-base-openai-detector` | Default. OpenAI GPT-2 detector. Good for GPT-2 family. |
| `Hello-SimpleAI/chatgpt-detector-roberta` | Fine-tuned for ChatGPT text. |
| `andreas122001/roberta-academic-detector` | Fine-tuned on academic text. |

See [`models/README.md`](models/README.md) for more details.

---

## Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Usage

### Analyzing Documents

1. Open the app in your browser.
2. **Upload PDFs** by dragging and dropping or clicking "Browse files".
3. Review the file list.
4. Click **"🔍 Analyze Documents"**.
5. Wait for the progress bar to complete.
6. View results in the dashboard, charts, and table.

### Inspecting Individual Documents

- Scroll to **Individual Document Inspection**.
- Select a PDF from the dropdown.
- View the AI risk score, confidence, detection signals (with visual bars), and detected patterns.
- Expand **Linguistic Statistics** for detailed metrics.

### Exporting to Excel

- Click **"⬇️ Export Results to Excel"**.
- Download `AI_Authorship_Analysis.xlsx`.

The Excel file contains:

| Sheet | Contents |
|---|---|
| Summary | PDF Name, Risk Score, Classification, Confidence, Word Count, Page Count, Status |
| Detailed Analysis | All signals, linguistic statistics, detected patterns |
| Errors | Failed/encrypted PDFs with error messages |

The file includes: bold headers, freeze row, autofilter, conditional color formatting on Risk Score, and sensible column widths.

---

## Score Interpretation

| Score | Classification |
|---|---|
| 81–100 | High AI-authorship indicators |
| 61–80 | Elevated AI-authorship indicators |
| 41–60 | Uncertain / mixed signals |
| 21–40 | Relatively low AI-authorship indicators |
| 0–20 | Low AI-authorship indicators |
| N/A | Insufficient text (< 300 words) |

---

## Testing

Install dependencies and run:

```bash
pytest tests/ -v
```

Run a specific test file:

```bash
pytest tests/test_scoring.py -v
```

Tests cover: PDF extraction, text preprocessing, feature extraction, scoring engine, and Excel export. Unit tests use synthetic in-memory PDFs and text — no external data required.

---

## Advanced: Sidebar Settings

| Setting | Description |
|---|---|
| Detection Model | HuggingFace model ID (change and re-run to reload) |
| Minimum Word Count | Documents with fewer words are marked "Insufficient text" |
| Advanced Scoring Weights | Sliders for all 5 signal weights (auto-normalized) |

---

## Limitations

- **Text-only (V1)**: Scanned PDFs without selectable text cannot be analyzed. They will be marked "No extractable text".
- **Model generalization**: The default model was trained on GPT-2 output. It may not generalize well to GPT-4, Claude, Gemini, or heavily edited AI text.
- **False positives**: Some human writing styles (e.g., highly formal academic writing, technical reports) can resemble AI-generated text and may receive elevated scores.
- **False negatives**: AI-generated text that has been edited or paraphrased may score low.
- **Short documents**: Documents under ~500 words produce less reliable scores regardless of content.
- **Language**: This system is designed for English text. Performance on other languages is undefined.
- **No scientific validation**: The scoring weights are a configurable baseline, not academically validated parameters.

---

## Ethical Considerations

AI-authorship detection is probabilistic and error-prone. This tool is intended as one input in a broader review process.

- **Do not use this score as definitive proof** of AI authorship or academic dishonesty.
- **Do not take punitive action** based solely on this score.
- A score of 87/100 does not mean "87% certain this is AI-written." It means the system detected elevated AI-authorship signals relative to its heuristics.
- Human reviewers must exercise independent judgment.
- Provide the accused with an opportunity to respond before any conclusions are drawn.

---

## Privacy

- All PDF processing happens locally on your machine.
- No document text is sent to external APIs.
- No uploaded PDFs are stored permanently. Data exists only during the current Streamlit session.
- Model weights are downloaded from Hugging Face and cached locally (`~/.cache/huggingface`).

---

## Future (V2 Roadmap — Not Implemented)

- Custom fine-tuned classifier training
- Multilingual detection
- Document comparison
- Batch history and session storage
- OCR / scanned PDF support
- REST API
- Cloud deployment
- User accounts

---

## License

Internal use only. Not for distribution without authorization.
