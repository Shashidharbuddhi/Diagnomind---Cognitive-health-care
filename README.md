# DiagnoMind (RPC Prototype)

DiagnoMind is a retrieval-augmented clinical reasoning prototype with:
- A **FastAPI backend** for case retrieval and reasoning generation
- A **React + Vite frontend** for entering notes/vitals and viewing results
- A **MIMIC-IV preprocessing + embedding + FAISS indexing pipeline**

## What It Does

Given patient notes and vitals, the backend:
1. Embeds the patient summary with **Bio_ClinicalBERT**
2. Retrieves similar historical cases from a **FAISS HNSW index**
3. Generates structured diagnostic reasoning using **FLAN-T5**
4. Returns retrieved cases, distances, and clinician-style reasoning text

## Project Structure

```text
RPC/
├── backend/
│   ├── main.py                         # FastAPI app (/status, /query)
│   ├── reasoner.py                     # FLAN-T5 reasoning pipeline
│   ├── preprocess.py                   # Build cases.csv from MIMIC-IV
│   ├── embeddings.py                   # Build case_embeddings.npy
│   ├── build_index.py                  # Build faiss_index.idx
│   ├── evaluate_confusion_matrix.py    # Retrieval evaluation
│   ├── requirements.txt
│   └── data/
└── frontend/
    ├── src/App.jsx                     # Main UI
    ├── package.json
    └── ...
```

## Prerequisites

- Python 3.10+ recommended
- Node.js 18+ recommended
- Access to MIMIC-IV files (only if rebuilding data from scratch)
- Enough RAM/storage for model downloads (`Bio_ClinicalBERT`, `flan-t5-large`)

## Quick Start

### 1) Start Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Backend runs on: `http://localhost:8000`

Health check:
```bash
curl http://localhost:8000/status
```

### 2) Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on: `http://localhost:5173`

## API Usage

### `POST /query`

Example:

```bash
curl -X POST "http://localhost:8000/query?top_k=5" \
  -H "Content-Type: application/json" \
  -d '{
    "subject_id": 0,
    "hadm_id": 0,
    "vitals": {"heart_rate": 110, "bp_systolic": 90, "spo2": 88},
    "notes": "Patient presents with shortness of breath and chest tightness."
  }'
```

Returns JSON with:
- `patient_summary`
- `retrieved_cases`
- `distances`
- `reasoning`

### `GET /query`

```bash
curl "http://localhost:8000/query?notes=Chest%20pain&vitals=hr=120,bp=90&top_k=5"
```

## Data Pipeline (Optional Rebuild)

Use this only if you want to regenerate `backend/data/*`.

### Step 1: Build `cases.csv`

Edit `backend/preprocess.py`:
- Set `MIMIC_IV_ROOT` to your local MIMIC-IV directory

Then run:

```bash
cd backend
python preprocess.py
```

### Step 2: Generate Embeddings

```bash
python embeddings.py
```

### Step 3: Build FAISS Index

```bash
python build_index.py
```

## Evaluation

Run retrieval quality evaluation:

```bash
cd backend
python evaluate_confusion_matrix.py --k 3 --top-labels 15
```

Outputs are written to `backend/data/evaluation/`:
- `predictions.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `metrics.json`

## Known Notes

- `backend/main.py` currently formats retrieved cases using fields like `notes_summary`, `vitals_summary`, and `dx_codes`.
- The generated `cases.csv` from `preprocess.py` stores a single `case_text` field instead.
- The app still runs, but retrieved case summaries may be sparse unless those columns are aligned.

## Troubleshooting

- If you get `FAISS index not found`, run:
  - `python preprocess.py`
  - `python embeddings.py`
  - `python build_index.py`
- If model load is slow on first request, that is expected (Hugging Face model download + warmup).
- If frontend cannot call backend, ensure backend is running on port `8000` and CORS allows `localhost:5173`.
