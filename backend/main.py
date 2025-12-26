# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import os
import numpy as np
import pandas as pd
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

from reasoner import generate_reasoning

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
CASES_CSV = os.path.join(DATA_DIR, "cases.csv")
EMB_PATH = os.path.join(DATA_DIR, "case_embeddings.npy")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.idx")

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
embed_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
embed_model.eval()

faiss_index = None
cases_df = None

def mean_pool(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def load_index():
    global faiss_index, cases_df
    if faiss_index is not None:
        return
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("FAISS index not found. Run build_index.py first.")
    if not os.path.exists(CASES_CSV):
        raise FileNotFoundError("cases.csv not found. Run preprocess.py first.")
    faiss_index = faiss.read_index(INDEX_PATH)
    cases_df = pd.read_csv(CASES_CSV)
    print("[main] Loaded FAISS index and cases.")

def embed_text(text: str) -> np.ndarray:
    enc = tokenizer(
        [text],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )
    input_ids = enc["input_ids"].to(device)
    att = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = embed_model(input_ids=input_ids, attention_mask=att, return_dict=True)
        embs = mean_pool(out.last_hidden_state, att)
    return embs.cpu().numpy().astype("float32")

def search_index(patient_text: str, top_k: int = 5):
    """
    Search FAISS index for similar cases.
    
    Returns:
        cases (list): List of case text summaries
        distances (list): L2 distances from query
    """
    load_index()
    
    # Embed the patient text
    query_emb = embed_text(patient_text)
    
    # Search FAISS
    distances, indices = faiss_index.search(query_emb, top_k)
    
    # Retrieve case data
    cases = []
    distances_list = []
    
    for idx, distance in zip(indices[0], distances[0]):
        if idx >= 0 and idx < len(cases_df):
            row = cases_df.iloc[idx]
            # Build case summary with patient notes and vitals
            case_summary = (
                f"SUBJECT_ID: {row.get('subject_id', 'N/A')}\n"
                f"HADM_ID: {row.get('hadm_id', 'N/A')}\n"
                f"NOTE: {row.get('notes_summary', '')}\n"
                f"VITALS: {row.get('vitals_summary', '')}\n"
                f"Primary diagnoses (ICD codes): {row.get('dx_codes', '')}"
            )
            cases.append(case_summary)
            distances_list.append(float(distance))
    
    return cases, distances_list

class QueryPayload(BaseModel):
    subject_id: Optional[int] = None
    hadm_id: Optional[int] = None
    vitals: Dict[str, float] = {}
    notes: str

app = FastAPI(title="DiagnoMind Prototype")

# 🔥 CORS FIX
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
def status():
    return {"status": "ok"}

@app.get("/query")
@app.post("/query")
def query_endpoint(
    notes: str = None,
    vitals: str = "",
    top_k: int = 5,
    payload: QueryPayload = None,
):
    """
    Query endpoint that retrieves similar cases and generates reasoning.
    
    Accepts both GET and POST:
    - GET: Query params - notes, vitals, top_k
    - POST: JSON body with QueryPayload
    """
    try:
        load_index()
        
        # Handle POST request with JSON body
        if payload is not None:
            notes = payload.notes
            vitals = ", ".join([f"{k}={v}" for k, v in payload.vitals.items()]) if payload.vitals else ""
        
        # Validate notes parameter
        if not notes:
            raise ValueError("'notes' parameter is required")
        
        # Build patient text summary
        patient_text = f"NOTES: {notes}\nVITALS: {vitals}"
        
        # Retrieve similar cases using embeddings
        cases, distances = search_index(patient_text, top_k=top_k)
        
        # Generate clinical reasoning using FLAN T5
        reasoning = generate_reasoning(patient_text, cases)
        
        return {
            "status": "success",
            "patient_summary": patient_text,
            "retrieved_count": len(cases),
            "retrieved_cases": cases,
            "reasoning": reasoning,
            "distances": distances,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during reasoning: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
