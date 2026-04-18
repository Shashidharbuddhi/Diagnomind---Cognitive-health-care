# backend/embeddings.py
import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
CASES_CSV = os.path.join(DATA_DIR, "cases.csv")
EMB_OUT = os.path.join(DATA_DIR, "case_embeddings.npy")
IDX_MAP_OUT = os.path.join(DATA_DIR, "case_index_map.csv")

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mean_pool(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def embed_cases(batch_size=8):
    if not os.path.exists(CASES_CSV):
        raise FileNotFoundError(f"{CASES_CSV} not found. Run preprocess.py first.")

    print("[embeddings] Loading cases...")
    df = pd.read_csv(CASES_CSV)
    texts = df["case_text"].astype(str).tolist()

    print("[embeddings] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            input_ids = enc["input_ids"].to(device)
            att = enc["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=att, return_dict=True)
            embs = mean_pool(out.last_hidden_state, att)
            embs = embs.cpu().numpy()
            embeddings.append(embs)
            print(f"[embeddings] Embedded {i+len(batch_texts)}/{len(texts)}")

    embeddings = np.vstack(embeddings)
    np.save(EMB_OUT, embeddings)
    df[["case_id", "subject_id", "hadm_id", "admittime"]].to_csv(IDX_MAP_OUT, index=False)
    print(f"[embeddings] Saved {embeddings.shape[0]} embeddings to {EMB_OUT}")

if __name__ == "__main__":
    embed_cases()
