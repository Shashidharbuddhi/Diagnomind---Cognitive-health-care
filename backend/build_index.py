# backend/build_index.py
import os
import numpy as np
import faiss

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
EMB_PATH = os.path.join(DATA_DIR, "case_embeddings.npy")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.idx")

def build_faiss_index():
    if not os.path.exists(EMB_PATH):
        raise FileNotFoundError(f"{EMB_PATH} not found. Run embeddings.py first.")
    emb = np.load(EMB_PATH)
    d = emb.shape[1]
    print(f"[faiss] Building HNSW index for {emb.shape[0]} vectors, dim={d}")
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64
    index.add(emb.astype("float32"))
    faiss.write_index(index, INDEX_PATH)
    print(f"[faiss] Wrote index to {INDEX_PATH} (ntotal={index.ntotal})")

if __name__ == "__main__":
    build_faiss_index()
