"""
ingest.py — Load cleaned CSV → embed → store in ChromaDB
"""
import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# Explicitly use the new segmented API to avoid warnings
os.environ["CHROMA_API_IMPL"] = "chromadb.api.segment.SegmentAPI"

DATA_PATH = "data/cleaned_prescriptions.csv"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "prescriptions"
EMBED_MODEL = "all-MiniLM-L6-v2"

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop rows without diagnosis or medicines to keep DB quality high
    df = df.dropna(subset=["diagnosis", "medicines"], how="all").reset_index(drop=True)
    df.fillna("Not specified", inplace=True)
    return df

def build_document(row: pd.Series) -> str:
    """Convert prescription row into rich text chunk for vector embedding."""
    age_str = str(row.get('age_in_years', 'Not specified'))
    return (
        f"PATIENT TYPE: {row.get('patient_type', 'Unspecified')}\n"
        f"DIAGNOSIS: {row.get('diagnosis', 'Unspecified')}\n"
        f"MEDICINES: {row.get('medicines', 'Unspecified')}\n"
        f"DOSAGE: {row.get('dosage_intake', 'Unspecified')}\n"
        f"AGE: {age_str} | GENDER: {row.get('gender', 'Unspecified')}\n"
    )

def ingest():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")
        
    df = load_data(DATA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    documents, metadatas, ids = [], [], []

    for i, row in df.iterrows():
        doc = build_document(row)
        documents.append(doc)
        metadatas.append({
            "diagnosis": str(row.get("diagnosis", "")),
            "medicines": str(row.get("medicines", "")),
            "dosage": str(row.get("dosage_intake", "")),
            "gender": str(row.get("gender", "")),
            "age": str(row.get("age_in_years", "")),
        })
        ids.append(f"rx_{i:04d}")

    # Batch upsert
    BATCH = 50
    for start in range(0, len(documents), BATCH):
        collection.add(
            documents=documents[start:start+BATCH],
            metadatas=metadatas[start:start+BATCH],
            ids=ids[start:start+BATCH],
        )
    print(f"✅ Knowledge base ready — {collection.count()} records in ChromaDB")

if __name__ == "__main__":
    ingest()