"""
rag_pipeline.py — Gemini-powered RAG core with Multimodal Parsing
"""
import os
from google import genai
from google.genai import types
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "prescriptions"
EMBED_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
TOP_K = 4

SYSTEM_PROMPT = """You are a clinical decision-support AI assistant helping doctors.
You are given a patient query (or parsed prescription summary) and similar past prescription cases.
Provide a structured response:
## TREATMENT SUMMARY
## DRUG RECOMMENDATIONS
## PRECAUTIONS & ADVICE
Rules: Professional tone, cite retrieved cases, and state that the doctor must verify all dosages."""

class MedicalRAG:
    def __init__(self, chroma_dir: str = CHROMA_DIR, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("Gemini API Key is missing.")
            
        self.client = genai.Client(api_key=self.api_key)
        
        # Load Vector DB
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        client_db = chromadb.PersistentClient(path=chroma_dir)
        self.collection = client_db.get_collection(name=COLLECTION_NAME, embedding_function=ef)

    def parse_prescription(self, file_bytes: bytes, mime_type: str) -> str:
        """Parses an uploaded image or PDF and extracts clinical entities."""
        vision_prompt = """Analyze this prescription document. Extract and structure the following details:
        - Patient Details (Age, Gender)
        - Diagnosis / Symptoms
        - Prescribed Medicines & Dosages
        Return ONLY the extracted clinical details as a clean, concise text summary."""
        
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                vision_prompt
            ]
        )
        return response.text

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        cases = []
        # Ensure we have results before zipping
        if results and results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                cases.append({"document": doc, "metadata": meta, "similarity": round(1 - dist, 3)})
        return cases

    def stream_generate(self, query: str, top_k: int = TOP_K):
        cases = self.retrieve(query, top_k=top_k)
        context = "\n".join([f"Case {i+1}:\n{c['document']}" for i, c in enumerate(cases)])
        
        user_message = f"### DOCTOR'S QUERY / PATIENT CASE\n{query}\n\n### RETRIEVED SIMILAR CASES\n{context}"

        response_stream = self.client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
        )
        
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text