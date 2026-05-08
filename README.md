<<<<<<< HEAD
# 🏥 Medical Prescription RAG System

A Retrieval-Augmented Generation (RAG) system that helps doctors get **drug recommendations** and **treatment summaries** by finding similar past prescription cases and generating clinical advice using Claude AI.

---

## 📁 Project Structure

```
medical-rag/
├── data/
│   └── structured_output01.csv   ← Your prescription dataset
├── chroma_db/                     ← Auto-created after ingest
├── ingest.py                      ← Build the vector knowledge base
├── rag_pipeline.py                ← Core RAG logic (retrieve + generate)
├── app.py                         ← Streamlit web UI
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx
```

Or create a `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx
```

### 3. Build the knowledge base (run once)

```bash
python ingest.py
```

This will:
- Clean and normalize the 86 prescription records
- Embed each record using `all-MiniLM-L6-v2` (free, runs locally)
- Store embeddings in ChromaDB (`chroma_db/` folder)

### 4. Launch the web app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🧠 How It Works

```
Doctor's Query
     │
     ▼
[Embedding Model]  ← all-MiniLM-L6-v2 (local, free)
     │
     ▼
[ChromaDB Vector Search]  ← cosine similarity over 86 prescription records
     │
     ▼
Top-K Similar Cases (with metadata)
     │
     ▼
[Claude claude-opus-4-5]  ← RAG prompt: query + retrieved cases
     │
     ▼
┌─────────────────────────────────────┐
│  TREATMENT SUMMARY                  │
│  DRUG RECOMMENDATIONS               │
│  PRECAUTIONS & ADVICE               │
└─────────────────────────────────────┘
```

---

## 💡 Example Queries

| Query | Expected output focus |
|---|---|
| "60yr male, hypertension + CHF" | Cardiac meds (Furosemide, ACE inhibitors) |
| "Child with acute bronchospasm" | Bronchodilators, inhaled steroids |
| "28yr female, bacterial infection" | Azithromycin / Amoxicillin based |
| "Fungal infection, immunocompromised" | Amphotericin B based antifungals |

---

## 🔧 Customization

### Add more data
Replace `data/structured_output01.csv` with a larger CSV (same column format), then re-run `python ingest.py`.

### Change the LLM
In `rag_pipeline.py`, update the model name:
```python
model="claude-opus-4-5"        # highest quality
model="claude-sonnet-4-5"      # faster, cheaper
```

### Adjust retrieval
In `rag_pipeline.py`, change:
```python
TOP_K = 4   # increase for more context, decrease for speed
```

---

## ⚕️ Disclaimer

This is a clinical decision-support tool. All AI-generated recommendations must be reviewed and verified by a qualified physician before prescribing. This system does not replace professional medical judgment.
=======
# Medical-RAG
>>>>>>> 015b75dc483fa38fe4d7fe0e6069d8920e1e293b
