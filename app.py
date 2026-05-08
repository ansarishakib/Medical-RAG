"""
app.py — Streamlit web UI for the Medical RAG System
Run: streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
from ingest import ingest
from rag_pipeline import MedicalRAG

st.set_page_config(page_title="Medical RAG — Clinical Decision Support", page_icon="🏥", layout="wide")

st.markdown("""
<style>
.case-card { background: #f8fafc; border-left: 4px solid #3b82f6; border-radius: 6px; padding: 12px 16px; margin-bottom: 10px; font-size: 0.85rem; }
.similarity-badge { background: #dbeafe; color: #1e40af; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/google-logo.png", width=60)
    st.title("⚙️ Settings")

    api_key = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
    st.divider()
    top_k = st.slider("Similar cases to retrieve", min_value=2, max_value=8, value=4)
    st.divider()
    
    if st.button("🔄 (Re)Build Knowledge Base", use_container_width=True):
        with st.spinner("Embedding prescriptions into ChromaDB..."):
            try:
                ingest()
                st.success("✅ Knowledge base built!")
                st.session_state["kb_ready"] = True
            except Exception as e:
                st.error(f"Error: {e}")

    if os.path.exists("chroma_db"):
        st.session_state.setdefault("kb_ready", True)
        st.info("✅ Knowledge base found")
    else:
        st.warning("⚠️ Build the knowledge base first")

st.title("🏥 Medical RAG — Clinical Decision Support")
st.markdown("Upload a prescription (PDF/Image) to parse it, OR enter a patient case manually. The system will retrieve similar past cases and generate treatment recommendations.")

# Initialize RAG globally if key exists so we can use vision parsing before clicking generate
rag = None
if api_key and st.session_state.get("kb_ready"):
    try:
        rag = MedicalRAG(api_key=api_key)
    except Exception as e:
        st.sidebar.error(f"Failed to load RAG: {e}")

# ── File Upload & Parsing ──
st.markdown("### 1. Input Patient Data")
upload_col, text_col = st.columns([1, 1])

with upload_col:
    uploaded_file = st.file_uploader("Upload Prescription (Image or PDF)", type=["png", "jpg", "jpeg", "pdf"])
    if uploaded_file and rag:
        if st.button("📄 Parse Uploaded Document", use_container_width=True):
            with st.spinner("Analyzing document with Gemini Vision..."):
                try:
                    mime_type = uploaded_file.type
                    parsed_text = rag.parse_prescription(uploaded_file.read(), mime_type)
                    st.session_state["query"] = parsed_text
                    st.success("Document parsed! Extracted text populated.")
                except Exception as e:
                    st.error(f"Failed to parse document: {e}")
    elif uploaded_file and not rag:
        st.warning("Please enter your API Key and build the knowledge base to enable parsing.")

with text_col:
    query = st.text_area(
        "Patient Case / Extracted Summary",
        value=st.session_state.get("query", ""),
        height=150,
        placeholder="Upload a document to auto-fill, or type manually e.g. 55-year-old male with hypertension...",
    )

st.markdown("### 2. Clinical Support")
generate_btn = st.button("🔍 Generate Recommendations", type="primary", use_container_width=True)

# ── Generate Response ──
if generate_btn:
    if not api_key:
        st.error("⚠️ Please enter your Gemini API key.")
        st.stop()
    if not query.strip():
        st.error("⚠️ Please enter a patient query or parse a document.")
        st.stop()
    if not rag:
        st.error("⚠️ System not initialized properly. Check API key and Knowledge Base.")
        st.stop()

    with st.spinner("🔍 Retrieving similar cases..."):
        cases = rag.retrieve(query, top_k=top_k)

    st.divider()
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown("## 🤖 AI Clinical Response")
        response_placeholder = st.empty()
        full_response = ""
        with st.spinner("Generating..."):
            try:
                for chunk in rag.stream_generate(query, top_k=top_k):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Generation error: {e}")

    with right_col:
        st.markdown("## 📚 Retrieved Similar Cases")
        for i, case in enumerate(cases, 1):
            sim_pct = int(case["similarity"] * 100)
            meta = case["metadata"]
            st.markdown(
                f"""<div class="case-card">
                <b>Case {i}</b> &nbsp; <span class="similarity-badge">{sim_pct}% match</span><br><br>
                <b>🩺 Diagnosis:</b> {meta.get('diagnosis', '—')}<br>
                <b>💊 Medicines:</b> {meta.get('medicines', '—')}<br>
                <b>📏 Dosage:</b> {meta.get('dosage', '—')}<br>
                <b>👤 Age/Gender:</b> {meta.get('age', '—')} / {meta.get('gender', '—')}
                </div>""", unsafe_allow_html=True
            )