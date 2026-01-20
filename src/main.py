python
import json
import streamlit as st
import faiss
import numpy as np
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer

from src.config import *
from src.index import load_prebuilt_index, process_pdfs
from src.pdf_utils import pdf_to_text, split_chunks
from src.api_client import query_ollama, check_ollama_health

def answer_query(query: str, index: faiss.IndexFlatIP, meta: List[dict], embedder: SentenceTransformer) -> str:
    """Retrieve top-k chunks and build prompt for Doctor Apollyon."""
    try:
        q_emb = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        _, I = index.search(q_emb, k=5)  # top-5 most similar
        
        context_parts = []
        for idx in I[0]:
            if idx < 0 or idx >= len(meta):
                continue
                
            src = meta[idx]
            pdf_path = BASE_DIR / src["source"]
            
            if not pdf_path.exists():
                continue
            
            raw = pdf_to_text(pdf_path)
            chunks = split_chunks(raw)
            
            if src["chunk_id"] < len(chunks):
                chunk = chunks[src["chunk_id"]]
                context_parts.append(f"[{src['pdf_name']} – chunk {src['chunk_id']}] {chunk}")
        
        if not context_parts:
            return "⚠️ No relevant context found in the knowledge base."
        
        context = "\n---\n".join(context_parts)
        
        prompt = (
            f"You are Doctor Apollyon, a seasoned cyber-mage who teaches ethical hacking.\n"
            f"Use the following excerpts from your grim grimoire (the supplied PDFs) as absolute truth:\n\n"
            f"{context}\n\n"
            f"Answer the user's question with technical depth, precision, and your characteristic dark wit.\n"
            f"Question: {query}\n\n"
            f"Answer concisely but thoroughly:"
        )
        
        return prompt
        
    except Exception as e:
        return f"Error building query: {str(e)}"

def initialize_session():
    """Initialize or load the knowledge base."""
    if "initialized" not in st.session_state:
        st.session_state.update({
            "index": None,
            "meta": [],
            "embedder": None,
            "initialized": False
        })
        
        # Try to load existing index
        idx, meta = load_prebuilt_index()
        if idx is not None:
            st.session_state.update({
                "index": idx,
                "meta": meta,
                "embedder": SentenceTransformer(EMBED_MODEL),
                "initialized": True
            })
            st.success("📚 Loaded existing knowledge base")

def main():
    st.set_page_config(
        page_title="🦴 Doctor Apollyon",
        page_icon="💀",
        layout="centered"
    )
    
    # CSS theme
    seal_url = "https://i.etsystatic.com/40811848/r/il/c670c3/5283635364/il_1080xN.5283635364_ki78.jpg"
    st.markdown(
        f"""
        <style>
        .main {{
            background-image: url('{seal_url}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-color: #0b0b0b;
            color: #e0e0e0;
        }}
        .stButton > button {{
            background-color: #2c0a0a;
            color: #ff4d4d;
            border: 2px solid #ff1a1a;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
        }}
        .stButton > button:hover {{
            background-color: #4a0f0f;
            box-shadow: 0 0 15px #ff0000;
        }}
        .stTextInput > div > div > input {{
            background-color: #1a1a1a;
            color: #ffdddd;
            border: 1px solid #444;
            border-radius: 5px;
        }}
        h1, h2, h3 {{
            color: #ff6666 !important;
            text-shadow: 0 0 10px #ff0000;
        }}
        .stAlert {{
            background-color: #2a0a0a;
            color: #ffcccc;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.title(":skull_and_crossbones: Doctor Apollyon")
    st.subheader("Ask the dark doctor anything about ethical hacking & the occult digital arts")
    
    # Initialize
    initialize_session()
    
    # Health check
    if not st.session_state.get("health_checked"):
        with st.spinner("Checking Ollama connection..."):
            if check_ollama_health():
                st.session_state.health_checked = True
            else:
                st.error("⚠️ Ollama is not ready. Please check the sidebar for setup instructions.")
                with st.sidebar:
                    st.markdown("""
                    ### 🔧 Ollama Setup Required
                    
                    1. **Install Ollama**: https://ollama.ai
                    2. **Start Ollama**: `ollama serve`
                    3. **Pull a model**: `ollama pull llama2-uncensored`
                    4. **Refresh** this page
                    
                    You can change the model in `.env` or `src/config.py`
                    """)
                return
    
    # ── PDF Upload ────────────────────────────────────────────────
    st.markdown("### 📂 Feed the Abyss")
    uploaded = st.file_uploader(
        "Drop new PDFs here (they'll be copied into abyss_feed/):",
        type=["pdf"],
        accept_multiple_files=True,
    )
    
    if uploaded:
        PDF_ROOT.mkdir(parents=True, exist_ok=True)
        for uf in uploaded:
            dest = PDF_ROOT / uf.name
            dest.write_bytes(uf.getbuffer())
        st.success(f"✅ Copied {len(uploaded)} PDF(s) into **abyss_feed/**")
        st.info("Click 'Process PDFs' to update the knowledge base")
    
    # ── Build Index Button ────────────────────────────────────────
    if st.button("⚙️ Process PDFs (build/re-build index)", type="primary"):
        try:
            with st.spinner("Converting PDFs to dark knowledge..."):
                idx, meta = process_pdfs()
                st.session_state.update({
                    "index": idx,
                    "meta": meta,
                    "embedder": SentenceTransformer(EMBED_MODEL),
                    "initialized": True
                })
            st.success("✅ Knowledge base is ready! The doctor is in.")
            st.balloons()
        except Exception as e:
            st.error(f"❌ Failed to process PDFs: {str(e)}")
    
    # ── Query Interface ───────────────────────────────────────────
    if st.session_state.get("initialized") and st.session_state.get("index") is not None:
        st.markdown("### 💀 Consult Doctor Apollyon")
        
        # Show knowledge base stats
        st.caption(f"📚 Knowledge base: {len(st.session_state.meta)} chunks from {len(set(m['source'] for m in st.session_state.meta))} PDFs")
        
        user_q = st.text_input("Whisper your query into the void:", placeholder="e.g., What is a buffer overflow attack?")
        
        if st.button("🔮 Summon Apollyon"):
            if not user_q.strip():
                st.warning("The void is silent… type a question first.")
            else:
                with st.spinner("Doctor Apollyon is consulting the grimoire..."):
                    try:
                        # Build prompt
                        prompt = answer_query(
                            user_q,
                            st.session_state.index,
                            st.session_state.meta,
                            st.session_state.embedder
                        )
                        
                        if prompt.startswith("Error"):
                            st.error(prompt)
                        else:
                            # Get answer from Ollama
                            answer = query_ollama(prompt)
                            st.markdown("**🗣️ Doctor Apollyon replies:**")
                            st.markdown(f"```{answer}```")
                            
                            # Add to history
                            if "history" not in st.session_state:
                                st.session_state.history = []
                            st.session_state.history.append((user_q, answer))
                            
                    except Exception as e:
                        st.error(f"❌ Consultation failed: {str(e)}")
        
        # Show history
        if st.session_state.get("history"):
            with st.expander("📜 Consultation History"):
                for i, (q, a) in enumerate(st.session_state.history[-5:], 1):
                    st.markdown(f"**Q{i}:** {q}")
                    st.markdown(f"**A{i}:** {a[:100]}...")
    
    else:
        st.info("🕸️ No knowledge base loaded. Upload PDFs and press **Process PDFs** to begin.")
    
    # ── Sidebar Info ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.caption(f"**Embedding Model:** {EMBED_MODEL}")
        st.caption(f"**Ollama Model:** {OLLAMA_MODEL}")
        st.caption(f"**PDF Directory:** {PDF_ROOT}")
        
        st.markdown("### 📖 Grimoire Contents")
        if st.session_state.get("meta"):
            pdf_list = sorted(set(m["pdf_name"] for m in st.session_state.meta))
            for pdf in pdf_list:
                st.caption(f"📄 {pdf}")
        else:
            st.caption("Empty abyss")

if __name__ == "__main__":
    main()
