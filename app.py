import streamlit as st
import os
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv

# Load .env explicitly from project folder
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from document_processor import extract_text, get_document_stats, chunk_text
from embedder import TarkaEmbedder
from llm_handler import ask_gemini, check_api_key

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TarkaRAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d1b2a 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(99, 102, 241, 0.3);
}
.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #a5b4fc;
    margin: 0;
}
.main-subtitle { color: #94a3b8; font-size: 0.95rem; margin-top: 0.3rem; }
.model-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    margin-right: 6px;
}
.stat-card {
    background: #0f172a;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #a5b4fc;
    line-height: 1;
}
.stat-label {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.chat-bubble-user {
    background: linear-gradient(135deg, #312e81, #1e1b4b);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px 12px 4px 12px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    color: #e0e7ff;
}
.chat-bubble-ai {
    background: #0f172a;
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 12px 12px 12px 4px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    color: #cbd5e1;
    line-height: 1.7;
}
.chunk-card {
    background: #0f172a;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.83rem;
    color: #94a3b8;
    font-family: 'Space Mono', monospace;
}
.score-pill {
    background: rgba(99,102,241,0.2);
    color: #a5b4fc;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
}
.status-ok  { color: #34d399; font-size: 0.85rem; }
.status-err { color: #f87171; font-size: 0.85rem; }
.step-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    padding: 0.5rem 1.5rem;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #818cf8, #6366f1);
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "embedder": TarkaEmbedder(),
        "doc_stats": None,
        "doc_text": None,
        "chunks": [],
        "indexed": False,
        "chat_history": [],
        "current_file": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-title">🔍 TarkaRAG</div>
    <div class="main-subtitle">
        Document Intelligence powered by
        <span class="model-badge">Tarka-Embedding-150M</span>
        <span class="model-badge">Gemini 2.5 Flash</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    api_ok, api_msg = check_api_key()
    if api_ok:
        st.markdown('<div class="status-ok">✅ Gemini API connected</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-err">❌ {api_msg}</div>', unsafe_allow_html=True)
        st.info("Add your GEMINI_API_KEY to the .env file and restart.")

    # Show current API key status for debugging
    current_key = os.getenv("GEMINI_API_KEY", "")
    if current_key and current_key != "your_gemini_api_key_here":
        st.caption(f"Key loaded: {current_key[:8]}...{current_key[-4:]}")
    else:
        st.caption("No key detected in environment")

    st.divider()
    st.markdown("### 🧩 Chunking Settings")
    chunk_size  = st.slider("Chunk size (words)", 200, 1000, 500, 50)
    chunk_overlap = st.slider("Overlap (words)", 0, 200, 50, 10)
    top_k = st.slider("Top-K chunks for retrieval", 1, 10, 5)

    st.divider()
    st.markdown("### 📋 About")
    st.markdown("""
    <small style='color:#64748b;'>
    <b>Embedding:</b> Tarka-150M (768-dim)<br>
    <b>Vector DB:</b> FAISS (cosine similarity)<br>
    <b>LLM:</b> Gemini 2.5 Flash<br>
    <b>Supported:</b> PDF, DOCX, TXT
    </small>
    """, unsafe_allow_html=True)

    if st.session_state.indexed:
        st.divider()
        if st.button("🗑️ Reset & Upload New Doc"):
            st.session_state.embedder.reset()
            st.session_state.doc_stats   = None
            st.session_state.doc_text    = None
            st.session_state.chunks      = []
            st.session_state.indexed     = False
            st.session_state.chat_history = []
            st.session_state.current_file = None
            st.rerun()


# ─── Main layout ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.3], gap="large")


# ─── LEFT: Upload + Stats + Embed ────────────────────────────────────────────
with col_left:

    st.markdown('<div class="step-label">Step 01 — Upload Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your document here",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
    )

    if uploaded_file and uploaded_file.name != st.session_state.current_file:
        with st.spinner("Analysing document..."):
            try:
                file_bytes = BytesIO(uploaded_file.read())
                text, pages = extract_text(file_bytes, uploaded_file.name)
                stats = get_document_stats(text, pages, uploaded_file.name)
                st.session_state.doc_text     = text
                st.session_state.doc_stats    = stats
                st.session_state.indexed      = False
                st.session_state.current_file = uploaded_file.name
                st.session_state.chat_history = []
                st.session_state.embedder.reset()
            except Exception as e:
                st.error(f"Error reading file: {e}")

    if st.session_state.doc_stats:
        stats = st.session_state.doc_stats
        st.markdown('<div class="step-label" style="margin-top:1.5rem;">Step 02 — Document Analysis</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{stats["pages"]}</div><div class="stat-label">Pages</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{stats["words"]:,}</div><div class="stat-label">Words</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{stats["tokens"]:,}</div><div class="stat-label">Tokens</div></div>', unsafe_allow_html=True)

        c4, c5, c6 = st.columns(3)
        with c4:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{stats["sentences"]}</div><div class="stat-label">Sentences</div></div>', unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{stats["estimated_read_time_min"]}m</div><div class="stat-label">Read Time</div></div>', unsafe_allow_html=True)
        with c6:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{stats["avg_words_per_page"]}</div><div class="stat-label">Words/Page</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="step-label" style="margin-top:1.5rem;">Step 03 — Embed with Tarka-150M</div>', unsafe_allow_html=True)

        if not st.session_state.indexed:
            if st.button("🚀 Embed Document", use_container_width=True):
                status_box = st.empty()
                progress   = st.progress(0)

                def update_status(msg):
                    status_box.info(msg)

                with st.spinner(""):
                    try:
                        update_status("Chunking document...")
                        progress.progress(15)
                        chunks = chunk_text(
                            st.session_state.doc_text,
                            chunk_size=chunk_size,
                            overlap=chunk_overlap,
                        )
                        st.session_state.chunks = chunks
                        progress.progress(30)

                        update_status("Loading Tarka-Embedding-150M-V1 model...")
                        progress.progress(45)
                        st.session_state.embedder.load_model(update_status)
                        progress.progress(65)

                        update_status(f"Embedding {len(chunks)} chunks...")
                        st.session_state.embedder.build_index(chunks, update_status)
                        progress.progress(90)

                        st.session_state.indexed = True
                        progress.progress(100)
                        status_box.success(f"✅ Indexed {len(chunks)} chunks! Ready to chat.")

                    except Exception as e:
                        status_box.error(f"Embedding failed: {e}")
                        progress.empty()
        else:
            st.success(f"✅ {len(st.session_state.chunks)} chunks indexed — Ready!")


# ─── RIGHT: Chat ──────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="step-label">Step 04 — Ask Questions</div>', unsafe_allow_html=True)

    if not st.session_state.indexed:
        st.markdown("""
        <div style="background:#0f172a;border:1px dashed rgba(99,102,241,0.3);
             border-radius:12px;padding:2rem;text-align:center;color:#475569;">
            <div style="font-size:2.5rem;margin-bottom:0.5rem;">💬</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.85rem;">
                Upload & embed a document first<br>to start asking questions
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-bubble-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble-ai">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
                if msg.get("chunks"):
                    with st.expander("📎 Retrieved context chunks", expanded=False):
                        for i, chunk in enumerate(msg["chunks"]):
                            st.markdown(
                                f'<div class="chunk-card">'
                                f'<span class="score-pill">score {chunk["score"]:.3f}</span> '
                                f'chunk #{chunk["index"]+1}<br><br>'
                                f'{chunk["chunk"][:300]}{"..." if len(chunk["chunk"]) > 300 else ""}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

        st.divider()
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Ask a question",
                placeholder="e.g. What is the main topic? Summarize key findings...",
                label_visibility="collapsed",
            )
            col_btn1, col_btn2 = st.columns([3, 1])
            with col_btn1:
                submitted = st.form_submit_button("Send ↗", use_container_width=True)
            with col_btn2:
                clear = st.form_submit_button("Clear", use_container_width=True)

        if clear:
            st.session_state.chat_history = []
            st.rerun()

        if submitted and user_input.strip():
            if not api_ok:
                st.error("❌ Gemini API key not working. Check your .env file.")
            else:
                with st.spinner("Searching document & generating answer..."):
                    try:
                        results = st.session_state.embedder.search(user_input, top_k=top_k)
                        answer  = ask_gemini(user_input, results)
                        st.session_state.chat_history.append({"role": "user", "content": user_input})
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "chunks": results,
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Suggestions
        if not st.session_state.chat_history:
            st.markdown('<div style="color:#475569;font-size:0.8rem;margin-top:0.5rem;">💡 Try asking:</div>', unsafe_allow_html=True)
            suggestions = ["Summarize this document", "What are the key points?", "What is the main conclusion?"]
            cols = st.columns(3)
            for i, sug in enumerate(suggestions):
                with cols[i]:
                    if st.button(sug, key=f"sug_{i}", use_container_width=True):
                        with st.spinner("Generating answer..."):
                            try:
                                results = st.session_state.embedder.search(sug, top_k=top_k)
                                answer  = ask_gemini(sug, results)
                                st.session_state.chat_history.append({"role": "user", "content": sug})
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": answer,
                                    "chunks": results,
                                })
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))