# 🔍 RAG System — Intelligent Document Q&A System

<div align="center">

![TarkaRAG Banner](https://img.shields.io/badge/TarkaRAG-Document%20Intelligence-6366f1?style=for-the-badge&logo=bookstack&logoColor=white)

[![HuggingFace Space](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Space-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/prasanthr0416/Tarka_rag_system)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

**Upload any document. Ask anything. Get precise answers.**

[🚀 Live Demo](https://huggingface.co/spaces/prasanthr0416/Tarka_rag_system) · [📖 How it works](#how-it-works) · [⚡ Quick Start](#quick-start) · [🛠️ Tech Stack](#tech-stack)

</div>

---

## ✨ What is RAG System?

RAG System is a local **Retrieval-Augmented Generation (RAG)** system that lets you upload any document and ask questions about it in natural language. It uses **Tarka-Embedding-150M-V1** for semantic search and **Gemini 2.5 Flash** to generate precise, grounded answers.

No hallucinations. No guessing. Only answers from your document.

---

## 🎯 Live Demo

> 🟢 **Try it now:** [https://huggingface.co/spaces/prasanthr0416/Tarka_rag_system](https://huggingface.co/spaces/prasanthr0416/Tarka_rag_system)

Upload a PDF, DOCX, or TXT file and start asking questions instantly!

---

## 🚀 Features

- 📄 **Multi-format support** — PDF, DOCX, TXT
- 📊 **Document analytics** — pages, words, tokens, read time shown instantly
- 🧠 **Tarka-Embedding-150M-V1** — state-of-the-art semantic embeddings
- ⚡ **FAISS vector search** — blazing fast similarity search
- 🤖 **Gemini 2.5 Flash** — accurate, grounded answers
- 💬 **Chat interface** — full conversation history
- 🔍 **Context transparency** — see exactly which chunks were used to answer
- 🎛️ **Adjustable settings** — chunk size, overlap, Top-K retrieval
- 🌙 **Dark themed UI** — clean, modern Streamlit interface

---

## 🧠 How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                        UPLOAD PHASE                         │
│                                                             │
│  PDF/DOCX/TXT  →  Extract Text  →  Show Stats (pages,      │
│                                    words, tokens)           │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                       EMBED PHASE                           │
│                                                             │
│  Full Text  →  Split into 500-word chunks                   │
│             →  Tarka-150M encodes each chunk                │
│             →  768-dim vectors stored in FAISS (RAM)        │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                       QUERY PHASE                           │
│                                                             │
│  User Question  →  Tarka-150M encodes question              │
│                 →  FAISS finds Top-5 similar chunks         │
│                 →  Gemini 2.5 Flash reads chunks            │
│                 →  Returns grounded answer                  │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- Gemini API key (free at [aistudio.google.com](https://aistudio.google.com/app/apikey))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/tarka-rag-system.git
cd tarka-rag-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
echo "GEMINI_API_KEY=your_key_here" > .env

# 5. Run the app
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser!

---

## 📁 Project Structure

```
tarka-rag-system/
├── app.py                  # Streamlit UI — main entry point
├── document_processor.py   # PDF/DOCX/TXT extraction + document stats
├── embedder.py             # Tarka-150M embedding + FAISS index
├── llm_handler.py          # Gemini 2.5 Flash answer generation
├── requirements.txt        # Python dependencies
├── .env                    # API key (local only, not committed)
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **UI** | Streamlit | Web interface |
| **Embedding** | Tarka-Embedding-150M-V1 | Text → 768-dim vectors |
| **ML Framework** | PyTorch + sentence-transformers | Model inference |
| **Vector DB** | FAISS | Fast similarity search |
| **LLM** | Gemini 2.5 Flash | Answer generation |
| **PDF parsing** | PyPDF2 | Extract text from PDFs |
| **DOCX parsing** | python-docx | Extract text from Word files |
| **Tokenizer** | tiktoken | Token counting |
| **API SDK** | google-genai | Gemini API calls |

---

## ⚙️ Configuration

Adjust these settings in the sidebar:

| Setting | Default | Description |
|---|---|---|
| Chunk size | 500 words | Words per chunk |
| Overlap | 50 words | Shared words between chunks |
| Top-K | 5 | Chunks sent to Gemini |

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Your Google Gemini API key |

**Local:** Add to `.env` file
```
GEMINI_API_KEY=AIzaSy...
```

**HuggingFace:** Add as Secret in Space Settings

---

## 📊 Capacity & Performance

| Document Size | Pages | CPU Time |
|---|---|---|
| Small | 1–50 pages | ~30 sec |
| Medium | 50–200 pages | ~2 min |
| Large | 200–500 pages | ~5 min |

> **No word limit!** Documents are chunked, so any size works within your RAM.

---

## 🤗 Deploy on HuggingFace Spaces

1. Fork this repo
2. Create a new Space at [huggingface.co](https://huggingface.co)
3. Select **Streamlit** SDK
4. Upload all files
5. Add `GEMINI_API_KEY` in **Settings → Secrets**
6. Done! 🎉

---

## 🙏 Acknowledgements

- [Tarka Labs](https://huggingface.co/Tarka-AIR) for the Tarka-Embedding models
- [Google](https://ai.google.dev) for Gemini 2.5 Flash API
- [Meta](https://github.com/facebookresearch/faiss) for FAISS
- [HuggingFace](https://huggingface.co) for model hosting

---

<div align="center">

Made with ❤️ using Tarka-150M + Gemini 2.5 Flash

⭐ Star this repo if you found it useful!

</div>
