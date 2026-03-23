import os
import PyPDF2
import docx
import tiktoken


def extract_text_from_pdf(file) -> tuple[str, int]:
    """Extract text from PDF and return (text, page_count)."""
    reader = PyPDF2.PdfReader(file)
    page_count = len(reader.pages)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip(), page_count


def extract_text_from_docx(file) -> tuple[str, int]:
    """Extract text from DOCX and return (text, estimated_pages)."""
    doc = docx.Document(file)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
    text = "\n".join(full_text)
    # Estimate pages: ~250 words per page
    word_count = len(text.split())
    estimated_pages = max(1, round(word_count / 250))
    return text.strip(), estimated_pages


def extract_text_from_txt(file) -> tuple[str, int]:
    """Extract text from TXT and return (text, estimated_pages)."""
    text = file.read().decode("utf-8", errors="ignore")
    word_count = len(text.split())
    estimated_pages = max(1, round(word_count / 250))
    return text.strip(), estimated_pages


def extract_text(file, filename: str) -> tuple[str, int]:
    """Extract text from uploaded file based on extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file)
    elif ext == ".docx":
        return extract_text_from_docx(file)
    elif ext == ".txt":
        return extract_text_from_txt(file)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: PDF, DOCX, TXT")


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base encoding)."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback: approximate 1 token per 4 characters
        return len(text) // 4


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def get_document_stats(text: str, page_count: int, filename: str) -> dict:
    """Return a stats dictionary for the uploaded document."""
    word_count = len(text.split())
    char_count = len(text)
    token_count = count_tokens(text)
    sentence_count = text.count(".") + text.count("!") + text.count("?")
    avg_words_per_page = round(word_count / max(page_count, 1))

    return {
        "filename": filename,
        "pages": page_count,
        "words": word_count,
        "characters": char_count,
        "tokens": token_count,
        "sentences": sentence_count,
        "avg_words_per_page": avg_words_per_page,
        "estimated_read_time_min": max(1, round(word_count / 200)),
    }
