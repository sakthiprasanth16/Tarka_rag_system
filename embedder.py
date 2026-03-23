import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List

MODEL_NAME = "Tarka-AIR/Tarka-Embedding-150M-V1"


class TarkaEmbedder:
    """Handles embedding with Tarka-Embedding-150M-V1 and FAISS vector store."""

    def __init__(self):
        self.model = None
        self.index = None
        self.chunks: List[str] = []
        self.dimension = 768  # Tarka-150M output dimension

    def load_model(self, status_callback=None):
        """Load the Tarka embedding model."""
        if self.model is None:
            if status_callback:
                status_callback("Loading Tarka-Embedding-150M-V1... (first run downloads model)")
            self.model = SentenceTransformer(
                MODEL_NAME,
                trust_remote_code=True,
                tokenizer_kwargs={"padding_side": "left"},
            )
            if status_callback:
                status_callback("Model loaded successfully!")
        return self.model

    def embed_chunks(self, chunks: List[str], status_callback=None) -> np.ndarray:
        """Embed a list of text chunks and return numpy array."""
        if self.model is None:
            self.load_model(status_callback)

        if status_callback:
            status_callback(f"Embedding {len(chunks)} chunks with Tarka-150M...")

        embeddings = self.model.encode(
            chunks,
            batch_size=16,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def build_index(self, chunks: List[str], status_callback=None):
        """Build FAISS index from chunks."""
        self.chunks = chunks
        embeddings = self.embed_chunks(chunks, status_callback)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        if status_callback:
            status_callback(f"FAISS index built with {self.index.ntotal} vectors.")

        return self.index

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Search the FAISS index for the most relevant chunks."""
        if self.index is None or not self.chunks:
            raise ValueError("Index not built. Please embed documents first.")

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        query_embedding = np.array(query_embedding, dtype=np.float32)

        scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(score),
                    "index": int(idx),
                })
        return results

    def reset(self):
        """Clear index and chunks."""
        self.index = None
        self.chunks = []