import os
from google import genai
from google.genai import types

MODEL_NAME = "gemini-2.5-flash"


def get_api_key() -> str:
    """Get API key from environment (works for both local .env and HuggingFace secrets)."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return key


def get_gemini_client():
    """Initialize and return Gemini client."""
    api_key = get_api_key()
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY not set. Please add your API key to the .env file or HuggingFace Secrets.")
    client = genai.Client(api_key=api_key)
    return client


def build_prompt(question: str, context_chunks: list) -> str:
    """Build a RAG prompt from question and retrieved context chunks."""
    context_text = "\n\n---\n\n".join(
        [f"[Chunk {i+1} | Relevance: {c['score']:.2f}]\n{c['chunk']}"
         for i, c in enumerate(context_chunks)]
    )
    prompt = f"""You are a precise and helpful document assistant. Answer the user's question strictly based on the provided document context.

RULES:
- Answer ONLY from the context provided below.
- If the answer is not in the context, say: "I couldn't find relevant information in the document for this question."
- Be concise yet thorough. Use bullet points when listing multiple items.
- Quote directly from the document when it helps the answer.
- Do NOT make up information.

DOCUMENT CONTEXT:
{context_text}

USER QUESTION:
{question}

ANSWER:"""
    return prompt


def ask_gemini(question: str, context_chunks: list) -> str:
    """Send question + context to Gemini and return the answer."""
    client = get_gemini_client()
    prompt = build_prompt(question, context_chunks)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=2048,
        ),
    )
    return response.text


def check_api_key() -> tuple:
    """Check if API key is valid."""
    try:
        api_key = get_api_key()
        if not api_key or api_key == "your_gemini_api_key_here":
            return False, "GEMINI_API_KEY not set in .env file or HuggingFace Secrets"
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents="Say OK",
            config=types.GenerateContentConfig(max_output_tokens=5),
        )
        return True, "API key is valid."
    except ValueError as e:
        return False, str(e)
    except Exception as e:
        return False, f"API error: {str(e)}"
