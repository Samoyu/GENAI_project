import os 
from typing import List, Dict, Any
from openai import OpenAI

EMBED_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"

# -----------------------------
# Helpers
# -----------------------------
def chunk_text(text: str, max_tokens: int = 256, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    step = max(1, max_tokens - overlap)
    while i < len(words):
        chunk = words[i:i + max_tokens]
        chunks.append(" ".join(chunk))
        i += step
    return chunks

# -----------------------------
# Embedding (index-time) client
# -----------------------------
def embed_texts_index_mode(client: OpenAI, texts: List[str]) -> List[List[float]]:
    cleaned = [t for t in texts if isinstance(t, str) and t.strip()]
    if not cleaned:
        return []
    # IMPORTANT: use "passage" (a.k.a. document) for indexing
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=cleaned,
        encoding_format="float",
        extra_body={"input_type": "passage", "truncate": "NONE"}
    )
    return [d.embedding for d in resp.data]
    
def get_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v