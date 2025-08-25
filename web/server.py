import os
import json
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.prompts import PromptTemplate

# -----------------------------
# Config / Env
# -----------------------------
load_dotenv()
INDEX_NAME       = os.getenv("PINECONE_INDEX", "techcrunch-articles")
NAMESPACE        = os.getenv("PINECONE_NAMESPACE", "") or None

EMBED_MODEL      = os.getenv("EMBED_MODEL", "nvidia/llama-3.2-nv-embedqa-1b-v2")  # base model
LLM_MODEL        = os.getenv("LLM_MODEL", "meta/llama-3.1-8b-instruct")

# IMPORTANT: langchain_nvidia expects base URL WITHOUT /v1
NVIDIA_BASE_URL  = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

nvidia_key   = os.getenv("nvidia_key")
pinecone_key = os.getenv("pinecone_key")
if not (nvidia_key and pinecone_key):
    raise RuntimeError("Please set nvidia_key and pinecone_key in your .env")

DATA_DIR = Path(__file__).parent.resolve()

REPORT_FILES = {
    "weekly_trends": DATA_DIR / "weekly_report_structured.json",
    "emerging_technologies": DATA_DIR / "emerging_technologies_report.json",
    "pricing_watch": DATA_DIR / "pricing_watch.json",
    "startup_funding": DATA_DIR / "startup_funding_report.json",
}

# -----------------------------
# Retrieval clients
# -----------------------------
pc = Pinecone(api_key=pinecone_key)
index = pc.Index(INDEX_NAME)

# Use base model; embed_query() handles query mode internally
embedder_query = NVIDIAEmbeddings(
    model=EMBED_MODEL,
    base_url=NVIDIA_BASE_URL,  # no /v1
    api_key=nvidia_key,
)

llm = ChatNVIDIA(
    model=LLM_MODEL,
    base_url=NVIDIA_BASE_URL,  # no /v1
    api_key=nvidia_key,
    temperature=0.2,
)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embedder_query,
    text_key="text",
    namespace=NAMESPACE,
)

PROMPT = PromptTemplate.from_template(
    "You are an assistant that answers based ONLY on the context below.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer concisely:"
)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="VC Intel Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Serve static (index.html lives next to server.py)
app.mount("/static", StaticFiles(directory=str(DATA_DIR)), name="static")

@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/api/reports")
def list_reports():
    out = []
    for key, path in REPORT_FILES.items():
        out.append({"key": key, "exists": path.exists(), "filename": path.name})
    return out

@app.get("/api/report/{report_key}")
def get_report(report_key: str):
    path = REPORT_FILES.get(report_key)
    if not path or not path.exists():
        return JSONResponse({"error": "report not found"}, status_code=404)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/chat")
def chat(payload: Dict[str, Any] = Body(...)):
    message = (payload.get("message") or "").strip()
    if not message:
        return {"answer": "Please type a question.", "sources": []}

    # retrieve
    try:
        docs = vector_store.similarity_search(message, k=5)
    except Exception as e:
        return {"answer": f"Retrieval error: {e}", "sources": []}

    context = "\n\n".join(d.page_content for d in docs)

    # generate (return ONLY textual content)
    try:
        msg = (PROMPT | llm).invoke({"question": message, "context": context})
        answer_text = getattr(msg, "content", str(msg)).strip()
    except Exception as e:
        return {"answer": f"LLM error: {e}", "sources": []}

    sources = []
    for d in docs:
        md = d.metadata or {}
        if md.get("url"):
            sources.append({"title": md.get("title") or "(untitled)", "url": md.get("url")})

    return {"answer": answer_text, "sources": sources[:5]}

@app.get("/")
def root():
    return FileResponse(str(DATA_DIR / "index.html"))
