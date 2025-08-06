import os, json, re, logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.prompts import PromptTemplate

# -----------------------------
# Config
# -----------------------------
INDEX_NAME = "techcrunch-articles"
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")
EMBED_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
LLM_MODEL = "meta/llama-3.1-8b-instruct"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

TOP_K = 8
FETCH_K = 32
DAYS_BACK = 7
MAX_SNIPPETS = 50
MAX_CONTEXT_TOKENS = 6000
MAX_CONTEXT_CHARS  = 4 * MAX_CONTEXT_TOKENS
WRITER_MAX_TOKENS  = 1500

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("emerging_technologies")

# -----------------------------
# Env
# -----------------------------
load_dotenv()
nvidia_key = os.getenv("nvidia_key")
pinecone_key = os.getenv("pinecone_key")
if not nvidia_key or not pinecone_key:
    raise RuntimeError("Set nvidia_key and pinecone_key in your environment (.env).")

# -----------------------------
# Helpers
# -----------------------------
ISO_PATTERNS = ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d")

def parse_iso_maybe(s: str):
    if not s or not isinstance(s, str): return None
    s = s.replace("Z", "+00:00")
    try: return datetime.fromisoformat(s)
    except Exception: pass
    for p in ISO_PATTERNS:
        try: return datetime.strptime(s, p)
        except Exception: continue
    return None

def as_aware_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None: return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def in_last_n_days(iso_str: str, n: int, now_utc: datetime) -> bool:
    dt = parse_iso_maybe(iso_str)
    if not dt: return False
    dt = as_aware_utc(dt)
    return (now_utc - dt) <= timedelta(days=n)

def clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def trim_to_char_budget(texts: List[str], max_chars: int) -> str:
    combined, total = [], 0
    for t in texts:
        t = t.strip()
        if not t: continue
        need = len(t) + 2
        if total + need > max_chars:
            if not combined and max_chars > 0:
                combined.append(t[:max_chars])
            break
        combined.append(t)
        total += need
    return "\n\n".join(combined)

# -----------------------------
# Clients
# -----------------------------
pc = Pinecone(api_key=pinecone_key)
pc_index = pc.Index(INDEX_NAME)

embedder = NVIDIAEmbeddings(model=EMBED_MODEL, base_url=NVIDIA_BASE_URL, api_key=nvidia_key)
llm = ChatNVIDIA(model=LLM_MODEL, base_url=NVIDIA_BASE_URL, api_key=nvidia_key, temperature=0.2)

vector_store = PineconeVectorStore(
    index=pc_index,
    embedding=embedder,
    text_key="text",
    namespace=NAMESPACE or None,
)

# -----------------------------
# Structured Schemas
# -----------------------------
REPORT_JSON_SCHEMA: Dict[str, Any] = {
    "title": "EmergingTechnologiesReport",
    "type": "object",
    "properties": {
        "as_of_utc": {"type": "string"},
        "window_start_utc": {"type": "string"},
        "window_end_utc": {"type": "string"},
        "emerging_technologies": {
            "type": "array", "minItems": 3, "maxItems": 15,
            "items": {
                "type": "object",
                "properties": {
                    "technology": {"type": "string"},
                    "explanation": {"type": "string"},
                    "maturity": {"type": "string"},  # e.g., research, prototype, early adoption, mainstream
                    "key_players": {"type": "array", "items": {"type": "string"}},
                    "sectors_impacted": {"type": "array", "items": {"type": "string"}},
                    "evidence_urls": {"type": "array", "minItems": 1, "items": {"type": "string"}}
                },
                "required": ["technology", "explanation", "evidence_urls"],
                "additionalProperties": False
            }
        },
        "highlights": {
            "type": "array", "minItems": 0, "maxItems": 10,
            "items": {"type": "string"}
        }
    },
    "required": ["emerging_technologies"],
    "additionalProperties": False
}

OUTLINE_JSON_SCHEMA: Dict[str, Any] = {
    "title": "EmergingOutline",
    "type": "object",
    "properties": {
        "sections": {
            "type": "array", "minItems": 3, "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "focus_question": {"type": "string"}
                },
                "required": ["title", "focus_question"],
                "additionalProperties": False
            }
        }
    },
    "required": ["sections"],
    "additionalProperties": False
}

planner = llm.with_structured_output(OUTLINE_JSON_SCHEMA).with_retry(
    stop_after_attempt=3, wait_exponential_jitter=True
)  # Runnable.with_retry per docs. :contentReference[oaicite:3]{index=3}

writer = llm.with_structured_output(REPORT_JSON_SCHEMA, max_tokens=WRITER_MAX_TOKENS).with_retry(
    stop_after_attempt=3, wait_exponential_jitter=True
)

# -----------------------------
# Retrieval (MMR + recency)
# -----------------------------
def mmr_recent(query: str, now_utc: datetime, days_back: int = DAYS_BACK, k: int = TOP_K, fetch_k: int = FETCH_K):
    docs = vector_store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, lambda_mult=0.5)
    # MMR & fetch_k available on PineconeVectorStore. :contentReference[oaicite:4]{index=4}
    return [d for d in docs if in_last_n_days(clean((d.metadata or {}).get("date", "")), days_back, now_utc)]

def build_context_blocks(queries: List[str], now_utc: datetime) -> List[str]:
    seen, blocks = set(), []
    for q in queries:
        for d in mmr_recent(q, now_utc):
            s_clean = " ".join((d.page_content or "").split())[:2000]
            if s_clean and s_clean not in seen:
                seen.add(s_clean)
                blocks.append(s_clean)
                if len(blocks) >= MAX_SNIPPETS:
                    return blocks
    return blocks

# -----------------------------
# Prompts
# -----------------------------
PLAN_PROMPT = PromptTemplate.from_template(
    "Plan an emerging technologies report with 3–6 sections.\n"
    "Each section has one 'focus_question' to identify novel tech/themes in the last {days} days\n"
    "from TechCrunch coverage. Audience: venture capitalists. Return ONLY structured data."
)

WRITER_PROMPT = PromptTemplate.from_template(
    "You are a VC analyst. Using ONLY the CONTEXT below, produce a structured emerging technologies report with:\n"
    "- emerging_technologies: each item has technology, explanation, maturity, key_players[], sectors_impacted[], evidence_urls[]\n"
    "- highlights: 0–10 concise bullets about patterns/opportunities (optional)\n\n"
    "Rules:\n"
    "- No fabrication; ground every item in the context.\n"
    "- Aggregate duplicates across articles and include URLs as evidence.\n"
    "- Be concise and investor-oriented.\n\n"
    "=== CONTEXT START ===\n{context}\n=== CONTEXT END ==="
)

# -----------------------------
# Main pipeline
# -----------------------------
def generate_report(topic_hint: Optional[str] = None) -> Dict[str, Any]:
    now_utc = datetime.now(timezone.utc)

    plan = (PLAN_PROMPT | planner).invoke({"days": DAYS_BACK})
    seed_queries = [s["focus_question"] for s in plan.get("sections", [])]
    if topic_hint:
        seed_queries = [f"{q} {topic_hint}" for q in seed_queries]

    blocks = build_context_blocks(seed_queries, now_utc)
    context = trim_to_char_budget(blocks, MAX_CONTEXT_CHARS)

    try:
        result = (WRITER_PROMPT | writer).invoke({"context": context})
    except Exception as e:
        log.warning("Writer failed once (%s). Retrying with half context...", str(e)[:200])
        half = context[: max(1000, len(context)//2)]
        result = (WRITER_PROMPT | writer).invoke({"context": half})

    result["as_of_utc"] = now_utc.isoformat()
    result["window_start_utc"] = (now_utc - timedelta(days=DAYS_BACK)).isoformat()
    result["window_end_utc"] = now_utc.isoformat()
    return result


payload = generate_report()
with open("emerging_technologies_report.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print("Saved emerging_technologies_report.json")
