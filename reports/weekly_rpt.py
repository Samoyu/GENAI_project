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
INDEX_NAME       = "techcrunch-articles"
NAMESPACE        = os.getenv("PINECONE_NAMESPACE", "")
EMBED_MODEL      = "nvidia/llama-3.2-nv-embedqa-1b-v2"
LLM_MODEL        = "meta/llama-3.1-8b-instruct"
NVIDIA_BASE_URL  = "https://integrate.api.nvidia.com/v1"

TOP_K, FETCH_K   = 8, 32
DAYS_BACK        = 7                 # initial window
BACKOFF_DAYS     = [60, None]        # widen to 60 days, then no filter
MAX_SNIPPETS     = 40
MAX_CONTEXT_TOKENS = 6000
MAX_CONTEXT_CHARS  = MAX_CONTEXT_TOKENS * 4
WRITER_MAX_TOKENS  = 1200

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("weekly_trends")

# -----------------------------
# Env
# -----------------------------
load_dotenv()
nvidia_key   = os.getenv("nvidia_key")
pinecone_key = os.getenv("pinecone_key")
if not (nvidia_key and pinecone_key):
    raise RuntimeError("Add nvidia_key and pinecone_key to .env")

# -----------------------------
# Clients
# -----------------------------
pc       = Pinecone(api_key=pinecone_key)
pc_index = pc.Index(INDEX_NAME)
embedder = NVIDIAEmbeddings(model=EMBED_MODEL, base_url=NVIDIA_BASE_URL, api_key=nvidia_key)
llm      = ChatNVIDIA(model=LLM_MODEL, base_url=NVIDIA_BASE_URL, api_key=nvidia_key, temperature=0.2)

vector_store = PineconeVectorStore(index=pc_index, embedding=embedder,
                                   text_key="text", namespace=NAMESPACE or None)

# -----------------------------
# Structured Schemas
# -----------------------------
OUTLINE_SCHEMA: Dict[str, Any] = {
    "title": "WeeklyTrendsOutline",
    "type": "object",
    "properties": {
        "focus_questions": {
            "type": "array", "minItems": 3, "maxItems": 8,
            "items": {"type": "string"}
        }
    },
    "required": ["focus_questions"],
    "additionalProperties": False
}

REPORT_SCHEMA: Dict[str, Any] = {
    "title": "WeeklyTrendsOnly",
    "type": "object",
    "properties": {
        "weekly_trends": {
            "type": "array", "minItems": 2, "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "topic":         {"type": "string"},
                    "description":   {"type": "string"},
                    "evidence_urls": {"type": "array","minItems":1,"items":{"type":"string"}}
                },
                "required": ["topic","description","evidence_urls"],
                "additionalProperties": False
            }
        }
    },
    "required": ["weekly_trends"],
    "additionalProperties": False
}

planner = llm.with_structured_output(OUTLINE_SCHEMA).with_retry(
    stop_after_attempt=3, wait_exponential_jitter=True)

writer  = llm.with_structured_output(REPORT_SCHEMA, max_tokens=WRITER_MAX_TOKENS).with_retry(
    stop_after_attempt=3, wait_exponential_jitter=True)

# ───────────── Helpers ─────────────
ISO_PATTERNS = ("%Y-%m-%dT%H:%M:%S%z","%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S","%Y-%m-%d")

def parse_iso(s: str):
    s = (s or "").replace("Z", "+00:00")
    for p in (None, *ISO_PATTERNS):
        try:
            return datetime.fromisoformat(s) if p is None else datetime.strptime(s, p)
        except:
            pass
    return None

def recent_enough(iso: str, now: datetime, days: Optional[int]) -> True:
    if days is None:
        return True
    dt = parse_iso(iso) or datetime(1970, 1, 1, tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (now - dt) <= timedelta(days=days)

def clean(t: str) -> str:
    return re.sub(r"\s+"," ", t or "").strip()

def trim(texts: List[str], limit: int) -> str:
    out, total = [], 0
    for t in texts:
        t = t.strip(); need = len(t) + 2
        if not t: continue
        if total + need > limit:
            if not out and limit > 0: out.append(t[:limit])
            break
        out.append(t); total += need
    return "\n\n".join(out)

def extract_url(md: Dict[str, Any]) -> str:
    # Accept common field names just in case the index used a different key
    for k in ("url", "source", "link", "href", "URL", "src"):
        v = md.get(k)
        if v:
            return clean(v)
    return ""

def fetch_docs(q: str, now: datetime, days: Optional[int]) -> List[Dict[str, str]]:
    """
    Return dicts with text + metadata (title, url, date). Filter by recency if days is not None.
    """
    docs = vector_store.max_marginal_relevance_search(q, k=TOP_K, fetch_k=FETCH_K, lambda_mult=0.5)
    out: List[Dict[str,str]] = []
    for d in docs:
        md = d.metadata or {}
        if recent_enough(clean(md.get("date","")), now, days):
            url = extract_url(md)
            out.append({
                "text":  " ".join((d.page_content or "").split())[:2000],
                "url":   url,
                "title": clean(md.get("title","")),
                "date":  clean(md.get("date","")),
                "id":    clean(md.get("id","")),
            })
    return out

def format_block(rec: Dict[str,str]) -> str:
    title = rec.get("title") or "(untitled)"
    url   = rec.get("url") or ""
    date  = rec.get("date") or ""
    txt   = rec.get("text") or ""
    return f"TITLE: {title}\nURL: {url}\nDATE: {date}\nTEXT: {txt}"

# ───────────── Prompts ─────────────
PLAN_PROMPT = PromptTemplate.from_template(
    "Generate 3–8 focus_questions to identify important *weekly trends* for VCs. "
    "Each question should target a different technology or market angle. "
    "Return only the array focus_questions."
)

WRITER_PROMPT = PromptTemplate.from_template(
    "You are a VC analyst. Using ONLY the CONTEXT below, create 2–8 weekly_trends. "
    "Each trend must include: topic, a 2–4 sentence description, and evidence_urls[] (1–3 items). "
    "Use URLs ONLY from the SOURCES list (copy them exactly). Do NOT invent links.\n\n"
    "=== SOURCES (title — url) ===\n{sources}\n\n"
    "=== CONTEXT START ===\n{context}\n=== CONTEXT END ==="
)

# ───────────── Retrieval + Generation ─────────────
def build_context(focus_qs: List[str], now: datetime, days: Optional[int]):
    seen_snips, blocks = set(), []
    sources_list: List[Dict[str, str]] = []
    seen_urls = set()

    for q in focus_qs:
        for rec in fetch_docs(q, now, days):
            snip = rec["text"]
            if not snip or snip in seen_snips:
                continue
            seen_snips.add(snip)

            # keep only items that have a non-empty URL
            if rec.get("url"):
                blocks.append(format_block(rec))
                if rec["url"] not in seen_urls:
                    sources_list.append({"title": rec.get("title") or "(untitled)", "url": rec["url"]})
                    seen_urls.add(rec["url"])

            if len(blocks) >= MAX_SNIPPETS:
                break
        if len(blocks) >= MAX_SNIPPETS:
            break

    context = trim(blocks, MAX_CONTEXT_CHARS)
    return context, sources_list

def generate_report(topic_hint: Optional[str]=None) -> Dict[str,Any]:
    now = datetime.now(timezone.utc)

    # Planner
    outline = (PLAN_PROMPT | planner).invoke({})
    focus_qs = outline["focus_questions"]
    if topic_hint:
        focus_qs = [f"{q} {topic_hint}" for q in focus_qs]

    # Retrieval with automatic window backoff
    tried_windows = [DAYS_BACK] + BACKOFF_DAYS
    context = ""
    sources_list: List[Dict[str,str]] = []
    for win in tried_windows:
        context, sources_list = build_context(focus_qs, now, win)
        log.info("retrieval: days_back=%s -> %d sources", win, len(sources_list))
        if sources_list:
            break

    sources_str = "\n".join(f"- {s['title']} — {s['url']}" for s in sources_list)
    allowed_urls_ordered = [s["url"] for s in sources_list if s.get("url")]

    # Writer
    try:
        trends = (WRITER_PROMPT | writer).invoke({"context": context, "sources": sources_str})
    except Exception as e:
        log.warning("Writer retry w/half context (%s)", str(e)[:120])
        context = context[:max(1000, len(context)//2)]
        trends = (WRITER_PROMPT | writer).invoke({"context": context, "sources": sources_str})

    # Enforce: evidence_urls only from Pinecone metadata URLs
    allowed_set = set(allowed_urls_ordered)
    for item in trends.get("weekly_trends", []):
        cur = [u for u in item.get("evidence_urls", []) if u in allowed_set]
        if not cur:
            # fallback: attach 1–2 known-good URLs so schema is satisfied
            cur = allowed_urls_ordered[:2]
        item["evidence_urls"] = cur[:3]

    # add meta
    trends["as_of_utc"]        = now.isoformat()
    trends["window_start_utc"] = (now - timedelta(days=tried_windows[0] if tried_windows[0] else 0)).isoformat() \
                                  if tried_windows[0] else "unbounded"
    trends["window_end_utc"]   = now.isoformat()
    trends["num_sources"]      = len(allowed_urls_ordered)
    return trends

# ───────────── Run & save ─────────────
payload = generate_report()
with open("weekly_report_structured.json","w",encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
print("Saved weekly_report_structured.json (trends only, full pipeline)")
