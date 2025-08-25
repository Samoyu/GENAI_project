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
NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "")
EMBED_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
LLM_MODEL   = "meta/llama-3.1-8b-instruct"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Increase recall
TOP_K = 12            # kept after MMR (was 8)
FETCH_K = 36         # pre-MMR candidates (was 32)

# Date windows & thresholds
DAYS_BACK = 30        # initial window
BACKOFF_DAYS = [90, None]  # widen gradually → unbounded
MIN_SOURCES = 25      # require at least this many sources before stopping

# Context/output budgets
MAX_SNIPPETS = 120
MAX_CONTEXT_TOKENS = 9000
MAX_CONTEXT_CHARS  = 4 * MAX_CONTEXT_TOKENS   # ~36k chars
WRITER_MAX_TOKENS  = 2200
TARGET_MIN_EVENTS  = 20   # if the writer returns fewer, we take a second pass with a wider window

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("startup_funding")

# Funding-focused query boosters
FUNDING_BOOST_QUERIES = [
    "seed funding raises round",
    "Series A raise investment led by",
    "Series B funding round investors",
    "Series C growth equity valuation",
    "pre-seed angel round accelerators",
    "venture capital financing announced",
    "company raised $ million led by",
    "acquisition or merger deal value",
]

# -----------------------------
# Env
# -----------------------------
load_dotenv()
nvidia_key   = os.getenv("nvidia_key")
pinecone_key = os.getenv("pinecone_key")
if not nvidia_key or not pinecone_key:
    raise RuntimeError("Set nvidia_key and pinecone_key in your environment (.env).")

# -----------------------------
# Helpers
# -----------------------------
ISO_PATTERNS = ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d")

FUNDING_SENTENCE_RE = re.compile(
    r"""(?ix)
    (?:
      (raised|raises|raise|funding|round|seed|pre[-\s]?seed|series\s?[abcde]|bridge|
       valuation|led\s+by|participat(?:ed|es)\s+in|financing|investment|investors|
       acquisition|acquired|merger|\$[\d,.]+|€[\d,.]+|£[\d,.]+|NT\$[\d,.]+)
    )
    """
)

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

def recent_enough(iso_str: str, now_utc: datetime, days: Optional[int]) -> bool:
    if days is None:  # unbounded
        return True
    dt = parse_iso_maybe(iso_str)
    if not dt: return False
    dt = as_aware_utc(dt)
    return (now_utc - dt) <= timedelta(days=days)

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

def extract_funding_bits(text: str, max_chars: int = 800) -> str:
    """
    Keep only sentences around funding cues to pack more docs in context.
    """
    # naive sentence split
    sents = re.split(r'(?<=[\.\?!])\s+', text.strip())
    keep = []
    for i, s in enumerate(sents):
        if FUNDING_SENTENCE_RE.search(s):
            # include neighboring context
            left  = sents[i-1] if i-1 >= 0 else ""
            right = sents[i+1] if i+1 < len(sents) else ""
            snippet = " ".join([left, s, right]).strip()
            keep.append(snippet)
    if not keep:
        # fallback: first N chars
        return text[:max_chars]
    clipped = " ".join(keep)
    return clipped[:max_chars]

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
    "title": "StartupFundingReport",
    "type": "object",
    "properties": {
        "as_of_utc": {"type": "string"},
        "window_start_utc": {"type": "string"},
        "window_end_utc": {"type": "string"},
        "highlights": {"type": "array", "minItems": 0, "maxItems": 10, "items": {"type": "string"}},
        "funding_events": {
            "type": "array", "minItems": 0, "maxItems": 200,
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "sector": {"type": "string"},
                    "round": {"type": "string"},
                    "amount": {"type": "string"},
                    "date": {"type": "string"},
                    "investors": {"type": "array", "items": {"type": "string"}},
                    "source_url": {"type": "string"},
                    "summary": {"type": "string"}
                },
                "required": ["company", "sector", "round", "amount", "date", "source_url", "summary"],
                "additionalProperties": False
            }
        }
    },
    "required": ["funding_events"],
    "additionalProperties": False
}

OUTLINE_JSON_SCHEMA: Dict[str, Any] = {
    "title": "FundingOutline",
    "type": "object",
    "properties": {
        "sections": {
            "type": "array", "minItems": 3, "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "focus_question": {"type": "string"}},
                "required": ["title", "focus_question"], "additionalProperties": False
            }
        }
    },
    "required": ["sections"], "additionalProperties": False
}

planner = llm.with_structured_output(OUTLINE_JSON_SCHEMA).with_retry(
    stop_after_attempt=3, wait_exponential_jitter=True
)
writer = llm.with_structured_output(REPORT_JSON_SCHEMA, max_tokens=WRITER_MAX_TOKENS).with_retry(
    stop_after_attempt=3, wait_exponential_jitter=True
)

# -----------------------------
# Retrieval with URL-aware context
# -----------------------------
def fetch_docs(query: str, now_utc: datetime, days: Optional[int]):
    """MMR search + recency filter; return dicts with text + URL/title/date."""
    docs = vector_store.max_marginal_relevance_search(query, k=TOP_K, fetch_k=FETCH_K, lambda_mult=0.5)
    out = []
    for d in docs:
        md = d.metadata or {}
        if recent_enough(clean(md.get("date", "")), now_utc, days):
            txt = " ".join((d.page_content or "").split())
            out.append({
                "text":  txt[:3000],  # keep some buffer for funding extraction
                "url":   clean(md.get("url", "")),
                "title": clean(md.get("title", "")),
                "date":  clean(md.get("date", "")),
                "id":    clean(md.get("id", "")),
            })
    return out

def build_context_and_sources(queries: List[str], now_utc: datetime, days: Optional[int]):
    """
    Build a context of many short funding-focused snippets and a SOURCES list.
    """
    seen_texts, blocks = set(), []
    sources_list, seen_urls = [], set()
    for q in queries:
        for rec in fetch_docs(q, now_utc, days):
            if not rec.get("url"):  # we only keep docs with URLs for SOURCES mapping
                continue
            snip = extract_funding_bits(rec["text"], max_chars=900)
            # dedupe by text body
            s_clean = " ".join(snip.split())
            if not s_clean or s_clean in seen_texts:
                continue
            seen_texts.add(s_clean)
            blocks.append(s_clean)
            if rec["url"] not in seen_urls:
                sources_list.append({"title": rec.get("title") or "(untitled)", "url": rec["url"]})
                seen_urls.add(rec["url"])
            if len(blocks) >= MAX_SNIPPETS:
                break
        if len(blocks) >= MAX_SNIPPETS:
            break
    context = trim_to_char_budget(blocks, MAX_CONTEXT_CHARS)
    return context, sources_list

# -----------------------------
# Prompts
# -----------------------------
PLAN_PROMPT = PromptTemplate.from_template(
    "Plan a startup funding report with 3–6 sections.\n"
    "Provide 'title' and one 'focus_question' per section to discover funding events\n"
    "from the last {days} days (TechCrunch/Crunchbase corpus).\n"
    "Audience: venture capitalists. Return ONLY structured data."
)

WRITER_PROMPT = PromptTemplate.from_template(
    "You are an investment analyst. Using ONLY the CONTEXT below, produce a structured\n"
    "startup funding report with:\n"
    "- funding_events: each has company, sector, round, amount, date, investors[], source_url, summary\n"
    "- highlights: 0–10 concise bullets about notable patterns (optional)\n\n"
    "Rules:\n"
    "- Use exactly ONE URL from the SOURCES list as source_url for each event; do NOT invent links.\n"
    "- No fabrication; every item must be grounded in the context.\n"
    "- Include as many distinct funding events as the context supports (aim high, up to 200).\n"
    "- Aggregate duplicates across multiple articles and keep summaries concise.\n\n"
    "=== SOURCES (title — url) ===\n{sources}\n\n"
    "=== CONTEXT START ===\n{context}\n=== CONTEXT END ==="
)

# -----------------------------
# Main pipeline
# -----------------------------
def generate_report(topic_hint: Optional[str] = None) -> Dict[str, Any]:
    now_utc = datetime.now(timezone.utc)

    # Planner
    plan = (PLAN_PROMPT | planner).invoke({"days": DAYS_BACK})
    seed_queries = [s["focus_question"] for s in plan.get("sections", [])]

    # Add funding booster queries
    seed_queries = seed_queries + FUNDING_BOOST_QUERIES
    if topic_hint:
        seed_queries = [f"{q} {topic_hint}" for q in seed_queries]

    # Retrieval with automatic date-window backoff until we meet MIN_SOURCES
    tried_windows = [DAYS_BACK] + BACKOFF_DAYS
    context = ""
    sources_list: List[Dict[str, str]] = []
    for win in tried_windows:
        context, sources_list = build_context_and_sources(seed_queries, now_utc, win)
        log.info("retrieval: days_back=%s -> %d sources", win, len(sources_list))
        if len(sources_list) >= MIN_SOURCES:
            break

    sources_str = "\n".join(f"- {s['title']} — {s['url']}" for s in sources_list)
    allowed_urls_ordered = [s["url"] for s in sources_list if s.get("url")]
    allowed_set = set(allowed_urls_ordered)

    # Writer
    try:
        result = (WRITER_PROMPT | writer).invoke({"context": context, "sources": sources_str})
    except Exception as e:
        log.warning("Writer failed once (%s). Retrying with half context...", str(e)[:200])
        half = context[: max(1000, len(context)//2)]
        result = (WRITER_PROMPT | writer).invoke({"context": half, "sources": sources_str})

    # If too few events, widen once more (best-effort)
    if len(result.get("funding_events", []) or []) < TARGET_MIN_EVENTS:
        log.info("Few events returned (%d). Widening window for a second pass.",
                 len(result.get("funding_events", []) or []))
        # take the widest window and rebuild context
        context2, sources2 = build_context_and_sources(seed_queries, now_utc, None)
        src2_str = "\n".join(f"- {s['title']} — {s['url']}" for s in sources2)
        allowed2 = [s["url"] for s in sources2 if s.get("url")]
        allowed_set2 = set(allowed2)
        try:
            result2 = (WRITER_PROMPT | writer).invoke({"context": context2, "sources": src2_str})
            # prefer the larger list
            if len(result2.get("funding_events", []) or []) > len(result.get("funding_events", []) or []):
                result = result2
                allowed_urls_ordered = allowed2
                allowed_set = allowed_set2
        except Exception as e:
            log.warning("Second pass failed: %s", str(e)[:200])

    # Post-sanitize: ensure source_url is one of the allowed URLs
    events = result.get("funding_events", []) or []
    for ev in events:
        url = ev.get("source_url", "")
        if url not in allowed_set:
            ev["source_url"] = allowed_urls_ordered[0] if allowed_urls_ordered else ""
    result["funding_events"] = events

    # Meta
    result["as_of_utc"] = now_utc.isoformat()
    result["window_start_utc"] = (now_utc - timedelta(days=DAYS_BACK)).isoformat()
    result["window_end_utc"] = now_utc.isoformat()
    return result


payload = generate_report()
with open("startup_funding_report.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print("Saved startup_funding_report.json")
