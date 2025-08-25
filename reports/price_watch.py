import os, re, json, time, logging
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
INDEX_NAME = os.getenv("PINECONE_INDEX", "techcrunch-articles")
NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "")
EMBED_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
LLM_MODEL   = "meta/llama-3.1-8b-instruct"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Recall knobs
TOP_K, FETCH_K   = 20, 100         # more candidates -> more recall
DAYS_BACK        = int(os.getenv("REPORT_DAYS_BACK", "14"))
BACKOFF_WINDOWS  = [90, None]      # widen to 90 days, then unbounded
MIN_SOURCES_GOAL = int(os.getenv("PRICING_MIN_SOURCES", "12"))  # stop once we have at least N unique URLs
MAX_SNIPPETS     = 120             # allow deeper context for more items
MAX_CONTEXT_TOKENS = 7000
MAX_CONTEXT_CHARS  = 4 * MAX_CONTEXT_TOKENS
WRITER_MAX_TOKENS  = 1600

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pricing_watch")

# -----------------------------
# Env & Clients
# -----------------------------
load_dotenv()
nvidia_key  = os.getenv("nvidia_key")
pinecone_key= os.getenv("pinecone_key")
if not (nvidia_key and pinecone_key):
    raise RuntimeError("Set nvidia_key and pinecone_key in your .env")

pc = Pinecone(api_key=pinecone_key)
pc_index = pc.Index(INDEX_NAME)

embedder = NVIDIAEmbeddings(model=EMBED_MODEL, base_url=NVIDIA_BASE_URL, api_key=nvidia_key)
llm      = ChatNVIDIA(model=LLM_MODEL, base_url=NVIDIA_BASE_URL, api_key=nvidia_key, temperature=0.2)

vector_store = PineconeVectorStore(
    index=pc_index,
    embedding=embedder,
    text_key="text",
    namespace=NAMESPACE or None,
)

# -----------------------------
# Helpers
# -----------------------------
ISO_PATTERNS = ("%Y-%m-%dT%H:%M:%S%z","%Y-%m-%dT%H:%M:%S.%f%z","%Y-%m-%dT%H:%M:%S","%Y-%m-%d")

def parse_iso(s:str):
    s=(s or"").replace("Z","+00:00")
    for p in (None,*ISO_PATTERNS):
        try: return datetime.fromisoformat(s) if p is None else datetime.strptime(s,p)
        except: pass
    return None

def recent_enough(iso:str, now:datetime, days:Optional[int])->bool:
    if days is None: return True
    dt=parse_iso(iso) or datetime(1970,1,1,tzinfo=timezone.utc)
    if dt.tzinfo is None: dt=dt.replace(tzinfo=timezone.utc)
    return (now-dt)<=timedelta(days=days)

def clean(s:str)->str: return re.sub(r"\s+"," ",(s or "")).strip()

def trim(texts:List[str], limit:int)->str:
    out,total=[],0
    for t in texts:
        t=t.strip()
        if not t: continue
        need=len(t)+2
        if total+need>limit:
            if not out and limit>0: out.append(t[:limit])
            break
        out.append(t); total+=need
    return "\n\n".join(out)

# Much broader pricing detector (URL, title, AND snippet content)
PRICING_PAT = re.compile(
    r"(pricing|price|prices|fees?|rates?|plan[s]?|tier[s]?|package|packaging|bundle|"
    r"subscription|per[- ](seat|user|month|request|call|credit|token)|"
    r"usage[- ]?based|consumption|metered|overage[s]?|discount[s]?|"
    r"free|freemium|trial|paywall|rate[- ]card|list[- ]price|msrp)",
    re.I,
)

def looks_like_pricing(md:Dict[str,Any], text_sample:str)->bool:
    url = (md or {}).get("url","")
    title = (md or {}).get("title","")
    hay = " ".join([url, title, text_sample or ""])
    if PRICING_PAT.search(hay):
        return True
    if (md or {}).get("source","") in {"pricing_page","pricing"}:
        return True
    return False

# -----------------------------
# Retrieval
# -----------------------------
def fetch_docs(q:str, now:datetime, days:Optional[int]):
    docs=vector_store.max_marginal_relevance_search(q, k=TOP_K, fetch_k=FETCH_K, lambda_mult=0.45)
    out=[]
    for d in docs:
        md=d.metadata or {}
        text=" ".join((d.page_content or "").split())
        if not looks_like_pricing(md, text[:600]):   # <-- check snippet too
            continue
        if recent_enough(clean(md.get("date","")), now, days):
            out.append({
                "text": text[:2000],
                "url": clean(md.get("url","")),
                "title": clean(md.get("title","")),
                "date": clean(md.get("date","")),
                "id": clean(md.get("id","")),
            })
    return out

def build_context_and_sources(queries:List[str], now:datetime, days:Optional[int],
                              agg_seen_texts:Optional[set]=None,
                              agg_seen_urls:Optional[set]=None,
                              agg_blocks:Optional[list]=None,
                              agg_sources:Optional[list]=None):
    """
    Accumulate across calls/windows so we don't lose earlier results.
    """
    seen_texts = agg_seen_texts or set()
    seen_urls  = agg_seen_urls or set()
    blocks     = agg_blocks or []
    sources    = agg_sources or []

    for q in queries:
        for rec in fetch_docs(q, now, days):
            snip=rec["text"]
            if not snip or snip in seen_texts: 
                continue
            seen_texts.add(snip)

            if rec.get("url"):
                blocks.append(snip)
                if rec["url"] not in seen_urls:
                    sources.append({"title": rec.get("title") or "(untitled)","url":rec["url"]})
                    seen_urls.add(rec["url"])
            if len(blocks)>=MAX_SNIPPETS: 
                break
        if len(blocks)>=MAX_SNIPPETS: 
            break

    # no trimming here (we may continue accumulating); return raw lists + sets
    return seen_texts, seen_urls, blocks, sources

# -----------------------------
# Schemas & Prompts
# -----------------------------
REPORT_SCHEMA: Dict[str,Any] = {
    "title": "PricingWatch",
    "type": "object",
    "properties": {
        "as_of_utc": {"type":"string"},
        "window_start_utc": {"type":"string"},
        "window_end_utc": {"type":"string"},
        "num_sources": {"type":"number"},
        "pricing_changes": {
            "type":"array","minItems":0,"maxItems":50,
            "items":{
                "type":"object",
                "properties":{
                    "company":{"type":"string"},
                    "products_affected":{"type":"array","items":{"type":"string"}},
                    "change_type":{"type":"string"},
                    "summary":{"type":"string"},
                    "implications":{"type":"string"},
                    "evidence_urls":{"type":"array","minItems":1,"items":{"type":"string"}},
                },
                "required":["company","summary","evidence_urls"],"additionalProperties":False
            }
        },
        "highlights":{"type":"array","minItems":0,"maxItems":10,"items":{"type":"string"}}
    },
    "required":["pricing_changes"],"additionalProperties":False
}

OUTLINE_SCHEMA: Dict[str,Any] = {
    "title":"PricingOutline",
    "type":"object",
    "properties":{"focus_questions":{"type":"array","minItems":3,"maxItems":10,"items":{"type":"string"}}},
    "required":["focus_questions"],"additionalProperties":False
}

planner = llm.with_structured_output(OUTLINE_SCHEMA).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
writer  = llm.with_structured_output(REPORT_SCHEMA, max_tokens=WRITER_MAX_TOKENS).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)

PLAN_PROMPT = PromptTemplate.from_template(
    "Generate 3–10 focus_questions to detect pricing/packaging changes in the last {days} days. "
    "Cover new tiers, metered pricing, usage-based/consumption pricing, per-seat changes, bundles, discounts, "
    "credits/overage, token/credit pricing, and paywall/freemium shifts. "
    "Audience: investors/founders. Return only focus_questions."
)

WRITER_PROMPT = PromptTemplate.from_template(
    "You are a pricing analyst. Using ONLY the CONTEXT, produce a Pricing Watch report. "
    "Fill the schema. Use URLs ONLY from the SOURCES list. No fabrication.\n\n"
    "=== SOURCES (title — url) ===\n{sources}\n\n"
    "=== CONTEXT START ===\n{context}\n=== CONTEXT END ==="
)

# Extra seed queries to boost recall (merged with model plan)
SEED_QUERIES = [
    "pricing change", "price increase", "price cut", "subscription price",
    "new pricing tier", "enterprise plan pricing", "usage-based pricing", "consumption pricing",
    "per-seat price", "token pricing", "credit pricing", "overage fees", "bundle pricing",
    "discounted plan", "freemium to paid", "rate card", "list price"
]

# -----------------------------
# Main
# -----------------------------
def run(topic_hint: Optional[str]=None)->Dict[str,Any]:
    now=datetime.now(timezone.utc)

    outline=(PLAN_PROMPT|planner).invoke({"days":DAYS_BACK})
    focus=(outline.get("focus_questions") or [])
    if topic_hint:
        focus=[f"{q} {topic_hint}" for q in focus]
    # merge with manual seeds (dedupe)
    merged = []
    seen = set()
    for q in (focus + SEED_QUERIES):
        if q not in seen:
            merged.append(q); seen.add(q)

    # Accumulate across windows until we hit MIN_SOURCES_GOAL (or run out)
    seen_texts, seen_urls, blocks, sources = set(), set(), [], []
    for win in [DAYS_BACK] + BACKOFF_WINDOWS:
        seen_texts, seen_urls, blocks, sources = build_context_and_sources(
            merged, now, win, seen_texts, seen_urls, blocks, sources
        )
        log.info("retrieval: days_back=%s -> total_sources=%d", win, len(sources))
        if len(sources) >= MIN_SOURCES_GOAL or len(blocks) >= MAX_SNIPPETS:
            break

    # Build prompt strings
    context = trim(blocks, MAX_CONTEXT_CHARS)
    sources_str = "\n".join(f"- {s['title']} — {s['url']}" for s in sources)
    allowed_urls_ordered = [s["url"] for s in sources if s.get("url")]
    allowed_set = set(allowed_urls_ordered)

    # Writer
    try:
        rpt=(WRITER_PROMPT|writer).invoke({"context":context,"sources":sources_str})
    except Exception as e:
        log.warning("writer retry half context: %s", str(e)[:160])
        half=context[:max(1200,len(context)//2)]
        rpt=(WRITER_PROMPT|writer).invoke({"context":half,"sources":sources_str})

    # Enforce URL allowlist
    for ch in rpt.get("pricing_changes",[]) or []:
        urls=[u for u in ch.get("evidence_urls",[]) if u in allowed_set]
        if not urls: urls = allowed_urls_ordered[:2]
        ch["evidence_urls"]=urls[:3]

    rpt["as_of_utc"]=now.isoformat()
    rpt["window_start_utc"]=(now - timedelta(days=DAYS_BACK)).isoformat()
    rpt["window_end_utc"]=now.isoformat()
    rpt["num_sources"]=len(allowed_urls_ordered)
    return rpt


payload=run()
with open("pricing_watch.json","w",encoding="utf-8") as f:
    json.dump(payload,f,indent=2,ensure_ascii=False)
print("Saved pricing_watch.json")
