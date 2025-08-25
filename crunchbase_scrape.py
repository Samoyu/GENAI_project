import os
import re
import time
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI                 # NVIDIA via OpenAI-compatible endpoint
from pinecone import Pinecone, ServerlessSpec

# -----------------------------
# Config
# -----------------------------
INDEX_NAME        = "techcrunch-articles"      # reuse the same index
EMBED_MODEL       = "nvidia/llama-3.2-nv-embedqa-1b-v2"
NVIDIA_BASE_URL   = "https://integrate.api.nvidia.com/v1"
EMBED_DIM         = 2048
UPSERT_BATCH_SIZE = 100
SCRAPE_DELAY_SEC  = 1.0
PAGES_TO_SCRAPE   = 50
USER_AGENT        = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)
BASE = "https://news.crunchbase.com/"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("crunchbase_news_pipeline")

# -----------------------------
# Utils
# -----------------------------
def chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks, i = [], 0
    while i < len(text):
        j = min(i + max_chars, len(text))
        k = text.rfind(". ", i, j)  # try to break at sentence
        if k == -1 or j == len(text):
            k = j
        else:
            k += 1  # include the period
        chunks.append(text[i:k].strip())
        i = max(k - overlap, k)
    return [c for c in chunks if c]

def embed_texts_index_mode(nv_client, texts):
    resp = nv_client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=EMBED_DIM,
        extra_body={"input_type": "passage"}  # << goes straight into JSON body
    )
    vecs = [d.embedding for d in resp.data]
    assert all(len(v) == EMBED_DIM for v in vecs), f"Embedding dim mismatch (expected {EMBED_DIM})"
    return vecs


def _clean(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def make_id(url: str, idx: int) -> str:
    # Hash the full URL to avoid collisions; prefix to identify source
    h = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
    return f"cb_{h}_{idx}"

# -----------------------------
# Scraper
# -----------------------------
class CrunchbaseNewsScraper:
    def __init__(self, delay: float = 1.0, session: Optional[requests.Session] = None):
        self.delay = delay
        self.sess = session or requests.Session()
        self.sess.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})

    def get(self, url: str) -> Optional[str]:
        try:
            r = self.sess.get(url, timeout=20)
            if r.status_code != 200:
                logger.warning("GET %s -> %s", url, r.status_code)
                return None
            time.sleep(self.delay)
            return r.text
        except Exception as e:
            logger.warning("GET %s failed: %s", url, e)
            return None

    def listing_urls(self, page: int) -> List[str]:
        url = BASE if page == 1 else urljoin(BASE, f"page/{page}/")
        html = self.get(url)
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")

        cand = []
        for a in soup.select("h2 a[href], h3 a[href]"):
            href = a.get("href") or ""
            if not href:
                continue
            full = urljoin(BASE, href)
            path = urlparse(full).path.rstrip("/")
            if not full.startswith(BASE):
                continue
            if any(seg in path.split("/")[:2] for seg in ["page", "tag", "sections", "category"]):
                continue
            cand.append(full)

        seen, out = set(), []
        for u in cand:
            if u not in seen:
                seen.add(u)
                out.append(u)
        logger.info("page %s -> %d article URLs", page, len(out))
        return out

    def parse_article(self, url: str) -> Optional[Dict[str, Any]]:
        html = self.get(url)
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")

        title = None
        h1 = soup.find("h1")
        if h1:
            title = _clean(h1.get_text(" ", strip=True))
        if not title:
            og = soup.find("meta", {"property": "og:title"})
            if og and og.get("content"):
                title = _clean(og["content"])

        author = None
        a_meta = soup.find("meta", {"name": "author"})
        if a_meta and a_meta.get("content"):
            author = _clean(a_meta["content"])
        if not author:
            a_rel = soup.select_one('a[rel="author"], a.author, span.author, .byline a')
            if a_rel:
                author = _clean(a_rel.get_text(" ", strip=True))

        date = None
        t = soup.find("time")
        if t and (t.get("datetime") or t.get_text(strip=True)):
            date = t.get("datetime") or t.get_text(strip=True)
        if not date:
            m = soup.find("meta", {"property": "article:published_time"})
            if m and m.get("content"):
                date = m["content"]

        body = ""
        candidates = [
            "article .entry-content","article .post-content","article .single-content","article .content",
            "article",".entry-content",".post-content",".single-content",".content__body",".content","#content",
        ]
        for sel in candidates:
            node = soup.select_one(sel)
            if not node:
                continue
            parts = []
            for p in node.find_all(["p", "li", "h2", "h3"], recursive=True):
                txt = _clean(p.get_text(" ", strip=True))
                if txt:
                    parts.append(txt)
            if len(" ".join(parts)) > 400:
                body = " ".join(parts)
                break
        if not body:
            parts = [_clean(p.get_text(" ", strip=True)) for p in soup.find_all("p")]
            body = " ".join([p for p in parts if p])[:20000]

        if not title or not body:
            logger.warning("skip (missing title/body): %s", url)
            return None

        return {"url": url, "title": title, "author": author or "", "date": date or "", "content": body}

    def scrape_multiple_pages(self, start_page: int = 1, max_pages: int = 5) -> List[Dict[str, Any]]:
        seen, out = set(), []
        for p in range(start_page, start_page + max_pages):
            urls = self.listing_urls(p)
            if not urls:
                break
            for u in urls:
                if u in seen:
                    continue
                seen.add(u)
                art = self.parse_article(u)
                if art and art.get("content"):
                    out.append(art)
        return out

# -----------------------------
# Keys
# -----------------------------
load_dotenv()
nvidia_key   = os.getenv("nvidia_key")
pinecone_key = os.getenv("pinecone_key")
namespace    = os.getenv("PINECONE_NAMESPACE", "") or None
assert nvidia_key and pinecone_key, "Add nvidia_key and pinecone_key to .env"

# -----------------------------
# Clients
# -----------------------------
nvidia_client = OpenAI(api_key=nvidia_key, base_url=NVIDIA_BASE_URL)
pc = Pinecone(api_key=pinecone_key)

# -----------------------------
# Index (create if missing - robust across SDK versions)
# -----------------------------
existing = pc.list_indexes()
try:
    existing_names = set(existing.names())  # new SDK
except AttributeError:
    existing_names = {getattr(i, "name", i) for i in (existing or [])}  # fallback

if INDEX_NAME not in existing_names:
    logger.info(f"Creating Pinecone index '{INDEX_NAME}' (dim={EMBED_DIM})...")
    try:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        time.sleep(5)  # control plane settle
    except Exception as e:
        if getattr(e, "status", None) == 409:
            logger.info("Index already exists (409). Proceeding.")
        else:
            raise
else:
    logger.info(f"Index '{INDEX_NAME}' already exists.")
index = pc.Index(INDEX_NAME)

# -----------------------------
# Scrape Crunchbase News
# -----------------------------
scraper = CrunchbaseNewsScraper(delay=SCRAPE_DELAY_SEC)
articles = scraper.scrape_multiple_pages(start_page=1, max_pages=PAGES_TO_SCRAPE)
logger.info("Total Crunchbase News articles with content: %d", len(articles))

# -----------------------------
# Upsert
# -----------------------------
vectors = []
for art in articles:
    chunks = [c for c in chunk_text(art["content"]) if c.strip()]
    if not chunks:
        continue
    embeddings = embed_texts_index_mode(nvidia_client, chunks)  # uses input_type="passage"

    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vid = make_id(art["url"], idx)
        meta = {
            "text": chunk,
            "title": art.get("title", ""),
            "url": art.get("url", ""),
            "author": art.get("author", ""),
            "date": art.get("date", ""),
            "chunk_index": idx,
            "source": "crunchbase_news",
        }
        vectors.append((vid, emb, meta))

logger.info("Upserting %d vectors...", len(vectors))
for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
    batch = vectors[i:i + UPSERT_BATCH_SIZE]
    if not batch:
        continue
    index.upsert(vectors=batch, namespace=namespace)
    logger.info("Upserted batch %d–%d (%d vectors).", i, i + len(batch) - 1, len(batch))
    time.sleep(0.5)

# -----------------------------
# Verify
# -----------------------------
try:
    stats = index.describe_index_stats()
    if namespace:
        ns_stats = stats.get("namespaces", {}).get(namespace, {})
        logger.info("Stats for namespace '%s': %s", namespace, json.dumps(ns_stats, indent=2))
    else:
        logger.info("Index stats: %s", json.dumps(stats, indent=2))
except Exception as e:
    logger.warning("describe_index_stats failed: %s", e)

logger.info(
    "Done. Inserted Crunchbase News chunks into '%s'%s",
    INDEX_NAME, f" (namespace={namespace})" if namespace else ""
)
