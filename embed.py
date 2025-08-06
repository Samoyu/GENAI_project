import os
import time
import json
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv         
from openai import OpenAI                  
from pinecone import Pinecone, ServerlessSpec 
from scrape import TechCrunchScraper
from utils import chunk_text, embed_texts_index_mode


# -----------------------------
# Config
# -----------------------------
INDEX_NAME = "techcrunch-articles"  
EMBED_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
LLM_MODEL = "meta/llama-3.1-8b-instruct"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
PAGES_TO_SCRAPE = 50
EMBED_DIM = 2048
UPSERT_BATCH_SIZE = 100
SCRAPE_DELAY_SEC = 1.0
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("techcrunch_pipeline")

# -------------------------
# Keys
# -------------------------
load_dotenv()  
nvidia_key = os.getenv("nvidia_key")
pinecone_key = os.getenv("pinecone_key")


# -------------------------
# Client
# -------------------------
nvidia_client = OpenAI(api_key=nvidia_key, base_url=NVIDIA_BASE_URL)
pc = Pinecone(api_key=pinecone_key)


# -------------------------
# Index
# -------------------------
existing = pc.list_indexes()
if INDEX_NAME not in existing:
    logger.info(f"Creating Pinecone index '{INDEX_NAME}' (dim={EMBED_DIM})...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait a bit for the control plane to be ready
    time.sleep(5)
else:
    logger.info(f"Index '{INDEX_NAME}' already exists.")
index = pc.Index(INDEX_NAME)


# -------------------------
# Scrape
# -------------------------
scraper = TechCrunchScraper(delay=SCRAPE_DELAY_SEC)
articles = scraper.scrape_multiple_pages(start_page=1, max_pages=PAGES_TO_SCRAPE)
logger.info(f"Total scraped articles with content: {len(articles)}")


# -------------------------
# Upsert
# -------------------------
vectors = []
external_store: Dict[str, str] = {}
for art in articles:
    art_id = art["url"].rstrip("/").split("/")[-1] or str(abs(hash(art["url"])))
    chunks = [c for c in chunk_text(art["content"]) if c.strip()]
    if not chunks:
        continue

    embeddings = embed_texts_index_mode(nvidia_client, chunks)
    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vid = f"{art_id}_{idx}"
        meta = {
            "text": chunk,               
            "title": art.get("title", ""),
            "url": art.get("url", ""),
            "author": art.get("author", ""),
            "date": art.get("date", ""),
            "chunk_index": idx
        }
        vectors.append((vid, emb, meta))
        external_store[vid] = chunk

# Upsert in batches
logger.info(f"Upserting {len(vectors)} vectors...")
for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
    batch = vectors[i:i + UPSERT_BATCH_SIZE]
    index.upsert(vectors=batch)
    logger.info(f"Upserted batch {i}-{i+len(batch)-1} ({len(batch)} vectors).")
    time.sleep(0.5)

# # Save external chunk store (optional)
# with open("chunks_external.json", "w", encoding="utf-8") as f:
#     json.dump(external_store, f, indent=2, ensure_ascii=False)
# logger.info("Saved chunk texts to chunks_external.json")
