import time
import logging
import requests
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from openai import OpenAI      
from utils import chunk_text

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("techcrunch_pipeline")


# -----------------------------
# Scraper
# -----------------------------
class TechCrunchScraper:
    def __init__(self, delay: float = 1.0):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        })
        self.delay = delay
        self.base_url = "https://techcrunch.com"
        self.articles_data: List[Dict[str, Any]] = []

    def _get(self, url: str):
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            logger.error(f"GET error for {url}: {e}")
            return None

    def extract_article_links(self, soup: BeautifulSoup) -> List[str]:
        links = set()
        # Primary: new TC list uses data-destinationlink
        for el in soup.find_all(attrs={"data-destinationlink": True}):
            href = el.get("data-destinationlink")
            if href and href.startswith("https://techcrunch.com/"):
                links.add(href)

        # Fallback: standard anchors under article cards
        for a in soup.select("a[href^='https://techcrunch.com/']"):
            href = a.get("href")
            if href and "/20" in href and "/tag/" not in href and href.startswith("https://techcrunch.com/"):
                links.add(href)

        return list(links)

    def scrape_article_content(self, url: str) -> Optional[Dict[str, Any]]:
        logger.info(f"Scraping article: {url}")
        resp = self._get(url)
        if not resp:
            return None

        soup = BeautifulSoup(resp.content, "html.parser")

        data: Dict[str, Any] = {
            "url": url,
            "title": "",
            "content": "",
            "author": "",
            "date": "",
            "tags": []
        }

        # Title
        title_el = soup.find("h1") or soup.find("title")
        if title_el:
            data["title"] = title_el.get_text(strip=True)

        # Content (robust selector)
        content_el = soup.select_one(
            "div.entry-content, div.wp-block-post-content, div[class*='wp-block-post-content']"
        )
        if content_el:
            paras = content_el.find_all(["p", "h2", "h3", "h4", "h5", "h6", "li"])
            text_blocks = []
            for p in paras:
                t = p.get_text(" ", strip=True)
                if t:
                    text_blocks.append(t)
            data["content"] = "\n\n".join(text_blocks)

        # Author
        author_el = soup.select_one("a[rel='author'], a.author, span.author")
        if author_el:
            data["author"] = author_el.get_text(strip=True)

        # Date
        time_el = soup.find("time")
        if time_el and time_el.get("datetime"):
            data["date"] = time_el.get("datetime")
        elif time_el:
            data["date"] = time_el.get_text(strip=True)

        # Tags
        tags = [a.get_text(strip=True) for a in soup.select("a[href*='/tag/']")]
        data["tags"] = list({t for t in tags if t})

        return data

    def scrape_latest_page(self, page_url: str) -> List[Dict[str, Any]]:
        logger.info(f"Scraping list page: {page_url}")
        resp = self._get(page_url)
        if not resp:
            return []

        soup = BeautifulSoup(resp.content, "html.parser")
        links = self.extract_article_links(soup)
        logger.info(f"Found {len(links)} article links")

        results = []
        for link in links:
            art = self.scrape_article_content(link)
            if art and art.get("content"):
                results.append(art)
                self.articles_data.append(art)
            time.sleep(self.delay)
        return results

    def scrape_multiple_pages(self, start_page: int = 1, max_pages: int = 2) -> List[Dict[str, Any]]:
        logger.info(f"Begin scraping pages {start_page}..{start_page + max_pages - 1}")
        for p in range(start_page, start_page + max_pages):
            page_url = f"{self.base_url}/latest/" if p == 1 else f"{self.base_url}/latest/page/{p}/"
            page_articles = self.scrape_latest_page(page_url)
            if not page_articles:
                logger.warning(f"No articles on page {p}; stopping.")
                break
            logger.info(f"Scraped {len(page_articles)} from page {p}")
            time.sleep(self.delay * 2)
        logger.info(f"Scraping complete. Total: {len(self.articles_data)}")
        return self.articles_data