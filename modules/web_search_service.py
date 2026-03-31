# modules/web_search_service.py
import os
from tavily import TavilyClient, AsyncTavilyClient
import asyncio
import httpx
from bs4 import BeautifulSoup, NavigableString
from config import TAVILY_MAX_RESULTS


def convert_table_to_markdown(table_tag) -> str:
    """Convert a BeautifulSoup <table> tag to Markdown string."""
    rows = []

    # Headers
    headers = [th.get_text(strip=True) for th in table_tag.find_all("th")]
    if headers:
        rows.append("| " + " | ".join(headers) + " |")
        rows.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Data rows
    for tr in table_tag.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cells:
            rows.append("| " + " | ".join(cells) + " |")

    return "\n" + "\n".join(rows) + "\n"


async def acrawl_beautifulsoup(url: str) -> dict:
    """
    Crawl a URL using httpx + BeautifulSoup.
    Returns: {'raw_html': ..., 'clean_text': ...}
    """
    print(f"[WebSearch] Crawling: {url}")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            body = soup.find("main") or soup.find("body")
            if not body:
                return {"raw_html": "", "clean_text": ""}

            raw_html = str(body)

            # Remove junk tags
            for tag in body(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                tag.decompose()

            # Convert tables to Markdown
            tables = body.find_all("table")
            if tables:
                print(f"[WebSearch] Converting {len(tables)} tables to Markdown")
                for table in tables:
                    table.replace_with(NavigableString(convert_table_to_markdown(table)))

            # Extract and clean text
            text = body.get_text(separator="\n", strip=True)
            clean_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

            return {
                "raw_html": raw_html,
                "clean_text": clean_text[:15000],  # Limit to avoid prompt overflow
            }

    except httpx.HTTPStatusError as e:
        print(f"[WebSearch] ⚠️ HTTP error crawling {url}: {e.response.status_code}")
        return {"raw_html": "", "clean_text": f"Lỗi: Không thể truy cập URL (mã lỗi {e.response.status_code})."}
    except Exception as e:
        print(f"[WebSearch] ❌ Error crawling {url}: {e}")
        return {"raw_html": "", "clean_text": "Lỗi: Không thể crawl URL."}


class WebSearchService:
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        self.tavily_client = TavilyClient(api_key=api_key)
        self.async_tavily_client = AsyncTavilyClient(api_key=api_key)
        self.max_results = TAVILY_MAX_RESULTS
        print("[WebSearch] ✅ Tavily search service initialized")

    def search_tavily(self, query: str) -> list:
        """Synchronous Tavily search."""
        try:
            print(f"[WebSearch] Searching: {query}")
            response = self.tavily_client.search(
                query=query,
                max_results=self.max_results,
                search_depth="advanced",
            )
            return response.get("results", [])
        except Exception as e:
            print(f"[WebSearch] ❌ Search error: {e}")
            return []

    async def asearch_tavily(self, query: str) -> list:
        """Async Tavily search."""
        try:
            print(f"[WebSearch] Async searching: {query}")
            response = await self.async_tavily_client.search(
                query=query,
                max_results=self.max_results,
                search_depth="advanced",
            )
            return response.get("results", [])
        except Exception as e:
            print(f"[WebSearch] ❌ Async search error: {e}")
            return []

    def crawl_tavily(self, url: str) -> dict:
        """Sync wrapper for BeautifulSoup crawl."""
        try:
            return asyncio.run(acrawl_beautifulsoup(url))
        except Exception as e:
            print(f"[WebSearch] ❌ Sync crawl error: {e}")
            return {"raw_html": "", "clean_text": ""}

    async def acrawl_tavily(self, url: str) -> dict:
        """Async BeautifulSoup crawl."""
        return await acrawl_beautifulsoup(url)


# Singleton instance
web_search_service = WebSearchService()
