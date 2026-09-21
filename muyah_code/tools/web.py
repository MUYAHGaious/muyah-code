"""WebSearch (DuckDuckGo, no API key) and WebFetch (HTML -> readable text, optional focused extraction)."""

from __future__ import annotations

import re
import warnings
from urllib.parse import urlparse

import httpx

from muyah_code.tools.base import NETWORK, Tool, ToolError, ToolResult, truncate_middle

USER_AGENT = "Mozilla/5.0 (compatible; MUYAH-CODE/1.0; +https://github.com/MUYAHGaious)"
MAX_FETCH_BYTES = 5 * 1024 * 1024
MAX_TEXT = 40000


def _ddgs():
    try:
        from ddgs import DDGS  # type: ignore[import-not-found]
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore[import-not-found]
        except ImportError as e:
            raise ToolError("Web search needs the 'ddgs' or 'duckduckgo_search' package: pip install ddgs") from e
    return DDGS


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, markdown-ish text) keeping headings, list items, code and links readable."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "form", "nav", "footer", "header", "aside"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    for pre in root.find_all("pre"):
        pre.replace_with(soup.new_string("\n```\n" + pre.get_text() + "\n```\n"))
    for code in root.find_all("code"):
        code.replace_with(soup.new_string("`" + code.get_text() + "`"))
    for level in range(1, 7):
        for h in root.find_all(f"h{level}"):
            h.replace_with(soup.new_string("\n\n" + "#" * level + " " + h.get_text(" ", strip=True) + "\n"))
    for li in root.find_all("li"):
        li.insert_before(soup.new_string("\n- "))
    for a in root.find_all("a"):
        href = a.get("href") or ""
        label = a.get_text(" ", strip=True)
        if href.startswith("http") and label and label != href:
            a.replace_with(soup.new_string(f"{label} ({href})"))
    text = root.get_text("\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return title, text.strip()


class WebSearchTool(Tool):
    name = "WebSearch"
    kind = NETWORK
    description = (
        "Search the web (DuckDuckGo). Returns titles, URLs and snippets. Use for current docs, error "
        "messages, library versions. Follow up with WebFetch on the most relevant URL."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "Number of results (default 6, max 15)"},
        },
        "required": ["query"],
    }

    def permission_subject(self, args, ctx):
        return args.get("query", "")

    def run(self, args, ctx):
        query = args["query"].strip()
        n = min(max(int(args.get("max_results") or 6), 1), 15)
        DDGS = _ddgs()
        try:
            with warnings.catch_warnings():  # library deprecation notices must not print into the UI
                warnings.simplefilter("ignore")
                with DDGS() as d:
                    results = list(d.text(query, max_results=n))
        except Exception as e:
            raise ToolError(f"Search failed: {e.__class__.__name__}: {e}") from e
        if not results:
            return ToolResult(f"No results for '{query}'.", summary="0 results")
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', '').strip()}\n   {r.get('href') or r.get('url', '')}\n"
                         f"   {(r.get('body') or '').strip()[:300]}")
        return ToolResult("\n".join(lines), summary=f"{len(results)} results")


class WebFetchTool(Tool):
    name = "WebFetch"
    kind = NETWORK
    description = (
        "Fetch a URL and return its readable text (HTML converted to markdown-like text; JSON/plain text as-is). "
        "Optionally pass 'prompt' describing what to extract - the page is then condensed to just that, which "
        "saves context on long pages. Only http(s) URLs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "http(s) URL"},
            "prompt": {"type": "string", "description": "What information to extract from the page"},
        },
        "required": ["url"],
    }

    def permission_subject(self, args, ctx):
        return args.get("url", "")

    def run(self, args, ctx):
        url = args["url"].strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ToolError(f"Invalid URL: {url}")
        try:
            with httpx.Client(follow_redirects=True, timeout=30, headers={"User-Agent": USER_AGENT}) as client:
                resp = client.get(url)
        except httpx.HTTPError as e:
            raise ToolError(f"Fetch failed: {e.__class__.__name__}: {e}") from e
        if resp.status_code >= 400:
            raise ToolError(f"HTTP {resp.status_code} for {url}")
        ctype = resp.headers.get("content-type", "")
        raw = resp.content[:MAX_FETCH_BYTES]
        if "html" in ctype or raw[:200].lstrip().lower().startswith((b"<!doctype html", b"<html")):
            title, text = html_to_text(raw.decode(resp.encoding or "utf-8", errors="replace"))
        elif any(t in ctype for t in ("json", "text", "xml", "javascript", "yaml")) or not ctype:
            title, text = "", raw.decode(resp.encoding or "utf-8", errors="replace")
        else:
            return ToolResult(f"{url} returned non-text content ({ctype}, {len(resp.content)} bytes).",
                              summary=ctype)
        header = f"URL: {resp.url}\n" + (f"Title: {title}\n" if title else "") + "\n"
        prompt = (args.get("prompt") or "").strip()
        llm = ctx.service("llm")
        if prompt and llm is not None and len(text) > 3000:
            window = int(ctx.service("context_window") or 16384)
            condensed = self._extract(llm, text, prompt, budget_chars=int(window * 3.5 * 0.55))
            if condensed:
                return ToolResult(header + condensed, summary=f"Extracted from {len(text)} chars")
        return ToolResult(header + truncate_middle(text, MAX_TEXT, "page"), summary=f"{len(text)} chars")

    @staticmethod
    def _extract(llm, text: str, prompt: str, budget_chars: int) -> str | None:
        snippet = text[: max(6000, budget_chars)]
        messages = [
            {"role": "system", "content": "You extract information from web pages. Answer only from the page. "
                                          "Quote code and commands exactly. Be concise."},
            {"role": "user", "content": f"Page content:\n<<<\n{snippet}\n>>>\n\nTask: {prompt}"},
        ]
        try:
            return llm.chat(messages, max_tokens=1500, temperature=0, purpose="webfetch").content.strip() or None
        except Exception:
            return None
