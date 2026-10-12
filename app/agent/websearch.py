"""联网检索：DuckDuckGo 搜索 + 网页正文抽取（httpx + trafilatura）。"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx
import trafilatura

log = logging.getLogger("xueyu.websearch")

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"


def search_web(query: str, *, max_results: int = 6) -> list[dict]:
    """同步搜索（调用方用 asyncio.to_thread 包装）。返回 [{url,title,snippet}]。"""
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            try:
                rows = list(ddgs.text(query, region="cn-zh", safesearch="moderate",
                                      max_results=max_results))
            except TypeError:  # 兼容不同 ddgs 版本参数
                rows = list(ddgs.text(query, max_results=max_results))
        return [{"url": r.get("href") or r.get("url") or "",
                 "title": r.get("title") or "",
                 "snippet": r.get("body") or r.get("snippet") or ""}
                for r in rows if (r.get("href") or r.get("url"))]
    except Exception as e:  # noqa: BLE001 搜索是尽力而为，不让单次失败炸掉任务
        log.warning("web search failed: %s", e)
        return []


async def asearch_web(query: str, *, max_results: int = 6) -> list[dict]:
    return await asyncio.to_thread(search_web, query, max_results=max_results)


def fetch_page_sync(url: str, *, timeout: float = 12.0, cap: int = 6000) -> Optional[str]:
    """同步抓取正文（聊天等同步上下文使用）。"""
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent": UA}) as client:
            resp = client.get(url)
            if resp.status_code >= 400 or "text/html" not in resp.headers.get("content-type", "html"):
                return None
            html = resp.text
        text = trafilatura.extract(html, include_comments=False, include_tables=True,
                                   no_fallback=False, favor_recall=True)
        return (text or "").strip()[:cap] or None
    except Exception:  # noqa: BLE001
        return None


async def afetch_page(url: str, *, timeout: float = 12.0, cap: int = 6000) -> Optional[str]:
    """抓取并抽取网页正文；失败返回 None。"""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout,
                                     headers={"User-Agent": UA}) as client:
            resp = await client.get(url)
            if resp.status_code >= 400 or "text/html" not in resp.headers.get("content-type", "html"):
                return None
            html = resp.text
    except Exception:  # noqa: BLE001
        return None

    def _extract() -> Optional[str]:
        try:
            text = trafilatura.extract(html, include_comments=False, include_tables=True,
                                       no_fallback=False, favor_recall=True)
        except Exception:  # noqa: BLE001
            text = None
        return (text or "").strip()[:cap] or None

    return await asyncio.to_thread(_extract)
