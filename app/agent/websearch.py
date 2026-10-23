"""联网检索：DuckDuckGo 搜索 + 网页正文抽取（httpx + trafilatura）。"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx
import trafilatura

log = logging.getLogger("xueyu.websearch")

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
MAX_REDIRECTS = 3
MAX_BYTES = 2_000_000


# 明确封掉的内网段：回环、RFC1918、链路本地（含云元数据）、CGNAT、IPv6 私有段
_BLOCKED_NETS = tuple(
    __import__("ipaddress").ip_network(n)
    for n in (
        "0.0.0.0/8", "10.0.0.0/8", "127.0.0.0/8", "169.254.0.0/16",
        "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10",
        "::1/128", "fc00::/7", "fe80::/10", "ff00::/8",
    )
)
# 说明：198.18.0.0/15 是部分代理软件（Clash/Surge 等）的 DNS 假地址段，故意放行，
# 否则开了代理的用户所有网页抓取都会被拦。


def _host_allowed(host: str) -> bool:
    """只放行公网主机，挡掉回环、内网、链路本地（云元数据）等地址。"""
    import ipaddress
    import socket

    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if any(ip in net for net in _BLOCKED_NETS if net.version == ip.version):
            return False
    return True


def safe_url(url: str) -> bool:
    from urllib.parse import urlparse

    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return False
    return _host_allowed(p.hostname or "")


async def _get_html(url: str, *, timeout: float, client: httpx.AsyncClient) -> Optional[str]:
    """逐跳校验重定向后抓取（httpx 自动跟随会绕过首跳校验）。"""
    from urllib.parse import urljoin

    current = url
    for _ in range(MAX_REDIRECTS + 1):
        if not safe_url(current):
            log.warning("blocked unsafe url: %s", current)
            return None
        resp = await client.get(current, follow_redirects=False)
        if resp.is_redirect:
            current = urljoin(current, resp.headers.get("location", ""))
            continue
        if resp.status_code >= 400:
            return None
        if "text/html" not in resp.headers.get("content-type", "html"):
            return None
        content = resp.content
        if len(content) > MAX_BYTES:
            content = content[:MAX_BYTES]
        return content.decode(resp.encoding or "utf-8", errors="replace")
    return None


def _extract_text(html: str, cap: int) -> Optional[str]:
    try:
        text = trafilatura.extract(html, include_comments=False, include_tables=True,
                                   no_fallback=False, favor_recall=True)
    except Exception:  # noqa: BLE001
        text = None
    return (text or "").strip()[:cap] or None


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
    from urllib.parse import urljoin

    if not safe_url(url):
        return None
    try:
        current = url
        with httpx.Client(follow_redirects=False, timeout=timeout, headers={"User-Agent": UA}) as client:
            for _ in range(MAX_REDIRECTS + 1):
                if not safe_url(current):
                    return None
                resp = client.get(current)
                if resp.is_redirect:
                    current = urljoin(current, resp.headers.get("location", ""))
                    continue
                if resp.status_code >= 400:
                    return None
                if "text/html" not in resp.headers.get("content-type", "html"):
                    return None
                html = resp.content[:MAX_BYTES].decode(resp.encoding or "utf-8", errors="replace")
                return _extract_text(html, cap)
    except Exception:  # noqa: BLE001
        return None
    return None


async def afetch_page(url: str, *, timeout: float = 12.0, cap: int = 6000) -> Optional[str]:
    """抓取并抽取网页正文；失败返回 None。"""
    if not safe_url(url):
        return None
    try:
        async with httpx.AsyncClient(follow_redirects=False, timeout=timeout,
                                     headers={"User-Agent": UA}) as client:
            html = await _get_html(url, timeout=timeout, client=client)
    except Exception:  # noqa: BLE001
        return None
    if not html:
        return None
    return await asyncio.to_thread(_extract_text, html, cap)
