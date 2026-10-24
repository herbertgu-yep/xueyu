"""联网检索：DuckDuckGo 搜索 + 网页正文抽取（httpx + trafilatura）。"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpcore
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


def _resolve_checked(host: str) -> list[str]:
    """解析主机并校验全部地址；任何一个是内网/回环就整体拒绝。"""
    import ipaddress
    import socket

    if not host:
        return []
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return []
    ips = []
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if any(ip in net for net in _BLOCKED_NETS if net.version == ip.version):
            return []
        ips.append(str(ip))
    return ips


def _host_allowed(host: str) -> bool:
    return bool(_resolve_checked(host))


def _proxy_configured() -> bool:
    import os

    return any(os.environ.get(k) for k in
               ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"))


class _PinnedBackend:
    """把 TCP 连接固定到校验过的 IP，避免 DNS 重绑定（TLS SNI 仍是原域名）。"""

    def __init__(self, host: str, ip: str, inner):
        self.host, self.ip, self.inner = host, ip, inner

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        target = self.ip if host == self.host else host
        return self.inner.connect_tcp(target, port, timeout, local_address, socket_options)

    def connect_unix_socket(self, path, timeout=None, socket_options=None):
        return self.inner.connect_unix_socket(path, timeout, socket_options)

    def sleep(self, seconds):
        return self.inner.sleep(seconds)


def _pinned_get_sync(url: str, timeout: float) -> Optional[tuple[int, dict, bytes]]:
    import ssl as _ssl
    from urllib.parse import urlparse

    p = urlparse(url)
    host = p.hostname or ""
    ips = _resolve_checked(host)
    if not ips:
        return None
    ctx = _ssl.create_default_context() if p.scheme == "https" else None
    pool = httpcore.ConnectionPool(
        ssl_context=ctx, network_backend=_PinnedBackend(host, ips[0], httpcore.SyncBackend()))
    try:
        resp = pool.request("GET", url, headers={"User-Agent": UA},
                            extensions={"timeout": {"connect": timeout, "read": timeout,
                                                    "write": timeout, "pool": timeout}})
        headers = {k.decode().lower(): v.decode() for k, v in resp.headers}
        status = resp.status
        content = bytes(resp.content)[:MAX_BYTES]   # pool.request 已把响应读完
        resp.close()
        return status, headers, content
    except Exception as e:  # noqa: BLE001
        log.warning("pinned fetch failed: %s (%s)", url, e)
        return None
    finally:
        pool.close()


async def _pinned_get_async(url: str, timeout: float) -> Optional[tuple[int, dict, bytes]]:
    import ssl as _ssl
    from urllib.parse import urlparse

    p = urlparse(url)
    host = p.hostname or ""
    ips = _resolve_checked(host)
    if not ips:
        return None
    ctx = _ssl.create_default_context() if p.scheme == "https" else None
    pool = httpcore.AsyncConnectionPool(
        ssl_context=ctx, network_backend=_PinnedBackend(host, ips[0], httpcore.AnyIOBackend()))
    try:
        resp = await pool.request("GET", url, headers={"User-Agent": UA},
                                  extensions={"timeout": {"connect": timeout, "read": timeout,
                                                          "write": timeout, "pool": timeout}})
        headers = {k.decode().lower(): v.decode() for k, v in resp.headers}
        status = resp.status
        content = bytes(resp.content)[:MAX_BYTES]
        await resp.aclose()
        return status, headers, content
    except Exception as e:  # noqa: BLE001
        log.warning("pinned fetch failed: %s (%s)", url, e)
        return None
    finally:
        await pool.aclose()


def safe_url(url: str) -> bool:
    from urllib.parse import urlparse

    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return False
    return _host_allowed(p.hostname or "")


async def _get_html(url: str, *, timeout: float, client: Optional[httpx.AsyncClient] = None) -> Optional[str]:
    """逐跳校验重定向后抓取；无代理时把连接固定到已校验的 IP。"""
    from urllib.parse import urljoin

    current = url
    for _ in range(MAX_REDIRECTS + 1):
        if not safe_url(current):
            log.warning("blocked unsafe url: %s", current)
            return None
        if client is None:
            got = await _pinned_get_async(current, timeout)
            if not got:
                return None
            status, headers, content = got
        else:
            resp = await client.get(current, follow_redirects=False)
            status = resp.status_code
            headers = {k.lower(): v for k, v in resp.headers.items()}
            content = resp.content[:MAX_BYTES]
            if resp.is_redirect:
                current = urljoin(current, headers.get("location", ""))
                continue
            if status < 400:
                return content.decode(resp.encoding or "utf-8", errors="replace")
            return None
        if 300 <= status < 400 and headers.get("location"):
            current = urljoin(current, headers["location"])
            continue
        if status >= 400:
            return None
        if "text/html" not in headers.get("content-type", "html"):
            return None
        return content.decode("utf-8", errors="replace")
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
    pinned = not _proxy_configured()
    try:
        current = url
        client = None if pinned else httpx.Client(
            follow_redirects=False, timeout=timeout, headers={"User-Agent": UA})
        try:
            for _ in range(MAX_REDIRECTS + 1):
                if not safe_url(current):
                    return None
                if client is None:
                    got = _pinned_get_sync(current, timeout)
                    if not got:
                        return None
                    status, headers, content = got
                    if 300 <= status < 400 and headers.get("location"):
                        current = urljoin(current, headers["location"])
                        continue
                    if status >= 400 or "text/html" not in headers.get("content-type", "html"):
                        return None
                    return _extract_text(content.decode("utf-8", errors="replace"), cap)
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
        finally:
            if client is not None:
                client.close()
    except Exception as e:  # noqa: BLE001
        log.warning("fetch failed: %s (%s)", url, e)
        return None
    return None


async def afetch_page(url: str, *, timeout: float = 12.0, cap: int = 6000) -> Optional[str]:
    """抓取并抽取网页正文；失败返回 None。"""
    if not safe_url(url):
        return None
    try:
        if _proxy_configured():
            async with httpx.AsyncClient(follow_redirects=False, timeout=timeout,
                                         headers={"User-Agent": UA}) as client:
                html = await _get_html(url, timeout=timeout, client=client)
        else:
            html = await _get_html(url, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        log.warning("fetch failed: %s (%s)", url, e)
        return None
    if not html:
        return None
    return await asyncio.to_thread(_extract_text, html, cap)
