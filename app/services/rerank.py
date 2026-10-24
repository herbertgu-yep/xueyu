"""检索重排：本地 cross-encoder 或远程 rerank API。

- 本地：sentence-transformers 的 CrossEncoder，直接给「查询—片段」打相关性分；
- 远程：DashScope 原生 rerank 协议（阿里云百炼 / 千问 MaaS）。

配置在设置表 rerank 里：mode=local 用本地模型，mode=api 用远程接口；
远程接口失败自动退回本地模型，本地模型不可用则返回 None，由调用方决定下一步。
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional

log = logging.getLogger("xueyu.rerank")

DEFAULT_MODEL = os.environ.get("XUEYU_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

_lock = threading.Lock()
_model: Any = None
_failed: Optional[str] = None
_api_failed: Optional[str] = None
_api_failed_at: float = 0.0
_API_RETRY_SECONDS = 60


def _cfg() -> dict:
    from app.services import settings_store

    return settings_store.get("rerank")


def _api_usable(cfg: dict) -> bool:
    return (cfg.get("mode") == "api" and bool(cfg.get("api_base_url"))
            and bool(cfg.get("api_key")) and bool(cfg.get("api_model")))


def model_name() -> str:
    cfg = _cfg()
    if _api_usable(cfg):
        return str(cfg.get("api_model"))
    return DEFAULT_MODEL


def _load() -> Any:
    """懒加载 cross-encoder；失败后记住原因，避免每次查询都重试下载。"""
    global _model, _failed
    if _model is not None or _failed:
        return _model
    with _lock:
        if _model is not None or _failed:
            return _model
        try:
            from sentence_transformers import CrossEncoder

            _model = CrossEncoder(DEFAULT_MODEL, max_length=512)
            log.info("rerank model loaded: %s", DEFAULT_MODEL)
        except Exception as e:  # noqa: BLE001
            _failed = f"{type(e).__name__}: {e}"[:200]
            log.warning("rerank model unavailable: %s", _failed)
        return _model


def available() -> bool:
    if _api_usable(_cfg()):
        return True
    return _load() is not None


def using_api() -> bool:
    return _api_usable(_cfg())


def unavailable_reason() -> Optional[str]:
    if _api_usable(_cfg()):
        return _api_failed
    _load()
    return _failed


def reset_api_state() -> None:
    """设置改了之后清掉失败缓存，下一次调用立刻按新配置重试。"""
    global _api_failed, _api_failed_at
    _api_failed = None
    _api_failed_at = 0.0


def _normalize(ranked: list[dict], chunks: list[dict], limit: Optional[int]) -> list[dict]:
    """API/本地只重排了前 limit 个，这里补回没参与精排的尾部。"""
    return ranked + (chunks[limit:] if limit else [])


def _rerank_api(query: str, chunks: list[dict], cfg: dict, limit: Optional[int],
                text_limit: int) -> Optional[list[dict]]:
    """DashScope 原生 rerank；失败返回 None（由调用方退回本地模型）。"""
    global _api_failed, _api_failed_at
    if _api_failed and time.time() - _api_failed_at < _API_RETRY_SECONDS:
        return None

    import httpx

    cand = chunks[:limit] if limit else list(chunks)
    if not cand:
        return []
    url = (str(cfg["api_base_url"]).rstrip("/")
           + "/services/rerank/text-rerank/text-rerank")
    body = {
        "model": str(cfg["api_model"]),
        "input": {"query": query,
                  "documents": [str(c.get("text") or "")[:text_limit] for c in cand]},
        "parameters": {"top_n": len(cand), "return_documents": False},
    }
    try:
        resp = httpx.post(url, headers={"Authorization": f"Bearer {cfg['api_key']}"},
                          json=body, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:160]}")
        results = ((resp.json().get("output") or {}).get("results") or [])
        by_index: dict[int, dict] = {}
        for item in results:
            i = int(item["index"])
            if 0 <= i < len(cand):
                by_index[i] = {**cand[i], "rerank_score": round(float(item["relevance_score"]), 4)}
        if not by_index:
            raise RuntimeError("返回结果为空")
    except Exception as e:  # noqa: BLE001
        _api_failed = f"{type(e).__name__}: {e}"[:200]
        _api_failed_at = time.time()
        log.warning("rerank api failed, falling back to local: %s", _api_failed)
        return None

    _api_failed = None
    ranked = [by_index[i] for i in sorted(by_index, key=lambda i: -by_index[i]["rerank_score"])]
    ranked.extend(c for i, c in enumerate(cand) if i not in by_index)
    return _normalize(ranked, chunks, limit)


def _rerank_local(query: str, chunks: list[dict], limit: Optional[int],
                  text_limit: int) -> Optional[list[dict]]:
    global _failed
    model = _load()
    if model is None or not chunks:
        return None
    cand = chunks[:limit] if limit else list(chunks)
    pairs = [(query, str(c.get("text") or "")[:text_limit]) for c in cand]
    try:
        scores = model.predict(pairs, batch_size=8, show_progress_bar=False)
    except Exception as e:  # noqa: BLE001
        _failed = f"{type(e).__name__}: {e}"[:200]
        log.warning("rerank inference failed: %s", _failed)
        return None
    ranked = sorted(zip(cand, scores), key=lambda x: -float(x[1]))
    out = [{**c, "rerank_score": round(float(s), 4)} for c, s in ranked]
    return _normalize(out, chunks, limit)


def rerank(query: str, chunks: list[dict], *, limit: Optional[int] = None,
           text_limit: int = 1200) -> Optional[list[dict]]:
    """按相关性重排候选；所有后端都不可用时返回 None。

    limit=None 重排整池；产品链路传 12——实测把候选从 12 扩到 40，
    top-3 命中率没有提升，耗时却从约 2s 涨到约 10s。
    """
    cfg = _cfg()
    if _api_usable(cfg):
        out = _rerank_api(query, chunks, cfg, limit, text_limit)
        if out is not None:
            return out
    return _rerank_local(query, chunks, limit, text_limit)
