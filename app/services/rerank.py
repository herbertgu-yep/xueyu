"""Cross-encoder 重排：本地模型直接给「查询—片段」打分。

相比让大模型读一遍候选再排序，cross-encoder 是专门为相关性训练的双塔合体模型：
输入一对文本，直接输出一个相关性分数，不生成文字，因此快两个数量级、也不花 API 费用。

模型按需下载，加载一次常驻；不可用时返回 None，由调用方回退到别的重排方式。
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

log = logging.getLogger("xueyu.rerank")

DEFAULT_MODEL = os.environ.get("XUEYU_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

_lock = threading.Lock()
_model: Any = None
_failed: Optional[str] = None


def model_name() -> str:
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
    return _load() is not None


def unavailable_reason() -> Optional[str]:
    _load()
    return _failed


def rerank(query: str, chunks: list[dict], *, limit: Optional[int] = None,
           text_limit: int = 1200) -> Optional[list[dict]]:
    """按相关性重排候选；模型不可用或推理失败返回 None。

    limit=None 重排整池；产品链路传 12——实测把候选从 12 扩到 40，
    top-3 命中率没有提升，耗时却从约 2s 涨到约 10s。
    """
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
    return out + chunks[limit:]
