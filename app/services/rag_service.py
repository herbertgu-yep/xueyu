"""本地向量化（RAG）：LlamaIndex + sentence-transformers + Chroma 持久化。

- 首次使用会下载 embedding 模型（约 130MB，缓存于 ~/.cache/huggingface）
- 单工作线程串行建索引；上传即入队，重建按钮清空后全部重嵌
- 更换 embedding 模型会自动触发全量重建（维度不兼容）
"""
from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict
import time
import traceback
from typing import Any, Optional

from sqlmodel import Session, select

from app.config import CHROMA_DIR
from app.storage.db import engine
from app.storage.models import Document
from app.services import settings_store

log = logging.getLogger("xueyu.rag")

_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "running": False,
    "current": "",       # 正在处理的文档
    "done": 0,
    "total": 0,
    "error": None,
    "last_finished_at": None,
}

_embed_model = None
_embed_lock = threading.RLock()  # _get_index 持锁时会再调 _get_embed_model，必须可重入
_index = None
_chroma_client = None
_chroma_collection = None
# Chroma 的 Rust 运行时对同路径多实例敏感：客户端/集合/索引全程单例
_chroma_lock = threading.RLock()


def status() -> dict:
    with _state_lock:
        st = dict(_state)
    with Session(engine) as s:
        docs = s.exec(select(Document)).all()
        st["documents"] = len(docs)
        st["indexed_documents"] = sum(1 for d in docs if d.indexed)
        st["chunks"] = sum(d.chunk_count for d in docs)
        st["failed"] = [d.title for d in docs if d.index_error]
    emb = settings_store.get("embedding")
    if (emb.get("mode") or "local") == "api":
        st["embedding_model"] = emb.get("api_model") or "text-embedding-3-small"
        st["embedding_mode"] = "api"
    else:
        st["embedding_model"] = emb.get("model") or "BAAI/bge-small-zh-v1.5"
        st["embedding_mode"] = "local"
    return st


def _get_collection():
    """单例 chroma PersistentClient + collection（同进程多实例会死锁）。"""
    global _chroma_client, _chroma_collection
    with _chroma_lock:
        if _chroma_collection is None:
            import chromadb

            _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            _chroma_collection = _chroma_client.get_or_create_collection(
                "shelf", metadata={"hnsw:space": "cosine"})
        return _chroma_collection


def _hf_model_cached(model_id: str) -> bool:
    from pathlib import Path

    d = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    return d.exists() and any(d.iterdir())


def _set_hf_offline(offline: bool) -> None:
    """同时改 env 与已加载模块的运行时常量（huggingface_hub 在 import 时固化 env）。"""
    import os

    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HUB_OFFLINE = offline
    except Exception:
        pass
    try:
        from transformers.utils import hub as tf_hub

        tf_hub._is_offline_mode = offline
    except Exception:
        pass


def _get_embed_cfg() -> dict:
    return settings_store.get("embedding")


def _make_local_embedding(model_name: str):
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    try:
        # 已缓存 → 强制离线加载（在线 etag 检查遇 HF 限流会指数退避数分钟）
        offline = _hf_model_cached(model_name)
        _set_hf_offline(offline)
        return HuggingFaceEmbedding(model_name=model_name, device="cpu")
    finally:
        _set_hf_offline(False)


def _make_api_embedding(cfg: dict):
    """OpenAI 兼容 embedding API（litellm 统一各家：openai/通义/智谱/…）。"""
    import litellm as _litellm
    from llama_index.core.embeddings import BaseEmbedding

    base_url = (cfg.get("api_base_url") or "").strip() or None
    api_key = (cfg.get("api_key") or "").strip() or "sk-empty"
    api_model = (cfg.get("api_model") or "text-embedding-3-small").strip()

    def _embed(texts: list[str]) -> list[list[float]]:
        resp = _litellm.embedding(model=api_model, input=texts, api_key=api_key,
                                  base_url=base_url, timeout=30)
        return [item["embedding"] for item in resp.data]

    class ApiEmbedding(BaseEmbedding):
        """LlamaIndex BaseEmbedding 适配：远程 /embeddings 调用。"""

        def __init__(self):
            super().__init__(embed_batch_size=32, model_name=api_model)

        def _get_query_embedding(self, query: str):
            return _embed([query])[0]

        async def _aget_query_embedding(self, query: str):
            return _embed([query])[0]

        def _get_text_embedding(self, text: str):
            return _embed([text])[0]

        async def _aget_text_embedding(self, text: str):
            return _embed([text])[0]

        def _get_text_embeddings(self, texts: list[str]):
            return _embed(texts)

        async def _aget_text_embeddings(self, texts: list[str]):
            return _embed(texts)

    return ApiEmbedding()


def _make_dashscope_embedding(cfg: dict):
    """DashScope 原生 embedding 协议（阿里云百炼 / 千问 MaaS）。

    与 OpenAI 兼容接口不同，这条路径是 /services/embeddings/text-embedding/text-embedding，
    请求体 {"model", "input": {"texts": [...]}}，返回 output.embeddings[].embedding。
    """
    import httpx
    from llama_index.core.embeddings import BaseEmbedding

    base = (cfg.get("api_base_url") or "https://dashscope.aliyuncs.com/api/v1").rstrip("/")
    url = f"{base}/services/embeddings/text-embedding/text-embedding"
    api_key = (cfg.get("api_key") or "").strip()
    api_model = (cfg.get("api_model") or "text-embedding-v3").strip()
    batch_size = 16

    def _embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                resp = httpx.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": api_model, "input": {"texts": batch}},
                    timeout=60,
                )
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"DashScope embedding 请求失败：{e}") from e
            if resp.status_code != 200:
                raise RuntimeError(
                    f"DashScope embedding HTTP {resp.status_code}：{resp.text[:200]}")
            items = ((resp.json().get("output") or {}).get("embeddings") or [])
            if len(items) != len(batch):
                raise RuntimeError(
                    f"DashScope embedding 返回条数不符：{len(items)} != {len(batch)}")
            out.extend(item["embedding"] for item in items)
        return out

    class DashScopeEmbedding(BaseEmbedding):
        def __init__(self):
            super().__init__(embed_batch_size=batch_size, model_name=api_model)

        def _get_query_embedding(self, query: str):
            return _embed([query])[0]

        async def _aget_query_embedding(self, query: str):
            import asyncio

            return await asyncio.to_thread(_embed, [query])[0]

        def _get_text_embedding(self, text: str):
            return _embed([text])[0]

        async def _aget_text_embedding(self, text: str):
            import asyncio

            return await asyncio.to_thread(_embed, [text])[0]

        def _get_text_embeddings(self, texts: list[str]):
            return _embed(texts)

        async def _aget_text_embeddings(self, texts: list[str]):
            import asyncio

            return await asyncio.to_thread(_embed, texts)

    return DashScopeEmbedding()


def local_embedding_available() -> bool:
    """打包版剔除了 torch/transformers，本地 embedding 仅开发环境可用。"""
    import sys

    if not getattr(sys, "frozen", False):
        return True
    try:
        import importlib

        importlib.import_module("sentence_transformers")
        return True
    except ImportError:
        return False


def _get_embed_model():
    """双模式 embedding：local（sentence-transformers，仅开发环境）/ api（OpenAI 兼容）。"""
    global _embed_model
    with _embed_lock:
        if _embed_model is None:
            cfg = _get_embed_cfg()
            if (cfg.get("mode") or "local") == "local" and not local_embedding_available():
                raise RuntimeError(
                    "打包版不含本地向量化模型（用于精简体积），请在「设置 → 向量索引」中切换为 "
                    "API 服务并填入支持 embeddings 的服务商（OpenAI / 通义 / 智谱等）"
                )
            if (cfg.get("mode") or "local") == "api":
                if (cfg.get("api_provider") or "openai") == "dashscope":
                    _embed_model = _make_dashscope_embedding(cfg)
                else:
                    _embed_model = _make_api_embedding(cfg)
            else:
                _embed_model = _make_local_embedding(cfg.get("model") or "BAAI/bge-small-zh-v1.5")
        return _embed_model


def _get_index():
    """懒加载 Chroma 集合 + LlamaIndex VectorStoreIndex。"""
    global _index
    with _embed_lock:
        if _index is None:
            from llama_index.core import VectorStoreIndex
            from llama_index.core.storage.storage_context import StorageContext
            from llama_index.vector_stores.chroma import ChromaVectorStore

            collection = _get_collection()
            storage_context = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=collection))
            _index = VectorStoreIndex.from_vector_store(
                vector_store=ChromaVectorStore(chroma_collection=collection),
                embed_model=_get_embed_model(),
                storage_context=storage_context,
            )
        return _index


def reset_client() -> None:
    """更换 embedding 模型后重置内存态（触发全量重建）。"""
    global _index, _embed_model, _chroma_client, _chroma_collection
    with _embed_lock, _chroma_lock:
        _index = None
        _embed_model = None
        _chroma_client = None
        _chroma_collection = None


# ---------------------------------------------------------------- 索引构建

CHUNK_SIZE = 480
CHUNK_OVERLAP = 60
CE_RERANK_LIMIT = 12  # cross-encoder 精排的候选上限（按 RRF 顺序取前 N）

_PAGE_MARK = re.compile(r"【第\s*(\d+)\s*页】")


def _page_of(text: str, which: str = "start") -> Optional[int]:
    """从分块里的「【第 N 页】」标记取起止页；没有标记返回 None。"""
    nums = [int(x) for x in _PAGE_MARK.findall(text or "")]
    if not nums:
        return None
    return nums[0] if which == "start" else nums[-1]


def _pages_for_chunks(pieces: list[str]) -> list[tuple[Optional[int], Optional[int]]]:
    """按顺序补全页码：块里没有标记时沿用上一块的页码。"""
    out: list[tuple[Optional[int], Optional[int]]] = []
    current: Optional[int] = None
    for piece in pieces:
        nums = [int(x) for x in _PAGE_MARK.findall(piece or "")]
        if nums:
            current = nums[-1]
        out.append((nums[0] if nums else current, nums[-1] if nums else current))
    return out


def split_chunks(text: str) -> list[str]:
    """统一的切块口径，向量索引与 BM25 都走这里，保证两路能对齐到同一个 chunk。"""
    from llama_index.core import Document as LlamaDocument
    from llama_index.core.node_parser import SentenceSplitter

    text = (text or "").strip()
    if not text:
        return []
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents([LlamaDocument(text=text)], show_progress=False)
    return [n.get_content().strip() for n in nodes if n.get_content().strip()]


def index_document(doc_id: int) -> dict:
    """把单个文档的解析文本切块并向量化入库。"""
    with Session(engine) as s:
        doc = s.get(Document, doc_id)
        if not doc:
            raise KeyError("文档不存在")
        if not doc.text_content.strip():
            doc.indexed = False
            doc.index_error = "无可索引文本（可能为扫描件且未启用 OCR）"
            s.add(doc)
            s.commit()
            return {"indexed": False, "error": doc.index_error}

        with _state_lock:
            _state["current"] = doc.title

        # 旧向量马上要删，先把状态改成"未索引"：中途失败时不会出现
        # indexed=True 但集合里已经没有这块数据的情况
        doc.indexed = False
        doc.chunk_count = 0
        doc.index_error = None
        s.add(doc)
        s.commit()
        try:
            vector_store = _chroma_store()
            try:
                vector_store.delete(str(doc.id))  # 旧块先删，保证幂等
            except Exception as e:  # noqa: BLE001
                log.debug("delete old vectors failed doc=%s: %s", doc.id, e)

            # 与 BM25 共用 split_chunks()：带 metadata 切块会改变边界，
            # 两路的 chunk_id 就对不上，RRF 的"同一块两路命中"也就失效了。
            from llama_index.core.schema import TextNode

            pieces = split_chunks(doc.text_content)
            pages = _pages_for_chunks(pieces)
            nodes = [
                TextNode(
                    text=piece,
                    id_=f"{doc.id}:{i}",
                    metadata={"doc_id": str(doc.id), "title": doc.title,
                              "chunk_id": f"{doc.id}:{i}",
                              "page_start": pages[i][0],
                              "page_end": pages[i][1]},
                    # 元数据只用于回溯出处，不参与向量化，避免标题泄漏进检索
                    excluded_embed_metadata_keys=["doc_id", "title", "chunk_id", "page_start", "page_end"],
                    excluded_llm_metadata_keys=["doc_id", "title", "chunk_id", "page_start", "page_end"],
                )
                for i, piece in enumerate(pieces)
            ]
            _get_index().insert_nodes(nodes, show_progress=False)
        except Exception as e:
            doc.indexed = False
            doc.chunk_count = 0
            doc.index_error = f"向量化失败：{e}"[:300]
            s.add(doc)
            s.commit()
            raise

        doc.indexed = True
        doc.indexed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        doc.chunk_count = len(nodes)
        doc.index_error = None
        s.add(doc)
        s.commit()
        return {"indexed": True, "chunks": len(nodes)}


def remove_document(doc_id: int) -> None:
    try:
        _chroma_store().delete(str(doc_id))
    except Exception as e:  # noqa: BLE001
        log.warning("删除向量失败 doc=%s: %s", doc_id, e)


def _chroma_store():
    from llama_index.vector_stores.chroma import ChromaVectorStore

    return ChromaVectorStore(chroma_collection=_get_collection())


def reindex_all() -> dict:
    """清空集合后全量重建（后台线程执行，状态走 status()）。"""
    with _state_lock:
        if _state["running"]:
            return {"started": False, "reason": "已有重建任务在进行中"}
        _state.update({"running": True, "done": 0, "error": None, "current": ""})

    with Session(engine) as s:
        doc_ids = [d.id for d in s.exec(select(Document)).all() if d.text_content.strip()]

    with _state_lock:
        _state["total"] = len(doc_ids)

    def worker():
        try:
            try:
                _get_collection()._client.delete_collection("shelf")
            except Exception as e:  # noqa: BLE001
                log.debug("清空集合失败（首次重建时正常）：%s", e)
            reset_client()
            _get_index()
            for i, did in enumerate(doc_ids):
                with _state_lock:
                    _state["done"] = i
                try:
                    index_document(did)
                except Exception:
                    with Session(engine) as s:
                        doc = s.get(Document, did)
                        if doc:
                            doc.index_error = traceback.format_exc(limit=2)[-500:]
                            s.add(doc)
                            s.commit()
            with _state_lock:
                _state.update({"running": False, "current": "", "done": len(doc_ids),
                               "last_finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        except Exception as e:  # noqa: BLE001
            with _state_lock:
                _state.update({"running": False, "error": str(e), "current": ""})

    threading.Thread(target=worker, name="xueyu-reindex", daemon=True).start()
    return {"started": True, "total": len(doc_ids)}


# ---------------------------------------------------------------- 混合检索（BM25 + 向量 + RRF）

_bm25_cache: "OrderedDict[tuple, tuple[Any, list[dict]]]" = OrderedDict()
_BM25_CACHE_MAX = 4


def _norm_text(text: str) -> str:
    """去空白、标点、大小写，用于判断两条改写是否重复。"""
    import re as _re

    return _re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(text).lower())


def _tokenize_zh(text: str) -> list[str]:
    """中文按字 + 英文按词的轻量分词（BM25 用，无外部依赖）。"""
    import re as _re

    tokens = []
    for word in _re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text):
        tokens.append(word.lower())
    return tokens


def _bm25_for_docs(doc_ids: Optional[list[int]] = None) -> tuple[Any, list[dict]]:
    """从 DB 构建内存 BM25 索引（全量或限定文档），小型 LRU 缓存。

    签名带每篇文档的 id/长度/索引时间：内容被替换但总长度没变时也会重建。
    不同文档集合各占一个槽位，来回切换不会反复重建。
    """
    with Session(engine) as s:
        if doc_ids:
            docs = s.exec(select(Document).where(Document.id.in_(doc_ids))).all()
        else:
            docs = s.exec(select(Document)).all()
        sig = (tuple(sorted(doc_ids)) if doc_ids else "all", len(docs),
               tuple(sorted((d.id, len(d.text_content or ""), d.indexed_at or "") for d in docs)))
        with _state_lock:
            hit = _bm25_cache.get(sig)
            if hit is not None:
                _bm25_cache.move_to_end(sig)
                return hit

        chunks = []
        for d in docs:
            if not d.text_content:
                continue
            # 与向量索引共用 split_chunks，融合键才能对上
            for i, piece in enumerate(split_chunks(d.text_content)):
                chunks.append({"chunk_id": f"{d.id}:{i}", "doc_id": d.id,
                               "title": d.title, "text": piece})
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([_tokenize_zh(c["text"][:3000]) for c in chunks]) if chunks else None
    with _state_lock:
        _bm25_cache[sig] = (bm25, chunks)
        _bm25_cache.move_to_end(sig)
        while len(_bm25_cache) > _BM25_CACHE_MAX:
            _bm25_cache.popitem(last=False)
    return bm25, chunks


def _rrf_fuse(vector_hits: list[dict], bm25_hits: list[dict], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion：融合两路检索排名（标准常数 k=60）。"""
    scores: dict[str, dict] = {}
    for rank, h in enumerate(vector_hits):
        key = h.get("chunk_id") or f"{h['doc_id']}:{h['text'][:80]}"
        scores.setdefault(key, {**h, "rrf": 0.0, "sources": []})
        scores[key]["rrf"] += 1 / (k + rank + 1)
        scores[key]["sources"].append("vec")
    for rank, h in enumerate(bm25_hits):
        key = h.get("chunk_id") or f"{h['doc_id']}:{h['text'][:80]}"
        entry = scores.get(key)
        if entry is None:
            entry = {**h, "score": None, "rrf": 0.0, "sources": []}
            scores[key] = entry
        entry["rrf"] += 1 / (k + rank + 1)
        entry["sources"].append("bm25")
    fused = sorted(scores.values(), key=lambda x: -x["rrf"])
    for h in fused:
        h["hybrid"] = True
        h["retrieved_by"] = "+".join(sorted(set(h.pop("sources", []))))
    return fused


def _rrf_fuse_weighted(ranked_lists: list[list[dict]], weights: list[float],
                       k: int = 60) -> list[dict]:
    """跨查询的加权 RRF：每个查询先各自融合两路，再按查询权重累加排名分。

    原查询权重 1.0、改写查询权重 0.7，避免改写跑偏时把原问题挤下去。
    一个块被多个查询同时召回时分数自然叠加，排名贡献不会像按 chunk_id 去重那样丢掉。
    """
    scores: dict[str, dict] = {}
    for ranked, w in zip(ranked_lists, weights):
        for rank, h in enumerate(ranked):
            key = h.get("chunk_id") or f"{h['doc_id']}:{h['text'][:80]}"
            entry = scores.get(key)
            if entry is None:
                entry = {**h, "rrf": 0.0, "hits": 0}
                scores[key] = entry
            entry["rrf"] += w / (k + rank + 1)
            entry["hits"] += 1
    fused = sorted(scores.values(), key=lambda x: -x["rrf"])
    for h in fused:
        hits = h.pop("hits", 1)
        src = h.get("retrieved_by") or "hybrid"
        h["retrieved_by"] = f"{src}+multi" if hits > 1 else src
    return fused


async def _recall_and_fuse(queries: list[str], doc_ids: Optional[list[int]],
                           recall_k: int, weights: Optional[list[float]] = None
                           ) -> tuple[list[dict], dict]:
    """多查询召回：每个查询内部先做向量 + BM25 的 RRF，再跨查询加权融合。"""
    import asyncio as _asyncio

    vec_lists = await _asyncio.gather(
        *[_asyncio.to_thread(query_shelf, q, doc_ids=doc_ids, k=recall_k) for q in queries],
        return_exceptions=True)

    bm25, corpus = _bm25_for_docs(doc_ids)
    per_query: list[list[dict]] = []
    vec_keys: set[str] = set()
    bm_keys: set[str] = set()
    vector_down = None

    for q, vec_or_err in zip(queries, vec_lists):
        if isinstance(vec_or_err, RuntimeError):      # embedding 不可用 → 该路置空
            vector_down = str(vec_or_err)[:120]
            vec = []
        elif isinstance(vec_or_err, BaseException):
            vec = []
        else:
            vec = vec_or_err
        for h in vec:
            vec_keys.add(h.get("chunk_id") or f"{h['doc_id']}:{h['text'][:80]}")

        bm: list[dict] = []
        if bm25 and corpus:
            scores = bm25.get_scores(_tokenize_zh(q))
            ranked = sorted(zip(corpus, scores), key=lambda x: -x[1])[:recall_k]
            bm = [{**c, "score": round(float(sc), 3)} for c, sc in ranked if sc > 0]
            for c in bm:
                bm_keys.add(c.get("chunk_id") or f"{c['doc_id']}:{c['text'][:80]}")

        per_query.append(_rrf_fuse(vec, bm))

    fused = _rrf_fuse_weighted(per_query, weights or [1.0] * len(queries))
    stats = {"vec_hits": len(vec_keys), "bm25_hits": len(bm_keys)}
    if vector_down:
        stats["vector_down"] = vector_down
    return fused, stats


def _rerank_limit() -> int:
    """精排候选数：设置里可改，默认 12。"""
    try:
        n = int(settings_store.get("rerank").get("api_top_n") or CE_RERANK_LIMIT)
        return min(50, max(1, n))
    except Exception:  # noqa: BLE001
        return CE_RERANK_LIMIT


async def _cross_encoder_rerank(query: str, chunks: list[dict]) -> Optional[list[dict]]:
    """本地 cross-encoder 或远程 rerank API；都不可用返回 None。

    只精排前 N 个候选：实测再往后放候选不再提升 top-3 命中率，
    但耗时随候选数线性增长（12 个约 2-3s，40 个约 10s）。
    """
    import asyncio as _asyncio

    from app.services import rerank as rerank_mod

    return await _asyncio.to_thread(rerank_mod.rerank, query, chunks, limit=_rerank_limit())


def _rerank_model_name() -> str:
    from app.services import rerank as rerank_mod

    return rerank_mod.model_name()


def _rerank_using_api() -> bool:
    from app.services import rerank as rerank_mod

    return rerank_mod.using_api()


def _rerank_fallback_count() -> int:
    from app.services import rerank as rerank_mod

    return rerank_mod.fallback_count()


def _rerank_threshold() -> Optional[float]:
    from app.services import rerank as rerank_mod

    return rerank_mod.threshold()


def _annotate_confidence(meta: dict, ranked: list[dict]) -> None:
    """记录最高分，并按当前模型阈值标记低置信度（不在这里过滤，调用方决定）。"""
    top = ranked[0].get("rerank_score") if ranked else None
    if top is not None:
        meta["top_score"] = float(top)
    threshold = _rerank_threshold()
    if threshold is not None:
        meta["rerank_threshold"] = threshold
        if top is not None and float(top) < threshold:
            meta["low_confidence"] = True


async def _llm_listwise_rerank(query: str, chunks: list[dict]) -> list[dict]:
    """让 LLM 按相关性输出候选编号顺序（listwise）；调用失败抛异常，由调用方记录。"""
    from app.services import llm as llm_mod

    cand = chunks[: min(12, len(chunks))]
    listing = "\n".join(f"[{i + 1}] {c['title']}：{c['text'][:150]}" for i, c in enumerate(cand))
    data = await llm_mod.ask_json(
        "你是检索重排器。按与查询的相关性从高到低排列候选片段编号 JSON："
        '{\"order\": [1, 3, 2, …]}（只含给出的编号，不要新增）；只输出 JSON。',
        f"查询：{query}\n\n候选：\n{listing}", temperature=0.0, fn="research")
    order = [int(x) - 1 for x in data.get("order", []) if 1 <= int(x) <= len(cand)]
    seen, reranked = set(), []
    for oi in order:
        if oi not in seen:
            reranked.append(cand[oi])
            seen.add(oi)
    for c in cand:
        if id(c) not in {id(x) for x in reranked}:
            reranked.append(c)
    return reranked + chunks[len(cand):]


async def query_shelf_hybrid(query: str, *, doc_ids: Optional[list[int]] = None, k: int = 5,
                             multi_query: bool = False, rerank: bool = False) -> tuple[list[dict], dict]:
    """混合检索主入口：向量 + BM25 → RRF 融合 →（可选）multi-query 扩展 →（可选）重排。

    返回 (chunks, meta)：meta 记录检索过程信息（向量/BM25 命中数、改写与重排状态），供日志与调试。
    """
    meta = {"mode": "hybrid", "queries": [query], "vec_hits": 0, "bm25_hits": 0, "reranked": False}
    queries = [query]
    if multi_query:
        # Multi-Query Retrieval：面向"中文口语问英文论文"做跨语言改写
        try:
            from app.services import llm as llm_mod

            data = await llm_mod.ask_json(
                "你是检索查询改写器，检索目标是英文论文库，用户可能用中文口语提问。"
                "把问题改写成 3 条检索式，输出 JSON：{\"variants\": [\"…\"]}。"
                "要求：① 一条中文同义改写；② 一条给出对应的英文术语/缩写；"
                "③ 一条贴近论文表达的英文检索式。保留原问题的关键实体，不要重复；只输出 JSON。",
                f"原查询：{query}", temperature=0.2, fn="research")
            seen = {_norm_text(query)}
            variants = []
            for v in data.get("variants", []):
                v = str(v).strip()
                if v and _norm_text(v) not in seen:
                    seen.add(_norm_text(v))
                    variants.append(v)
            queries = [query] + variants[:3]
            meta["queries"] = queries
        except Exception as e:  # noqa: BLE001
            meta["rewrite_failed"] = str(e)[:120]

    # 启用重排时扩大召回池：先粗召回再精排，重排才有意义
    recall_k = max(k * 4, 12) if rerank else k
    fused, stats = await _recall_and_fuse(
        queries, doc_ids, recall_k, [1.0] + [0.7] * max(0, len(queries) - 1))
    meta.update(stats)

    if rerank and fused:
        # 首选 cross-encoder：本地模型直接打相关性分，不生成文字，速度快、不花钱
        # XUEYU_RERANK_ENGINE=llm 可切回 LLM listwise，用于对比实验
        import os as _os

        engine = (_os.environ.get("XUEYU_RERANK_ENGINE") or "cross_encoder").lower()
        if engine != "llm":
            before_fallback = _rerank_fallback_count()
            ranked = await _cross_encoder_rerank(query, fused)
            if ranked is not None:
                if _rerank_using_api() and _rerank_fallback_count() > before_fallback:
                    meta["rerank_fallback"] = True   # 远程失败退回了本地，评测视为不可用
                meta["reranked"] = "rerank_api" if _rerank_using_api() else "cross_encoder"
                meta["rerank_model"] = _rerank_model_name()
                _annotate_confidence(meta, ranked)
                return ranked[:k], meta
        # cross-encoder 不可用，或显式指定 llm 时走 LLM listwise 重排
        try:
            fused = await _llm_listwise_rerank(query, fused)
            meta["reranked"] = "llm"
            meta["rerank_model"] = "llm"
        except Exception as e:  # noqa: BLE001
            meta["rerank_failed"] = str(e)[:120]
    return fused[:k], meta


# ---------------------------------------------------------------- 检索

def query_shelf(query: str, *, doc_ids: Optional[list[int]] = None, k: int = 5) -> list[dict]:
    """向量检索。doc_ids 为空 = 整个书架；否则限定在这些文档内。"""
    if not query.strip():
        return []
    index = _get_index()
    try:
        if doc_ids:
            from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

            filters = MetadataFilters(filters=[ExactMatchFilter(key="doc_id", value=str(d)) for d in doc_ids])
            retriever = index.as_retriever(similarity_top_k=k, filters=filters)
            nodes = retriever.retrieve(query)
        else:
            nodes = index.as_retriever(similarity_top_k=k).retrieve(query)
    except Exception:
        # 过滤器不可用等异常时退化为检索后过滤
        nodes = index.as_retriever(similarity_top_k=max(k * 3, 12)).retrieve(query)
        if doc_ids:
            want = {str(d) for d in doc_ids}
            nodes = [n for n in nodes if str(n.metadata.get("doc_id")) in want][:k]

    out = []
    for n in nodes:
        meta = n.metadata or {}
        score = getattr(n, "score", None)
        out.append({
            "chunk_id": meta.get("chunk_id"),
            "doc_id": meta.get("doc_id"),
            "title": meta.get("title", ""),
            "page_start": meta.get("page_start"),
            "page_end": meta.get("page_end"),
            "text": n.get_content().strip(),
            "score": round(score, 4) if score is not None else None,
        })
    return out


def collection_count() -> int:
    try:
        return _get_collection().count()
    except Exception:
        return 0
