"""本地向量化（RAG）：LlamaIndex + sentence-transformers + Chroma 持久化。

- 首次使用会下载 embedding 模型（约 130MB，缓存于 ~/.cache/huggingface）
- 单工作线程串行建索引；上传即入队，重建按钮清空后全部重嵌
- 更换 embedding 模型会自动触发全量重建（维度不兼容）
"""
from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Optional

from sqlmodel import Session, select

from app.config import CHROMA_DIR
from app.storage.db import engine
from app.storage.models import Document
from app.services import settings_store

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
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.embeddings import MockEmbedding

    base_url = (cfg.get("api_base_url") or "").strip() or None
    api_key = (cfg.get("api_key") or "").strip() or "sk-empty"
    api_model = (cfg.get("api_model") or "text-embedding-3-small").strip()

    class ApiEmbedding(MockEmbedding):
        """LlamaIndex BaseEmbedding 适配：远程 /embeddings 调用。"""

        def __init__(self):
            super().__init__(embed_batch_size=32, model_name=api_model)
            self.callback_manager = CallbackManager([])

        def _get_query_embedding(self, query: str):
            return self._embed([query])[0]

        async def _aget_query_embedding(self, query: str):
            return self._embed([query])[0]

        def _get_text_embeddings(self, texts: list[str]):
            return self._embed(texts)

        def _embed(self, texts: list[str]) -> list[list[float]]:
            resp = _litellm.embedding(model=api_model, input=texts, api_key=api_key,
                                      base_url=base_url, timeout=30)
            return [item["embedding"] for item in resp.data]

    return ApiEmbedding()


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

def index_document(doc_id: int) -> dict:
    """把单个文档的解析文本切块并向量化入库。"""
    from llama_index.core import Document as LlamaDocument
    from llama_index.core.node_parser import SentenceSplitter

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

        vector_store = _chroma_store()
        try:
            vector_store.delete(str(doc.id))  # 旧块先删，保证幂等
        except Exception:
            pass

        splitter = SentenceSplitter(chunk_size=480, chunk_overlap=60)
        nodes = splitter.get_nodes_from_documents(
            [LlamaDocument(id_=str(doc.id), text=doc.text_content,
                           metadata={"doc_id": str(doc.id), "title": doc.title})],
            show_progress=False,
        )
        _get_index().insert_nodes(nodes, show_progress=False)

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
    except Exception:
        pass


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
            except Exception:
                pass
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

_bm25_cache: dict[str, Any] = {"corpus_sig": None, "bm25": None, "chunks": []}


def _tokenize_zh(text: str) -> list[str]:
    """中文按字 + 英文按词的轻量分词（BM25 用，无外部依赖）。"""
    import re as _re

    tokens = []
    for word in _re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text):
        tokens.append(word.lower())
    return tokens


def _bm25_for_docs(doc_ids: Optional[list[int]] = None) -> tuple[Any, list[dict]]:
    """从 DB 构建内存 BM25 索引（全量或限定文档），带签名缓存。"""
    with Session(engine) as s:
        if doc_ids:
            docs = s.exec(select(Document).where(Document.id.in_(doc_ids))).all()
        else:
            docs = s.exec(select(Document)).all()
        chunks = []
        for d in docs:
            if not d.text_content:
                continue
            size = 1200
            text = d.text_content
            for i in range(0, len(text), size):
                piece = text[i:i + size]
                if piece.strip():
                    chunks.append({"doc_id": d.id, "title": d.title, "text": piece})
    sig = (tuple(sorted(doc_ids)) if doc_ids else "all", sum(len(c["text"]) for c in chunks))
    with _state_lock:
        if _bm25_cache["corpus_sig"] == sig:
            return _bm25_cache["bm25"], _bm25_cache["chunks"]
        from rank_bm25 import BM25Okapi

        bm25 = BM25Okapi([_tokenize_zh(c["text"][:3000]) for c in chunks]) if chunks else None
        _bm25_cache.update({"corpus_sig": sig, "bm25": bm25, "chunks": chunks})
    return bm25, chunks


def _rrf_fuse(vector_hits: list[dict], bm25_hits: list[dict], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion：融合两路检索排名（标准常数 k=60）。"""
    scores: dict[str, dict] = {}
    for rank, h in enumerate(vector_hits):
        key = f"{h['doc_id']}:{h['text'][:80]}"
        scores.setdefault(key, {**h, "rrf": 0.0, "sources": []})
        scores[key]["rrf"] += 1 / (k + rank + 1)
        scores[key]["sources"].append("vec")
    for rank, h in enumerate(bm25_hits):
        key = f"{h['doc_id']}:{h['text'][:80]}"
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


async def query_shelf_hybrid(query: str, *, doc_ids: Optional[list[int]] = None, k: int = 5,
                             multi_query: bool = False, rerank: bool = False) -> tuple[list[dict], dict]:
    """混合检索主入口：向量 + BM25 → RRF 融合 →（可选）multi-query 扩展 →（可选）LLM 重排。

    返回 (chunks, meta)：meta 记录检索过程信息（向量/BM25 命中数、改写与重排状态），供日志与调试。
    """
    meta = {"mode": "hybrid", "queries": [query], "vec_hits": 0, "bm25_hits": 0, "reranked": False}
    queries = [query]
    if multi_query:
        # Multi-Query Retrieval：LLM 改写出多视角检索式，并行检索后融合
        try:
            from app.services import llm as llm_mod

            data = await llm_mod.ask_json(
                "你是检索查询改写器。把查询改写为 2-3 个不同视角的检索式 JSON："
                '{\"variants\": [\"…\"]}（含原查询的关键概念、同义词、上下位词）；只输出 JSON。',
                f"原查询：{query}", temperature=0.2, fn="research")
            variants = [str(v).strip() for v in data.get("variants", []) if str(v).strip()][:3]
            queries = [query] + variants
            meta["queries"] = queries
        except Exception:
            pass

    vector_hits: dict[str, dict] = {}
    bm25_hits: dict[str, dict] = {}
    vector_down = None
    for q in queries:
        try:
            vec = query_shelf(q, doc_ids=doc_ids, k=k)
        except RuntimeError as e:  # embedding 不可用（打包版未配 API）→ 降级 BM25 关键词检索
            vector_down = str(e)[:120]
            vec = []
        for h in vec:
            key = f"{h['doc_id']}:{h['text'][:80]}"
            if key not in vector_hits or (h.get("score") or 0) > (vector_hits[key].get("score") or 0):
                vector_hits[key] = h
        try:
            bm25, corpus = _bm25_for_docs(doc_ids)
            if bm25 and corpus:
                scores = bm25.get_scores(_tokenize_zh(q))
                ranked = sorted(zip(corpus, scores), key=lambda x: -x[1])[:k]
                for c, sc in ranked:
                    if sc > 0:
                        key = f"{c['doc_id']}:{c['text'][:80]}"
                        if key not in bm25_hits:
                            bm25_hits[key] = {**c, "score": round(float(sc), 3)}
        except Exception:
            pass
    meta["vec_hits"] = len(vector_hits)
    meta["bm25_hits"] = len(bm25_hits)
    if vector_down:
        meta["vector_down"] = vector_down
    fused = _rrf_fuse(list(vector_hits.values()), list(bm25_hits.values()))
    if not fused and bm25_hits:
        fused = [{**h, "hybrid": True, "retrieved_by": "bm25"} for h in bm25_hits.values()]

    if rerank and fused:
        # LLM listwise 重排：对 top 候选按与查询的相关性重新排序
        try:
            from app.services import llm as llm_mod

            cand = fused[: min(12, len(fused))]
            listing = "\n".join(f"[{i + 1}] {c['title']}：{c['text'][:150]}" for i, c in enumerate(cand))
            data = await llm_mod.ask_json(
                "你是检索重排器。按与查询的相关性从高到低排列候选片段编号 JSON："
                '{\"order\": [1, 3, 2, …]}（只含给出的编号，不要新增）；只输出 JSON。',
                f"查询：{query}\n\n候选：\n{listing}", temperature=0.0, fn="research")
            order = [int(x) - 1 for x in data.get("order", []) if 1 <= int(x) <= len(cand)]
            seen, reranked = set(), []
            for oi in order:
                if oi not in seen:
                    reranked.append(cand[oi]); seen.add(oi)
            for c in cand:
                if id(c) not in {id(x) for x in reranked}:
                    reranked.append(c)
            fused = reranked + fused[len(cand):]
            meta["reranked"] = True
        except Exception:
            pass
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
            "doc_id": meta.get("doc_id"),
            "title": meta.get("title", ""),
            "text": n.get_content().strip(),
            "score": round(score, 4) if score is not None else None,
        })
    return out


def collection_count() -> int:
    try:
        return _get_collection().count()
    except Exception:
        return 0
