"""针对审查修复的回归测试：缓存作用域 / OCR 标记 / 页码 / 上传限额。"""
from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.parser import documents as parser  # noqa: E402
from app.services import llm  # noqa: E402
from app.services.rag_service import _page_of  # noqa: E402


def test_semantic_cache_isolated_by_scope(monkeypatch):
    llm._semcache.clear()
    monkeypatch.setattr(llm, "_cache_enabled", lambda: True)
    monkeypatch.setattr(llm, "_query_embedding", lambda text: [1.0, 0.0])
    monkeypatch.setattr(llm, "_cache_scope", lambda fn, scope: (scope, "m1"))

    llm.semantic_cache_put("问题", "答案 A", scope="doc:1:v1", fn="doc")
    assert llm.semantic_cache_get("问题", scope="doc:1:v1", fn="doc") == "答案 A"
    assert llm.semantic_cache_get("问题", scope="doc:1:v2", fn="doc") is None
    assert llm.semantic_cache_get("问题", scope="doc:2:v1", fn="doc") is None


def test_semantic_cache_isolated_by_model(monkeypatch):
    llm._semcache.clear()
    monkeypatch.setattr(llm, "_cache_enabled", lambda: True)
    monkeypatch.setattr(llm, "_query_embedding", lambda text: [1.0, 0.0])
    monkeypatch.setattr(llm, "_cache_scope", lambda fn, scope: (scope, "model-a"))
    llm.semantic_cache_put("问题", "答案 A", scope="doc:1", fn="doc")

    monkeypatch.setattr(llm, "_cache_scope", lambda fn, scope: (scope, "model-b"))
    assert llm.semantic_cache_get("问题", scope="doc:1", fn="doc") is None


def test_page_of_reads_markers():
    text = "【第 2 页】\n开头\n\n【第 3 页】\n结尾"
    assert _page_of(text, "start") == 2
    assert _page_of(text, "end") == 3
    assert _page_of("没有页码标记") is None


def test_parse_any_keeps_ocr_flag(monkeypatch):
    monkeypatch.setattr(parser, "get_ocr", lambda: (lambda _b: "ocr text"))
    monkeypatch.setattr(parser, "parse_pdf", lambda path, ocr=None: ("文本", 2, True))
    out = parser.parse_any(pathlib.Path("demo.pdf"))
    assert out["ocr_used"] is True
    assert out["pages"] == 2


def test_read_limited_rejects_oversize():
    from app.api.shelf import _read_limited

    class FakeUpload:
        def __init__(self, chunks):
            self._chunks = list(chunks)

        async def read(self, n):
            return self._chunks.pop(0) if self._chunks else b""

    ok = asyncio.run(_read_limited(FakeUpload([b"a" * 10, b"b" * 10]), 25))
    assert len(ok) == 20

    try:
        asyncio.run(_read_limited(FakeUpload([b"a" * 10, b"b" * 10]), 15))
    except ValueError as e:
        assert "限制" in str(e)
    else:
        raise AssertionError("超限时应当报错")