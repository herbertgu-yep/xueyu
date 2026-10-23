"""检索链路的回归测试：分块口径、RRF 融合、信源去重、引用校验。

只测不依赖 LLM 的纯逻辑，因此不需要配置模型即可运行。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.rag_service import _rrf_fuse, split_chunks  # noqa: E402


def test_split_chunks_is_deterministic():
    text = "句子。" * 400
    first = split_chunks(text)
    second = split_chunks(text)
    assert first == second, "同一文本多次切块结果必须一致"
    assert len(first) > 1, "长文本应切成多块"


def test_split_chunks_respects_size():
    text = "这是一句用于测试切块的中文句子。" * 200
    chunks = split_chunks(text)
    assert all(len(c) <= 480 + 60 for c in chunks), "单块不应超过 chunk_size + overlap"


def test_rrf_prefers_chunks_hit_by_both_retrievers():
    vec = [
        {"chunk_id": "1:0", "doc_id": 1, "text": "a"},
        {"chunk_id": "1:1", "doc_id": 1, "text": "b"},
    ]
    bm25 = [
        {"chunk_id": "1:1", "doc_id": 1, "text": "b"},
        {"chunk_id": "2:0", "doc_id": 2, "text": "c"},
    ]
    fused = _rrf_fuse(vec, bm25)
    assert fused[0]["chunk_id"] == "1:1", "两路都命中的块应排第一"
    assert fused[0]["retrieved_by"] == "bm25+vec", "同一块两路都命中时来源应合并"


def test_rrf_keys_by_chunk_id_not_text_prefix():
    """两路口径一致时，同一块即使长文本前缀不同也要合并。"""
    vec = [{"chunk_id": "9:3", "doc_id": 9, "text": "甲" * 200}]
    bm25 = [{"chunk_id": "9:3", "doc_id": 9, "text": "乙" * 200}]
    fused = _rrf_fuse(vec, bm25)
    assert len(fused) == 1, "chunk_id 相同应融合为一条"


def test_add_source_keeps_multiple_chunks_of_same_document():
    from app.agent.research_agent import ResearchRunner

    runner = ResearchRunner(
        task_id="t", query="q", workspace={"type": "web"}, thinking_level="quick",
        web_enabled=False, emit=lambda *a, **k: None,
    )
    a = runner.add_source(origin="doc", title="论文（片段）", url="doc://1", text="第一段")
    b = runner.add_source(origin="doc", title="论文（片段）", url="doc://1", text="第二段")
    assert a and b and a is not b, "同一文档的不同片段都要保留"
    assert len(runner.sources) == 2


def test_add_source_full_text_overrides_snippet():
    from app.agent.research_agent import ResearchRunner

    runner = ResearchRunner(
        task_id="t", query="q", workspace={"type": "web"}, thinking_level="quick",
        web_enabled=False, emit=lambda *a, **k: None,
    )
    runner.add_source(origin="web", title="页面", url="https://e.com/a", text="摘要")
    runner.add_source(origin="web", title="页面", url="https://e.com/a", text="摘要" + "正文" * 100)
    assert len(runner.sources) == 1, "同一 URL 只保留一条信源"
    assert len(runner.sources[0]["text"]) > 10, "抓到全文后应覆盖摘要"


def test_verify_citations_marks_unknown_refs():
    from app.agent.research_agent import ResearchRunner
    from app.agent.websearch import safe_url

    runner = ResearchRunner(
        task_id="t", query="q", workspace={"type": "web"}, thinking_level="quick",
        web_enabled=False, emit=lambda *a, **k: None,
    )
    runner.add_source(origin="web", title="a", url="https://e.com/a", text="x")
    report, note = runner._verify_citations("结论一[1]，结论二[9]。")
    assert "[1]" in report and "[9]" not in report and "[?]" in report
    assert note

    # SSRF 防护：内网与元数据地址一律拒绝
    assert not safe_url("http://127.0.0.1:8765/")
    assert not safe_url("http://169.254.169.254/latest/meta-data/")
    assert not safe_url("file:///etc/passwd")
