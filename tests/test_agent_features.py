"""Skills 渐进加载、MCP 工具校验、置信度阈值与 SSRF 解析的测试。"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.agent import websearch  # noqa: E402
from app.agent.research_agent import ResearchRunner, validate_tool_args  # noqa: E402
from app.services import llm, rag_service, settings_store  # noqa: E402


def _runner(**kw):
    base = dict(task_id="t", query="测试问题", workspace={"type": "web"},
                thinking_level="quick", web_enabled=False, emit=lambda *_: None)
    base.update(kw)
    return ResearchRunner(**base)


def test_validate_tool_args():
    schema = {"type": "object", "required": ["q"], "properties": {
        "q": {"type": "string"}, "n": {"type": "integer"},
        "mode": {"type": "string", "enum": ["a", "b"]}}}
    assert validate_tool_args(schema, {"q": "x", "n": 1, "mode": "a"}) is None
    assert "缺少必填" in validate_tool_args(schema, {})
    assert "类型" in validate_tool_args(schema, {"q": 1})
    assert "整数" in validate_tool_args(schema, {"q": "x", "n": True})
    assert "允许取值" in validate_tool_args(schema, {"q": "x", "mode": "c"})


def test_skill_meta_reads_frontmatter(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_store, "skills_dir", lambda: tmp_path)
    (tmp_path / "rigor.md").write_text(
        "---\nname: 严谨模式\ndescription: 引用要标号\nwhen_to_use: 学术研究\n---\n正文内容",
        encoding="utf-8")
    meta = settings_store.skill_meta("rigor")
    assert meta == {"id": "rigor", "name": "严谨模式",
                    "description": "引用要标号", "when_to_use": "学术研究"}


def test_skill_progressive_loading(monkeypatch):
    skills = [{"id": "a", "name": "A", "description": "d1", "when_to_use": ""},
              {"id": "b", "name": "B", "description": "d2", "when_to_use": ""},
              {"id": "c", "name": "C", "description": "d3", "when_to_use": ""}]

    async def fake_ask_json(system, user, **kw):
        return {"skills": ["b"]}

    monkeypatch.setattr(llm, "ask_json", fake_ask_json)
    runner = _runner(skills=skills, skill_loader=lambda sid: f"{sid}-正文")
    import asyncio

    asyncio.run(runner._prepare_skills())
    assert [s["id"] for s in runner.loaded_skills] == ["b"]
    prompt = runner.skills_prompt()
    assert "a-正文" not in prompt and "b-正文" in prompt
    assert "load_skill" in prompt
    assert runner._load_skill("zzz").startswith("技能 zzz 不在")


def test_mcp_alias_unique_and_collision_safe():
    class FakeClient:
        def __init__(self, name):
            self.name = name
            self.tools = [{"name": "search", "description": "s",
                           "inputSchema": {"type": "object", "properties": {}}}]

    runner = _runner(mcp_clients=[FakeClient("server-one"), FakeClient("server-two")])
    specs = runner._mcp_tool_specs()
    names = [s["function"]["name"] for s in specs]
    assert len(names) == len(set(names)) == 2
    assert all(len(n) <= 64 for n in names)
    assert all(n in runner._mcp_alias for n in names)


def test_low_confidence_annotation(monkeypatch):
    monkeypatch.setattr(rag_service, "_rerank_threshold", lambda: 0.5)
    meta: dict = {}
    rag_service._annotate_confidence(meta, [{"rerank_score": 0.4}])
    assert meta["low_confidence"] is True and meta["top_score"] == 0.4
    meta2: dict = {}
    rag_service._annotate_confidence(meta2, [{"rerank_score": 0.9}])
    assert "low_confidence" not in meta2


def test_ssrf_resolution_blocks_private():
    assert websearch._resolve_checked("127.0.0.1") == []
    assert websearch._resolve_checked("10.0.0.1") == []
    assert websearch._host_allowed("169.254.169.254") is False
    assert isinstance(websearch._proxy_configured(), bool)

def test_skill_selection_empty_means_none(monkeypatch):
    skills = [{"id": c, "name": c, "description": "", "when_to_use": ""} for c in "abc"]

    async def fake_ask_json(system, user, **kw):
        return {"skills": []}

    monkeypatch.setattr(llm, "ask_json", fake_ask_json)
    runner = _runner(skills=skills, skill_loader=lambda sid: f"{sid}-body")
    import asyncio

    asyncio.run(runner._prepare_skills())
    assert runner.loaded_skills == []


def test_skill_truncation_is_marked():
    runner = _runner(skills=[{"id": "long", "name": "L", "description": "", "when_to_use": ""}],
                     skill_loader=lambda sid: "字" * 5000)
    runner._load_skill("long")
    assert "已截断" in runner.skills_prompt()


def test_mcp_sources_separated_by_arguments():
    class FakeClient:
        name = "srv"
        tools = [{"name": "query", "description": "read", "inputSchema": {},
                  "annotations": {"readOnlyHint": True}}]

        async def call(self, tool, args):
            return f"result-{args['q']}"

    runner = _runner(mcp_clients=[FakeClient()])
    runner._mcp_tool_specs()
    alias = next(iter(runner._mcp_alias))
    import asyncio

    asyncio.run(runner._exec_tool(alias, {"q": "a"}))
    asyncio.run(runner._exec_tool(alias, {"q": "b"}))
    assert len(runner.sources) == 2
    assert runner.sources[0]["text"] != runner.sources[1]["text"]


def test_mcp_start_failure_cleans_up():
    from app.agent.mcp_client import McpStdioClient

    client = McpStdioClient("bad", "definitely-not-a-real-command-xyz")
    import asyncio

    try:
        asyncio.run(client.start())
    except Exception:
        pass
    else:
        raise AssertionError("不存在的命令应当启动失败")
    assert client.proc is None and client.started is False and client._reader_task is None


def test_tool_risk_levels():
    from app.agent.research_agent import tool_risk_level

    assert tool_risk_level("delete_file", "", {}) == "high"
    assert tool_risk_level("get_weather", "", {}) == "read"
    assert tool_risk_level("frobnicate", "", {}) == "write"
    assert tool_risk_level("anything", "", {"readOnlyHint": True}) == "read"
    assert tool_risk_level("anything", "", {"destructiveHint": True}) == "high"


def test_pages_for_chunks_carries_context():
    from app.services.rag_service import _pages_for_chunks

    pages = _pages_for_chunks(["【第 2 页】a", "b", "【第 3 页】c", "d"])
    assert pages == [(2, 2), (2, 2), (3, 3), (3, 3)]


def test_mcp_high_risk_waits_for_user_approval():
    events = []
    runner = _runner(emit=lambda e, p: events.append((e, p)))
    import asyncio

    async def go():
        for approved in (True, False):
            task = asyncio.create_task(runner._ask_approval("srv", "delete_file", {"id": 1}))
            await asyncio.sleep(0.05)
            rid = next(iter(runner._approvals))
            runner.resolve_approval(rid, approved)
            assert await task is approved

    asyncio.run(go())
    assert any(e == "approval" for e, _ in events)


def test_index_failure_marks_document_unindexed(monkeypatch):
    from sqlmodel import Session, SQLModel, create_engine

    from app.storage.models import Document

    eng = create_engine("sqlite://")
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        doc = Document(title="t", filename="t.pdf", path="docs/t.pdf", fmt="pdf",
                       size_bytes=10, text_content="正文内容" * 200)
        s.add(doc)
        s.commit()
        s.refresh(doc)
        doc_id = doc.id

    import app.services.rag_service as rs
    monkeypatch.setattr(rs, "engine", eng)
    monkeypatch.setattr(rs, "Session", Session)

    def boom():
        raise RuntimeError("chroma down")

    monkeypatch.setattr(rs, "_chroma_store", boom)
    try:
        rs.index_document(doc_id)
    except Exception:
        pass
    else:
        raise AssertionError("索引失败应当抛出异常")

    with Session(eng) as s:
        fresh = s.get(Document, doc_id)
    assert fresh.indexed is False
    assert fresh.chunk_count == 0
    assert "向量化失败" in (fresh.index_error or "")
