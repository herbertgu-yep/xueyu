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