"""系统 API：设置 / 概览 / 全局搜索 / 工具。"""
from __future__ import annotations

import re
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import APP_NAME, APP_VERSION, DATA_DIR
from app.services import notes_service, plan_engine, research_service, settings_store, shelf_service

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/overview")
def overview():
    """首页 + 侧栏数据（一次拉全）。"""
    plans = plan_engine.list_items()
    active_plans = [p for p in plans if p["status"] == "active"]
    research = research_service.list_tasks()
    running = [t for t in research if t["status"] in ("running", "queued")]
    recent_reports = [t for t in research if t["status"] == "completed"][:3]
    notes = notes_service.list_notes()
    folders = notes_service.list_folders()
    stats = plan_engine.stats()
    shelf = shelf_service.list_documents()
    cats = shelf_service.list_categories()
    recent_notes = sorted(notes, key=lambda n: n["updated"], reverse=True)[:4]
    return {
        "app": {"name": APP_NAME, "version": APP_VERSION, "data_dir": str(DATA_DIR)},
        "stats": stats,
        "plans": active_plans[:6],
        "today": {"total": stats["today_total"], "done": stats["today_done"]},
        "research_running": running,
        "research_recent": recent_reports,
        "notes_count": len(notes),
        "folders": folders,
        "recent_notes": recent_notes,
        "docs_count": len(shelf),
        "categories": cats,
        "ai_configured": settings_store.is_configured(),
    }


# ---------------------------------------------------------------- 设置

def _embedding_local_available() -> bool:
    from app.services.rag_service import local_embedding_available

    return local_embedding_available()


def _mask(key: str) -> str:
    return ("•" * 8 + key[-4:]) if key else ""


@router.get("/settings")
def get_settings():
    entries = settings_store.list_provider_entries()
    routing = settings_store.get_routing()
    return {
        "providers_catalog": [
            {"id": pid, "label": p["label"], "base_url": p["base_url"], "models": p["models"]}
            for pid, p in settings_store.PROVIDERS.items()
        ],
        "provider_entries": [{k: (v if k != "api_key" else _mask(v)) for k, v in e.items()} for e in entries],
        "routing": routing,
        "routing_functions": settings_store.ROUTING_FUNCTIONS,
        "model_configured": settings_store.is_configured(),
        "embedding": settings_store.get("embedding"),
        "embedding_local_available": _embedding_local_available(),
        "reminder": settings_store.get("reminder"),
        "general": settings_store.get("general"),
        "embedding_presets": settings_store.EMBEDDING_PRESETS,
    }


class ProvidersIn(BaseModel):
    providers: list[dict]


@router.put("/settings/providers")
def put_providers(body: ProvidersIn):
    """整表保存服务商条目；掩码 Key（• 开头）保留原值。"""
    old_entries = {e["id"]: e for e in settings_store.list_provider_entries()}
    for e in body.providers:
        key = (e.get("api_key") or "").strip()
        eid = (e.get("id") or "").strip()
        if key.startswith("•") and eid in old_entries:
            e["api_key"] = old_entries[eid].get("api_key", "")
        elif key.startswith("•"):
            e["api_key"] = ""
    saved = settings_store.save_provider_entries(body.providers)
    # 路由指向被删除的条目时清理
    routing = settings_store.get_routing()
    valid_ids = {e["id"] for e in saved}
    if any(v not in valid_ids for v in routing.values()):
        settings_store.save_routing({k: v for k, v in routing.items() if v in valid_ids})
    return {"ok": True, "configured": settings_store.is_configured(),
            "providers": [{k: (v if k != "api_key" else _mask(v)) for k, v in e.items()} for e in saved]}


class RoutingIn(BaseModel):
    routing: dict[str, str]


@router.put("/settings/routing")
def put_routing(body: RoutingIn):
    saved = settings_store.save_routing(body.routing)
    return {"ok": True, "routing": saved}


class TestModelIn(BaseModel):
    prompt: str = "用一个词回答：1+1等于几？"
    provider_id: Optional[str] = None      # 测试某个已保存条目
    config: Optional[dict] = None          # 或直接内联测试一份未保存配置


@router.post("/settings/test")
async def test_model(body: TestModelIn):
    import asyncio

    return await asyncio.to_thread(_test_model_impl, body)


def _test_model_impl(body: TestModelIn):
    from app.services import llm
    import litellm as _litellm

    try:
        if body.config:
            # 测试未保存的表单配置（不落盘）
            cfg = {"provider": body.config.get("provider", "openai"),
                   "api_key": body.config.get("api_key", ""),
                   "base_url": body.config.get("base_url", ""),
                   "model": body.config.get("model", ""),
                   "temperature": 0}
            if not cfg["model"]:
                raise llm.LLMConfigError("模型名称未填写")
            kwargs = {"model": llm._model_string(cfg), "messages": [{"role": "user", "content": body.prompt}],
                      **llm._call_kwargs(cfg), "temperature": 0}
            try:
                resp = _litellm.completion(**kwargs)
            except Exception as e:  # noqa: BLE001
                raise llm.LLMCallError(llm._brief(e)) from e
            return {"ok": True, "reply": (resp.choices[0].message.content or "")[:200]}
        fn_cfg = None
        if body.provider_id:
            entries = settings_store.list_provider_entries()
            fn_cfg = next((e for e in entries if e["id"] == body.provider_id), None)
            if not fn_cfg:
                raise HTTPException(404, "找不到该服务商条目，请先保存")
        reply = llm.acompletion([{"role": "user", "content": body.prompt}], temperature=0,
                                fn="note" if fn_cfg is None else None)
        if fn_cfg is not None and settings_store.get_routing().get("note") != fn_cfg["id"]:
            # 测试指定条目：临时用该条目的连接参数直连一次
            kwargs = {"model": llm._model_string(fn_cfg),
                      "messages": [{"role": "user", "content": body.prompt}],
                      **llm._call_kwargs(fn_cfg), "temperature": 0}
            try:
                resp = _litellm.completion(**kwargs)
            except Exception as e:  # noqa: BLE001
                raise llm.LLMCallError(llm._brief(e)) from e
            return {"ok": True, "reply": (resp.choices[0].message.content or "")[:200]}
        return {"ok": True, "reply": reply[:200]}
    except (llm.LLMConfigError, llm.LLMCallError) as e:
        raise HTTPException(400, str(e))


@router.get("/settings/research")
def get_research_settings():
    return settings_store.get("research")


class ResearchSettingsIn(BaseModel):
    paradigm: str = "plan_execute"
    react_max_steps: int = 14
    auto_memory: bool = True
    mcp_enabled: bool = True
    hybrid_multi_query: bool = True
    hybrid_rerank: bool = False
    semantic_cache: bool = True


@router.put("/settings/research")
def put_research_settings(body: ResearchSettingsIn):
    if body.paradigm not in ("plan_execute", "react", "multi_agent"):
        raise HTTPException(400, "未知范式")
    if not (4 <= body.react_max_steps <= 40):
        raise HTTPException(400, "步数应在 4-40 之间")
    settings_store.put("research", body.model_dump(), merge=False)
    return {"ok": True}


@router.get("/settings/mcp")
def list_mcp():
    return settings_store.get("mcp_servers").get("items") or []


class McpServerIn(BaseModel):
    id: Optional[str] = None
    name: str
    command: str
    args: list[str] = []
    env: dict = {}
    enabled: bool = True


@router.post("/settings/mcp")
def add_mcp(body: McpServerIn):
    import uuid

    items = settings_store.get("mcp_servers").get("items") or []
    item = body.model_dump()
    item["id"] = body.id or f"mcp_{uuid.uuid4().hex[:6]}"
    items.append(item)
    settings_store.put("mcp_servers", {"items": items}, merge=False)
    return item


@router.put("/settings/mcp/{mid}")
def update_mcp(mid: str, body: McpServerIn):
    items = settings_store.get("mcp_servers").get("items") or []
    for i, it in enumerate(items):
        if it.get("id") == mid:
            new_item = body.model_dump()
            new_item["id"] = mid
            items[i] = new_item
            settings_store.put("mcp_servers", {"items": items}, merge=False)
            return new_item
    raise HTTPException(404, "MCP 服务器不存在")


@router.delete("/settings/mcp/{mid}")
def delete_mcp(mid: str):
    items = settings_store.get("mcp_servers").get("items") or []
    items = [it for it in items if it.get("id") != mid]
    settings_store.put("mcp_servers", {"items": items}, merge=False)
    return {"ok": True}


class McpTestIn(BaseModel):
    command: str
    args: list[str] = []
    env: dict = {}


@router.post("/settings/mcp/test")
async def test_mcp(body: McpTestIn):
    """真实连接 MCP 服务器并列出工具。"""
    from app.agent.mcp_client import McpStdioClient

    client = McpStdioClient("test", body.command, body.args, body.env, timeout=12)
    try:
        await client.start()
        return {"ok": True, "tools": [t["name"] for t in client.tools]}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"MCP 连接失败：{e}")
    finally:
        await client.stop()


class EmbeddingIn(BaseModel):
    mode: str = "local"            # local | api
    model: str = "BAAI/bge-small-zh-v1.5"
    api_base_url: str = ""
    api_key: str = ""
    api_model: str = "text-embedding-3-small"


@router.put("/settings/embedding")
def put_embedding(body: EmbeddingIn):
    if body.mode not in ("local", "api"):
        raise HTTPException(400, "embedding 模式只能是 local 或 api")
    old = settings_store.get("embedding")
    new_cfg = body.model_dump()
    # 掩码 Key 保留原值
    if new_cfg["api_key"].startswith("•"):
        new_cfg["api_key"] = old.get("api_key", "")
    changed = (old.get("mode") != new_cfg["mode"] or old.get("model") != new_cfg["model"]
               or old.get("api_base_url") != new_cfg["api_base_url"]
               or old.get("api_key") != new_cfg["api_key"] or old.get("api_model") != new_cfg["api_model"])
    settings_store.put("embedding", new_cfg, merge=False)
    if changed:
        from app.services import rag_service

        rag_service.reset_client()
        rag_service.reindex_all()  # 向量空间变化，自动全量重建
        return {"ok": True, "reindexed": True}
    return {"ok": True, "reindexed": False}


class EmbeddingTestIn(BaseModel):
    mode: str = "local"
    model: str = "BAAI/bge-small-zh-v1.5"
    api_base_url: str = ""
    api_key: str = ""
    api_model: str = "text-embedding-3-small"


@router.post("/settings/embedding/test")
async def test_embedding(body: EmbeddingTestIn):
    """真实调用 embedding（本地或 API），返回维度与耗时。"""
    import time as _time

    from app.services.rag_service import _make_api_embedding, _make_local_embedding

    try:
        if body.mode == "local" and not _embedding_local_available():
            raise HTTPException(400, "打包版不含本地向量化模型，请使用 API 服务模式")
        t0 = _time.time()
        emb = _make_local_embedding(body.model) if body.mode == "local" else _make_api_embedding(body.model_dump())
        vec = emb.get_text_embedding("间隔重复测试")
        ms = int((_time.time() - t0) * 1000)
        return {"ok": True, "dim": len(vec), "ms": ms}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"embedding 测试失败：{e}")


class ReminderIn(BaseModel):
    enabled: bool
    time: str = "09:00"


@router.put("/settings/reminder")
def put_reminder(body: ReminderIn):
    if not re.match(r"^\d{2}:\d{2}$", body.time):
        raise HTTPException(400, "时间格式应为 HH:MM")
    settings_store.put("reminder", body.model_dump())
    from app.main import reschedule_reminder

    reschedule_reminder()
    return {"ok": True}


class GeneralIn(BaseModel):
    nickname: str = "同学"
    daily_goal_minutes: int = 60


@router.put("/settings/general")
def put_general(body: GeneralIn):
    settings_store.put("general", body.model_dump())
    return {"ok": True}


# ---------------------------------------------------------------- 全局搜索 ⌘K

@router.get("/search")
def global_search(q: str):
    if not q.strip():
        return {"notes": [], "docs": [], "reports": [], "plans": []}
    ql = q.lower()
    plans = [{"id": p["id"], "title": p["title"], "stage_label": p["stage_label"]}
             for p in plan_engine.list_items() if ql in p["title"].lower()][:8]
    docs = [{"id": d["id"], "title": d["title"], "fmt": d["fmt"]}
            for d in shelf_service.list_documents() if ql in d["title"].lower()][:8]
    reports = [{"id": t["id"], "title": t.get("report_title") or t["query"]}
               for t in research_service.list_tasks() if t["status"] == "completed"
               and (ql in (t.get("report_title") or "").lower() or ql in t["query"].lower())][:8]
    return {"notes": notes_service.search(q, 8), "docs": docs, "reports": reports, "plans": plans}


# ---------------------------------------------------------------- 工具

class ImportIn(BaseModel):
    path: str
    category_id: Optional[int] = None
    title: Optional[str] = None


@router.post("/tools/import_local")
def import_local(body: ImportIn):
    """从本地路径导入文件（研究页上传材料等场景）。"""
    try:
        return shelf_service.import_local(body.path, category_id=body.category_id, title=body.title)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/tools/dump_config")
def dump_config():
    """导出脱敏配置（排障用）。"""
    cfg = settings_store.get("model")
    return {"provider": cfg.get("provider"), "model": cfg.get("model"),
            "base_url": cfg.get("base_url"), "api_key_set": bool(cfg.get("api_key")),
            "data_dir": str(DATA_DIR)}
