"""设置存取（SQLite key-value）。模型 API Key 仅存本机。"""
from __future__ import annotations

from typing import Any, Optional

from sqlmodel import Session

from app.storage.db import engine
from app.storage.models import Setting

# LLM 供应方预设：决定 LiteLLM model 前缀 / 默认 base_url
PROVIDERS = {
    "openai": {"label": "OpenAI", "prefix": "openai", "base_url": "", "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]},
    "anthropic": {"label": "Claude", "prefix": "anthropic", "base_url": "", "models": ["claude-sonnet-4-5", "claude-haiku-4-5"]},
    "deepseek": {"label": "DeepSeek", "prefix": "deepseek", "base_url": "", "models": ["deepseek-chat", "deepseek-reasoner"]},
    "qwen": {"label": "通义千问", "prefix": "openai", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "models": ["qwen-plus", "qwen-max", "qwen-turbo"]},
    "glm": {"label": "智谱 GLM", "prefix": "openai", "base_url": "https://open.bigmodel.cn/api/paas/v4", "models": ["glm-4.6", "glm-4.5-air"]},
    "moonshot": {"label": "月之暗面 Kimi", "prefix": "openai", "base_url": "https://api.moonshot.cn/v1", "models": ["moonshot-v1-32k", "kimi-k2-0905-preview"]},
    "ollama": {"label": "Ollama（本地）", "prefix": "ollama", "base_url": "http://localhost:11434", "models": ["qwen2.5:7b", "llama3.1:8b"]},
    "custom": {"label": "自定义（OpenAI 兼容）", "prefix": "openai", "base_url": "", "models": []},
}

EMBEDDING_PRESETS = [
    {"id": "BAAI/bge-small-zh-v1.5", "label": "BGE 中文 small（推荐，约 130MB）"},
    {"id": "paraphrase-multilingual-MiniLM-L12-v2", "label": "多语言 MiniLM（约 470MB）"},
    {"id": "BAAI/bge-m3", "label": "BGE-M3（效果最好，约 2.3GB）"},
]

DEFAULTS: dict[str, Any] = {
    "model": {  # 旧版单模型配置（保留做迁移源）
        "provider": "openai",
        "api_key": "",
        "base_url": "",
        "model": "gpt-4o-mini",
        "temperature": 0.4,
    },
    "providers": {"items": []},   # 多服务商配置：[{id, provider, api_key, base_url, model, temperature, label}]
    "routing": {},                # 功能路由：{note|doc|research: provider_entry_id}
    "research": {"paradigm": "plan_execute", "react_max_steps": 14, "auto_memory": True,
                 "hybrid_multi_query": True, "hybrid_rerank": False},
    "mcp_servers": {"items": []},   # [{id, name, command, args[], env{}, enabled}]
    "embedding": {"mode": "local", "model": "BAAI/bge-small-zh-v1.5",
                  "api_base_url": "", "api_key": "", "api_model": "text-embedding-3-small"},
    "reminder": {"enabled": False, "time": "09:00"},
    "general": {"nickname": "同学", "daily_goal_minutes": 60},
}

# 功能 → 说明（供 API/前端展示）
ROUTING_FUNCTIONS = {
    "note": "笔记 AI 助手",
    "doc": "文档分析",
    "research": "深度研究",
}


def get(key: str) -> dict:
    with Session(engine) as s:
        row = s.get(Setting, key)
        data = row.data() if row else {}
    return {**DEFAULTS.get(key, {}), **data}


def put(key: str, value: dict, *, merge: bool = True) -> None:
    with Session(engine) as s:
        row = s.get(Setting, key)
        if row is None:
            row = Setting(key=key, value="{}")
            s.add(row)
        if merge:
            merged = {**DEFAULTS.get(key, {}), **(row.data() if row else {}), **value}
        else:
            merged = {**DEFAULTS.get(key, {}), **value}
        import json

        row.value = json.dumps(merged, ensure_ascii=False)
        s.add(row)
        s.commit()


def _provider_entries() -> list[dict]:
    return get("providers").get("items") or []


def _migrate_legacy_model() -> None:
    # 旧版单模型配置 → providers 列表（一次性迁移）
    if _provider_entries():
        return
    legacy = get("model")
    if (legacy.get("api_key") or legacy.get("provider") == "ollama") and legacy.get("model"):
        entry = {k: legacy.get(k, "") for k in ("provider", "api_key", "base_url", "model", "temperature")}
        entry["id"] = "legacy"
        preset_label = PROVIDERS.get(legacy.get("provider", ""), {}).get("label", legacy["provider"])
        entry["label"] = f"{preset_label} · {legacy['model']}"
        put("providers", {"items": [entry]})


def list_provider_entries() -> list[dict]:
    _migrate_legacy_model()
    return _provider_entries()


def save_provider_entries(entries: list[dict]) -> list[dict]:
    # 整表保存：生成缺失 id 与 label
    import uuid

    cleaned = []
    for e in entries:
        provider = (e.get("provider") or "openai").strip()
        model = (e.get("model") or "").strip()
        if not model:
            continue
        eid = (e.get("id") or "").strip() or f"p_{uuid.uuid4().hex[:6]}"
        preset = PROVIDERS.get(provider, PROVIDERS["custom"])
        cleaned.append({
            "id": eid,
            "provider": provider,
            "api_key": (e.get("api_key") or "").strip(),
            "base_url": (e.get("base_url") or "").strip(),
            "model": model,
            "temperature": float(e.get("temperature", 0.4)),
            "label": e.get("label") or f"{preset['label']} · {model}",
        })
    put("providers", {"items": cleaned}, merge=False)
    return cleaned


def get_routing() -> dict[str, str]:
    raw = get("routing")
    return {k: v for k, v in raw.items() if k in ROUTING_FUNCTIONS}


def save_routing(routing: dict[str, str]) -> dict[str, str]:
    # 全量写（merge=False），允许把路由清空
    valid = {k: v for k, v in (routing or {}).items()
             if k in ROUTING_FUNCTIONS and any(e["id"] == v for e in _provider_entries())}
    put("routing", valid, merge=False)
    return valid


def llm_config(fn: str | None = None) -> dict:
    # 解析某功能应使用的模型配置：路由指定 → 第一个条目 → 旧版单配置
    entries = list_provider_entries()
    if fn:
        entry_id = get_routing().get(fn)
        if entry_id:
            for e in entries:
                if e["id"] == entry_id:
                    return e
    if entries:
        return entries[0]
    return get("model")


def is_configured(fn: str | None = None) -> bool:
    cfg = llm_config(fn)
    provider = cfg.get("provider", "")
    if provider == "ollama":
        return bool(cfg.get("model"))
    return bool(cfg.get("api_key") or provider == "ollama") and bool(cfg.get("model"))


# ---------------------------------------------------------------- 研究技能（Markdown playbooks）

def skills_dir():
    import pathlib as _p

    from app.config import DATA_DIR

    d = _p.Path(DATA_DIR) / "skills"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_skills() -> list[dict]:
    out = []
    for f in sorted(skills_dir().glob("*.md")):
        out.append({"id": f.stem, "name": f.stem, "size": f.stat().st_size,
                    "updated": f.stat().st_mtime})
    return out


def read_skill(skill_id: str) -> str:
    if "/" in skill_id or ".." in skill_id:
        raise ValueError("非法技能名")
    f = skills_dir() / f"{skill_id}.md"
    if not f.exists():
        raise KeyError("技能不存在")
    return f.read_text(encoding="utf-8")


def save_skill(skill_id: str, content: str, *, old_id: Optional[str] = None) -> str:
    if "/" in skill_id or ".." in skill_id or not skill_id.strip():
        raise ValueError("非法技能名")
    sid = skill_id.strip()[:40]
    f = skills_dir() / f"{sid}.md"
    f.write_text(content, encoding="utf-8")
    if old_id and old_id != sid:
        old = skills_dir() / f"{old_id}.md"
        if old.exists():
            old.unlink()
    return sid


def delete_skill(skill_id: str) -> None:
    f = skills_dir() / f"{skill_id}.md"
    if f.exists():
        f.unlink()
