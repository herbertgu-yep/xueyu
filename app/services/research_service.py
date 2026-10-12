"""研究任务服务：DB 持久化 + WebSocket 广播 + Runner 生命周期管理。"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import WebSocket
from sqlmodel import Session, select

from app.agent.research_agent import CancelledByUser, ResearchRunner
from app.config import REPORTS_DIR
from app.storage.db import engine
from app.storage.models import ResearchTask

_subscribers: dict[str, set[WebSocket]] = {}
_runners: dict[str, ResearchRunner] = {}
_tasks: dict[str, asyncio.Task] = {}
_log_files: dict[str, Path] = {}


# ---------------------------------------------------------------- WS 广播

async def subscribe(task_id: str, ws: WebSocket) -> None:
    await ws.accept()
    _subscribers.setdefault(task_id, set()).add(ws)
    try:
        # 先补发当前快照
        task = get_task(task_id)
        await ws.send_json(serialize(task))
        while True:
            await ws.receive_text()  # 保活：客户端不发数据也行，仅挂住连接
    except Exception:
        pass
    finally:
        _subscribers.get(task_id, set()).discard(ws)


async def _broadcast(task_id: str, payload: dict) -> None:
    conns = list(_subscribers.get(task_id, set()))
    if not conns:
        return
    data = json.dumps(payload, ensure_ascii=False)
    for ws in conns:
        try:
            await ws.send_text(data)
        except Exception:
            _subscribers.get(task_id, set()).discard(ws)


# ---------------------------------------------------------------- 任务生命周期

def _log_path(task_id: str) -> Path:
    d = REPORTS_DIR / task_id
    d.mkdir(parents=True, exist_ok=True)
    return d / "activity.log"


def _append_log(task_id: str, line: str) -> None:
    try:
        with open(_log_path(task_id), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def create_task(*, query: str, workspace: dict, thinking_level: str, web_enabled: bool,
                paradigm: str = "plan_execute", skill_ids: Optional[list[str]] = None) -> dict:
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    skill_ids = skill_ids or []
    with Session(engine) as s:
        task = ResearchTask(
            id=task_id,
            query=query.strip(),
            workspace_json=json.dumps(workspace, ensure_ascii=False),
            thinking_level=thinking_level,
            web_enabled=web_enabled,
            paradigm=paradigm if paradigm in ("plan_execute", "react", "multi_agent") else "plan_execute",
            skills_json=json.dumps(skill_ids, ensure_ascii=False),
            status="queued",
        )
        s.add(task)
        s.commit()
        _append_log(task_id, f"[{time.strftime('%H:%M:%S')}] 任务已创建：{query.strip()[:60]}（范式 {task.paradigm}）")
    start_runner(task_id)
    return get_task(task_id)


def start_runner(task_id: str) -> None:
    with Session(engine) as s:
        task = s.get(ResearchTask, task_id)
        if not task:
            raise KeyError("任务不存在")

        def emit(event: str, payload: dict) -> None:
            """Runner 事件 → DB + WS（在 runner 的 loop 里回调）。"""
            _handle_event_sync(task_id, event, payload)

        # 技能内容
        from app.services import settings_store

        skills = []
        for sid in json.loads(task.skills_json or "[]"):
            try:
                skills.append({"name": sid, "content": settings_store.read_skill(sid)})
            except Exception:
                pass
        cfg = settings_store.get("research")
        memory_path = Path(REPORTS_DIR) / task_id / "memory.md"

        runner = ResearchRunner(
            task_id=task_id,
            query=task.query,
            workspace=json.loads(task.workspace_json or "{}"),
            thinking_level=task.thinking_level,
            web_enabled=task.web_enabled,
            emit=emit,
            paradigm=getattr(task, "paradigm", None) or "plan_execute",
            skills=skills,
            react_max_steps=cfg.get("react_max_steps", 14),
            auto_memory=cfg.get("auto_memory", True),
            memory_path=memory_path,
            hybrid_multi_query=cfg.get("hybrid_multi_query", True),
            hybrid_rerank=cfg.get("hybrid_rerank", False),
        )
        _runners[task_id] = runner
        _set_status(task_id, "running", step="排队中")

        async def _run():
            mcp_clients = []
            try:
                # MCP：连接启用的服务器，把外部工具并入 Agent
                servers = settings_store.get("mcp_servers").get("items") or []
                if servers and cfg.get("mcp_enabled", True):
                    from app.agent.mcp_client import connect_all

                    mcp_clients, errs = await connect_all(servers)
                    for err in errs:
                        _append_log(task_id, f"[{time.strftime('%H:%M:%S')}] ⚠️ MCP 连接失败：{err}")
                    if mcp_clients:
                        runner.mcp_clients = mcp_clients
                        n = sum(len(c.tools) for c in mcp_clients)
                        _append_log(task_id, f"[{time.strftime('%H:%M:%S')}] 🔌 MCP 已连接 {len(mcp_clients)} 个服务器 / {n} 个工具")
                result = await runner.run()
                _finalize(task_id, result)
            except CancelledByUser:
                _set_status(task_id, "cancelled", finished=True, step="已取消")
                _broadcast_sync(task_id, {"type": "status", "status": "cancelled"})
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                _set_status(task_id, "failed", error=msg, finished=True, step="失败")
                _append_log(task_id, f"[{time.strftime('%H:%M:%S')}] ❌ 任务失败：{msg}")
                _broadcast_sync(task_id, {"type": "status", "status": "failed", "error": msg})
            finally:
                for c in mcp_clients:
                    await c.stop()
                _runners.pop(task_id, None)
                _tasks.pop(task_id, None)

        _tasks[task_id] = asyncio.get_running_loop().create_task(_run())


def _handle_event_sync(task_id: str, event: str, payload: dict) -> None:
    """在事件循环内被 runner 调用：更新 DB 并广播。"""
    if event == "log":
        _append_log(task_id, f"[{payload.get('time')}] {payload['message']}")
        _broadcast_sync(task_id, {"type": "log", **payload})
    elif event == "progress":
        _update(task_id, progress=payload["progress"], current_step=payload["step"])
        _broadcast_sync(task_id, {"type": "progress", **payload})
    elif event == "sources":
        _update(task_id, sources_visited=payload["sources_visited"])
        _broadcast_sync(task_id, {"type": "sources", **payload})
    elif event == "conflicts":
        _update(task_id, conflicts_found=payload["conflicts_found"])
        _broadcast_sync(task_id, {"type": "conflicts", **payload})


def _broadcast_sync(task_id: str, payload: dict) -> None:
    """runner 与 loop 同处一个事件循环，直接调度广播。"""
    asyncio.get_running_loop().create_task(_broadcast(task_id, payload))


def _set_status(task_id: str, status: str, *, error: Optional[str] = None,
                finished: bool = False, step: str = "") -> None:
    fields = {"status": status, "current_step": step}
    if error is not None:
        fields["error"] = error
    if finished:
        fields["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if status == "cancelled":
            fields["progress"] = 0.0
    _update(task_id, **fields)


def _update(task_id: str, **fields) -> None:
    with Session(engine) as s:
        task = s.get(ResearchTask, task_id)
        if not task:
            return
        for k, v in fields.items():
            setattr(task, k, v)
        s.add(task)
        s.commit()


def _finalize(task_id: str, result: dict) -> None:
    """报告落盘 + DB 收尾 + 系统通知。"""
    d = REPORTS_DIR / task_id
    d.mkdir(parents=True, exist_ok=True)
    report_md = result["report_md"]
    sources = result["sources"]
    conflicts = result["conflicts"]
    title = result.get("report_title") or "研究报告"

    refs_md = "\n\n## 参考文献\n\n" + "\n".join(
        f"{i + 1}. {s['title']}" + (f" — {s['url']}" if s["url"] and not s["url"].startswith(("doc://", "note://")) else "")
        + (f"（本地信源：{s['origin']}）" if s["url"].startswith(("doc://", "note://")) else "")
        for i, s in enumerate(sources)
    )
    (d / "report.md").write_text(report_md + refs_md, encoding="utf-8")
    meta = {
        "title": title,
        "citations": [{"ref_id": s["ref_id"], "title": s["title"], "url": s["url"],
                       "origin": s["origin"], "accessed_at": s["accessed_at"]} for s in sources],
        "conflicts": conflicts,
        "word_count": len(report_md),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (d / "task.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    _update(task_id, status="completed", progress=1.0, current_step="已完成",
            report_title=title, report_word_count=len(report_md),
            report_json=json.dumps({"citations": meta["citations"], "conflicts": conflicts},
                                   ensure_ascii=False),
            report_dir=str(d), sources_visited=len(sources), conflicts_found=len(conflicts),
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    _append_log(task_id, f"[{time.strftime('%H:%M:%S')}] 🎉 研究完成：《{title}》")
    _broadcast_sync(task_id, {"type": "status", "status": "completed", "report_title": title})

    # macOS 系统通知（后台运行的核心承诺）
    from app.services.notify import notify

    notify("学屿 · 研究完成 🎉", f"《{title}》已生成，含 {len(sources)} 条引用", task_id=task_id)


# ---------------------------------------------------------------- 查询 / 控制

def serialize(task: ResearchTask, *, with_report: bool = False) -> dict:
    report = None
    if task.report_json:
        try:
            report = json.loads(task.report_json)
        except Exception:
            report = None
    out = {
        "id": task.id,
        "query": task.query,
        "workspace": json.loads(task.workspace_json or "{}"),
        "thinking_level": task.thinking_level,
        "web_enabled": task.web_enabled,
        "paradigm": getattr(task, "paradigm", "plan_execute"),
        "status": task.status,
        "progress": task.progress,
        "current_step": task.current_step,
        "sources_visited": task.sources_visited,
        "conflicts_found": task.conflicts_found,
        "error": task.error,
        "created_at": task.created_at,
        "finished_at": task.finished_at,
        "report_title": task.report_title,
        "report_word_count": task.report_word_count,
        "report": report,
    }
    if with_report:
        rp = (Path(task.report_dir) / "report.md") if task.report_dir else None
        out["report_md"] = rp.read_text(encoding="utf-8") if rp and rp.exists() else None
    return out


def get_task(task_id: str, *, with_report: bool = False) -> dict:
    with Session(engine) as s:
        task = s.get(ResearchTask, task_id)
        if not task:
            raise KeyError("研究任务不存在")
        return serialize(task, with_report=with_report)


def list_tasks() -> list[dict]:
    with Session(engine) as s:
        rows = s.exec(select(ResearchTask).order_by(ResearchTask.created_at.desc())).all()
        return [serialize(t) for t in rows]


def recent_logs(task_id: str, limit: int = 200) -> list[str]:
    p = _log_path(task_id)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    return lines[-limit:]


def running_count() -> int:
    with Session(engine) as s:
        return sum(1 for t in s.exec(select(ResearchTask)).all() if t.status in ("running", "queued"))


def pause_task(task_id: str) -> dict:
    runner = _runners.get(task_id)
    if not runner:
        raise ValueError("任务不在运行中")
    runner.pause()
    _set_status(task_id, "paused", step="已暂停")
    _broadcast_sync(task_id, {"type": "status", "status": "paused"})
    return get_task(task_id)


def resume_task(task_id: str) -> dict:
    runner = _runners.get(task_id)
    if not runner:
        raise ValueError("任务不在运行中")
    runner.resume()
    _set_status(task_id, "running", step="继续运行")
    _broadcast_sync(task_id, {"type": "status", "status": "running"})
    return get_task(task_id)


def cancel_task(task_id: str) -> dict:
    runner = _runners.get(task_id)
    if not runner:
        _set_status(task_id, "cancelled", finished=True, step="已取消")
        return get_task(task_id)
    runner.cancel()
    return get_task(task_id)


def delete_task(task_id: str) -> None:
    with Session(engine) as s:
        task = s.get(ResearchTask, task_id)
        if task:
            s.delete(task)
            s.commit()
    import shutil

    shutil.rmtree(REPORTS_DIR / task_id, ignore_errors=True)


def interrupted_on_restart() -> None:
    """服务启动时把上次遗留的 running/queued 任务标记为失败（诚实处理，不假装还在跑）。"""
    with Session(engine) as s:
        rows = s.exec(select(ResearchTask)).all()
        for t in rows:
            if t.status in ("running", "queued", "paused"):
                t.status = "failed"
                t.error = "应用重启导致任务中断，请重新发起研究"
                t.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                s.add(t)
        s.commit()


# ---------------------------------------------------------------- 报告后处理

def load_report_context(task_id: str) -> tuple[str, list[dict]]:
    """(report_md, sources)；供追问 / 转笔记 / 复习计划使用。"""
    data = get_task(task_id, with_report=True)
    report_md = data.get("report_md") or ""
    sources = (data.get("report") or {}).get("citations", [])
    return report_md, sources


def export_markdown(task_id: str) -> str:
    data = get_task(task_id, with_report=True)
    if data.get("report_md"):
        return data["report_md"]
    raise KeyError("报告尚未生成")
