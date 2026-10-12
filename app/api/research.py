"""深度研究 API + WebSocket。"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.services import research_service

router = APIRouter(prefix="/api/research", tags=["research"])


class Workspace(BaseModel):
    type: str = "web"  # web | notes | docs | shelf_rag
    note_ids: list[str] = []
    doc_ids: list[int] = []


class ResearchIn(BaseModel):
    query: str
    workspace: Workspace = Workspace()
    thinking_level: str = "standard"  # quick | standard | deep
    web_enabled: bool = True
    paradigm: str = "plan_execute"    # plan_execute | react | multi_agent
    skill_ids: list[str] = []


@router.get("")
def list_tasks():
    return research_service.list_tasks()


@router.post("")
async def create_task(body: ResearchIn):
    if len(body.query.strip()) < 6:
        raise HTTPException(400, "研究问题太短了——说清楚范围和关心的维度，研究质量才高")
    if body.thinking_level not in ("quick", "standard", "deep"):
        raise HTTPException(400, "思考程度只能是 quick / standard / deep")
    return research_service.create_task(
        query=body.query,
        workspace=body.workspace.model_dump(),
        thinking_level=body.thinking_level,
        web_enabled=body.web_enabled,
        paradigm=body.paradigm,
        skill_ids=body.skill_ids,
    )


# ---------------- 技能（研究 playbook） ----------------
@router.get("/skills")
def list_skills():
    from app.services import settings_store

    return settings_store.list_skills()


@router.get("/skills/{skill_id}")
def get_skill(skill_id: str):
    from app.services import settings_store

    try:
        return {"id": skill_id, "content": settings_store.read_skill(skill_id)}
    except (KeyError, ValueError) as e:
        raise HTTPException(404, str(e))


class SkillIn(BaseModel):
    name: str
    content: str


@router.post("/skills")
def create_skill(body: SkillIn):
    from app.services import settings_store

    try:
        sid = settings_store.save_skill(body.name, body.content)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": sid}


@router.put("/skills/{skill_id}")
def update_skill(skill_id: str, body: SkillIn):
    from app.services import settings_store

    try:
        sid = settings_store.save_skill(body.name, body.content, old_id=skill_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": sid}


@router.delete("/skills/{skill_id}")
def delete_skill(skill_id: str):
    from app.services import settings_store

    settings_store.delete_skill(skill_id)
    return {"ok": True}


@router.get("/{task_id}")
def get_task(task_id: str, with_report: bool = False):
    try:
        return research_service.get_task(task_id, with_report=with_report)
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.get("/{task_id}/logs", response_class=PlainTextResponse)
def logs(task_id: str):
    try:
        return "\n".join(research_service.recent_logs(task_id))
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.post("/{task_id}/pause")
def pause(task_id: str):
    try:
        return research_service.pause_task(task_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/{task_id}/resume")
def resume(task_id: str):
    try:
        return research_service.resume_task(task_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/{task_id}/cancel")
def cancel(task_id: str):
    try:
        return research_service.cancel_task(task_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.delete("/{task_id}")
def delete(task_id: str):
    research_service.delete_task(task_id)
    return {"ok": True}


class FollowupIn(BaseModel):
    question: str
    history: Optional[list[dict]] = None


@router.post("/{task_id}/followup")
async def followup(task_id: str, body: FollowupIn):
    if not body.question.strip():
        raise HTTPException(400, "追问内容不能为空")
    from app.services import llm as llm_mod

    try:
        report_md, sources = research_service.load_report_context(task_id)
        if not report_md:
            raise HTTPException(400, "报告尚未生成")
        refs = "\n".join(f"[{s['ref_id']}] {s['title']} {s.get('url', '')}" for s in sources)
        msgs = [{"role": "system", "content":
                 "你是研究助理。基于给定的研究报告与信源列表回答追问；可以继续用 [n] 引用编号信源；"
                 "报告没有覆盖的内容要明确说明。输出 Markdown。"}]
        for h in (body.history or [])[-6:]:
            msgs.append({"role": "assistant" if h.get("role") in ("assistant", "bot") else "user",
                         "content": str(h.get("content", ""))[:2500]})
        msgs.append({"role": "user",
                     "content": f"研究报告：\n{report_md[:16000]}\n\n信源列表：\n{refs}\n\n追问：{body.question}"})
        answer = await llm_mod.achat(msgs, temperature=0.3, fn="research", compact=True)
        return {"answer": answer}
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(404, str(e))
    except (llm_mod.LLMConfigError, llm_mod.LLMCallError) as e:
        raise HTTPException(400, str(e))


@router.get("/{task_id}/export", response_class=PlainTextResponse)
def export_md(task_id: str):
    try:
        return research_service.export_markdown(task_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.post("/{task_id}/note")
def to_note(task_id: str):
    """报告存入笔记。"""
    from app.services import notes_service

    try:
        data = research_service.get_task(task_id, with_report=True)
    except KeyError as e:
        raise HTTPException(404, str(e))
    if not data.get("report_md"):
        raise HTTPException(400, "报告尚未生成")
    meta = data.get("report") or {}
    tags = ["研究报告"]
    if meta.get("conflicts"):
        tags.append("有冲突标注")
    note = notes_service.create_note(
        title=data.get("report_title") or data["query"],
        folder="研究", tags=tags, content=data["report_md"])
    return note


@router.post("/{task_id}/plan")
async def to_plan(task_id: str):
    """按报告生成复习计划（素材 = 结论先行 + 要点）。"""
    from app.services import notes_ai, plan_engine

    try:
        data = research_service.get_task(task_id, with_report=True)
    except KeyError as e:
        raise HTTPException(404, str(e))
    report_md = data.get("report_md") or ""
    if not report_md:
        raise HTTPException(400, "报告尚未生成")
    title = data.get("report_title") or data["query"]
    content = report_md[:9000]
    flashcards = None
    try:
        flashcards = await notes_ai.generate_flashcards_async(content, title=title, count=10)
    except Exception:  # noqa: BLE001 卡片可选，报告计划本身必须创建成功
        flashcards = None
    return plan_engine.create_plan(
        title=f"研究回顾：{title[:36]}", content=content, source_type="report",
        source_id=task_id, daily_minutes=30, flashcards=flashcards)
