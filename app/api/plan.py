"""学习计划 / 复习 API。"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services import plan_engine

router = APIRouter(prefix="/api/plan", tags=["plan"])


class PlanIn(BaseModel):
    title: str = ""
    content: str = ""
    source_type: str = "manual"
    source_id: Optional[str] = None
    daily_minutes: int = Field(default=30, ge=5, le=480)
    flashcards: Optional[list[dict]] = None
    schedule_type: str = "normal"  # normal | short | long | temp


@router.get("/today")
def today(with_content: bool = False):
    return plan_engine.today_queue(with_content=with_content)


@router.get("/items")
def items(include_archived: bool = False):
    return plan_engine.list_items(include_archived=include_archived)


@router.get("/upcoming")
def upcoming(limit: int = 8):
    return plan_engine.upcoming_points(limit=limit)


@router.get("/day")
def day(date: Optional[str] = None, with_content: bool = False):
    return plan_engine.day_queue(date, with_content=with_content)


@router.get("/week")
def week(anchor: Optional[str] = None):
    return plan_engine.week_view(anchor)


@router.get("/month")
def month(year: int = Query(...), month: int = Query(...)):
    return plan_engine.month_view(year, month)


@router.get("/schedule/{item_id}")
def schedule(item_id: int):
    try:
        return plan_engine.plan_schedule(item_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


class LeaveIn(BaseModel):
    days: int


@router.post("/leave")
def leave(body: LeaveIn):
    return plan_engine.apply_leave(body.days)


@router.get("/stats")
def stats():
    return plan_engine.stats()


@router.post("")
def create_plan(body: PlanIn):
    if not body.title.strip() and not body.content.strip():
        raise HTTPException(400, "计划标题或内容不能都为空")
    return plan_engine.create_plan(
        title=body.title or body.content.strip().splitlines()[0][:40],
        content=body.content, source_type=body.source_type,
        source_id=body.source_id, daily_minutes=body.daily_minutes,
        flashcards=body.flashcards, schedule_type=body.schedule_type,
    )


@router.get("/{item_id}")
def get_plan(item_id: int):
    try:
        return plan_engine.get_item(item_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.patch("/{item_id}")
def update_plan(item_id: int, body: dict):
    try:
        return plan_engine.update_item(
            item_id, title=body.get("title"), daily_minutes=body.get("daily_minutes"))
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.delete("/{item_id}")
def delete_plan(item_id: int):
    plan_engine.delete_item(item_id)
    return {"ok": True}


class FlashcardsGen(BaseModel):
    count: int = Field(default=12, ge=3, le=40)


@router.post("/{item_id}/flashcards")
async def gen_flashcards(item_id: int, body: FlashcardsGen | None = None):
    """用 AI 从素材生成复习卡片（真实 LLM 调用）。"""
    from app.services import llm, notes_ai, plan_engine

    count = (body.count if body else 12) or 12
    try:
        item = plan_engine.get_item(item_id)
    except KeyError:
        raise HTTPException(404, "复习项不存在")
    content = item.get("content") or ""
    if not content.strip():
        raise HTTPException(400, "该计划没有素材内容，无法生成卡片")
    try:
        cards = await notes_ai.generate_flashcards_async(content, title=item["title"], count=count)
    except (llm.LLMConfigError, llm.LLMCallError) as e:
        raise HTTPException(400, str(e))
    return plan_engine.set_flashcards(item_id, cards)


# ---------------------------------------------------------------- 复习计时

review_router = APIRouter(prefix="/api/review", tags=["review"])


@review_router.post("/{item_id}/start")
def review_start(item_id: int):
    try:
        return plan_engine.start_review(item_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@review_router.post("/{item_id}/pause")
def review_pause(item_id: int):
    try:
        return plan_engine.pause_review(item_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


@review_router.post("/{item_id}/reset")
def review_reset(item_id: int):
    try:
        return plan_engine.reset_timer(item_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


class FinishIn(BaseModel):
    known: int = 0
    unknown: int = 0
    accuracy: Optional[float] = None


@review_router.post("/{item_id}/finish")
def review_finish(item_id: int, body: FinishIn):
    try:
        return plan_engine.finish_review(
            item_id, known=body.known, unknown=body.unknown, accuracy=body.accuracy)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
