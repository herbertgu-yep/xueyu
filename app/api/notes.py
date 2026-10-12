"""笔记 API：CRUD（frontmatter md）+ AI 操作 + 历史版本。"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import notes_ai, notes_service

router = APIRouter(prefix="/api/notes", tags=["notes"])


@router.get("/folders")
def folders():
    return notes_service.list_folders()


@router.get("/search")
def search(q: str, limit: int = 30):
    return notes_service.search(q, limit)


@router.get("")
def list_notes(folder: Optional[str] = None, q: Optional[str] = None):
    return notes_service.list_notes(folder=folder, q=q)


class NoteIn(BaseModel):
    title: str = ""
    folder: str = "未分类"
    tags: Optional[list[str]] = None
    content: str = ""
    ai_assisted: bool = False


@router.post("")
def create(body: NoteIn):
    return notes_service.create_note(**body.model_dump())


class FolderRename(BaseModel):
    old: str
    new: str


@router.put("/folders")
def rename_folder(body: FolderRename):
    try:
        return notes_service.rename_folder(body.old, body.new)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/folders/{name}")
def delete_folder(name: str):
    try:
        return notes_service.delete_folder(name)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/{note_id}")
def read(note_id: str):
    try:
        return notes_service.read_note(note_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.put("/{note_id}")
def update(note_id: str, body: dict):
    try:
        return notes_service.update_note(
            note_id,
            title=body.get("title"), folder=body.get("folder"), tags=body.get("tags"),
            content=body.get("content"), ai_assisted=body.get("ai_assisted"),
        )
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.delete("/{note_id}")
def delete(note_id: str):
    notes_service.delete_note(note_id)
    return {"ok": True}


class AiIn(BaseModel):
    action: str  # summarize | continue | polish | quiz | chat
    message: Optional[str] = None
    history: Optional[list[dict]] = None
    web: bool = False
    thinking: str = "standard"


@router.post("/{note_id}/ai")
async def ai(note_id: str, body: AiIn):
    # 同步 LLM 调用放线程池——直接在事件循环里跑会阻塞整站（致命）
    import asyncio

    try:
        return await asyncio.to_thread(
            notes_ai.run_action, note_id, body.action, body.message, body.history,
            web=body.web, thinking=body.thinking)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except (notes_ai.llm.LLMConfigError, notes_ai.llm.LLMCallError) as e:
        raise HTTPException(400, str(e))


class PlanFromNoteIn(BaseModel):
    daily_minutes: int = 30
    generate_flashcards: bool = False
    schedule_type: str = "normal"


@router.post("/{note_id}/plan")
async def make_plan(note_id: str, body: PlanFromNoteIn):
    """把笔记加入复习计划（可选 AI 生成卡片）。"""
    from app.services import llm, plan_engine

    try:
        note = notes_service.read_note(note_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    content = note["content"]
    flashcards = None
    if body.generate_flashcards and content.strip():
        try:
            flashcards = await notes_ai.generate_flashcards_async(
                content, title=note["title"], count=12)
        except (llm.LLMConfigError, llm.LLMCallError) as e:
            raise HTTPException(400, f"卡片生成失败（计划仍会创建）：{e}")
    plan = plan_engine.create_plan(
        title=note["title"], content=content, source_type="note", source_id=note_id,
        daily_minutes=body.daily_minutes, flashcards=flashcards,
        schedule_type=body.schedule_type,
    )
    return plan


class StreamAiIn(BaseModel):
    action: str  # summarize | continue | polish | chat
    message: Optional[str] = None
    history: Optional[list[dict]] = None
    thinking: str = "standard"
    web: bool = False


@router.post("/{note_id}/ai/stream")
async def ai_stream(note_id: str, body: StreamAiIn):
    """SSE 流式 AI 输出（delta → done）。"""
    import json as _json

    from fastapi.responses import StreamingResponse
    from app.agent import websearch as _ws

    async def gen():
        try:
            # 联网场景：先检索，把结果作为消息上下文注入（检索阶段发一个 status 事件）
            web_ctx = None
            if body.web and body.action == "chat":
                yield f"data: {_json.dumps({'type': 'status', 'text': '🌐 联网检索中…'}, ensure_ascii=False)}\n\n"
                results = await _ws.asearch_web(body.message or "", max_results=4)
                parts = []
                sources = []
                for i, r in enumerate(results[:3], 1):
                    text = await _ws.afetch_page(r["url"]) or r["snippet"]
                    if text:
                        parts.append(f"[{i}] {r['title']}\n{r['url']}\n{text[:2000]}")
                        sources.append({"ref_id": i, "title": r["title"], "url": r["url"]})
                if parts:
                    web_ctx = "【联网检索结果】回答用 [n] 标注引用：\n" + "\n\n".join(parts)
                yield f"data: {_json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

            async for ev in notes_ai.stream_action(
                note_id, body.action, body.message, body.history,
                thinking=body.thinking, extra_context=web_ctx,
            ):
                yield f"data: {_json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"data: {_json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class PolishIn(BaseModel):
    count: int = 3


@router.post("/{note_id}/polish")
async def polish(note_id: str, body: PolishIn | None = None):
    """润色给出多个风格方案，供选择。"""
    try:
        variants = await notes_ai.polish_variants(note_id, (body.count if body else 3) or 3)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except (notes_ai.llm.LLMConfigError, notes_ai.llm.LLMCallError) as e:
        raise HTTPException(400, str(e))
    return {"variants": variants}


class ChatSessionIn(BaseModel):
    title: str = "新对话"


@router.get("/{note_id}/chats")
def chats(note_id: str):
    try:
        notes_service.read_note(note_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return notes_service.list_chats(note_id)


@router.post("/{note_id}/chats")
def new_chat(note_id: str, body: ChatSessionIn | None = None):
    try:
        notes_service.read_note(note_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return notes_service.create_chat(note_id, (body.title if body else "新对话") or "新对话")


@router.get("/{note_id}/chats/{cid}")
def get_chat(note_id: str, cid: str):
    try:
        return notes_service.read_chat(note_id, cid)
    except KeyError as e:
        raise HTTPException(404, str(e))


class ChatSaveIn(BaseModel):
    title: Optional[str] = None
    messages: list[dict]


@router.put("/{note_id}/chats/{cid}")
def put_chat(note_id: str, cid: str, body: ChatSaveIn):
    try:
        return notes_service.save_chat(note_id, cid, title=body.title, messages=body.messages)
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.delete("/{note_id}/chats/{cid}")
def del_chat(note_id: str, cid: str):
    notes_service.delete_chat(note_id, cid)
    return {"ok": True}


class QuizGen(BaseModel):
    count: int = 6


@router.post("/{note_id}/quiz")
async def gen_quiz(note_id: str, body: QuizGen | None = None):
    """生成自测题并存为该笔记的附属文件（不进笔记正文）。"""
    notes_ai.QUIZ_COUNT["count"] = (body.count if body else 6) or 6
    import asyncio

    try:
        r = await asyncio.to_thread(notes_ai.run_action, note_id, "quiz")
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except (notes_ai.llm.LLMConfigError, notes_ai.llm.LLMCallError) as e:
        raise HTTPException(400, str(e))
    name = notes_service.save_asset(note_id, "自测题", r["reply"])
    return {"name": name, "content": r["reply"]}


@router.get("/{note_id}/assets")
def assets(note_id: str):
    try:
        notes_service.read_note(note_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return notes_service.list_assets(note_id)


@router.get("/{note_id}/assets/{name}")
def asset_content(note_id: str, name: str):
    try:
        return {"name": name, "content": notes_service.read_asset(note_id, name)}
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.delete("/{note_id}/assets/{name}")
def asset_delete(note_id: str, name: str):
    notes_service.delete_asset(note_id, name)
    return {"ok": True}


class SnapshotIn(BaseModel):
    label: str = ""


@router.post("/{note_id}/snapshot")
def snapshot(note_id: str, body: SnapshotIn | None = None):
    """把当前内容存入历史版本（AI 覆盖等高风险操作前调用）。"""
    try:
        note = notes_service.read_note(note_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    notes_service.save_version(note_id, note["content"], label=(body.label if body else "") or "manual")
    return {"ok": True}


@router.get("/{note_id}/versions")
def versions(note_id: str):
    return notes_service.list_versions(note_id)


class RestoreIn(BaseModel):
    file: str


@router.post("/{note_id}/restore")
def restore(note_id: str, body: RestoreIn):
    try:
        return notes_service.restore_version(note_id, body.file)
    except KeyError as e:
        raise HTTPException(404, str(e))
