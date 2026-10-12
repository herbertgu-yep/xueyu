"""书架 + 文档 API。"""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from sqlmodel import Session

from app.storage.db import engine
from app.storage.models import Document
from app.services import doc_analysis, rag_service, shelf_service

router = APIRouter(prefix="/api/shelf", tags=["shelf"])
docs_router = APIRouter(prefix="/api/docs", tags=["docs"])


@router.get("")
def shelf(fmt: Optional[str] = None, analyzed: Optional[bool] = None,
          category_id: Optional[int] = None):
    return {
        "categories": shelf_service.list_categories(),
        "documents": shelf_service.list_documents(fmt=fmt, analyzed=analyzed, category_id=category_id),
        "index": rag_service.status(),
    }


@router.get("/index/status")
def index_status():
    return rag_service.status()


@router.post("/reindex")
def reindex():
    return rag_service.reindex_all()


class CategoryIn(BaseModel):
    name: str
    icon: str = "📂"


@router.post("/category")
def add_category(body: CategoryIn):
    return shelf_service.create_category(body.name, body.icon)


class CategoryUpdate(BaseModel):
    name: Optional[str] = None
    icon: Optional[str] = None


@router.put("/category/{cat_id}")
def update_category(cat_id: int, body: CategoryUpdate):
    try:
        return shelf_service.rename_category(cat_id, name=body.name, icon=body.icon)
    except KeyError as e:
        raise HTTPException(404, str(e))


class ReorderIn(BaseModel):
    ids: list[int]


@router.post("/category/reorder")
def reorder_categories(body: ReorderIn):
    shelf_service.reorder_categories(body.ids)
    return {"ok": True}


@router.delete("/category/{cat_id}")
def del_category(cat_id: int):
    shelf_service.delete_category(cat_id)
    return {"ok": True}


@router.post("/upload")
async def upload(files: list[UploadFile] = File(...), category_id: Optional[int] = None,
                 title: Optional[str] = None):
    """多文件上传。逐个处理；单个失败不影响其它，错误汇总返回。

    解析 + 向量化是重 IO/CPU 操作，放线程池执行避免阻塞事件循环。
    """
    results, errors = [], []
    for f in files:
        try:
            data = await f.read()
            doc = await asyncio.to_thread(
                shelf_service.save_upload, f.filename or "unnamed", data,
                category_id=category_id, title=title)
            results.append(doc)
        except ValueError as e:
            errors.append({"file": f.filename, "error": str(e)})
        except Exception as e:  # noqa: BLE001
            errors.append({"file": f.filename, "error": f"处理失败：{e}"})
    return {"documents": results, "errors": errors}


# ---------------------------------------------------------------- 单文档

@docs_router.get("")
def all_docs():
    return shelf_service.list_documents()


@docs_router.get("/{doc_id}")
def get_doc(doc_id: int, with_text: bool = False):
    try:
        return shelf_service.get_document(doc_id, with_text=with_text)
    except KeyError as e:
        raise HTTPException(404, str(e))


@docs_router.patch("/{doc_id}")
def update_doc(doc_id: int, body: dict):
    kwargs = {}
    if "category_id" in body:
        kwargs["category_id"] = body.get("category_id")
    if "title" in body:
        kwargs["title"] = body.get("title")
    try:
        return shelf_service.update_document(doc_id, **kwargs)
    except KeyError as e:
        raise HTTPException(404, str(e))


@docs_router.delete("/{doc_id}")
def del_doc(doc_id: int):
    try:
        shelf_service.delete_document(doc_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return {"ok": True}


@docs_router.get("/{doc_id}/file")
def doc_file(doc_id: int):
    """在浏览器/客户端内打开或下载原始文件。"""
    from pathlib import Path

    from app.config import DATA_DIR

    doc = shelf_service.get_document(doc_id)
    path = DATA_DIR / doc["path"]
    if not Path(path).exists():
        raise HTTPException(404, "文件已丢失")
    return FileResponse(path, filename=doc["filename"])


@docs_router.post("/{doc_id}/reindex")
def reindex_doc(doc_id: int):
    try:
        return rag_service.index_document(doc_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


class AnalyzeIn(BaseModel):
    kind: str = "all"  # summary | mindmap | quiz | all


@docs_router.post("/{doc_id}/analyze")
async def analyze(doc_id: int, body: AnalyzeIn):
    try:
        result = await asyncio.to_thread(doc_analysis.analyze, doc_id, body.kind)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    if result.get("error"):
        raise HTTPException(400, result["error"])
    return result["doc"]


class AskIn(BaseModel):
    question: str
    history: Optional[list[dict]] = None


@docs_router.post("/{doc_id}/ask")
async def ask_doc(doc_id: int, body: AskIn):
    if not body.question.strip():
        raise HTTPException(400, "问题不能为空")
    try:
        return await asyncio.to_thread(doc_analysis.ask, doc_id, body.question, body.history)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except (doc_analysis.llm.LLMConfigError, doc_analysis.llm.LLMCallError) as e:
        raise HTTPException(400, str(e))


class DocPlanIn(BaseModel):
    daily_minutes: int = 30
    generate_flashcards: bool = True
    schedule_type: str = "normal"


@docs_router.post("/{doc_id}/plan")
async def to_plan(doc_id: int, body: DocPlanIn):
    """加入复习计划：素材取 AI 摘要（若有）或原文节选。"""
    from app.services import llm, notes_ai, plan_engine

    doc = shelf_service.get_document(doc_id, with_text=True)
    content = doc.get("summary_md") or (doc.get("text") or "")[:8000]
    if not content.strip():
        raise HTTPException(400, "该文档没有可用文本")
    flashcards = None
    if body.generate_flashcards:
        try:
            flashcards = await notes_ai.generate_flashcards_async(
                content, title=doc["title"], count=12)
        except (llm.LLMConfigError, llm.LLMCallError) as e:
            raise HTTPException(400, f"卡片生成失败（计划仍会创建）：{e}")
    return plan_engine.create_plan(
        title=doc["title"], content=content, source_type="document",
        source_id=str(doc_id), daily_minutes=body.daily_minutes, flashcards=flashcards,
        schedule_type=body.schedule_type)


@docs_router.get("/{doc_id}/preview")
def preview(doc_id: int):
    """原文预览：md/txt 返回原文（前端渲染），docx/pptx 返回服务端转的 HTML，pdf 前端走 PDF.js。"""
    doc = shelf_service.get_document(doc_id)
    if doc["fmt"] in ("md", "txt"):
        full = shelf_service.get_document(doc_id, with_text=True)
        return {"kind": "md", "text": full.get("text") or ""}
    if doc["fmt"] in ("docx", "pptx"):
        with Session(engine) as s:
            d = s.get(Document, doc_id)
            return {"kind": "html", "html": shelf_service.build_preview_html(d)}
    return {"kind": doc["fmt"]}


@docs_router.post("/{doc_id}/reveal")
def reveal(doc_id: int):
    """在访达中显示。"""
    import subprocess

    from app.config import DATA_DIR

    doc = shelf_service.get_document(doc_id)
    path = (DATA_DIR / doc["path"]).resolve()
    if not path.exists():
        raise HTTPException(404, "文件已丢失")
    subprocess.Popen(["open", "-R", str(path)])
    return {"ok": True}
