"""学屿 FastAPI 主应用：REST + WebSocket + 静态托管 UI。

- 开发：`uvicorn app.main:app --port 8765` 然后浏览器打开
- 客户端：`python main.py`（pywebview 壳自动选端口并开窗口）
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse

from app.api import notes, plan, research, shelf, system
from app.services import research_service, ws_hub
from app.storage.db import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("xueyu")

UI_DIR = Path(__file__).resolve().parent.parent / "ui"

PAGES = {
    "/": "index.html",
    "/plan": "plan.html",
    "/review": "review.html",
    "/notes": "notes.html",
    "/shelf": "shelf.html",
    "/doc": "doc.html",
    "/research": "research.html",
    "/report": "report.html",
    "/settings": "settings.html",
}


def create_app() -> FastAPI:
    init_db()
    research_service.interrupted_on_restart()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        _app.state.ticker = asyncio.create_task(ws_hub.review_ticker())
        start_reminder()
        yield
        ticker = getattr(_app.state, "ticker", None)
        if ticker:
            ticker.cancel()

    app = FastAPI(title="学屿 XueYu", version="0.1.0", docs_url=None, redoc_url=None, lifespan=lifespan)

    app.include_router(plan.router)
    app.include_router(plan.review_router)
    app.include_router(notes.router)
    app.include_router(shelf.router)
    app.include_router(shelf.docs_router)
    app.include_router(research.router)
    app.include_router(system.router)

    # ------------------------------------------------ WebSocket
    @app.websocket("/ws/review/{item_id}")
    async def ws_review(ws: WebSocket, item_id: int):
        await ws_hub.review_ws_handler(ws, item_id)

    @app.websocket("/ws/research/{task_id}")
    async def ws_research(ws: WebSocket, task_id: str):
        await research_service.subscribe(task_id, ws)

    # ------------------------------------------------ 页面路由
    @app.get("/{page:path}", include_in_schema=False)
    def page_or_static(page: str):
        key = "/" + page
        if key in PAGES:
            return FileResponse(UI_DIR / PAGES[key])
        target = UI_DIR / page
        if page and target.is_file():
            return FileResponse(target)
        if not page:
            return FileResponse(UI_DIR / "index.html")
        return FileResponse(UI_DIR / "404.html", status_code=404)

    # ------------------------------------------------ 静态资源不缓存（本地客户端更新立即可见）
    @app.middleware("http")
    async def no_cache_html(request: Request, call_next):
        resp = await call_next(request)
        path = request.url.path
        if not path.startswith("/api") and not path.startswith("/ws"):
            if path.endswith((".html", "/")) or "." not in path.rsplit("/", 1)[-1]:
                resp.headers["Cache-Control"] = "no-cache"
            elif path.endswith((".js", ".css")):
                resp.headers["Cache-Control"] = "no-cache"
        return resp

    # ------------------------------------------------ 全局异常 → 友好 JSON
    @app.exception_handler(Exception)
    async def on_error(request: Request, e: Exception):
        from app.services.llm import LLMCallError, LLMConfigError

        if isinstance(e, LLMConfigError):
            return JSONResponse(status_code=400, content={"detail": str(e)})
        if isinstance(e, LLMCallError):
            return JSONResponse(status_code=400, content={"detail": str(e)})
        if isinstance(e, KeyError):
            return JSONResponse(status_code=404, content={"detail": str(e).strip("'")})
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": f"服务器内部错误：{e}"})

    return app


# ---------------------------------------------------------------- 复习提醒（APScheduler）

_scheduler = None


def start_reminder() -> None:
    """每天定时提醒今日待复习（P1，可在设置关闭）。"""
    global _scheduler
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        if _scheduler is None:
            _scheduler = AsyncIOScheduler()
            _scheduler.start()
        reschedule_reminder()
    except Exception as e:  # noqa: BLE001
        log.warning("reminder scheduler failed: %s", e)


def reschedule_reminder() -> None:
    if _scheduler is None:
        return
    from app.services import settings_store

    cfg = settings_store.get("reminder")
    job = _scheduler.get_job("review-reminder")
    if not cfg.get("enabled"):
        if job:
            job.remove()
        return
    hh, mm = cfg.get("time", "09:00").split(":")
    if job:
        job.reschedule(trigger="cron", hour=int(hh), minute=int(mm))
    else:
        _scheduler.add_job(
            _send_reminder, "cron", id="review-reminder",
            hour=int(hh), minute=int(mm),
        )


async def _send_reminder() -> None:
    from app.config import APP_NAME
    from app.services import notify, plan_engine

    data = plan_engine.today_queue()
    pending = data["total"] - data["done"]
    if pending > 0:
        notify(APP_NAME, f"今日还有 {pending} 项复习待完成，坚持一下 💪")


app = create_app()
