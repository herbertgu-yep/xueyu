"""WebSocket 中枢：复习计时每秒广播 + 连接管理。

- /ws/review/{item_id}：服务端每秒推 {type:"tick", item, elapsed_sec, run_state}
- /ws/research/{task_id} 的订阅逻辑在 research_service 里（活动日志/进度推送）
"""
from __future__ import annotations

import asyncio

from fastapi import WebSocket

from app.services import plan_engine, research_service

_review_clients: dict[int, set[WebSocket]] = {}


def add_review_client(item_id: int, ws: WebSocket) -> None:
    _review_clients.setdefault(item_id, set()).add(ws)


def remove_review_client(item_id: int, ws: WebSocket) -> None:
    _review_clients.get(item_id, set()).discard(ws)


async def review_ws_handler(ws: WebSocket, item_id: int) -> None:
    await ws.accept()
    add_review_client(item_id, ws)
    try:
        # 立即推一次快照
        await _push_item(ws, item_id)
        while True:
            await ws.receive_text()
    except Exception:
        pass
    finally:
        remove_review_client(item_id, ws)


async def _push_item(ws: WebSocket, item_id: int) -> None:
    import json

    try:
        item = plan_engine.get_item(item_id, with_content=False)
        await ws.send_text(json.dumps({
            "type": "tick",
            "item_id": item_id,
            "elapsed_sec": item["elapsed_sec"],
            "run_state": item["run_state"],
        }, ensure_ascii=False))
    except KeyError:
        pass


async def review_ticker() -> None:
    """每秒向订阅的窗口广播运行中/暂停项的计时（服务端权威时钟）。"""
    while True:
        try:
            import json

            items = plan_engine.list_items()
            for it in items:
                if it["run_state"] not in ("running", "paused"):
                    continue
                conns = list(_review_clients.get(it["id"], set()))
                if not conns:
                    continue
                payload = json.dumps({
                    "type": "tick", "item_id": it["id"],
                    "elapsed_sec": it["elapsed_sec"], "run_state": it["run_state"],
                }, ensure_ascii=False)
                for ws in conns:
                    try:
                        await ws.send_text(payload)
                    except Exception:
                        remove_review_client(it["id"], ws)
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(1.0)


research_ws_handler = research_service.subscribe  # /ws/research/{task_id}
