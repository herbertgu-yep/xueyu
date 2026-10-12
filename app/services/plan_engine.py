"""计划引擎：艾宾浩斯排期 + SM-2 变体 + 服务端权威计时。

- 每条 ReviewItem 即一条学习计划：学习日（新学）→ D1/D2/D4/D7/D15/D30 六个复习点 → 毕业
- 计时以服务端为唯一权威（running_since + accumulated_sec），暂停时间不计入
- 完成复习按正确率换算 SM-2 质量 q，更新 EF 与下一次间隔（艾宾浩斯阶梯 × EF 缩放）
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Optional

from sqlmodel import Session, select

from app.storage.db import engine
from app.storage.models import ReviewItem, ReviewLog

POINT_LABELS = ["D1", "D2", "D4", "D7", "D15", "D30"]

# 排期档位：间隔天数表。temp 无间隔表（完成即毕业）
SCHEDULE_TYPES = {
    "temp":  {"label": "临时计划", "intervals": []},
    "normal": {"label": "艾宾浩斯 · 正常", "intervals": [1, 2, 4, 7, 15, 30]},
    "short": {"label": "艾宾浩斯 · 短周期", "intervals": [1, 2, 3, 5, 7, 10]},
    "long":  {"label": "艾宾浩斯 · 长周期", "intervals": [1, 3, 7, 21, 60, 120]},
}


def intervals_for(item: ReviewItem) -> list[int]:
    return SCHEDULE_TYPES.get(getattr(item, "schedule_type", "normal") or "normal", SCHEDULE_TYPES["normal"])["intervals"]


def _today() -> date:
    return datetime.now().astimezone().date()


def parse_date(s: str) -> date:
    return date.fromisoformat(s[:10])


# ---------------------------------------------------------------- 创建计划

def create_plan(
    *,
    title: str,
    content: str = "",
    source_type: str = "manual",  # note | document | manual | report
    source_id: Optional[str] = None,
    daily_minutes: int = 30,
    flashcards: Optional[list[dict]] = None,
    start_today: bool = True,
    schedule_type: str = "normal",
) -> dict:
    with Session(engine) as s:
        item = ReviewItem(
            title=title.strip() or "未命名计划",
            source_type=source_type,
            source_id=source_id,
            content=content,
            daily_minutes=max(5, min(480, int(daily_minutes or 30))),
            schedule_type=schedule_type if schedule_type in SCHEDULE_TYPES else "normal",
            due_date=_today().isoformat() if start_today else (_today() + timedelta(days=1)).isoformat(),
        )
        if flashcards:
            from app.storage.models import put_json

            item.flashcards_json = put_json(flashcards)
        s.add(item)
        s.commit()
        s.refresh(item)
        return serialize(item)


def serialize(item: ReviewItem, *, with_content: bool = False) -> dict:
    elapsed = elapsed_sec(item)
    stage = current_stage(item)
    ivals = intervals_for(item)
    flashcards = []
    if item.flashcards_json:
        try:
            import json

            flashcards = json.loads(item.flashcards_json)
        except Exception:
            flashcards = []
    out = {
        "id": item.id,
        "title": item.title,
        "source_type": item.source_type,
        "source_id": item.source_id,
        "daily_minutes": item.daily_minutes,
        "created_at": item.created_at,
        "reps": item.reps,
        "stage": stage,  # 0=学习日待完成, 1..6=第 n 个复习点, 7=已毕业
        "stage_label": stage_label(item),
        "easiness_factor": round(item.easiness_factor, 2),
        "interval_days": item.interval_days,
        "due_date": item.due_date,
        "due_in_days": (parse_date(item.due_date) - _today()).days if item.due_date else None,
        "last_reviewed_at": item.last_reviewed_at,
        "last_accuracy": item.last_accuracy,
        "run_state": item.run_state,
        "elapsed_sec": elapsed,
        "total_time_sec": item.total_time_sec,
        "status": item.status,
        "points_done": max(item.reps - 1, 0) if item.status != "graduated" else len(ivals),
        "flashcards": flashcards if with_content else None,
        "flashcard_count": len(flashcards),
        "has_content": bool(item.content),
        "schedule": projected_schedule(item),
        "schedule_type": getattr(item, "schedule_type", "normal") or "normal",
        "schedule_label": SCHEDULE_TYPES.get(getattr(item, "schedule_type", "normal") or "normal", SCHEDULE_TYPES["normal"])["label"],
        "total_points": len(ivals),
    }
    if with_content:
        out["content"] = item.content
    return out


def current_stage(item: ReviewItem) -> int:
    """0=学习日(新学)待完成；1..n=待第 n 个复习点；n+1=毕业。"""
    ivals = intervals_for(item)
    if item.status == "graduated":
        return len(ivals) + 1
    return min(item.reps, len(ivals))


def stage_label(item: ReviewItem) -> str:
    ivals = intervals_for(item)
    if item.status == "graduated":
        return "已毕业"
    if item.reps == 0:
        return "临时" if not ivals else "新学"
    return f"D{ivals[item.reps - 1]}" if item.reps <= len(ivals) else "复习点"


def projected_schedule(item: ReviewItem) -> list[dict]:
    """外推剩余复习点落点：学习日后第 1/2/4/7/15/30 天。

    - reps=0（尚未学习）：D_k = 学习日(due_date) + INTERVALS[k-1]
    - reps>0：下一点 = due_date（真实动态排期）；更远的点从 due_date 按剩余偏移差推算
    """
    if item.status == "graduated" or not item.due_date:
        return []
    d0 = parse_date(item.due_date)
    reps = item.reps
    ivals = intervals_for(item)
    if not ivals:  # 临时计划：仅学习日
        return [{"point": "学习日", "index": 1, "date": d0.isoformat(), "is_current": True}]
    offset0 = 0 if reps == 0 else ivals[reps - 1]
    out = []
    for k in range(reps + 1, len(ivals) + 1):
        if reps == 0:
            date = d0 + timedelta(days=intervals_for(item)[k - 1])
        else:
            date = d0 if k == reps + 1 else d0 + timedelta(days=max(1, intervals_for(item)[k - 1] - offset0))
        out.append({
            "point": POINT_LABELS[k - 1],
            "index": k,
            "date": date.isoformat(),
            "is_current": k == reps + 1,
        })
    return out


# ---------------------------------------------------------------- 计时状态机

def elapsed_sec(item: ReviewItem) -> int:
    acc = item.accumulated_sec
    if item.run_state == "running" and item.running_since:
        started = datetime.fromisoformat(item.running_since)
        acc += max(0, int((datetime.now().astimezone() - started).total_seconds()))
    return acc


def start_review(item_id: int) -> dict:
    with Session(engine) as s:
        item = s.get(ReviewItem, item_id)
        if not item:
            raise KeyError("复习项不存在")
        if item.status == "graduated":
            raise ValueError("该计划已毕业")
        if item.due_date and parse_date(item.due_date) > _today():
            raise ValueError("还没到复习日期，可在计划页查看排期")
        # 单焦点：同时只允许一项在计时，其余自动暂停
        running = s.exec(select(ReviewItem).where(ReviewItem.run_state == "running")).all()
        for other in running:
            if other.id != item_id:
                _fold_pause(s, other)
        if item.run_state == "running":
            pass
        elif item.run_state == "paused":
            item.running_since = datetime.now().astimezone().isoformat(timespec="seconds")
        else:
            item.accumulated_sec = 0
            item.running_since = datetime.now().astimezone().isoformat(timespec="seconds")
        item.run_state = "running"
        s.add(item)
        s.commit()
        s.refresh(item)
        return serialize(item)


def pause_review(item_id: int) -> dict:
    with Session(engine) as s:
        item = s.get(ReviewItem, item_id)
        if not item:
            raise KeyError("复习项不存在")
        _fold_pause(s, item)
        s.add(item)
        s.commit()
        s.refresh(item)
        return serialize(item)


def _fold_pause(s: Session, item: ReviewItem) -> None:
    """把 running 的时间折进 accumulated，冻结计时。"""
    if item.run_state == "running" and item.running_since:
        started = datetime.fromisoformat(item.running_since)
        item.accumulated_sec += max(0, int((datetime.now().astimezone() - started).total_seconds()))
    item.running_since = None
    item.run_state = "paused"


def finish_review(
    item_id: int,
    *,
    known: int = 0,
    unknown: int = 0,
    accuracy: Optional[float] = None,
) -> dict:
    """完成复习：结算计时 + 写日志 + SM-2 更新 + 推进复习点。"""
    with Session(engine) as s:
        item = s.get(ReviewItem, item_id)
        if not item:
            raise KeyError("复习项不存在")
        # 结算计时
        if item.run_state == "running" and item.running_since:
            started = datetime.fromisoformat(item.running_since)
            item.accumulated_sec += max(0, int((datetime.now().astimezone() - started).total_seconds()))
        duration = item.accumulated_sec
        item.running_since = None
        item.run_state = "idle"
        item.accumulated_sec = 0
        item.total_time_sec += duration

        # 正确率 → SM-2 质量 q
        if accuracy is None and (known or unknown):
            total = known + unknown
            accuracy = known / total if total else None
        q = quality_from_accuracy(accuracy)
        item.last_accuracy = accuracy
        item.last_quality = q

        # SM-2 变体：EF 更新 + 间隔 = 档位基准 × EF 缩放
        old_reps = item.reps
        ivals = intervals_for(item)
        if q >= 3:
            item.reps = old_reps + 1
            ef = item.easiness_factor + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
            item.easiness_factor = max(1.3, round(ef, 3))
            if not ivals or item.reps >= len(ivals):
                # 临时计划 / 全部复习点完成 → 毕业
                item.status = "graduated"
                item.due_date = ""
                item.interval_days = ivals[-1] if ivals else 0
            else:
                # 下一段间隔 = 档位表相邻差值 × EF 缩放
                base = ivals[item.reps - 1] - (ivals[item.reps - 2] if item.reps >= 2 else 0)
                scaled = max(1, round(base * item.easiness_factor / 2.5))
                item.interval_days = scaled
                item.due_date = (_today() + timedelta(days=scaled)).isoformat()
        else:
            # 未通过：明天重做当前复习点，间隔不推进
            item.interval_days = 1
            item.due_date = (_today() + timedelta(days=1)).isoformat()
        item.last_reviewed_at = datetime.now().astimezone().isoformat(timespec="seconds")

        log = ReviewLog(
            item_id=item.id,
            duration_sec=duration,
            accuracy=accuracy,
            quality=q,
            known=known,
            unknown=unknown,
        )
        s.add(log)
        s.add(item)
        s.commit()
        s.refresh(item)
        out = serialize(item)
        out["last_session"] = {
            "duration_sec": duration,
            "accuracy": accuracy,
            "quality": q,
            "graduated": item.status == "graduated",
            "next": None if item.status == "graduated" else item.due_date,
        }
        return out


def quality_from_accuracy(accuracy: Optional[float]) -> int:
    """正确率 → SM-2 0-5。无卡片素材时按自评默认 4。"""
    if accuracy is None:
        return 4
    a = max(0.0, min(1.0, accuracy))
    if a >= 0.95:
        return 5
    if a >= 0.85:
        return 4
    if a >= 0.7:
        return 3
    if a >= 0.5:
        return 2
    if a >= 0.3:
        return 1
    return 0


def reset_timer(item_id: int) -> dict:
    """放弃本次计时（清零未结算秒数）。"""
    with Session(engine) as s:
        item = s.get(ReviewItem, item_id)
        if not item:
            raise KeyError("复习项不存在")
        item.accumulated_sec = 0
        item.running_since = None
        item.run_state = "idle"
        s.add(item)
        s.commit()
        s.refresh(item)
        return serialize(item)


# ---------------------------------------------------------------- 队列与统计

def today_queue(*, with_content: bool = False) -> dict:
    today = _today().isoformat()
    with Session(engine) as s:
        items = s.exec(
            select(ReviewItem)
            .where(ReviewItem.status == "active")
            .where(ReviewItem.due_date <= today)
        ).all()
        running_first = sorted(
            items,
            key=lambda it: ({"running": 0, "paused": 1}.get(it.run_state, 2), it.due_date, it.id),
        )
        logs_today = s.exec(select(ReviewLog).where(ReviewLog.reviewed_date == today)).all()
        done_item_ids = {l.item_id for l in logs_today}
        queue = [serialize(it, with_content=with_content) for it in running_first]
        for it, raw in zip(queue, running_first):
            it["done_today"] = raw.id in done_item_ids
        focus_today = sum(l.duration_sec for l in logs_today)
        focus_today += sum(elapsed_sec(it) for it in items if it.run_state == "running" and it.id not in done_item_ids)
        return {
            "date": today,
            "queue": queue,
            "total": len(queue),
            "done": sum(1 for it in running_first if it.id in done_item_ids),
            "focus_today_sec": focus_today,
        }


def items_due_on(items: list[ReviewItem], ds: str, today: date) -> list[ReviewItem]:
    """某日期应复习的项目集合 —— day/week/month 三视图共用的唯一事实源。

    - 今天：到期日 <= 今天的全部（含逾期未完成，自动顺延）
    - 未来：到期日精确命中 + 外推落点命中
    - 过去：到期日 <= 该日的（供「当天没完成」展示）
    """
    if ds >= today.isoformat():
        out = [it for it in items if it.due_date and it.due_date <= ds] if ds == today.isoformat() \
            else [it for it in items if it.due_date == ds]
        if ds > today.isoformat():
            for it in items:
                if it.due_date == ds:
                    continue
                for p in projected_schedule(it):
                    if p["date"] == ds:
                        out.append(it)
                        break
        return out
    return [it for it in items if it.due_date and it.due_date <= ds]


def day_queue(date_str: Optional[str] = None, *, with_content: bool = False) -> dict:
    """任意一天的复习安排：今天含逾期未做；未来显示计划落点；过去显示当天完成记录。"""
    d = parse_date(date_str) if date_str else _today()
    ds = d.isoformat()
    today = _today()
    with Session(engine) as s:
        items = s.exec(select(ReviewItem).where(ReviewItem.status == "active")).all()
        logs = s.exec(select(ReviewLog).where(ReviewLog.reviewed_date == ds)).all()
        done_ids = {l.item_id for l in logs}
        future_points: dict[int, dict] = {}
        if d > today:
            for it in items:
                if it.due_date == ds:
                    continue
                for p in projected_schedule(it):
                    if p["date"] == ds:
                        future_points[it.id] = p
                        break
        due = items_due_on(items, ds, today)
        running_first = sorted(
            due, key=lambda it: ({"running": 0, "paused": 1}.get(it.run_state, 2), it.due_date, it.id)
            if d == today else (it.due_date, it.id))
        queue = [serialize(it, with_content=with_content) for it in running_first]
        for it, raw in zip(queue, running_first):
            it["done_today"] = raw.id in done_ids
            p = future_points.get(raw.id)
            it["future_point"] = p["point"] if p else ""
            it["future_index"] = p["index"] if p else None
        log_rows = []
        logged_ids = set()
        for l in logs:
            logged_ids.add(l.item_id)
            it = next((x for x in items if x.id == l.item_id), None)
            log_rows.append({
                "item_id": l.item_id,
                "title": it.title if it else "（已删除的计划）",
                "duration_sec": l.duration_sec,
                "accuracy": l.accuracy,
                "known": l.known,
                "unknown": l.unknown,
            })
        # 未完成（已自动顺延）：当天到期但当天没有完成记录、且现在仍未毕业的计划
        undone = []
        if d < today:
            for it in items:
                if it.id in logged_ids or it.status != "active":
                    continue
                if it.due_date and parse_date(it.due_date) <= d:
                    undone.append({"item_id": it.id, "title": it.title,
                                   "stage_label": stage_label(it), "due_date": it.due_date})
        focus = sum(l.duration_sec for l in logs)
        if d == today:
            focus += sum(elapsed_sec(it) for it in due if it.run_state == "running" and it.id not in done_ids)
        return {
            "date": ds,
            "is_today": d == today,
            "is_future": d > today,
            "is_past": d < today,
            "queue": queue,
            "completed": log_rows,
            "undone": undone,
            "total": len(queue),
            "done": sum(1 for it in running_first if it.id in done_ids),
            "focus_sec": focus,
            "weekday": "一二三四五六日"[d.weekday()],
        }


def list_items(*, include_archived: bool = False) -> list[dict]:
    with Session(engine) as s:
        q = select(ReviewItem).order_by(ReviewItem.created_at.desc())
        items = s.exec(q).all()
        return [serialize(it) for it in items if include_archived or it.status != "archived"]


def get_item(item_id: int, *, with_content: bool = True) -> dict:
    with Session(engine) as s:
        item = s.get(ReviewItem, item_id)
        if not item:
            raise KeyError("复习项不存在")
        return serialize(item, with_content=with_content)


def upcoming_points(limit: int = 8) -> list[dict]:
    """跨计划聚合「接下来的复习点」（当前待完成点 + 未来外推点）。"""
    today = _today()
    rows: list[dict] = []
    with Session(engine) as s:
        items = s.exec(select(ReviewItem).where(ReviewItem.status == "active")).all()
        for it in items:
            sched = projected_schedule(it)
            for p in sched:
                if parse_date(p["date"]) >= today:
                    rows.append({
                        "date": p["date"],
                        "item_id": it.id,
                        "title": it.title,
                        "point": p["point"],
                        "point_index": p["index"],
                        "points_done": max(it.reps - 1, 0),
                        "last_accuracy": it.last_accuracy,
                    })
                    break
    rows.sort(key=lambda r: (r["date"], r["item_id"]))
    return rows[:limit]


def week_view(anchor: Optional[str] = None) -> dict:
    """周视图：7 天各自的 复习任务数/完成数 与当日统计。"""
    start = parse_date(anchor) if anchor else _today() - timedelta(days=_today().weekday())
    start = start - timedelta(days=start.weekday())
    days: list[dict] = []
    today = _today()
    with Session(engine) as s:
        items = s.exec(select(ReviewItem).where(ReviewItem.status == "active")).all()
        logs = s.exec(select(ReviewLog)).all()
        for i in range(7):
            d = start + timedelta(days=i)
            ds = d.isoformat()
            due = len(items_due_on(items, ds, today))
            done = sum(1 for l in logs if l.reviewed_date == ds)
            secs = sum(l.duration_sec for l in logs if l.reviewed_date == ds)
            days.append({
                "date": ds,
                "weekday": "一二三四五六日"[d.weekday()],
                "day": d.day,
                "is_today": d == _today(),
                "future": d > _today(),
                "due": due,
                "done": done,
                "seconds": secs,
            })
        accs = [l.accuracy for l in logs if l.accuracy is not None and l.accuracy >= 0]
        return {
            "start": start.isoformat(),
            "days": days,
            "week_done": sum(d["done"] for d in days),
            "week_seconds": sum(d["seconds"] for d in days),
            "week_accuracy": round(sum(accs) / len(accs), 3) if accs else None,
        }


def month_view(year: int, month: int) -> dict:
    """月视图：按复习点落点显示任务圆点（含未来外推）。"""
    first = date(year, month, 1)
    last = (first + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    per_day: dict[str, dict] = {}
    with Session(engine) as s:
        items = s.exec(select(ReviewItem).where(ReviewItem.status == "active")).all()
        logs = s.exec(select(ReviewLog).where(ReviewLog.reviewed_date >= first.isoformat(),
                                               ReviewLog.reviewed_date <= last.isoformat())).all()
        for i in range((last - first).days + 1):
            ds = (first + timedelta(days=i)).isoformat()
            n = len(items_due_on(items, ds, _today()))
            if n:
                slot = per_day.setdefault(ds, {"due": 0, "done": 0})
                slot["due"] += n
        for l in logs:
            slot = per_day.setdefault(l.reviewed_date, {"due": 0, "done": 0})
            slot["done"] += 1
    return {
        "year": year, "month": month,
        "days": [
            {"date": (first + timedelta(days=i)).isoformat(),
             "day": (first + timedelta(days=i)).day,
             "is_today": (first + timedelta(days=i)) == _today(),
             **per_day.get((first + timedelta(days=i)).isoformat(), {"due": 0, "done": 0})}
            for i in range((last - first).days + 1)
        ],
    }


def stats() -> dict:
    """首页统计：保持率 / 本周专注 / 连续打卡 / 今日概况。"""
    today = _today()
    with Session(engine) as s:
        items = s.exec(select(ReviewItem).where(ReviewItem.status != "archived")).all()
        logs = s.exec(select(ReviewLog)).all()

        # 记忆保持率：R = exp(-Δt / S)，S = 基准间隔 × (EF/2.5) × 1.5
        retentions = []
        for it in items:
            if it.status == "graduated" or not it.last_reviewed_at or not it.interval_days:
                continue
            dt = (today - parse_date(it.last_reviewed_at)).days
            if dt < 0:
                continue
            strength = max(0.5, it.interval_days * (it.easiness_factor / 2.5) * 1.5)
            retentions.append(math.exp(-dt / strength))
        retention = round(sum(retentions) / len(retentions) * 100) if retentions else None

        week_start = today - timedelta(days=today.weekday())
        week_logs = [l for l in logs if parse_date(l.reviewed_date) >= week_start]
        prev_start = week_start - timedelta(days=7)
        prev_logs = [l for l in logs if prev_start <= parse_date(l.reviewed_date) < week_start]
        week_sec = sum(l.duration_sec for l in week_logs)
        prev_sec = sum(l.duration_sec for l in prev_logs)

        # 连续打卡
        days_with = {l.reviewed_date for l in logs}
        streak = 0
        d = today if today.isoformat() in days_with else today - timedelta(days=1)
        while d.isoformat() in days_with:
            streak += 1
            d -= timedelta(days=1)

        accs = [l.accuracy for l in logs if l.accuracy is not None]
        today_queue_data = today_queue()
        return {
            "retention": retention,
            "week_focus_sec": week_sec,
            "week_focus_delta": (week_sec - prev_sec) / prev_sec if prev_sec else None,
            "streak": streak,
            "avg_accuracy": round(sum(accs) / len(accs) * 100) if accs else None,
            "active_plans": sum(1 for it in items if it.status == "active"),
            "graduated": sum(1 for it in items if it.status == "graduated"),
            "today_total": today_queue_data["total"],
            "today_done": today_queue_data["done"],
            "focus_today_sec": today_queue_data["focus_today_sec"],
        }


def plan_schedule(item_id: int) -> dict:
    """单个计划的完整时间表：学习日 + 六个复习点（已完成带成绩，未完成外推日期）。"""
    with Session(engine) as s:
        item = s.get(ReviewItem, item_id)
        if not item:
            raise KeyError("复习项不存在")
        logs = s.exec(
            select(ReviewLog).where(ReviewLog.item_id == item_id).order_by(ReviewLog.created_at)
        ).all()
        proj = {p["index"]: p for p in projected_schedule(item)}
        try:
            created_d = parse_date(item.created_at)
        except Exception:
            created_d = _today()

        def log_at(i):  # 第 i 个复习点（1-based）的日志
            return logs[i - 1] if 0 < i <= len(logs) else None

        points = [{
            "index": 0, "label": "学习日",
            "date": (log_at(1).reviewed_date if item.reps >= 1 and len(logs) >= 1 else created_d.isoformat()),
            "status": "done" if item.reps >= 1 else "pending",
            "accuracy": log_at(1).accuracy if item.reps >= 1 and len(logs) >= 1 else None,
            "duration_sec": log_at(1).duration_sec if item.reps >= 1 and len(logs) >= 1 else None,
        }]
        for i in range(1, 7):
            lg = log_at(i)
            if lg is not None:
                points.append({"index": i, "label": POINT_LABELS[i - 1], "date": lg.reviewed_date,
                               "status": "done", "accuracy": lg.accuracy,
                               "duration_sec": lg.duration_sec})
            elif item.status == "graduated":
                points.append({"index": i, "label": POINT_LABELS[i - 1], "date": "",
                               "status": "skipped", "accuracy": None, "duration_sec": None})
            else:
                p = proj.get(i)
                points.append({"index": i, "label": POINT_LABELS[i - 1],
                               "date": p["date"] if p else "", "status": "pending",
                               "accuracy": None, "duration_sec": None})
        return {"item": serialize(item), "points": points,
                "graduated": item.status == "graduated"}


def apply_leave(days: int, *, start: Optional[str] = None) -> dict:
    """请假：所有未毕业计划的到期日顺延 N 天（从 start 起算的天数直接整体平移）。"""
    n = max(1, min(90, int(days)))
    shifted = 0
    with Session(engine) as s:
        items = s.exec(select(ReviewItem).where(ReviewItem.status == "active")).all()
        for it in items:
            if it.due_date:
                it.due_date = (parse_date(it.due_date) + timedelta(days=n)).isoformat()
                s.add(it)
                shifted += 1
        s.commit()
    # 留痕（设置页不展示，仅记录最近一次）
    try:
        from app.services import settings_store

        leaves = settings_store.get("leaves") or {}
        items = leaves.get("items") or []
        items.append({"date": _today().isoformat(), "days": n})
        settings_store.put("leaves", {"items": items[-20:]})
    except Exception:
        pass
    return {"shifted": shifted, "days": n}


def delete_item(item_id: int) -> None:
    with Session(engine) as s:
        item = s.get(ReviewItem, item_id)
        if item:
            item.status = "archived"
            item.run_state = "idle"
            item.running_since = None
            s.add(item)
            s.commit()


def update_item(item_id: int, *, title: Optional[str] = None, daily_minutes: Optional[int] = None) -> dict:
    with Session(engine) as s:
        item = s.get(ReviewItem, item_id)
        if not item:
            raise KeyError("复习项不存在")
        if title is not None:
            item.title = title.strip() or item.title
        if daily_minutes is not None:
            item.daily_minutes = max(5, min(480, int(daily_minutes)))
        s.add(item)
        s.commit()
        s.refresh(item)
        return serialize(item)


def set_flashcards(item_id: int, flashcards: list[dict]) -> dict:
    from app.storage.models import put_json

    with Session(engine) as s:
        item = s.get(ReviewItem, item_id)
        if not item:
            raise KeyError("复习项不存在")
        item.flashcards_json = put_json(flashcards)
        s.add(item)
        s.commit()
        s.refresh(item)
        return serialize(item, with_content=True)
