"""SQLModel 数据模型：全部持久化实体的表定义。"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlmodel import Column, Field, SQLModel, Text


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def today_str() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d")


class ReviewItem(SQLModel, table=True):
    """复习项（= 一条艾宾浩斯学习计划），SM-2 字段对齐 Anki 惯例。"""

    __tablename__ = "review_item"

    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(index=True)
    source_type: str = Field(default="manual")  # note | document | manual | report
    source_id: Optional[str] = Field(default=None, index=True)
    content: str = Field(default="", sa_column=Column(Text))  # 复习素材（Markdown）
    flashcards_json: Optional[str] = Field(default=None, sa_column=Column(Text))  # [{front,back}]
    daily_minutes: int = Field(default=30)  # 每日计划投入时长
    schedule_type: str = Field(default="normal")  # normal | short | long | temp
    created_at: str = Field(default_factory=now_iso)

    # SM-2 / 艾宾浩斯
    reps: int = Field(default=0)  # 已完成复习次数（第几个复习点，0=尚未学习）
    easiness_factor: float = Field(default=2.5)  # EF 难度因子
    interval_days: int = Field(default=0)
    due_date: str = Field(default_factory=today_str, index=True)  # 下次复习日 YYYY-MM-DD
    last_reviewed_at: Optional[str] = None
    last_quality: Optional[int] = None  # 0-5，由正确率换算
    last_accuracy: Optional[float] = None

    # 计时状态机（服务端权威）：idle / running / paused / done_today
    run_state: str = Field(default="idle", index=True)
    running_since: Optional[str] = None  # ISO 时刻，running 时有效
    accumulated_sec: int = Field(default=0)  # 本次未结算的累计秒数（暂停不计入）
    total_time_sec: int = Field(default=0)  # 历史累计用时

    status: str = Field(default="active", index=True)  # active | graduated | archived


class ReviewLog(SQLModel, table=True):
    """每次复习的结算记录，用于打卡 / 周统计 / 正确率曲线。"""

    __tablename__ = "review_log"

    id: Optional[int] = Field(default=None, primary_key=True)
    item_id: int = Field(index=True, foreign_key="review_item.id")
    reviewed_date: str = Field(default_factory=today_str, index=True)  # YYYY-MM-DD
    duration_sec: int = Field(default=0)
    accuracy: Optional[float] = None  # 0-1
    quality: Optional[int] = None  # SM-2 0-5
    known: int = Field(default=0)
    unknown: int = Field(default=0)
    created_at: str = Field(default_factory=now_iso)


class Category(SQLModel, table=True):
    """书架分类。"""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    icon: str = "📂"
    position: int = Field(default=0)


class Document(SQLModel, table=True):
    """书架文档。原文落在 DOCS_DIR，解析文本存库（供搜索 / RAG / 分析）。"""

    __tablename__ = "document"

    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(index=True)
    filename: str
    path: str  # 原始文件相对 DATA_DIR 的路径
    fmt: str = Field(default="pdf", index=True)  # pdf | docx | pptx | md | txt
    size_bytes: int = Field(default=0)
    pages: Optional[int] = None
    category_id: Optional[int] = Field(default=None, foreign_key="category.id", index=True)
    uploaded_at: str = Field(default_factory=now_iso)

    text_content: str = Field(default="", sa_column=Column(Text))  # 解析后的纯文本
    text_chars: int = Field(default=0)
    ocr_used: bool = Field(default=False)

    # AI 分析产物
    summary_md: Optional[str] = Field(default=None, sa_column=Column(Text))
    mindmap_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    quiz_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    analyzed_at: Optional[str] = None
    analysis_error: Optional[str] = None

    # 向量索引状态
    indexed: bool = Field(default=False, index=True)
    indexed_at: Optional[str] = None
    chunk_count: int = Field(default=0)
    index_error: Optional[str] = None


class ResearchTask(SQLModel, table=True):
    """深度研究任务。活动日志 / 报告正文落在 REPORTS_DIR，DB 存元数据与产物摘要。"""

    __tablename__ = "research_task"

    id: str = Field(primary_key=True)  # task_xxxxxx
    query: str
    workspace_json: str = Field(default="{}")
    thinking_level: str = Field(default="standard")  # quick | standard | deep
    paradigm: str = Field(default="plan_execute")    # plan_execute | react | multi_agent
    skills_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    web_enabled: bool = Field(default=True)
    status: str = Field(default="queued", index=True)  # queued|running|paused|completed|failed|cancelled
    progress: float = Field(default=0.0)
    current_step: str = Field(default="")
    sources_visited: int = Field(default=0)
    conflicts_found: int = Field(default=0)
    error: Optional[str] = None
    created_at: str = Field(default_factory=now_iso)
    finished_at: Optional[str] = None
    report_title: Optional[str] = None
    report_word_count: Optional[int] = None
    report_json: Optional[str] = Field(default=None, sa_column=Column(Text))  # citations 等
    report_dir: Optional[str] = None


class Setting(SQLModel, table=True):
    """键值设置（模型配置、提醒时间、embedding 模型等）。Key 仅存本机。"""

    key: str = Field(primary_key=True)
    value: str = Field(default="{}", sa_column=Column(Text))  # JSON

    def data(self) -> Any:
        try:
            return json.loads(self.value)
        except Exception:
            return {}


def put_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
