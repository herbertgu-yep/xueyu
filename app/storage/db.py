"""SQLite 引擎与会话管理。"""
from __future__ import annotations

from sqlmodel import Session, SQLModel, create_engine, select

from app.config import DB_PATH
from app.storage import models as _models  # noqa: F401  # 注册表模型

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
    echo=False,
)


def init_db() -> None:
    _models  # 副作用导入：注册 SQLModel 表模型
    SQLModel.metadata.create_all(engine)
    _migrate_columns()
    _seed_default_categories()


def _migrate_columns() -> None:
    """轻量列迁移：旧库补新列。"""
    targets = {
        "research_task": (("paradigm", "VARCHAR(20) DEFAULT 'plan_execute'"),
                          ("skills_json", "TEXT")),
        "review_item": (("schedule_type", "VARCHAR(20) DEFAULT 'normal'"),),
    }
    with engine.connect() as conn:
        for table, columns in targets.items():
            cols = [r[1] for r in conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()]
            for col, ddl in columns:
                if col not in cols:
                    try:
                        conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
                        conn.commit()
                    except Exception:
                        pass


def _seed_default_categories() -> None:
    with Session(engine) as s:
        if s.exec(select(Category)).first() is not None:
            return
        defaults = [("计算机", "💻"), ("英语", "🔤"), ("数学", "📐"), ("论文", "📑")]
        for i, (name, icon) in enumerate(defaults):
            s.add(Category(name=name, icon=icon, position=i))
        s.commit()


from app.storage.models import Category  # noqa: E402
