"""笔记服务 —— 以 .md 文件 + YAML frontmatter 落盘（Obsidian 兼容），SQLite 不存正文。

目录结构：NOTES_DIR/<文件夹>/<id>-<slug>.md
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import frontmatter

from app.config import NOTES_DIR

_ID_RE = re.compile(r"^note_([0-9a-f]{6})")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _slugify(name: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|#^\[\]]", "", name).strip()
    s = s.replace(" ", "_")
    return s[:40] or "untitled"


def _note_files(folder: Optional[str] = None) -> list[Path]:
    if folder:
        d = NOTES_DIR / folder
        return sorted(d.glob("*.md")) if d.exists() else []
    return sorted(NOTES_DIR.rglob("*.md"))


def _path_of(note_id: str, folder: Optional[str] = None) -> Optional[Path]:
    """按 id 定位文件（改名/移动后仍可寻址）。"""
    for f in _note_files(folder):
        if _extract_id(f) == note_id:
            return f
    return None


def _extract_id(path: Path) -> str:
    m = _ID_RE.search(path.stem)
    return m.group(1) if m else path.stem


def _id_in_name(path: Path) -> bool:
    return bool(_ID_RE.search(path.stem))


def _plain_excerpt(text: str, limit: int = 60) -> str:
    """剥离 Markdown 标记后截断，用于列表摘要。"""
    t = re.sub(r"```[\s\S]*?```", " [代码] ", text)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"!\[.*?\]\(.*?\)", " ", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"^#{1,6}\s+", "", t, flags=re.M)
    t = re.sub(r"^[>\-+*]\s+", "", t, flags=re.M)
    t = re.sub(r"[*_~#|]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()[:limit]


def list_folders() -> list[dict]:
    if not NOTES_DIR.exists():
        return []
    out = []
    for d in sorted(NOTES_DIR.iterdir()):
        if d.is_dir():
            count = len(list(d.glob("*.md")))
            if count or any(d.iterdir()):
                out.append({"folder": d.name, "count": count})
    return out


def ensure_folder(name: str) -> Path:
    name = _slugify(name)
    d = NOTES_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_notes(folder: Optional[str] = None, q: Optional[str] = None) -> list[dict]:
    out = []
    for f in _note_files(folder):
        try:
            post = frontmatter.load(f)
        except Exception:
            continue
        title = str(post.get("title") or f.stem)
        body = post.content
        if q and q.lower() not in title.lower() and q.lower() not in body.lower():
            continue
        meta_path = f.relative_to(NOTES_DIR).parent
        out.append({
            "id": _extract_id(f),
            "title": title,
            "folder": meta_path.as_posix() if str(meta_path) != "." else "",
            "tags": post.get("tags") or [],
            "ai_assisted": bool(post.get("ai_assisted", False)),
            "updated": str(post.get("updated") or datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")),
            "created": str(post.get("created") or ""),
            "excerpt": _plain_excerpt(body),
        })
    out.sort(key=lambda n: n["updated"], reverse=True)
    return out


def read_note(note_id: str) -> dict:
    path = _path_of(note_id)
    if not path or not path.exists():
        raise KeyError("笔记不存在")
    post = frontmatter.load(path)
    rel = path.relative_to(NOTES_DIR).parent
    return {
        "id": _extract_id(path),
        "title": str(post.get("title") or path.stem),
        "folder": rel.as_posix() if str(rel) != "." else "",
        "tags": post.get("tags") or [],
        "ai_assisted": bool(post.get("ai_assisted", False)),
        "created": str(post.get("created") or ""),
        "updated": str(post.get("updated") or ""),
        "content": post.content,
        "path": str(path),
    }


def _write(note_id: str, folder: str, title: str, tags: list[str], content: str,
           ai_assisted: bool, created: str) -> Path:
    d = ensure_folder(folder or "未分类")
    path = d / f"note_{note_id}-{_slugify(title)}.md"
    post = frontmatter.Post(content, **{
        "id": f"note_{note_id}",
        "title": title,
        "folder": folder or "未分类",
        "tags": tags,
        "created": created,
        "updated": _now(),
        "ai_assisted": ai_assisted,
    })
    path.write_text(frontmatter.dumps(post), encoding="utf-8")
    return path


def create_note(*, title: str, folder: str = "未分类", tags: Optional[list[str]] = None,
                content: str = "", ai_assisted: bool = False) -> dict:
    note_id = uuid.uuid4().hex[:6]
    now = _now()
    path = _write(note_id, folder, title or "无标题笔记", tags or [], content, ai_assisted, now)
    return read_note(_extract_id(path))


def update_note(note_id: str, *, title: Optional[str] = None, folder: Optional[str] = None,
                tags: Optional[list[str]] = None, content: Optional[str] = None,
                ai_assisted: Optional[bool] = None, touch: bool = True) -> dict:
    path = _path_of(note_id)
    if not path or not path.exists():
        raise KeyError("笔记不存在")
    old = frontmatter.load(path)
    new_title = title if title is not None else str(old.get("title") or path.stem)
    new_folder = folder if folder is not None else str(old.get("folder") or "")
    new_tags = tags if tags is not None else (old.get("tags") or [])
    new_content = content if content is not None else old.content
    ai = ai_assisted if ai_assisted is not None else bool(old.get("ai_assisted", False))
    created = str(old.get("created") or _now())
    old_path = path
    new_path = _write(_extract_id(path), new_folder, new_title, list(new_tags), new_content, ai, created)
    if old_path.resolve() != new_path.resolve() and old_path.exists():
        old_path.unlink()  # 改名/换文件夹后清理旧文件
    _prune_versions(old_path.parent / f".versions-{_extract_id(path)}")
    return read_note(_extract_id(new_path))


_VERSIONS_KEEP = 20


def save_version(note_id: str, content: str, label: str = "") -> None:
    """保存历史版本（P1：保存前快照，保留最近 20 份）。"""
    path = _path_of(note_id)
    if not path:
        return
    vdir = path.parent / f".versions-{_extract_id(path)}"
    vdir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    (vdir / f"{ts}{'-' + _slugify(label) if label else ''}.md").write_text(content, encoding="utf-8")
    _prune_versions(vdir)


def list_versions(note_id: str) -> list[dict]:
    path = _path_of(note_id)
    if not path:
        return []
    vdir = path.parent / f".versions-{_extract_id(path)}"
    if not vdir.exists():
        return []
    out = []
    for f in sorted(vdir.glob("*.md"), reverse=True):
        out.append({"file": f.name, "time": f.stem, "size": f.stat().st_size})
    return out


def restore_version(note_id: str, version_file: str) -> dict:
    path = _path_of(note_id)
    vdir = path.parent / f".versions-{_extract_id(path)}"
    vf = (vdir / version_file).resolve()
    if not vf.exists() or vdir.resolve() not in vf.parents:
        raise KeyError("版本不存在")
    current = read_note(note_id)
    save_version(note_id, current["content"], label="restore-backup")
    return update_note(note_id, content=vf.read_text(encoding="utf-8"))


def _prune_versions(vdir: Path) -> None:
    if not vdir.exists():
        return
    files = sorted(vdir.glob("*.md"), reverse=True)
    for f in files[_VERSIONS_KEEP:]:
        f.unlink()


def delete_note(note_id: str) -> None:
    path = _path_of(note_id)
    if path and path.exists():
        path.unlink()
        vdir = path.parent / f".versions-{_extract_id(path)}"
        if vdir.exists():
            shutil.rmtree(vdir, ignore_errors=True)


def search(q: str, limit: int = 30) -> list[dict]:
    if not q.strip():
        return []
    ql = q.lower()
    hits = []
    for f in _note_files():
        try:
            post = frontmatter.load(f)
        except Exception:
            continue
        title = str(post.get("title") or f.stem)
        if ql in title.lower():
            hits.append({"id": _extract_id(f), "title": title,
                         "snippet": re.sub(r"\s+", " ", post.content[:120]), "score": 2})
        elif ql in post.content.lower():
            idx = post.content.lower().find(ql)
            hits.append({"id": _extract_id(f), "title": title,
                         "snippet": re.sub(r"\s+", " ", post.content[max(0, idx - 40): idx + 80]), "score": 1})
    hits.sort(key=lambda h: -h["score"])
    return hits[:limit]


# ---------------------------------------------------------------- 附件（自测题等）

def _asset_dir(note_id: str) -> Path:
    d = NOTES_DIR.parent / "note_assets" / note_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_asset(note_id: str, kind: str, content: str) -> str:
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{kind}-{ts}.md"
    (_asset_dir(note_id) / name).write_text(content, encoding="utf-8")
    return name


def list_assets(note_id: str) -> list[dict]:
    d = NOTES_DIR.parent / "note_assets" / note_id
    if not d.exists():
        return []
    out = []
    for f in sorted(d.glob("*.md"), reverse=True):
        out.append({"name": f.name, "size": f.stat().st_size,
                    "time": datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")})
    return out


def read_asset(note_id: str, name: str) -> str:
    if "/" in name or ".." in name or not name.endswith(".md"):
        raise KeyError("附件不存在")
    f = _asset_dir(note_id) / name
    if not f.exists():
        raise KeyError("附件不存在")
    return f.read_text(encoding="utf-8")


def delete_asset(note_id: str, name: str) -> None:
    if "/" in name or ".." in name or not name.endswith(".md"):
        raise KeyError("附件不存在")
    f = _asset_dir(note_id) / name
    if f.exists():
        f.unlink()


# ---------------------------------------------------------------- 文件夹管理

def rename_folder(old: str, new: str) -> dict:
    src = NOTES_DIR / old
    new_slug = _slugify(new)
    dst = NOTES_DIR / new_slug
    if not src.is_dir() or old == new_slug:
        raise ValueError("文件夹不存在或名称无效")
    if dst.exists():
        raise ValueError(f"文件夹「{new_slug}」已存在")
    src.rename(dst)
    for f in dst.glob("*.md"):
        post = frontmatter.load(f)
        post["folder"] = new_slug
        f.write_text(frontmatter.dumps(post), encoding="utf-8")
    return {"folder": new_slug}


def delete_folder(name: str) -> dict:
    """删除文件夹：里面的笔记自动归入「未分类」。"""
    src = NOTES_DIR / name
    if not src.is_dir():
        raise ValueError("文件夹不存在")
    target = ensure_folder("未分类")
    moved = 0
    for f in list(src.glob("*.md")):
        post = frontmatter.load(f)
        post["folder"] = "未分类"
        nf = target / f.name
        nf.write_text(frontmatter.dumps(post), encoding="utf-8")
        f.unlink()
        moved += 1
    shutil.rmtree(src, ignore_errors=True)
    return {"moved": moved}


# ---------------------------------------------------------------- AI 会话历史

def _chat_dir(note_id: str) -> Path:
    d = NOTES_DIR.parent / "note_assets" / note_id / "chats"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_chats(note_id: str) -> list[dict]:
    out = []
    for f in sorted(_chat_dir(note_id).glob("chat-*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            out.append({"id": f.stem, "title": data.get("title") or "新对话",
                        "count": len(data.get("messages") or []),
                        "updated": data.get("updated") or datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")})
        except Exception:
            continue
    return out


def create_chat(note_id: str, title: str = "新对话") -> dict:
    cid = f"chat-{uuid.uuid4().hex[:8]}"
    data = {"id": cid, "title": title, "created": _now(), "updated": _now(), "messages": []}
    (_chat_dir(note_id) / f"{cid}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return {"id": cid, "title": title}


def read_chat(note_id: str, cid: str) -> dict:
    if "/" in cid or ".." in cid:
        raise KeyError("会话不存在")
    f = _chat_dir(note_id) / f"{cid}.json"
    if not f.exists():
        raise KeyError("会话不存在")
    return json.loads(f.read_text(encoding="utf-8"))


def save_chat(note_id: str, cid: str, *, title: Optional[str] = None,
              messages: Optional[list[dict]] = None) -> dict:
    data = read_chat(note_id, cid)
    if title:
        data["title"] = title
    if messages is not None:
        data["messages"] = messages[-60:]  # 上限保护
    data["updated"] = _now()
    (_chat_dir(note_id) / f"{cid}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def delete_chat(note_id: str, cid: str) -> None:
    f = _chat_dir(note_id) / f"{cid}.json"
    if f.exists():
        f.unlink()


def note_text(note_id: str) -> str:
    """供研究工作区引用的纯文本。"""
    n = read_note(note_id)
    return f"# {n['title']}\n\n{n['content']}"
