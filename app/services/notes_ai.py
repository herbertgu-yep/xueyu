"""笔记 AI：总结 / 续写 / 润色 / 自测题 / 自由对话。

输出一律为 Markdown，前端可一键插入笔记正文。
"""
from __future__ import annotations

from typing import Optional

from app.services import llm, notes_service
from app.services.llm import LLMCallError

_BASE = "你是「学屿」里的学习助手，贴合用户笔记的语境回答，输出 Markdown，中文为主。"


QUIZ_COUNT = {"count": 6}


def run_action(note_id: str, action: str, message: Optional[str] = None,
               history: Optional[list[dict]] = None, *, web: bool = False,
               thinking: str = "standard") -> dict:
    note = notes_service.read_note(note_id)
    title, content = note["title"], note["content"]
    tags = "、".join(note.get("tags") or [])

    if action == "chat":
        if not (message or "").strip():
            raise ValueError("消息不能为空")
        reply, sources = _chat(title, content, message or "", history or [], web=web, thinking=thinking)
        out = {"reply": reply, "action": action}
        if sources:
            out["sources"] = sources
        return out

    if action == "summarize":
        prompt = (f"总结这篇笔记，输出 3-8 条要点（Markdown 无序列表），突出可复用的结论与易错点：\n\n"
                  f"# {title}\n\n{content[:12000]}")
        system = _BASE + "总结要短而准，别复述原文结构。"
    elif action == "continue":
        req = (message or "").strip()
        requirement = f"\n\n用户的续写要求：{req}" if req else ""
        prompt = (f"续写这篇笔记（从原文结尾自然接续，风格一致，150-400 字，可直接拼在原文后）。"
                  f"{requirement}\n\n# {title}\n\n{content[-6000:]}")
        system = _BASE + "只输出续写内容，不要重复原文。"
    elif action == "polish":
        prompt = (f"润色改写这篇笔记：纠错、理顺表述、补小标题，保留全部知识点与原意，"
                  f"输出润色后的完整 Markdown 正文：\n\n# {title}\n\n{content[:12000]}")
        system = _BASE + "只输出润色后的正文，不要解释改了什么。"
    elif action == "quiz":
        n = QUIZ_COUNT["count"] or 6
        prompt = (f"基于这篇笔记出 {n} 道自测题，输出 Markdown：题目用加粗，答案跟在后面并用 > 引用块折叠提示，"
                  f"题型混合（问答/填空/应用）：\n\n# {title}\n\n{content[:12000]}")
        system = _BASE + "题目要能暴露没学牢的点。"
    else:
        raise ValueError(f"未知操作：{action}")

    temp = {"quick": 0.2, "standard": 0.4, "deep": 0.6}.get(thinking, 0.4)
    reply = llm.acompletion([
        {"role": "system", "content": system + (f"\n笔记标签：{tags}" if tags else "")},
        {"role": "user", "content": prompt},
    ], temperature=temp, fn="note")
    return {"reply": reply, "action": action}


async def stream_action(note_id: str, action: str, message: Optional[str] = None,
                        history: Optional[list[dict]] = None, *, thinking: str = "standard",
                        extra_context: Optional[str] = None):
    """流式执行 AI 操作（SSE 消费端）：yield {'type':'delta','text'} ... {'type':'done','text'}。

    支持 summarize / continue / polish / chat。真实流式调用（litellm astream）。
    """
    note = notes_service.read_note(note_id)
    title, content = note["title"], note["content"]
    temp = {"quick": 0.2, "standard": 0.4, "deep": 0.6}.get(thinking, 0.4)

    if action == "chat":
        system = _BASE + f"当前笔记：《{title}》全文会附在最后一条消息里，回答要贴合笔记内容；若用户在问与笔记无关的问题，也正常回答。"
        msgs = [{"role": "system", "content": system}]
        for h in (history or [])[-8:]:
            role = "assistant" if h.get("role") in ("assistant", "bot") else "user"
            msgs.append({"role": role, "content": str(h.get("content", ""))[:3000]})
        user_content = f"【当前笔记《{title}》】\n{content[:8000]}\n\n【用户消息】{message or ''}"
        if extra_context:
            user_content += "\n\n" + extra_context
        msgs.append({"role": "user", "content": user_content})
    elif action == "summarize":
        system = _BASE + "总结要短而准，别复述原文结构。"
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": f"总结这篇笔记，输出 3-8 条要点（Markdown 无序列表），突出可复用的结论与易错点：\n\n# {title}\n\n{content[:12000]}"}]
    elif action == "continue":
        req = (message or "").strip()
        requirement = f"\n\n用户的续写要求：{req}" if req else ""
        system = _BASE + "只输出续写内容，不要重复原文。"
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": f"续写这篇笔记（从原文结尾自然接续，风格一致，150-400 字，可直接拼在原文后）。{requirement}\n\n# {title}\n\n{content[-6000:]}"}]
    elif action == "polish":
        system = _BASE + "只输出润色后的正文，不要解释改了什么。"
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": f"润色改写这篇笔记：纠错、理顺表述、补小标题，保留全部知识点与原意，输出润色后的完整 Markdown 正文：\n\n# {title}\n\n{content[:12000]}"}]
    else:
        raise ValueError(f"该操作不支持流式：{action}")

    chunks = []
    async for delta in llm.astream(msgs, temperature=temp):
        chunks.append(delta)
        yield {"type": "delta", "text": delta}
    full = "".join(chunks)
    yield {"type": "done", "text": full, "action": action}


async def polish_variants(note_id: str, count: int = 3) -> list[dict]:
    """润色一次给出多个风格方案，供用户挑选（真实并发 LLM 调用）。"""
    import asyncio as _asyncio

    note = notes_service.read_note(note_id)
    title, content = note["title"], note["content"]
    styles = [
        ("精炼版", "润色时追求简洁清晰：合并重复内容、删冗词、句子更短促有力，保留全部知识点。"),
        ("深度版", "润色时适度展开：为关键结论补充解释和例子，让笔记自学时可读性更强，但不得偏离原意。"),
        ("结构版", "润色时重组结构：补小标题、要点编号、层级清晰，适合复习浏览。"),
    ][:max(2, min(4, count))]
    base = "只输出润色后的完整 Markdown 正文，不要解释改了什么。"

    async def one(style_name: str, style_hint: str) -> dict:
        text = await llm.achat([
            {"role": "system", "content": _BASE + style_hint + " " + base},
            {"role": "user", "content": f"润色这篇笔记：\n\n# {title}\n\n{content[:12000]}"},
        ], temperature=0.4, fn="note")
        return {"style": style_name, "text": text}

    return list(await _asyncio.gather(*(one(n, h) for n, h in styles)))


def _chat(title: str, content: str, message: str, history: list[dict], *,
          web: bool = False, thinking: str = "standard") -> tuple[str, list[dict]]:
    sources: list[dict] = []
    web_ctx = ""
    if web:
        from app.agent import websearch

        results = websearch.search_web(message, max_results=5)
        parts = []
        for i, r in enumerate(results[:3], 1):
            body = websearch.fetch_page_sync(r["url"]) or r["snippet"]
            if body:
                parts.append(f"[{i}] {r['title']}\n{r['url']}\n{body[:2500]}")
                sources.append({"ref_id": i, "title": r["title"], "url": r["url"]})
        if parts:
            web_ctx = "\n\n【联网检索结果】（回答时用 [n] 标注引用）\n" + "\n\n".join(parts)

    temp = {"quick": 0.2, "standard": 0.5, "deep": 0.75}.get(thinking, 0.5)
    depth = {"quick": "回答简洁直接。", "standard": "",
             "deep": "回答要深入展开：给出原理、例子与注意事项。"}[thinking]
    msgs = [{"role": "system", "content":
             _BASE + f"当前笔记：《{title}》全文会附在最后一条消息里，回答要贴合笔记内容；"
                     "若用户在问与笔记无关的问题，也正常回答。" + depth}]
    for h in history[-8:]:
        role = "assistant" if h.get("role") in ("assistant", "bot") else "user"
        msgs.append({"role": role, "content": str(h.get("content", ""))[:3000]})
    msgs.append({"role": "user", "content": f"【当前笔记《{title}》】\n{content[:8000]}\n\n【用户消息】{message}{web_ctx}"})
    # 长对话自动压缩（同步调用场景：先按预算裁剪历史）
    ctx = llm.model_context_tokens()
    budget = int(ctx * 0.45)
    while len(msgs) > 4 and llm.estimate_messages_tokens(msgs) > budget:
        drop = 1 if msgs[0].get("role") != "system" else 1
        if msgs[drop].get("role") == "system" or len(msgs) <= 4:
            break
        msgs.pop(drop)
    return llm.acompletion(msgs, temperature=temp, fn="note"), sources


def generate_flashcards(text: str, *, title: str = "", count: int = 12) -> list[dict]:
    """从素材生成复习卡片（正面=提问，背面=要点答案）。"""
    data = llm.acompletion([
        {"role": "system", "content":
            "你是出题助手。根据学习材料生成间隔复习卡片 JSON："
            f'[{{"front": "问题/提示", "back": "要点答案，1-3 句"}}]，共 {count} 张左右，覆盖核心考点；'
            "只输出 JSON，不要解释。"},
        {"role": "user", "content": f"材料《{title}》：\n\n{text[:16000]}"},
    ], temperature=0.4)
    return _parse_cards(llm.parse_json(data))


async def generate_flashcards_async(text: str, *, title: str = "", count: int = 12) -> list[dict]:
    data = await llm.ask_json(
        "你是出题助手。根据学习材料生成间隔复习卡片 JSON："
        f'[{{"front": "问题/提示", "back": "要点答案，1-3 句"}}]，共 {count} 张左右，覆盖核心考点；只输出 JSON。',
        f"材料《{title}》：\n\n{text[:16000]}",
        temperature=0.4,
        fn="note",
    )
    return _parse_cards(data)


def _parse_cards(data) -> list[dict]:
    if not isinstance(data, list) or not data:
        raise LLMCallError("卡片生成结果为空")
    return [{"front": str(x.get("front", "")).strip(), "back": str(x.get("back", "")).strip()}
            for x in data if x.get("front")]
