"""深度研究 Agent 平台 —— 三种范式 + Function Calling + 任务记忆 + 技能 + MCP 工具。

范式（可在设置页与研究页选择）：
- plan_execute  智能规划：拆解 → 检索 →（覆盖度不足自动补检）→ 交叉验证 → 报告（默认，稳）
- react         ReAct 循环：LLM 自主选择内置工具/MCP 工具，边想边查，步数可配
- multi_agent   多智能体：每个子课题一个并行研究员，汇总裁判与撰写

共享能力：任务记忆（memory.md，句句沉淀事实）、技能注入（Markdown playbooks）、
        MCP 外部工具（stdio 客户端）、信源统一编号、冲突并列标注、暂停/取消。
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Callable, Optional

from app.agent import websearch
from app.config import THINKING_PROFILES
from sqlmodel import Session, select

from app.services import llm, settings_store, notes_service, rag_service


class CancelledByUser(Exception):
    pass


EMIT = Callable[[str, dict], None]

RESEARCH_TOOLS = [
    {"type": "function", "function": {
        "name": "web_search", "description": "联网搜索。返回标题+URL+摘要列表。",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "open_page", "description": "抓取网页正文全文。",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "shelf_search", "description": "在书架文档的本地向量索引中检索。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "note_search", "description": "在用户笔记中全文搜索。",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "memory_save", "description": "把关键事实/结论存入任务记忆（写报告时会使用）。",
        "parameters": {"type": "object", "properties": {"fact": {"type": "string"}}, "required": ["fact"]}}},
    {"type": "function", "function": {
        "name": "finish_research", "description": "检索已充分，结束研究阶段进入报告撰写。",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]


class ResearchRunner:
    def __init__(self, *, task_id: str, query: str, workspace: dict,
                 thinking_level: str, web_enabled: bool, emit: EMIT,
                 paradigm: str = "plan_execute", skills: Optional[list[dict]] = None,
                 mcp_clients: Optional[list] = None, react_max_steps: int = 14,
                 auto_memory: bool = True, memory_path: Optional[Path] = None,
                 hybrid_multi_query: bool = True, hybrid_rerank: bool = False):
        self.task_id = task_id
        self.query = query
        self.workspace = workspace or {"type": "web"}
        self.thinking_level = thinking_level
        self.profile = THINKING_PROFILES.get(thinking_level, THINKING_PROFILES["standard"])
        self.web_enabled = web_enabled
        self.emit = emit
        self.paradigm = paradigm if paradigm in ("plan_execute", "react", "multi_agent") else "plan_execute"
        self.skills = skills or []
        self.mcp_clients = mcp_clients or []
        self.react_max_steps = max(4, int(react_max_steps or 14))
        self.auto_memory = auto_memory
        self.memory_path = memory_path
        self.hybrid_multi_query = hybrid_multi_query
        self.hybrid_rerank = hybrid_rerank
        self.memory: list[str] = []
        self.pause_evt = asyncio.Event()
        self.pause_evt.set()
        self.cancel_flag = asyncio.Event()
        self.sources: list[dict] = []
        self._url_seen: set[str] = set()
        self._conflicts: list[dict] = []

    # ---------------- 控制 ----------------
    async def gate(self) -> None:
        while True:
            if self.cancel_flag.is_set():
                raise CancelledByUser()
            if self.pause_evt.is_set():
                return
            await self.pause_evt.wait()

    def pause(self) -> None:
        self.pause_evt.clear()

    def resume(self) -> None:
        self.pause_evt.set()

    def cancel(self) -> None:
        self.cancel_flag.set()
        self.pause_evt.set()

    def check_cancel(self) -> None:
        if self.cancel_flag.is_set():
            raise CancelledByUser()

    # ---------------- 事件 / 记忆 ----------------
    def log(self, message: str, *, kind: str = "info", url: str = "") -> None:
        self.emit("log", {"kind": kind, "message": message,
                          "time": time.strftime("%H:%M:%S"), "url": url})

    def set_progress(self, progress: float, step: str) -> None:
        self.emit("progress", {"progress": round(progress, 3), "step": step})

    def memory_save(self, fact: str) -> None:
        fact = str(fact).strip()[:400]
        if not fact or fact in self.memory:
            return
        self.memory.append(fact)
        if self.memory_path:
            try:
                self.memory_path.parent.mkdir(parents=True, exist_ok=True)
                self.memory_path.write_text("\n".join(f"- {m}" for m in self.memory), encoding="utf-8")
            except Exception:
                pass

    def _workspace_prompt(self) -> str:
        """让拆解/撰写阶段知道工作区里有什么——尤其 docs 全文工作区。"""
        ws_type = self.workspace.get("type", "web")
        if ws_type == "docs":
            doc_ids = self.workspace.get("doc_ids") or []
            titles = []
            from app.storage.db import engine as _engine
            from app.storage.models import Document as _Document
            from sqlmodel import Session as _Session

            with _Session(_engine) as s:
                for did in doc_ids:
                    d = s.get(_Document, did)
                    if d:
                        titles.append(f"《{d.title}》（{len(d.text_content or '')} 字）")
            if titles:
                return (
                    "\n\n【重要上下文】用户为本次研究选择了本地书架文件：" + "、".join(titles)
                    + "。用户问题中的「这个文件」「该文档」即指这些文件。"
                    + "你的任务是基于这些文件的**内容本身**展开研究（核心论点、方法、结论、应用），"
                    + "不要把问题理解为研究文件格式、文件类型识别等元信息话题；"
                    + "这些文件的完整原文会在检索阶段自动纳入，无需联网检索。"
                )
        if ws_type == "shelf_rag":
            return "\n\n【重要上下文】本研究的资料来自用户书架的向量索引（用户上传的文档内容）。"
        return ""

    def skills_prompt(self) -> str:
        if not self.skills:
            return ""
        return "\n\n【用户启用的研究技能（必须遵守）】\n" + "\n".join(
            f"### 技能：{s.get('name', '技能')}\n{s.get('content', '')[:1500]}" for s in self.skills)

    def memory_prompt(self) -> str:
        if not self.memory:
            return ""
        return "\n\n【任务记忆（研究过程中沉淀的关键事实）】\n" + "\n".join(f"- {m}" for m in self.memory[:40])

    # ---------------- 信源 ----------------
    def add_source(self, *, origin: str, title: str, url: str, text: str, snippet: str = "") -> Optional[dict]:
        url_key = url or f"{origin}:{title}"
        if url_key in self._url_seen:
            return None
        self._url_seen.add(url_key)
        ref = {
            "ref_id": len(self.sources) + 1,
            "origin": origin,
            "title": (title or url or "未命名信源").strip()[:160],
            "url": url,
            "text": (text or snippet or "").strip(),
            "accessed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self.sources.append(ref)
        return ref

    def _sources_listing(self, cap: int = 700, doc_cap: Optional[int] = None) -> str:
        parts = []
        for s in self.sources:
            text = (s["text"] or "").strip().replace("\n", " ")
            limit = doc_cap if (s["origin"] == "doc" and doc_cap) else cap
            parts.append(f"[{s['ref_id']}] {s['title']}（{s['origin']}）{s['url']}\n    {text[:limit]}")
        return "\n".join(parts)

    def _format_source(self, s: dict, cap: int) -> str:
        text = (s["text"] or "").strip().replace("\n", " ")
        return f"[{s['ref_id']}] {s['title']}（{s['origin']}）{s['url']}\n    {text[:cap]}"

    def _report_sources_listing(self) -> tuple[str, str]:
        """撰写阶段的信源清单：按模型上下文窗口动态分配预算。

        本地文档（doc）源优先完整原文纳入，放不下时按比例截断并如实告知；
        网络信源固定 1100 字上限。返回 (清单文本, 状态说明)。
        """
        ctx_tokens = llm.model_context_tokens(settings_store.llm_config("research"))
        budget = max(8000, int(ctx_tokens * 0.85) - 12000 - 1500)  # 预留报告输出与指令开销

        doc_sources = [s for s in self.sources if s["origin"] == "doc"]
        other_sources = [s for s in self.sources if s["origin"] != "doc"]
        others_tokens = sum(llm.estimate_tokens((s["text"] or "")[:1100]) + 12 for s in other_sources)
        doc_tokens_total = sum(llm.estimate_tokens(s["text"] or "") for s in doc_sources)

        note = ""
        caps: dict[int, int] = {}
        if doc_sources and doc_tokens_total + others_tokens > budget:
            allowed = max(3000, budget - others_tokens)
            ratio = allowed / max(1, doc_tokens_total)
            for s in doc_sources:
                caps[id(s)] = max(2000, int(len(s["text"] or "") * ratio))
            note = f"（文档原文超出上下文预算，已按比例截断至约 {int(ratio * 100)}%）"
            self.log(f"⚠️ 文档原文超出模型上下文预算，按比例截断至约 {int(ratio * 100)}%", kind="warn")
        elif doc_sources:
            note = f"（全部文档原文完整纳入，约 {int(doc_tokens_total)} tokens）"
            self.log(f"✓ 全部文档原文完整纳入撰写上下文（约 {int(doc_tokens_total)} tokens）", kind="ok")

        parts = []
        for s in self.sources:
            cap = caps.get(id(s), len(s["text"] or "") if s["origin"] == "doc" else 1100)
            parts.append(self._format_source(s, cap))
        return "\n".join(parts), note

    # ---------------- 入口 ----------------
    async def run(self) -> dict:
        if self.paradigm == "react":
            await self._react()
        elif self.paradigm == "multi_agent":
            await self._multi_agent()
        else:
            await self._plan_execute()
        report = await self._synthesize()
        return {"sources": self.sources, "conflicts": self._conflicts,
                "report_md": report["md"], "report_title": report["title"]}

    # ---------------- 范式一：Plan-Execute-Replan ----------------
    async def _plan_execute(self) -> None:
        self.set_progress(0.02, "拆解研究问题")
        self.log("🧩 正在拆解研究问题…")
        n = self.profile["subquestions"]
        data = await llm.ask_json(
            "你是研究策划。把研究问题拆成可检索的子课题 JSON：{\"subquestions\": [\"…\"]}"
            f"，{n} 个左右；子课题彼此独立、可分别检索、覆盖不同维度；只输出 JSON。",
            f"研究问题：{self.query}{self._workspace_prompt()}{self.skills_prompt()}",
            temperature=0.3, fn="research")
        subs = [str(x).strip() for x in data.get("subquestions", []) if str(x).strip()][:n] or [self.query]
        await self.gate()
        self.log(f"拆解完成 → {len(subs)} 个子课题：{' / '.join(subs)}", kind="ok")

        if self.workspace.get("type", "web") == "docs":
            # 选中的书架文件：完整原文一次性纳入（不随子课题重复、不联网）
            await self.gate()
            self.log("📄 将选中的书架文件以完整原文纳入研究材料", kind="ok")
            await self._full_docs_for(self.workspace.get("doc_ids") or [])
            self.set_progress(0.2, "研读文档原文")
        await self._retrieve_round(subs, progress_from=0.08, progress_to=0.66)

        # Replan：评估覆盖度，不足则补一轮
        await self.gate()
        self.set_progress(0.70, "评估覆盖度")
        gaps = await llm.ask_json(
            "你是研究督导。对照子课题与已收集信源，判断是否有关键维度缺失。"
            "输出 JSON：{\"sufficient\": true/false, \"missing\": [\"缺失角度的补充检索式\"]}；只输出 JSON。",
            f"研究问题：{self.query}\n子课题：{subs}\n\n已收集信源：\n{self._sources_listing(cap=300)}",
            temperature=0.1, fn="research")
        missing = [str(m) for m in (gaps.get("missing") or []) if str(m).strip()][:2]
        if not gaps.get("sufficient", True) and missing:
            self.log(f"🔁 覆盖度不足，补充检索：{' / '.join(missing)}", kind="warn")
            await self._retrieve_round(missing, progress_from=0.70, progress_to=0.80, light=True)
        else:
            self.log("覆盖度评估：信源已足够", kind="ok")

        await self._verify(0.84)
        self.set_progress(0.86, "准备撰写")

    async def _retrieve_round(self, subs: list[str], *, progress_from: float,
                              progress_to: float, light: bool = False) -> None:
        span = progress_to - progress_from
        for i, sub in enumerate(subs):
            await self.gate()
            self.log(f"🔍 检索：{sub}")
            await self._retrieve_one(sub, light=light)
            if self.auto_memory and self.sources:
                await self._extract_facts(sub)
            self.set_progress(progress_from + span * (i + 1) / len(subs), "检索中")
            self.emit("sources", {"sources_visited": len(self.sources)})

    async def _retrieve_one(self, sub: str, light: bool = False) -> None:
        ws_type = self.workspace.get("type", "web")
        if ws_type == "notes":
            nids = self.workspace.get("note_ids", [])
            if nids:
                for nid in nids:
                    await self.gate()
                    try:
                        meta = notes_service.read_note(nid)
                        self.add_source(origin="note", title=meta["title"], url=f"note://{nid}",
                                        text=notes_service.note_text(nid)[:8000])
                    except Exception:
                        self.log(f"⚠️ 笔记 {nid} 读取失败", kind="warn")
            else:
                self.log("⚠️ 未指定笔记，回退到整个书架 RAG", kind="warn")
                await self._rag_for(sub, doc_ids=None)
        elif ws_type == "shelf_rag":
            unindexed = self._unindexed_docs()
            if unindexed:
                raise RuntimeError(
                    f"书架有 {len(unindexed)} 份文档未完成向量化（{'、'.join(unindexed[:3])} 等），"
                    "请到「书架」页点击「重建索引」完成向量化后再使用「整个书架 · RAG」"
                )
            await self._rag_for(sub, doc_ids=None)
        elif self.web_enabled:
            await self._web_for(sub, light=light)
        else:
            self.log("⚠️ 全网工作区但联网关闭，跳过", kind="warn")
        # docs 全文工作区：原文已在检索阶段前整体纳入，这里不按子课题重复、不叠加联网

    def _unindexed_docs(self) -> list[str]:
        from app.storage.db import engine as _engine
        from app.storage.models import Document as _Document

        with Session(_engine) as s:
            docs = s.exec(select(_Document)).all()
        return [d.title for d in docs if d.text_content and not d.indexed]

    async def _full_docs_for(self, doc_ids: list[int]) -> None:
        """书架文件工作区：将选中文档的完整原文直接纳入研究（不走 RAG）。"""
        from app.storage.db import engine as _engine
        from app.storage.models import Document as _Document

        for did in doc_ids:
            await self.gate()
            with Session(_engine) as s:
                doc = s.get(_Document, did)
            if not doc:
                self.log(f"⚠️ 文档 {did} 不存在，已跳过", kind="warn")
                continue
            if not (doc.text_content or "").strip():
                self.log(f"⚠️ 《{doc.title}》没有可分析的文本（可能为扫描件），已跳过", kind="warn")
                continue
            text = doc.text_content
            if len(text) > 60000:
                text = text[:40000] + "\n…（中间内容略）…\n" + text[-15000:]
            self.add_source(origin="doc", title=doc.title, url=f"doc://{doc.id}", text=text)
            self.log(f"📄 已完整纳入《{doc.title}》（{len(doc.text_content)} 字）", kind="ok")
        self.emit("sources", {"sources_visited": len(self.sources)})

    async def _web_for(self, sub: str, light: bool = False) -> None:
        results = await websearch.asearch_web(sub, max_results=self.profile["results_per_search"])
        fetched = 0
        for r in results:
            await self.gate()
            if fetched >= (1 if light else self.profile["fetch_pages"]):
                if r["snippet"]:
                    self.add_source(origin="web", title=r["title"], url=r["url"], text=r["snippet"])
                continue
            text = await websearch.afetch_page(r["url"])
            fetched += 1
            if text:
                self.add_source(origin="web", title=r["title"], url=r["url"], text=text)
                self.log(f"已读取 {r['url']}：{r['title'][:50]}", kind="ok", url=r["url"])
            elif r["snippet"]:
                self.add_source(origin="web", title=r["title"], url=r["url"], text=r["snippet"])
        if not results:
            self.log(f"「{sub}」没有搜到公开结果", kind="warn")

    async def _rag_for(self, sub: str, doc_ids: Optional[list[int]]) -> None:
        try:
            chunks, meta = await rag_service.query_shelf_hybrid(
                sub, doc_ids=doc_ids,
                k=self.profile["results_per_search"],
                multi_query=self.hybrid_multi_query, rerank=self.hybrid_rerank)
            if len(meta.get("queries", [])) > 1:
                self.log(f"🔀 multi-query 改写为 {len(meta['queries'])} 个检索式", kind="ok")
            self.log(f"混合检索：向量 {meta['vec_hits']} + BM25 {meta['bm25_hits']} → RRF 融合"
                     + (" → LLM 重排" if meta.get("reranked") else ""), kind="ok")
        except Exception as e:  # noqa: BLE001
            self.log(f"⚠️ 混合检索失败：{e}", kind="warn")
            return
        for c in chunks:
            await self.gate()
            if self.add_source(origin="doc", title=f"{c['title']}（片段）",
                               url=f"doc://{c['doc_id']}", text=c["text"]):
                self.log(f"命中书架片段：《{c['title']}》", kind="ok")

    async def _extract_facts(self, topic: str) -> None:
        """把当前信源里的关键事实沉淀进任务记忆（一次便宜调用）。"""
        await self.gate()
        recent = self._sources_listing(cap=250)[-3000:]
        try:
            facts = await llm.ask_json(
                "从信源中提取 2-4 条关键事实/数字/结论 JSON：{\"facts\": [\"…\"]}，每条一句话；只输出 JSON。",
                f"主题：{topic}\n信源：\n{recent}", temperature=0.1, fn="research")
            for f in facts.get("facts", [])[:4]:
                self.memory_save(str(f))
        except Exception:
            pass

    # ---------------- 范式二：ReAct ----------------
    async def _react(self) -> None:
        self.set_progress(0.05, "ReAct 循环启动")
        mcp_tools = self._mcp_tool_specs()
        all_tools = RESEARCH_TOOLS + mcp_tools
        self.log(f"🤖 ReAct 循环：{len(all_tools)} 个工具可用"
                 + (f"（含 MCP：{', '.join(t['function']['name'] for t in mcp_tools)}）" if mcp_tools else ""), kind="ok")
        messages: list[dict] = [
            {"role": "system", "content":
                "你是深度研究 Agent。围绕研究问题自主使用工具收集信源："
                "先想清楚要查什么，逐个 web_search / shelf_search / note_search，"
                "重要网页用 open_page 读全文；随时 memory_save 沉淀关键事实；"
                "信息足够后调用 finish_research 结束。"
                "要求：多信源交叉验证；记住每条信源的编号；至少收集 6 个信源再结束。"
                + self.skills_prompt()},
            {"role": "user", "content": f"研究问题：{self.query}"},
        ]
        steps = 0
        finished = False
        while steps < self.react_max_steps and not finished:
            await self.gate()
            self.check_cancel()
            steps += 1
            self.set_progress(0.05 + 0.7 * steps / self.react_max_steps, f"ReAct 第 {steps} 步")
            try:
                resp = await llm.achat_raw(messages, tools=all_tools, fn="research")
            except llm.LLMCallError as e:
                self.log(f"⚠️ 模型调用失败：{e}", kind="warn")
                break
            msg = resp["message"]
            tool_calls = resp.get("tool_calls") or []
            if tool_calls:
                messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
                for tc in tool_calls:
                    await self.gate()
                    fname = tc["function"]["name"]
                    try:
                        fargs = json.loads(tc["function"]["arguments"] or "{}")
                    except Exception:
                        fargs = {}
                    self.log(f"🔧 {fname}（{json.dumps(fargs, ensure_ascii=False)[:80]}）", kind="now")
                    result = await self._exec_tool(fname, fargs)
                    if fname == "finish_research":
                        finished = True
                        result = "已确认结束。"
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            else:
                content = (msg.get("content") or "").strip()
                if content:
                    self.log(f"💭 {content[:120]}")
                messages.append({"role": "assistant", "content": content})
                if steps >= 3 and len(self.sources) >= 4:
                    break
        if self.sources:
            self.log(f"ReAct 结束：共 {steps} 步，收集 {len(self.sources)} 个信源，记忆 {len(self.memory)} 条", kind="ok")
        else:
            self.log("⚠️ ReAct 未收集到信源，回退到规划模式补检", kind="warn")
            await self._plan_execute()
            return
        await self._verify(0.80)

    def _mcp_tool_specs(self) -> list[dict]:
        specs = []
        for client in self.mcp_clients:
            for t in client.tools:
                specs.append({"type": "function", "function": {
                    "name": f"mcp__{client.name}__{t['name']}"[:64],
                    "description": f"[MCP:{client.name}] {t['description']}"[:300],
                    "parameters": t.get("inputSchema") or {"type": "object", "properties": {}}}})
        return specs

    async def _exec_tool(self, name: str, args: dict) -> str:
        try:
            if name == "web_search":
                results = await websearch.asearch_web(args.get("query", ""), max_results=self.profile["results_per_search"])
                for r in results[:4]:
                    self.add_source(origin="web", title=r["title"], url=r["url"], text=r["snippet"])
                return "\n".join(f"{r['title']} | {r['url']}\n{r['snippet']}" for r in results[:6]) or "无结果"
            if name == "open_page":
                text = await websearch.afetch_page(args.get("url", "")) or ""
                self.add_source(origin="web", title=args.get("url", "")[:80], url=args.get("url", ""), text=text)
                return text[:4000] or "抓取失败"
            if name == "shelf_search":
                chunks, _meta = await rag_service.query_shelf_hybrid(
                    args.get("query", ""), k=int(args.get("k", 5)), multi_query=False, rerank=False)
                for c in chunks:
                    self.add_source(origin="doc", title=f"{c['title']}（片段）",
                                    url=f"doc://{c['doc_id']}", text=c["text"])
                return "\n\n".join(f"《{c['title']}》：{c['text'][:500]}" for c in chunks) or "无结果"
            if name == "note_search":
                hits = notes_service.search(args.get("query", ""), 5)
                for h in hits:
                    self.add_source(origin="note", title=h["title"], url=f"note://{h['id']}", text=h["snippet"])
                return "\n".join(f"《{h['title']}》：{h['snippet']}" for h in hits) or "无结果"
            if name == "memory_save":
                self.memory_save(args.get("fact", ""))
                return "已存入任务记忆"
            if name == "finish_research":
                return "已确认结束。"
            if name.startswith("mcp__"):
                _, server, tool = name.split("__", 2)
                client = next((c for c in self.mcp_clients if c.name == server), None)
                if not client:
                    return "MCP 服务器未连接"
                return await client.call(tool, args)
            return f"未知工具 {name}"
        except CancelledByUser:
            raise
        except Exception as e:  # noqa: BLE001
            return f"工具执行失败：{e}"

    # ---------------- 范式三：Multi-Agent ----------------
    async def _multi_agent(self) -> None:
        self.set_progress(0.04, "拆解任务分工")
        n = max(2, self.profile["subquestions"])
        data = await llm.ask_json(
            "你是研究组长。把研究问题拆成子课题并分配研究员 JSON："
            "{\"agents\": [{\"name\": \"研究员A\", \"topic\": \"…\"}]}，"
            f"{min(4, n)} 个以内，视角互补；只输出 JSON。",
            f"研究问题：{self.query}{self.skills_prompt()}", temperature=0.3, fn="research")
        agents = data.get("agents") or []
        if not agents:
            agents = [{"name": f"研究员{i + 1}", "topic": self.query} for i in range(2)]
        agents = agents[:4]
        self.log(f"👥 {len(agents)} 个并行研究员：{'、'.join(a.get('name', '?') for a in agents)}", kind="ok")
        if self.workspace.get("type", "web") == "docs":
            await self._full_docs_for(self.workspace.get("doc_ids") or [])

        async def mini(agent: dict, idx: int) -> None:
            topic = agent.get("topic", self.query)
            name = agent.get("name", f"研究员{idx + 1}")
            base = len(self.sources)
            await self.gate()
            self.log(f"[{name}] 开始研究：{topic}", kind="now")
            if self.web_enabled:
                results = await websearch.asearch_web(topic, max_results=self.profile["results_per_search"])
                fetched = 0
                for r in results:
                    await self.gate()
                    text = await websearch.afetch_page(r["url"]) if fetched < 2 else None
                    if text:
                        fetched += 1
                    self.add_source(origin="web", title=r["title"], url=r["url"], text=text or r["snippet"])
            if self.workspace.get("type") == "shelf_rag":
                await self._rag_for(topic, doc_ids=None)
            got = len(self.sources) - base
            self.log(f"[{name}] 收集 {got} 个信源，提炼要点…", kind="ok")
            if got and self.auto_memory:
                await self._extract_facts(topic)

        results = await asyncio.gather(*(mini(a, i) for i, a in enumerate(agents)),
                                       return_exceptions=True)
        for r in results:
            if isinstance(r, Exception) and not isinstance(r, CancelledByUser):
                self.log(f"⚠️ 一个研究员异常退出：{r}", kind="warn")
        self.set_progress(0.75, "汇总裁判")
        self.log(f"全部研究员完成，共 {len(self.sources)} 个信源", kind="ok")
        await self._verify(0.82)

    # ---------------- 交叉验证 ----------------
    async def _verify(self, progress: float) -> None:
        await self.gate()
        self.check_cancel()
        self.set_progress(progress, "交叉验证信源")
        if len(self.sources) < 2:
            self.log("信源不足 2 个，跳过交叉验证", kind="warn")
            return
        self.log("⚖️ 正在交叉验证各信源的关键数据与结论…")
        data = await llm.ask_json(
            "你是事实核查员。对照编号信源，找出彼此矛盾的关键论断 JSON："
            "{\"conflicts\": [{\"topic\": \"冲突主题\", \"side_a\": \"观点A\", \"refs_a\": [编号], "
            "\"side_b\": \"观点B\", \"refs_b\": [编号], \"note\": \"差异原因\"}]}；没有输出 {\"conflicts\": []}；只输出 JSON。",
            f"研究问题：{self.query}\n\n编号信源：\n{self._sources_listing(cap=700)}{self.memory_prompt()}",
            temperature=0.1, fn="research")
        self._conflicts = [c for c in data.get("conflicts", []) if c.get("topic")]
        for c in self._conflicts:
            self.log(f"⚠️ 信源冲突：{c['topic']} —— {c.get('side_a', '')} vs {c.get('side_b', '')}", kind="warn")
        if not self._conflicts:
            self.log("未发现明显冲突的信源", kind="ok")
        self.emit("conflicts", {"conflicts_found": len(self._conflicts)})

    # ---------------- 撰写 ----------------
    async def _synthesize(self) -> dict:
        await self.gate()
        self.check_cancel()
        self.set_progress(0.9, "撰写研究报告")
        self.log("✍️ 正在汇总撰写研究报告…")
        conflicts = self._conflicts
        conflict_note = ""
        if conflicts:
            conflict_note = "\n\n已知信源冲突（报告必须用「信源冲突提示」段并列标注，不得擅自裁决）：\n" + "\n".join(
                f"- {c['topic']}: A={c.get('side_a')}（{c.get('refs_a')}） vs B={c.get('side_b')}（{c.get('refs_b')}）"
                for c in conflicts)
        words = self.profile["report_words"]
        listing, listing_note = self._report_sources_listing()
        report = await llm.achat([
            {"role": "system", "content":
                "你是深度研究撰写者。撰写一份严谨的中文研究报告（Markdown）：\n"
                "1. 第一行必须是 `# 报告标题`\n"
                "2. 结构：`## 一、结论先行`（3-6 条要点）→ 分章详述 → `## 信源冲突提示`（若有）→ `## 风险与局限`\n"
                "3. 每个关键论断句尾必须带上标引用 [n]，n 只能用编号信源列表里的编号；不得编造信源\n"
                "4. 数据注明出处编号；区分「事实」与「推断」\n"
                f"5. 篇幅目标 {words} 字；不要写参考文献列表（系统会自动附加）\n"
                "6. 语气客观克制，不用营销话术" + self.skills_prompt() + self.memory_prompt()},
            {"role": "user", "content":
                f"研究问题：{self.query}{self._workspace_prompt()}\n\n编号信源（共 {len(self.sources)} 个）{listing_note}：\n{listing}{conflict_note}"},
        ], temperature=0.4, fn="research", compact=True)
        title = self.query
        for line in report.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        self.set_progress(0.98, "生成参考文献")
        self.log(f"报告初稿完成：《{title}》，{len(self.sources)} 条引用", kind="ok")
        return {"md": report, "title": title}
