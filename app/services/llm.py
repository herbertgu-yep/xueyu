"""LLM 调用收口层 —— 所有模型调用统一走 LiteLLM。

- 支持 OpenAI / Claude / DeepSeek / 通义 / GLM / Kimi / Ollama / 任意 OpenAI 兼容端点
- 未配置模型时抛 LLMConfigError（API 层转成友好提示，绝不返回假数据）
- ask_json：让模型输出 JSON 并做健壮解析（剥代码栏、截取平衡大括号）
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from typing import Any, AsyncIterator, Iterable, Optional

import litellm

from app.services import settings_store

litellm.suppress_debug_info = True


class LLMConfigError(RuntimeError):
    """模型未配置或配置无效。"""


class LLMCallError(RuntimeError):
    """模型调用失败（网络 / 鉴权 / 限流）。"""


_FN_NAMES = {"note": "笔记 AI", "doc": "文档分析", "research": "深度研究"}

# ---------------------------------------------------------------- 语义缓存（Semantic Cache）

_semcache_lock = threading.Lock()
_semcache: list[dict] = []          # [{query_emb, response, fn, ts}]
_SEMCACHE_MAX = 128
_SEMCACHE_SIM = 0.94                # 余弦相似度命中阈值


def _cache_enabled() -> bool:
    try:
        from app.services import settings_store

        return bool(settings_store.get("research").get("semantic_cache", True))
    except Exception:
        return False


def _query_embedding(text: str) -> Optional[list[float]]:
    """查询向量：优先用 RAG 的本地 embedding，失败返回 None。"""
    try:
        from app.services import rag_service

        emb = rag_service._get_embed_model()
        return emb.get_text_embedding(text[:512])
    except Exception:
        return None


@staticmethod
def _cosine(a: list[float], b: list[float]) -> float:
    import math

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def semantic_cache_get(query: str) -> Optional[str]:
    """语义缓存命中：相似度 ≥ 阈值则直接返回历史回答。"""
    if not _cache_enabled():
        return None
    qemb = _query_embedding(query)
    if not qemb:
        return None
    with _semcache_lock:
        best, best_sim = None, 0.0
        for item in _semcache:
            sim = _cosine(qemb, item["emb"])
            if sim > best_sim:
                best, best_sim = item, sim
    if best and best_sim >= _SEMCACHE_SIM:
        return best["response"]
    return None


def semantic_cache_put(query: str, response: str) -> None:
    if not _cache_enabled() or not response:
        return
    qemb = _query_embedding(query)
    if not qemb:
        return
    with _semcache_lock:
        _semcache.append({"emb": qemb, "response": response, "ts": time.time()})
        if len(_semcache) > _SEMCACHE_MAX:
            _semcache.pop(0)

# ---------------------------------------------------------------- 上下文窗口与压缩

_CONTEXT_FALLBACK = 8192


def model_context_tokens(cfg: Optional[dict] = None) -> int:
    """查模型上下文窗口：litellm model_cost 优先，失败回退 8k。"""
    try:
        cfg = cfg or settings_store.llm_config()
        name = (cfg.get("model") or "").strip()
        info = litellm.model_cost.get(name) or {}
        for key in ("max_input_tokens", "max_tokens"):
            v = info.get(key)
            if v and v > 1000:
                return int(v)
    except Exception:
        pass
    return _CONTEXT_FALLBACK


def estimate_tokens(text: str) -> int:
    """粗估 token：中英文混合按 max(字数/1.6, 词数) 保守取高。"""
    if not text:
        return 0
    return max(int(len(text) / 1.6), len(text.split()))


def estimate_messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(str(m.get("content", ""))) + 4 for m in messages)


async def compact_messages(messages: list[dict], *, fn: Optional[str] = None,
                           keep_recent: int = 6, budget_ratio: float = 0.45) -> list[dict]:
    """超预算时把较早的对话压缩成一条摘要（真实 LLM 调用）。

    - 预算 = 模型上下文 × budget_ratio（给输出留空间）
    - 保留 system 消息 + 最近 keep_recent 条；其余压缩成 [对话摘要]
    - 未超预算原样返回
    """
    ctx = model_context_tokens(fn and settings_store.llm_config(fn))
    budget = int(ctx * budget_ratio)
    if estimate_messages_tokens(messages) <= budget:
        return messages
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    recent = rest[-keep_recent:]
    older = rest[:-keep_recent]
    if not older:
        return messages
    transcript = "\n".join(f"[{m.get('role')}] {str(m.get('content', ''))[:1500]}" for m in older)
    summary = await achat([
        {"role": "system", "content":
            "把以下对话/材料压缩成一份紧凑的要点摘要（保留：结论、数字、用户偏好、未完成事项），"
            "500 字以内，直接输出摘要本身。"},
        {"role": "user", "content": transcript[:30000]},
    ], temperature=0.1, fn=fn)
    compressed = {"role": "system", "content": f"[早前对话摘要，已压缩以控制上下文长度]\n{summary}"}
    result = system + [compressed] + recent
    # 压缩后仍超预算 → 只留最近消息硬截断
    while len(result) > 2 and estimate_messages_tokens(result) > budget:
        idx = 1 if result[0].get("role") == "system" else 0
        if len(result) - 1 <= keep_recent:
            break
        result.pop(idx)
    return result


def _unconfigured_msg(fn: Optional[str]) -> str:
    name = _FN_NAMES.get(fn or "", "")
    where = f"（{name}当前使用的模型）" if name else ""
    return f"尚未配置 AI 模型{where}，请到「设置」页添加服务商并填入 API Key"


def _model_string(cfg: dict) -> str:
    preset = settings_store.PROVIDERS.get(cfg.get("provider", ""), settings_store.PROVIDERS["custom"])
    model = (cfg.get("model") or "").strip()
    if not model:
        raise LLMConfigError("模型名称未配置，请到「设置」补全")
    prefix = preset["prefix"]
    if model.startswith(f"{prefix}/"):
        return model
    return f"{prefix}/{model}"


def _call_kwargs(cfg: dict) -> dict:
    preset = settings_store.PROVIDERS.get(cfg.get("provider", ""), settings_store.PROVIDERS["custom"])
    kwargs: dict[str, Any] = {"temperature": float(cfg.get("temperature", 0.4))}
    base_url = (cfg.get("base_url") or "").strip() or preset.get("base_url", "")
    if base_url:
        kwargs["base_url"] = base_url
    api_key = (cfg.get("api_key") or "").strip()
    if cfg.get("provider") == "ollama":
        kwargs["api_key"] = api_key or "ollama"  # litellm 对 ollama 仍要求非空
    elif api_key:
        kwargs["api_key"] = api_key
    return kwargs


def acompletion(messages: list[dict], *, temperature: Optional[float] = None, max_tokens: Optional[int] = None, fn: Optional[str] = None) -> str:
    """同步上下文里调用；返回文本。阻塞放线程池。fn = 功能路由（note/doc/research）。"""
    cfg = settings_store.llm_config(fn)
    if not settings_store.is_configured():
        raise LLMConfigError(_unconfigured_msg(fn))
    kwargs: dict[str, Any] = {
        "model": _model_string(cfg),
        "messages": messages,
        **_call_kwargs(cfg),
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    try:
        resp = litellm.completion(**kwargs)
    except Exception as e:  # noqa: BLE001 统一转成可读错误
        raise LLMCallError(f"模型调用失败：{_brief(e)}") from e
    return (resp.choices[0].message.content or "").strip()


async def achat(
    messages: list[dict],
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    fn: Optional[str] = None,
    compact: bool = False,
) -> str:
    """异步调用（事件循环内使用）。compact=True 时超上下文自动压缩。"""
    cfg = settings_store.llm_config(fn)
    if compact:
        try:
            messages = await compact_messages(messages, fn=fn)
        except Exception:
            pass
    if not settings_store.is_configured():
        raise LLMConfigError(_unconfigured_msg(fn))
    kwargs: dict[str, Any] = {
        "model": _model_string(cfg),
        "messages": messages,
        **_call_kwargs(cfg),
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    try:
        resp = await litellm.acompletion(**kwargs)
    except Exception as e:  # noqa: BLE001
        raise LLMCallError(f"模型调用失败：{_brief(e)}") from e
    return (resp.choices[0].message.content or "").strip()


async def astream(messages: list[dict], *, temperature: Optional[float] = None, fn: Optional[str] = None) -> AsyncIterator[str]:
    """流式对话（AI 面板自由对话使用）。"""
    cfg = settings_store.llm_config(fn)
    if not settings_store.is_configured():
        raise LLMConfigError(_unconfigured_msg(fn))
    kwargs: dict[str, Any] = {"model": _model_string(cfg), "messages": messages, **_call_kwargs(cfg)}
    if temperature is not None:
        kwargs["temperature"] = temperature
    try:
        resp = await litellm.acompletion(**kwargs, stream=True)
        async for chunk in resp:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
    except Exception as e:  # noqa: BLE001
        raise LLMCallError(f"模型调用失败：{_brief(e)}") from e


async def achat_raw(messages: list[dict], *, tools: Optional[list[dict]] = None,
                    fn: Optional[str] = None) -> dict:
    """Function Calling 原始响应：返回 {"message": {...}, "tool_calls": [...]}。

    messages 需含 tool 结果时用 {"role": "tool", "tool_call_id", "content"}。
    """
    cfg = settings_store.llm_config(fn)
    if not settings_store.is_configured(fn):
        raise LLMConfigError(_unconfigured_msg(fn))
    kwargs: dict[str, Any] = {"model": _model_string(cfg), "messages": messages, **_call_kwargs(cfg)}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    try:
        resp = await litellm.acompletion(**kwargs)
    except Exception as e:  # noqa: BLE001
        raise LLMCallError(f"模型调用失败：{_brief(e)}") from e
    msg = resp.choices[0].message
    tool_calls = None
    if getattr(msg, "tool_calls", None):
        tool_calls = [{"id": tc.id, "type": "function",
                       "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                      for tc in msg.tool_calls]
    return {"message": {"role": "assistant", "content": msg.content or ""}, "tool_calls": tool_calls}


async def achat_batch(prompts: Iterable[list[dict]], *, temperature: Optional[float] = None) -> list[str]:
    """并发跑多个 prompt（如批量出题）。"""
    tasks = [achat(p, temperature=temperature) for p in prompts]
    return await asyncio.gather(*tasks)


def ask_json_sync(system: str, user: str, *, temperature: float = 0.2, max_tokens: Optional[int] = None, fn: Optional[str] = None) -> Any:
    """同步上下文的 JSON 问答（内部用 litellm.completion 阻塞调用）。"""
    cfg = settings_store.llm_config(fn)
    if not settings_store.is_configured(fn):
        raise LLMConfigError(_unconfigured_msg(fn))
    kwargs: dict[str, Any] = {
        "model": _model_string(cfg),
        "messages": [
            {"role": "system", "content": system + "\n\n只输出合法 JSON，不要输出任何解释、注释或 Markdown 代码栏标记。"},
            {"role": "user", "content": user},
        ],
        **_call_kwargs(cfg),
        "temperature": temperature,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    try:
        resp = litellm.completion(**kwargs)
    except Exception as e:  # noqa: BLE001
        raise LLMCallError(f"模型调用失败：{_brief(e)}") from e
    text = (resp.choices[0].message.content or "").strip()
    try:
        return parse_json(text)
    except LLMCallError as e:
        raise LLMCallError(str(e) + f"｜模型原始输出片段：{text[:200]}") from e


async def ask_json(system: str, user: str, *, temperature: float = 0.2, max_tokens: Optional[int] = None, fn: Optional[str] = None) -> Any:
    """要求模型返回 JSON；做健壮解析。失败抛 LLMCallError。"""
    text = await achat(
        [{"role": "system", "content": system + "\n\n只输出合法 JSON，不要输出任何解释、注释或 Markdown 代码栏标记。"},
         {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
        fn=fn,
    )
    try:
        return parse_json(text)
    except LLMCallError as e:
        raise LLMCallError(str(e) + f"｜模型原始输出片段：{text[:200]}") from e


def parse_json(text: str) -> Any:
    text = text.strip()
    # 剥 ```json ... ```
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 截取首个平衡的 {} 或 []
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise LLMCallError(f"模型未返回可解析的 JSON：{text[:200]}")


def _brief(e: Exception) -> str:
    msg = str(e)
    if "api key" in msg.lower() or "401" in msg:
        return "API Key 无效或未填写"
    if "429" in msg:
        return "触发限流，请稍后重试"
    if "Connection" in msg or "timeout" in msg.lower() or "connect" in msg.lower():
        return "网络连接失败（本地模型请确认 Ollama 已启动）"
    return msg[:300]
