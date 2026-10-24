"""DashScope 原生 embedding / rerank 协议适配测试（不联网）。"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import rerank  # noqa: E402
from app.services.rag_service import _make_dashscope_embedding  # noqa: E402


class FakeResp:
    def __init__(self, payload: dict, status: int = 200):
        self.status_code = status
        self._payload = payload
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


CFG = {
    "mode": "api",
    "api_provider": "dashscope",
    "api_base_url": "https://example.test/api/v1",
    "api_key": "sk-test",
    "api_model": "qwen3.7-text-rerank",
}


def test_rerank_api_orders_by_relevance(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, json))
        return FakeResp({"output": {"results": [
            {"index": 1, "relevance_score": 0.12},
            {"index": 0, "relevance_score": 0.91},
        ]}})

    monkeypatch.setattr(httpx, "post", fake_post)
    rerank.reset_api_state()
    out = rerank._rerank_api("查询", [{"text": "相关"}, {"text": "无关"}], CFG, None, 1200)

    assert [c["text"] for c in out] == ["相关", "无关"]
    assert out[0]["rerank_score"] == 0.91
    assert calls[0][0] == "https://example.test/api/v1/services/rerank/text-rerank/text-rerank"
    assert calls[0][1]["parameters"]["top_n"] == 2


def test_rerank_api_failure_returns_none(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResp({"message": "invalid key"}, status=401)

    monkeypatch.setattr(httpx, "post", fake_post)
    rerank.reset_api_state()
    assert rerank._rerank_api("查询", [{"text": "a"}], CFG, None, 1200) is None
    assert "401" in (rerank._api_failed or "")


def test_dashscope_embedding_batches(monkeypatch):
    batches = []

    def fake_post(url, headers=None, json=None, timeout=None):
        texts = json["input"]["texts"]
        batches.append(len(texts))
        return FakeResp({"output": {"embeddings": [{"embedding": [0.1, 0.2]} for _ in texts]}})

    monkeypatch.setattr(httpx, "post", fake_post)
    emb = _make_dashscope_embedding({**CFG, "api_model": "qwen3.7-text-embedding-flash"})
    vecs = emb.get_text_embedding_batch([f"t{i}" for i in range(20)])

    assert len(vecs) == 20 and len(vecs[0]) == 2
    assert batches == [16, 4]
