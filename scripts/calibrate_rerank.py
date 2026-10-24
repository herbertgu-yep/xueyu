"""校准重排的拒答阈值：用 80 道有答案题 + 10 道负样本找最佳分界点。

做法：每题走同一套召回（不做多查询改写），用当前重排模型给前 12 个候选打分，
取最高分。正负样本的最高分分布决定阈值，按 balanced accuracy 选点。

用法：
  XUEYU_DATA=<评测库> python -m scripts.calibrate_rerank
  XUEYU_DATA=<评测库> python -m scripts.calibrate_rerank --save   # 写进当前模型设置
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

QUESTIONS_PATH = pathlib.Path(__file__).resolve().parent.parent / "tests" / "data" / "eval_questions.json"
RECALL_K = 12


def _top_score(query: str) -> float | None:
    from app.services import rag_service, rerank

    fused, _stats = asyncio.run(
        rag_service._recall_and_fuse([query], None, RECALL_K, [1.0]))
    ranked = rerank.rerank(query, fused, limit=RECALL_K)
    if not ranked:
        return None
    score = ranked[0].get("rerank_score")
    return float(score) if score is not None else None


def _auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return 0.0
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="把最佳阈值写进 rerank 设置")
    args = ap.parse_args()

    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    from app.services import rerank, settings_store

    model = rerank.model_name()
    print(f"重排模型：{model}")

    pos, neg = [], []
    for q in data["questions"]:
        score = _top_score(q["q"])
        if score is not None:
            pos.append(score)
    for q in data.get("negatives", []):
        score = _top_score(q["q"])
        if score is not None:
            neg.append(score)
    if not pos or not neg:
        raise SystemExit("没有拿到有效分数：检查重排配置与评测库")

    cuts = sorted(set(pos + neg))
    best = None
    for t in cuts:
        tpr = sum(s >= t for s in pos) / len(pos)
        tnr = sum(s < t for s in neg) / len(neg)
        bal = (tpr + tnr) / 2
        if best is None or bal > best["balanced_acc"]:
            best = {"threshold": round(t, 4), "balanced_acc": round(bal, 3),
                    "positive_recall": round(tpr, 3), "negative_reject": round(tnr, 3)}

    print(f"正样本 {len(pos)}（中位 {sorted(pos)[len(pos) // 2]:.4f}） / "
          f"负样本 {len(neg)}（中位 {sorted(neg)[len(neg) // 2]:.4f}）")
    print(f"AUC {_auc(pos, neg):.3f}；最佳阈值 {best['threshold']}，"
          f"有答案保留 {best['positive_recall']:.1%}，无答案拒绝 {best['negative_reject']:.1%}")

    if args.save:
        cfg = settings_store.get("rerank")
        thresholds = dict(cfg.get("thresholds") or {})
        thresholds[model] = best["threshold"]
        cfg["thresholds"] = thresholds
        settings_store.put("rerank", cfg, merge=False)
        rerank.reset_api_state()
        print(f"已写入 rerank.thresholds[{model}] = {best['threshold']}")


if __name__ == "__main__":
    main()