# 检索质量测评报告

> 所有数字均为本机实测，问题集、语料清单与脚本随仓库提供，可复现。无估算，无外推。

## 1. 目的

回答两个问题：

1. 混合检索（BM25 + 向量 + RRF 融合 + Multi-Query 扩展 + 重排）相比单路检索，提升多少？
2. 重排用本地 cross-encoder 还是再调一次大模型？质量与代价分别是多少？

## 2. 测评设置

| 项 | 值 |
| --- | --- |
| 语料 | 20 篇 arXiv 公开论文（Transformer / RAG / Agent / 扩散模型 / CoT 等主题），共 1436 个分块 |
| 问题集 | 21 题：术语型 10 题（含专业词）+ 口语型 11 题（大白话），每题标注正确答案文档 |
| 命中判定 | 标准答案文档的任一分块进入检索结果前 3 名记为命中（规则在跑数前确定） |
| 分块 | 向量与 BM25 共用 `split_chunks()`：480 字、重叠 60 字，每个分块带 `chunk_id` |
| Embedding | BAAI/bge-small-zh-v1.5（本地模式） |
| 多查询改写 | DeepSeek（OpenAI 兼容接口），temperature 0.2 |
| 重排 A | `BAAI/bge-reranker-v2-m3` cross-encoder，本地打分，精排 RRF 前 12 个候选 |
| 重排 B | LLM listwise 排序，同样只看前 12 个候选（每候选 150 字） |
| 对照模式 | 向量单路 / BM25 单路 / 混合 RRF / 混合全开（CE）/ 混合全开（LLM） |

问题集见 `tests/data/eval_questions.json`，脚本见 `scripts/eval_retrieval.py` 与 `scripts/eval_rerank.py`。
逐题数据：`docs/evaluation-data.json`（CE 全链路）、`docs/evaluation-data-llm-rerank.json`（LLM 全链路）、
`docs/rerank-comparison.json`（固定候选池的重排对比）。
语料由 `scripts/fetch_eval_corpus.py` 下载，下载后逐篇核对首页关键词，确保文件内容与题目标注的论文一致。

## 3. 结果

前三档（不依赖 LLM）各跑一次，结果稳定：

| 模式 | top-3 命中率 | 术语型 | 口语型 | p50 延迟 |
| --- | --- | --- | --- | --- |
| 向量单路 | 38.1% | 40.0% | 36.4% | 8ms |
| BM25 单路 | 38.1% | 50.0% | 27.3% | 4ms |
| 混合 RRF | 47.6% | 50.0% | 45.5% | 14ms |

全开模式包含一次 LLM 改写，改写结果每次略有不同，因此重复运行取区间：

| 模式 | top-3 命中率（多次运行） | 术语型 | 口语型 | p50 延迟 |
| --- | --- | --- | --- | --- |
| 混合全开（CE） | 85.7%–100%（6 次，均值 92.9%，中位数 92.9%） | 90%–100% | 72.7%–100% | 3.9–4.8s |
| 混合全开（LLM） | 90.5%–95.2%（4 次，均值 94.0%，中位数 95.2%） | 100% | 81.8%–90.9% | 6.4–8.2s |

为了排除改写波动、单独比较两个重排器，用同一批候选池固定重排输入
（`scripts/eval_rerank.py`，改写结果缓存复用）：

| 重排器 | top-3 命中 | 单题重排耗时 |
| --- | --- | --- |
| cross-encoder（前 12 候选） | 20/21 | 2.0s |
| LLM listwise（前 12 候选） | 21/21 | 5.7s |

## 4. 结论

1. **单路召回 38.1%**：20 篇主题相近的论文干扰项密集。BM25 对术语型有效（50%）、对口语型只有 27.3%；
   向量单路反过来，口语型 36.4%。两路互补，RRF 融合后到 47.6%。
2. **Multi-Query + 重排把命中率拉到 90% 以上**：改写扩大召回覆盖，重排把正确文档顶到前列。
   术语型 10/10 全中。
3. **cross-encoder 与 LLM 重排质量接近**：固定候选池下 20/21 对 21/21，差一题；
   全链路重复运行时 CE 均值 92.9%、LLM 均值 94.0%，差距在一个问题以内。
   CE 单题 2.0s、不产生 API 费用，LLM 5.7s 且每次都要计费，
   因此默认用 CE，LLM 重排保留为模型不可用时的回退路径（也可用 `XUEYU_RERANK_ENGINE=llm` 显式切换）。
4. **精排候选数不需要很多**：同一批召回结果上，CE 只排前 12 个是 20/21、2.0s；
   放到前 40 个仍是 20/21 左右，耗时涨到 10s。所以产品里截断在前 12。
5. **延迟主要花在改写，不在重排**：全开模式 p50 约 4–8s，其中一次 Multi-Query 改写调用占大头。
   默认产品路径是混合 RRF（14ms），深度检索才开全开模式。
6. **漏检集中在个别口语题**：CE 固定候选池下漏 toolformer 口语题；全链路运行时，
   LLM 的漏检题在不同运行间变化（toolformer / ddpm）。21 题里差一题，没有继续调参去拟合问题集。

## 5. 复现

```bash
# 1. 下载并校验语料（20 篇，逐篇核对首页关键词）
python -m scripts.fetch_eval_corpus --out ~/.xueyu_eval_docs

# 2. 建立干净的评测库并建索引
python -m scripts.fetch_eval_corpus --out ~/.xueyu_eval_docs \
    --ingest --data ~/.xueyu_eval_clean

# 3. 配置 research 模型（多查询改写用），然后跑全链路测评
XUEYU_DATA=~/.xueyu_eval_clean python -m scripts.eval_retrieval

# 4. 对比 LLM listwise 重排
XUEYU_DATA=~/.xueyu_eval_clean XUEYU_RERANK_ENGINE=llm python -m scripts.eval_retrieval

# 5. 固定候选池，单独比较两个重排器
XUEYU_DATA=~/.xueyu_eval_clean python -m scripts.eval_rerank --ce-limits 12
```

评测脚本会自检：未配置模型直接报错退出；任何一题的改写或重排失败也会报错退出，
避免把「降级成纯 RRF」的数据当成全开模式的结果。运行时打印重排方式与实际模型名。

## 6. 语料校验

评测语料按文件名标注标准答案，一旦 PDF 与文件名错配，命中率就会失真且很难从结果里看出来。
`fetch_eval_corpus.py` 为每篇论文记录一个必须出现在正文前两页的关键词，下载后逐篇校验，
不通过直接中止。这一步是评测可信度的前提。
