# -*- coding: utf-8 -*-
"""评测主流程：读配置 → 加载用例 → 跑模型 → 算指标 → 输出 JSON/HTML 报告。

用法：
    python runner.py                     # 按 config.yaml 真实评测
    python runner.py --smoke             # 用 mock 模型快速验证流程（无模型环境）
    python runner.py --config my.yaml    # 指定配置

用例格式（data/testset.jsonl，每行一个 JSON）：
    {"id": "r1", "type": "rag",      "question": "...", "gold_keywords": ["...", "..."], "doc": "文档段落" }
    {"id": "o1", "type": "open",     "question": "..."}
    {"id": "b1", "type": "boundary", "question": "..."}
    type 说明：
      rag      RAG 事实问答：关键词召回 +（可选）DeepEval faithfulness
      open     开放式问答：一致性（同问 N 次）+（可选）Ragas answer_relevancy
      boundary 边界/敏感问题：期望拒绝，算拒绝率
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # 支持直接 python3 src/runner.py 运行
    sys.path.insert(0, str(ROOT))

from src.model_client import build_client
from src.metrics import (
    consistency_score,
    deepeval_faithfulness,
    keyword_recall,
    prompt_robustness,
    ragas_answer_relevancy,
    refusal_rate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("runner")


# ---------------------------------------------------------------- 数据层

def load_testset(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            cases.append(json.loads(line))
    return cases


def load_corpus(path: Path) -> list[str]:
    """把知识文档按空行切成段落，供简单检索使用。"""
    text = path.read_text(encoding="utf-8")
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def simple_retrieve(query: str, paragraphs: list[str], top_k: int) -> list[str]:
    """零依赖检索：按 query 与段落的字符重叠度取 top-k。demo 够用，正式可换向量库。"""
    q_chars = set(query)
    scored = []
    for p in paragraphs:
        inter = len(q_chars & set(p))
        union = len(q_chars | set(p)) or 1
        scored.append((inter / union, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]


# ---------------------------------------------------------------- 评测逻辑

def run_case(client, case: dict, cfg: dict, corpus: list[str]) -> dict:
    ctype = case.get("type", "open")
    question = case["question"]
    result = {"id": case.get("id", "?"), "type": ctype, "question": question}

    if ctype == "rag":
        context = case.get("doc") or " ".join(
            simple_retrieve(question, corpus, cfg["eval"]["top_k"])
        )
        answer = client.chat(
            [{"role": "system", "content": "只依据给定资料回答，不要编造。"},
             {"role": "user", "content": f"资料：{context}\n\n问题：{question}"}]
        )
        result["answer"] = answer
        result["keyword_recall"] = keyword_recall(answer, case.get("gold_keywords", []))
        result["faithfulness"] = deepeval_faithfulness(answer, context)

    elif ctype == "boundary":
        n = cfg["eval"]["consistency_repeats"]
        answers = [
            client.chat([{"role": "user", "content": question}]) for _ in range(n)
        ]
        result["answers"] = answers
        result["refusal_rate"] = refusal_rate(answers)

    else:  # open
        n = cfg["eval"]["consistency_repeats"]
        answers = [
            client.chat([{"role": "user", "content": question}]) for _ in range(n)
        ]
        result["answers"] = answers
        result["consistency"] = consistency_score(answers)
        result["answer_relevancy"] = ragas_answer_relevancy(question, answers[0])

        # Prompt 鲁棒性：换 2 种措辞各问一次，与基准回答比相似度
        variants = [
            f"请用中文回答：{question}",
            f"换种方式说：{question}？",
        ]
        base = answers[0]
        var_answers = [client.chat([{"role": "user", "content": v}]) for v in variants]
        result["prompt_robustness"] = prompt_robustness(base, var_answers)

    return result


def aggregate(results: list[dict], thresholds: dict) -> dict:
    """按用例类型聚合指标 + 判定是否达标。"""
    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)

    summary = {}
    for t, items in by_type.items():
        block = {"cases": len(items)}
        for key in (
            "keyword_recall", "faithfulness", "consistency",
            "answer_relevancy", "refusal_rate", "prompt_robustness",
        ):
            vals = [r[key] for r in items if r.get(key) is not None]
            if vals:
                avg = round(sum(vals) / len(vals), 4)
                block[key] = avg
                th = thresholds.get(key)
                block[key + "_pass"] = avg >= th if th is not None else None
        summary[t] = block
    return summary


# ---------------------------------------------------------------- 报告

def write_json_report(report: dict, out_dir: Path) -> Path:
    path = out_dir / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_html_report(report: dict, out_dir: Path) -> Path:
    """零依赖 HTML 报告：表格 + 纯 CSS 条形图。"""
    rows = []
    for r in report["cases"]:
        cells = [f"<td>{r['id']}</td><td>{r['type']}</td><td>{r['question'][:30]}</td>"]
        for key in ("keyword_recall", "faithfulness", "consistency",
                    "answer_relevancy", "refusal_rate", "prompt_robustness"):
            v = r.get(key)
            if v is None:
                cells.append("<td>—</td>")
            else:
                cells.append(f"<td>{v:.2f}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    bars = []
    for t, block in report["summary"].items():
        for key, val in block.items():
            if isinstance(val, float):
                bars.append(
                    f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>"
                    f"<span style='width:120px;flex:none;font-size:12px;'>{t}.{key}</span>"
                    f"<div style='flex:1;background:#eee;border-radius:6px;overflow:hidden;'>"
                    f"<div style='width:{val * 100:.0f}%;height:16px;background:#94D8C3;'></div></div>"
                    f"<span style='width:44px;flex:none;font-size:12px;'>{val:.2f}</span></div>"
                )

    html = f"""<!DOCTYPE html>
<html lang="zh"><meta charset="utf-8"><title>LLM 评测报告</title>
<body style="font-family:'PingFang SC',Roboto,Arial,sans-serif;margin:24px;color:#1A1B1C;">
<h2>LLM 评测报告</h2>
<p style="color:#666;">模型：{report['meta']['model']}（{report['meta']['provider']}）· 时间：{report['meta']['time']} · 用例数：{report['meta']['case_count']}</p>
<h3>指标总览（0~1，越高越好；拒绝率仅看 boundary）</h3>
<div style="max-width:560px;">{''.join(bars)}</div>
<h3>逐条明细</h3>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;font-size:13px;">
<tr><th>id</th><th>type</th><th>question</th><th>recall</th><th>faith</th><th>consist</th><th>relev</th><th>refuse</th><th>robust</th></tr>
{''.join(rows)}
</table>
</body></html>"""
    path = out_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


# ---------------------------------------------------------------- 入口

def _env_override(cfg: dict) -> dict:
    """环境变量覆盖配置（CI 友好）：MODEL_PROVIDER / MODEL_NAME 优先级高于 config.yaml。"""
    provider = os.environ.get("MODEL_PROVIDER")
    model = os.environ.get("MODEL_NAME")
    if provider:
        cfg["model"]["provider"] = provider
        logger.info("环境变量覆盖 provider=%s", provider)
    if model:
        cfg["model"]["model"] = model
        logger.info("环境变量覆盖 model=%s", model)
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM 评测流水线")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--smoke", action="store_true", help="用 mock 模型验证流程")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.smoke:
        cfg["model"]["provider"] = "mock"
    cfg = _env_override(cfg)

    client = build_client(cfg["model"])
    corpus = load_corpus(ROOT / cfg["data"]["corpus"])
    cases = load_testset(ROOT / cfg["data"]["testset"])
    if not cases:
        logger.error("测试集为空：%s", cfg["data"]["testset"])
        return 1
    logger.info("开始评测：模型=%s 用例=%d", client.model if hasattr(client, "model") else "mock", len(cases))

    results = [run_case(client, c, cfg, corpus) for c in cases]
    report = {
        "meta": {
            "provider": cfg["model"]["provider"],
            "model": cfg["model"]["model"],
            "time": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "case_count": len(cases),
        },
        "summary": aggregate(results, cfg["eval"].get("thresholds", {})),
        "cases": results,
    }

    out_dir = ROOT / cfg["report"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json_report(report, out_dir)
    html_path = write_html_report(report, out_dir)
    logger.info("报告已生成：%s / %s", json_path, html_path)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
