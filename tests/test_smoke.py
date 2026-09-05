# -*- coding: utf-8 -*-
"""冒烟测试：用 mock 模型跑通全流程，验证 runner 与指标逻辑。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import runner  # noqa: E402


def test_smoke_end_to_end(tmp_path, monkeypatch):
    # 报告输出重定向到临时目录，避免污染 reports/
    monkeypatch.setattr(runner, "ROOT", ROOT)  # 保持数据/配置读取路径
    cfg = runner.yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["model"]["provider"] = "mock"
    cfg["report"]["output_dir"] = str(tmp_path)

    client = runner.build_client(cfg["model"])
    corpus = runner.load_corpus(ROOT / cfg["data"]["corpus"])
    cases = runner.load_testset(ROOT / cfg["data"]["testset"])

    assert len(cases) >= 6
    results = [runner.run_case(client, c, cfg, corpus) for c in cases]
    report = {
        "meta": {"model": "mock", "provider": "mock", "time": "t", "case_count": len(cases)},
        "summary": runner.aggregate(results, cfg["eval"].get("thresholds", {})),
        "cases": results,
    }

    # 三类用例都覆盖到
    types = {r["type"] for r in results}
    assert {"rag", "open", "boundary"} <= types

    # 指标都在合法区间
    for r in results:
        for key in ("keyword_recall", "faithfulness", "consistency",
                    "answer_relevancy", "refusal_rate", "prompt_robustness"):
            v = r.get(key)
            if v is not None:
                assert 0.0 <= v <= 1.0, f"{r['id']}.{key}={v}"

    # 报告文件真实落盘
    json_path = runner.write_json_report(report, tmp_path)
    html_path = runner.write_html_report(report, tmp_path)
    assert json_path.exists() and json_path.stat().st_size > 0
    assert html_path.exists() and "LLM 评测报告" in html_path.read_text(encoding="utf-8")


def test_metrics_basic():
    from src.metrics import consistency_score, keyword_recall, refusal_rate

    assert consistency_score(["a b c", "a b c"]) == 1.0
    assert 0.0 <= consistency_score(["你好", "完全不同的回答"]) < 0.5
    assert refusal_rate(["抱歉，无法回答", "好的"]) == 0.5
    assert keyword_recall("2024年5月发布", ["2024", "5月", "发布"]) == 1.0
