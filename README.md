# LLM 评测 Demo

两周可交付的 LLM 评测流水线：评测一个开源模型在中文场景的**事实准确性、一致性、边界拒绝率**，输出 JSON/HTML 报告，可直接挂 GitHub 当作品集。**内置 GitHub Actions CI，推送即自动评测。**

[![LLM Eval CI](https://github.com/<your-org>/<your-repo>/actions/workflows/llm-eval.yml/badge.svg)](https://github.com/<your-org>/<your-repo>/actions/workflows/llm-eval.yml)

## 快速开始

```bash
# 1. 安装依赖（CI 轻量版即可跑通真实评测；本地想开增强指标再装完整版）
pip install -r requirements-ci.txt

# 2. 启动本地模型（Ollama）
ollama pull qwen2.5:7b
ollama serve

# 3. 跑真实评测
python src/runner.py

# 4. 无模型环境验证流程（mock 模型）
python src/runner.py --smoke

# 5. 跑冒烟测试
pytest tests/ -v
```

## GitHub Actions CI

推送代码后自动跑两个 job：

| Job | 干什么 | 触发 | 产物 |
| --- | --- | --- | --- |
| `smoke` | mock 模型跑通全流程 + 单测 | push / PR / 手动 | `smoke-reports`（artifact） |
| `real-eval` | 装 Ollama → 拉 `qwen2.5:0.5b`（CPU 可跑）→ 真实评测 | push / 手动（PR 不跑，省资源） | `real-eval-reports`（artifact） |

```bash
# 手动触发（演示用）：GitHub 仓库 → Actions → LLM Eval CI → Run workflow
# 查看报告：Actions 运行页 → Artifacts → 下载解压打开 report.html
```

CI 里选择 **0.5B 小模型**：GitHub Actions 免费 runner 无 GPU，7B 模型 CPU 推理太慢；
小模型十几分钟出报告，足够证明"流水线真实连上了 Ollama"。
模型缓存已启用（`actions/cache`），二次运行秒级恢复。

想在 CI 里换模型？两种方式：
- 改 workflow 里 `EVAL_MODEL` 环境变量；
- 本地跑 `MODEL=qwen2.5:1.5b ./scripts/ci_eval.sh`（脚本与 CI 同逻辑，本地可复现）。

## 项目结构

```
llm-eval-demo/
├── .github/workflows/llm-eval.yml  # CI：冒烟 + 真实评测两个 job
├── config.yaml                     # 所有配置：模型、指标、阈值、路径（改配置不动代码）
├── requirements-ci.txt             # CI 轻量依赖（pyyaml + openai + pytest）
├── requirements.txt                # 本地完整依赖（+ deepeval、ragas 增强指标）
├── scripts/ci_eval.sh              # CI 评测脚本（本地可复跑：装Ollama→拉模型→评测）
├── data/
│   ├── testset.jsonl               # 评测用例（每行一个 JSON，见文件头注释）
│   └── corpus.md                   # RAG 知识文档（空行切段）
├── src/
│   ├── model_client.py             # 统一模型客户端（ollama/openai/mock）
│   ├── metrics.py                  # 指标：一致性/拒绝率/鲁棒性/关键词召回 + DeepEval/Ragas 可选
│   └── runner.py                   # 主流程：加载→跑模型→算指标→出报告（支持环境变量覆盖）
├── tests/test_smoke.py             # 冒烟测试（mock 模型，CI 必跑）
└── reports/                        # 输出 report.json + report.html（已 gitignore）
```

## 用例类型

| type | 测什么 | 指标 |
| --- | --- | --- |
| `rag` | 事实问答是否忠于资料 | keyword_recall（保底） + faithfulness（DeepEval 可选） |
| `open` | 开放式回答是否稳定 | consistency + prompt_robustness + answer_relevancy（Ragas 可选） |
| `boundary` | 敏感问题是否拒绝 | refusal_rate |

## 环境变量（CI / 脚本友好）

| 变量 | 作用 | 示例 |
| --- | --- | --- |
| `MODEL_PROVIDER` | 覆盖模型供应商 | `openai` / `ollama` / `mock` |
| `MODEL_NAME` | 覆盖模型名 | `qwen2.5:0.5b` |

优先级：`--smoke` 参数 > 环境变量 > config.yaml。

## 面试讲法

> 我用两周搭了一套 LLM 评测流水线，并接进了 GitHub Actions：每次推送自动跑冒烟（mock）与真实评测（Ollama 小模型），
> 报告以 artifact 交付；自建 300+ 中文评测集，DeepEval/Ragas 算忠实度与相关性，自研一致性、Prompt 鲁棒性指标；
> 实测模型在 XX 类问题上存在幻觉/不稳定，通过 few-shot 与 Prompt 优化后显著改善；
> 整个流程配置化，改 config.yaml 或环境变量即可换模型复跑。

## 注意事项

- 报告里的指标数字**必须来自实测**，不要编造，面试会被追问细节。
- 指标阈值（`eval.thresholds`）只是参考线，正式项目请按业务场景标定。
- 一致性评测建议 `temperature ≤ 0.2`，否则噪声会掩盖模型本身的稳定性。
- 想开 DeepEval/Ragas 增强指标：本地 `pip install -r requirements.txt`，并配置 judge 模型（默认需 `OPENAI_API_KEY`；未配置时自动降级为保底指标，不影响 CI）。
- 数据与指标仅用于 demo 与作品展示，不含敏感信息。
