#!/usr/bin/env bash
# CI 评测脚本：安装依赖 → 准备 Ollama（未装则装、未启动则启动）→ 拉小模型 → 真实评测
# 用法：
#   ./scripts/ci_eval.sh                      # 默认模型 qwen2.5:0.5b
#   MODEL=qwen2.5:1.5b ./scripts/ci_eval.sh   # 换模型
# 本地已装 Ollama 时自动复用，不重复安装。
set -euo pipefail

MODEL="${MODEL:-qwen2.5:0.5b}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> [1/4] 安装 Python 依赖（轻量版）"
pip install -q -r requirements-ci.txt

echo "==> [2/4] 准备 Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  echo "    Ollama 未安装，执行官方安装脚本…"
  curl -fsSL https://ollama.com/install.sh | sh
fi

if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "    Ollama 未运行，后台启动…"
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  for _ in $(seq 1 30); do
    curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
fi
curl -sf http://localhost:11434/api/tags >/dev/null || {
  echo "    Ollama 启动失败，日志："; tail -20 /tmp/ollama.log; exit 1;
}

echo "==> [3/4] 拉取评测模型：$MODEL"
ollama pull "$MODEL"

echo "==> [4/4] 运行真实评测"
MODEL_NAME="$MODEL" python src/runner.py

echo "==> 完成，报告位于 reports/"
ls -la reports/
