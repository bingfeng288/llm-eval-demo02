# -*- coding: utf-8 -*-
"""统一模型客户端：ollama / openai 兼容 API / mock。

所有 provider 暴露同一个接口：
    chat(messages: list[dict], temperature: float | None, max_tokens: int | None) -> str

用法：
    from src.model_client import build_client
    client = build_client(config["model"])
    print(client.chat([{"role": "user", "content": "你好"}]))
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ModelClient:
    """OpenAI 兼容客户端：同时支持 Ollama(base_url=.../v1) 与任意 OpenAI 兼容服务。"""

    def __init__(self, cfg: dict):
        self.provider = cfg.get("provider", "ollama")
        self.base_url = cfg.get("base_url", "http://localhost:11434/v1")
        self.model = cfg.get("model", "qwen2.5:7b")
        self.temperature = cfg.get("temperature", 0.2)
        self.max_tokens = cfg.get("max_tokens", 1024)
        self._client = None
        if self.provider != "mock":
            self._client = self._init_openai_client()

    def _init_openai_client(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                f"provider={self.provider!r} 需要 openai 库：pip install openai"
            )
        # Ollama 不校验 key，占位即可；真实服务请改为环境变量注入
        return OpenAI(base_url=self.base_url, api_key=cfg_key())

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def __repr__(self):  # pragma: no cover
        return f"<ModelClient {self.provider}:{self.model}>"


class MockClient:
    """无模型环境用：确定性返回，保证 runner 全流程可跑通、可做 CI 冒烟。"""

    def __init__(self, cfg: dict):
        self.model = cfg.get("model", "mock")

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        question = messages[-1]["content"][:24]
        if "敏感" in question or "违法" in question:
            return "抱歉，我无法回答这个问题。"
        return f"这是模拟回答（用于验证评测流水线）：{question}"


def build_client(cfg: dict):
    """按配置构造客户端。"""
    provider = cfg.get("provider", "ollama")
    if provider == "mock":
        return MockClient(cfg)
    return ModelClient(cfg)


def cfg_key() -> str:
    import os

    return os.environ.get("OPENAI_API_KEY", "ollama")
