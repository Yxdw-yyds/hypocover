"""LLM clients behind one interface.

Model spec strings (``--model``):
    openai:gpt-4o-mini                 OpenAI API          (OPENAI_API_KEY)
    openrouter:deepseek/deepseek-chat  OpenRouter          (OPENROUTER_API_KEY)
    vllm:Qwen/Qwen3-4B                 local vLLM server   (VLLM_BASE_URL, default http://localhost:8000/v1)
    mock:zipf                          offline simulator, see mock.py
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

try:  # optional: load keys from .env
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass


@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0

    def add(self, c: Completion) -> None:
        self.calls += 1
        self.prompt_tokens += c.prompt_tokens
        self.completion_tokens += c.completion_tokens
        self.cost += c.cost

    def to_dict(self) -> dict:
        return dict(calls=self.calls, prompt_tokens=self.prompt_tokens,
                    completion_tokens=self.completion_tokens, cost=round(self.cost, 6))


class LLM:
    name: str = "llm"
    is_mock: bool = False

    def generate(self, prompt: str, *, meta: dict | None = None) -> Completion:
        raise NotImplementedError

    # Mock needs to know the instance; real models ignore this.
    def bind(self, task, inst) -> "LLM":
        return self


_PROVIDERS = {
    "openai": (None, "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "vllm": (None, None),
}


@dataclass
class OpenAICompatLLM(LLM):
    provider: str
    model: str
    temperature: float = 0.7
    top_p: float | None = None
    max_tokens: int | None = None
    max_attempts: int = 6
    extra_body: dict = field(default_factory=dict)

    def __post_init__(self):
        from openai import OpenAI

        base, key_env = _PROVIDERS[self.provider]
        if self.provider == "vllm":
            base = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
            key = os.environ.get("VLLM_API_KEY", "EMPTY")
        else:
            key = os.environ.get(key_env or "", "")
            if not key:
                raise RuntimeError(f"{key_env} is not set (put it in .env)")
        self.client = OpenAI(base_url=base, api_key=key, timeout=300)
        self.name = f"{self.provider}:{self.model}"
        if self.provider == "openrouter":
            self.extra_body.setdefault("usage", {"include": True})

    def generate(self, prompt: str, *, meta: dict | None = None) -> Completion:
        kwargs: dict[str, Any] = dict(model=self.model,
                                      messages=[{"role": "user", "content": prompt}])
        # reasoning models (o-series / gpt-5) reject temperature
        if not self.model.split("/")[-1].startswith(("o1", "o3", "o4", "gpt-5")):
            kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        delay = 2.0
        for attempt in range(self.max_attempts):
            try:
                r = self.client.chat.completions.create(**kwargs)
                text = (r.choices[0].message.content or "") if r.choices else ""
                u = r.usage
                cost = 0.0
                if u is not None:
                    cost = getattr(u, "cost", None) or (u.model_extra or {}).get("cost", 0.0)
                return Completion(text=text,
                                  prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                                  completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                                  cost=float(cost or 0.0))
            except Exception as e:  # rate limits, timeouts, 5xx
                if attempt == self.max_attempts - 1:
                    return Completion(text=f"Error querying {self.name}: {e}")
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise AssertionError("unreachable")


def make_llm(spec: str, **kw) -> LLM:
    provider, _, model = spec.partition(":")
    if provider == "mock":
        from .mock import MockLLM
        return MockLLM(**{k: v for k, v in kw.items() if k in ("seed",)})
    if provider not in _PROVIDERS:
        raise ValueError(f"unknown provider in {spec!r}; use one of {list(_PROVIDERS)} or mock")
    return OpenAICompatLLM(provider=provider, model=model, **kw)

