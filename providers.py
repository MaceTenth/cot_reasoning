"""Thin wrappers around the three providers.

Model IDs are configurable via env vars / CLI because the exact IDs evolve:
  OPENAI_MODEL     (default: gpt-5.5)
  ANTHROPIC_MODEL  (default: claude-opus-4-8)
  GEMINI_MODEL     (default: gemini-3)

Each provider implements .generate(system, user) -> ModelResult.
SDKs are imported lazily so the harness still runs with only some installed.
Token usage is captured from each response for cost estimation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ModelResult:
    text: str
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---------------------------------------------------------------------------
# Per-model pricing (USD per 1M tokens).  Add / update as pricing is published.
# fmt: (input_per_1M, output_per_1M)
# ---------------------------------------------------------------------------
PRICE_TABLE: dict[str, tuple[float, float]] = {
    # OpenAI — verified 2026-06-03 https://openai.com/api/pricing/
    # Standard tier, short context window pricing
    "gpt-5.5":              ( 5.00,   30.00),  # $5/MTok in, $30/MTok out
    "gpt-5.5-pro":          (30.00,  180.00),
    "gpt-5.4":              ( 2.50,   15.00),
    "gpt-5.4-mini":         ( 0.75,    4.50),
    "gpt-5.4-nano":         ( 0.20,    1.25),
    "gpt-5.4-pro":          (30.00,  180.00),
    "gpt-4o":               ( 2.50,   10.00),
    "gpt-4o-mini":          ( 0.15,    0.60),
    "gpt-4.1":              ( 2.00,    8.00),
    "o3":                   (10.00,   40.00),
    # Anthropic — verified 2026-06-03 https://docs.anthropic.com/en/about-claude/pricing
    "claude-opus-4-8":      ( 5.00,   25.00),  # $5/MTok in, $25/MTok out
    "claude-opus-4-7":      ( 5.00,   25.00),
    "claude-opus-4-6":      ( 5.00,   25.00),
    "claude-opus-4-5":      ( 5.00,   25.00),
    "claude-opus-4-1":      (15.00,   75.00),  # legacy Opus 4.1
    "claude-sonnet-4-6":    ( 3.00,   15.00),
    "claude-sonnet-4-5":    ( 3.00,   15.00),
    "claude-haiku-4-5":     ( 1.00,    5.00),
    "claude-haiku-3-5":     ( 0.80,    4.00),
    # Google — verify at https://cloud.google.com/vertex-ai/generative-ai/pricing
    "gemini-3":             ( 7.00,   21.00),  # unconfirmed — update when published
    "gemini-2.5-pro":       ( 3.50,   10.50),
    "gemini-2.5-flash":     ( 0.15,    0.60),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Return estimated USD cost, or None if the model isn't in the price table."""
    key = next((k for k in PRICE_TABLE if model.startswith(k)), None)
    if key is None:
        return None
    inp, out = PRICE_TABLE[key]
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.5")
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def generate(self, system: str, user: str, temperature: float = 0.0) -> ModelResult:
        try:
            client = self._client_lazy()
            kwargs: dict = dict(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as inner:
                # Reasoning models (o-series, gpt-5.x) reject custom temperature.
                if "temperature" in str(inner).lower() and "unsupported" in str(inner).lower():
                    kwargs.pop("temperature")
                    resp = client.chat.completions.create(**kwargs)
                else:
                    raise
            usage = resp.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()
            return ModelResult(
                text=resp.choices[0].message.content or "",
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
        except Exception as e:  # noqa: BLE001
            return ModelResult(text="", error=f"{type(e).__name__}: {e}")


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic()
        return self._client

    def generate(self, system: str, user: str, temperature: float = 0.0) -> ModelResult:
        try:
            client = self._client_lazy()
            kwargs: dict = dict(
                model=self.model,
                max_tokens=2048,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            try:
                resp = client.messages.create(**kwargs)
            except Exception as inner:
                # claude-opus-4+ deprecated temperature parameter.
                if "temperature" in str(inner).lower() and "deprecated" in str(inner).lower():
                    kwargs.pop("temperature")
                    resp = client.messages.create(**kwargs)
                else:
                    raise
            parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
            usage = resp.usage
            return ModelResult(
                text="".join(parts),
                prompt_tokens=getattr(usage, "input_tokens", 0),
                completion_tokens=getattr(usage, "output_tokens", 0),
            )
        except Exception as e:  # noqa: BLE001
            return ModelResult(text="", error=f"{type(e).__name__}: {e}")


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3")
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client()
        return self._client

    def generate(self, system: str, user: str, temperature: float = 0.0) -> ModelResult:
        try:
            from google.genai import types

            client = self._client_lazy()
            try:
                resp = client.models.generate_content(
                    model=self.model,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=temperature,
                    ),
                )
            except Exception as inner:
                # Some Gemini models don't accept temperature.
                if "temperature" in str(inner).lower():
                    resp = client.models.generate_content(
                        model=self.model,
                        contents=user,
                        config=types.GenerateContentConfig(system_instruction=system),
                    )
                else:
                    raise
            meta = getattr(resp, "usage_metadata", None)
            return ModelResult(
                text=resp.text or "",
                prompt_tokens=getattr(meta, "prompt_token_count", 0) or 0,
                completion_tokens=getattr(meta, "candidates_token_count", 0) or 0,
            )
        except Exception as e:  # noqa: BLE001
            return ModelResult(text="", error=f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# A deterministic offline provider so you can test the whole pipeline with no
# API keys. It naively tries to follow some constraints when in 'direct' mode
# and adds reasoning chatter in 'cot' mode (to mimic CoT pollution).
# --------------------------------------------------------------------------- #
class MockProvider:
    name = "mock"

    def __init__(self, model: str | None = None):
        self.model = model or "mock-1"

    def generate(self, system: str, user: str, temperature: float = 0.0) -> ModelResult:
        is_cot = "step by step" in user.lower() or "reason carefully" in system.lower()
        body = "Here is a concise answer that addresses the request directly."
        if is_cot:
            text = (
                "Let me think about this. First, I need to consider the "
                "requirements, weigh the options, and plan my response. "
                "Step 1: understand. Step 2: draft.\n\n"
                "Final answer: " + body
            )
        else:
            text = body
        return ModelResult(text=text)


PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "mock": MockProvider,
}


def build_provider(name: str, model: str | None = None):
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider '{name}'. choices: {list(PROVIDERS)}")
    return PROVIDERS[name](model=model)
