"""LLM client factory.

Centralises model selection so every agent gets the right model tier
without hard-coding model names. Tier routing lets us swap models via
env vars (useful for cost control and A/B testing in production).

NOTE: Currently wired to OpenAI for testing. Switch back to Anthropic
before publishing by reverting this file and pyproject.toml.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

Tier = Literal["fast", "strong"]

_DEFAULTS: dict[Tier, str] = {
    "fast": "gpt-4o-mini",
    "strong": "gpt-4o",
}

_ENV_KEYS: dict[Tier, str] = {
    "fast": "FAST_MODEL",
    "strong": "STRONG_MODEL",
}


@lru_cache(maxsize=2)
def get_llm(tier: Tier) -> ChatOpenAI:
    """Return a ChatOpenAI instance for the requested capability tier.

    'fast'   → gpt-4o-mini  (Profiler, Lineage)
    'strong' → gpt-4o       (Semantic Agent, Trust Scorer)

    Model identifiers are overridable via FAST_MODEL / STRONG_MODEL env vars.
    """
    model = os.getenv(_ENV_KEYS[tier], _DEFAULTS[tier])
    return ChatOpenAI(
        model=model,
        max_tokens=4096,
        temperature=0,
    )
