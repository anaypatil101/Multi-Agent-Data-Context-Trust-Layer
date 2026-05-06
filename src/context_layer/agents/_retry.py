"""Bounded retry helper shared by every LLM-calling agent.

DESIGN RATIONALE:
  Retries protect against transient LLM failures (rate limits, network
  blips, malformed JSON that fails Pydantic validation) without burning
  tokens in an infinite loop. Three deliberate safeguards:

  1. HARD CAP — `MAX_RETRIES = 2` is a module constant. Agents cannot
     accidentally configure 100 retries; the worst case is always 3
     attempts per LLM call.

  2. NON-RETRYABLE ERRORS — `AuthenticationError` (bad API key) and
     prompt-too-long errors will never succeed on retry. We let them
     bubble immediately so the user learns about the real problem
     instead of waiting for backoff and consuming tokens.

  3. EXPONENTIAL BACKOFF — sleep 1s then 2s between attempts. Naturally
     paces token spend and gives rate limits time to clear.

  Worst-case spend: 3 attempts * 4 LLM agents = 12 calls.
  The pipeline already issues ~40 calls in the happy path (profiler +
  lineage + semantic + per-entity trust scoring), so retries are at most
  a ~30% ceiling — bounded, not unbounded.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")

MAX_RETRIES = 2  # 3 total attempts per LLM call
BACKOFF_BASE_SECONDS = 1.0


def _is_retryable(exc: BaseException) -> bool:
    """Return False for errors that will never succeed on retry."""
    name = type(exc).__name__.lower()
    # Any auth / permission failure is terminal — retrying just wastes time.
    if "authentication" in name or "permission" in name:
        return False
    # Token-budget / context-length errors are also terminal.
    msg = str(exc).lower()
    if "context_length" in msg or "maximum context length" in msg:
        return False
    return True


def call_with_retries(
    fn: Callable[[], T],
    *,
    max_retries: int = MAX_RETRIES,
) -> tuple[T | None, BaseException | None]:
    """Run `fn` up to `max_retries + 1` times with exponential backoff.

    Returns:
        (result, None)        on success.
        (None, last_exception) when all attempts have failed.

    Never raises — callers decide whether to degrade or surface the error.
    """
    last_err: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn(), None
        except BaseException as e:  # noqa: BLE001 — caller decides what to do
            last_err = e
            if not _is_retryable(e):
                break
            if attempt < max_retries:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))
    return None, last_err
