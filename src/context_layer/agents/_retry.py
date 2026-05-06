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
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")

MAX_RETRIES = 2  # 3 total attempts per LLM call
BACKOFF_BASE_SECONDS = 1.0


@dataclass
class RetryResult(Generic[T]):
    """Structured result from call_with_retries.

    Carries timing and attempt metadata alongside the value/error so
    callers can log observability data without ad-hoc timers.
    """

    value: T | None
    error: BaseException | None
    attempts: int
    latency_ms: float


def _is_retryable(exc: BaseException) -> bool:
    """Return False for errors that will never succeed on retry."""
    name = type(exc).__name__.lower()
    if "authentication" in name or "permission" in name:
        return False
    msg = str(exc).lower()
    if "context_length" in msg or "maximum context length" in msg:
        return False
    return True


def call_with_retries(
    fn: Callable[[], T],
    *,
    max_retries: int = MAX_RETRIES,
) -> RetryResult[T]:
    """Run `fn` up to `max_retries + 1` times with exponential backoff.

    Returns a RetryResult carrying the value or error, plus the number
    of attempts and wall-clock latency across all attempts.

    Never raises — callers decide whether to degrade or surface the error.
    """
    last_err: BaseException | None = None
    t0 = time.monotonic()
    attempts = 0
    for attempt in range(max_retries + 1):
        attempts += 1
        try:
            result = fn()
            elapsed = (time.monotonic() - t0) * 1000
            return RetryResult(value=result, error=None, attempts=attempts, latency_ms=elapsed)
        except BaseException as e:  # noqa: BLE001 — caller decides what to do
            last_err = e
            if not _is_retryable(e):
                break
            if attempt < max_retries:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))
    elapsed = (time.monotonic() - t0) * 1000
    return RetryResult(value=None, error=last_err, attempts=attempts, latency_ms=elapsed)
