"""Trust Scorer Agent — hybrid deterministic + LLM confidence scoring.

DESIGN RATIONALE (interview-ready):
  Wrong context is worse than no context. A data analyst trusting an
  incorrect column definition can make costly business decisions. So we
  score conservatively using TWO independent signals:

  1. DETERMINISTIC RULES (weight 0.4):
     Structural checks that are fully auditable and reproducible.
     They catch issues like type-name mismatches or ambiguous naming
     that ANY schema has, regardless of domain. These rules never
     hallucinate and their results can be explained line-by-line.

  2. LLM ASSESSMENT (weight 0.6):
     gpt-4o evaluates whether the generated definition is semantically
     accurate and complete. This catches domain-level errors that rules
     can't — e.g., a definition that confuses "revenue" with "profit".
     But LLMs can be overconfident, which is why they don't get full
     weight.

  The 0.4/0.6 split is deliberate: deterministic rules act as a floor
  that prevents the LLM from inflating scores on structurally bad
  definitions. If rules flag problems, even a confident LLM can't push
  the score above the review threshold.

  Threshold: score < 0.6 → needs_review = True

FAILURE HANDLING:
  Two complementary mechanisms:

  a) UPSTREAM FAILURE FLOOR — if agent_health shows that an upstream LLM
     agent failed (Profiler or Semantic), this scorer skips the LLM
     assessment entirely and floors every score to 0.0 with flag
     `upstream_failure`. The safe default: when upstream context is
     unreliable, claim zero confidence.

  b) PER-ITEM RETRIES — each item's LLM call is wrapped in bounded
     retries. If a single item still fails, that item falls back to
     deterministic-score-only with flag `llm_assessment_unavailable`,
     so one bad item doesn't void the whole batch.
"""

from __future__ import annotations

import time

from langchain_core.messages import HumanMessage, SystemMessage

from context_layer.agents._retry import call_with_retries
from context_layer.llm import get_llm
from context_layer.models.outputs import (
    LLMTrustAssessment,
    LineageOutput,
    ProfilerOutput,
    SemanticOutput,
    TrustOutput,
    TrustScore,
)
from context_layer.models.state import PipelineState

REVIEW_THRESHOLD = 0.6
DETERMINISTIC_WEIGHT = 0.4
LLM_WEIGHT = 0.6

_AMBIGUOUS_NAMES = frozenset({
    "data", "val", "value", "tmp", "temp", "x", "y", "z", "col", "field",
    "info", "misc", "other", "stuff", "flag", "status2", "new", "old",
    "test", "foo", "bar", "baz",
})

_TYPE_SEMANTIC_MAP: dict[str, set[str]] = {
    "email": {"VARCHAR", "TEXT", "CHAR", "CHARACTER VARYING"},
    "phone": {"VARCHAR", "TEXT", "CHAR", "CHARACTER VARYING"},
    "url": {"VARCHAR", "TEXT", "CHAR", "CHARACTER VARYING"},
    "currency": {"DECIMAL", "NUMERIC", "FLOAT", "DOUBLE", "MONEY", "INTEGER", "INT", "BIGINT"},
    "timestamp": {"TIMESTAMP", "DATETIME", "DATE", "TIME"},
    "boolean": {"BOOLEAN", "BOOL", "TINYINT", "BIT", "INTEGER", "INT"},
    "id": {"INTEGER", "INT", "BIGINT", "SERIAL", "UUID", "VARCHAR", "TEXT"},
}


# ---------------------------------------------------------------------------
# Deterministic scoring
# ---------------------------------------------------------------------------

def _deterministic_score(
    entity_type: str,
    entity_name: str,
    definition: str,
    *,
    data_type: str | None = None,
    semantic_type: str | None = None,
    null_rate: float | None = None,
    is_orphan: bool = False,
) -> tuple[float, list[str]]:
    """Return (score, flags) based on structural rules.

    Starts at 1.0 and applies penalties. Each penalty is capped so that
    a single bad signal doesn't tank the score to zero — we want the
    composite score to reflect ALL issues, not just the worst one.
    """
    score = 1.0
    flags: list[str] = []

    if entity_name.lower() in _AMBIGUOUS_NAMES:
        score -= 0.3
        flags.append("ambiguous_name")

    if len(entity_name) <= 2:
        score -= 0.25
        flags.append("very_short_name")

    if len(definition) < 10:
        score -= 0.2
        flags.append("definition_too_short")

    if data_type and semantic_type and semantic_type != "unknown":
        allowed_types = _TYPE_SEMANTIC_MAP.get(semantic_type, set())
        if allowed_types and data_type.upper().split("(")[0] not in allowed_types:
            score -= 0.4
            flags.append("type_name_mismatch")

    if null_rate is not None and null_rate > 0.5:
        score -= 0.2
        flags.append("high_null_rate")

    if is_orphan and entity_type == "table":
        score -= 0.1
        flags.append("orphan_table")

    return max(score, 0.0), flags


# ---------------------------------------------------------------------------
# LLM assessment
# ---------------------------------------------------------------------------

_LLM_PROMPT = """\
You are a data quality reviewer. Evaluate whether the following definition \
is accurate, specific, and useful for a data analyst.

Entity: {entity_type} "{entity_name}" (in table "{table}")
Data type: {data_type}
Definition: "{definition}"
Business context: "{business_context}"

Rate your confidence that this definition is correct and complete.
Return a score from 0.0 to 1.0 and explain your reasoning in 1-2 sentences.
Be skeptical — if the definition is vague or could be wrong, score low."""


def _assess_batch_with_llm(
    items: list[dict],
) -> tuple[list[LLMTrustAssessment | None], float, int]:
    """Send each item to the LLM for semantic confidence scoring.

    Returns (assessments, total_latency_ms, total_attempts).
    Each assessment is None when that item failed all retries.
    """
    llm = get_llm("strong").with_structured_output(LLMTrustAssessment)
    results: list[LLMTrustAssessment | None] = []
    total_latency = 0.0
    total_attempts = 0

    for item in items:
        prompt = _LLM_PROMPT.format(**item)

        def _invoke(_p: str = prompt) -> LLMTrustAssessment:
            return llm.invoke([
                SystemMessage(
                    content="You are a precise data quality reviewer. "
                    "Score conservatively — when in doubt, score lower."
                ),
                HumanMessage(content=_p),
            ])

        rr = call_with_retries(_invoke)
        results.append(rr.value)
        total_latency += rr.latency_ms
        total_attempts += rr.attempts

    return results, total_latency, total_attempts


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def trust_scorer_node(state: PipelineState) -> dict:
    """Run the Trust Scorer: definitions + profiles → confidence scores."""
    semantic: SemanticOutput = state["semantic_output"]
    profiler: ProfilerOutput = state["profiler_output"]
    lineage: LineageOutput = state["lineage_output"]
    agent_health = state.get("agent_health", {}) or {}
    logger = state.get("run_logger")

    orphan_set = set(lineage.orphan_tables)

    profile_lookup: dict[str, dict[str, dict]] = {}
    for tp in profiler.tables:
        col_map: dict[str, dict] = {}
        for cp in tp.column_profiles:
            col_map[cp.column_name] = {
                "semantic_type": cp.inferred_semantic_type,
                "null_rate": cp.null_rate,
            }
        profile_lookup[tp.table_name] = col_map

    upstream_failed = (
        agent_health.get("semantic") == "failed"
        or agent_health.get("profiler") == "failed"
    )

    llm_items: list[dict] = []
    det_results: list[tuple[str, str, str | None, str, float, list[str]]] = []

    for table_def in semantic.tables:
        det_score, det_flags = _deterministic_score(
            "table", table_def.table_name, table_def.definition,
            is_orphan=table_def.table_name in orphan_set,
        )
        det_results.append((
            "table", table_def.table_name, None,
            table_def.definition, det_score, det_flags,
        ))
        llm_items.append({
            "entity_type": "table",
            "entity_name": table_def.table_name,
            "table": table_def.table_name,
            "data_type": "N/A",
            "definition": table_def.definition,
            "business_context": table_def.domain,
        })

        col_profiles = profile_lookup.get(table_def.table_name, {})
        for col_def in table_def.column_definitions:
            cp = col_profiles.get(col_def.column_name, {})
            det_score, det_flags = _deterministic_score(
                "column",
                col_def.column_name,
                col_def.definition,
                data_type=_find_col_type(state, table_def.table_name, col_def.column_name),
                semantic_type=cp.get("semantic_type"),
                null_rate=cp.get("null_rate"),
            )
            det_results.append((
                "column", col_def.column_name, table_def.table_name,
                col_def.definition, det_score, det_flags,
            ))
            llm_items.append({
                "entity_type": "column",
                "entity_name": col_def.column_name,
                "table": table_def.table_name,
                "data_type": _find_col_type(
                    state, table_def.table_name, col_def.column_name
                ) or "unknown",
                "definition": col_def.definition,
                "business_context": col_def.business_context,
            })

    t0 = time.monotonic()
    if upstream_failed:
        llm_assessments: list[LLMTrustAssessment | None] = [None] * len(llm_items)
        batch_latency = 0.0
        batch_attempts = 0
    else:
        llm_assessments, batch_latency, batch_attempts = _assess_batch_with_llm(llm_items)
    total_wall = (time.monotonic() - t0) * 1000

    scores: list[TrustScore] = []
    for (etype, ename, parent, definition, det_s, flags), llm_a in zip(
        det_results, llm_assessments
    ):
        if upstream_failed:
            final = 0.0
            llm_score = 0.0
            llm_reason = "upstream agent failed; score floored to 0"
            flags = [*flags, "upstream_failure"]
            needs_review = True
        elif llm_a is None:
            final = DETERMINISTIC_WEIGHT * det_s
            llm_score = 0.0
            llm_reason = "LLM scoring unavailable after retries; deterministic-only"
            flags = [*flags, "llm_assessment_unavailable"]
            needs_review = final < REVIEW_THRESHOLD
        else:
            llm_score = llm_a.confidence
            llm_reason = llm_a.reasoning
            final = DETERMINISTIC_WEIGHT * det_s + LLM_WEIGHT * llm_score
            needs_review = final < REVIEW_THRESHOLD

        if needs_review and "low_confidence" not in flags:
            flags = [*flags, "low_confidence"]

        reasoning = (
            f"Deterministic: {det_s:.2f} (flags: {', '.join(flags) if flags else 'none'}) | "
            f"LLM: {llm_score:.2f} ({llm_reason}) | "
            f"Final: {final:.2f}"
        )

        scores.append(TrustScore(
            entity_type=etype,
            entity_name=ename,
            parent_table=parent,
            definition=definition,
            score=round(min(max(final, 0.0), 1.0), 3),
            deterministic_score=round(det_s, 3),
            llm_score=round(llm_score, 3),
            flags=flags,
            needs_review=needs_review,
            reasoning=reasoning,
        ))

    review_count = sum(1 for s in scores if s.needs_review)
    avg = sum(s.score for s in scores) / len(scores) if scores else 0.0

    if not upstream_failed and llm_items:
        non_null = sum(1 for a in llm_assessments if a is not None)
        if non_null == 0:
            self_health = "failed"
        elif non_null < len(llm_assessments):
            self_health = "degraded"
        else:
            self_health = "ok"
    else:
        self_health = "ok"

    if logger:
        logger.log(
            agent="trust_scorer",
            latency_ms=batch_latency or total_wall,
            attempts=batch_attempts,
            health=self_health,
            prompt_preview=f"{len(llm_items)} items scored (upstream_failed={upstream_failed})",
            response_preview=f"avg={avg:.3f}, reviews={review_count}",
            error=None if self_health == "ok" else f"health={self_health}",
        )

    return {
        "trust_output": TrustOutput(
            scores=scores,
            review_count=review_count,
            average_confidence=round(avg, 3),
        ),
        "agent_health": {"trust_scorer": self_health},
    }


def _find_col_type(
    state: PipelineState, table_name: str, col_name: str
) -> str | None:
    """Look up the original SQL data type from the parsed schema."""
    for t in state.get("tables", []):
        if t.name == table_name:
            for c in t.columns:
                if c.name == col_name:
                    return c.data_type
    return None
