"""Profiler Agent — analyses column types, null rates, value patterns, anomalies.

Uses gpt-4o-mini (fast tier) because profiling is a structured extraction task
that doesn't need deep reasoning. The LLM reads the DDL and returns a
typed ColumnProfile for every column; we lean on structured output to
guarantee the schema.

FAILURE HANDLING:
  Wrapped in `call_with_retries` (bounded retries with exponential backoff).
  On terminal failure we emit an empty ProfilerOutput and write
  `agent_health["profiler"] = "failed"`. The pipeline keeps running so the
  user still gets PII flags, lineage, etc. — partial output beats no output.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from context_layer.agents._retry import call_with_retries
from context_layer.llm import get_llm
from context_layer.models.outputs import ProfilerOutput, TableProfile
from context_layer.models.schema import TableSchema
from context_layer.models.state import PipelineState


_SYSTEM_PROMPT = """\
You are a database schema profiler. Given a list of tables and their columns \
(from SQL DDL), produce a structured profile for every column.

For each column, assess:
- inferred_semantic_type: one of id, email, phone, currency, timestamp, \
boolean, enum, text, numeric, url, address, name, unknown
- null_rate: estimate from DDL constraints (NOT NULL → 0.0, nullable with \
default → 0.3, nullable no default → 0.5, use judgment)
- distinct_ratio: estimate cardinality (PKs → 1.0, FKs → 0.7, booleans → low, \
enums → low, free text → high)
- pattern: regex or short description if a pattern is apparent from the \
column name / type (e.g., email columns → email pattern)
- anomalies: flag anything suspicious (type-name mismatch, unusual defaults, \
ambiguous naming)

Also provide a one-sentence estimated_purpose for each table.

Be precise and conservative. If unsure, say "unknown" rather than guessing."""


def _build_schema_text(tables: list[TableSchema]) -> str:
    parts: list[str] = []
    for t in tables:
        cols = "\n".join(
            f"  {c.name} {c.data_type}"
            f"{' NOT NULL' if not c.nullable else ''}"
            f"{' PRIMARY KEY' if c.is_primary_key else ''}"
            f"{' DEFAULT ' + c.default_value if c.default_value else ''}"
            for c in t.columns
        )
        parts.append(f"TABLE {t.name}:\n{cols}")
    return "\n\n".join(parts)


def _degraded_output(tables: list[TableSchema]) -> ProfilerOutput:
    """Empty per-table profile — used when all retries are exhausted."""
    return ProfilerOutput(tables=[
        TableProfile(
            table_name=t.name,
            column_profiles=[],
            estimated_purpose="(profiler degraded — LLM call failed after retries)",
        )
        for t in tables
    ])


def profiler_node(state: PipelineState) -> dict:
    """Run the Profiler Agent: schema → column profiles + anomalies."""
    tables: list[TableSchema] = state["tables"]
    schema_text = _build_schema_text(tables)
    logger = state.get("run_logger")

    llm = get_llm("fast").with_structured_output(ProfilerOutput)
    prompt_text = f"Profile these tables:\n\n{schema_text}"

    def _invoke() -> ProfilerOutput:
        return llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ])

    rr = call_with_retries(_invoke)

    health = "ok" if rr.value is not None else "failed"
    if logger:
        logger.log(
            agent="profiler",
            latency_ms=rr.latency_ms,
            attempts=rr.attempts,
            health=health,
            prompt_preview=prompt_text,
            response_preview=rr.value.model_dump_json() if rr.value else None,
            error=str(rr.error) if rr.error else None,
        )

    if rr.value is not None:
        return {"profiler_output": rr.value, "agent_health": {"profiler": "ok"}}

    return {
        "profiler_output": _degraded_output(tables),
        "agent_health": {"profiler": "failed"},
    }
