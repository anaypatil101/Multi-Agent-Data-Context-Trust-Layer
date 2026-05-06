"""Semantic Agent — generates human-readable definitions for every table and column.

Uses gpt-4o (strong tier) because definition quality is the core deliverable
of this pipeline. Cheaper models produce vague definitions; gpt-4o can
incorporate profiler context (detected patterns, anomalies) and lineage
context (how tables relate) to produce definitions that actually help a
data consumer understand what they're looking at.

PII HANDLING:
  Before any column reaches this agent, the upstream PII Detector flags
  sensitive columns. For flagged columns, we substitute the original DDL
  fragment with a [MASKED:<category>] placeholder so the LLM sees only
  the column name, type, and PII category. The LLM is instructed to write
  a generic definition based on the category alone — no speculation about
  actual values.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from context_layer.agents._retry import call_with_retries
from context_layer.llm import get_llm
from context_layer.models.outputs import (
    ColumnDefinition,
    LineageOutput,
    PIIOutput,
    ProfilerOutput,
    SemanticOutput,
    TableDefinition,
)
from context_layer.models.schema import TableSchema
from context_layer.models.state import PipelineState


_SYSTEM_PROMPT = """\
You are a data documentation expert. Given database tables with their column \
profiles and inter-table relationships, write clear, concise definitions.

For each TABLE, provide:
- definition: 1-2 sentences explaining what this table stores and why it exists
- domain: the business domain (e.g., "e-commerce", "user management", "payments")

For each COLUMN, provide:
- definition: 1 sentence explaining what the column represents
- business_context: how this column is used in business processes

Guidelines:
- Write for a data analyst who has never seen this schema before
- Reference relationships when relevant ("links to the users table via user_id")
- If the profiler detected anomalies, acknowledge them in the definition
- Be specific: "unique identifier for a customer order" is better than "an ID field"
- Do NOT pad definitions with filler words

SENSITIVE COLUMN HANDLING:
- Columns marked [SENSITIVE:<category>] contain PII. Their content has been \
masked. Write a generic definition based ONLY on the category and column name.
- Do NOT speculate about format, example values, or specific contents.
- Mention that the field is sensitive and indicate the PII category in business_context."""


def _build_context(
    tables: list[TableSchema],
    profiler: ProfilerOutput,
    lineage: LineageOutput,
    pii: PIIOutput,
) -> str:
    profile_lookup: dict[str, dict[str, str]] = {}
    for tp in profiler.tables:
        col_map: dict[str, str] = {}
        for cp in tp.column_profiles:
            parts = [f"type={cp.inferred_semantic_type}"]
            if cp.pattern:
                parts.append(f"pattern={cp.pattern}")
            if cp.anomalies:
                parts.append(f"anomalies={cp.anomalies}")
            col_map[cp.column_name] = ", ".join(parts)
        profile_lookup[tp.table_name] = col_map

    pii_lookup: dict[tuple[str, str], str] = {
        (f.table_name, f.column_name): f.pii_category
        for f in pii.flagged_columns
    }

    rel_text = "\n".join(
        f"  {r.source_table}.{r.source_column} → "
        f"{r.target_table}.{r.target_column} ({r.relationship_type}, "
        f"conf={r.confidence})"
        for r in lineage.relationships
    )

    sections: list[str] = []
    for t in tables:
        col_lines: list[str] = []
        t_profiles = profile_lookup.get(t.name, {})
        for c in t.columns:
            pii_category = pii_lookup.get((t.name, c.name))

            if pii_category:
                col_lines.append(
                    f"  {c.name} {c.data_type}"
                    f"{' NOT NULL' if not c.nullable else ''}"
                    f"{' PK' if c.is_primary_key else ''}"
                    f"  [SENSITIVE:{pii_category}]"
                )
            else:
                profile_info = t_profiles.get(c.name, "")
                col_lines.append(
                    f"  {c.name} {c.data_type}"
                    f"{' NOT NULL' if not c.nullable else ''}"
                    f"{' PK' if c.is_primary_key else ''}"
                    + (f"  [{profile_info}]" if profile_info else "")
                )
        sections.append(f"TABLE {t.name}:\n" + "\n".join(col_lines))

    schema_section = "\n\n".join(sections)
    return (
        f"=== SCHEMA WITH PROFILES ===\n{schema_section}\n\n"
        f"=== RELATIONSHIPS ===\n{rel_text or '(none detected)'}\n\n"
        f"=== ORPHAN TABLES ===\n{', '.join(lineage.orphan_tables) or '(none)'}"
    )


def _degraded_output(tables: list[TableSchema]) -> SemanticOutput:
    """Empty-but-typed semantic output used when all retries fail."""
    return SemanticOutput(tables=[
        TableDefinition(
            table_name=t.name,
            definition="",
            domain="unknown",
            column_definitions=[
                ColumnDefinition(
                    column_name=c.name,
                    definition="",
                    business_context="",
                )
                for c in t.columns
            ],
        )
        for t in tables
    ])


def semantic_node(state: PipelineState) -> dict:
    """Run the Semantic Agent: schema + profiles + lineage + PII flags → definitions."""
    tables: list[TableSchema] = state["tables"]
    profiler: ProfilerOutput = state["profiler_output"]
    lineage: LineageOutput = state["lineage_output"]
    pii: PIIOutput = state["pii_output"]
    logger = state.get("run_logger")

    context = _build_context(tables, profiler, lineage, pii)

    llm = get_llm("strong").with_structured_output(SemanticOutput)
    prompt_text = f"Generate definitions for:\n\n{context}"

    def _invoke() -> SemanticOutput:
        return llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ])

    rr = call_with_retries(_invoke)

    health = "ok" if rr.value is not None else "failed"
    if logger:
        logger.log(
            agent="semantic",
            latency_ms=rr.latency_ms,
            attempts=rr.attempts,
            health=health,
            prompt_preview=prompt_text,
            response_preview=rr.value.model_dump_json() if rr.value else None,
            error=str(rr.error) if rr.error else None,
        )

    if rr.value is not None:
        return {"semantic_output": rr.value, "agent_health": {"semantic": "ok"}}

    return {
        "semantic_output": _degraded_output(tables),
        "agent_health": {"semantic": "failed"},
    }
