"""Lineage Agent — infers relationships between tables.

Two-pass design:
  1. Deterministic pass: explicit FK constraints already parsed by InputParser
     get confidence 1.0 — these are facts, not guesses.
  2. LLM-assisted pass: gpt-4o-mini scans column names across tables to find
     implicit relationships (e.g., orders.user_id → users.id). These get
     confidence 0.5–0.9 because naming conventions aren't guarantees.

This split matters because deterministic extraction is auditable and
reproducible, while LLM inference adds coverage at the cost of certainty.

FAILURE HANDLING:
  Lineage gets *partial* degradation: the deterministic pass always
  succeeds, so when the LLM-inference pass exhausts its retries we still
  emit explicit FKs and mark health = "degraded" (not "failed"). This is
  honest — half the work succeeded — and preserves the high-trust
  explicit FKs the rest of the pipeline depends on.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from context_layer.agents._retry import call_with_retries
from context_layer.llm import get_llm
from context_layer.models.outputs import LineageOutput, Relationship
from context_layer.models.schema import TableSchema
from context_layer.models.state import PipelineState


class _InferredRelationships(BaseModel):
    """LLM output schema for the inference pass."""

    relationships: list[_InferredRel] = Field(default_factory=list)


class _InferredRel(BaseModel):
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


# Fix forward ref — _InferredRelationships references _InferredRel
_InferredRelationships.model_rebuild()


_SYSTEM_PROMPT = """\
You are a database relationship analyst. Given a set of table definitions, \
identify IMPLICIT foreign-key relationships that are NOT declared as \
explicit FK constraints.

Look for:
- Column names ending in _id that match another table's name + .id
- Naming patterns like table_name_fk, ref_table, etc.
- Columns with the same name across tables that suggest a join

For each relationship, provide a confidence score:
- 0.9: very strong signal (e.g., orders.user_id → users.id)
- 0.7: moderate signal (shared column name, plausible join)
- 0.5: weak signal (possible but ambiguous)

Do NOT include relationships that are already declared as FOREIGN KEY \
constraints — those are handled separately. Only return NEW inferred ones."""


def _extract_explicit(tables: list[TableSchema]) -> list[Relationship]:
    """Pass 1: deterministic extraction of declared FK constraints."""
    rels: list[Relationship] = []
    for table in tables:
        for fk in table.foreign_keys:
            rels.append(
                Relationship(
                    source_table=table.name,
                    source_column=fk.source_column,
                    target_table=fk.target_table,
                    target_column=fk.target_column,
                    relationship_type="explicit_fk",
                    confidence=1.0,
                )
            )
    return rels


def _build_schema_text(tables: list[TableSchema]) -> str:
    parts: list[str] = []
    for t in tables:
        cols = ", ".join(f"{c.name} {c.data_type}" for c in t.columns)
        fks = ", ".join(
            f"FK({fk.source_column} → {fk.target_table}.{fk.target_column})"
            for fk in t.foreign_keys
        )
        line = f"TABLE {t.name} ({cols})"
        if fks:
            line += f" [Declared FKs: {fks}]"
        parts.append(line)
    return "\n".join(parts)


def _find_orphans(
    tables: list[TableSchema], relationships: list[Relationship]
) -> list[str]:
    mentioned = set()
    for r in relationships:
        mentioned.add(r.source_table)
        mentioned.add(r.target_table)
    return [t.name for t in tables if t.name not in mentioned]


def lineage_node(state: PipelineState) -> dict:
    """Run the Lineage Agent: schema → relationships + orphan tables."""
    tables: list[TableSchema] = state["tables"]
    logger = state.get("run_logger")

    explicit = _extract_explicit(tables)

    schema_text = _build_schema_text(tables)
    llm = get_llm("fast").with_structured_output(_InferredRelationships)
    prompt_text = f"Analyze these tables:\n\n{schema_text}"

    def _invoke() -> _InferredRelationships:
        return llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ])

    rr = call_with_retries(_invoke)

    health: str
    if rr.value is not None:
        inferred = [
            Relationship(
                source_table=r.source_table,
                source_column=r.source_column,
                target_table=r.target_table,
                target_column=r.target_column,
                relationship_type="inferred",
                confidence=r.confidence,
            )
            for r in rr.value.relationships
        ]
        health = "ok"
    else:
        inferred = []
        health = "degraded"

    if logger:
        logger.log(
            agent="lineage",
            latency_ms=rr.latency_ms,
            attempts=rr.attempts,
            health=health,
            prompt_preview=prompt_text,
            response_preview=rr.value.model_dump_json() if rr.value else None,
            error=str(rr.error) if rr.error else None,
        )

    all_rels = explicit + inferred
    orphans = _find_orphans(tables, all_rels)

    return {
        "lineage_output": LineageOutput(
            relationships=all_rels,
            orphan_tables=orphans,
        ),
        "agent_health": {"lineage": health},
    }
