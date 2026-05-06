"""Aggregator — compiles all agent outputs into the final ContextLayer.

Purely deterministic: joins outputs from all upstream agents into one
governed asset record. No LLM calls — the heavy reasoning is done.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from context_layer.models.outputs import (
    ContextLayer,
    ContextLayerColumn,
    ContextLayerMetadata,
    ContextLayerRelationship,
    ContextLayerTable,
    LineageOutput,
    PIIOutput,
    ProfilerOutput,
    SemanticOutput,
    TrustOutput,
)
from context_layer.models.schema import TableSchema
from context_layer.models.state import PipelineState


def aggregator_node(state: PipelineState) -> dict:
    """Merge all agent outputs into a single ContextLayer."""
    t0 = time.monotonic()

    tables: list[TableSchema] = state["tables"]
    profiler: ProfilerOutput = state["profiler_output"]
    lineage: LineageOutput = state["lineage_output"]
    pii: PIIOutput = state["pii_output"]
    semantic: SemanticOutput = state["semantic_output"]
    trust: TrustOutput = state["trust_output"]
    logger = state.get("run_logger")
    run_id = state.get("run_id")

    pii_map: dict[tuple[str, str], str] = {
        (f.table_name, f.column_name): f.pii_category
        for f in pii.flagged_columns
    }

    profile_map: dict[str, dict[str, object]] = {}
    for tp in profiler.tables:
        col_map = {}
        for cp in tp.column_profiles:
            col_map[cp.column_name] = cp
        profile_map[tp.table_name] = col_map

    semantic_map: dict[str, object] = {}
    col_def_map: dict[tuple[str, str], object] = {}
    for td in semantic.tables:
        semantic_map[td.table_name] = td
        for cd in td.column_definitions:
            col_def_map[(td.table_name, cd.column_name)] = cd

    trust_map: dict[tuple[str, str | None], object] = {}
    for ts in trust.scores:
        trust_map[(ts.entity_name, ts.parent_table)] = ts

    rel_map: dict[str, list[ContextLayerRelationship]] = {}
    for r in lineage.relationships:
        ctx_rel = ContextLayerRelationship(
            source_table=r.source_table,
            source_column=r.source_column,
            target_table=r.target_table,
            target_column=r.target_column,
            relationship_type=r.relationship_type,
            confidence=r.confidence,
        )
        rel_map.setdefault(r.source_table, []).append(ctx_rel)
        if r.target_table != r.source_table:
            rel_map.setdefault(r.target_table, []).append(ctx_rel)

    ctx_tables: list[ContextLayerTable] = []
    total_cols = 0

    for table in tables:
        td = semantic_map.get(table.name)
        table_trust = trust_map.get((table.name, None))

        cols: list[ContextLayerColumn] = []
        for col in table.columns:
            cd = col_def_map.get((table.name, col.name))
            cp = (profile_map.get(table.name) or {}).get(col.name)
            ct = trust_map.get((col.name, table.name))

            pii_cat = pii_map.get((table.name, col.name))
            cols.append(ContextLayerColumn(
                column_name=col.name,
                data_type=col.data_type,
                definition=cd.definition if cd else "No definition generated",
                business_context=cd.business_context if cd else "",
                semantic_type=cp.inferred_semantic_type if cp else "unknown",
                trust_score=ct.score if ct else 0.0,
                needs_review=ct.needs_review if ct else True,
                trust_flags=ct.flags if ct else ["no_score"],
                is_sensitive=pii_cat is not None,
                pii_category=pii_cat,
            ))

        total_cols += len(cols)

        ctx_tables.append(ContextLayerTable(
            table_name=table.name,
            definition=td.definition if td else "No definition generated",
            domain=td.domain if td else "unknown",
            columns=cols,
            relationships=rel_map.get(table.name, []),
            trust_score=table_trust.score if table_trust else 0.0,
            needs_review=table_trust.needs_review if table_trust else True,
        ))

    metadata = ContextLayerMetadata(
        generated_at=datetime.now(timezone.utc),
        schema_type=state.get("schema_type", "sql"),
        models_used={
            "profiler": "gpt-4o-mini (fast)",
            "lineage": "gpt-4o-mini (fast)",
            "pii_detector": "deterministic (no LLM)",
            "semantic": "gpt-4o (strong)",
            "trust_scorer": "gpt-4o (strong)",
            "aggregator": "deterministic (no LLM)",
        },
        table_count=len(ctx_tables),
        column_count=total_cols,
        average_trust=trust.average_confidence,
        review_count=trust.review_count,
        sensitive_column_count=pii.sensitive_column_count,
        agent_health=dict(state.get("agent_health", {}) or {}),
        run_id=run_id,
    )

    elapsed = (time.monotonic() - t0) * 1000

    if logger:
        logger.log(
            agent="aggregator",
            latency_ms=elapsed,
            health="ok",
            response_preview=f"{len(ctx_tables)} tables, {total_cols} cols assembled",
        )
        logger.flush()

    return {
        "context_layer": ContextLayer(tables=ctx_tables, metadata=metadata)
    }
