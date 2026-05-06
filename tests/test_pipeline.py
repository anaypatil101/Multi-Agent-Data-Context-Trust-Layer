"""End-to-end pipeline test with mocked LLM calls.

Validates that all seven nodes execute in the correct order, the parallel
fan-out/fan-in works, PII detection masks sensitive columns, and the final
ContextLayer is correctly assembled — all without burning Anthropic credits.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from pathlib import Path

from context_layer.models.outputs import (
    ColumnDefinition,
    ColumnProfile,
    ContextLayer,
    LLMTrustAssessment,
    LineageOutput,
    PIIOutput,
    ProfilerOutput,
    Relationship,
    SemanticOutput,
    TableDefinition,
    TableProfile,
)


SAMPLE_DDL = Path(__file__).resolve().parent.parent / "samples" / "ecommerce.sql"


# ---------------------------------------------------------------------------
# Fake LLM responses for each agent
# ---------------------------------------------------------------------------

def _fake_profiler_output(tables) -> ProfilerOutput:
    profiles = []
    for t in tables:
        cols = []
        for c in t.columns:
            sem = "unknown"
            if "email" in c.name:
                sem = "email"
            elif "phone" in c.name:
                sem = "phone"
            elif c.name in ("name", "first_name", "last_name"):
                sem = "name"
            elif c.name.endswith("_id") or c.name == "id":
                sem = "id"
            elif "price" in c.name or c.name in ("total", "amount"):
                sem = "currency"
            cols.append(ColumnProfile(
                column_name=c.name,
                inferred_semantic_type=sem,
                null_rate=0.0 if not c.nullable else 0.3,
                distinct_ratio=1.0 if c.is_primary_key else 0.5,
                pattern=None,
                anomalies=[],
            ))
        profiles.append(TableProfile(
            table_name=t.name,
            column_profiles=cols,
            estimated_purpose=f"Stores {t.name} data",
        ))
    return ProfilerOutput(tables=profiles)


def _fake_lineage_output() -> LineageOutput:
    return LineageOutput(
        relationships=[
            Relationship(
                source_table="orders",
                source_column="user_id",
                target_table="users",
                target_column="id",
                relationship_type="inferred",
                confidence=0.9,
            ),
        ],
        orphan_tables=[],
    )


def _fake_semantic_output(tables) -> SemanticOutput:
    defs = []
    for t in tables:
        col_defs = [
            ColumnDefinition(
                column_name=c.name,
                definition=f"The {c.name} column in {t.name}",
                business_context=f"Used in {t.name} business logic",
            )
            for c in t.columns
        ]
        defs.append(TableDefinition(
            table_name=t.name,
            definition=f"The {t.name} table stores core data",
            column_definitions=col_defs,
            domain="e-commerce",
        ))
    return SemanticOutput(tables=defs)


def _fake_trust_assessment() -> LLMTrustAssessment:
    return LLMTrustAssessment(confidence=0.85, reasoning="Definition looks reasonable")


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

def test_full_pipeline_with_mocked_llm():
    """Run the entire graph with mocked LLMs and verify the output."""
    from context_layer.agents.input_parser import input_parser_node

    ddl = SAMPLE_DDL.read_text()
    parsed = input_parser_node({"raw_schema": ddl, "schema_type": "sql"})
    tables = parsed["tables"]

    profiler_out = _fake_profiler_output(tables)
    lineage_out = _fake_lineage_output()
    semantic_out = _fake_semantic_output(tables)
    trust_assess = _fake_trust_assessment()

    # Mock the LLM factory so no real API calls are made.
    # Each with_structured_output() call returns a mock whose .invoke()
    # returns the pre-built output for that agent.
    call_count = {"profiler": 0, "lineage": 0, "semantic": 0, "trust": 0}

    def make_mock_llm(tier):
        mock_llm = MagicMock()

        def with_structured_output(schema):
            inner = MagicMock()

            def invoke(messages):
                if schema is ProfilerOutput:
                    call_count["profiler"] += 1
                    return profiler_out
                elif schema is SemanticOutput:
                    call_count["semantic"] += 1
                    return semantic_out
                elif schema is LLMTrustAssessment:
                    call_count["trust"] += 1
                    return trust_assess
                elif schema.__name__ == "_InferredRelationships":
                    call_count["lineage"] += 1
                    from context_layer.agents.lineage import _InferredRelationships
                    return _InferredRelationships(relationships=[])
                else:
                    raise ValueError(f"Unexpected schema: {schema}")

            inner.invoke = invoke
            return inner

        mock_llm.with_structured_output = with_structured_output
        return mock_llm

    with patch("context_layer.agents.profiler.get_llm", make_mock_llm), \
         patch("context_layer.agents.lineage.get_llm", make_mock_llm), \
         patch("context_layer.agents.semantic.get_llm", make_mock_llm), \
         patch("context_layer.agents.trust_scorer.get_llm", make_mock_llm):

        from context_layer.graph import build_graph
        graph = build_graph()
        result = graph.invoke({"raw_schema": ddl, "schema_type": "sql"})

    # --- Verify the output ---
    ctx: ContextLayer = result["context_layer"]

    # 1. All 6 tables present
    table_names = {t.table_name for t in ctx.tables}
    assert table_names == {
        "users", "categories", "products", "orders", "order_items", "payments"
    }, f"Expected 6 tables, got {table_names}"

    # 2. Metadata is populated
    assert ctx.metadata.table_count == 6
    assert ctx.metadata.column_count > 0
    assert ctx.metadata.average_trust > 0

    # 3. PII detection fired — email, phone, name columns flagged
    pii: PIIOutput = result["pii_output"]
    pii_cols = {(f.table_name, f.column_name) for f in pii.flagged_columns}
    assert ("users", "email") in pii_cols, "users.email should be flagged as PII"
    assert ("users", "phone") in pii_cols, "users.phone should be flagged as PII"
    assert pii.sensitive_column_count > 0

    # 4. Sensitivity propagated to final output
    users_table = next(t for t in ctx.tables if t.table_name == "users")
    email_col = next(c for c in users_table.columns if c.column_name == "email")
    assert email_col.is_sensitive is True
    assert email_col.pii_category == "email"

    phone_col = next(c for c in users_table.columns if c.column_name == "phone")
    assert phone_col.is_sensitive is True
    assert phone_col.pii_category == "phone"

    # Non-PII column should not be flagged
    id_col = next(c for c in users_table.columns if c.column_name == "id")
    assert id_col.is_sensitive is False
    assert id_col.pii_category is None

    # 5. sensitive_column_count in metadata
    assert ctx.metadata.sensitive_column_count == pii.sensitive_column_count

    # 6. All agents were called
    assert call_count["profiler"] == 1
    assert call_count["lineage"] == 1
    assert call_count["semantic"] == 1
    assert call_count["trust"] > 0  # Called once per entity

    # 7. Trust scores exist on every column
    for tbl in ctx.tables:
        for col in tbl.columns:
            assert 0.0 <= col.trust_score <= 1.0, f"{tbl.table_name}.{col.column_name} has bad trust"

    print(f"\n  Pipeline OK: {ctx.metadata.table_count} tables, "
          f"{ctx.metadata.column_count} columns, "
          f"avg trust {ctx.metadata.average_trust:.0%}, "
          f"{ctx.metadata.sensitive_column_count} sensitive, "
          f"{ctx.metadata.review_count} need review")


def test_pipeline_degrades_on_semantic_failure():
    """Semantic Agent always raises → pipeline must degrade, not crash.

    Verifies Gap 6 contract:
      - Pipeline completes without exception even when an LLM agent fails
        every retry.
      - `agent_health["semantic"]` is "failed" in the final ContextLayer.
      - Trust Scorer recognises the upstream failure and floors EVERY
        column to score=0.0 with `needs_review=True` and the
        `upstream_failure` flag.
      - Definitions come back as empty strings (the typed fallback)
        instead of crashing the aggregator.
      - Profiler / Lineage / PII Detector still produce normal output —
        partial results survive an isolated failure.
    """
    from context_layer.agents.input_parser import input_parser_node

    ddl = SAMPLE_DDL.read_text()
    parsed = input_parser_node({"raw_schema": ddl, "schema_type": "sql"})
    tables = parsed["tables"]

    profiler_out = _fake_profiler_output(tables)

    # ------------------------------------------------------------------
    # Patch time.sleep inside the retry helper so the test doesn't wait
    # 1+2 = 3 seconds per failed LLM call (~12s total).
    # ------------------------------------------------------------------
    with patch("context_layer.agents._retry.time.sleep", lambda *_: None):

        # Build a mock LLM factory: profiler/lineage succeed, semantic always
        # raises, trust_scorer must never be called for LLM scoring on
        # upstream failure (we'll assert that).
        semantic_attempts = {"count": 0}
        trust_llm_calls = {"count": 0}

        def make_mock_llm(tier):
            mock_llm = MagicMock()

            def with_structured_output(schema):
                inner = MagicMock()

                def invoke(messages):
                    if schema is ProfilerOutput:
                        return profiler_out
                    if schema is SemanticOutput:
                        semantic_attempts["count"] += 1
                        raise RuntimeError("simulated semantic LLM failure")
                    if schema is LLMTrustAssessment:
                        trust_llm_calls["count"] += 1
                        return LLMTrustAssessment(confidence=0.85, reasoning="ok")
                    if schema.__name__ == "_InferredRelationships":
                        from context_layer.agents.lineage import _InferredRelationships
                        return _InferredRelationships(relationships=[])
                    raise ValueError(f"Unexpected schema: {schema}")

                inner.invoke = invoke
                return inner

            mock_llm.with_structured_output = with_structured_output
            return mock_llm

        with patch("context_layer.agents.profiler.get_llm", make_mock_llm), \
             patch("context_layer.agents.lineage.get_llm", make_mock_llm), \
             patch("context_layer.agents.semantic.get_llm", make_mock_llm), \
             patch("context_layer.agents.trust_scorer.get_llm", make_mock_llm):

            from context_layer.graph import build_graph
            graph = build_graph()
            # MUST NOT raise — the whole point of Gap 6.
            result = graph.invoke({"raw_schema": ddl, "schema_type": "sql"})

    ctx: ContextLayer = result["context_layer"]

    # --- Retries actually happened: 3 attempts (1 initial + 2 retries) ---
    assert semantic_attempts["count"] == 3, (
        f"Expected exactly 3 semantic attempts (initial + 2 retries), "
        f"got {semantic_attempts['count']}"
    )

    # --- agent_health surfaced in metadata ---
    health = ctx.metadata.agent_health
    assert health.get("semantic") == "failed", (
        f"semantic should be 'failed', got {health.get('semantic')}"
    )
    assert health.get("profiler") == "ok"
    assert health.get("lineage") == "ok"

    # --- Trust Scorer skipped LLM (token-burn protection) ---
    assert trust_llm_calls["count"] == 0, (
        "Trust Scorer should not have called the LLM after upstream "
        "Semantic failure — it should floor scores instead."
    )

    # --- Every column floored to 0 + needs_review + upstream_failure flag ---
    for tbl in ctx.tables:
        for col in tbl.columns:
            assert col.trust_score == 0.0, (
                f"{tbl.table_name}.{col.column_name} should be floored "
                f"to 0.0 on upstream failure, got {col.trust_score}"
            )
            assert col.needs_review is True
            assert "upstream_failure" in col.trust_flags, (
                f"{tbl.table_name}.{col.column_name} missing upstream_failure flag"
            )
            # Definition should be the typed empty fallback.
            assert col.definition == "" or col.definition == "No definition generated"

    # --- PII detection still ran (deterministic — survives LLM failures) ---
    pii: PIIOutput = result["pii_output"]
    assert pii.sensitive_column_count > 0, "PII detection should still work"

    # --- Tables and columns still enumerated correctly ---
    assert ctx.metadata.table_count == 6
    assert ctx.metadata.column_count > 0

    print(
        f"\n  Degraded pipeline OK: semantic failed after "
        f"{semantic_attempts['count']} attempts, "
        f"all {ctx.metadata.column_count} cols floored to 0, "
        f"PII still flagged {pii.sensitive_column_count} cols, "
        f"trust scorer made {trust_llm_calls['count']} LLM calls"
    )


if __name__ == "__main__":
    test_full_pipeline_with_mocked_llm()
    test_pipeline_degrades_on_semantic_failure()
    print("\n  All assertions passed.")
