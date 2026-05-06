"""End-to-end pipeline test with mocked LLM calls.

Validates that all seven nodes execute in the correct order, the parallel
fan-out/fan-in works, PII detection masks sensitive columns, the final
ContextLayer is correctly assembled, and the audit trail captures every
agent's execution — all without burning API credits.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

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
from context_layer.run_logger import RunLogger


SAMPLE_DDL = Path(__file__).resolve().parent.parent / "samples" / "ecommerce.sql"
_TEST_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


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
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm_factory(
    profiler_out, semantic_out, trust_assess, call_count=None
):
    """Build a mock get_llm that dispatches by structured output schema."""
    if call_count is None:
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

    return make_mock_llm, call_count


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_full_pipeline_with_mocked_llm():
    """Run the entire graph with mocked LLMs and verify the output + audit trail."""
    from context_layer.agents.input_parser import input_parser_node

    ddl = SAMPLE_DDL.read_text()
    parsed = input_parser_node({"raw_schema": ddl, "schema_type": "sql"})
    tables = parsed["tables"]

    profiler_out = _fake_profiler_output(tables)
    semantic_out = _fake_semantic_output(tables)
    trust_assess = _fake_trust_assessment()

    make_mock_llm, call_count = _make_mock_llm_factory(
        profiler_out, semantic_out, trust_assess
    )

    logger = RunLogger(run_id="test-happy")

    with patch("context_layer.agents.profiler.get_llm", make_mock_llm), \
         patch("context_layer.agents.lineage.get_llm", make_mock_llm), \
         patch("context_layer.agents.semantic.get_llm", make_mock_llm), \
         patch("context_layer.agents.trust_scorer.get_llm", make_mock_llm):

        from context_layer.graph import build_graph
        graph = build_graph()
        result = graph.invoke({
            "raw_schema": ddl,
            "schema_type": "sql",
            "run_id": logger.run_id,
            "run_logger": logger,
        })

    ctx: ContextLayer = result["context_layer"]

    # --- Core pipeline assertions ---

    table_names = {t.table_name for t in ctx.tables}
    assert table_names == {
        "users", "categories", "products", "orders", "order_items", "payments"
    }, f"Expected 6 tables, got {table_names}"

    assert ctx.metadata.table_count == 6
    assert ctx.metadata.column_count > 0
    assert ctx.metadata.average_trust > 0

    pii: PIIOutput = result["pii_output"]
    pii_cols = {(f.table_name, f.column_name) for f in pii.flagged_columns}
    assert ("users", "email") in pii_cols, "users.email should be flagged as PII"
    assert ("users", "phone") in pii_cols, "users.phone should be flagged as PII"
    assert pii.sensitive_column_count > 0

    users_table = next(t for t in ctx.tables if t.table_name == "users")
    email_col = next(c for c in users_table.columns if c.column_name == "email")
    assert email_col.is_sensitive is True
    assert email_col.pii_category == "email"

    phone_col = next(c for c in users_table.columns if c.column_name == "phone")
    assert phone_col.is_sensitive is True
    assert phone_col.pii_category == "phone"

    id_col = next(c for c in users_table.columns if c.column_name == "id")
    assert id_col.is_sensitive is False
    assert id_col.pii_category is None

    assert ctx.metadata.sensitive_column_count == pii.sensitive_column_count

    assert call_count["profiler"] == 1
    assert call_count["lineage"] == 1
    assert call_count["semantic"] == 1
    assert call_count["trust"] > 0

    for tbl in ctx.tables:
        for col in tbl.columns:
            assert 0.0 <= col.trust_score <= 1.0, f"{tbl.table_name}.{col.column_name} has bad trust"

    # --- Audit trail assertions ---

    assert ctx.metadata.run_id == "test-happy"

    entries = logger.entries
    agent_names = [e["agent"] for e in entries]
    assert "input_parser" in agent_names
    assert "profiler" in agent_names
    assert "lineage" in agent_names
    assert "pii_detector" in agent_names
    assert "semantic" in agent_names
    assert "trust_scorer" in agent_names
    assert "aggregator" in agent_names

    for entry in entries:
        assert entry["run_id"] == "test-happy"
        assert "timestamp" in entry
        assert "latency_ms" in entry
        assert entry["health"] in ("ok", "degraded", "failed")

    # JSONL file was flushed by the aggregator
    jsonl_path = _TEST_RUNS_DIR / "test-happy.jsonl"
    assert jsonl_path.exists(), "Audit trail JSONL was not flushed"
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == len(entries)
    for line in lines:
        parsed_entry = json.loads(line)
        assert "agent" in parsed_entry
        assert "run_id" in parsed_entry

    # Clean up test JSONL
    jsonl_path.unlink(missing_ok=True)

    print(f"\n  Pipeline OK: {ctx.metadata.table_count} tables, "
          f"{ctx.metadata.column_count} columns, "
          f"avg trust {ctx.metadata.average_trust:.0%}, "
          f"{ctx.metadata.sensitive_column_count} sensitive, "
          f"{ctx.metadata.review_count} need review, "
          f"audit trail: {len(entries)} entries")


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
      - Audit trail captures the failure with error and retry count.
    """
    from context_layer.agents.input_parser import input_parser_node

    ddl = SAMPLE_DDL.read_text()
    parsed = input_parser_node({"raw_schema": ddl, "schema_type": "sql"})
    tables = parsed["tables"]

    profiler_out = _fake_profiler_output(tables)

    with patch("context_layer.agents._retry.time.sleep", lambda *_: None):

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

        logger = RunLogger(run_id="test-degraded")

        with patch("context_layer.agents.profiler.get_llm", make_mock_llm), \
             patch("context_layer.agents.lineage.get_llm", make_mock_llm), \
             patch("context_layer.agents.semantic.get_llm", make_mock_llm), \
             patch("context_layer.agents.trust_scorer.get_llm", make_mock_llm):

            from context_layer.graph import build_graph
            graph = build_graph()
            result = graph.invoke({
                "raw_schema": ddl,
                "schema_type": "sql",
                "run_id": logger.run_id,
                "run_logger": logger,
            })

    ctx: ContextLayer = result["context_layer"]

    # --- Retries actually happened ---
    assert semantic_attempts["count"] == 3, (
        f"Expected exactly 3 semantic attempts (initial + 2 retries), "
        f"got {semantic_attempts['count']}"
    )

    # --- agent_health surfaced in metadata ---
    health = ctx.metadata.agent_health
    assert health.get("semantic") == "failed"
    assert health.get("profiler") == "ok"
    assert health.get("lineage") == "ok"

    # --- Trust Scorer skipped LLM ---
    assert trust_llm_calls["count"] == 0

    # --- Every column floored ---
    for tbl in ctx.tables:
        for col in tbl.columns:
            assert col.trust_score == 0.0
            assert col.needs_review is True
            assert "upstream_failure" in col.trust_flags
            assert col.definition == "" or col.definition == "No definition generated"

    # --- PII detection still ran ---
    pii: PIIOutput = result["pii_output"]
    assert pii.sensitive_column_count > 0

    assert ctx.metadata.table_count == 6
    assert ctx.metadata.column_count > 0

    # --- Audit trail captured failure ---
    assert ctx.metadata.run_id == "test-degraded"

    entries = logger.entries
    semantic_entry = next(e for e in entries if e["agent"] == "semantic")
    assert semantic_entry["health"] == "failed"
    assert semantic_entry["error"] is not None
    assert semantic_entry["attempts"] == 3

    # JSONL was flushed
    jsonl_path = _TEST_RUNS_DIR / "test-degraded.jsonl"
    assert jsonl_path.exists()
    jsonl_path.unlink(missing_ok=True)

    print(
        f"\n  Degraded pipeline OK: semantic failed after "
        f"{semantic_attempts['count']} attempts, "
        f"all {ctx.metadata.column_count} cols floored to 0, "
        f"PII still flagged {pii.sensitive_column_count} cols, "
        f"trust scorer made {trust_llm_calls['count']} LLM calls, "
        f"audit trail: {len(entries)} entries"
    )


if __name__ == "__main__":
    test_full_pipeline_with_mocked_llm()
    test_pipeline_degrades_on_semantic_failure()
    print("\n  All assertions passed.")
