"""Output models for each agent and the final context layer.

Every agent reads from and writes to typed Pydantic models — no raw dicts
cross agent boundaries. This makes contracts explicit and catchable at
validation time rather than runtime KeyErrors deep in the pipeline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------
# `ok`       — agent ran successfully on first attempt (or after a retry)
# `degraded` — agent produced partial output (e.g. lineage emitted explicit
#              FKs but the LLM-inference pass failed)
# `failed`   — all retries exhausted; agent emitted a typed empty fallback
AgentHealth = Literal["ok", "degraded", "failed"]


# ---------------------------------------------------------------------------
# Profiler Agent outputs
# ---------------------------------------------------------------------------

class ColumnProfile(BaseModel):
    column_name: str
    inferred_semantic_type: str = Field(
        description="High-level semantic category: id, email, phone, currency, "
        "timestamp, boolean, enum, text, numeric, unknown"
    )
    null_rate: float = Field(ge=0.0, le=1.0, description="Estimated fraction of NULLs")
    distinct_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Estimated ratio of distinct values to total rows",
    )
    pattern: str | None = Field(
        default=None,
        description="Regex or description of detected value pattern",
    )
    anomalies: list[str] = Field(default_factory=list)


class TableProfile(BaseModel):
    table_name: str
    column_profiles: list[ColumnProfile]
    estimated_purpose: str = Field(
        description="One-sentence guess at the table's business role"
    )


class ProfilerOutput(BaseModel):
    tables: list[TableProfile]


# ---------------------------------------------------------------------------
# Lineage Agent outputs
# ---------------------------------------------------------------------------

class Relationship(BaseModel):
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relationship_type: Literal["explicit_fk", "inferred"] = Field(
        description="explicit_fk = parsed from DDL constraint; "
        "inferred = guessed from naming conventions"
    )
    confidence: float = Field(ge=0.0, le=1.0)


class LineageOutput(BaseModel):
    relationships: list[Relationship]
    orphan_tables: list[str] = Field(
        default_factory=list,
        description="Tables with zero detected relationships",
    )


# ---------------------------------------------------------------------------
# PII Detector outputs
# ---------------------------------------------------------------------------

class PIIColumnFlag(BaseModel):
    """A column flagged as containing PII or sensitive data."""

    table_name: str
    column_name: str
    pii_category: str = Field(
        description="One of: email, phone, name, address, ssn, dob, "
        "financial, ip, credential, generic_pii"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Combined confidence from name match + profiler corroboration",
    )
    masked_ddl_fragment: str = Field(
        description="DDL fragment with sensitive substrings replaced by [MASKED:<category>]"
    )
    reasoning: str = Field(
        description="Which rules fired and why this column was flagged"
    )


class PIIOutput(BaseModel):
    flagged_columns: list[PIIColumnFlag]
    sensitive_column_count: int = 0


# ---------------------------------------------------------------------------
# Semantic Agent outputs
# ---------------------------------------------------------------------------

class ColumnDefinition(BaseModel):
    column_name: str
    definition: str = Field(description="Human-readable definition")
    business_context: str = Field(
        description="How this column fits into broader business logic"
    )


class TableDefinition(BaseModel):
    table_name: str
    definition: str
    column_definitions: list[ColumnDefinition]
    domain: str = Field(
        description="Business domain this table belongs to, e.g. 'e-commerce', 'payments'"
    )


class SemanticOutput(BaseModel):
    tables: list[TableDefinition]


# ---------------------------------------------------------------------------
# Trust Scorer outputs
# ---------------------------------------------------------------------------

class TrustScore(BaseModel):
    entity_type: Literal["table", "column"]
    entity_name: str
    parent_table: str | None = Field(
        default=None,
        description="Set when entity_type is 'column'",
    )
    definition: str
    score: float = Field(ge=0.0, le=1.0)
    deterministic_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score from rule-based checks alone — fully auditable",
    )
    llm_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score from LLM semantic assessment",
    )
    flags: list[str] = Field(default_factory=list)
    needs_review: bool = False
    reasoning: str = Field(
        description="Human-readable explanation of the score breakdown"
    )


class TrustOutput(BaseModel):
    scores: list[TrustScore]
    review_count: int = 0
    average_confidence: float = 0.0


# ---------------------------------------------------------------------------
# LLM sub-models used for structured output calls
# ---------------------------------------------------------------------------

class LLMTrustAssessment(BaseModel):
    """Schema the LLM returns when asked to evaluate a definition."""

    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


# ---------------------------------------------------------------------------
# Final aggregated context layer
# ---------------------------------------------------------------------------

class ContextLayerColumn(BaseModel):
    column_name: str
    data_type: str
    definition: str
    business_context: str
    semantic_type: str
    trust_score: float
    needs_review: bool
    trust_flags: list[str] = Field(default_factory=list)
    is_sensitive: bool = Field(
        default=False,
        description="True when PII detection flagged this column as sensitive",
    )
    pii_category: str | None = Field(
        default=None,
        description="PII category if flagged: email, phone, name, ssn, etc.",
    )


class ContextLayerRelationship(BaseModel):
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relationship_type: str
    confidence: float


class ContextLayerTable(BaseModel):
    table_name: str
    definition: str
    domain: str
    columns: list[ContextLayerColumn]
    relationships: list[ContextLayerRelationship] = Field(default_factory=list)
    trust_score: float
    needs_review: bool


class ContextLayerMetadata(BaseModel):
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    schema_type: str
    models_used: dict[str, str] = Field(
        description="Agent name → model identifier"
    )
    table_count: int = 0
    column_count: int = 0
    average_trust: float = 0.0
    review_count: int = 0
    sensitive_column_count: int = 0
    agent_health: dict[str, AgentHealth] = Field(
        default_factory=dict,
        description="Per-agent execution health: ok / degraded / failed. "
        "Lets consumers reason about which parts of the context layer are "
        "trustworthy when an upstream agent ran in fallback mode.",
    )
    run_id: str | None = Field(
        default=None,
        description="Unique run identifier. Use GET /runs/{run_id} to "
        "retrieve the full per-agent audit trail for this run.",
    )
    demo: bool = Field(
        default=False,
        description="True when this context layer was generated by the /analyze/demo "
        "endpoint using the hardcoded sample e-commerce schema.",
    )


class ContextLayer(BaseModel):
    """The single deliverable: a fully annotated, trust-scored context layer."""

    tables: list[ContextLayerTable]
    metadata: ContextLayerMetadata
