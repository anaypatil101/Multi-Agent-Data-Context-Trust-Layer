"""LangGraph pipeline state.

Uses TypedDict (not Pydantic) because LangGraph's StateGraph expects
TypedDict for state schema. Each output key is owned by exactly one agent,
so no reducer annotations are needed for them — parallel agents (Profiler,
Lineage) write to disjoint keys and LangGraph merges them without conflict.

`agent_health` is the one shared key. Profiler and Lineage both write it
in the same parallel super-step, so it needs a merge reducer to prevent
the second writer from clobbering the first.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from context_layer.models.outputs import (
    AgentHealth,
    ContextLayer,
    LineageOutput,
    PIIOutput,
    ProfilerOutput,
    SemanticOutput,
    TrustOutput,
)
from context_layer.models.schema import TableSchema


def _merge_health(
    a: dict[str, AgentHealth] | None,
    b: dict[str, AgentHealth] | None,
) -> dict[str, AgentHealth]:
    """Reducer for agent_health.

    Agents only ever set their own key (e.g. {"profiler": "ok"}), so a
    plain dict-merge is safe — no two agents ever write the same key.
    """
    out: dict[str, AgentHealth] = {}
    if a:
        out.update(a)
    if b:
        out.update(b)
    return out


class PipelineState(TypedDict, total=False):
    """Typed state threaded through the entire agent graph.

    `total=False` lets agents return partial updates — each node only
    sets the keys it owns. LangGraph shallow-merges the returned dict
    into the running state.
    """

    raw_schema: str
    schema_type: str
    tables: list[TableSchema]
    profiler_output: ProfilerOutput
    lineage_output: LineageOutput
    pii_output: PIIOutput
    semantic_output: SemanticOutput
    trust_output: TrustOutput
    context_layer: ContextLayer
    agent_health: Annotated[dict[str, AgentHealth], _merge_health]
