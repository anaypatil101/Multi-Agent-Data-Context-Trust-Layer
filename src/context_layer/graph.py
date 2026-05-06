"""LangGraph StateGraph wiring.

This is the orchestration layer — it defines HOW agents execute, not
WHAT they do. The graph structure encodes three key architectural decisions:

1. PARALLEL FAN-OUT: Profiler and Lineage both receive the parsed schema
   from InputParser and run in the same LangGraph "super-step". They write
   to disjoint state keys, so no reducer is needed.

2. PII GATE BEFORE SEMANTIC: After the parallel branches converge, a
   deterministic PII Detector runs. It must execute before any LLM sees
   the schema metadata, so sensitive columns are masked in the Semantic
   Agent's prompt. Both Profiler and Lineage fan in at this gate.

3. FAN-IN AT PII DETECTOR: The PII Detector waits for both parallel
   branches to complete before it runs. This is automatic — LangGraph
   won't execute a node until ALL its inbound edges have been satisfied.

We use static edges (add_edge) rather than the Send API because the
branching is fixed at graph-compile time, not data-dependent.
"""

from langgraph.graph import END, START, StateGraph

from context_layer.agents.aggregator import aggregator_node
from context_layer.agents.input_parser import input_parser_node
from context_layer.agents.lineage import lineage_node
from context_layer.agents.pii_detector import pii_detector_node
from context_layer.agents.profiler import profiler_node
from context_layer.agents.semantic import semantic_node
from context_layer.agents.trust_scorer import trust_scorer_node
from context_layer.models.state import PipelineState


def build_graph() -> StateGraph:
    """Construct and compile the agent pipeline graph."""
    builder = StateGraph(PipelineState)

    builder.add_node("input_parser", input_parser_node)
    builder.add_node("profiler", profiler_node)
    builder.add_node("lineage", lineage_node)
    builder.add_node("pii_detector", pii_detector_node)
    builder.add_node("semantic", semantic_node)
    builder.add_node("trust_scorer", trust_scorer_node)
    builder.add_node("aggregator", aggregator_node)

    # InputParser is the entry point
    builder.add_edge(START, "input_parser")

    # Fan-out: Profiler and Lineage run in parallel (same super-step)
    builder.add_edge("input_parser", "profiler")
    builder.add_edge("input_parser", "lineage")

    # Fan-in at PII Detector: blocks Semantic until sensitive columns are flagged
    builder.add_edge("profiler", "pii_detector")
    builder.add_edge("lineage", "pii_detector")

    # Sequential tail: PII → Semantic → Trust Scorer → Aggregator
    builder.add_edge("pii_detector", "semantic")
    builder.add_edge("semantic", "trust_scorer")
    builder.add_edge("trust_scorer", "aggregator")
    builder.add_edge("aggregator", END)

    return builder.compile()


graph = build_graph()
