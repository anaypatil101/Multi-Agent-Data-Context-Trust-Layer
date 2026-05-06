# Agentic Context Layer for Structured Data

> A multi-agent **LangGraph** pipeline that transforms raw database schemas into a **trustworthy context layer** — governed business definitions, semantic ownership, data lineage, policy-enforced PII classification, and calibrated trust scores for every data asset.

Enterprise AI fails where data governance fails. This pipeline is an automated, auditable first pass at the work Atlan does at scale: enriching ungoverned data assets with the context, semantics, and trust signals that make them safe to reason over.

## The governance gap this solves

Modern data stacks accumulate hundreds of tables and thousands of columns with no business definitions, no ownership, and no documented lineage. When an AI assistant is asked "show me top customers by revenue", it has no way to know that `t_03.amt_v2` is the column it needs — and no way to know whether to trust that column's meaning even if it finds it.

The result is **ungoverned context**: AI that hallucinates answers because the underlying assets have no semantic anchoring, no lineage to explain where the data came from, and no trust signal to say whether the definition is reliable.

This pipeline produces that context layer automatically:

- **Business definitions** — human-readable descriptions of what every column and table actually means, grounded in profiled data patterns
- **Semantic typing** — inferred types (`email`, `currency`, `timestamp`, `identifier`) that AI agents can act on without guessing
- **Data lineage** — explicit foreign key relationships plus inferred cross-table dependencies with confidence scores
- **Policy enforcement** — deterministic PII detection and masking before any sensitive metadata reaches an LLM
- **Trust scores** — calibrated 0–1 confidence per definition, with a full audit trail of which rules fired and what the LLM reasoned
- **Governance health** — per-agent reliability signals (`ok` / `degraded` / `failed`) so consumers know which parts of the context layer to act on and which to treat with caution

**Wrong context is worse than no context.** Every output is scored. Anything below the trust threshold is flagged for human review rather than silently propagated.

## Pipeline at a glance

```
                       ┌─────────────────┐
                       │   InputParser   │  pure Python — no LLM
                       │  DDL / CSV      │
                       └────────┬────────┘
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
         ┌────────────────┐        ┌────────────────────┐
         │   Profiler     │        │      Lineage       │   ← parallel super-step
         │   (fast LLM)   │        │      (fast LLM)    │
         │ types · nulls  │        │ explicit FKs +     │
         │ patterns       │        │ inferred lineage   │
         └────────┬───────┘        └─────────┬──────────┘
                  │                          │
                  └────────────┬─────────────┘
                               ▼
                     ┌──────────────────┐
                     │   PII Detector   │  deterministic — no LLM
                     │  policy gate:    │  classify · mask · flag
                     │  flag + mask     │
                     └─────────┬────────┘
                               ▼
                     ┌──────────────────┐
                     │     Semantic     │  strong LLM
                     │  business        │  governed definitions
                     │  definitions     │  + semantic typing
                     └─────────┬────────┘
                               ▼
                     ┌──────────────────┐
                     │   Trust Scorer   │  strong LLM + rules
                     │  calibrated      │  auditable confidence
                     │  trust · 0→1     │  review flagging
                     └─────────┬────────┘
                               ▼
                     ┌──────────────────┐
                     │    Aggregator    │  deterministic merge
                     │  → ContextLayer  │  governed asset record
                     └──────────────────┘
```

Each agent owns one governance concern. Each agent communicates via strictly typed Pydantic contracts. Each agent's output is independently inspectable and auditable.

## Key design decisions

These are intentional architectural choices, not implementation details.

### 1. LangGraph `StateGraph` — auditable, pausable orchestration

Every agent has a typed input/output contract via Pydantic. The graph state is a `TypedDict` where each agent writes to exactly one key — no shared mutable state, no merge conflicts during parallel execution. You can pause the graph after any node and inspect exactly what it produced. This is the foundation for an auditable governance pipeline: every enrichment decision has a clear owner and a clear output.

### 2. Parallel fan-out for Profiler and Lineage

They're independent governance concerns — data profiling and lineage inference have no data dependency on each other. Running them in the same LangGraph super-step halves wall-clock latency without adding orchestration complexity. The fan-in is automatic: LangGraph won't advance to the PII gate until both branches have written their results to state.

### 3. PII detection is a policy gate, not an LLM call

The PII Detector is the compliance boundary between raw schema data and the LLM layer. It uses **rule-based name matching corroborated by the profiler's semantic-type signals** to classify sensitive assets. Three reasons deterministic rules, not an LLM:

- **Auditability** — governance teams need to explain *why* a column was classified as sensitive. Rules produce a traceable classification decision.
- **Privacy** — sending column metadata to an LLM to ask "is this PII?" is itself a data governance violation. The policy gate exists to prevent exactly this.
- **Reliability** — PII categories (`email`, `ssn`, `dob`, `phone`) are well-defined regulatory concepts. Pattern matching catches them with higher precision than probabilistic inference.

When a column is flagged, its DDL fragment is replaced with `[MASKED:<category>]` before the Semantic Agent sees it. The Semantic Agent produces a governance-safe definition based on the classification alone, with no speculation about actual values.

### 4. Hybrid trust scoring — governance-grade confidence

**Ungoverned definitions are worse than missing definitions.** The Trust Scorer combines two independent signals to produce calibrated, explainable confidence:

| Component             | Weight | Catches                                              | Weakness                              |
|-----------------------|:------:|------------------------------------------------------|---------------------------------------|
| Deterministic rules   |  0.4   | Type-name mismatches, ambiguous names, high null rates | Can't evaluate semantic accuracy      |
| LLM assessment        |  0.6   | Wrong business domain, vague or contradictory definitions | Can hallucinate confidence       |

The deterministic component acts as a **governance floor**: if structural signals indicate a low-quality definition, no LLM confidence level can push the score above the `0.6` review threshold. Every score ships with a full breakdown — which rules fired, what the LLM reasoned — making trust explainable to data stewards and stakeholders.

### 5. Strict Pydantic contracts — fail at the governance boundary

No raw `dict` crosses an agent boundary. If the LLM returns semantically malformed output, the failure surfaces at the Pydantic validation layer — not three agents downstream as a silent data quality error. Governance pipelines must fail loudly and early; silent errors propagate as trusted context.

### 6. Graceful degradation — reliability without false confidence

Every LLM-calling agent is wrapped in **bounded retries with exponential backoff** via a shared helper (`agents/_retry.py`). Hard caps prevent token burn and infinite loops:

- **3 attempts maximum** (1 initial + 2 retries) — a module-level constant, not configurable per call-site.
- **Non-retryable errors break immediately** — auth failures and context-length errors will never succeed on retry; continuing wastes tokens.
- **Exponential backoff** — 1s, then 2s — gives rate-limited APIs time to recover.

When retries are exhausted, each agent emits a **degraded but type-safe output** rather than crashing the pipeline. The context layer is still produced — but its trust signals communicate exactly how much of it to believe:

| Agent | On failure | Governance impact |
|---|---|---|
| Profiler | Empty `ProfilerOutput` — downstream runs on structure alone | Definitions lose data-pattern grounding |
| Lineage | Explicit FKs preserved (deterministic); inferred lineage omitted | Lineage graph is incomplete, not wrong |
| Semantic | Empty definitions for every column | Trust Scorer floors all affected scores |
| Trust Scorer | All scores floored to 0.0, `needs_review=True`, `upstream_failure` flag | Human review required before use |

The Trust Scorer reads `agent_health` from upstream state. If Semantic failed, it skips its own LLM batch entirely — no point scoring empty definitions, and no reason to emit false-confident scores on ungoverned data.

Every run surfaces governance health in the final context layer:

```jsonc
"metadata": {
  "agent_health": {
    "profiler":      "ok",
    "lineage":       "ok",
    "semantic":      "ok",
    "trust_scorer":  "ok"
  }
}
```

Possible values per agent: `"ok"` / `"degraded"` / `"failed"`. Consumers can gate on this before acting on the definitions.

## Output: what a governed data asset looks like

For every column, the pipeline emits a governed asset record:

```jsonc
{
  "column_name": "email",
  "data_type": "VARCHAR(255)",
  "definition": "Contact email address for the registered user account.",
  "business_context": "Sensitive PII — used for transactional and marketing communication.",
  "semantic_type": "email",
  "is_sensitive": true,
  "pii_category": "email",
  "trust_score": 0.92,
  "needs_review": false,
  "trust_flags": []
}
```

For every table: a business definition, inferred domain, data lineage (explicit FKs and confidence-scored inferred relationships), table-level trust score, and a review flag.

The top-level `ContextLayer` includes aggregate governance metadata: asset counts, average trust across the layer, review queue size, sensitive asset count, models used per agent, generation timestamp, and per-agent execution health.

## Quick start

### Prerequisites
- Python 3.11+
- An OpenAI API key (or Anthropic — see `llm.py`)

### Install
```bash
git clone <repo-url> && cd agentic-data-context-layer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env       # add OPENAI_API_KEY
```

### Run via CLI

```bash
python -m context_layer samples/ecommerce.sql            # Rich-formatted output
python -m context_layer samples/ecommerce.sql --json     # raw JSON context layer
python -m context_layer samples/ecommerce.csv --type csv # CSV schema input
```

### Run via API

```bash
uvicorn context_layer.api:app --reload
```

```bash
# JSON governed context layer
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"schema_text": "CREATE TABLE users (id INT PRIMARY KEY, email VARCHAR(255));", "schema_type": "sql"}'

# Rendered HTML governance report
curl -X POST http://localhost:8000/analyze/html \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{"schema_text": "CREATE TABLE users (id INT PRIMARY KEY, email VARCHAR(255));", "schema_type": "sql"}
EOF
```

Auto-generated OpenAPI docs: `http://localhost:8000/docs`.

### Run tests (no live LLM calls)

```bash
pytest tests/ -v
```

Two test scenarios:
- **Happy path** — all agents succeed, full governed output verified end-to-end.
- **Degraded path** — Semantic Agent fails all retries; asserts pipeline completes, Trust Scorer makes zero LLM calls, every definition is floored to 0.0 trust with `upstream_failure` flag, `agent_health["semantic"] == "failed"`.

## Project structure

```
src/context_layer/
├── models/
│   ├── schema.py        # Input: TableSchema, ColumnSchema, ForeignKeyConstraint
│   ├── outputs.py       # Agent outputs + ContextLayer + AgentHealth
│   └── state.py         # LangGraph PipelineState (TypedDict + agent_health reducer)
├── agents/
│   ├── _retry.py        # Bounded retry helper (MAX_RETRIES=2, non-retryable detection)
│   ├── input_parser.py  # DDL / CSV → structured asset definitions (pure Python)
│   ├── profiler.py      # Data profiling: types, null rates, value patterns (fast LLM + retry)
│   ├── lineage.py       # Data lineage: explicit FKs + inferred relationships (fast LLM + retry)
│   ├── pii_detector.py  # Policy gate: PII classification + masking (deterministic)
│   ├── semantic.py      # Business definitions + semantic typing (strong LLM + retry + fallback)
│   ├── trust_scorer.py  # Trust calibration: rules × LLM + upstream-failure floor
│   └── aggregator.py    # Governed asset record assembly (deterministic)
├── graph.py             # StateGraph: parallel lineage/profiling, policy gate, trust scoring
├── llm.py               # Tier-based LLM factory (fast/strong, provider-agnostic)
├── api.py               # FastAPI — governed context layer as a service
└── __main__.py          # Rich CLI

templates/report.html    # Governance report template (Jinja2)
samples/                 # ecommerce.sql, ecommerce.csv — sample ungoverned schemas
tests/                   # Mocked end-to-end tests: happy path + degraded governance
```

## Models used per agent

| Agent          | Model              | Why this tier                                                            |
|----------------|--------------------|--------------------------------------------------------------------------|
| InputParser    | none               | Pure Python — DDL parsing needs no LLM inference                         |
| Profiler       | gpt-4o-mini        | Structured data extraction; speed matters more than semantic depth       |
| Lineage        | gpt-4o-mini        | Name-pattern relationship inference; strong reasoning not required       |
| PII Detector   | none               | Policy enforcement demands deterministic, auditable classification rules |
| Semantic       | gpt-4o             | Business definition quality is the core governance deliverable           |
| Trust Scorer   | gpt-4o             | Calibrated trust assessment requires strong critical reasoning           |
| Aggregator     | none               | Deterministic assembly — no LLM needed for merging typed structs         |

Model identifiers are env-overridable via `FAST_MODEL` and `STRONG_MODEL` in `.env`. Swap providers by changing `langchain_openai` to `langchain_anthropic` in `llm.py`.

## What this project demonstrates

- **Data governance pipeline** — automated enrichment of ungoverned schemas into trustworthy context assets, with full audit trails
- **Semantic understanding at scale** — business definitions, semantic typing, and domain classification without manual annotation
- **Lineage inference** — explicit FK detection plus confidence-scored cross-table relationship inference
- **Policy-enforced privacy** — deterministic PII classification gate that prevents sensitive metadata from reaching the LLM layer
- **Calibrated trust, not blind confidence** — hybrid rules × LLM scoring with a deterministic floor; every score is explainable to data stewards
- **Governance-aware reliability** — bounded retries, typed degraded fallbacks, upstream-failure flooring, `agent_health` in every response so consumers know what to trust
- **LangGraph orchestration** — typed state, parallel fan-out/fan-in, interleaved deterministic and LLM nodes, auditable at every step
- **Context layer as a service** — FastAPI endpoint, OpenAPI docs, Rich CLI, and a self-contained HTML governance report
