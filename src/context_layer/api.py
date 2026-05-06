"""FastAPI application exposing the context layer pipeline.

Endpoints:
  POST /analyze        — JSON in, JSON out (for programmatic consumers)
  POST /analyze/html   — JSON in, rendered HTML report (for humans)
  GET  /analyze/demo   — full pipeline run against a hardcoded e-commerce schema (JSON)
  GET  /demo/html      — live governance report for the hardcoded schema, with loading UI (HTML)
  GET  /runs           — list recent run audit trails
  GET  /runs/{id}      — retrieve full per-agent audit trail for a run
  GET  /health         — liveness check
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field

from context_layer.graph import graph
from context_layer.models.outputs import ContextLayer
from context_layer.run_logger import RunLogger

app = FastAPI(
    title="Agentic Context Layer",
    description="Multi-agent pipeline that generates trusted context for database schemas",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Demo schema — hardcoded so deployment has no file-path dependency
# ---------------------------------------------------------------------------
_DEMO_SCHEMA = """\
-- E-commerce sample schema.
-- Includes explicit FKs, intentionally ambiguous columns (data, val, misc_flag),
-- and type-name mismatches to exercise trust scoring edge cases.

CREATE TABLE users (
    id          INTEGER      NOT NULL PRIMARY KEY,
    email       VARCHAR(255) NOT NULL,
    name        VARCHAR(128) NOT NULL,
    phone       VARCHAR(20),
    created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status      VARCHAR(20)  NOT NULL DEFAULT 'active',
    data        TEXT
);

CREATE TABLE categories (
    id          INTEGER      NOT NULL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    parent_id   INTEGER,
    description TEXT,
    FOREIGN KEY (parent_id) REFERENCES categories(id)
);

CREATE TABLE products (
    id          INTEGER      NOT NULL PRIMARY KEY,
    sku         VARCHAR(50)  NOT NULL,
    name        VARCHAR(255) NOT NULL,
    description TEXT,
    price       DECIMAL(10,2) NOT NULL,
    category_id INTEGER      NOT NULL,
    created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    val         INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(id)
);

CREATE TABLE orders (
    id          INTEGER      NOT NULL PRIMARY KEY,
    user_id     INTEGER      NOT NULL,
    order_date  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    total       DECIMAL(10,2) NOT NULL,
    status      VARCHAR(20)  NOT NULL DEFAULT 'pending',
    misc_flag   BOOLEAN      DEFAULT FALSE,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE order_items (
    id          INTEGER      NOT NULL PRIMARY KEY,
    order_id    INTEGER      NOT NULL,
    product_id  INTEGER      NOT NULL,
    quantity    INTEGER      NOT NULL DEFAULT 1,
    unit_price  DECIMAL(10,2) NOT NULL,
    FOREIGN KEY (order_id)   REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

CREATE TABLE payments (
    id              INTEGER      NOT NULL PRIMARY KEY,
    order_id        INTEGER      NOT NULL,
    amount          DECIMAL(10,2) NOT NULL,
    method          VARCHAR(30)  NOT NULL,
    paid_at         TIMESTAMP,
    confirmation_id VARCHAR(100),
    x               INTEGER,
    FOREIGN KEY (order_id) REFERENCES orders(id)
);
"""

# In-memory cache so repeated visits to /demo/html are instant.
# The first request runs the full pipeline (~25-35 s); all subsequent
# requests return the cached ContextLayer with no additional LLM cost.
_demo_result: ContextLayer | None = None
_demo_lock = asyncio.Lock()

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "templates"
_jinja_env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)))


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    schema_text: str = Field(
        ...,
        description="Raw SQL DDL or CSV content to analyse",
        min_length=1,
    )
    schema_type: Literal["sql", "csv"] = Field(
        default="sql",
        description="Format of the input schema",
    )


# ---------------------------------------------------------------------------
# Pipeline endpoints
# ---------------------------------------------------------------------------

@app.post("/analyze", response_model=ContextLayer)
async def analyze(req: AnalyzeRequest) -> ContextLayer:
    """Run the full agent pipeline and return the context layer as JSON."""
    logger = RunLogger()
    try:
        result = await graph.ainvoke({
            "raw_schema": req.schema_text,
            "schema_type": req.schema_type,
            "run_id": logger.run_id,
            "run_logger": logger,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}") from e

    return result["context_layer"]


@app.post("/analyze/html", response_class=HTMLResponse)
async def analyze_html(req: AnalyzeRequest) -> HTMLResponse:
    """Run the pipeline and return an HTML report."""
    logger = RunLogger()
    try:
        result = await graph.ainvoke({
            "raw_schema": req.schema_text,
            "schema_type": req.schema_type,
            "run_id": logger.run_id,
            "run_logger": logger,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}") from e

    ctx: ContextLayer = result["context_layer"]
    template = _jinja_env.get_template("report.html")
    html = template.render(context=ctx)
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# Demo endpoints — no user input, hardcoded e-commerce schema
# ---------------------------------------------------------------------------

async def _run_demo_pipeline() -> ContextLayer:
    """Run the full pipeline against the hardcoded demo schema.

    Result is cached in-process after the first call so subsequent visitors
    get an instant response with no additional LLM cost.
    """
    global _demo_result
    async with _demo_lock:
        if _demo_result is not None:
            return _demo_result
        logger = RunLogger()
        try:
            result = await graph.ainvoke({
                "raw_schema": _DEMO_SCHEMA,
                "schema_type": "sql",
                "run_id": logger.run_id,
                "run_logger": logger,
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Demo pipeline failed: {e}") from e
        ctx: ContextLayer = result["context_layer"]
        ctx.metadata.demo = True
        _demo_result = ctx
        return ctx


@app.get("/analyze/demo", response_model=ContextLayer)
async def analyze_demo() -> ContextLayer:
    """Demo endpoint — runs the full 6-agent pipeline against a hardcoded e-commerce schema.

    No user input required. The OpenAI API key is still required server-side.
    Results are cached after the first run so subsequent calls are instant.
    The response includes ``metadata.demo = true`` so downstream consumers
    know this is sample data, not a production schema.
    """
    return await _run_demo_pipeline()


@app.get("/demo/html", response_class=HTMLResponse)
async def demo_html() -> HTMLResponse:
    """Serves the live governance report for the hardcoded e-commerce demo schema.

    Opens a browser-friendly page with a loading animation while the pipeline
    runs, then renders the full context layer report inline via JavaScript.
    No user input required.
    """
    template = _jinja_env.get_template("demo.html")
    return HTMLResponse(content=template.render())


# ---------------------------------------------------------------------------
# Audit trail endpoints
# ---------------------------------------------------------------------------

@app.get("/runs")
async def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    """List recent pipeline runs with timestamps, newest first."""
    return RunLogger.list_runs(limit=limit)


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> list[dict[str, Any]]:
    """Retrieve the full per-agent audit trail for a specific run."""
    entries = RunLogger.read(run_id)
    if not entries:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return entries


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
