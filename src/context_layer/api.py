"""FastAPI application exposing the context layer pipeline.

Endpoints:
  POST /analyze      — JSON in, JSON out (for programmatic consumers)
  POST /analyze/html — JSON in, rendered HTML report (for humans)
  GET  /runs         — list recent run audit trails
  GET  /runs/{id}    — retrieve full per-agent audit trail for a run
  GET  /health       — liveness check
"""

from __future__ import annotations

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
