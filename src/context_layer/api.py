"""FastAPI application exposing the context layer pipeline.

Three endpoints:
  POST /analyze      — JSON in, JSON out (for programmatic consumers)
  POST /analyze/html — JSON in, rendered HTML report (for humans)
  GET  /health       — liveness check
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field

from context_layer.graph import graph
from context_layer.models.outputs import ContextLayer

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
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/analyze", response_model=ContextLayer)
async def analyze(req: AnalyzeRequest) -> ContextLayer:
    """Run the full agent pipeline and return the context layer as JSON."""
    try:
        result = await graph.ainvoke({
            "raw_schema": req.schema_text,
            "schema_type": req.schema_type,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}") from e

    return result["context_layer"]


@app.post("/analyze/html", response_class=HTMLResponse)
async def analyze_html(req: AnalyzeRequest) -> HTMLResponse:
    """Run the pipeline and return an HTML report."""
    try:
        result = await graph.ainvoke({
            "raw_schema": req.schema_text,
            "schema_type": req.schema_type,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}") from e

    ctx: ContextLayer = result["context_layer"]
    template = _jinja_env.get_template("report.html")
    html = template.render(context=ctx)
    return HTMLResponse(content=html)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
