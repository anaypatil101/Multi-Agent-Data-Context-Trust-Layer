"""Per-run audit trail logger.

Every pipeline run gets a UUID. Each agent logs an entry with its prompt,
response preview, latency, retry count, and health status. The entries
are flushed as JSONL to `runs/{run_id}.jsonl` at the end of the run,
giving full observability into what every agent did — without coupling
the agents to a specific logging backend.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_PREVIEW_LIMIT = 500
_RUNS_DIR = Path(__file__).resolve().parent.parent.parent / "runs"


def _preview(text: str | None, limit: int = _PREVIEW_LIMIT) -> str:
    if text is None:
        return ""
    return text[:limit] + ("..." if len(text) > limit else "")


class RunLogger:
    """Collects per-agent audit entries for a single pipeline run."""

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._entries: list[dict[str, Any]] = []

    def log(
        self,
        *,
        agent: str,
        latency_ms: float = 0.0,
        attempts: int = 1,
        health: str = "ok",
        prompt_preview: str | None = None,
        response_preview: str | None = None,
        error: str | None = None,
    ) -> None:
        """Append one audit entry for an agent execution."""
        self._entries.append({
            "run_id": self.run_id,
            "agent": agent,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": round(latency_ms, 1),
            "attempts": attempts,
            "health": health,
            "prompt_preview": _preview(prompt_preview),
            "response_preview": _preview(response_preview),
            "error": error,
        })

    def flush(self) -> Path:
        """Write all entries as JSONL and return the file path."""
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        path = _RUNS_DIR / f"{self.run_id}.jsonl"
        with open(path, "w") as f:
            for entry in self._entries:
                f.write(json.dumps(entry) + "\n")
        return path

    @property
    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    @staticmethod
    def read(run_id: str) -> list[dict[str, Any]]:
        """Read a previously flushed audit trail by run_id."""
        path = _RUNS_DIR / f"{run_id}.jsonl"
        if not path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    @staticmethod
    def list_runs(limit: int = 20) -> list[dict[str, Any]]:
        """List recent run IDs with timestamps, newest first."""
        if not _RUNS_DIR.exists():
            return []
        files = sorted(_RUNS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        results: list[dict[str, Any]] = []
        for f in files[:limit]:
            run_id = f.stem
            first_line = ""
            with open(f) as fh:
                first_line = fh.readline().strip()
            ts = None
            if first_line:
                try:
                    ts = json.loads(first_line).get("timestamp")
                except (json.JSONDecodeError, KeyError):
                    pass
            results.append({
                "run_id": run_id,
                "timestamp": ts,
                "size_bytes": f.stat().st_size,
            })
        return results
