"""InputParser node — pure-Python extraction, no LLM calls.

Transforms raw DDL or CSV text into a list of structured TableSchema objects.
Keeping this deterministic avoids burning tokens on something regex can handle
and makes the first pipeline step fast and fully reproducible.
"""

from __future__ import annotations

import csv
import io
import re

from context_layer.models.schema import ColumnSchema, ForeignKeyConstraint, TableSchema
from context_layer.models.state import PipelineState


# ---------------------------------------------------------------------------
# SQL DDL parsing
# ---------------------------------------------------------------------------

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"[`\"']?(\w+)[`\"']?\s*\((.*?)\)\s*;",
    re.IGNORECASE | re.DOTALL,
)

_FK_INLINE_RE = re.compile(
    r"FOREIGN\s+KEY\s*\(\s*[`\"']?(\w+)[`\"']?\s*\)\s*"
    r"REFERENCES\s+[`\"']?(\w+)[`\"']?\s*\(\s*[`\"']?(\w+)[`\"']?\s*\)",
    re.IGNORECASE,
)

_COL_RE = re.compile(
    r"^\s*[`\"']?(\w+)[`\"']?\s+"
    r"([\w]+(?:\s*\([^)]*\))?)"
    r"(.*)",
    re.IGNORECASE,
)


def _split_columns(body: str) -> list[str]:
    """Split a CREATE TABLE body on commas, respecting parenthesised groups.

    Naive str.split(',') breaks types like DECIMAL(10,2). This tracks
    paren depth so commas inside parens are preserved.
    """
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in body:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def _parse_ddl(raw: str) -> list[TableSchema]:
    tables: list[TableSchema] = []

    for match in _CREATE_TABLE_RE.finditer(raw):
        table_name = match.group(1)
        body = match.group(2)
        columns: list[ColumnSchema] = []
        foreign_keys: list[ForeignKeyConstraint] = []

        for fk in _FK_INLINE_RE.finditer(body):
            foreign_keys.append(
                ForeignKeyConstraint(
                    source_column=fk.group(1),
                    target_table=fk.group(2),
                    target_column=fk.group(3),
                )
            )

        for line in _split_columns(body):
            line = line.strip()
            if not line:
                continue
            upper = line.upper()
            if any(
                upper.startswith(kw)
                for kw in ("PRIMARY KEY", "FOREIGN KEY", "UNIQUE", "INDEX", "CONSTRAINT", "CHECK")
            ):
                if upper.startswith("PRIMARY KEY"):
                    pk_match = re.search(r"\((.+?)\)", line)
                    if pk_match:
                        pk_cols = {
                            c.strip().strip("`\"'")
                            for c in pk_match.group(1).split(",")
                        }
                        for col in columns:
                            if col.name in pk_cols:
                                col.is_primary_key = True
                continue

            col_match = _COL_RE.match(line)
            if not col_match:
                continue

            name = col_match.group(1)
            data_type = col_match.group(2).upper()
            remainder = col_match.group(3).upper()

            columns.append(
                ColumnSchema(
                    name=name,
                    data_type=data_type,
                    nullable="NOT NULL" not in remainder,
                    is_primary_key="PRIMARY KEY" in remainder,
                    default_value=_extract_default(remainder),
                    raw_ddl_fragment=line.strip(),
                )
            )

        tables.append(
            TableSchema(
                name=table_name,
                columns=columns,
                foreign_keys=foreign_keys,
                raw_ddl=match.group(0),
            )
        )
    return tables


def _extract_default(remainder: str) -> str | None:
    m = re.search(r"DEFAULT\s+(\S+)", remainder, re.IGNORECASE)
    return m.group(1).strip("'\"") if m else None


# ---------------------------------------------------------------------------
# CSV parsing (header-row schema inference)
# ---------------------------------------------------------------------------

_TYPE_INFERENCE: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\d{4}-\d{2}-\d{2}"), "TIMESTAMP"),
    (re.compile(r"^-?\d+\.\d+$"), "DECIMAL"),
    (re.compile(r"^-?\d+$"), "INTEGER"),
    (re.compile(r"^(true|false)$", re.I), "BOOLEAN"),
]


def _infer_csv_type(values: list[str]) -> str:
    non_empty = [v for v in values if v.strip()]
    if not non_empty:
        return "TEXT"
    for pattern, sql_type in _TYPE_INFERENCE:
        if all(pattern.match(v) for v in non_empty):
            return sql_type
    return "TEXT"


def _parse_csv(raw: str) -> list[TableSchema]:
    reader = csv.reader(io.StringIO(raw))
    rows = list(reader)
    if not rows:
        return []

    headers = [h.strip() for h in rows[0]]
    sample_rows = rows[1:11]

    col_values: dict[int, list[str]] = {i: [] for i in range(len(headers))}
    for row in sample_rows:
        for i, val in enumerate(row):
            if i < len(headers):
                col_values[i].append(val.strip())

    columns = [
        ColumnSchema(
            name=h,
            data_type=_infer_csv_type(col_values.get(i, [])),
            nullable=True,
            raw_ddl_fragment=h,
        )
        for i, h in enumerate(headers)
    ]

    return [
        TableSchema(
            name="imported_table",
            columns=columns,
            raw_ddl=raw[:500],
        )
    ]


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

def input_parser_node(state: PipelineState) -> dict:
    """Parse raw schema text into structured TableSchema objects."""
    raw = state["raw_schema"]
    schema_type = state.get("schema_type", "sql")

    if schema_type == "csv":
        tables = _parse_csv(raw)
    else:
        tables = _parse_ddl(raw)

    return {"tables": tables}
