"""CLI entry point — run the pipeline on a local file and print Rich output.

Usage:
    python -m context_layer samples/ecommerce.sql
    python -m context_layer samples/ecommerce.csv --type csv
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from context_layer.graph import graph
from context_layer.models.outputs import ContextLayer


def _render(ctx: ContextLayer, console: Console) -> None:
    console.print()
    sensitive = ctx.metadata.sensitive_column_count
    sensitive_text = (
        f"[orange1]Sensitive: {sensitive}[/orange1]"
        if sensitive
        else f"Sensitive: {sensitive}"
    )
    console.print(
        Panel(
            f"[bold]Context Layer Report[/bold]\n"
            f"Tables: {ctx.metadata.table_count}  |  "
            f"Columns: {ctx.metadata.column_count}  |  "
            f"Avg Trust: {ctx.metadata.average_trust:.0%}  |  "
            f"Needs Review: {ctx.metadata.review_count}  |  "
            f"{sensitive_text}",
            border_style="blue",
        )
    )

    for tbl in ctx.tables:
        trust_color = (
            "green" if tbl.trust_score >= 0.8
            else "yellow" if tbl.trust_score >= 0.6
            else "red"
        )
        review = " [red]⚠ NEEDS REVIEW[/red]" if tbl.needs_review else ""

        console.print()
        console.print(
            f"[bold]{tbl.table_name}[/bold]  "
            f"[dim]({tbl.domain})[/dim]  "
            f"[{trust_color}]{tbl.trust_score:.0%} trust[/{trust_color}]"
            f"{review}"
        )
        console.print(f"  [dim]{tbl.definition}[/dim]")

        col_table = Table(show_header=True, header_style="bold dim", padding=(0, 1))
        col_table.add_column("Column", style="magenta")
        col_table.add_column("Type", style="dim")
        col_table.add_column("Definition")
        col_table.add_column("Sensitive")
        col_table.add_column("Trust", justify="right")

        for col in tbl.columns:
            tc = (
                "green" if col.trust_score >= 0.8
                else "yellow" if col.trust_score >= 0.6
                else "red"
            )
            review_mark = " ⚠" if col.needs_review else ""
            sensitive_cell = (
                f"[orange1]PII: {col.pii_category}[/orange1]"
                if col.is_sensitive else "[dim]—[/dim]"
            )
            col_table.add_row(
                col.column_name,
                col.data_type,
                col.definition,
                sensitive_cell,
                f"[{tc}]{col.trust_score:.0%}{review_mark}[/{tc}]",
            )

        console.print(col_table)

        if tbl.relationships:
            for rel in tbl.relationships:
                console.print(
                    f"  [blue]→[/blue] {rel.source_table}.{rel.source_column} → "
                    f"{rel.target_table}.{rel.target_column}  "
                    f"[dim]({rel.relationship_type}, {rel.confidence:.0%})[/dim]"
                )


async def _run(file_path: str, schema_type: str) -> ContextLayer:
    raw = Path(file_path).read_text()
    result = await graph.ainvoke({
        "raw_schema": raw,
        "schema_type": schema_type,
    })
    return result["context_layer"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic Context Layer CLI")
    parser.add_argument("file", help="Path to SQL DDL or CSV file")
    parser.add_argument(
        "--type", dest="schema_type", default="sql", choices=["sql", "csv"],
        help="Schema format (default: sql)",
    )
    parser.add_argument(
        "--json", dest="output_json", action="store_true",
        help="Output raw JSON instead of Rich table",
    )
    args = parser.parse_args()

    console = Console()
    with console.status("[bold blue]Running agent pipeline..."):
        ctx = asyncio.run(_run(args.file, args.schema_type))

    if args.output_json:
        console.print_json(ctx.model_dump_json(indent=2))
    else:
        _render(ctx, console)


if __name__ == "__main__":
    main()
