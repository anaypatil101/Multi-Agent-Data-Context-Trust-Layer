"""Input schema models — represent the raw database structure after parsing."""

from pydantic import BaseModel, Field


class ColumnSchema(BaseModel):
    """A single column extracted from a DDL statement or CSV header."""

    name: str
    data_type: str = Field(description="SQL type as written in DDL, or inferred from CSV")
    nullable: bool = True
    is_primary_key: bool = False
    default_value: str | None = None
    raw_ddl_fragment: str = Field(
        default="",
        description="Original DDL text for this column, preserved for downstream agents",
    )


class ForeignKeyConstraint(BaseModel):
    """An explicit FK constraint parsed from DDL."""

    source_column: str
    target_table: str
    target_column: str


class TableSchema(BaseModel):
    """A full table definition with columns and explicit constraints."""

    name: str
    columns: list[ColumnSchema]
    foreign_keys: list[ForeignKeyConstraint] = Field(default_factory=list)
    raw_ddl: str = Field(
        default="",
        description="Complete CREATE TABLE statement, kept for LLM context",
    )
