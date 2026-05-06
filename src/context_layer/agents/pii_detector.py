"""PII Detector — deterministic identification of sensitive columns.

DESIGN RATIONALE:
  PII detection is deliberately deterministic (no LLM) for three reasons:

  1. AUDITABILITY: Compliance teams need to explain WHY a column was
     flagged. Rule-based matches give a paper trail; LLM judgments don't.

  2. PRIVACY: Sending column metadata to an LLM to ask "is this PII?"
     would itself be a potential leak. The whole point of PII detection
     is to prevent sensitive data from reaching the LLM downstream.

  3. PRECISION: PII categories are well-defined. Column names follow
     industry conventions (`email`, `ssn`, `dob`). A regex/lookup catches
     these reliably and quickly without API costs.

  This node sits AFTER Profiler and Lineage (so we can corroborate with
  the profiler's inferred semantic type) and BEFORE Semantic (so we can
  mask sensitive content in the LLM prompt).
"""

from __future__ import annotations

import re

from context_layer.models.outputs import (
    PIIColumnFlag,
    PIIOutput,
    ProfilerOutput,
)
from context_layer.models.schema import TableSchema
from context_layer.models.state import PipelineState


# ---------------------------------------------------------------------------
# Detection rules
# ---------------------------------------------------------------------------
# Column-name keywords mapped to PII category. Matched as whole-word or
# snake_case substring against the lower-cased column name.
PII_PATTERNS: dict[str, list[str]] = {
    "email": ["email", "e_mail", "email_address", "mail_address"],
    "phone": ["phone", "phone_number", "mobile", "cell", "telephone", "fax"],
    "name": [
        "first_name", "last_name", "full_name", "surname", "given_name",
        "family_name", "middle_name", "maiden_name",
    ],
    "address": [
        "address", "street", "city", "state_code", "zip", "zipcode",
        "postal", "postcode", "country", "region",
    ],
    "ssn": [
        "ssn", "social_security", "social_security_number", "sin",
        "national_id", "tax_id", "tin",
    ],
    "dob": [
        "dob", "date_of_birth", "birth_date", "birthday", "birthdate",
    ],
    "financial": [
        "credit_card", "card_number", "card_no", "cc_number", "ccn",
        "account_number", "acct_number", "routing_number", "iban",
        "swift", "bank_account",
    ],
    "ip": [
        "ip_address", "ip_addr", "client_ip", "remote_ip", "src_ip", "dst_ip",
    ],
    "credential": [
        "password", "passwd", "pwd", "secret", "token", "api_key",
        "access_token", "refresh_token", "private_key", "auth_token",
    ],
}

# Profiler-detected semantic types that map directly to PII categories.
# Used for corroboration: name + profiler agreement = high confidence.
SEMANTIC_TO_PII: dict[str, str] = {
    "email": "email",
    "phone": "phone",
    "address": "address",
    "name": "name",
}


def _classify_column(
    column_name: str, semantic_type: str | None
) -> tuple[str | None, float, str]:
    """Return (pii_category, confidence, reasoning) for a column.

    Confidence model:
      - Strong name match alone: 0.85
      - Strong name match + profiler agreement: 1.0
      - Profiler-only signal (no name match): 0.7
      - No signal: returns (None, 0.0, "").
    """
    name_lower = column_name.lower()
    name_match: str | None = None

    # Pass 1: exact-word match (highest precision, e.g. "email" == "email")
    for category, keywords in PII_PATTERNS.items():
        if name_lower in keywords:
            name_match = category
            break

    # Pass 2: substring match on snake_case parts (e.g. "user_email" → email)
    if name_match is None:
        parts = re.split(r"[_\s]+", name_lower)
        for category, keywords in PII_PATTERNS.items():
            keyword_set = set(keywords)
            single_words = {kw for kw in keyword_set if "_" not in kw}
            if any(p in single_words for p in parts):
                name_match = category
                break
            # Also catch full-keyword substrings inside the name
            if any(kw in name_lower for kw in keyword_set if "_" in kw):
                name_match = category
                break

    profiler_pii = SEMANTIC_TO_PII.get((semantic_type or "").lower())

    if name_match and profiler_pii:
        if name_match == profiler_pii:
            return (
                name_match,
                1.0,
                f"name keyword '{name_lower}' + profiler semantic_type='{semantic_type}' agree",
            )
        # Conflicting signals: trust the name (more specific) but lower confidence
        return (
            name_match,
            0.75,
            f"name suggests {name_match} but profiler said {profiler_pii}; deferring to name",
        )

    if name_match:
        return (
            name_match,
            0.85,
            f"name keyword match for '{name_lower}' → {name_match}",
        )

    if profiler_pii:
        return (
            profiler_pii,
            0.7,
            f"profiler-only signal: semantic_type='{semantic_type}' → {profiler_pii}",
        )

    return None, 0.0, ""


def _mask_ddl_fragment(raw: str, column_name: str, category: str) -> str:
    """Replace any in-fragment values that could leak PII signals.

    The raw DDL fragment is mostly type info, but DEFAULT clauses can
    contain example values. We replace the entire post-type remainder
    with a redaction marker, keeping only the column name + type.
    """
    if not raw:
        return f"{column_name} [MASKED:{category}]"

    # Keep "name TYPE", strip everything after (DEFAULTs, comments, etc.)
    match = re.match(r"^\s*[`\"']?\w+[`\"']?\s+([\w]+(?:\([^)]*\))?)", raw)
    if match:
        col_type = match.group(1)
        return f"{column_name} {col_type} [MASKED:{category}]"
    return f"{column_name} [MASKED:{category}]"


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def pii_detector_node(state: PipelineState) -> dict:
    """Identify sensitive columns; produce flags + masked DDL fragments."""
    import time as _time

    tables: list[TableSchema] = state["tables"]
    profiler: ProfilerOutput = state["profiler_output"]
    logger = state.get("run_logger")

    t0 = _time.monotonic()

    semantic_lookup: dict[tuple[str, str], str] = {}
    for tp in profiler.tables:
        for cp in tp.column_profiles:
            semantic_lookup[(tp.table_name, cp.column_name)] = (
                cp.inferred_semantic_type
            )

    flags: list[PIIColumnFlag] = []
    for table in tables:
        for col in table.columns:
            semantic_type = semantic_lookup.get((table.name, col.name))
            category, confidence, reasoning = _classify_column(
                col.name, semantic_type
            )
            if category is None:
                continue

            flags.append(PIIColumnFlag(
                table_name=table.name,
                column_name=col.name,
                pii_category=category,
                confidence=round(confidence, 3),
                masked_ddl_fragment=_mask_ddl_fragment(
                    col.raw_ddl_fragment, col.name, category
                ),
                reasoning=reasoning,
            ))

    elapsed = (_time.monotonic() - t0) * 1000

    if logger:
        logger.log(
            agent="pii_detector",
            latency_ms=elapsed,
            health="ok",
            response_preview=f"{len(flags)} sensitive columns flagged",
        )

    return {
        "pii_output": PIIOutput(
            flagged_columns=flags,
            sensitive_column_count=len(flags),
        )
    }
