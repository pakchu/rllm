from __future__ import annotations

import hashlib
from pathlib import Path


DECISION = Path(
    "docs/daily-treasury-fiscal-flow-breadth-source-axis-decision-2026-07-20.md"
)
EXPECTED_SHA256 = (
    "7ed2748506645aa4b4a9589d7c7d3dd1baa4456c78a7c7f56845358c02f18c5b"
)


def test_dffb_source_axis_decision_is_hash_frozen() -> None:
    assert hashlib.sha256(DECISION.read_bytes()).hexdigest() == EXPECTED_SHA256


def test_dffb_decision_freezes_causal_and_outcome_blind_boundaries() -> None:
    text = DECISION.read_text(encoding="utf-8")
    normalized = " ".join(text.replace("**", "").split()).replace("- ", "-")
    required_clauses = (
        "official historical PDF archive snapshot acquired now",
        "Research stages are assigned by causal",
        "boundary_quarantine",
        "16:00:00 America/New_York",
        "No DTS report PDF or DTS API data row dated `2024-01-01` or later",
        "No BTC OHLC, trade, order-book",
        "Every build binds the parser source SHA-256",
        "every announcement-workbook row with an effective date on or before",
        "New York decision-date Jaccard exceeds `0.30`",
        "absolute signed occupied-",
        "exposure correlation against either prior primary strategy exceeds `0.40`",
    )
    for clause in required_clauses:
        assert clause.replace("- ", "-") in normalized


def test_dffb_decision_does_not_freeze_a_posthoc_rule() -> None:
    text = DECISION.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "does **not** choose" in normalized
    assert (
        "trading rule, feature threshold, sign, holding period, leverage, or model"
        in normalized
    )
    assert "A plain DTS operating-cash/TGA level" in normalized
    assert "may not be repaired in place" in normalized
