from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from training import (
    build_paired_intrinsic_venue_orderflow_topology_support as support,
)


REPORT = Path(
    "results/paired_intrinsic_venue_orderflow_topology_"
    "support_2026-07-24.json"
)
CLOCK = Path(
    "data/paired_intrinsic_venue_orderflow_topology_"
    "states_2020_2023.csv.gz"
)
REPORT_SHA256 = (
    "d20a2647017bce5b6c8a8c8993d3b5aca9307aae457e433a59128f1c6dd2db5b"
)
CLOCK_SHA256 = (
    "2828d9f0092ada7578e7297420cc95c6f6fa76c050458dfd9b5b3ed55ec5ae3e"
)
MANIFEST_HASH = (
    "b7690b1d2a6bc864c9a16b162917407b98011f6c00c5428a4b658732bb63d186"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_support_artifacts_are_hash_frozen() -> None:
    assert _sha256(REPORT) == REPORT_SHA256
    assert _sha256(CLOCK) == CLOCK_SHA256
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    core = {
        key: value for key, value in report.items() if key != "manifest_hash"
    }
    assert support.canonical_hash(core) == MANIFEST_HASH
    assert report["manifest_hash"] == MANIFEST_HASH
    assert report["clock"]["sha256"] == CLOCK_SHA256


def test_support_failure_retires_pivot_before_outcomes() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["source_support_passed"] is False
    assert report["first_failing_check"] == "eval_opportunities"
    assert report["decision"] == "retire_PIVOT_72_unchanged_before_outcomes"
    assert report["authorized_next_stage"] is None
    assert report["outcomes_opened"] is False
    assert report["market_values_loaded"] is False
    assert report["funding_values_loaded"] is False
    assert report["comparator_rows_decoded"] == 0
    assert report["post_2023_values_decoded"] is False
    assert report["outcome_boundary"] == {
        "source_rows_decoded": 420_768,
        "market_rows_decoded": 0,
        "funding_rows_decoded": 0,
        "comparator_rows_decoded": 0,
        "post_entry_price_rows_decoded": 0,
        "future_return_rows_decoded": 0,
        "return_or_pnl_fields_decoded": 0,
        "network_calls": 0,
    }


def test_support_failure_is_localized_to_2023_coverage_and_one_quartile() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    failed = {
        name
        for name, passed in report["source_support_checks"].items()
        if not passed
    }
    assert failed == {
        "eval_opportunities",
        "eval_each_half",
        "eval_each_quarter",
        "eval_active_months",
        "eval_maximum_month_share",
        "eval_maximum_entry_gap",
        "eval:spot_late_abs_flow_q:Q3",
    }
    temporal = report["temporal_statistics"]
    assert temporal["global"]["opportunities"] == 1_012
    assert temporal["train"]["opportunities"] == 546
    assert temporal["selection"]["opportunities"] == 316
    assert temporal["eval"]["opportunities"] == 150
    assert temporal["eval"]["active_months"] == 9
    assert temporal["eval"]["half_counts"] == [109, 41]
    assert temporal["eval"]["quarter_gate_counts"] == [49, 60, 3, 38]
    assert temporal["eval"]["maximum_entry_gap_days"] > 125.0
    assert report["token_statistics"]["eval"]["levels"][
        "spot_late_abs_flow_q"
    ]["shares"]["Q3"] == 0.5


def test_clock_contains_only_source_state_and_no_outcome_fields() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    with gzip.open(CLOCK, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\n").split(",")
        rows = sum(1 for _ in handle)
    assert header == list(support.CLOCK_COLUMNS)
    assert rows == 1_014
    assert report["clock"]["rows"] == rows
    assert report["clock"]["globally_reserved_rows"] == 1_013
    assert report["clock"]["split_contained_rows"] == 1_012
    assert not any(
        token in column.lower()
        for column in header
        for token in support.FORBIDDEN_CLOCK_TOKENS
    )
