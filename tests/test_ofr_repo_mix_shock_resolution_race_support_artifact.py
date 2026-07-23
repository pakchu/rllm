from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

from training import build_ofr_repo_mix_shock_resolution_race_support as support


REPORT = Path("results/ofr_repo_mix_shock_resolution_race_support_2026-07-23.json")
CLOCK = Path("results/ofr_repo_mix_shock_resolution_race_clocks_2026-07-23.csv.gz")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_support_artifacts_are_hash_bound_and_rejected() -> None:
    report = payload()
    assert sha256(REPORT) == (
        "d42b97bb85f75eba4cb45ea3487af27a44e8bc659a1ee07d73656d3ec5f23cf9"
    )
    assert sha256(CLOCK) == (
        "bed2f522bd68fd389cb0af3d655980999324c0e6adb4f1dc027c6b6ea2cc5ae6"
    )
    assert report["manifest_hash"] == (
        "019ebd8ee55689b44c1e027f82ca57c62c2b05385657d5fa4dbd2fc099d016cd"
    )
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == support.canonical_hash(core)
    assert report["clock_artifact"]["sha256"] == sha256(CLOCK)
    assert report["support_builder"]["sha256"] == (
        "d00fa29f04c5eb09ffbc7787ccdf959643d579614f3eca8caa52d3ce8c18100d"
    )
    assert report["disposition"] == "reject RMSR-72 unchanged before outcomes"
    assert report["advance_to_evaluator_freeze"] is False


def test_primary_fails_only_frozen_gap_and_terminal_balance_gates() -> None:
    report = payload()
    assert report["source_support_passed"] is False
    failed = {name for name, passed in report["source_checks"].items() if not passed}
    assert failed == {
        "maximum_entry_gap",
        "selection_each_terminal_type",
        "train_each_terminal_type",
    }
    train = report["primary_support_summaries"]["train"]
    selection = report["primary_support_summaries"]["selection"]
    assert (train["events"], train["longs"], train["shorts"]) == (37, 15, 22)
    assert train["terminal_type_counts"] == {
        "PRICE_CONFIRMATION": 6,
        "QUANTITY_ABSORPTION": 31,
    }
    assert train["maximum_entry_gap_elapsed_days"] == 129.0
    assert (selection["events"], selection["longs"], selection["shorts"]) == (
        33,
        26,
        7,
    )
    assert selection["terminal_type_counts"] == {
        "PRICE_CONFIRMATION": 3,
        "QUANTITY_ABSORPTION": 30,
    }


def test_source_and_race_audits_are_exact() -> None:
    report = payload()
    assert report["source_audit"] == {
        "equal_availability_rows_suppressed": 417,
        "invalid_materiality_dates": 0,
        "invalid_missing_or_null_dates": 4,
        "normalized_rows_read": 77369,
        "required_rows_read": 9976,
        "source_dates_seen": 1249,
        "valid_feature_dates": 1245,
    }
    assert report["race_audits"]["primary"] == {
        "already_priced": 50,
        "ambiguous_same_date": 1,
        "armed": 82,
        "continuity_cancellations": 1,
        "lead_extreme_transitions": 132,
        "price_confirmation": 13,
        "quantity_absorption": 67,
        "timeouts": 0,
    }
    assert report["dominance_diagnostics"]["train"][
        "non_tie_maximum_share"
    ] == 5 / 6
    assert report["dominance_diagnostics"]["selection"][
        "non_tie_maximum_share"
    ] == 16 / 27


def test_clock_is_source_only_and_outcome_boundary_remained_closed() -> None:
    with gzip.open(CLOCK, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == payload()["clock_artifact"]["rows"]
    primary = [row for row in rows if row["control"] == "primary"]
    assert len(primary) == 70
    assert {row["terminal_type"] for row in primary} == {
        "PRICE_CONFIRMATION",
        "QUANTITY_ABSORPTION",
    }
    forbidden = {
        "open",
        "high",
        "low",
        "close",
        "return",
        "pnl",
        "cagr",
        "mdd",
        "funding",
    }
    assert not forbidden.intersection(name.lower() for name in rows[0])

    report = payload()
    boundary = report["outcome_boundary"]
    assert boundary["source_observation_rows_read"] == 77369
    assert boundary["post_2023_source_rows_read"] == 0
    assert boundary["comparator_rows_read"] == 0
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["pnl_cagr_mdd_opened"] is False
    assert boundary["comparator_access_short_circuited_on_source_failure"] is True
    assert report["novelty"]["evaluated"] is False
