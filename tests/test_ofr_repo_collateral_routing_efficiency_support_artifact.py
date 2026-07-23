from __future__ import annotations

import csv
import gzip
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from training import build_ofr_repo_collateral_routing_efficiency_support as support


REPORT = Path(
    "results/ofr_repo_collateral_routing_efficiency_support_2026-07-23.json"
)
CLOCK = Path(
    "results/ofr_repo_collateral_routing_efficiency_clocks_2026-07-23.csv.gz"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_support_artifacts_are_hash_bound_and_rejected_before_outcomes() -> None:
    report = payload()
    assert sha256(REPORT) == (
        "cd0ce324dfd5661898cee30603500eaf3e76f33604097392c765d7d1386e6451"
    )
    assert sha256(CLOCK) == (
        "cbe4e5f6fc52b66062abbf931e46ea4aa0d1f3c0157ffd365d0638aa573c2826"
    )
    assert report["manifest_hash"] == (
        "d84ff0313b3d2dc0762799d90a959e75f6a0a57ed8b9186b7155c3567f872e9b"
    )
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == support.canonical_hash(core)
    assert report["clock_artifact"]["sha256"] == sha256(CLOCK)
    assert report["support_builder"]["sha256"] == (
        "be032fd7b2f17c9aa3cc5c42fef9e7da045586580c723ea9deac40756f29fd70"
    )
    assert report["source_support_passed"] is False
    assert report["advance_to_evaluator_freeze"] is False
    assert report["disposition"] == "reject RCRE-72 unchanged before outcomes"


def test_primary_fails_only_frozen_quadrant_generality_gates() -> None:
    report = payload()
    failed = {name for name, passed in report["source_checks"].items() if not passed}
    assert failed == {
        "selection_each_quadrant",
        "train_each_quadrant",
        "train_quadrant_concentration",
    }
    train = report["primary_support_summaries"]["train"]
    selection = report["primary_support_summaries"]["selection"]
    assert (train["events"], train["longs"], train["shorts"]) == (75, 19, 56)
    assert train["maximum_entry_gap_elapsed_days"] == 40.0
    assert train["quadrant_counts"] == {
        "q+r+": 48,
        "q+r-": 18,
        "q-r+": 1,
        "q-r-": 8,
    }
    assert (selection["events"], selection["longs"], selection["shorts"]) == (
        39,
        19,
        20,
    )
    assert selection["maximum_entry_gap_elapsed_days"] == 26.0
    assert selection["quadrant_counts"] == {"q+r+": 20, "q+r-": 19}


def test_source_and_control_audits_are_exact() -> None:
    report = payload()
    assert report["source_audit"] == {
        "equal_availability_rows_suppressed": 417,
        "invalid_materiality_dates": 0,
        "invalid_missing_or_null_dates": 4,
        "normalized_rows_read": 77_369,
        "required_rows_read": 9_976,
        "source_dates_seen": 1_249,
        "valid_feature_dates": 1_245,
        "venue_swap_dates_checked": 1_245,
        "venue_swap_identity_failures": 0,
    }
    assert report["rank_ready_rows"] == 827
    assert report["decision_state_rows"] == 827
    assert report["control_diagnostics"] == {
        "label_pair_controls_economically_falsifying": False,
        "quantity_gap_label_pair_exact": True,
        "rate_gap_label_pair_exact": True,
    }
    assert report["clock_artifact"]["rows"] == 1_988
    assert report["clock_artifact"]["primary_rows"] == 114


def test_clock_is_exact_source_only_and_globally_nonoverlapping() -> None:
    with gzip.open(CLOCK, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == payload()["clock_artifact"]["rows"]
    assert tuple(rows[0]) == support.CLOCK_COLUMNS
    primary = [row for row in rows if row["control"] == "primary"]
    assert len(primary) == 114
    for row in primary:
        quantity = Fraction(row["quantity_gap"])
        rate = Fraction(row["rate_gap"])
        product = Fraction(row["routing_pressure"])
        unit = Fraction(row["u_routing_pressure"])
        assert quantity != 0 and rate != 0 and product == quantity * rate
        if int(row["state"]) == 1:
            assert product > 0 and unit >= Fraction(1, 2)
            assert row["side"] == "-1"
        else:
            assert product < 0 and unit <= Fraction(-1, 2)
            assert row["side"] == "1"

    for control in support.CONTROL_NAMES:
        selected = [row for row in rows if row["control"] == control]
        entries = [support.rmsr_support._timestamp(row["entry_time"]) for row in selected]
        exits = [support.rmsr_support._timestamp(row["exit_time"]) for row in selected]
        assert len(entries) == len(set(entries))
        assert all(
            current >= previous
            for previous, current in zip(exits, entries[1:])
        )

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


def test_comparator_and_market_boundaries_remained_closed() -> None:
    report = payload()
    boundary = report["outcome_boundary"]
    assert boundary["source_observation_rows_read"] == 77_369
    assert boundary["signed_features_and_rcre_incidence_opened"] is True
    assert boundary["post_2023_source_rows_read"] == 0
    assert boundary["comparator_rows_read"] == 0
    assert boundary["comparator_access_short_circuited_on_source_failure"] is True
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["pnl_cagr_mdd_opened"] is False
    assert report["novelty"] == {
        "checks": {},
        "evaluated": False,
        "metrics": {},
        "passed": False,
        "qualifying_groups": [],
        "reason": "source support failed before comparator access",
    }
    assert report["common_window_audit"]["comparators"] == {}
    assert report["common_window_audit"]["intervals_clipped"] == 0
