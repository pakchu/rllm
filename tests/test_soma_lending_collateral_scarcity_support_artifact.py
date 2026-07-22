from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

from training import build_soma_lending_collateral_scarcity_support as support


REPORT = Path("results/soma_lending_collateral_scarcity_support_2026-07-23.json")
CLOCK = Path("results/soma_lending_collateral_scarcity_clocks_2026-07-23.csv.gz")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_support_artifacts_are_hash_bound_and_rejected() -> None:
    report = payload()
    assert sha256(REPORT) == "354f3edb9f1d9bdbac1f609e50882f2e4d1df6ee8cfa555287ca99a15148a738"
    assert sha256(CLOCK) == "b3fe0dc8c895f9a8974cdf08b5bed9d58ff693b8aca7ed59c224627de930a948"
    assert report["manifest_hash"] == "95d6e4b3220645bc63d323b7834286beb6b9b7f02bdf8fe2f6db1f6bfc52ad4b"
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == support.canonical_hash(core)
    assert report["clocks"]["sha256"] == sha256(CLOCK)
    assert report["disposition"] == "REJECT_BEFORE_OUTCOMES_NO_REPAIR"
    assert report["advance_to_evaluator_freeze"] is False


def test_primary_fails_only_frozen_gap_and_rrp_novelty_gates() -> None:
    report = payload()
    assert report["source_support_passed"] is False
    failed_source = {
        name for name, passed in report["source_support_gates"].items() if not passed
    }
    assert failed_source == {"train_maximum_gap_45d"}
    primary = report["clock_summaries"]["primary"]
    assert primary["train"]["events"] == 75
    assert primary["train"]["maximum_entry_gap_elapsed_days"] == 78.0
    assert primary["selection"]["events"] == 35
    assert report["novelty_passed"] is False
    failed_novelty = {
        name
        for name, checks in report["novelty_checks"].items()
        if not all(checks.values())
    }
    assert failed_novelty == {"overnight_rrp_flow_release"}
    rrp = report["novelty_metrics"]["overnight_rrp_flow_release"]
    assert rrp["slcs_one_day_containment"] == 41 / 110
    assert rrp["signed_5m_occupied_exposure_correlation"] < 0.05


def test_clock_is_source_only_and_causal() -> None:
    with gzip.open(CLOCK, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1685
    primary = [row for row in rows if row["control"] == "primary"]
    assert len(primary) == 110
    assert {row["side"] for row in primary} == {"-1", "1"}
    forbidden = {"open", "high", "low", "close", "return", "pnl", "cagr", "mdd", "funding"}
    assert not forbidden.intersection(name.lower() for name in rows[0])


def test_outcome_boundary_remained_closed() -> None:
    report = payload()
    boundary = report["outcome_boundary"]
    assert boundary["source_operation_rows_read"] == 1259
    assert boundary["source_detail_rows_read"] == 182616
    assert boundary["comparator_clock_rows_read"] == 4123
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["return_rows_read"] == 0
    assert boundary["pnl_cagr_mdd_opened"] is False
    assert boundary["economic_simulations_run"] == 0
