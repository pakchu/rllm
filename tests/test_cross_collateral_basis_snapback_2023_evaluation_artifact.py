from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.preregister_cross_collateral_basis_snapback import canonical_hash


RESULT = Path("results/cross_collateral_basis_snapback_2023_evaluation_2026-07-17.json")


def test_2023_evaluation_artifact_is_hash_bound_and_keeps_2024_sealed() -> None:
    report = json.loads(RESULT.read_text())
    body = {key: value for key, value in report.items() if key not in {"as_of", "content_hash"}}
    assert canonical_hash(body) == report["content_hash"]
    assert report["content_hash"] == (
        "1d41d55f29dec340f207d81600ee8d8f3a595c641d952756079ecf571d423f9a"
    )
    assert report["mode"] == "2023_outcome_blind_development_not_pristine_oos"
    assert report["gate_passed"] is False
    assert report["disposition"] == "REJECT_2023_KEEP_2024_SEALED"
    assert report["config"]["period_end"] == "2024-01-01"
    assert "2024" not in report


def test_2023_absolute_return_cagr_strict_mdd_and_cost_failure_are_frozen() -> None:
    report = json.loads(RESULT.read_text())
    base = report["base_cost"]
    stress = report["stress_cost"]
    assert base["absolute_return_pct"] == pytest.approx(-5.489758615178997)
    assert base["cagr_pct"] == pytest.approx(-5.493413500420741)
    assert base["strict_mdd_pct"] == pytest.approx(6.727095880793743)
    assert base["cagr_to_strict_mdd"] == pytest.approx(-0.8166099603403545)
    assert base["trades"] == 58
    assert stress["absolute_return_pct"] == pytest.approx(-9.780732431949268)
    assert stress["strict_mdd_pct"] == pytest.approx(10.926863806889608)
    assert base["pre_cost_pnl"] == pytest.approx(0.012905936729976643)
    assert base["transaction_cost"] == pytest.approx(0.06780352288176712)
    assert base["pre_cost_pnl_to_cost"] == pytest.approx(0.19034315890166154)


def test_mechanism_converges_but_both_branches_and_cost_gate_fail() -> None:
    report = json.loads(RESULT.read_text())
    base = report["base_cost"]
    gates = report["gates"]
    assert base["median_signed_wedge_convergence"] > 0.0
    assert base["signed_wedge_convergence_hit_rate"] == pytest.approx(40 / 58)
    assert gates["median_signed_wedge_convergence_positive"] is True
    assert gates["signed_wedge_convergence_hit_rate_at_least_55pct"] is True
    assert gates["pre_cost_pnl_exceeds_transaction_cost"] is False
    assert gates["um_rich_branch_positive"] is False
    assert gates["cm_rich_branch_positive"] is False
    assert gates["monthly_signflip_pvalue_at_most_10pct"] is False


def test_every_frozen_trade_is_2023_and_respects_reserved_clock() -> None:
    report = json.loads(RESULT.read_text())
    trades = report["base_cost_trades"]
    assert len(trades) == 58
    assert all(trade["entry_time"].startswith("2023-") for trade in trades)
    assert all(trade["exit_time"].startswith("2023-") for trade in trades)
    assert {trade["rich_leg"] for trade in trades} == {"um", "cm"}
