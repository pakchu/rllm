from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.preregister_cross_venue_radial_refill_compression import canonical_hash


PATH = Path(
    "results/cross_venue_radial_refill_compression_selection_2023_2026-07-17.json"
)


def test_crrc_is_rejected_before_any_later_window() -> None:
    payload = json.loads(PATH.read_text())
    body = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    assert canonical_hash(body) == payload["manifest_hash"]
    assert payload["outcomes_opened"] is True
    assert payload["opened_window"] == ["2023-01-01", "2024-01-01"]
    assert payload["decision"] == "rejected_before_2024"
    assert payload["2024_test_opened"] is False
    assert payload["2025_eval_opened"] is False
    assert payload["2026_holdout_opened"] is False
    assert payload["outcome_opening_head"] == (
        "b32694c365ab03fdd60c5c358bcd77fbb515f1f1"
    )


def test_primary_stats_and_full_calendar_cagr_are_frozen() -> None:
    payload = json.loads(PATH.read_text())
    annual = payload["evaluation"]["primary"]["2023"]
    assert annual["absolute_return_pct"] == pytest.approx(-1.0175265640873454)
    assert annual["cagr_pct"] == pytest.approx(annual["absolute_return_pct"])
    assert annual["strict_mdd_pct"] == pytest.approx(9.498132218709577)
    assert annual["cagr_to_strict_mdd"] == pytest.approx(-0.10712912187966861)
    assert annual["trades"] == 156
    assert annual["longs"] == 91
    assert annual["shorts"] == 65
    assert annual["transaction_cost_pct_initial"] == pytest.approx(9.524116873323347)


def test_failure_is_broad_and_not_a_single_control_artifact() -> None:
    payload = json.loads(PATH.read_text())
    evaluation = payload["evaluation"]
    assert evaluation["primary"]["q1"]["absolute_return_pct"] > 0.0
    for quarter in ("q2", "q3", "q4"):
        assert evaluation["primary"][quarter]["absolute_return_pct"] < 0.0
    assert evaluation["long_only"]["absolute_return_pct"] == pytest.approx(
        4.046332866294056
    )
    assert evaluation["short_only"]["absolute_return_pct"] == pytest.approx(
        -4.866927349461259
    )
    assert evaluation["ten_bp_notional_side_cost_stress"]["absolute_return_pct"] < 0.0
    assert evaluation["entry_and_exit_delay_plus_5m"]["absolute_return_pct"] < 0.0
    assert evaluation["monthly_cluster_signflip"]["raw_p_value"] == pytest.approx(
        0.5714214289285535
    )
    assert evaluation["passes_2023_selection"] is False


def test_exact_failed_gate_set_is_frozen() -> None:
    payload = json.loads(PATH.read_text())
    failed = {
        key
        for key, passed in payload["evaluation"]["selection_gates"].items()
        if not passed
    }
    assert failed == {
        "annual_absolute_return_positive",
        "annual_cagr_to_strict_mdd_at_least_3",
        "every_quarter_absolute_return_positive",
        "short_only_absolute_return_positive",
        "ten_bp_stress_absolute_return_positive",
        "delay_plus_5m_absolute_return_positive",
        "monthly_cluster_signflip_p_at_most_0_10",
    }
    assert "no sign, threshold, hold, scale" in payload["anti_repair"]
