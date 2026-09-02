from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from training import evaluate_gross9_qtr_distill_economics as qtr


def _market(start="2024-01-01T00:00:00Z", periods=7, opens=None, highs=None, lows=None):
    dates = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    if opens is None:
        opens = [100.0] * periods
    if highs is None:
        highs = [o + 1 for o in opens]
    if lows is None:
        lows = [o - 1 for o in opens]
    return pd.DataFrame({"date": dates, "open": opens, "high": highs, "low": lows, "close": opens})


def _funding(times=None, rates=None, marks=None):
    times = times or ["2024-01-01T00:00:00Z"]
    rates = rates or [0.0] * len(times)
    marks = marks or [100.0] * len(times)
    return pd.DataFrame({"date": pd.to_datetime(times, utc=True), "funding_rate": rates, "mark_price": marks})


def _four_sleeve_clock(rows):
    base = []
    for i, sleeve in enumerate(["a", "b", "c", "d"]):
        base.append({"sleeve": sleeve, "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:25:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:30:00Z"), "side": 1})
    base.extend(rows)
    return pd.DataFrame(base)


def test_same_timestamp_exits_entries_are_net_costed_and_new_quantities_use_same_pre_equity():
    market = _market(periods=5)
    funding = _funding(times=["2024-01-01T00:00:00Z"], rates=[0.0])
    clock = pd.DataFrame([
        {"sleeve": "a", "weight": 0.5, "entry_time": pd.Timestamp("2024-01-01T00:00:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": 1},
        {"sleeve": "b", "weight": 0.5, "entry_time": pd.Timestamp("2024-01-01T00:10:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:20:00Z"), "side": 1},
        {"sleeve": "c", "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:15:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:20:00Z"), "side": 1},
        {"sleeve": "d", "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:15:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:20:00Z"), "side": -1},
    ])
    result = qtr.simulate_portfolio(clock, market, funding, market.date.iloc[0], market.date.iloc[-1], cost=0.001, initial_equity=100_000.0)
    transitions = {row["time"]: row for row in result["transition_rows"]}

    first = transitions[pd.Timestamp("2024-01-01T00:00:00Z")]
    assert first["q_before"] == 0.0
    assert first["q_after"] == pytest.approx(500.0)
    assert first["fee"] == pytest.approx(50.0)

    handoff = transitions[pd.Timestamp("2024-01-01T00:10:00Z")]
    assert handoff["exits"] == ["a"]
    assert handoff["entries"] == ["b"]
    # Exiting + entering same side at same open nets to a tiny fee only for the
    # equity-after-fee sizing difference, not two full notionals.
    assert abs(handoff["delta_q"]) < 1.0
    assert handoff["fee"] < 0.10
    assert handoff["q_after"] == pytest.approx(0.5 * handoff["equity_pre"] / 100.0)


def test_fixed_quantities_do_not_rebalance_after_price_move_and_final_exit_charges_cost():
    market = _market(periods=4, opens=[100, 110, 120, 130], highs=[100, 110, 120, 130], lows=[100, 110, 120, 130])
    funding = _funding()
    clock = _four_sleeve_clock([
        {"sleeve": "a", "weight": 0.5, "entry_time": pd.Timestamp("2024-01-01T00:00:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:15:00Z"), "side": 1},
    ])
    # Remove dummy future rows outside market by using exactly four active sleeves inside the window.
    clock = pd.DataFrame([
        {"sleeve": "a", "weight": 0.5, "entry_time": pd.Timestamp("2024-01-01T00:00:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:15:00Z"), "side": 1},
        {"sleeve": "b", "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:05:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": 1},
        {"sleeve": "c", "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:05:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": -1},
        {"sleeve": "d", "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:05:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": 1},
    ])
    result = qtr.simulate_portfolio(clock, market, funding, market.date.iloc[0], market.date.iloc[-1], cost=0.0, initial_equity=100_000.0)
    entry = result["transition_rows"][0]
    assert entry["q_after"] == pytest.approx(500.0)
    # Deterministic value includes b/c/d fixed-quantity intraday sleeves; no interval is resized after later price moves.
    assert result["final_equity"] == pytest.approx(115_954.54545454546, rel=1e-9)


def test_funding_uses_post_transition_aggregate_q_so_exiting_position_excluded_and_new_entry_included():
    market = _market(periods=3)
    funding = _funding(times=["2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z"], rates=[0.01, 0.01], marks=[100.0, 100.0])
    clock = pd.DataFrame([
        {"sleeve": "a", "weight": 0.5, "entry_time": pd.Timestamp("2024-01-01T00:00:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": 1},
        {"sleeve": "b", "weight": 0.25, "entry_time": pd.Timestamp("2024-01-01T00:10:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z") + pd.Timedelta(minutes=0), "side": 1},
        {"sleeve": "c", "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:05:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": -1},
        {"sleeve": "d", "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:05:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": 1},
    ])
    # b must be non-zero interval; extend market one more row.
    market = _market(periods=4)
    clock.loc[1, "exit_time"] = pd.Timestamp("2024-01-01T00:15:00Z")
    result = qtr.simulate_portfolio(clock, market, funding, market.date.iloc[0], market.date.iloc[-1], cost=0.0, initial_equity=100_000.0)
    effects = {row["time"]: row for row in result["equity_effect_rows"]}
    assert effects[pd.Timestamp("2024-01-01T00:00:00Z")]["funding_cash"] == pytest.approx(-500.0)
    # At 00:10 a exits before funding and b enters before funding, so only b's post-transition q pays.
    assert effects[pd.Timestamp("2024-01-01T00:10:00Z")]["funding_cash"] == pytest.approx(-248.75, rel=1e-6)


def test_strict_mdd_uses_favorable_then_adverse_global_hwm_with_virtual_liquidation_cost():
    market = _market(periods=3, opens=[100, 100, 100], highs=[120, 100, 100], lows=[80, 100, 100])
    funding = _funding()
    clock = pd.DataFrame([
        {"sleeve": "a", "weight": 1.0, "entry_time": pd.Timestamp("2024-01-01T00:00:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": 1},
        {"sleeve": "b", "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:05:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": -1},
        {"sleeve": "c", "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:05:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": 1},
        {"sleeve": "d", "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:05:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": 1},
    ])
    result = qtr.simulate_portfolio(clock, market, funding, market.date.iloc[0], market.date.iloc[-1], cost=0.001, initial_equity=100_000.0)
    # Peak after favorable high is 119900 (entry fee paid); adverse equity is 79820 (includes virtual cost at low).
    assert result["strict_mdd_pct"] == pytest.approx((1 - 79_820.0 / 119_900.0) * 100.0, rel=1e-6)


def test_cluster_signflip_is_utc_week_deterministic():
    rows = [
        {"time": pd.Timestamp("2024-01-01T00:00:00Z"), "log_effect": 0.01},
        {"time": pd.Timestamp("2024-01-03T00:00:00Z"), "log_effect": 0.02},
        {"time": pd.Timestamp("2024-01-08T00:00:00Z"), "log_effect": -0.01},
    ]
    a = qtr.cluster_signflip(rows, draws=1000, seed=7)
    b = qtr.cluster_signflip(rows, draws=1000, seed=7)
    assert a == b
    assert a["clusters"] == 2
    assert a["draws"] == 1000


def test_stage_checks_train_reports_legacy_bonferroni_but_oos_p_point_one_authorizes(monkeypatch, tmp_path):
    primary = {
        "base": {"absolute_return_pct": 1, "cagr_to_strict_mdd": 3, "strict_mdd_pct": 10, "mean_exposure_weighted_gross_edge_bp": 1},
        "stress": {"absolute_return_pct": 1, "cagr_to_strict_mdd": 2.5},
        "calendar_halves": {"first": {"absolute_return_pct": 1}, "second": {"absolute_return_pct": 1}},
        "cluster_signflip": {"pvalue": 0.09},
    }
    assert "oos_cluster_signflip_p_max_0_1" not in qtr.stage_checks("train", primary)
    assert qtr.stage_checks("test", primary)["oos_cluster_signflip_p_max_0_1"] is True
    primary["cluster_signflip"] = {"pvalue": 0.11}
    assert qtr.stage_checks("test", primary)["oos_cluster_signflip_p_max_0_1"] is False
    assert qtr.TRAIN_LEGACY_BONFERRONI_P_MAX == pytest.approx(0.1 / 72)


def test_predecessor_hash_pass_gate_blocks_before_loader_opening(monkeypatch, tmp_path):
    called = {"load": False}
    monkeypatch.setattr(qtr, "load_portfolio_clock", lambda *a, **k: (_ for _ in ()).throw(AssertionError("loader opened")))
    outputs = {stage: tmp_path / f"{stage}.json" for stage in qtr.STAGES}
    with pytest.raises(RuntimeError, match="missing predecessor train"):
        qtr.run("test", output=tmp_path / "test.json", sleeves=[], outputs=outputs)

    core = {"protocol_version": qtr.PROTOCOL_VERSION, "policy_id": qtr.POLICY_ID, "stage": "train", "passed": False, "advance_to_next_stage": False}
    outputs["train"].write_text(json.dumps({**core, "manifest_hash": qtr.canonical_hash(core)}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="predecessor did not pass"):
        qtr.run("test", output=tmp_path / "test.json", sleeves=[], outputs=outputs)


def test_run_writes_manifest_and_never_opens_later_stage_without_passed_predecessor(monkeypatch, tmp_path):
    start = pd.Timestamp("2023-07-01T00:00:00Z")
    end = pd.Timestamp("2024-01-01T00:00:00Z")
    clock = pd.DataFrame([
        {"sleeve": "a", "weight": 0.1, "entry_time": start, "exit_time": start + pd.Timedelta(minutes=5), "side": 1},
        {"sleeve": "b", "weight": 0.1, "entry_time": start, "exit_time": start + pd.Timedelta(minutes=5), "side": 1},
        {"sleeve": "c", "weight": 0.1, "entry_time": start, "exit_time": start + pd.Timedelta(minutes=5), "side": -1},
        {"sleeve": "d", "weight": 0.1, "entry_time": start, "exit_time": start + pd.Timedelta(minutes=5), "side": 1},
    ])
    monkeypatch.setattr(qtr, "load_portfolio_clock", lambda *a, **k: clock)
    monkeypatch.setattr(qtr, "load_sources", lambda stage, s, e: (_market(str(s), periods=2, opens=[100, 110], highs=[110, 110], lows=[100, 110]), _funding(times=[str(s)], rates=[0.0]), {"mode": "unit"}))
    monkeypatch.setattr(qtr, "validate_market", lambda *a, **k: None)
    monkeypatch.setattr(qtr, "validate_funding", lambda *a, **k: None)
    monkeypatch.setattr(qtr, "evaluate_primary", lambda *a, **k: {
        "base": {"absolute_return_pct": 1, "cagr_to_strict_mdd": 3, "strict_mdd_pct": 10, "mean_exposure_weighted_gross_edge_bp": 1},
        "stress": {"absolute_return_pct": 1, "cagr_to_strict_mdd": 2.5},
        "calendar_halves": {"first": {"absolute_return_pct": 1}, "second": {"absolute_return_pct": 1}},
        "cluster_signflip": {"pvalue": 0.5},
    })
    monkeypatch.setattr(qtr, "STAGES", {**qtr.STAGES, "train": ("train", str(start), str(start + pd.Timedelta(minutes=5)))})

    output = tmp_path / "train.json"
    result = qtr.run("train", output=output, sleeves=qtr.default_sleeves())
    written = json.loads(output.read_text())
    assert written["manifest_hash"] == qtr.canonical_hash({k: v for k, v in written.items() if k != "manifest_hash"})
    assert result["later_stage_outcomes_opened"] is False
    assert result["train_legacy_cluster_diagnostic"]["reported_not_pass_authorizing"] is True
