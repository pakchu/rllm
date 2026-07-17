from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from training import evaluate_cboe_institutional_hedge_migration as evaluator


def test_frozen_schedule_family_and_stage_counts() -> None:
    schedules = evaluator.load_schedules()
    assert tuple(schedules) == evaluator.ALL_CLOCK_NAMES
    primary = schedules["primary"]
    contained = evaluator._window_schedule(primary, evaluator.STAGE1)
    # One source event enters in 2022 but exits in 2023 and is therefore
    # excluded from the physically bounded Stage-1 simulation.
    assert len(contained) == 151
    assert bool(contained["side"].eq(-1).all())
    assert bool(schedules["direction_flip"]["side"].eq(1).all())
    assert schedules["direction_flip"]["entry_time"].equals(primary["entry_time"])


def test_evaluator_freeze_reads_no_outcomes(tmp_path: Path) -> None:
    path = tmp_path / "freeze.json"
    frozen = evaluator.freeze_evaluator(path)
    assert frozen["opened_windows"] == []
    assert frozen["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert frozen["funding_rows_parsed_during_freeze"] == 0
    assert frozen["simulation_run_during_freeze"] is False
    assert evaluator.verify_evaluator_freeze(path) == frozen


def test_two_sided_signflip_is_deterministic() -> None:
    trades = pd.DataFrame(
        {
            "entry_time": pd.date_range("2021-01-01", periods=30, freq="7D", tz="UTC"),
            "net_return": [0.01] * 30,
        }
    )
    left = evaluator.weekly_cluster_signflip_two_sided(trades, draws=2_000, seed=7)
    right = evaluator.weekly_cluster_signflip_two_sided(trades, draws=2_000, seed=7)
    assert left == right
    assert 0.0 <= left["p_value_two_sided"] <= 1.0


def test_stage2_fails_closed_without_passing_stage1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing-stage1.json"
    monkeypatch.setattr(evaluator, "STAGE1_OUTPUT", missing)
    with pytest.raises(ValueError, match="has not been run"):
        evaluator._verified_passing_stage1("missing")


def test_stage2_contract_matches_smaller_sealed_support() -> None:
    registration, _ = evaluator._verify_static_inputs()
    _, gates = evaluator._verify_evaluation_contract(registration, stage2=True)
    assert gates["minimum_trades"] == 60
    assert gates["minimum_short_trades"] == 60
    assert gates["each_subperiod_minimum_trades"] == 25
