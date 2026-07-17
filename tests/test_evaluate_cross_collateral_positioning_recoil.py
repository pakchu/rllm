from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_cross_collateral_positioning_recoil as evaluator


def test_static_inputs_support_and_contract_are_frozen() -> None:
    registration, support = evaluator._verify_static_inputs()
    assert registration["policy"]["policy_id"] == "CCPR-1"
    assert support["selected_q"] == 0.85
    assert support["outcomes_opened"] is False
    assert (
        evaluator._verify_evaluation_contract(registration)
        == evaluator.EvaluationConfig()
    )


def test_schedules_use_frozen_holds_delay_sides_and_nonoverlap() -> None:
    schedules = evaluator.load_schedules()
    assert set(schedules) == set(evaluator.CANDIDATE_IDS)
    for candidate_id, clocks in schedules.items():
        assert set(clocks) == set(evaluator.ALL_CLOCK_NAMES)
        hold = pd.Timedelta(minutes=5 * evaluator.CANDIDATE_HOLDS[candidate_id])
        for name, schedule in clocks.items():
            delay = pd.Timedelta(minutes=70 if name == "entry_shift_plus_1h" else 10)
            assert schedule["entry_time"].eq(schedule["signal_time"] + delay).all()
            assert schedule["exit_time"].eq(schedule["entry_time"] + hold).all()
            assert schedule["side"].isin((-1, 1)).all()
            assert (
                schedule["entry_time"]
                .iloc[1:]
                .reset_index(drop=True)
                .ge(schedule["exit_time"].iloc[:-1].reset_index(drop=True))
                .all()
            )
        primary = clocks["primary"]
        flipped = clocks["direction_flip"]
        assert primary[["entry_time", "exit_time"]].equals(
            flipped[["entry_time", "exit_time"]]
        )
        assert (
            flipped["side"]
            .reset_index(drop=True)
            .eq(-primary["side"].reset_index(drop=True))
            .all()
        )


def test_freeze_does_not_parse_or_simulate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("freeze opened an outcome")

    monkeypatch.setattr(evaluator, "load_execution_window", forbidden)
    monkeypatch.setattr(evaluator, "simulate_schedule", forbidden)
    report = evaluator.freeze_evaluator(tmp_path / "freeze.json")
    assert report["opened_windows"] == []
    assert report["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert report["funding_rows_parsed_during_freeze"] == 0
    assert report["simulation_run_during_freeze"] == 0
    assert all(report["schedule_invariant_checks"].values())


def test_two_sided_cluster_signflip_is_deterministic_and_sign_symmetric() -> None:
    entry = pd.date_range("2022-01-03", periods=12, freq="7D", tz="UTC")
    positive = pd.DataFrame(
        {"entry_time": entry, "net_return": np.linspace(0.001, 0.012, len(entry))}
    )
    negative = positive.assign(net_return=-positive["net_return"])
    first = evaluator.weekly_cluster_signflip_two_sided(positive, draws=2_000, seed=17)
    second = evaluator.weekly_cluster_signflip_two_sided(positive, draws=2_000, seed=17)
    mirrored = evaluator.weekly_cluster_signflip_two_sided(
        negative, draws=2_000, seed=17
    )
    assert first == second
    assert first["p_value_two_sided"] == mirrored["p_value_two_sided"]


def test_strict_engine_schedule_smoke() -> None:
    dates = pd.date_range("2022-01-01", periods=5, freq="5min", tz="UTC")
    market = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 102.0, 101.0, 100.0, 100.0],
            "low": [100.0, 98.0, 99.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
    funding = pd.DataFrame(
        columns=["funding_time", "symbol", "funding_rate", "settlement_mark_price"]
    )
    schedule = pd.DataFrame(
        {
            "candidate_id": ["CCPR-H4"],
            "clock_name": ["primary"],
            "signal_time": [dates[0]],
            "signal_day": [dates[0]],
            "entry_time": [dates[1]],
            "exit_time": [dates[3]],
            "side": [1],
        }
    )
    metrics = evaluator.simulate_schedule(
        market,
        funding,
        schedule,
        period_start=dates[0],
        period_end=dates[-1],
        cost_rate=0.0,
        cfg=evaluator.EvaluationConfig(),
    )
    assert metrics["trades"] == 1
    assert metrics["strict_mdd_pct"] > 0.0
    assert "p_value_two_sided" in metrics["weekly_cluster_signflip"]


def test_forged_stage1_is_rejected_before_stage2_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = {
        "protocol_version": "cross_collateral_positioning_recoil_stage1_v1",
        "policy_id": "CCPR-1",
        "stage": "stage1_2021_2022",
        "evaluator_freeze_manifest_hash": "x",
        "evaluator_source_sha256": "x",
        "config": {},
        "execution_diagnostics": {},
        "candidate_results": [],
        "selected_candidate_id": "CCPR-H4",
        "stage1_passed": True,
        "opened_windows": ["stage1_2021_2022"],
        "sealed_windows": ["stage2_2023", "2024_plus"],
        "disposition": "ADVANCE_TO_SEALED_2023",
    }
    fake["manifest_hash"] = evaluator._canonical_hash(fake)
    path = tmp_path / "fake.json"
    path.write_text(json.dumps(fake), encoding="utf-8")
    monkeypatch.setattr(evaluator, "STAGE1_OUTPUT", path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("stage2 loader reached")

    monkeypatch.setattr(evaluator, "load_execution_window", forbidden)
    with pytest.raises(ValueError, match="candidate family changed"):
        evaluator._verified_passing_stage1(expected_freeze_hash="x")


def test_written_freeze_replays_after_generation() -> None:
    report = evaluator.verify_evaluator_freeze()
    assert report["policy_id"] == "CCPR-1"
    assert report["selected_q"] == 0.85
    assert report["evaluator_source_sha256"] == evaluator._sha256(
        evaluator.EVALUATOR_SOURCE
    )


def test_written_stage1_is_hash_valid_and_physically_stops_before_2023() -> None:
    report = evaluator._load_json(evaluator.STAGE1_OUTPUT)
    evaluator._validate_stored_stage1_structure(report)
    assert report["stage1_passed"] is False
    assert report["selected_candidate_id"] is None
    assert report["sealed_windows"] == ["stage2_2023", "2024_plus"]
    diagnostics = report["execution_diagnostics"]
    assert diagnostics["physical_window"] == [
        "2021-07-08T00:00:00+00:00",
        "2023-01-01T00:00:00+00:00",
    ]
    assert diagnostics["market"]["rows"] == 156_096
    assert diagnostics["market"]["last_timestamp"] == "2022-12-31T23:55:00+00:00"
    assert diagnostics["funding"]["rows"] == 1_626
    assert diagnostics["funding"]["last_timestamp"] == "2022-12-31T16:00:00+00:00"


def test_failed_real_stage1_blocks_stage2_before_execution_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("2023 execution loader reached")

    monkeypatch.setattr(evaluator, "load_execution_window", forbidden)
    freeze = evaluator.verify_evaluator_freeze()
    with pytest.raises(ValueError, match="Stage1 failed; 2023 remains sealed"):
        evaluator._verified_passing_stage1(expected_freeze_hash=freeze["manifest_hash"])
