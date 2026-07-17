from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from training import evaluate_sofr_rate_dislocation as evaluator


def _synthetic_simulation(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    cost_rate: float,
    cfg: evaluator.EvaluationConfig,
) -> dict[str, object]:
    del market, funding, cost_rate, cfg
    clock_name = str(schedule["clock_name"].iloc[0])
    primary = clock_name == "primary"
    trade_details = [
        {
            "entry_time": evaluator._timestamp(
                period_start
                + pd.DateOffset(months=index % 12)
                + pd.Timedelta(days=index // 12)
            ).isoformat(),
            "side": 1 if index % 2 == 0 else -1,
        }
        for index in range(50)
    ]
    return {
        "absolute_return_pct": 5.0 if primary else -1.0,
        "cagr_pct": 5.0 if primary else -1.0,
        "strict_mdd_pct": 1.0,
        "cagr_to_strict_mdd": 4.0 if primary else -1.0,
        "trades": 50,
        "mean_gross_underlying_bp": 40.0 if primary else -10.0,
        "weekly_cluster_signflip": {"p_value_one_sided": 0.05},
        "trade_details": trade_details,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
    }


def test_source_only_schedules_are_causal_nonoverlapping_and_frozen() -> None:
    schedules = evaluator.load_schedules()
    assert set(schedules) == set(evaluator.ALL_CLOCK_NAMES)
    assert len(evaluator._window_schedule(schedules["primary"], evaluator.STAGE1)) == 48
    assert len(evaluator._window_schedule(schedules["primary"], evaluator.STAGE2)) == 40
    for schedule in schedules.values():
        assert bool(
            schedule["entry_time"]
            .eq(schedule["signal_day"] + pd.Timedelta(minutes=5))
            .all()
        )
        assert bool(
            schedule["exit_time"]
            .eq(schedule["entry_time"] + pd.Timedelta(days=5))
            .all()
        )
        assert bool(schedule["exit_time"].le(evaluator.STAGE2[1]).all())
        if len(schedule) > 1:
            assert schedule["entry_time"].iloc[1:].reset_index(drop=True).ge(
                schedule["exit_time"].iloc[:-1].reset_index(drop=True)
            ).all()
    assert schedules["direction_flip"][["entry_time", "exit_time"]].equals(
        schedules["primary"][["entry_time", "exit_time"]]
    )
    assert schedules["random_side"][["entry_time", "exit_time"]].equals(
        schedules["primary"][["entry_time", "exit_time"]]
    )
    train_distribution = evaluator._entry_distribution(
        evaluator._window_schedule(schedules["primary"], evaluator.STAGE1)
    )
    stage2_distribution = evaluator._entry_distribution(
        evaluator._window_schedule(schedules["primary"], evaluator.STAGE2)
    )
    assert train_distribution["trades"] == 48
    assert train_distribution["long_trades"] == 31
    assert train_distribution["short_trades"] == 17
    assert sum(train_distribution["entry_month_counts"].values()) == 48
    assert train_distribution["max_single_entry_month_count"] == 5
    assert train_distribution["max_single_entry_month_share"] == 5 / 48
    assert stage2_distribution["trades"] == 40
    assert stage2_distribution["long_trades"] == 20
    assert stage2_distribution["short_trades"] == 20
    assert sum(stage2_distribution["entry_month_counts"].values()) == 40
    assert stage2_distribution["max_single_entry_month_count"] == 5
    assert stage2_distribution["max_single_entry_month_share"] == 5 / 40


def test_full_qualification_enforces_side_and_month_distribution() -> None:
    trade_details = [
        {
            "entry_time": (
                evaluator._timestamp(
                    evaluator._utc_timestamp("2021-01-01")
                    + pd.DateOffset(months=index % 15)
                ).isoformat()
            ),
            "side": 1 if index < 15 else -1,
        }
        for index in range(30)
    ]
    metrics = {
        "absolute_return_pct": 1.0,
        "cagr_to_strict_mdd": 3.0,
        "strict_mdd_pct": 15.0,
        "weekly_cluster_signflip": {"p_value_one_sided": 0.10},
        "trades": 30,
        "mean_gross_underlying_bp": 35.0,
        "trade_details": trade_details,
    }
    stress = {"absolute_return_pct": 0.1}
    subperiods = {"whole": {"absolute_return_pct": 0.1, "trades": 30}}
    gates = evaluator._full_qualification(
        metrics,
        stress,
        subperiods,
        total_trades_min=30,
        subperiod_trade_mins={"whole": 30},
        each_side_trades_min=15,
        max_single_entry_month_share=0.15,
    )
    assert set(gates) == evaluator.BASE_QUALIFICATION_GATE_NAMES
    assert all(gates.values())

    metrics["trade_details"][-1]["side"] = 1
    concentrated = [dict(row, entry_time="2021-01-01T00:00:00+00:00") for row in trade_details]
    side_failure = evaluator._full_qualification(
        metrics,
        stress,
        subperiods,
        total_trades_min=30,
        subperiod_trade_mins={"whole": 30},
        each_side_trades_min=15,
        max_single_entry_month_share=0.15,
    )
    metrics["trade_details"] = concentrated
    concentration_failure = evaluator._full_qualification(
        metrics,
        stress,
        subperiods,
        total_trades_min=30,
        subperiod_trade_mins={"whole": 30},
        each_side_trades_min=14,
        max_single_entry_month_share=0.15,
    )
    assert side_failure["minimum_each_side_trades"] is False
    assert concentration_failure["single_entry_month_share_at_most_15pct"] is False


def test_freeze_opens_no_market_funding_or_simulation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[str] = []
    real_sha = evaluator._sha256

    def tracking_sha(path: str | Path) -> str:
        seen.append(str(path))
        return real_sha(path)

    monkeypatch.setattr(evaluator, "_sha256", tracking_sha)
    monkeypatch.setattr(
        evaluator,
        "load_execution_window",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("freeze opened execution data")
        ),
    )
    output = tmp_path / "freeze.json"
    report = evaluator.freeze_evaluator(output)
    assert report["opened_windows"] == []
    assert report["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert report["funding_rows_parsed_during_freeze"] == 0
    assert report["simulation_run_during_freeze"] is False
    assert str(evaluator.MARKET) not in seen
    assert str(evaluator.FUNDING) not in seen
    assert evaluator.verify_evaluator_freeze(output) == report


def test_evaluator_freeze_is_byte_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "freeze.json"
    first = evaluator.freeze_evaluator(output)
    first_bytes = output.read_bytes()
    second = evaluator.freeze_evaluator(output)
    assert first == second
    assert output.read_bytes() == first_bytes
    assert first["selection_protocol"]["candidate_count"] == 1
    assert first["control_schedules"]["primary"][
        "stage1_entry_distribution"
    ]["short_trades"] == 17
    assert first["control_schedules"]["primary"][
        "stage2_entry_distribution"
    ]["long_trades"] == 20


def test_evaluator_freeze_rejects_rehashed_schedule_metadata(tmp_path: Path) -> None:
    output = tmp_path / "freeze.json"
    report = evaluator.freeze_evaluator(output)
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    core["control_schedules"]["primary"]["stage1_2021_2022"] = 47
    output.write_text(json.dumps(evaluator._seal(core)))
    with pytest.raises(ValueError, match="primary schedule changed"):
        evaluator.verify_evaluator_freeze(output)


def test_stage2_rejects_stage1_bound_to_another_freeze_before_outcome_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stage1_path = tmp_path / "stage1.json"
    report = evaluator._seal(
        {
            "protocol_version": "sofr_rate_dislocation_stage1_v1",
            "policy_id": "SFRD-1",
            "stage": "stage1_2021_2022",
            "evaluator_freeze_manifest_hash": "wrong-freeze",
            "evaluator_source_sha256": evaluator._sha256(evaluator.EVALUATOR_SOURCE),
            "config": asdict(evaluator.EvaluationConfig()),
            "execution_diagnostics": {
                "physical_window": [
                    evaluator.STAGE1[0].isoformat(),
                    evaluator.STAGE1[1].isoformat(),
                ]
            },
            "gates": {name: True for name in evaluator.STAGE1_GATE_NAMES},
            "gate_passed": True,
            "opened_windows": ["stage1_2021_2022"],
            "sealed_windows": ["stage2_2023", "2024", "2025", "2026_ytd"],
            "disposition": "PASS_STAGE1_OPEN_2023_ONCE",
        }
    )
    stage1_path.write_text(json.dumps(report))
    monkeypatch.setattr(evaluator, "STAGE1_OUTPUT", stage1_path)
    monkeypatch.setattr(
        evaluator,
        "verify_evaluator_freeze",
        lambda: {"manifest_hash": "current-freeze"},
    )

    def forbidden_outcome_open(*args: object, **kwargs: object) -> None:
        raise AssertionError("Stage 2 tried to open a sealed outcome")

    monkeypatch.setattr(evaluator, "load_execution_window", forbidden_outcome_open)
    with pytest.raises(ValueError, match="current evaluator freeze"):
        evaluator.evaluate_stage2()


def test_stage1_requests_only_the_frozen_2021_2022_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    opened: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    monkeypatch.setattr(
        evaluator,
        "verify_evaluator_freeze",
        lambda: {"manifest_hash": "frozen-evaluator"},
    )

    def fake_load_execution_window(
        window: tuple[pd.Timestamp, pd.Timestamp],
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
        opened.append(window)
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {"physical_window": [window[0].isoformat(), window[1].isoformat()]},
        )

    monkeypatch.setattr(evaluator, "load_execution_window", fake_load_execution_window)
    monkeypatch.setattr(evaluator, "simulate_schedule", _synthetic_simulation)
    monkeypatch.setattr(evaluator, "STAGE1_OUTPUT", tmp_path / "stage1.json")
    monkeypatch.setattr(evaluator, "STAGE1_DOC", tmp_path / "stage1.md")
    report = evaluator.evaluate_stage1()
    assert opened == [evaluator.STAGE1]
    assert report["opened_windows"] == ["stage1_2021_2022"]
    assert report["sealed_windows"] == [
        "stage2_2023",
        "2024",
        "2025",
        "2026_ytd",
    ]


def test_stage2_valid_stage1_opens_only_2023_and_keeps_later_years_sealed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    freeze_hash = "frozen-evaluator"
    stage1_path = tmp_path / "stage1.json"
    stage1 = evaluator._seal(
        {
            "protocol_version": "sofr_rate_dislocation_stage1_v1",
            "policy_id": "SFRD-1",
            "stage": "stage1_2021_2022",
            "evaluator_freeze_manifest_hash": freeze_hash,
            "evaluator_source_sha256": evaluator._sha256(evaluator.EVALUATOR_SOURCE),
            "config": asdict(evaluator.EvaluationConfig()),
            "execution_diagnostics": {
                "physical_window": [
                    evaluator.STAGE1[0].isoformat(),
                    evaluator.STAGE1[1].isoformat(),
                ]
            },
            "base_cost_by_clock": {
                name: {"cagr_to_strict_mdd": 4.0 if name == "primary" else -1.0}
                for name in evaluator.ALL_CLOCK_NAMES
            },
            "full_qualification_by_control": {
                "one_observation_delay": {"passed": False},
                "random_side": {"passed": False},
            },
            "gates": {name: True for name in evaluator.STAGE1_GATE_NAMES},
            "gate_passed": True,
            "opened_windows": ["stage1_2021_2022"],
            "sealed_windows": ["stage2_2023", "2024", "2025", "2026_ytd"],
            "disposition": "PASS_STAGE1_OPEN_2023_ONCE",
        }
    )
    stage1_path.write_text(json.dumps(stage1))
    monkeypatch.setattr(evaluator, "STAGE1_OUTPUT", stage1_path)
    monkeypatch.setattr(
        evaluator,
        "verify_evaluator_freeze",
        lambda: {"manifest_hash": freeze_hash},
    )
    opened: list[evaluator.TimeWindow] = []

    def fake_load_execution_window(
        window: evaluator.TimeWindow,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
        opened.append(window)
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {"physical_window": [window[0].isoformat(), window[1].isoformat()]},
        )

    monkeypatch.setattr(evaluator, "load_execution_window", fake_load_execution_window)
    monkeypatch.setattr(evaluator, "simulate_schedule", _synthetic_simulation)
    monkeypatch.setattr(evaluator, "STAGE2_OUTPUT", tmp_path / "stage2.json")
    monkeypatch.setattr(evaluator, "STAGE2_DOC", tmp_path / "stage2.md")
    report = evaluator.evaluate_stage2()
    assert opened == [evaluator.STAGE2]
    assert report["stage1_manifest_hash"] == stage1["manifest_hash"]
    assert report["opened_windows"] == ["stage1_2021_2022", "stage2_2023"]
    assert report["sealed_windows"] == ["2024", "2025", "2026_ytd"]
