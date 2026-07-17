from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from training import evaluate_federal_liquidity_component_concordance as evaluator


def _metric(
    *,
    absolute_return: float = 10.0,
    ratio: float = 4.0,
    mdd: float = 8.0,
    trades: int = 100,
    gross_bp: float = 50.0,
    p_value: float = 0.01,
) -> dict[str, object]:
    return {
        "absolute_return_pct": absolute_return,
        "cagr_pct": 30.0,
        "strict_mdd_pct": mdd,
        "cagr_to_strict_mdd": ratio,
        "trades": trades,
        "mean_gross_underlying_bp": gross_bp,
        "weekly_cluster_signflip": {
            "p_value_one_sided": p_value,
            "cluster_count": trades,
        },
    }


def _stored_metric(
    *,
    absolute_return: float = 10.0,
    ratio: float = 4.0,
    trades: int = 100,
) -> dict[str, Any]:
    return {
        "absolute_return_pct": absolute_return,
        "cagr_pct": 30.0,
        "strict_mdd_pct": 8.0,
        "cagr_to_strict_mdd": ratio,
        "trades": trades,
        "mean_gross_underlying_bp": 50.0,
        "weekly_cluster_signflip_p": 0.01,
        "weekly_clusters": trades,
    }


def _well_formed_forged_stage1(freeze_hash: str) -> dict[str, Any]:
    registration = evaluator._verify_static_inputs()
    rows: list[dict[str, Any]] = []
    for candidate_id in evaluator.CANDIDATE_IDS:
        support = registration["source_only_support"][candidate_id]["windows"]
        primary_distribution = {
            "trades": support["train"]["count"],
            "longs": support["train"]["long"],
            "shorts": support["train"]["short"],
            "max_single_month_count": support["train"]["max_single_month_count"],
            "max_single_month_share": support["train"]["max_single_month_share"],
        }
        control_distribution = {
            "trades": 100,
            "longs": 50,
            "shorts": 50,
            "max_single_month_count": 5,
            "max_single_month_share": 0.05,
        }
        controls = {
            name: _stored_metric(absolute_return=-1.0, ratio=1.0)
            for name in evaluator.ALL_CLOCK_NAMES
            if name != "primary"
        }
        distributions = {
            "primary": primary_distribution,
            **{
                name: dict(control_distribution)
                for name in evaluator.ALL_CLOCK_NAMES
                if name != "primary"
            },
        }
        control_details: dict[str, Any] = {}
        for name in evaluator.FALSIFICATION_CONTROLS:
            control_subperiods = {
                year: _stored_metric(absolute_return=-1.0, trades=30)
                for year in evaluator.STAGE1_SUBPERIODS
            }
            control_stress = _stored_metric(absolute_return=-1.0)
            control_details[name] = {
                "stress_10bp_per_side": control_stress,
                "subperiods": control_subperiods,
                "gates": evaluator._stored_performance_gates(
                    controls[name],
                    control_stress,
                    control_subperiods,
                    distributions[name],
                    p_max=0.025,
                    total_trades_min=90,
                    each_side_min=40,
                    subperiod_trade_mins={"2020": 20, "2021": 20, "2022": 20},
                    month_share_max=0.08,
                ),
            }
        row = {
            "candidate_id": candidate_id,
            "primary": _stored_metric(trades=support["train"]["count"]),
            "stress_10bp_per_side": _stored_metric(
                absolute_return=3.0, trades=support["train"]["count"]
            ),
            "subperiods": {
                year: _stored_metric(trades=support[year]["count"])
                for year in evaluator.STAGE1_SUBPERIODS
            },
            "controls": controls,
            "entry_distribution": distributions,
            "falsification_control_details": control_details,
            "control_overall_qualification": {},
            "gates": {},
            "qualified": True,
        }
        row["gates"], row["control_overall_qualification"] = (
            evaluator._recompute_stored_stage1_row(row, registration)
        )
        row["qualified"] = all(row["gates"].values())
        rows.append(row)
    qualified = sorted(rows, key=evaluator._stage1_selection_key)
    selected = qualified[0]["candidate_id"]
    core = {
        "protocol_version": evaluator.STAGE1_PROTOCOL,
        "family_id": "FLCC-1",
        "stage": evaluator.STAGE1_ID,
        "as_of_date": "2026-07-17",
        "evaluator_freeze_manifest_hash": freeze_hash,
        "evaluator_source_sha256": evaluator._sha256(evaluator.EVALUATOR_SOURCE),
        "config": asdict(evaluator.EvaluationConfig()),
        "physical_source_diagnostics": {
            "physical_window": [value.isoformat() for value in evaluator.STAGE1],
            "market": {},
            "funding": {},
            "full_market_file_sha256_not_recomputed": True,
            "full_funding_file_sha256_not_recomputed": True,
            "reason": "forged",
        },
        "stage1_window": [value.isoformat() for value in evaluator.STAGE1],
        "candidate_count": len(rows),
        "candidates": rows,
        "qualified_candidate_ids": [row["candidate_id"] for row in qualified],
        "selected_candidate_id": selected,
        "stage1_passed": True,
        "advance_to_stage2": True,
        "2023_outcomes_opened": False,
        "2023_execution_rows_parsed": 0,
        "2023_funding_rows_parsed": 0,
        "opened_windows": evaluator.STAGE1_OPENED_WINDOWS,
        "sealed_windows": evaluator.STAGE1_SEALED_WINDOWS,
        "disposition": "PASS_STAGE1_OPEN_2023_ONCE",
    }
    return evaluator._seal(core)


def test_schedule_rebuild_is_exact_causal_and_nonoverlapping() -> None:
    schedules = evaluator.load_schedules()
    assert set(schedules) == {
        "FLCC-H4-Q60",
        "FLCC-H4-Q65",
        "FLCC-H8-Q60",
        "FLCC-H8-Q65",
    }
    for candidate_schedules in schedules.values():
        assert set(candidate_schedules) == set(evaluator.ALL_CLOCK_NAMES)
        for frame in candidate_schedules.values():
            assert frame["signal_day"].equals(frame["signal_time"])
            assert (
                frame["entry_time"] - frame["signal_time"] == pd.Timedelta(minutes=5)
            ).all()
            assert (
                frame["exit_time"] - frame["entry_time"] == pd.Timedelta(days=5)
            ).all()
            assert (
                frame["entry_time"].iloc[1:].reset_index(drop=True)
                >= frame["exit_time"].iloc[:-1].reset_index(drop=True)
            ).all()
            assert frame["exit_time"].max() < pd.Timestamp("2024-01-01", tz="UTC")


def test_frozen_schedule_is_compatible_with_reused_strict_simulator() -> None:
    schedule = evaluator.load_schedules()["FLCC-H4-Q60"]["primary"].iloc[[0]].copy()
    entry = schedule["entry_time"].iloc[0]
    exit_time = schedule["exit_time"].iloc[0]
    timestamps = pd.date_range(entry, exit_time, freq="5min", inclusive="both")
    market = pd.DataFrame(
        {
            "date": timestamps,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
        }
    )
    funding = pd.DataFrame(
        columns=["funding_time", "funding_rate", "settlement_mark_price"]
    )
    result = evaluator.simulate_schedule(
        market,
        funding,
        schedule,
        period_start=schedule["signal_day"].iloc[0].floor("D"),
        period_end=exit_time,
        cost_rate=0.0,
    )
    assert result["trades"] == 1


def test_freeze_never_calls_execution_parsers_or_simulator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("outcome path opened during evaluator freeze")

    monkeypatch.setattr(evaluator, "_parse_market_window", forbidden)
    monkeypatch.setattr(evaluator, "_parse_funding_window", forbidden)
    monkeypatch.setattr(evaluator, "simulate_schedule", forbidden)
    result = evaluator.freeze_evaluator(tmp_path / "freeze.json")
    assert result["outcomes_opened"] is False
    assert result["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert result["funding_rows_parsed_during_freeze"] == 0
    assert result["simulation_run_during_freeze"] is False


def test_performance_gates_enforce_ratio_mdd_significance_and_distribution() -> None:
    metrics = _metric()
    stress = _metric(absolute_return=3.0)
    subperiods = {year: _metric(trades=30) for year in ("2020", "2021", "2022")}
    distribution = {
        "trades": 100,
        "longs": 50,
        "shorts": 50,
        "max_single_month_share": 0.05,
    }
    gates = evaluator._performance_gates(
        metrics,
        stress,
        subperiods,
        distribution,
        p_max=0.025,
        total_trades_min=90,
        each_side_min=40,
        subperiod_trade_mins={"2020": 20, "2021": 20, "2022": 20},
        month_share_max=0.08,
    )
    assert all(gates.values())

    failed = copy.deepcopy(metrics)
    failed["cagr_to_strict_mdd"] = 2.99
    failed_gates = evaluator._performance_gates(
        failed,
        stress,
        subperiods,
        distribution,
        p_max=0.025,
        total_trades_min=90,
        each_side_min=40,
        subperiod_trade_mins={"2020": 20, "2021": 20, "2022": 20},
        month_share_max=0.08,
    )
    assert failed_gates["cagr_to_strict_mdd_at_least_3"] is False


def test_falsification_control_requires_stress_and_subperiod_qualification() -> None:
    metrics = _stored_metric()
    stress = _stored_metric(absolute_return=-1.0)
    subperiods = {
        year: _stored_metric(trades=30) for year in evaluator.STAGE1_SUBPERIODS
    }
    distribution = {
        "trades": 100,
        "longs": 50,
        "shorts": 50,
        "max_single_month_count": 5,
        "max_single_month_share": 0.05,
    }
    gates = evaluator._stored_performance_gates(
        metrics,
        stress,
        subperiods,
        distribution,
        p_max=0.025,
        total_trades_min=90,
        each_side_min=40,
        subperiod_trade_mins={"2020": 20, "2021": 20, "2022": 20},
        month_share_max=0.08,
    )
    assert gates["stress_cost_absolute_return_positive"] is False
    assert not all(gates.values())


def test_stage1_selection_key_prioritizes_worst_year_then_overall() -> None:
    better_worst = {
        "candidate_id": "B",
        "primary": {"cagr_to_strict_mdd": 3.1},
        "stress_10bp_per_side": {"absolute_return_pct": 1.0},
        "subperiods": {
            "2020": {"cagr_to_strict_mdd": 2.0},
            "2021": {"cagr_to_strict_mdd": 2.1},
            "2022": {"cagr_to_strict_mdd": 2.2},
        },
    }
    better_overall = copy.deepcopy(better_worst)
    better_overall["candidate_id"] = "A"
    better_overall["primary"]["cagr_to_strict_mdd"] = 10.0
    better_overall["subperiods"]["2020"]["cagr_to_strict_mdd"] = 1.9
    assert (
        sorted([better_overall, better_worst], key=evaluator._stage1_selection_key)[0][
            "candidate_id"
        ]
        == "B"
    )


def test_stage2_refuses_missing_forged_or_tampered_stage1_before_data_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage1 = tmp_path / "stage1.json"
    monkeypatch.setattr(evaluator, "STAGE1_OUTPUT", stage1)
    with pytest.raises(ValueError, match="absent"):
        evaluator._verified_passing_stage1(expected_freeze_hash="freeze")

    forged_core = {
        "evaluator_freeze_manifest_hash": "freeze",
        "stage1_passed": True,
        "advance_to_stage2": True,
        "selected_candidate_id": "FLCC-H4-Q60",
        "2023_outcomes_opened": False,
    }
    stage1.write_text(json.dumps(evaluator._seal(forged_core)))

    def forbidden_loader(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("execution data opened for a forged Stage1 report")

    monkeypatch.setattr(evaluator, "load_execution_window", forbidden_loader)
    with pytest.raises(ValueError, match="identity changed"):
        evaluator._verified_passing_stage1(expected_freeze_hash="freeze")

    payload = evaluator._seal(forged_core)
    payload["selected_candidate_id"] = "FLCC-H8-Q65"
    stage1.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluator._verified_passing_stage1(expected_freeze_hash="freeze")


def test_well_formed_self_sealed_stage1_is_replayed_before_stage2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(json.dumps(_well_formed_forged_stage1("freeze")))
    monkeypatch.setattr(evaluator, "STAGE1_OUTPUT", stage1)
    monkeypatch.setattr(evaluator, "load_schedules", lambda: {"forged": {}})
    seen_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def stage1_only_loader(
        window: tuple[pd.Timestamp, pd.Timestamp],
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        seen_windows.append(window)
        assert window == evaluator.STAGE1
        diagnostics = {
            "physical_window": [value.isoformat() for value in window],
            "market": {},
            "funding": {},
            "full_market_file_sha256_not_recomputed": True,
            "full_funding_file_sha256_not_recomputed": True,
            "reason": "replayed",
        }
        return pd.DataFrame(), pd.DataFrame(), diagnostics

    monkeypatch.setattr(evaluator, "load_execution_window", stage1_only_loader)
    monkeypatch.setattr(
        evaluator,
        "_build_stage1_core",
        lambda **kwargs: {"replayed": bool(kwargs)},
    )
    with pytest.raises(ValueError, match="not reproducible from frozen sources"):
        evaluator._verified_passing_stage1(expected_freeze_hash="freeze")
    assert seen_windows == [evaluator.STAGE1]
