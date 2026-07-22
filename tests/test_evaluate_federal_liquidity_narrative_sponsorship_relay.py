from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from training import evaluate_federal_liquidity_narrative_sponsorship_relay as evaluator


def _headline(
    *,
    absolute_return: float = 10.0,
    ratio: float = 4.0,
    mdd: float = 8.0,
    gross_bp: float = 50.0,
    p_value: float = 0.01,
) -> dict[str, Any]:
    return {
        "absolute_return_pct": absolute_return,
        "cagr_pct": 30.0,
        "strict_mdd_pct": mdd,
        "cagr_to_strict_mdd": ratio,
        "trades": 20,
        "mean_gross_underlying_bp": gross_bp,
        "monthly_cluster_signflip_p": p_value,
        "monthly_clusters": 12,
    }


def test_frozen_schedules_are_exact_causal_split_contained_and_nonoverlapping() -> None:
    schedules = evaluator.load_schedules()
    assert set(schedules) == set(evaluator.ALL_CLOCK_NAMES)
    assert len(schedules["primary"]) == 89
    assert len(evaluator._window_schedule(schedules["primary"], evaluator.STAGE1)) == 67
    assert len(evaluator._window_schedule(schedules["primary"], evaluator.STAGE2)) == 22

    for frame in schedules.values():
        assert frame["signal_day"].equals(frame["signal_time"])
        assert (
            frame["entry_time"] - frame["signal_time"] == pd.Timedelta(minutes=10)
        ).all()
        assert (frame["exit_time"] - frame["entry_time"] == pd.Timedelta(days=7)).all()
        assert all(
            evaluator._belongs_to_exactly_one_stage(row)
            for row in frame.itertuples(index=False)
        )
        for window in (evaluator.STAGE1, evaluator.STAGE2):
            selected = evaluator._window_schedule(frame, window)
            assert (
                selected["entry_time"].iloc[1:].reset_index(drop=True)
                >= selected["exit_time"].iloc[:-1].reset_index(drop=True)
            ).all()


def test_one_extra_bar_delay_changes_only_clock_and_timestamps() -> None:
    primary = evaluator.load_schedules()["primary"]
    delayed = evaluator.one_extra_bar_delay_schedule(primary)
    assert delayed["clock_name"].eq(evaluator.DELAY_CLOCK_NAME).all()
    assert delayed["entry_time"].equals(primary["entry_time"] + pd.Timedelta(minutes=5))
    assert delayed["exit_time"].equals(primary["exit_time"] + pd.Timedelta(minutes=5))
    assert delayed["signal_time"].equals(primary["signal_time"])
    assert delayed["side"].equals(primary["side"])
    assert len(evaluator._window_schedule(delayed, evaluator.STAGE1)) == 67
    assert len(evaluator._window_schedule(delayed, evaluator.STAGE2)) == 22


def test_frozen_schedule_is_compatible_with_reused_strict_simulator() -> None:
    schedule = evaluator.load_schedules()["primary"].iloc[[0]].copy()
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
    result = evaluator._simulate(
        market,
        funding,
        schedule,
        window=(schedule["signal_day"].iloc[0].floor("D"), exit_time),
        cost_rate=0.0,
        cfg=evaluator.EvaluationConfig(),
    )
    assert result["trades"] == 1
    assert result["monthly_cluster_signflip"]["cluster_count"] == 1


def test_monthly_cluster_signflip_is_deterministic_and_clusters_by_entry_month() -> None:
    trades = [
        {"entry_time": "2020-01-02T00:00:00Z", "net_return": 0.01},
        {"entry_time": "2020-01-20T00:00:00Z", "net_return": 0.02},
        {"entry_time": "2020-02-03T00:00:00Z", "net_return": -0.005},
    ]
    first = evaluator.monthly_cluster_signflip(trades, draws=1_000, seed=7)
    second = evaluator.monthly_cluster_signflip(trades, draws=1_000, seed=7)
    assert first == second
    assert first["cluster_count"] == 2
    assert first["monthly_net_return_sums"] == {
        "2020-01": pytest.approx(0.03),
        "2020-02": pytest.approx(-0.005),
    }
    assert 0.0 < first["p_value_one_sided"] <= 1.0


def test_performance_gates_enforce_all_preregistered_economic_contracts() -> None:
    primary = _headline()
    stress = _headline(absolute_return=2.0)
    delayed = _headline(absolute_return=1.0)
    subperiods = {name: _headline() for name in ("2020", "2021", "2022")}
    controls = {
        "liquidity_only": _headline(gross_bp=40.0),
        "narrative_only": _headline(gross_bp=42.0),
        "disagreement": _headline(gross_bp=44.0),
    }
    gates, margins = evaluator._performance_gates(
        primary, stress, delayed, subperiods, controls
    )
    assert all(gates.values())
    assert margins == {
        "liquidity_only": 10.0,
        "narrative_only": 8.0,
        "disagreement": 6.0,
    }

    failed_primary = dict(primary, cagr_to_strict_mdd=2.99)
    failed, _ = evaluator._performance_gates(
        failed_primary, stress, delayed, subperiods, controls
    )
    assert failed["cagr_to_strict_mdd_at_least_3"] is False

    too_close = dict(controls)
    too_close["disagreement"] = _headline(gross_bp=45.01)
    failed, _ = evaluator._performance_gates(
        primary, stress, delayed, subperiods, too_close
    )
    assert (
        failed["primary_mean_gross_margin_over_each_component_at_least_5bp"]
        is False
    )


def test_falsification_controls_are_not_economic_gate_inputs() -> None:
    primary = _headline()
    controls = {
        "liquidity_only": _headline(gross_bp=40.0),
        "narrative_only": _headline(gross_bp=40.0),
        "disagreement": _headline(gross_bp=40.0),
    }
    gates, _ = evaluator._performance_gates(
        primary,
        _headline(),
        _headline(),
        {"year": _headline()},
        controls,
    )
    assert not any(name in gates for name in evaluator.FALSIFICATION_CONTROLS)


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
    assert result["opened_windows"] == []
    assert result["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert result["funding_rows_parsed_during_freeze"] == 0
    assert result["simulation_run_during_freeze"] is False


def test_stage2_refuses_missing_or_malformed_stage1_before_any_data_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage1 = tmp_path / "stage1.json"
    monkeypatch.setattr(evaluator, "STAGE1_OUTPUT", stage1)

    def forbidden_loader(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("execution data opened for absent or forged Stage1")

    monkeypatch.setattr(evaluator, "load_stage1_execution_window", forbidden_loader)
    with pytest.raises(ValueError, match="absent"):
        evaluator._verified_passing_stage1(expected_freeze_hash="freeze")

    forged = evaluator._seal(
        {
            "protocol_version": evaluator.STAGE1_PROTOCOL,
            "policy_id": evaluator.POLICY_ID,
            "stage": "forged",
        }
    )
    stage1.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="identity changed"):
        evaluator._verified_passing_stage1(expected_freeze_hash="freeze")


def test_well_sealed_stage1_is_replayed_on_train_only_before_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(json.dumps({"placeholder": True}))
    monkeypatch.setattr(evaluator, "STAGE1_OUTPUT", stage1)
    monkeypatch.setattr(evaluator, "_validate_stage1_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluator, "load_schedules", lambda: {"primary": pd.DataFrame()})
    seen: list[evaluator.TimeWindow] = []

    def train_only_loader() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        seen.append(evaluator.STAGE1)
        return pd.DataFrame(), pd.DataFrame(), {"physical_window": "train"}

    monkeypatch.setattr(evaluator, "load_stage1_execution_window", train_only_loader)
    monkeypatch.setattr(evaluator, "_build_stage1_core", lambda **kwargs: {"replayed": True})
    with pytest.raises(ValueError, match="not reproducible"):
        evaluator._verified_passing_stage1(expected_freeze_hash="freeze")
    assert seen == [evaluator.STAGE1]


def test_stage2_loader_cannot_bypass_stage1_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "verify_evaluator_freeze",
        lambda: {"manifest_hash": "freeze"},
    )

    def rejected(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ValueError("Stage1 failed; 2023 remains sealed")

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("2023 parser opened before Stage1 replay")

    monkeypatch.setattr(evaluator, "_verified_passing_stage1", rejected)
    monkeypatch.setattr(evaluator, "_parse_market_window", forbidden)
    monkeypatch.setattr(evaluator, "_parse_funding_window", forbidden)
    with pytest.raises(ValueError, match="remains sealed"):
        evaluator.load_stage2_execution_window()
    assert not hasattr(evaluator, "_load_physical_execution_window")
    assert "load_stage2_execution_window" not in evaluator.__all__
    assert evaluator.__all__ == [
        "freeze_evaluator",
        "evaluate_stage1",
        "evaluate_stage2",
        "main",
    ]


def test_stage1_prefix_hash_must_match_frozen_independent_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluator, "_verify_execution_manifests", lambda: None)
    monkeypatch.setattr(
        evaluator,
        "_parse_market_window",
        lambda *args: (pd.DataFrame(), {"window_line_sha256": "tampered"}),
    )
    monkeypatch.setattr(
        evaluator,
        "_parse_funding_window",
        lambda *args: (
            pd.DataFrame(),
            {"window_line_sha256": evaluator.STAGE1_FUNDING_WINDOW_LINE_SHA256},
        ),
    )
    with pytest.raises(ValueError, match="market prefix hash changed"):
        evaluator.load_stage1_execution_window()


def test_funding_parser_does_not_split_sealed_boundary_values(tmp_path: Path) -> None:
    path = tmp_path / "funding.csv.gz"
    header = (
        "funding_time_ms,funding_time_utc,symbol,funding_rate,"
        "settlement_mark_price,mark_open_time_ms\n"
    )
    opened = "0,2020-01-01T00:00:00,BTCUSDT,0.0001,10000,0\n"
    # These deliberately invalid values must never be split or converted.
    sealed = "1,2020-01-01T08:00:00,BTCUSDT,NOT_A_FLOAT,NOT_A_PRICE,1\n"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(header + opened + sealed)
    frame, diagnostics = evaluator._parse_funding_window_causal(
        path,
        pd.Timestamp("2020-01-01T00:00:00Z"),
        pd.Timestamp("2020-01-01T08:00:00Z"),
    )
    assert len(frame) == 1
    assert diagnostics["stopped_before_parsing_end_boundary"] is True
    assert diagnostics["sealed_boundary_values_parsed"] is False


def test_frozen_artifacts_are_write_once(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    assert evaluator._write_once_text(path, "one\n", label="test") == "created"
    assert (
        evaluator._write_once_text(path, "one\n", label="test")
        == "verified_existing"
    )
    with pytest.raises(ValueError, match="refusing to overwrite"):
        evaluator._write_once_text(path, "two\n", label="test")


def test_freeze_copies_exact_preregistered_economic_gate_object(tmp_path: Path) -> None:
    freeze = evaluator.freeze_evaluator(tmp_path / "freeze.json")
    registration = evaluator._load_json(evaluator.PREREGISTRATION)
    assert freeze["economic_gates"] == registration["economic_gates"]
    assert freeze["monthly_cluster_signflip"]["p_value_max"] == 0.05
