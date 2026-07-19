from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from training import evaluate_coinm_next_maturity_shock_relay as evaluator


def _market(*, high: float = 100.0, low: float = 100.0) -> pd.DataFrame:
    entry = pd.Timestamp("2022-01-01T00:00:00Z")
    dates = pd.date_range(entry, periods=evaluator.HOLD_BARS + 1, freq="5min")
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
        }
    )
    frame.loc[0, "high"] = high
    frame.loc[0, "low"] = low
    return frame


def _schedule() -> pd.DataFrame:
    signal = pd.Timestamp("2021-12-31T23:50:00Z")
    entry = signal + pd.Timedelta(minutes=10)
    return pd.DataFrame(
        {
            "candidate_id": [evaluator.CANDIDATE],
            "clock_name": ["primary"],
            "signal_time": [signal],
            "feature_available_time": [signal + pd.Timedelta(minutes=5)],
            "entry_time": [entry],
            "exit_time": [entry + pd.Timedelta(hours=3)],
            "side": [1],
            "pair": ["front|next"],
        }
    )


def _funding(rate: float, mark: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "funding_time": [pd.Timestamp("2022-01-01T00:00:00Z")],
            "funding_rate": [rate],
            "settlement_mark_price": [mark],
        }
    )


def _empty_funding() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "funding_time": pd.Series([], dtype="datetime64[ns, UTC]"),
            "funding_rate": pd.Series([], dtype=float),
            "settlement_mark_price": pd.Series([], dtype=float),
        }
    )


def _simulate(market: pd.DataFrame, funding: pd.DataFrame) -> dict[str, object]:
    return evaluator.simulate_strict(
        market,
        funding,
        _schedule(),
        period_start=cast(pd.Timestamp, pd.Timestamp("2021-12-31T23:50:00Z")),
        period_end=cast(pd.Timestamp, pd.Timestamp("2022-01-01T03:05:00Z")),
        leverage=0.5,
        cost_rate_per_side=0.0,
    )


def test_boundary_credit_is_dropped_but_settlement_mark_hits_strict_mdd() -> None:
    result = _simulate(_market(), _funding(rate=-0.01, mark=50.0))
    trade = cast(list[dict[str, Any]], result["trade_details"])[0]
    assert trade["funding_cash"] == 0.0
    assert trade["dropped_boundary_funding_credits"] == 1
    assert result["strict_mdd_pct"] == pytest.approx(25.0)


def test_boundary_debit_is_retained() -> None:
    result = _simulate(_market(), _funding(rate=0.01, mark=100.0))
    trade = cast(list[dict[str, Any]], result["trade_details"])[0]
    assert trade["funding_cash"] == pytest.approx(-0.005)
    assert trade["funding_events_applied"] == 1
    assert result["absolute_return_pct"] == pytest.approx(-0.5)


def test_each_held_bar_marks_favorable_before_adverse() -> None:
    result = _simulate(_market(high=120.0, low=80.0), _empty_funding())
    assert result["absolute_return_pct"] == pytest.approx(0.0)
    assert result["strict_mdd_pct"] == pytest.approx(100.0 * (1.0 - 0.9 / 1.1))


def test_freeze_builds_schedules_without_opening_execution_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("freeze attempted to parse an outcome row")

    monkeypatch.setattr(evaluator, "_parse_market_month_window", forbidden)
    monkeypatch.setattr(evaluator, "_parse_funding_index_window", forbidden)
    report = evaluator.freeze_evaluator(tmp_path / "freeze.json")
    assert report["opened_windows"] == []
    assert report["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert report["funding_rows_parsed_during_freeze"] == 0
    assert report["schedule_records"]["primary"]["train_events"] == 93
    assert report["schedule_records"]["primary"]["test_events"] == 65
    assert all(report["schedule_invariant_checks"].values())


def test_schedule_family_is_frozen_and_nonoverlapping() -> None:
    schedules = evaluator.load_schedules()
    assert tuple(schedules) == evaluator.ALL_CLOCK_NAMES
    for name, schedule in schedules.items():
        record = evaluator._schedule_freeze_record(schedule, name)
        assert record["feature_available_after_signal_5m"] is True
        assert record["entry_delay_exact"] is True
        assert record["hold_exact"] is True
        assert record["globally_nonoverlapping"] is True
        assert record["pre_2024_only"] is True
        assert record["valid_sides"] is True


def test_funding_row_slice_never_reads_boundary_row(tmp_path: Path) -> None:
    source = tmp_path / "funding.csv.gz"
    with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
        handle.write(
            "funding_time_utc,symbol,funding_rate,settlement_mark_price\n"
        )
        handle.write("2020-01-01T00:00:00.000000Z,BTCUSDT,0.0001,100\n")
        handle.write("2020-01-01T08:00:00.000000Z,BTCUSDT,0.0001,101\n")
        handle.write("THIS_BOUNDARY_ROW_MUST_NOT_BE_READ\n")
    start = cast(pd.Timestamp, pd.Timestamp("2020-01-01T00:00:00Z"))
    end = cast(pd.Timestamp, pd.Timestamp("2020-01-01T16:00:00Z"))
    frame, diagnostics = evaluator._parse_funding_index_window(
        source,
        start,
        end,
        source_start=start,
        start_row_inclusive=0,
        end_row_exclusive=2,
    )
    assert len(frame) == 2
    assert diagnostics["boundary_row_read"] is False
    assert diagnostics["source_rows_consumed"] == 2


def _stub_metrics(
    *, ratio: float = 3.0, trades: int = 90, stress: bool = False
) -> dict[str, Any]:
    return {
        "absolute_return_pct": 0.0001,
        "cagr_pct": 45.0,
        "strict_mdd_pct": 15.0,
        "cagr_to_strict_mdd": 2.5 if stress else ratio,
        "trades": trades,
        "long_trades": trades // 2,
        "short_trades": trades - trades // 2,
        "mean_gross_underlying_bp": 1.0,
        "mean_gross_trade_bp": 0.5,
        "mean_net_trade_bp": 0.1,
        "weekly_cluster_signflip": {
            "p_value_two_sided": 0.10,
            "cluster_count": 10,
        },
    }


def test_gate_boundaries_and_names_match_preregistration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedules = {
        name: pd.DataFrame({"clock_name": [name]})
        for name in evaluator.ALL_CLOCK_NAMES
    }

    def fake_simulate(
        market: pd.DataFrame,
        funding: pd.DataFrame,
        schedule: pd.DataFrame,
        *,
        window: evaluator.TimeWindow,
        cost_rate: float,
        cfg: evaluator.EvaluationConfig,
    ) -> dict[str, Any]:
        del market, funding, window, cfg
        name = str(schedule["clock_name"].iloc[0])
        if cost_rate == evaluator.EvaluationConfig().stress_cost_notional_per_side:
            return _stub_metrics(stress=True)
        if name in evaluator.MECHANISM_CONTROLS:
            return _stub_metrics(ratio=2.75)
        return _stub_metrics()

    monkeypatch.setattr(evaluator, "_simulate", fake_simulate)
    empty = pd.DataFrame()
    cfg = evaluator.EvaluationConfig()
    train = evaluator._evaluate_stage(
        "train",
        evaluator.TRAIN,
        evaluator.TRAIN_SUBPERIODS,
        empty,
        empty,
        schedules,
        cfg,
    )
    assert train["gates"] == {
        "absolute_return_positive": True,
        "cagr_to_strict_mdd_at_least_3": True,
        "strict_mdd_at_most_15pct": True,
        "minimum_trades": True,
        "weekly_cluster_signflip_p_at_most_10pct": True,
        "each_subperiod_absolute_return_positive": True,
        "stress_absolute_return_positive": True,
        "stress_cagr_to_strict_mdd_at_least_2_5": True,
        "mechanism_control_margin_at_least_0_25": True,
    }
    test_schedules = {
        name: frame.assign(clock_name=name) for name, frame in schedules.items()
    }
    test = evaluator._evaluate_stage(
        "test",
        evaluator.TEST,
        evaluator.TEST_SUBPERIODS,
        empty,
        empty,
        test_schedules,
        cfg,
    )
    assert "mechanism_control_margin_at_least_0_25" not in test["gates"]


def test_train_replay_mismatch_blocks_test_unseal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core = {
        "train_passed": True,
        "evaluator_freeze_manifest_hash": "freeze-hash",
        "evaluator_source_sha256": evaluator._sha256(evaluator.EVALUATOR_SOURCE),
    }
    stored = evaluator._seal(core)
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps(stored), encoding="utf-8")
    monkeypatch.setattr(evaluator, "TRAIN_OUTPUT", train_path)
    monkeypatch.setattr(
        evaluator,
        "_build_train_report",
        lambda: {**stored, "train_passed": False},
    )
    with pytest.raises(ValueError, match="does not exactly reproduce"):
        evaluator._verified_passing_train("freeze-hash")


def test_each_preregistered_gate_rejects_just_beyond_its_boundary() -> None:
    primary = _stub_metrics()
    stress = _stub_metrics(stress=True)
    subperiods = {"half": _stub_metrics()}
    margins = {name: 0.25 for name in evaluator.MECHANISM_CONTROLS}
    assert all(
        evaluator._stage_gates(
            "train", primary, stress, subperiods, margins
        ).values()
    )

    cases: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], dict[str, float]]] = []
    cases.append(
        (
            "absolute_return_positive",
            {**primary, "absolute_return_pct": 0.0},
            stress,
            subperiods,
            margins,
        )
    )
    cases.append(
        (
            "cagr_to_strict_mdd_at_least_3",
            {**primary, "cagr_to_strict_mdd": 2.999999},
            stress,
            subperiods,
            margins,
        )
    )
    cases.append(
        (
            "strict_mdd_at_most_15pct",
            {**primary, "strict_mdd_pct": 15.0001},
            stress,
            subperiods,
            margins,
        )
    )
    cases.append(
        (
            "minimum_trades",
            {**primary, "trades": 89},
            stress,
            subperiods,
            margins,
        )
    )
    cases.append(
        (
            "weekly_cluster_signflip_p_at_most_10pct",
            {
                **primary,
                "weekly_cluster_signflip": {
                    "p_value_two_sided": 0.100001,
                    "cluster_count": 10,
                },
            },
            stress,
            subperiods,
            margins,
        )
    )
    cases.append(
        (
            "each_subperiod_absolute_return_positive",
            primary,
            stress,
            {"half": {**_stub_metrics(), "absolute_return_pct": 0.0}},
            margins,
        )
    )
    cases.append(
        (
            "stress_absolute_return_positive",
            primary,
            {**stress, "absolute_return_pct": 0.0},
            subperiods,
            margins,
        )
    )
    cases.append(
        (
            "stress_cagr_to_strict_mdd_at_least_2_5",
            primary,
            {**stress, "cagr_to_strict_mdd": 2.499999},
            subperiods,
            margins,
        )
    )
    cases.append(
        (
            "mechanism_control_margin_at_least_0_25",
            primary,
            stress,
            subperiods,
            {**margins, evaluator.MECHANISM_CONTROLS[0]: 0.249999},
        )
    )
    for gate, candidate, stressed, halves, control_margins in cases:
        result = evaluator._stage_gates(
            "train", candidate, stressed, halves, control_margins
        )
        assert result[gate] is False, gate

    test_primary = {**primary, "trades": 59}
    test_gates = evaluator._stage_gates(
        "test", test_primary, stress, subperiods, margins
    )
    assert test_gates["minimum_trades"] is False
