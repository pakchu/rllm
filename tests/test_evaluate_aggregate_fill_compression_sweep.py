from __future__ import annotations

from dataclasses import asdict, replace
import gzip
import json

import numpy as np
import pandas as pd
import pytest

from training import evaluate_aggregate_fill_compression_sweep as evaluate


def _market(rows: int = 170, price: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=rows, freq="5min"),
            "open": np.full(rows, price),
            "high": np.full(rows, price),
            "low": np.full(rows, price),
            "close": np.full(rows, price),
        }
    )


def _schedule(*, side: int = 1, entry: int = 4, exit_: int = 148) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=200, freq="5min")
    signal = entry - 2
    return pd.DataFrame(
        [
            {
                "origin_position": signal,
                "signal_position": signal,
                "entry_position": entry,
                "exit_position": exit_,
                "origin_date": str(dates[signal]),
                "signal_date": str(dates[signal]),
                "entry_date": str(dates[entry]),
                "exit_date": str(dates[exit_]),
                "side": side,
                "branch": "afcs_144",
                "delay_bars": 2,
                "hold_bars": 144,
            }
        ]
    )


def _funding(
    rows: list[tuple[pd.Timestamp, float, float]] | None = None,
) -> pd.DataFrame:
    rows = [] if rows is None else rows
    return pd.DataFrame(
        [
            {
                "funding_time_ms": int(time.value // 1_000_000),
                "funding_time_utc": time.isoformat(),
                "funding_rate": rate,
                "settlement_mark_price": mark,
                "funding_time": time,
            }
            for time, rate, mark in rows
        ],
        columns=[
            "funding_time_ms",
            "funding_time_utc",
            "funding_rate",
            "settlement_mark_price",
            "funding_time",
        ],
    )


def _cfg(**changes: object) -> evaluate.EvaluationConfig:
    return replace(evaluate.EvaluationConfig(), cluster_permutations=64, **changes)


def test_market_parser_stops_before_sentinel_values(tmp_path) -> None:
    path = tmp_path / "market.csv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("date,open,high,low,close\n")
        handle.write("2022-12-31 23:55:00,100,101,99,100\n")
        handle.write(
            "2023-01-01 00:00:00,DO_NOT_PARSE,DO_NOT_PARSE,DO_NOT_PARSE,DO_NOT_PARSE\n"
        )
    frame = evaluate._parse_market_before(path, pd.Timestamp("2023-01-01"))
    assert len(frame) == 1
    assert frame.loc[0, "close"] == 100.0


def test_funding_parser_stops_before_sentinel_values(tmp_path) -> None:
    path = tmp_path / "funding.csv.gz"
    cutoff = pd.Timestamp("2023-01-01")
    cutoff_ms = int(cutoff.timestamp() * 1_000)
    with gzip.open(path, "wt") as handle:
        handle.write(
            "funding_time_ms,funding_time_utc,funding_rate,settlement_mark_price\n"
        )
        handle.write(f"{cutoff_ms - 1},2022-12-31T23:59:59.999Z,0.001,100\n")
        handle.write(f"{cutoff_ms},DO_NOT_PARSE,DO_NOT_PARSE,DO_NOT_PARSE\n")
    frame = evaluate._parse_funding_before(path, cutoff)
    assert len(frame) == 1
    assert frame.loc[0, "settlement_mark_price"] == 100.0


def test_fixed_quantity_ledger_and_full_clock_cagr() -> None:
    market = _market()
    market.loc[148, "open"] = 110.0
    metrics = evaluate.simulate_schedule(
        market,
        _funding(),
        _schedule(),
        period_start=pd.Timestamp("2020-01-01"),
        period_end=pd.Timestamp("2021-01-01"),
        cost_rate=0.0,
        cfg=_cfg(),
    )
    assert metrics["absolute_return_pct"] == pytest.approx(5.0)
    years = 366.0 / 365.25
    assert metrics["cagr_pct"] == pytest.approx((1.05 ** (1 / years) - 1) * 100)


def test_strict_mdd_applies_favorable_then_adverse_and_liquidation_cost() -> None:
    market = _market()
    market.loc[4, "high"] = 120.0
    market.loc[147, "low"] = 90.0
    metrics = evaluate.simulate_schedule(
        market,
        _funding(),
        _schedule(),
        period_start=pd.Timestamp("2020-01-01"),
        period_end=pd.Timestamp("2020-01-02"),
        cost_rate=0.001,
        cfg=_cfg(),
    )
    favorable = 1.0 - 0.0005 + 0.5 * 0.20
    adverse = 1.0 - 0.0005 + 0.5 * -0.10 - 0.0005 * 0.90
    expected = (1.0 - adverse / favorable) * 100.0
    assert metrics["strict_mdd_pct"] == pytest.approx(expected)


def test_funding_uses_exact_settlement_mark_and_half_open_interval() -> None:
    market = _market()
    inside = market.loc[100, "date"]
    at_exit = market.loc[148, "date"]
    metrics = evaluate.simulate_schedule(
        market,
        _funding([(inside, 0.01, 200.0), (at_exit, 0.01, 200.0)]),
        _schedule(side=1),
        period_start=pd.Timestamp("2020-01-01"),
        period_end=pd.Timestamp("2020-01-02"),
        cost_rate=0.0,
        cfg=_cfg(),
    )
    assert metrics["funding_settlement_count"] == 1
    assert metrics["funding_cash_pct_initial"] == pytest.approx(-1.0)
    assert metrics["absolute_return_pct"] == pytest.approx(-1.0)


def test_schedule_rejects_delay_or_hold_mutation() -> None:
    changed = _schedule(exit_=147)
    with pytest.raises(ValueError, match="frozen hold"):
        evaluate.simulate_schedule(
            _market(),
            _funding(),
            changed,
            period_start=pd.Timestamp("2020-01-01"),
            period_end=pd.Timestamp("2020-01-02"),
            cost_rate=0.0,
            cfg=_cfg(),
        )


def test_schedule_rejects_any_frozen_timestamp_mutation() -> None:
    changed = _schedule()
    changed.loc[0, "exit_date"] = str(
        pd.Timestamp(changed.loc[0, "exit_date"]) + pd.Timedelta(minutes=5)
    )
    with pytest.raises(ValueError, match="exit timestamp differs"):
        evaluate.simulate_schedule(
            _market(),
            _funding(),
            changed,
            period_start=pd.Timestamp("2020-01-01"),
            period_end=pd.Timestamp("2020-01-02"),
            cost_rate=0.0,
            cfg=_cfg(),
        )


def test_weekly_cluster_signflip_is_deterministic() -> None:
    args = ([0.01, 0.02, -0.01], ["2020-01-01", "2020-01-08", "2020-01-15"])
    first = evaluate.weekly_cluster_signflip(*args, permutations=1_000, seed=7)
    second = evaluate.weekly_cluster_signflip(*args, permutations=1_000, seed=7)
    assert first == second


def test_freeze_manifest_hash_detects_mutation(monkeypatch, tmp_path) -> None:
    payload = {
        "evaluation_source_sha256": evaluate._sha256(evaluate.EVALUATION_SOURCE),
        "support_commit": evaluate.SUPPORT_COMMIT,
        "evaluation_config": asdict(evaluate.EvaluationConfig()),
        "opened_windows": [],
        "mutable_parameters": [],
        "execution_ohlc_rows_parsed_during_freeze": 0,
        "funding_settlement_marks_loaded_during_freeze": 0,
        "execution_simulation_run_during_freeze": False,
        "control_schedules": {},
    }
    payload["manifest_hash"] = evaluate._canonical_hash(payload)
    payload["opened_windows"] = ["stage1"]
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(evaluate, "EVALUATION_FREEZE", path)
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        evaluate.verify_evaluation_freeze()


def test_stage1_tampering_cannot_unlock_stage2(monkeypatch, tmp_path) -> None:
    payload = {
        "candidate_id": "AFCS-144",
        "stage": "stage1_2020_2022",
        "stage1_qualifies": True,
        "gate": {"forged": True},
        "next_action": "open_2023",
        "manifest_hash": "forged",
    }
    path = tmp_path / "stage1.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(evaluate, "STAGE1_OUTPUT", path)
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        evaluate.verify_stage1_output()


def test_all_time_shift_and_random_side_placebos_are_gate_enforced() -> None:
    assert evaluate.REJECTION_PLACEBOS == (
        "one_hour_signal_delay",
        "one_day_shifted_clock",
        "random_side",
    )


def test_stage2_placebo_full_gate_requires_both_halves() -> None:
    metrics = {
        "absolute_return_pct": 1.0,
        "cagr_to_strict_mdd": 4.0,
        "strict_mdd_pct": 10.0,
        "weekly_cluster_signflip": {"p_value_one_sided": 0.05},
        "mean_gross_underlying_move_bp": 21.0,
        "trade_count": 60,
    }
    stress = {"absolute_return_pct": 0.1}
    halves = {
        "2023_h1": {"absolute_return_pct": 1.0, "trade_count": 25},
        "2023_h2": {"absolute_return_pct": -0.1, "trade_count": 25},
    }
    assert not evaluate._passes_stage2_gate(metrics, stress, halves)
    halves["2023_h2"]["absolute_return_pct"] = 0.1
    assert evaluate._passes_stage2_gate(metrics, stress, halves)
