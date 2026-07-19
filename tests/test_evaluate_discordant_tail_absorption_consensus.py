from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from training import evaluate_discordant_tail_absorption_consensus as evaluator


def _market(
    *,
    start: str = "2023-03-01T04:05:00Z",
    periods: int = 97,
    price: float = 100.0,
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="5min")
    return pd.DataFrame(
        {
            "date": dates,
            "open": np.full(periods, price),
            "high": np.full(periods, price),
            "low": np.full(periods, price),
            "close": np.full(periods, price),
        }
    )


def _clock(
    market: pd.DataFrame,
    *,
    side: int = 1,
    entry_position: int = 0,
) -> dict[str, Any]:
    entry = evaluator._utc(market.iloc[entry_position]["date"])
    exit_time = evaluator._utc(
        market.iloc[entry_position + evaluator.EvaluationConfig().hold_bars]["date"]
    )
    return {
        "candidate": evaluator.POLICY_ID,
        "control": "primary",
        "split": "synthetic",
        "decision_time": entry - pd.Timedelta(minutes=5),
        "feature_available_time": entry - pd.Timedelta(minutes=5),
        "entry_time": entry,
        "exit_time": exit_time,
        "side": side,
    }


def _funding(
    rows: list[tuple[pd.Timestamp, float, float]] | None = None,
) -> pd.DataFrame:
    rows = rows or []
    return pd.DataFrame(
        {
            "funding_time": pd.to_datetime(
                [row[0] for row in rows], utc=True, errors="raise"
            ),
            "symbol": ["BTCUSDT"] * len(rows),
            "funding_rate": [row[1] for row in rows],
            "settlement_mark_price": [row[2] for row in rows],
        }
    )


def _simulate(
    market: pd.DataFrame,
    clocks: list[dict[str, Any]],
    *,
    funding: pd.DataFrame | None = None,
    cost: float = 0.0,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    start = start or evaluator._utc(market.iloc[0]["date"])
    end = end or evaluator._utc(start + pd.Timedelta(days=365.25))
    return evaluator.simulate_strict(
        market,
        _funding() if funding is None else funding,
        pd.DataFrame(clocks),
        start=start,
        end=end,
        cost_rate_per_side=cost,
    )


def test_control_derivation_is_source_only_deterministic_and_stage_contained() -> None:
    frame = evaluator.derive_control_clocks()

    assert len(frame) == 695 * 6 + 679 + 703
    assert frame.groupby("control").size().to_dict() == {
        "all_six_flow_fade_side": 695,
        "all_six_premium_side": 695,
        "deterministic_random_side": 695,
        "direction_flip": 695,
        "extra_latency_1h": 695,
        "primary": 695,
        "stale_premium_pairing_24h": 679,
        "symbol_permuted_premium_pairing": 703,
    }
    schedules = {
        name: frame.loc[frame["control"].eq(name)].reset_index(drop=True)
        for name in evaluator.ALL_CONTROLS
    }
    primary = schedules["primary"]
    assert primary.groupby("split").size().to_dict() == {
        "train": 143,
        "test": 190,
        "eval": 247,
        "final": 115,
    }
    for name, schedule in schedules.items():
        expected_delay = pd.Timedelta(minutes=65 if name == "extra_latency_1h" else 5)
        assert (
            schedule["entry_time"]
            .sub(schedule["decision_time"])
            .eq(expected_delay)
            .all()
        )
        assert (
            schedule["exit_time"]
            .sub(schedule["entry_time"])
            .eq(pd.Timedelta(hours=8))
            .all()
        )
        assert schedule["side"].isin((-1, 1)).all()
        assert (
            schedule["entry_time"]
            .iloc[1:]
            .ge(schedule["exit_time"].iloc[:-1].to_numpy())
            .all()
        )
        for stage, (start, end) in evaluator.STAGE_WINDOWS.items():
            selected = schedule.loc[schedule["split"].eq(stage)]
            assert selected["entry_time"].ge(start).all()
            assert selected["exit_time"].le(end).all()
    for name in evaluator.SAME_CLOCK_CONTROLS:
        assert schedules[name][["decision_time", "entry_time", "exit_time"]].equals(
            primary[["decision_time", "entry_time", "exit_time"]]
        )
    assert schedules["direction_flip"]["side"].eq(-primary["side"]).all()
    expected_random = schedules["deterministic_random_side"]["decision_time"].map(
        evaluator._deterministic_random_side
    )
    assert schedules["deterministic_random_side"]["side"].eq(expected_random).all()
    assert (
        schedules["extra_latency_1h"]["entry_time"]
        .eq(primary["entry_time"] + pd.Timedelta(hours=1))
        .all()
    )


def test_random_control_uses_frozen_ascii_sha_contract() -> None:
    decision = evaluator._utc("2024-01-02T03:00:00Z")
    digest = hashlib.sha256(
        f"{evaluator.POLICY_ID}|2024-01-02T03:00:00Z".encode()
    ).hexdigest()

    assert evaluator._deterministic_random_side(decision) == (
        1 if int(digest[0], 16) % 2 == 0 else -1
    )


def test_evaluator_is_directly_executable_from_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, str(evaluator.EVALUATOR_SOURCE), "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--freeze" in completed.stdout
    assert "--stage {train,test,eval,final}" in completed.stdout


def test_freeze_opens_no_market_funding_or_simulation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[str] = []
    real_sha = evaluator._sha256

    def tracking_sha(path: str | Path) -> str:
        seen.append(str(path))
        return real_sha(path)

    def forbidden_outcome_open(*args: object, **kwargs: object) -> None:
        raise AssertionError("freeze tried to open an execution outcome")

    monkeypatch.setattr(evaluator, "CONTROL_CLOCKS", tmp_path / "controls.csv.gz")
    for stage in evaluator.STAGE_ORDER:
        monkeypatch.setitem(
            evaluator.STAGE_OUTPUTS, stage, tmp_path / f"unused-{stage}.json"
        )
    monkeypatch.setattr(evaluator, "_sha256", tracking_sha)
    monkeypatch.setattr(evaluator, "load_execution_window", forbidden_outcome_open)
    monkeypatch.setattr(evaluator, "simulate_strict", forbidden_outcome_open)
    output = tmp_path / "freeze.json"

    report = evaluator.freeze_evaluator(output)

    assert report["opened_windows"] == []
    assert report["sealed_windows"] == list(evaluator.STAGE_ORDER)
    assert report["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert report["funding_rows_parsed_during_freeze"] == 0
    assert str(evaluator.TRAIN_MARKET) not in seen
    assert str(evaluator.TRAIN_FUNDING) not in seen
    assert report["support_commit"] == "60b889c9e50fd8a365bcbf1b398635303393bc6d"
    assert evaluator.verify_evaluator_freeze(output) == report

    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    core["evaluation_config"] = {**core["evaluation_config"], "leverage": 1.0}
    output.write_text(json.dumps(evaluator._seal(core)), encoding="utf-8")
    with pytest.raises(ValueError, match="config changed"):
        evaluator.verify_evaluator_freeze(output)


def test_freeze_artifact_is_write_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(evaluator, "CONTROL_CLOCKS", tmp_path / "controls.csv.gz")
    for stage in evaluator.STAGE_ORDER:
        monkeypatch.setitem(
            evaluator.STAGE_OUTPUTS, stage, tmp_path / f"unused-{stage}.json"
        )
    output = tmp_path / "freeze.json"
    evaluator.freeze_evaluator(output)

    with pytest.raises(FileExistsError):
        evaluator.freeze_evaluator(output)


def test_schedule_loader_recomputes_and_rejects_control_side_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = evaluator.derive_control_clocks()
    changed = expected.copy()
    index = changed.index[changed["control"].eq("direction_flip")][0]
    changed.loc[index, "side"] = -int(changed.loc[index, "side"])
    path = tmp_path / "controls.csv.gz"
    monkeypatch.setattr(evaluator, "CONTROL_CLOCKS", path)
    first_hash = evaluator._write_control_clock(expected)
    assert evaluator._write_control_clock(expected) == first_hash
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        evaluator._write_control_clock(changed)

    evaluator.deterministic_gzip_csv(changed, path)
    monkeypatch.setattr(evaluator, "derive_control_clocks", lambda: expected)

    with pytest.raises(ValueError, match="control semantics changed"):
        evaluator.load_schedules()


def test_future_loader_checks_prior_gate_before_source_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "verify_evaluator_freeze",
        lambda: {"manifest_hash": "freeze"},
    )

    def blocked_prior(*args: object, **kwargs: object) -> list[dict[str, Any]]:
        raise ValueError("train did not pass; test remains sealed")

    def forbidden_source_open(*args: object, **kwargs: object) -> dict[str, Any]:
        raise AssertionError("future execution source opened before prior gate")

    monkeypatch.setattr(evaluator, "_verified_prior_reports", blocked_prior)
    monkeypatch.setattr(
        evaluator, "_load_future_source_contract", forbidden_source_open
    )

    with pytest.raises(ValueError, match="remains sealed"):
        evaluator.load_execution_window("test")


def test_prior_report_cannot_claim_future_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "train.json"
    report = evaluator._seal(
        {
            "candidate": evaluator.POLICY_ID,
            "stage": "train",
            "stage_passed": True,
            "evaluator_freeze_manifest_hash": "freeze",
            "evaluator_source_sha256": evaluator._sha256(evaluator.EVALUATOR_SOURCE),
            "opened_windows": ["train", "test"],
            "sealed_windows": ["eval", "final"],
        }
    )
    path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setitem(evaluator.STAGE_OUTPUTS, "train", path)

    with pytest.raises(ValueError, match="unexpected window"):
        evaluator._verified_prior_reports("test", freeze_hash="freeze")


def test_flat_trade_costs_are_six_and_ten_bp_per_notional_side() -> None:
    market = _market()
    clock = _clock(market)

    base = _simulate(market, [clock], cost=0.0006)
    stress = _simulate(market, [clock], cost=0.0010)

    assert base["absolute_return_pct"] == pytest.approx(-0.06)
    assert stress["absolute_return_pct"] == pytest.approx(-0.10)
    assert base["strict_mdd_pct"] == pytest.approx(0.06)
    assert stress["strict_mdd_pct"] == pytest.approx(0.10)


@pytest.mark.parametrize(("side", "funding_rate"), [(1, 0.001), (-1, -0.001)])
def test_entry_and_exit_boundary_funding_debits_are_retained(
    side: int, funding_rate: float
) -> None:
    market = _market()
    clock = _clock(market, side=side)
    entry = evaluator._utc(clock["entry_time"])
    exit_time = evaluator._utc(clock["exit_time"])
    funding = _funding([(entry, funding_rate, 100.0), (exit_time, funding_rate, 100.0)])

    result = _simulate(market, [clock], funding=funding)
    trade = result["trade_details"][0]

    assert trade["visited_funding_events"] == 2
    assert trade["funding_events"] == 2
    assert trade["dropped_boundary_funding_credits"] == 0
    assert trade["funding_cash"] == pytest.approx(-0.001)


@pytest.mark.parametrize(("side", "funding_rate"), [(1, -0.001), (-1, 0.001)])
def test_entry_and_exit_boundary_funding_credits_are_dropped_but_visited(
    side: int, funding_rate: float
) -> None:
    market = _market()
    clock = _clock(market, side=side)
    entry = evaluator._utc(clock["entry_time"])
    exit_time = evaluator._utc(clock["exit_time"])
    funding = _funding([(entry, funding_rate, 100.0), (exit_time, funding_rate, 100.0)])

    result = _simulate(market, [clock], funding=funding)
    trade = result["trade_details"][0]

    assert trade["visited_funding_events"] == 2
    assert trade["funding_events"] == 0
    assert trade["dropped_boundary_funding_credits"] == 2
    assert trade["funding_cash"] == 0.0


def test_exact_offset_is_interior_but_after_exit_is_outside() -> None:
    market = _market()
    clock = _clock(market)
    entry = evaluator._utc(clock["entry_time"])
    exit_time = evaluator._utc(clock["exit_time"])
    funding = _funding(
        [
            (evaluator._utc(entry + pd.Timedelta(milliseconds=47)), -0.001, 100.0),
            (evaluator._utc(exit_time + pd.Timedelta(milliseconds=47)), 0.001, 100.0),
        ]
    )

    result = _simulate(market, [clock], funding=funding)
    trade = result["trade_details"][0]

    assert trade["visited_funding_events"] == 1
    assert trade["funding_events"] == 1
    assert trade["dropped_boundary_funding_credits"] == 0
    assert trade["funding_cash"] == pytest.approx(0.0005)


def test_dropped_boundary_credit_mark_still_hits_strict_mdd() -> None:
    market = _market()
    clock = _clock(market)
    entry = evaluator._utc(clock["entry_time"])

    result = _simulate(market, [clock], funding=_funding([(entry, -0.001, 50.0)]))

    assert result["trade_details"][0]["funding_cash"] == 0.0
    assert result["strict_mdd_pct"] == pytest.approx(25.0)


def test_strict_mdd_uses_favorable_then_adverse_held_bar_path() -> None:
    market = _market()
    market.loc[0, ["open", "high", "low", "close"]] = [100.0, 110.0, 90.0, 100.0]

    result = _simulate(market, [_clock(market)])

    assert result["absolute_return_pct"] == 0.0
    assert result["strict_mdd_pct"] == pytest.approx((1.0 - 0.95 / 1.05) * 100.0)


def test_full_calendar_cagr_counts_idle_time() -> None:
    market = _market()
    market.loc[96, ["open", "high", "low", "close"]] = [
        110.0,
        110.0,
        110.0,
        110.0,
    ]
    clock = _clock(market)
    start = evaluator._utc(clock["entry_time"])
    one_year = _simulate(
        market,
        [clock],
        start=start,
        end=evaluator._utc(start + pd.Timedelta(days=365.25)),
    )
    two_years = _simulate(
        market,
        [clock],
        start=start,
        end=evaluator._utc(start + pd.Timedelta(days=730.5)),
    )

    assert one_year["absolute_return_pct"] == pytest.approx(5.0)
    assert two_years["absolute_return_pct"] == pytest.approx(5.0)
    assert one_year["cagr_pct"] == pytest.approx(5.0)
    assert two_years["cagr_pct"] == pytest.approx((1.05**0.5 - 1.0) * 100.0)


def test_ratio_uses_exact_zero_mdd_rule() -> None:
    assert evaluator._ratio(0.1, 0.0) == float("inf")
    assert evaluator._ratio(0.0, 0.0) == 0.0
    assert evaluator._ratio(-0.1, 0.0) == float("-inf")
    assert evaluator._ratio(0.1, 1e-15) == pytest.approx(1e14)


def test_frozen_config_rejects_unregistered_mutation() -> None:
    market = _market()
    start = evaluator._utc(market.iloc[0]["date"])

    with pytest.raises(ValueError, match="configuration is frozen"):
        evaluator.simulate_strict(
            market,
            _funding(),
            pd.DataFrame([_clock(market)]),
            start=start,
            end=evaluator._utc(start + pd.Timedelta(days=1)),
            cost_rate_per_side=0.0006,
            cfg=replace(evaluator.EvaluationConfig(), leverage=1.0),
        )


def test_margin_gate_uses_every_frozen_mechanism_control() -> None:
    support = evaluator._verify_static_inputs()
    base = {
        "absolute_return_pct": 10.0,
        "cagr_to_strict_mdd": 3.5,
        "strict_mdd_pct": 5.0,
        "trades": 100,
        "mean_gross_underlying_bp": 30.0,
        "weekly_cluster_signflip": {"p_value_two_sided": 0.05},
    }
    stress = {"absolute_return_pct": 5.0, "cagr_to_strict_mdd": 3.0}
    halves = {
        "left": {"absolute_return_pct": 1.0},
        "right": {"absolute_return_pct": 1.0},
    }
    controls = {
        name: {"cagr_to_strict_mdd": 0.0} for name in evaluator.MECHANISM_CONTROLS
    }
    controls["direction_flip"] = {"cagr_to_strict_mdd": 4.0}

    checks, ratios, margin = evaluator._stage_gates(
        "train", base, stress, halves, controls, support
    )

    assert set(ratios) == set(evaluator.MECHANISM_CONTROLS)
    assert margin == pytest.approx(-0.5)
    assert checks["mechanism_control_margin_at_least_0_25"] is False
