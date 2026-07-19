from __future__ import annotations

import gzip
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from training import evaluate_stablecoin_quote_flow_diffusion as evaluator


def _market(
    *,
    start: str = "2023-07-01T00:00:00Z",
    periods: int = 73,
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
    entry = evaluator._timestamp(market.iloc[entry_position]["date"])
    exit_time = evaluator._timestamp(
        market.iloc[entry_position + evaluator.EvaluationConfig().hold_bars]["date"]
    )
    return {
        "candidate": evaluator.POLICY_ID,
        "control": "primary",
        "split": "synthetic",
        "source_hour_start": entry - pd.Timedelta(hours=1, minutes=5),
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
    start = start or evaluator._timestamp(market.iloc[0]["date"])
    end = end or evaluator._timestamp(start + pd.Timedelta(days=365.25))
    return evaluator.simulate_strict(
        market,
        _funding() if funding is None else funding,
        pd.DataFrame(clocks),
        start=start,
        end=end,
        cost_rate_per_side=cost,
    )


def test_frozen_schedules_are_causal_nonoverlapping_and_stage_contained() -> None:
    schedules = evaluator.load_schedules()

    assert set(schedules) == set(evaluator.ALL_CONTROLS)
    assert len(schedules["primary"]) == 550
    for name, schedule in schedules.items():
        assert bool(
            schedule["feature_available_time"].eq(schedule["decision_time"]).all()
        )
        expected_delay = pd.Timedelta(minutes=65 if name == "extra_latency_1h" else 5)
        assert bool(
            schedule["entry_time"]
            .sub(schedule["decision_time"])
            .eq(expected_delay)
            .all()
        )
        assert bool(
            schedule["exit_time"]
            .sub(schedule["entry_time"])
            .eq(pd.Timedelta(hours=6))
            .all()
        )
        assert bool(
            schedule["entry_time"]
            .iloc[1:]
            .ge(schedule["exit_time"].iloc[:-1].to_numpy())
            .all()
        )
        for stage, (start, end) in evaluator.STAGE_WINDOWS.items():
            window = schedule.loc[schedule["split"].eq(stage)]
            assert window["source_hour_start"].ge(start).all()
            assert window["entry_time"].ge(start).all()
            assert window["exit_time"].le(end).all()
            assert not window["exit_time"].eq(end).any()
    random_control = schedules["deterministic_random_side"]
    expected = random_control["decision_time"].map(evaluator._deterministic_random_side)
    assert bool(random_control["side"].eq(expected).all())


def test_deterministic_random_side_uses_frozen_ascii_sha_contract() -> None:
    decision = evaluator._timestamp("2024-01-02T03:00:00Z")

    assert evaluator._deterministic_random_side(decision) == -1


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
    assert evaluator.verify_evaluator_freeze(output) == report


def test_verify_freeze_rejects_changed_stage_seal(tmp_path: Path) -> None:
    output = tmp_path / "freeze.json"
    report = evaluator.freeze_evaluator(output)
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    core["sealed_windows"] = list(evaluator.STAGE_ORDER[:-1])
    output.write_text(json.dumps(evaluator._seal(core)), encoding="utf-8")

    with pytest.raises(ValueError, match="stage seal"):
        evaluator.verify_evaluator_freeze(output)


def test_freeze_artifact_is_write_once(tmp_path: Path) -> None:
    output = tmp_path / "freeze.json"
    evaluator.freeze_evaluator(output)

    with pytest.raises(FileExistsError):
        evaluator.freeze_evaluator(output)


def _write_market(path: Path, *, include_boundary: bool) -> None:
    periods = 5 if include_boundary else 4
    rows = []
    for timestamp in pd.date_range("2023-07-01", periods=periods, freq="5min"):
        price = 200.0 if timestamp == pd.Timestamp("2023-07-01 00:20") else 100.0
        rows.append([timestamp, price, price, price, price])
    with gzip.open(path, "wt", newline="") as handle:
        pd.DataFrame(
            rows, columns=pd.Index(["date", "open", "high", "low", "close"])
        ).to_csv(handle, index=False)


def _write_funding(path: Path, *, include_boundary: bool) -> None:
    rows = [["2023-07-01T00:00:00.047Z", "BTCUSDT", 0.0001, 100.0]]
    if include_boundary:
        rows.append(["2023-07-01T08:00:00Z", "BTCUSDT", 0.0001, 101.0])
    with gzip.open(path, "wt", newline="") as handle:
        pd.DataFrame(
            rows,
            columns=pd.Index(
                [
                    "funding_time_utc",
                    "symbol",
                    "funding_rate",
                    "settlement_mark_price",
                ]
            ),
        ).to_csv(handle, index=False)


def test_market_parser_accepts_only_the_exact_half_open_grid(tmp_path: Path) -> None:
    path = tmp_path / "market.csv.gz"
    _write_market(path, include_boundary=False)
    start = evaluator._timestamp("2023-07-01T00:00:00Z")
    end = evaluator._timestamp("2023-07-01T00:20:00Z")

    market, diagnostics = evaluator._parse_market_window(path, start, end)

    assert len(market) == 4
    assert diagnostics["rows"] == 4
    assert market["open"].max() == 100.0


def test_market_parser_rejects_an_unneeded_end_boundary_row(tmp_path: Path) -> None:
    path = tmp_path / "market.csv.gz"
    _write_market(path, include_boundary=True)
    start = evaluator._timestamp("2023-07-01T00:00:00Z")
    end = evaluator._timestamp("2023-07-01T00:20:00Z")

    with pytest.raises(ValueError, match="unneeded end boundary"):
        evaluator._parse_market_window(path, start, end)


def test_market_parser_allows_a_required_exact_exit_boundary_open(
    tmp_path: Path,
) -> None:
    path = tmp_path / "market.csv.gz"
    _write_market(path, include_boundary=True)
    start = evaluator._timestamp("2023-07-01T00:00:00Z")
    end = evaluator._timestamp("2023-07-01T00:20:00Z")

    market, diagnostics = evaluator._parse_market_window(
        path, start, end, include_end_boundary=True
    )

    assert len(market) == 5
    assert diagnostics["last_timestamp"] == end.isoformat()


def test_funding_parser_rejects_an_unneeded_end_boundary_event(tmp_path: Path) -> None:
    path = tmp_path / "funding.csv.gz"
    _write_funding(path, include_boundary=True)
    start = evaluator._timestamp("2023-07-01T00:00:00Z")
    end = evaluator._timestamp("2023-07-01T08:00:00Z")

    with pytest.raises(ValueError, match="unneeded end boundary"):
        evaluator._parse_funding_window(path, start, end)


def test_funding_parser_allows_a_required_exact_end_boundary_event(
    tmp_path: Path,
) -> None:
    path = tmp_path / "funding.csv.gz"
    _write_funding(path, include_boundary=True)
    start = evaluator._timestamp("2023-07-01T00:00:00Z")
    end = evaluator._timestamp("2023-07-01T08:00:00Z")

    funding, diagnostics = evaluator._parse_funding_window(
        path, start, end, include_end_boundary=True
    )

    assert len(funding) == 2
    assert diagnostics["exit_boundary_events"] == 1
    assert diagnostics["maximum_absolute_grid_offset_ms"] == pytest.approx(47.0)


def test_flat_trade_costs_are_six_and_ten_bp_per_notional_side() -> None:
    market = _market()
    clock = _clock(market)

    base = _simulate(market, [clock], cost=0.0006)
    stress = _simulate(market, [clock], cost=0.0010)

    assert base["absolute_return_pct"] == pytest.approx(-0.06)
    assert stress["absolute_return_pct"] == pytest.approx(-0.10)
    assert base["strict_mdd_pct"] == pytest.approx(0.06)
    assert stress["strict_mdd_pct"] == pytest.approx(0.10)
    assert base["trade_details"][0]["net_return"] == pytest.approx(-0.0006)
    assert stress["trade_details"][0]["net_return"] == pytest.approx(-0.0010)


@pytest.mark.parametrize(
    ("side", "funding_rate"),
    [(1, 0.001), (-1, -0.001)],
)
def test_entry_and_exit_boundary_funding_debits_are_retained(
    side: int, funding_rate: float
) -> None:
    market = _market()
    clock = _clock(market, side=side)
    entry = evaluator._timestamp(clock["entry_time"])
    exit_time = evaluator._timestamp(clock["exit_time"])
    funding = _funding([(entry, funding_rate, 100.0), (exit_time, funding_rate, 100.0)])

    result = _simulate(market, [clock], funding=funding)
    trade = result["trade_details"][0]

    assert trade["visited_funding_events"] == 2
    assert trade["funding_events"] == 2
    assert trade["dropped_boundary_funding_credits"] == 0
    assert trade["funding_cash"] == pytest.approx(-0.001)
    assert result["absolute_return_pct"] == pytest.approx(-0.1)


@pytest.mark.parametrize(
    ("side", "funding_rate"),
    [(1, -0.001), (-1, 0.001)],
)
def test_entry_and_exit_boundary_funding_credits_are_dropped_but_visited(
    side: int, funding_rate: float
) -> None:
    market = _market()
    clock = _clock(market, side=side)
    entry = evaluator._timestamp(clock["entry_time"])
    exit_time = evaluator._timestamp(clock["exit_time"])
    funding = _funding([(entry, funding_rate, 100.0), (exit_time, funding_rate, 100.0)])

    result = _simulate(market, [clock], funding=funding)
    trade = result["trade_details"][0]

    assert trade["visited_funding_events"] == 2
    assert trade["funding_events"] == 0
    assert trade["dropped_boundary_funding_credits"] == 2
    assert trade["funding_cash"] == 0.0
    assert result["absolute_return_pct"] == 0.0


def test_dropped_boundary_credit_settlement_mark_still_hits_strict_mdd() -> None:
    market = _market()
    clock = _clock(market)
    entry = evaluator._timestamp(clock["entry_time"])
    funding = _funding([(entry, -0.001, 50.0)])

    result = _simulate(market, [clock], funding=funding)

    assert result["trade_details"][0]["funding_cash"] == 0.0
    assert result["trade_details"][0]["visited_funding_events"] == 1
    assert result["strict_mdd_pct"] == pytest.approx(25.0)


def test_interior_funding_keeps_credit_and_debit_symmetrically() -> None:
    market = _market()
    clock = _clock(market)
    entry = evaluator._timestamp(clock["entry_time"])
    funding = _funding(
        [
            (evaluator._timestamp(entry + pd.Timedelta(hours=1)), -0.001, 100.0),
            (evaluator._timestamp(entry + pd.Timedelta(hours=2)), 0.002, 100.0),
        ]
    )

    result = _simulate(market, [clock], funding=funding)
    trade = result["trade_details"][0]

    assert trade["visited_funding_events"] == 2
    assert trade["funding_events"] == 2
    assert trade["dropped_boundary_funding_credits"] == 0
    assert trade["funding_cash"] == pytest.approx(-0.0005)
    assert result["absolute_return_pct"] == pytest.approx(-0.05)


def test_exact_binance_funding_time_offset_is_an_interior_event() -> None:
    market = _market()
    clock = _clock(market)
    entry = evaluator._timestamp(clock["entry_time"])
    funding = _funding(
        [(evaluator._timestamp(entry + pd.Timedelta(milliseconds=47)), -0.001, 100.0)]
    )

    result = _simulate(market, [clock], funding=funding)
    trade = result["trade_details"][0]

    assert trade["visited_funding_events"] == 1
    assert trade["funding_events"] == 1
    assert trade["dropped_boundary_funding_credits"] == 0
    assert trade["funding_cash"] == pytest.approx(0.0005)
    assert result["absolute_return_pct"] == pytest.approx(0.05)


def test_strict_mdd_uses_favorable_then_adverse_held_bar_path() -> None:
    market = _market()
    market.loc[0, ["open", "high", "low", "close"]] = [100.0, 110.0, 90.0, 100.0]

    result = _simulate(market, [_clock(market)])

    assert result["absolute_return_pct"] == 0.0
    assert result["strict_mdd_pct"] == pytest.approx((1.0 - 0.95 / 1.05) * 100.0)


def test_simulator_rejects_clock_crossing_declared_window() -> None:
    market = _market()
    clock = _clock(market)
    start = evaluator._timestamp(clock["entry_time"] + pd.Timedelta(minutes=5))
    end = evaluator._timestamp(clock["exit_time"] + pd.Timedelta(days=1))

    with pytest.raises(ValueError, match="crosses the simulation window"):
        _simulate(market, [clock], start=start, end=end)


def test_full_calendar_cagr_counts_idle_time() -> None:
    market = _market()
    market.loc[72, ["open", "high", "low", "close"]] = [110.0, 110.0, 110.0, 110.0]
    clock = _clock(market)
    start = evaluator._timestamp(clock["entry_time"])
    one_year = _simulate(
        market,
        [clock],
        start=start,
        end=evaluator._timestamp(start + pd.Timedelta(days=365.25)),
    )
    two_years = _simulate(
        market,
        [clock],
        start=start,
        end=evaluator._timestamp(start + pd.Timedelta(days=730.5)),
    )

    assert one_year["absolute_return_pct"] == pytest.approx(5.0)
    assert two_years["absolute_return_pct"] == pytest.approx(5.0)
    assert one_year["cagr_pct"] == pytest.approx(5.0)
    assert two_years["cagr_pct"] == pytest.approx((1.05**0.5 - 1.0) * 100.0)


def test_exit_exactly_at_half_end_is_contained_for_exit_exclusive_position() -> None:
    market = _market(periods=86)
    clock = _clock(market, entry_position=13)
    start = evaluator._timestamp(market.iloc[0]["date"])
    end = evaluator._timestamp(clock["exit_time"])

    result = evaluator._simulate_window(
        market,
        _funding(),
        pd.DataFrame([clock]),
        start=start,
        end=end,
        cost=0.0,
        cfg=evaluator.EvaluationConfig(),
    )

    assert result["trades"] == 1
    assert result["trade_details"][0]["exit_time"] == end.isoformat()


def test_ratio_uses_exact_zero_mdd_rule_without_epsilon_floor() -> None:
    assert evaluator._ratio(0.1, 0.0) == float("inf")
    assert evaluator._ratio(0.0, 0.0) == 0.0
    assert evaluator._ratio(-0.1, 0.0) == float("-inf")
    assert evaluator._ratio(0.1, 1e-15) == pytest.approx(1e14)


def test_weekly_cluster_signflip_exact_matches_manual_enumeration() -> None:
    trades = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2023-01-02", "2023-01-09"], utc=True),
            "net_return": [0.01, 0.01],
        }
    )

    result = evaluator.weekly_cluster_signflip_two_sided(trades)

    assert result["method"] == "exact"
    assert result["cluster_count"] == 2
    assert result["draws"] == 4
    assert result["p_value_two_sided"] == pytest.approx(0.5)


def test_weekly_cluster_monte_carlo_contract_is_deterministic() -> None:
    trades = pd.DataFrame(
        {
            "entry_time": pd.date_range("2023-01-02", periods=21, freq="7D", tz="UTC"),
            "net_return": np.linspace(-0.01, 0.02, 21),
        }
    )

    first = evaluator.weekly_cluster_signflip_two_sided(trades)
    second = evaluator.weekly_cluster_signflip_two_sided(trades)

    assert first == second
    assert first["method"] == "monte_carlo"
    assert first["draws"] == 20_000


def test_frozen_config_rejects_unregistered_mutation() -> None:
    market = _market()
    with pytest.raises(ValueError, match="configuration is frozen"):
        evaluator.simulate_strict(
            market,
            _funding(),
            pd.DataFrame([_clock(market)]),
            start=evaluator._timestamp(market.iloc[0]["date"]),
            end=evaluator._timestamp(market.iloc[0]["date"] + pd.Timedelta(days=1)),
            cost_rate_per_side=0.0006,
            cfg=replace(evaluator.EvaluationConfig(), leverage=1.0),
        )


def test_margin_gate_includes_all_frozen_falsification_controls() -> None:
    prereg, _ = evaluator._verify_static_inputs()
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

    checks, ratios, minimum_margin = evaluator._stage_gates(
        "train", base, stress, halves, controls, prereg
    )

    assert set(ratios) == set(evaluator.MECHANISM_CONTROLS)
    assert minimum_margin == pytest.approx(-0.5)
    assert checks["mechanism_control_margin_at_least_0_25"] is False

    controls["direction_flip"] = {"cagr_to_strict_mdd": float("inf")}
    checks, _, minimum_margin = evaluator._stage_gates(
        "train",
        {**base, "cagr_to_strict_mdd": float("inf")},
        stress,
        halves,
        controls,
        prereg,
    )
    assert minimum_margin == float("-inf")
    assert checks["mechanism_control_margin_at_least_0_25"] is False


def test_prior_report_cannot_claim_a_future_window_opened(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "train.json"
    report = evaluator._seal(
        {
            "protocol_version": "stablecoin_quote_flow_diffusion_stage_v1",
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


def test_direct_future_loader_checks_prior_gate_before_source_open(
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


def test_future_source_manifest_cannot_change_frozen_boundary_requirement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = evaluator._future_source_spec("test")
    assert spec["exit_boundary_required"] is False
    path = tmp_path / "source.json"
    report = evaluator._seal(
        {
            "protocol_version": spec["required_protocol_version"],
            "candidate": evaluator.POLICY_ID,
            "stage": "test",
            "physical_window": spec["physical_window"],
            "strategy_outcomes_calculated": False,
            "official_checksums_verified": True,
            "physical_rows_limited_to_window": True,
            "exit_boundary_required": True,
            "market": {"path": "market.csv.gz", "sha256": "market"},
            "funding": {"path": "funding.csv.gz", "sha256": "funding"},
        }
    )
    path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setitem(evaluator.FUTURE_SOURCE_MANIFESTS, "test", path)

    with pytest.raises(ValueError, match="boundary contract changed"):
        evaluator._load_future_source_contract("test")
