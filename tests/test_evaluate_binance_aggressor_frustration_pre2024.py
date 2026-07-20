from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_binance_aggressor_frustration_pre2024 as evaluate


def _market(
    opens: list[float],
    *,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    values = np.asarray(opens, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=len(values), freq="5min"),
            "open": values,
            "high": values if highs is None else highs,
            "low": values if lows is None else lows,
            "close": values,
        }
    )


def _schedule(
    market: pd.DataFrame,
    *,
    entry_position: int,
    exit_position: int,
    side: int = 1,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "side": [side],
            "entry_date": [market["date"].iloc[entry_position]],
            "exit_date": [market["date"].iloc[exit_position]],
            "entry_position": [entry_position],
            "exit_position": [exit_position],
        }
    )


def _funding(
    market: pd.DataFrame,
    positions: list[int],
    rates: list[float],
    *,
    mark_prices: list[float] | None = None,
) -> pd.DataFrame:
    times = market["date"].iloc[positions].reset_index(drop=True)
    return pd.DataFrame(
        {
            "funding_time_ms": (
                times.astype("int64") // 1_000_000
            ).astype(np.int64),
            "funding_time": times,
            "funding_rate": np.asarray(rates, dtype=float),
            "settlement_mark_price": (
                market["open"].iloc[positions].to_numpy(float)
                if mark_prices is None
                else np.asarray(mark_prices, dtype=float)
            ),
        }
    )


def _run(
    market: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    funding: pd.DataFrame | None = None,
    cost: float = 0.0,
) -> dict[str, object]:
    return evaluate.simulate_schedule(
        market,
        _funding(market, [], []) if funding is None else funding,
        schedule,
        start="2020-01-01",
        end="2021-01-01",
        cost_notional_per_side=cost,
        cfg=evaluate.EvaluationConfig(),
        compute_cluster=False,
    )


def test_train_market_parser_skips_poison_and_stops_before_2023(tmp_path: Path) -> None:
    path = tmp_path / "market.csv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("date,open,high,low,close\n")
        handle.write("2019-12-31 23:55:00,POISON,POISON,POISON,POISON\n")
        handle.write("2020-01-01 00:00:00,100,101,99,100\n")
        handle.write("2022-12-31 23:55:00,110,111,109,110\n")
        handle.write("2023-01-01 00:00:00,POISON,POISON,POISON,POISON\n")
    frame = evaluate._parse_market_window(
        path, start="2020-01-01", end="2023-01-01"
    )
    assert frame["open"].tolist() == [100.0, 110.0]


def test_selection_market_parser_accepts_only_audited_eof(tmp_path: Path) -> None:
    path = tmp_path / "market.csv.gz"
    terminal = "2023-12-31 23:55:00"
    with gzip.open(path, "wt") as handle:
        handle.write("date,open,high,low,close\n")
        handle.write("2022-12-31 23:55:00,POISON,POISON,POISON,POISON\n")
        handle.write(f"{terminal},100,101,99,100\n")
    with pytest.raises(ValueError, match="physical or audited boundary"):
        evaluate._parse_market_window(
            path, start="2023-01-01", end="2024-01-01"
        )
    frame = evaluate._parse_market_window(
        path,
        start="2023-01-01",
        end="2024-01-01",
        audited_eof_last_timestamp=terminal,
    )
    assert frame["open"].tolist() == [100.0]


def test_train_funding_parser_skips_poison_and_stops_before_2023(
    tmp_path: Path,
) -> None:
    path = tmp_path / "funding.csv.gz"
    header = (
        "funding_time_ms,funding_time_utc,symbol,funding_rate,"
        "settlement_mark_price,funding_time_offset_ms,mark_source\n"
    )
    with gzip.open(path, "wt") as handle:
        handle.write(header)
        handle.write("1577836799000,POISON,POISON,POISON,POISON,POISON,POISON\n")
        handle.write(
            "1577836800000,2020-01-01 00:00:00+00:00,BTCUSDT,0.001,7000,0,"
            "binance_8h_mark_price_kline_open\n"
        )
        handle.write("1672531200000,POISON,POISON,POISON,POISON,POISON,POISON\n")
    frame = evaluate._parse_funding_window(
        path, start="2020-01-01", end="2023-01-01"
    )
    assert frame["funding_rate"].tolist() == ["0.001"]


def test_direct_control_formulas_are_exact() -> None:
    frame = pd.DataFrame(
        {
            "quote_notional": [100.0],
            "signed_quote_notional": [20.0],
            "tick_notional_imbalance": [-0.3],
            "strict_sell_frustrated_notional": [11.0],
            "strict_buy_frustrated_notional": [1.0],
            "carried_sell_frustrated_notional": [2.0],
            "carried_buy_frustrated_notional": [7.0],
        }
    )
    scores = evaluate.direct_feature_control_scores(frame)
    assert scores["aggressor_flow_only"].iloc[0] == pytest.approx(-0.2)
    assert scores["tick_direction_only"].iloc[0] == pytest.approx(-0.3)
    assert scores["strict_nonzero_tick_only"].iloc[0] == pytest.approx(0.1)
    assert scores["carried_zero_tick_only"].iloc[0] == pytest.approx(-0.05)


def test_completed_bar_rejection_requires_opposing_flow_and_return() -> None:
    frame = pd.DataFrame(
        {
            "quote_notional": [100.0, 100.0, 100.0],
            "signed_quote_notional": [20.0, 20.0, -30.0],
            "micro_log_return": [-0.01, 0.01, 0.01],
        }
    )
    assert evaluate.completed_bar_rejection_score(frame).tolist() == pytest.approx(
        [-0.2, 0.0, 0.3]
    )


def test_frozen_control_clocks_are_source_only_and_reproducible() -> None:
    clocks, metadata = evaluate.build_control_clocks()
    assert {name: len(clock) for name, clock in clocks.items()} == {
        "primary": 11248,
        "direction_flip": 11248,
        "aggressor_flow_only": 11281,
        "tick_direction_only": 11317,
        "strict_nonzero_tick_only": 11259,
        "carried_zero_tick_only": 11342,
        "completed_bar_rejection": 12388,
        "stale_1h": 11248,
        "stale_24h": 11240,
    }
    assert metadata["official_market_columns_parsed"] == ["date"]
    assert metadata["official_market_value_rows_parsed"] == 0
    assert metadata["funding_value_rows_parsed"] == 0
    assert metadata["post_entry_outcome_rows_loaded"] == 0
    assert metadata["strategy_outcomes_calculated"] is False
    assert clocks["direction_flip"]["side"].tolist() == (
        -clocks["primary"]["side"]
    ).tolist()
    for clock in clocks.values():
        assert (clock["entry_date"] - clock["signal_date"]).eq(evaluate.BAR).all()
        assert (clock["exit_date"] - clock["entry_date"]).eq(
            evaluate.BAR * 24
        ).all()
        if len(clock) > 1:
            assert clock["entry_date"].iloc[1:].reset_index(drop=True).ge(
                clock["exit_date"].iloc[:-1].reset_index(drop=True)
            ).all()


def test_outcome_boundary_scan_reads_only_first_timestamp_columns() -> None:
    boundaries = evaluate.scan_outcome_boundaries()
    assert boundaries["market"]["value_rows_parsed"] == 0
    assert boundaries["funding"]["value_rows_parsed"] == 0
    assert boundaries["market"]["window_value_row_counts"] == {
        "train": 315648,
        "selection": 105120,
    }
    assert boundaries["funding"]["window_value_row_counts"] == {
        "train": 3288,
        "selection": 1095,
    }


def test_score_clock_is_prior_only_nonoverlapping_and_allows_exit_reentry() -> None:
    dates = pd.Series(pd.date_range("2020-01-01", periods=16, freq="5min"))
    score = pd.Series(np.ones(len(dates)))
    clean = pd.Series(np.ones(len(dates), dtype=bool))
    cfg = replace(
        evaluate.EvaluationConfig(),
        baseline_clean_observations=2,
        baseline_minimum_observations=2,
        hold_bars=2,
    )
    clock = evaluate._build_score_clock(
        dates, score, clean, name="synthetic", cfg=cfg
    )
    assert len(clock) > 2
    assert clock["entry_date"].iloc[1] == clock["exit_date"].iloc[0]
    mutated = score.copy()
    mutated.iloc[-1] = 1_000_000.0
    changed = evaluate._build_score_clock(
        dates, mutated, clean, name="synthetic", cfg=cfg
    )
    cutoff = dates.iloc[-1]
    pd.testing.assert_frame_equal(
        clock.loc[clock["signal_date"].lt(cutoff)].reset_index(drop=True),
        changed.loc[changed["signal_date"].lt(cutoff)].reset_index(drop=True),
    )


def test_future_score_nan_cannot_cancel_an_already_observable_signal() -> None:
    dates = pd.Series(pd.date_range("2020-01-01", periods=12, freq="5min"))
    clean = pd.Series(np.ones(len(dates), dtype=bool))
    cfg = replace(
        evaluate.EvaluationConfig(),
        baseline_clean_observations=2,
        baseline_minimum_observations=2,
        hold_bars=2,
    )
    original = evaluate._build_score_clock(
        dates, pd.Series(np.ones(len(dates))), clean, name="synthetic", cfg=cfg
    )
    changed_score = pd.Series(np.ones(len(dates)))
    changed_score.iloc[4] = np.nan
    changed = evaluate._build_score_clock(
        dates, changed_score, clean, name="synthetic", cfg=cfg
    )
    pd.testing.assert_series_equal(original.iloc[0], changed.iloc[0])


def test_stale_clocks_shift_exactly_12_and_288_bars() -> None:
    primary = pd.DataFrame(
        {
            "control": ["primary"],
            "side": [1],
            "signal_date": [pd.Timestamp("2020-01-01")],
            "entry_date": [pd.Timestamp("2020-01-01 00:05")],
            "exit_date": [pd.Timestamp("2020-01-01 02:05")],
        }
    )
    one_hour = evaluate._shift_clock(primary, name="stale_1h", bars=12)
    one_day = evaluate._shift_clock(primary, name="stale_24h", bars=288)
    assert one_hour["entry_date"].iloc[0] - primary["entry_date"].iloc[0] == pd.Timedelta(
        hours=1
    )
    assert one_day["entry_date"].iloc[0] - primary["entry_date"].iloc[0] == pd.Timedelta(
        days=1
    )


def test_clock_normalization_does_not_sort_malformed_input() -> None:
    raw = pd.DataFrame(
        {
            "signal_date": ["2020-01-02", "2020-01-01"],
            "entry_date": ["2020-01-02 00:05", "2020-01-01 00:05"],
            "exit_date": ["2020-01-02 02:05", "2020-01-01 02:05"],
            "side": [1, -1],
        }
    )
    clock = evaluate._normalize_clock(raw, name="synthetic")
    with pytest.raises(ValueError, match="not sorted"):
        evaluate._validate_clock(
            clock, name="synthetic", cfg=evaluate.EvaluationConfig()
        )


def test_strict_mdd_marks_favorable_before_adverse_for_long() -> None:
    opens = [100.0] * 25
    opens[-1] = 110.0
    highs = [100.0] * 25
    lows = [100.0] * 25
    highs[0] = 120.0
    lows[0] = 90.0
    market = _market(opens, highs=highs, lows=lows)
    result = _run(market, _schedule(market, entry_position=0, exit_position=24))
    assert result["absolute_return_pct"] == pytest.approx(5.0)
    assert result["strict_mdd_pct"] == pytest.approx((1.0 - 0.95 / 1.10) * 100.0)


def test_strict_mdd_marks_favorable_before_adverse_for_short() -> None:
    opens = [100.0] * 25
    opens[-1] = 90.0
    highs = [100.0] * 25
    lows = [100.0] * 25
    highs[0] = 110.0
    lows[0] = 80.0
    market = _market(opens, highs=highs, lows=lows)
    result = _run(
        market, _schedule(market, entry_position=0, exit_position=24, side=-1)
    )
    assert result["absolute_return_pct"] == pytest.approx(5.0)
    assert result["strict_mdd_pct"] == pytest.approx((1.0 - 0.95 / 1.10) * 100.0)


def test_costs_scale_with_half_leverage_and_exit_notional() -> None:
    market = _market([100.0] * 25)
    result = _run(
        market,
        _schedule(market, entry_position=0, exit_position=24),
        cost=0.0006,
    )
    assert result["absolute_return_pct"] == pytest.approx(-0.06)
    assert result["strict_mdd_pct"] == pytest.approx(0.06)


def test_funding_uses_fixed_quantity_and_settlement_mark() -> None:
    market = _market([100.0] * 25)
    funding = _funding(market, [12], [0.01], mark_prices=[200.0])
    result = _run(
        market,
        _schedule(market, entry_position=0, exit_position=24),
        funding=funding,
    )
    assert result["absolute_return_pct"] == pytest.approx(-1.0)
    assert result["total_funding_cash_pct_of_entry_equity_sum"] == pytest.approx(-1.0)


def test_boundary_funding_debits_apply_and_credits_drop() -> None:
    market = _market([100.0] * 25)
    schedule = _schedule(market, entry_position=0, exit_position=24)
    debits = _run(
        market,
        schedule,
        funding=_funding(market, [0, 24], [0.01, 0.01]),
    )
    assert debits["absolute_return_pct"] == pytest.approx(-1.0)
    assert debits["applied_funding_settlement_count"] == 2
    credits = _run(
        market,
        schedule,
        funding=_funding(market, [0, 24], [-0.01, -0.01]),
    )
    assert credits["absolute_return_pct"] == pytest.approx(0.0)
    assert credits["dropped_boundary_funding_credits"] == 2
    assert credits["applied_funding_settlement_count"] == 0


def test_interior_funding_credit_carries_into_adverse_strict_path() -> None:
    opens = [100.0] * 25
    highs = [100.0] * 25
    lows = [100.0] * 25
    lows[0] = 90.0
    market = _market(opens, highs=highs, lows=lows)
    result = _run(
        market,
        _schedule(market, entry_position=0, exit_position=24),
        funding=_funding(market, [12], [-0.01]),
    )
    assert result["absolute_return_pct"] == pytest.approx(0.5)
    assert result["strict_mdd_pct"] == pytest.approx(
        (1.0 - 0.955 / 1.005) * 100.0
    )


def test_global_high_water_mark_carries_across_trades() -> None:
    opens = [100.0] * 49
    opens[24] = 120.0
    opens[48] = 120.0
    highs = opens.copy()
    lows = opens.copy()
    lows[24] = 60.0
    market = _market(opens, highs=highs, lows=lows)
    schedule = pd.concat(
        [
            _schedule(market, entry_position=0, exit_position=24),
            _schedule(market, entry_position=24, exit_position=48),
        ],
        ignore_index=True,
    )
    result = _run(market, schedule)
    assert result["strict_mdd_pct"] == pytest.approx(25.0)


def test_cagr_uses_full_declared_calendar() -> None:
    opens = [100.0] * 25
    opens[-1] = 110.0
    market = _market(opens)
    result = _run(market, _schedule(market, entry_position=0, exit_position=24))
    expected = (1.05 ** (365.25 / 366.0) - 1.0) * 100.0
    assert result["wall_clock_years"] == pytest.approx(366.0 / 365.25)
    assert result["cagr_pct"] == pytest.approx(expected)


def test_split_crossing_trade_is_excluded() -> None:
    schedule = pd.DataFrame(
        {
            "entry_date": pd.to_datetime(
                ["2020-12-31 23:00", "2021-01-02 00:00"]
            ),
            "exit_date": pd.to_datetime(["2021-01-01 01:00", "2021-01-02 02:00"]),
        }
    )
    sliced = evaluate._slice_schedule(
        schedule, start="2020-01-01", end="2021-01-01"
    )
    assert sliced.empty


def test_weekly_sign_flip_is_one_sided_and_exact() -> None:
    result = evaluate.weekly_cluster_sign_flip(
        [0.01, 0.02, 0.03, -0.01],
        pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15", "2020-01-22"]),
        permutations=100,
        seed=7,
    )
    assert result["cluster_count"] == 4
    assert result["method"] == "exact"
    assert result["p_value_one_sided"] == pytest.approx(0.1875)


def _passing_policy(*, stage: str = "train", ratio: float = 3.0) -> dict[str, object]:
    cfg = evaluate.EvaluationConfig()
    if stage == "train":
        count, per_side = cfg.minimum_train_trades, cfg.minimum_train_trades_each_side
        split_count = cfg.minimum_train_split_trades
        splits = ("train_2020", "train_2021", "train_2022")
        clusters = cfg.minimum_train_weekly_clusters
    else:
        count = cfg.minimum_selection_trades
        per_side = cfg.minimum_selection_trades_each_side
        split_count = cfg.minimum_selection_split_trades
        splits = ("selection_2023_h1", "selection_2023_h2")
        clusters = cfg.minimum_selection_weekly_clusters
    base = {
        "absolute_return_pct": 1.0,
        "cagr_to_strict_mdd": ratio,
        "strict_mdd_pct": 15.0,
        "zero_mdd_ratio_cap_applied": False,
        "trade_count": count,
        "long_count": per_side,
        "short_count": per_side,
        "mean_gross_underlying_move_bp": 24.0,
        "weekly_cluster_sign_flip": {
            "cluster_count": clusters,
            "p_value_one_sided": 0.10,
        },
    }
    stress = {
        "absolute_return_pct": 1.0,
        "cagr_to_strict_mdd": 2.5,
        "zero_mdd_ratio_cap_applied": False,
        "trade_count": count,
    }
    return {
        "base_6bp": base,
        "stress_10bp": stress,
        "splits_base_6bp": {
            name: {"absolute_return_pct": 0.1, "trade_count": split_count}
            for name in splits
        },
        "side_contributions_base_6bp": {
            "long_only": {"absolute_return_pct": 0.1},
            "short_only": {"absolute_return_pct": 0.1},
        },
    }


def test_gate_boundaries_are_inclusive() -> None:
    assert evaluate.performance_gate_failures(
        _passing_policy(), stage="train", cfg=evaluate.EvaluationConfig()
    ) == []


def test_qualification_rejects_independently_passing_mechanism_control() -> None:
    primary = _passing_policy(ratio=4.0)
    policies = {
        name: _passing_policy(ratio=3.0) for name in evaluate.POLICY_NAMES
    }
    policies["primary"] = primary
    verdict = evaluate.qualification(
        policies, stage="train", cfg=evaluate.EvaluationConfig()
    )
    assert verdict["qualifies"] is False
    assert set(verdict["passing_mechanism_controls"]) == set(
        evaluate.MECHANISM_REJECTION_CONTROLS
    )


def test_qualification_accepts_exact_control_margin_when_controls_fail() -> None:
    policies = {
        name: _passing_policy(ratio=2.75) for name in evaluate.POLICY_NAMES
    }
    policies["primary"] = _passing_policy(ratio=3.0)
    for name in evaluate.MECHANISM_REJECTION_CONTROLS:
        policies[name] = deepcopy(policies[name])
        policies[name]["base_6bp"]["absolute_return_pct"] = -1.0
    verdict = evaluate.qualification(
        policies, stage="train", cfg=evaluate.EvaluationConfig()
    )
    assert verdict["minimum_primary_control_ratio_margin"] == pytest.approx(0.25)
    assert verdict["qualifies"] is True


def test_selection_refuses_before_loading_outcomes_when_train_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluate, "TRAIN_OUTPUT", tmp_path / "missing-train.json")
    monkeypatch.setattr(evaluate, "SELECTION_OUTPUT", tmp_path / "selection.json")
    monkeypatch.setattr(evaluate, "verify_evaluation_freeze", lambda cfg: {})
    monkeypatch.setattr(evaluate, "build_control_clocks", lambda cfg: ({}, {}))

    def forbidden(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("selection outcome loader was reached")

    monkeypatch.setattr(evaluate, "_compute_stage_report", forbidden)
    with pytest.raises(PermissionError, match="train artifact is missing"):
        evaluate.evaluate_selection()
