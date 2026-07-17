from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

from training import evaluate_miner_cadence_recovery_pre2024 as evaluate


def _market(
    opens: list[float],
    *,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    values = np.asarray(opens, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=len(values), freq="5min"),
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
            "funding_time_ms": (times.astype("int64") // 1_000_000).astype(np.int64),
            "funding_time": times,
            "funding_rate": rates,
            "settlement_mark_price": (
                market["open"].iloc[positions].to_numpy(float)
                if mark_prices is None
                else mark_prices
            ),
        }
    )


def test_market_parser_physically_stops_before_2024_values(tmp_path) -> None:
    path = tmp_path / "market.csv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("date,open,high,low,close\n")
        handle.write("2023-12-31 23:55:00,100,101,99,100\n")
        handle.write("2024-01-01 00:00:00,NOT_PARSED,NOT_PARSED,NOT_PARSED,NOT_PARSED\n")
    frame = evaluate._parse_pre2024_market(path)
    assert len(frame) == 1
    assert frame.iloc[0]["open"] == 100.0


def test_market_parser_requires_the_2024_boundary(tmp_path) -> None:
    path = tmp_path / "market.csv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("date,open,high,low,close\n")
        handle.write("2023-12-31 23:55:00,100,101,99,100\n")
    with pytest.raises(ValueError, match="did not reach sealed 2024 boundary"):
        evaluate._parse_pre2024_market(path)


def test_control_clocks_preserve_primary_counts_and_nonoverlap() -> None:
    clocks, _, _ = evaluate.verify_support_and_control_clocks()
    assert {name: len(clock) for name, clock in clocks.items()} == {
        "primary": 65,
        "direction_flip": 65,
        "cadence_confirmation_removed": 96,
        "stale_hash_state_7d": 59,
        "random_clock": 65,
        "constant_long": 136,
        "one_bar_delayed_entry": 65,
    }
    primary_counts = clocks["primary"]["entry_date"].dt.year.value_counts().sort_index()
    random_counts = clocks["random_clock"]["entry_date"].dt.year.value_counts().sort_index()
    pd.testing.assert_series_equal(primary_counts, random_counts)
    assert clocks["direction_flip"]["side"].eq(-1).all()
    for clock in clocks.values():
        assert clock["exit_date"].max() < pd.Timestamp("2024-01-01")
        assert (clock["exit_date"] - clock["entry_date"] == pd.Timedelta(days=7)).all()


def test_monthly_sign_flip_is_exact_and_excludes_empty_months() -> None:
    result = evaluate.monthly_cluster_sign_flip(
        [0.01, 0.02, 0.03, -0.01],
        pd.to_datetime(["2023-01-01", "2023-01-10", "2023-02-01", "2023-03-01"]),
        permutations=100,
        seed=7,
    )
    assert result["cluster_count"] == 3
    assert result["method"] == "exact"
    assert result["p_value_one_sided"] == pytest.approx(0.25)


def test_strict_mdd_marks_favorable_before_adverse_for_long() -> None:
    market = _market(
        [100.0, 100.0, 110.0],
        highs=[120.0, 100.0, 110.0],
        lows=[90.0, 100.0, 110.0],
    )
    result = evaluate.simulate_schedule(
        market,
        _funding(market, [], []),
        _schedule(market, entry_position=0, exit_position=2),
        start="2023-01-01",
        end="2024-01-01",
        cost_notional_per_side=0.0,
        cfg=evaluate.EvaluationConfig(),
        compute_cluster=False,
    )
    assert result["absolute_return_pct"] == pytest.approx(5.0)
    assert result["strict_mdd_pct"] == pytest.approx((1.0 - 0.95 / 1.10) * 100.0)
    expected_cagr = (1.05 ** (365.25 / 365.0) - 1.0) * 100.0
    assert result["cagr_pct"] == pytest.approx(expected_cagr)


def test_strict_mdd_marks_favorable_before_adverse_for_short() -> None:
    market = _market(
        [100.0, 100.0, 90.0],
        highs=[110.0, 100.0, 90.0],
        lows=[80.0, 100.0, 90.0],
    )
    result = evaluate.simulate_schedule(
        market,
        _funding(market, [], []),
        _schedule(market, entry_position=0, exit_position=2, side=-1),
        start="2023-01-01",
        end="2024-01-01",
        cost_notional_per_side=0.0,
        cfg=evaluate.EvaluationConfig(),
        compute_cluster=False,
    )
    assert result["absolute_return_pct"] == pytest.approx(5.0)
    assert result["strict_mdd_pct"] == pytest.approx((1.0 - 0.95 / 1.10) * 100.0)


def test_global_hwm_carries_across_trades() -> None:
    market = _market(
        [100.0, 100.0, 120.0, 100.0, 100.0, 100.0],
        highs=[100.0, 100.0, 120.0, 100.0, 100.0, 100.0],
        lows=[100.0, 100.0, 120.0, 90.0, 100.0, 100.0],
    )
    first = _schedule(market, entry_position=0, exit_position=2)
    second = _schedule(market, entry_position=3, exit_position=5)
    schedule = pd.concat([first, second], ignore_index=True)
    result = evaluate.simulate_schedule(
        market,
        _funding(market, [], []),
        schedule,
        start="2023-01-01",
        end="2024-01-01",
        cost_notional_per_side=0.0,
        cfg=evaluate.EvaluationConfig(),
        compute_cluster=False,
    )
    assert result["strict_mdd_pct"] == pytest.approx(5.0)


def test_entry_exit_and_hypothetical_liquidation_costs_are_charged() -> None:
    market = _market([100.0, 100.0, 100.0])
    result = evaluate.simulate_schedule(
        market,
        _funding(market, [], []),
        _schedule(market, entry_position=0, exit_position=2),
        start="2023-01-01",
        end="2024-01-01",
        cost_notional_per_side=0.001,
        cfg=evaluate.EvaluationConfig(),
        compute_cluster=False,
    )
    assert result["absolute_return_pct"] == pytest.approx(-0.1)
    assert result["strict_mdd_pct"] == pytest.approx(0.1)


def test_funding_is_entry_inclusive_exit_exclusive_and_mark_scaled() -> None:
    market = _market([100.0, 110.0, 120.0, 130.0])
    funding = _funding(
        market,
        [0, 2, 3],
        [0.02, -0.01, 0.50],
        mark_prices=[100.0, 140.0, 130.0],
    )
    result = evaluate.simulate_schedule(
        market,
        funding,
        _schedule(market, entry_position=0, exit_position=3),
        start="2023-01-01",
        end="2024-01-01",
        cost_notional_per_side=0.0,
        cfg=evaluate.EvaluationConfig(),
        compute_cluster=False,
    )
    # -0.5*0.02*(100/100) + 0.5*0.01*(140/100) = -0.003.
    assert result["funding_settlement_count"] == 2
    assert result["absolute_return_pct"] == pytest.approx(14.7)
    assert result["total_funding_return_pct_of_entry_equity_sum"] == pytest.approx(-0.3)


def test_funding_credit_and_debit_form_conservative_mdd_envelope() -> None:
    market = _market([100.0, 100.0, 100.0])
    funding = _funding(market, [0, 1], [-0.02, 0.02])
    result = evaluate.simulate_schedule(
        market,
        funding,
        _schedule(market, entry_position=0, exit_position=2),
        start="2023-01-01",
        end="2024-01-01",
        cost_notional_per_side=0.0,
        cfg=evaluate.EvaluationConfig(),
        compute_cluster=False,
    )
    assert result["absolute_return_pct"] == pytest.approx(0.0)
    assert result["strict_mdd_pct"] == pytest.approx((1.0 - 0.99 / 1.01) * 100.0)


def test_split_slicing_excludes_a_trade_that_exits_on_boundary() -> None:
    schedule = pd.DataFrame(
        {
            "entry_date": pd.to_datetime(["2023-06-29", "2023-07-01"]),
            "exit_date": pd.to_datetime(["2023-07-01", "2023-07-08"]),
        }
    )
    h1 = evaluate._slice_schedule(schedule, start="2023-01-01", end="2023-07-01")
    h2 = evaluate._slice_schedule(schedule, start="2023-07-01", end="2024-01-01")
    assert h1.empty
    assert len(h2) == 1


def _passing_metrics() -> dict[str, object]:
    return {
        "absolute_return_pct": 10.0,
        "cagr_to_strict_mdd": 4.0,
        "strict_mdd_pct": 5.0,
        "mean_gross_underlying_move_bp": 100.0,
        "monthly_cluster_sign_flip": {"p_value_one_sided": 0.05},
    }


def _passing_policy_windows() -> dict[str, dict[str, dict[str, object]]]:
    return {
        name: {
            "base_6bp": _passing_metrics(),
            "stress_10bp": {**_passing_metrics(), "absolute_return_pct": 5.0},
        }
        for name in evaluate.WINDOWS
    }


def test_mechanism_null_that_passes_every_gate_rejects_primary() -> None:
    policies = {name: _passing_policy_windows() for name in evaluate.POLICY_NAMES}
    verdict = evaluate.qualification(policies, evaluate.EvaluationConfig())
    assert verdict["qualifies"] is False
    assert set(verdict["passing_mechanism_controls"]) == {
        "cadence_confirmation_removed",
        "stale_hash_state_7d",
    }


def test_diagnostic_controls_cannot_replace_a_passing_primary() -> None:
    policies = {name: _passing_policy_windows() for name in evaluate.POLICY_NAMES}
    for name in evaluate.MECHANISM_REJECTION_CONTROLS:
        policies[name]["train"]["base_6bp"]["absolute_return_pct"] = -1.0
    policies["direction_flip"]["train"]["base_6bp"]["absolute_return_pct"] = -100.0
    policies["constant_long"]["train"]["base_6bp"]["absolute_return_pct"] = 500.0
    verdict = evaluate.qualification(policies, evaluate.EvaluationConfig())
    assert verdict["qualifies"] is True
    decision = evaluate._selection_decision(verdict)
    assert decision["performance_candidate"] == "MCR-7"
    assert decision["selected_alpha"] is None
