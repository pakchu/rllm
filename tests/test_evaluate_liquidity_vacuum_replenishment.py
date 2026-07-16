from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from training import evaluate_liquidity_vacuum_replenishment as evaluate


def _market(
    opens: list[float],
    *,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    rows = len(opens)
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "open": opens,
            "high": opens if highs is None else highs,
            "low": opens if lows is None else lows,
            "close": opens,
        }
    )


def _schedule(
    *,
    signal: int = 1,
    entry: int = 2,
    exit: int = 14,
    side: int = 1,
    branch: str = "lvrt_r0",
) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=max(exit + 1, 20), freq="5min")
    return pd.DataFrame(
        [
            {
                "setup_position": signal - 1,
                "signal_position": signal,
                "entry_position": entry,
                "exit_position": exit,
                "setup_date": str(dates[signal - 1]),
                "signal_date": str(dates[signal]),
                "entry_date": str(dates[entry]),
                "exit_date": str(dates[exit]),
                "side": side,
                "branch": branch,
                "hold_bars": exit - entry,
            }
        ]
    )


def _funding(times: list[pd.Timestamp] | None = None, rates: list[float] | None = None) -> pd.DataFrame:
    times = [] if times is None else times
    rates = [] if rates is None else rates
    return pd.DataFrame(
        {
            "funding_time_ms": [int(time.value // 1_000_000) for time in times],
            "funding_time": times,
            "funding_rate": rates,
        }
    )


def _cfg(**changes: object) -> evaluate.EvaluationConfig:
    return replace(evaluate.EvaluationConfig(), cluster_permutations=64, **changes)


def _simulate(
    market: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    funding: pd.DataFrame | None = None,
    cost: float = 0.0,
) -> dict[str, object]:
    return evaluate.simulate_schedule(
        market,
        _funding() if funding is None else funding,
        schedule,
        start="2023-01-01",
        end="2023-01-02",
        cost_notional_per_side=cost,
        cfg=_cfg(),
        compute_cluster=True,
    )


def test_next_open_entry_and_fixed_scheduled_exit() -> None:
    opens = [50.0, 75.0, 100.0, *([100.0] * 11), 110.0, 200.0]
    stats = _simulate(_market(opens), _schedule())
    assert stats["absolute_return_pct"] == pytest.approx(5.0)
    assert stats["trade_count"] == 1


def test_exit_bar_extremes_are_excluded_from_held_path() -> None:
    opens = [100.0] * 18
    highs = opens.copy()
    lows = opens.copy()
    highs[14] = 1_000.0
    lows[14] = 1.0
    stats = _simulate(_market(opens, highs=highs, lows=lows), _schedule())
    assert stats["strict_mdd_pct"] == pytest.approx(0.0)


def test_strict_mdd_uses_favorable_then_adverse_and_liquidation_cost() -> None:
    opens = [100.0] * 18
    highs = opens.copy()
    lows = opens.copy()
    highs[2] = 120.0
    lows[13] = 90.0
    stats = _simulate(
        _market(opens, highs=highs, lows=lows),
        _schedule(),
        cost=0.001,
    )
    post_entry = 1.0 - 0.0005
    favorable = post_entry * 1.10
    adverse_liquidation = post_entry * 0.95 * (1.0 - 0.0005)
    expected = (1.0 - adverse_liquidation / favorable) * 100.0
    assert stats["strict_mdd_pct"] == pytest.approx(expected)


def test_flat_trade_charges_frozen_cost_per_notional_side() -> None:
    stats = _simulate(_market([100.0] * 18), _schedule(), cost=0.0006)
    expected = ((1.0 - 0.0006 * 0.5) ** 2 - 1.0) * 100.0
    assert stats["absolute_return_pct"] == pytest.approx(expected)


def test_funding_uses_half_open_interval_without_double_charge() -> None:
    market = _market([100.0] * 32)
    first = _schedule(signal=1, entry=2, exit=14, side=1)
    second = _schedule(signal=13, entry=14, exit=26, side=1)
    schedule = pd.concat([first, second], ignore_index=True)
    boundary = market.loc[14, "date"]
    funding = _funding([boundary], [0.01])
    stats = _simulate(market, schedule, funding=funding)
    assert stats["funding_settlement_count"] == 1
    assert stats["trades_with_funding"] == 1
    assert stats["absolute_return_pct"] == pytest.approx(-0.5)


def test_funding_credit_raises_intratrade_peak_before_adverse_path() -> None:
    opens = [100.0] * 18
    lows = opens.copy()
    lows[13] = 90.0
    market = _market(opens, lows=lows)
    funding = _funding([market.loc[6, "date"]], [-0.02])
    stats = _simulate(market, _schedule(side=1), funding=funding)
    expected = (1.0 - 0.95 * 1.01 / 1.01) * 100.0
    assert stats["strict_mdd_pct"] == pytest.approx(expected)


def test_cagr_uses_full_wall_clock_window() -> None:
    opens = [100.0] * 18
    opens[14] = 110.0
    market = _market(opens)
    stats = evaluate.simulate_schedule(
        market,
        _funding(),
        _schedule(),
        start="2023-01-01",
        end="2024-01-01",
        cost_notional_per_side=0.0,
        cfg=_cfg(),
        compute_cluster=False,
    )
    years = 365.0 / 365.25
    expected = (1.05 ** (1.0 / years) - 1.0) * 100.0
    assert stats["cagr_pct"] == pytest.approx(expected)


def test_weekly_cluster_sign_flip_is_deterministic() -> None:
    args = ([0.01, 0.02, -0.01], ["2023-01-01", "2023-01-02", "2023-01-09"])
    first = evaluate.weekly_cluster_sign_flip(
        *args, permutations=1_000, seed=20_260_717
    )
    second = evaluate.weekly_cluster_sign_flip(
        *args, permutations=1_000, seed=20_260_717
    )
    assert first == second
    assert first["cluster_count"] == 3


def _metrics(
    *,
    absolute: float = 10.0,
    ratio: float = 4.0,
    mdd: float = 5.0,
    trades: int = 150,
    gross_bp: float = 20.0,
    p: float = 0.05,
) -> dict[str, object]:
    return {
        "absolute_return_pct": absolute,
        "cagr_to_strict_mdd": ratio,
        "strict_mdd_pct": mdd,
        "trade_count": trades,
        "mean_gross_underlying_move_bp": gross_bp,
        "weekly_cluster_sign_flip": {"p_value_one_sided": p},
    }


def _passing_windows() -> dict[str, object]:
    return {
        "train": {"base": _metrics(), "stress_8bp": _metrics()},
        "select2023": {"base": _metrics(trades=100), "stress_8bp": _metrics()},
        "select2023_h1": {"base": _metrics(trades=50), "stress_8bp": _metrics()},
        "select2023_h2": {"base": _metrics(trades=50), "stress_8bp": _metrics()},
    }


def test_qualification_rejects_passing_time_shift_placebo() -> None:
    policies = {name: _passing_windows() for name in evaluate.POLICY_NAMES}
    policies["sign_permuted_confirmation"]["train"]["base"]["absolute_return_pct"] = -1.0
    verdict = evaluate.qualification(policies, _cfg())
    assert verdict["qualifies"] is False
    assert verdict["passing_rejection_placebos"] == ["one_day_shifted_setup"]


def test_qualification_accepts_primary_when_both_placebos_fail() -> None:
    policies = {name: _passing_windows() for name in evaluate.POLICY_NAMES}
    for name in evaluate.REJECTION_PLACEBOS:
        policies[name]["train"]["base"]["absolute_return_pct"] = -1.0
    verdict = evaluate.qualification(policies, _cfg())
    assert verdict["qualifies"] is True


def test_schedule_rejects_wrong_hold() -> None:
    with pytest.raises(ValueError, match="frozen hold"):
        _simulate(_market([100.0] * 18), _schedule(exit=13))


def test_same_count_control_clock_mutation_changes_hash() -> None:
    first = _schedule(side=1)
    second = _schedule(side=-1)
    assert len(first) == len(second)
    assert evaluate._clock_sha256(first) != evaluate._clock_sha256(second)


def test_verify_freeze_rejects_manifest_mutation(tmp_path, monkeypatch) -> None:
    hashes = {name: "0" * 64 for name in evaluate.POLICY_NAMES}
    hashes["primary"] = evaluate.EVENT_CLOCK_SHA256
    rows = {name: 1 for name in evaluate.POLICY_NAMES}
    from training import freeze_liquidity_vacuum_replenishment_evaluator as freeze

    payload = freeze.build_manifest(
        "a" * 40,
        policy_clock_sha256=hashes,
        policy_clock_rows=rows,
    )
    payload["funding_data_sha256"] = "f" * 64
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(evaluate, "EVALUATION_FREEZE", path)
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        evaluate.verify_evaluation_freeze()
