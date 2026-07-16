from __future__ import annotations

from dataclasses import replace
import json
from unittest.mock import patch

import pandas as pd
import pytest

from training import evaluate_perp_only_wick_rejection as evaluate


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
    entry: int = 4,
    exit: int = 16,
    side: int = 1,
    branch: str = "powr_12",
) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=max(exit + 1, 20), freq="5min")
    return pd.DataFrame(
        [
            {
                "signal_position": signal,
                "entry_position": entry,
                "exit_position": exit,
                "signal_date": str(dates[signal]),
                "entry_date": str(dates[entry]),
                "exit_date": str(dates[exit]),
                "side": side,
                "branch": branch,
                "entry_delay_bars": entry - signal,
                "hold_bars": exit - entry,
            }
        ]
    )


def _funding(
    times: list[pd.Timestamp] | None = None,
    rates: list[float] | None = None,
) -> pd.DataFrame:
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
    with patch.object(evaluate, "verify_evaluation_freeze", return_value={}):
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


def test_signal_plus_three_entry_and_fixed_scheduled_exit() -> None:
    opens = [50.0, 75.0, 80.0, 90.0, 100.0, *([100.0] * 11), 110.0, 200.0]
    stats = _simulate(_market(opens), _schedule())
    assert stats["absolute_return_pct"] == pytest.approx(5.0)
    assert stats["trade_count"] == 1


def test_public_simulator_refuses_outcomes_before_freeze(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(evaluate, "EVALUATION_FREEZE", tmp_path / "missing.json")
    with pytest.raises(ValueError, match="evaluator freeze is missing"):
        evaluate.simulate_schedule(
            _market([100.0] * 20),
            _funding(),
            _schedule(),
            start="2023-01-01",
            end="2023-01-02",
            cost_notional_per_side=0.0,
            cfg=_cfg(),
            compute_cluster=False,
        )


def test_exit_bar_extremes_are_excluded_from_held_path() -> None:
    opens = [100.0] * 18
    highs = opens.copy()
    lows = opens.copy()
    highs[16] = 1_000.0
    lows[16] = 1.0
    stats = _simulate(_market(opens, highs=highs, lows=lows), _schedule())
    assert stats["strict_mdd_pct"] == pytest.approx(0.0)


def test_strict_mdd_uses_favorable_then_adverse_and_liquidation_cost() -> None:
    opens = [100.0] * 18
    highs = opens.copy()
    lows = opens.copy()
    highs[4] = 120.0
    lows[15] = 90.0
    stats = _simulate(
        _market(opens, highs=highs, lows=lows),
        _schedule(),
        cost=0.001,
    )
    favorable = 1.0 - 0.0005 + 0.5 * 0.20
    adverse_liquidation = 1.0 - 0.0005 + 0.5 * -0.10 - 0.0005 * 0.90
    expected = (1.0 - adverse_liquidation / favorable) * 100.0
    assert stats["strict_mdd_pct"] == pytest.approx(expected)


def test_flat_trade_charges_frozen_cost_per_notional_side() -> None:
    stats = _simulate(_market([100.0] * 18), _schedule(), cost=0.0006)
    expected = -(0.0006 * 0.5 * 2.0) * 100.0
    assert stats["absolute_return_pct"] == pytest.approx(expected)


def test_funding_uses_half_open_interval_without_double_charge() -> None:
    market = _market([100.0] * 34)
    first = _schedule(signal=1, entry=4, exit=16, side=1)
    second = _schedule(signal=13, entry=16, exit=28, side=1)
    schedule = pd.concat([first, second], ignore_index=True)
    boundary = market.loc[16, "date"]
    funding = _funding([boundary], [0.01])
    stats = _simulate(market, schedule, funding=funding)
    assert stats["funding_settlement_count"] == 1
    assert stats["trades_with_funding"] == 1
    assert stats["absolute_return_pct"] == pytest.approx(-0.5)


def test_frozen_realized_funding_source_loads_without_mark_imputation() -> None:
    with patch.object(evaluate, "verify_evaluation_freeze", return_value={}):
        funding, source = evaluate.load_realized_funding()
    assert len(funding) == 4_383
    assert funding.columns.tolist() == [
        "funding_time_ms",
        "funding_time",
        "funding_rate",
    ]
    assert source["funding_data_sha256"] == evaluate.FUNDING_DATA_SHA256


def test_funding_credit_raises_intratrade_peak_before_adverse_path() -> None:
    opens = [100.0] * 18
    lows = opens.copy()
    lows[15] = 90.0
    market = _market(opens, lows=lows)
    funding = _funding([market.loc[8, "date"]], [-0.02])
    stats = _simulate(market, _schedule(side=1), funding=funding)
    expected = (1.0 - 0.96 / 1.01) * 100.0
    assert stats["strict_mdd_pct"] == pytest.approx(expected)


def test_cagr_uses_full_wall_clock_window() -> None:
    opens = [100.0] * 18
    opens[16] = 110.0
    market = _market(opens)
    with patch.object(evaluate, "verify_evaluation_freeze", return_value={}):
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


def test_qualification_rejects_passing_mechanism_controls() -> None:
    policies = {name: _passing_windows() for name in evaluate.POLICY_NAMES}
    verdict = evaluate.qualification(policies, _cfg())
    assert verdict["qualifies"] is False
    assert verdict["passing_mechanism_controls"] == list(
        evaluate.MECHANISM_REJECTION_CONTROLS
    )


def test_exactly_twelve_bp_fails_strict_gross_move_gate() -> None:
    windows = _passing_windows()
    windows["train"]["base"]["mean_gross_underlying_move_bp"] = 12.0
    failures = evaluate._performance_gate_failures(windows, _cfg())
    assert "train: mean gross move not above 12 bp" in failures


def test_qualification_accepts_primary_when_mechanism_controls_fail() -> None:
    policies = {name: _passing_windows() for name in evaluate.POLICY_NAMES}
    for name in evaluate.MECHANISM_REJECTION_CONTROLS:
        policies[name]["train"]["base"]["absolute_return_pct"] = -1.0
    verdict = evaluate.qualification(policies, _cfg())
    assert verdict["qualifies"] is True


def test_qualification_requires_positive_delayed_entry_windows() -> None:
    policies = {name: _passing_windows() for name in evaluate.POLICY_NAMES}
    for name in evaluate.MECHANISM_REJECTION_CONTROLS:
        policies[name]["train"]["base"]["absolute_return_pct"] = -1.0
    policies[evaluate.DELAYED_ENTRY_CONTROL]["select2023"]["base"][
        "absolute_return_pct"
    ] = -0.1
    verdict = evaluate.qualification(policies, _cfg())
    assert verdict["qualifies"] is False
    assert verdict["delayed_entry_gate_failures"] == [
        "select2023: one-bar-delayed entry non-positive absolute return"
    ]


def test_direction_flip_is_diagnostic_only() -> None:
    policies = {name: _passing_windows() for name in evaluate.POLICY_NAMES}
    for name in evaluate.MECHANISM_REJECTION_CONTROLS:
        policies[name]["train"]["base"]["absolute_return_pct"] = -1.0
    policies["direction_flip"]["train"]["base"]["absolute_return_pct"] = -99.0
    verdict = evaluate.qualification(policies, _cfg())
    assert verdict["qualifies"] is True
    assert verdict["direction_flip_is_diagnostic_only"] is True


def test_performance_pass_cannot_skip_final_orthogonality_gate() -> None:
    decision = evaluate._selection_decision({"qualifies": True})
    assert decision["selected_alpha"] is None
    assert decision["performance_candidate"] == "POWR-12"
    assert decision["orthogonality_evaluated"] is False
    assert decision["promotion_ready"] is False


def test_result_manifest_hash_detects_mutation() -> None:
    payload = evaluate._seal_result({"selection": {"status": "pending"}})
    evaluate.validate_result_hash(payload)
    payload["selection"]["status"] = "selected"
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        evaluate.validate_result_hash(payload)


def test_schedule_rejects_wrong_hold() -> None:
    with pytest.raises(ValueError, match="frozen hold"):
        _simulate(_market([100.0] * 20), _schedule(exit=15))


def test_schedule_rejects_unregistered_entry_delay() -> None:
    with pytest.raises(ValueError, match="entry delay"):
        _simulate(_market([100.0] * 22), _schedule(entry=6, exit=18))


def test_schedule_rejects_timestamp_position_mismatch() -> None:
    schedule = _schedule()
    schedule.loc[0, "entry_date"] = "2023-01-01 00:30:00"
    with pytest.raises(ValueError, match="entry timestamp"):
        _simulate(_market([100.0] * 20), schedule)


def test_same_count_control_clock_mutation_changes_hash() -> None:
    first = _schedule(side=1)
    second = _schedule(side=-1)
    assert len(first) == len(second)
    assert evaluate._clock_sha256(first) != evaluate._clock_sha256(second)


def test_verify_freeze_rejects_manifest_mutation(tmp_path, monkeypatch) -> None:
    hashes = {name: "0" * 64 for name in evaluate.POLICY_NAMES}
    hashes["primary"] = evaluate.EVENT_CLOCK_SHA256
    rows = {name: 1 for name in evaluate.POLICY_NAMES}
    from training import freeze_perp_only_wick_rejection_evaluator as freeze

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
