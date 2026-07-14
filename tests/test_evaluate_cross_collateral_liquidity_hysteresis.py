from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_cross_collateral_liquidity_hysteresis as evaluator


def _market(rows: int = 260) -> pd.DataFrame:
    close = np.linspace(100.0, 120.0, rows)
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        }
    )


def _schedule() -> pd.DataFrame:
    positions = [20, 80, 140, 200]
    return pd.DataFrame(
        {
            "signal_position": positions,
            "entry_position": [value + 1 for value in positions],
            "exit_position": [value + 25 for value in positions],
            "signal_date": [f"signal-{value}" for value in positions],
            "entry_date": [f"entry-{value}" for value in positions],
            "exit_date": [f"exit-{value}" for value in positions],
            "side": [-1, 1, -1, 1],
            "branch": [
                "bearish_hysteresis",
                "bullish_hysteresis",
                "bearish_hysteresis",
                "bullish_hysteresis",
            ],
            "hold_bars": [24] * 4,
        }
    )


def test_preregistration_hashes_and_support_are_frozen() -> None:
    result = evaluator.verify_preregistration()
    assert result["protocol"]["outcomes_opened_for_cclh"] is False
    assert result["support_calibration"]["parameters_searched"] is False
    assert result["support"]["nonoverlap_total"] == 167


def test_evaluator_source_is_frozen_before_outcomes() -> None:
    freeze = evaluator.verify_evaluation_freeze()
    assert freeze["outcomes_opened_for_cclh"] is False
    assert freeze["opened_windows"] == []
    assert freeze["evaluation_freeze_commit"].startswith("8ad4ebf")


def test_frozen_signal_and_clv_overlap_replay_without_returns() -> None:
    preregistration = evaluator.verify_preregistration()
    cfg = evaluator.SignalConfig()
    market, _ = evaluator.clvr.load_sources(cfg)
    signal = evaluator.build_signal(market, cfg)
    clv_signal = evaluator._clv_signal(market, cfg)
    evaluator.verify_signal_replay(
        signal,
        clv_signal,
        market,
        cfg,
        preregistration,
    )


def test_action_controls_preserve_reserved_clock() -> None:
    reserved = _schedule()
    market = _market()
    reverse = evaluator.policy_schedule(
        reserved,
        market,
        "reverse",
        permutation_seed=7,
        price_momentum_bars=12,
    )
    assert reverse["side"].tolist() == [1, -1, 1, -1]
    assert reverse["signal_position"].tolist() == [20, 80, 140, 200]

    always_long = evaluator.policy_schedule(
        reserved,
        market,
        "always_long",
        permutation_seed=7,
        price_momentum_bars=12,
    )
    always_short = evaluator.policy_schedule(
        reserved,
        market,
        "always_short",
        permutation_seed=7,
        price_momentum_bars=12,
    )
    assert always_long["side"].tolist() == [1, 1, 1, 1]
    assert always_short["side"].tolist() == [-1, -1, -1, -1]


def test_price_momentum_and_overlap_controls_use_only_reserved_events() -> None:
    reserved = _schedule()
    market = _market()
    momentum = evaluator.policy_schedule(
        reserved,
        market,
        "price_momentum",
        permutation_seed=7,
        price_momentum_bars=12,
    )
    assert momentum["side"].tolist() == [1, 1, 1, 1]
    assert momentum["signal_position"].tolist() == [20, 80, 140, 200]

    clv_reserved = reserved.iloc[[0]].copy()
    clv_reserved["signal_position"] = [21]
    removed = evaluator.policy_schedule(
        reserved,
        market,
        "clv_overlap_removed",
        permutation_seed=7,
        price_momentum_bars=12,
        clv_reserved_schedule=clv_reserved,
        clv_overlap_tolerance_bars=2,
    )
    assert removed["signal_position"].tolist() == [80, 140, 200]


def test_sign_permutation_is_reproducible_and_count_preserving() -> None:
    first = evaluator.policy_schedule(
        _schedule(),
        _market(),
        "permuted_sign",
        permutation_seed=7,
        price_momentum_bars=12,
    )
    second = evaluator.policy_schedule(
        _schedule(),
        _market(),
        "permuted_sign",
        permutation_seed=7,
        price_momentum_bars=12,
    )
    pd.testing.assert_frame_equal(first, second)
    assert sorted(first["side"].tolist()) == [-1, -1, 1, 1]


def test_unknown_policy_branch_and_momentum_lookback_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown CCLH control"):
        evaluator.policy_schedule(
            _schedule(),
            _market(),
            "repair",
            permutation_seed=1,
            price_momentum_bars=12,
        )
    broken = _schedule()
    broken.loc[0, "branch"] = "unknown"
    with pytest.raises(ValueError, match="unknown branch"):
        evaluator.policy_schedule(
            broken,
            _market(),
            "cclh",
            permutation_seed=1,
            price_momentum_bars=12,
        )
    early = _schedule().iloc[[0]].copy()
    early["signal_position"] = [5]
    with pytest.raises(ValueError, match="lacks causal lookback"):
        evaluator.policy_schedule(
            early,
            _market(),
            "price_momentum",
            permutation_seed=1,
            price_momentum_bars=12,
        )


def _metrics(
    *,
    absolute_return: float = 10.0,
    ratio: float = 4.0,
    mdd: float = 2.5,
    trades: int = 50,
    p_value: float = 0.05,
) -> dict[str, object]:
    return {
        "absolute_return_pct": absolute_return,
        "cagr_to_strict_mdd": ratio,
        "strict_mdd_pct": mdd,
        "trade_count": trades,
        "weekly_cluster_sign_flip": {"p_value_one_sided": p_value},
    }


def _passing_windows() -> dict[str, object]:
    windows: dict[str, object] = {}
    for name in evaluator.WINDOWS:
        base = _metrics(trades=25 if name.startswith("q") else 50)
        policies = {policy: dict(base) for policy in evaluator.POLICY_NAMES}
        for control in evaluator.QUALIFICATION_CONTROLS:
            policies[control] = _metrics(
                ratio=1.0,
                trades=25 if name.startswith("q") else 50,
            )
        windows[name] = policies
    return windows


def test_qualification_is_fixed_and_rejects_quarter_failure() -> None:
    passing = _passing_windows()
    assert evaluator._qualification(passing)["qualifies"] is True

    failing = _passing_windows()
    failing["q4"]["cclh"] = _metrics(absolute_return=-0.1, trades=19)
    result = evaluator._qualification(failing)
    assert result["qualifies"] is False
    assert "q4: non-positive absolute return" in result["failures"]
    assert "q4: fewer than 20 trades" in result["failures"]


def test_qualification_rejects_better_price_control() -> None:
    failing = _passing_windows()
    failing["train2023_h1"]["price_momentum"] = _metrics(ratio=5.0)
    failing["select2023_h2"]["price_momentum"] = _metrics(ratio=5.0)
    result = evaluator._qualification(failing)
    assert result["qualifies"] is False
    assert (
        "cclh: minimum train/select ratio does not beat price_momentum"
        in result["failures"]
    )


def test_structural_diagnostics_build_distinct_clocks_causally() -> None:
    cfg = replace(
        evaluator.SignalConfig(),
        robust_baseline_bars=4,
        robust_min_periods=2,
        confirmation_bars=2,
        exit_confirmation_bars=2,
        hold_bars=2,
    )
    rows = 12
    market = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "source_complete": [True] * rows,
        }
    )
    for venue, scale in (("um", 1.0), ("cm", 2.0)):
        for distance in range(1, 6):
            market[f"{venue}_depth_m{distance}"] = (
                scale * distance * np.linspace(1.0, 2.0, rows)
            )
            market[f"{venue}_depth_p{distance}"] = (
                distance * np.linspace(2.0, 1.0, rows)
            )
    diagnostics = evaluator.structural_signals(market, cfg)
    assert set(diagnostics) == set(evaluator.STRUCTURAL_DIAGNOSTICS)
    assert all(len(signal) == rows for signal in diagnostics.values())


def test_frozen_cclh_result_rejects_and_keeps_2024_sealed() -> None:
    path = Path(
        "results/cross_collateral_liquidity_hysteresis_selection_2026-07-14.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "475c927b5440ff08fe75b2b1c095c06271e9e11a13fef965e710a0d5eda37582"
    )
    result = json.loads(path.read_text())
    assert result["selection"] == {
        "selected_alpha": None,
        "rejected": True,
        "reason": "CCLH v1 failed at least one frozen 2023 gate",
    }
    assert result["protocol"]["evaluation_source_sha256"] == (
        "f44fb011fa229c84424143d6da5fed0c06f6d4adfedd71fdca51353c257a80f3"
    )
    assert result["protocol"]["sealed_windows"] == [
        "test2024",
        "eval2025",
        "ytd2026",
    ]
    assert result["windows"]["train2023_h1"]["cclh"][
        "absolute_return_pct"
    ] == pytest.approx(7.506351684776402)
    assert result["windows"]["select2023_h2"]["cclh"][
        "absolute_return_pct"
    ] == pytest.approx(-0.18411820214264685)
