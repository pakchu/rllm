from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from training import evaluate_cross_collateral_liquidity_vacuum as evaluator


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_position": [0, 10, 20, 30],
            "entry_position": [1, 11, 21, 31],
            "exit_position": [5, 15, 25, 35],
            "signal_date": [
                "2023-01-01 00:00:00",
                "2023-01-01 01:00:00",
                "2023-01-01 02:00:00",
                "2023-01-01 03:00:00",
            ],
            "entry_date": [
                "2023-01-01 00:05:00",
                "2023-01-01 01:05:00",
                "2023-01-01 02:05:00",
                "2023-01-01 03:05:00",
            ],
            "exit_date": [
                "2023-01-01 00:25:00",
                "2023-01-01 01:25:00",
                "2023-01-01 02:25:00",
                "2023-01-01 03:25:00",
            ],
            "side": [-1, 1, -1, 1],
            "branch": ["vacuum"] * 4,
            "hold_bars": [4] * 4,
        }
    )


def test_preregistration_hashes_and_support_are_frozen() -> None:
    result = evaluator.verify_preregistration()
    assert result["protocol"]["outcomes_opened_for_clv"] is False
    assert result["support_calibration"]["selected_score_quantile"] == 0.975
    assert result["support"]["nonoverlap_total"] == 521


def test_frozen_signal_replays_support_without_opening_returns() -> None:
    preregistration = evaluator.verify_preregistration()
    cfg = evaluator.SignalConfig()
    market, _ = evaluator.load_sources(cfg)
    features = evaluator.build_features(market, cfg)
    signal = evaluator.classify_vacuum(features, cfg)
    evaluator.verify_signal_replay(
        signal,
        market,
        cfg,
        preregistration,
    )


def test_control_actions_preserve_reserved_clock() -> None:
    reserved = _schedule()
    reverse = evaluator.policy_schedule(
        reserved,
        "reverse",
        permutation_seed=20_260_714,
    )
    assert reverse["side"].tolist() == [1, -1, 1, -1]
    assert reverse["entry_position"].tolist() == [1, 11, 21, 31]

    always_long = evaluator.policy_schedule(
        reserved,
        "always_long",
        permutation_seed=20_260_714,
    )
    always_short = evaluator.policy_schedule(
        reserved,
        "always_short",
        permutation_seed=20_260_714,
    )
    assert always_long["side"].tolist() == [1, 1, 1, 1]
    assert always_short["side"].tolist() == [-1, -1, -1, -1]
    assert always_long["entry_position"].tolist() == [1, 11, 21, 31]
    assert always_short["entry_position"].tolist() == [1, 11, 21, 31]


def test_sign_permutation_is_reproducible_and_count_preserving() -> None:
    first = evaluator.policy_schedule(
        _schedule(),
        "permuted_sign",
        permutation_seed=7,
    )
    second = evaluator.policy_schedule(
        _schedule(),
        "permuted_sign",
        permutation_seed=7,
    )
    pd.testing.assert_frame_equal(first, second)
    assert sorted(first["side"].tolist()) == [-1, -1, 1, 1]
    assert first["entry_position"].tolist() == [1, 11, 21, 31]


def test_clv_policy_uses_exact_costed_next_open_execution() -> None:
    market = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=7, freq="5min"),
            "open": [100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 110.0],
            "high": [100.0] * 7,
            "low": [100.0] * 7,
        }
    )
    schedule = pd.DataFrame(
        {
            "signal_position": [0],
            "entry_position": [1],
            "exit_position": [5],
            "signal_date": ["2023-01-01 00:00:00"],
            "entry_date": ["2023-01-01 00:05:00"],
            "exit_date": ["2023-01-01 00:25:00"],
            "side": [1],
            "branch": ["vacuum"],
            "hold_bars": [4],
        }
    )
    cfg = replace(evaluator.EvaluationConfig(), cluster_permutations=16)
    metrics = evaluator.evaluate_policy(
        market,
        schedule,
        policy="clv",
        start="2023-01-01",
        end="2023-01-02",
        cfg=cfg,
    )
    expected_equity = (1.0 - 0.0003) * (1.0 + 0.5 * 0.10) * (1.0 - 0.0003)
    assert metrics["absolute_return_pct"] == pytest.approx(
        (expected_equity - 1.0) * 100.0
    )
    assert metrics["trade_count"] == 1
    assert metrics["reserved_candidate_count"] == 1
    assert metrics["executed_candidate_count"] == 1


def test_unknown_policy_branch_and_side_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown CLV control"):
        evaluator.policy_schedule(
            _schedule(),
            "repair",
            permutation_seed=1,
        )
    broken_branch = _schedule()
    broken_branch.loc[0, "branch"] = "refill"
    with pytest.raises(ValueError, match="unknown branch"):
        evaluator.policy_schedule(
            broken_branch,
            "clv",
            permutation_seed=1,
        )
    broken_side = _schedule()
    broken_side.loc[0, "side"] = 0
    with pytest.raises(ValueError, match="invalid side"):
        evaluator.policy_schedule(
            broken_side,
            "clv",
            permutation_seed=1,
        )


def _metrics(
    *,
    absolute_return: float = 10.0,
    ratio: float = 4.0,
    mdd: float = 2.5,
    trades: int = 200,
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
        base = _metrics(trades=80 if name.startswith("q") else 200)
        policies = {policy: dict(base) for policy in evaluator.POLICY_NAMES}
        for control in ("reverse", "always_long", "always_short"):
            policies[control] = _metrics(
                ratio=1.0,
                trades=80 if name.startswith("q") else 200,
            )
        windows[name] = policies
    return windows


def test_qualification_gate_is_fixed_and_rejects_quarter_failure() -> None:
    passing = _passing_windows()
    assert evaluator._qualification(passing)["qualifies"] is True

    failing = _passing_windows()
    failing["q4"]["clv"] = _metrics(absolute_return=-0.1, trades=74)
    result = evaluator._qualification(failing)
    assert result["qualifies"] is False
    assert "q4: non-positive absolute return" in result["failures"]
    assert "q4: fewer than 75 trades" in result["failures"]


def test_qualification_rejects_control_that_generalizes_better() -> None:
    failing = _passing_windows()
    failing["train2023_h1"]["always_long"] = _metrics(ratio=5.0)
    failing["select2023_h2"]["always_long"] = _metrics(ratio=5.0)
    result = evaluator._qualification(failing)
    assert result["qualifies"] is False
    assert (
        "clv: minimum train/select ratio does not beat always_long"
        in result["failures"]
    )
