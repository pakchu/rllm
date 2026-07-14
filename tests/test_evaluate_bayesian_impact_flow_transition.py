from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from training import evaluate_bayesian_impact_flow_transition as evaluator


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_position": [0, 4],
            "entry_position": [1, 5],
            "exit_position": [3, 7],
            "signal_date": ["2023-01-01 00:00", "2023-01-01 00:20"],
            "entry_date": ["2023-01-01 00:05", "2023-01-01 00:25"],
            "exit_date": ["2023-01-01 00:15", "2023-01-01 00:35"],
            "side": [1, -1],
            "branch": ["propagation", "absorption"],
            "hold_bars": [2, 2],
            "origin_position": [0, 1],
        }
    )


def _metrics(
    *,
    ratio: float = 4.0,
    absolute_return: float = 10.0,
    trades: int = 100,
) -> dict:
    return {
        "absolute_return_pct": absolute_return,
        "cagr_pct": ratio * 2.0,
        "strict_mdd_pct": 2.0,
        "cagr_to_strict_mdd": ratio,
        "trade_count": trades,
        "weekly_cluster_sign_flip": {"p_value_one_sided": 0.05},
    }


def test_frozen_preregistration_hashes_and_support_are_valid() -> None:
    result = evaluator.verify_preregistration()
    assert result["protocol"]["outcomes_opened_for_bift"] is False
    assert result["support_calibration"]["selected_change_quantile"] == 0.925
    assert result["support"]["nonoverlap_total"] == 272


def test_controls_apply_after_candidate_clock_is_reserved() -> None:
    schedule = _schedule()
    follow = evaluator.policy_schedule(schedule, "always_follow", permutation_seed=1)
    fade = evaluator.policy_schedule(schedule, "always_fade", permutation_seed=1)
    propagation = evaluator.policy_schedule(
        schedule, "propagation_only", permutation_seed=1
    )
    absorption = evaluator.policy_schedule(
        schedule, "absorption_only", permutation_seed=1
    )
    assert follow["side"].tolist() == [1, 1]
    assert fade["side"].tolist() == [-1, -1]
    assert propagation["signal_position"].tolist() == [0]
    assert absorption["signal_position"].tolist() == [4]
    assert len(schedule) == 2


def test_permuted_branch_is_deterministic_and_preserves_branch_counts() -> None:
    schedule = pd.concat([_schedule()] * 10, ignore_index=True)
    first = evaluator.policy_schedule(
        schedule, "permuted_branch", permutation_seed=20_260_714
    )
    second = evaluator.policy_schedule(
        schedule, "permuted_branch", permutation_seed=20_260_714
    )
    pd.testing.assert_frame_equal(first, second)
    assert first["branch"].value_counts().to_dict() == {
        "propagation": 10,
        "absorption": 10,
    }


def test_evaluate_policy_uses_exact_next_open_multiplier() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=5, freq="5min"),
            "open": [100.0, 100.0, 105.0, 110.0, 110.0],
            "high": [100.0, 104.0, 112.0, 110.0, 110.0],
            "low": [100.0, 99.0, 103.0, 110.0, 110.0],
        }
    )
    schedule = pd.DataFrame(
        {
            "signal_position": [0],
            "entry_position": [1],
            "exit_position": [3],
            "side": [1],
            "branch": ["propagation"],
        }
    )
    cfg = replace(evaluator.EvaluationConfig(), cluster_permutations=10)
    result = evaluator.evaluate_policy(
        frame,
        schedule,
        policy="bift",
        start="2023-01-01",
        end="2023-01-02",
        cfg=cfg,
    )
    expected = (1.0 - 0.0003) * (1.0 + 0.5 * 0.10) * (1.0 - 0.0003)
    assert np.isclose(result["absolute_return_pct"], (expected - 1.0) * 100.0)
    assert result["trade_count"] == 1
    assert result["branch_counts"] == {"propagation": 1}


def test_qualification_enforces_economic_and_control_gates() -> None:
    windows = {
        "train": {
            "bift": _metrics(),
            "always_follow": _metrics(ratio=2.0),
            "always_fade": _metrics(ratio=1.0),
        },
        "select2023": {
            "bift": _metrics(trades=80),
            "always_follow": _metrics(ratio=2.0),
            "always_fade": _metrics(ratio=1.0),
        },
        "select2023_h1": {"bift": _metrics(trades=30)},
        "select2023_h2": {"bift": _metrics(trades=30)},
    }
    assert evaluator._qualification(windows)["qualifies"] is True
    windows["select2023"]["bift"] = _metrics(
        absolute_return=-1.0, trades=80
    )
    rejected = evaluator._qualification(windows)
    assert rejected["qualifies"] is False
    assert "select2023: non-positive absolute return" in rejected["failures"]
