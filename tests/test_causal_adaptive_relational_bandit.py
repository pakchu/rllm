from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from training import causal_adaptive_relational_bandit as bandit
from training import evaluate_causal_adaptive_relational_baselines as evaluate
from training.preregister_causal_adaptive_relational_tokens import TOKEN_COLUMNS


def _tokens(**updates: str) -> dict[str, str]:
    tokens = {column: "0" for column in TOKEN_COLUMNS}
    tokens.update(updates)
    return tokens


def _row(tokens: dict[str, str], follow: float, fade: float) -> dict[str, object]:
    outcomes = {
        "ABSTAIN": {"utility": 0.0},
        "FOLLOW": {"utility": follow},
        "FADE": {"utility": fade},
    }
    return {
        "tokens": tokens,
        "action_outcomes": outcomes,
        "oracle_best_action": bandit._best_action(outcomes),
    }


def _metrics(
    *,
    absolute_return: float = 5.0,
    ratio: float = 2.0,
    mdd: float = 5.0,
    trades: int = 50,
    p_value: float = 0.10,
    follow: int = 20,
    fade: int = 20,
) -> dict[str, object]:
    return {
        "absolute_return_pct": absolute_return,
        "cagr_to_strict_mdd": ratio,
        "strict_mdd_pct": mdd,
        "trade_count": trades,
        "long_count": max(0, trades // 2),
        "short_count": max(0, trades - trades // 2),
        "weekly_cluster_sign_flip": {"p_value_one_sided": p_value},
        "action_counts": {
            "ABSTAIN": max(0, 100 - follow - fade),
            "FOLLOW": follow,
            "FADE": fade,
        },
    }


def test_preregistration_hashes_and_support_are_frozen() -> None:
    result = bandit.verify_preregistration()
    assert result["protocol"]["outcomes_opened_for_carta"] is False
    assert result["support"]["nonoverlap_total"] == 559


def test_action_reward_uses_exact_multiplier_and_opposite_fade() -> None:
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0, 105.0, 110.0],
            "high": [100.0, 101.0, 108.0, 110.0],
            "low": [100.0, 99.0, 98.0, 110.0],
        }
    )
    schedule = pd.DataFrame(
        [
            {
                "signal_position": 0,
                "entry_position": 1,
                "exit_position": 3,
                "side": 1,
            }
        ]
    )
    row = next(schedule.itertuples(index=False))
    cfg = bandit.BanditConfig()
    outcomes = bandit.action_outcomes(frame, row, cfg)
    cost = (cfg.fee_rate + cfg.slippage_rate) * cfg.leverage
    follow_multiplier = (1.0 - cost) * (1.0 + 0.5 * 0.10) * (1.0 - cost)
    fade_multiplier = (1.0 - cost) * (1.0 - 0.5 * 0.10) * (1.0 - cost)
    assert outcomes["FOLLOW"]["account_multiplier"] == pytest.approx(
        follow_multiplier
    )
    assert outcomes["FADE"]["account_multiplier"] == pytest.approx(fade_multiplier)
    assert outcomes["FOLLOW"]["side"] == 1
    assert outcomes["FADE"]["side"] == -1
    expected = math.log(follow_multiplier) - (
        cfg.held_path_drawdown_penalty
        * float(outcomes["FOLLOW"]["held_path_drawdown"])
    )
    assert outcomes["FOLLOW"]["utility"] == pytest.approx(expected)


def test_prompt_contains_only_frozen_symbolic_tokens() -> None:
    prompt = bandit.prompt_from_tokens(_tokens(reference_side_token="LONG"))
    assert "reference_side_token=LONG" in prompt
    assert "date=" not in prompt
    assert "reward" not in prompt.lower()
    assert "2023-" not in prompt


def test_ridge_prediction_cannot_read_selection_outcomes() -> None:
    train = [
        _row(_tokens(reference_side_token="LONG"), 0.02, -0.02),
        _row(_tokens(reference_side_token="SHORT"), -0.02, 0.02),
        _row(_tokens(reference_side_token="LONG", tension_rank="4"), 0.01, -0.01),
        _row(_tokens(reference_side_token="SHORT", tension_rank="4"), -0.01, 0.01),
    ]
    cfg = bandit.BanditConfig(minimum_feature_count=1, ridge_alpha=1.0)
    policy = bandit.fit_ridge_policy(train, cfg)
    selection = [_row(_tokens(reference_side_token="LONG"), -999.0, 999.0)]
    baseline = bandit.predict_ridge(policy, selection)
    selection[0]["action_outcomes"] = {
        "ABSTAIN": {"utility": 0.0},
        "FOLLOW": {"utility": 999.0},
        "FADE": {"utility": -999.0},
    }
    assert bandit.predict_ridge(policy, selection) == baseline


def test_prediction_subset_does_not_release_candidate_clock() -> None:
    candidates = pd.DataFrame(
        [
            {
                "signal_position": 1,
                "entry_position": 2,
                "exit_position": 4,
                "side": 1,
                "branch": "carta_candidate",
                "hold_bars": 2,
            },
            {
                "signal_position": 5,
                "entry_position": 6,
                "exit_position": 8,
                "side": -1,
                "branch": "carta_candidate",
                "hold_bars": 2,
            },
        ]
    )
    schedule = bandit.prediction_schedule(candidates, ["ABSTAIN", "FADE"])
    assert schedule["signal_position"].tolist() == [5]
    assert schedule["side"].tolist() == [1]
    assert schedule["branch"].tolist() == ["fade"]


def test_shuffled_ridge_is_seed_deterministic() -> None:
    rows = [
        _row(_tokens(tension_rank=str(index % 5)), index / 100.0, -index / 100.0)
        for index in range(10)
    ]
    cfg = bandit.BanditConfig(minimum_feature_count=1, ridge_alpha=1.0)
    first = bandit.fit_ridge_policy(rows, cfg, shuffle_targets=True)
    second = bandit.fit_ridge_policy(rows, cfg, shuffle_targets=True)
    assert first == second


def test_learnability_gate_enforces_halves_controls_and_action_diversity() -> None:
    item = {
        "windows": {
            "select2023": _metrics(),
            "select2023_h1": _metrics(trades=20),
            "select2023_h2": _metrics(trades=20),
        }
    }
    control = {
        "policy": "always_follow",
        "absolute_return_pct": 1.0,
        "cagr_to_strict_mdd": 0.5,
    }
    assert evaluate.baseline_qualification(item, control)["qualifies"] is True
    item["windows"]["select2023_h2"]["absolute_return_pct"] = -1.0
    item["windows"]["select2023"]["action_counts"]["FADE"] = 0
    item["windows"]["select2023"]["short_count"] = 0
    result = evaluate.baseline_qualification(item, control)
    assert result["qualifies"] is False
    assert "select2023_h2: non-positive absolute return" in result["failures"]
    assert any("collapsed" in failure for failure in result["failures"])
    assert any("executed direction" in failure for failure in result["failures"])


def test_control_floor_includes_signature_memory() -> None:
    policies = {
        name: {
            "windows": {
                "select2023": _metrics(
                    absolute_return=10.0 if name == "signature_memory" else 1.0,
                    ratio=3.0 if name == "signature_memory" else 0.5,
                )
            }
        }
        for name in evaluate.CONTROL_POLICIES
    }
    assert evaluate._control_floor(policies)["policy"] == "signature_memory"


def test_evaluator_cannot_open_2024_or_later() -> None:
    assert max(end for _, end in evaluate.WINDOWS.values()) == "2024-01-01"
    assert set(evaluate.LEARNED_POLICIES) == {"relational_ridge", "naive_bayes"}
