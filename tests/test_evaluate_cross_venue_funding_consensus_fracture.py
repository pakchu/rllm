from __future__ import annotations

import pandas as pd
import pytest

from training import evaluate_cross_venue_funding_consensus_fracture as evaluator


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_position": [0, 10, 20, 30],
            "entry_position": [1, 11, 21, 31],
            "exit_position": [5, 15, 25, 35],
            "signal_date": [
                "2023-01-01 00:55:00",
                "2023-01-01 08:55:00",
                "2023-01-01 16:55:00",
                "2023-01-02 00:55:00",
            ],
            "entry_date": [
                "2023-01-01 01:00:00",
                "2023-01-01 09:00:00",
                "2023-01-01 17:00:00",
                "2023-01-02 01:00:00",
            ],
            "exit_date": [
                "2023-01-01 08:00:00",
                "2023-01-01 16:00:00",
                "2023-01-02 00:00:00",
                "2023-01-02 08:00:00",
            ],
            "side": [-1, 1, -1, 1],
            "branch": [
                "bybit_rich",
                "bybit_cheap",
                "bybit_rich",
                "bybit_cheap",
            ],
            "hold_bars": [84, 84, 84, 84],
        }
    )


def test_preregistration_hashes_and_support_are_frozen() -> None:
    result = evaluator.verify_preregistration()
    assert result["protocol"]["outcomes_opened_for_cfcf"] is False
    assert result["support_calibration"]["selected_crowding_quantile"] == 0.90
    assert result["support"]["nonoverlap_total"] == 223


def test_frozen_signal_replays_support_without_opening_returns() -> None:
    preregistration = evaluator.verify_preregistration()
    cfg = evaluator.SignalConfig()
    premium, funding, market, _ = evaluator.load_sources(cfg)
    settlements = evaluator.build_settlement_features(premium, funding, cfg)
    state = evaluator.classify_settlements(settlements, cfg)
    signal = evaluator.project_to_market(state, market)
    evaluator.verify_signal_replay(
        state,
        signal,
        market,
        cfg,
        preregistration,
    )


def test_control_actions_preserve_or_abstain_on_reserved_clock() -> None:
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

    rich = evaluator.policy_schedule(
        reserved,
        "bybit_rich_only",
        permutation_seed=20_260_714,
    )
    cheap = evaluator.policy_schedule(
        reserved,
        "bybit_cheap_only",
        permutation_seed=20_260_714,
    )
    assert rich["entry_position"].tolist() == [1, 21]
    assert cheap["entry_position"].tolist() == [11, 31]


def test_branch_permutation_is_reproducible_and_count_preserving() -> None:
    first = evaluator.policy_schedule(
        _schedule(),
        "permuted_branch",
        permutation_seed=7,
    )
    second = evaluator.policy_schedule(
        _schedule(),
        "permuted_branch",
        permutation_seed=7,
    )
    pd.testing.assert_frame_equal(first, second)
    assert first["branch"].value_counts().to_dict() == {
        "bybit_rich": 2,
        "bybit_cheap": 2,
    }
    expected_side = first["branch"].map(
        {"bybit_rich": -1, "bybit_cheap": 1}
    )
    assert first["side"].tolist() == expected_side.tolist()


def test_unknown_policy_and_branch_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown CFCF control"):
        evaluator.policy_schedule(
            _schedule(),
            "repair",
            permutation_seed=1,
        )
    broken = _schedule()
    broken.loc[0, "branch"] = "unknown"
    with pytest.raises(ValueError, match="unknown branch"):
        evaluator.policy_schedule(
            broken,
            "cfcf",
            permutation_seed=1,
        )


def _metrics(
    *,
    absolute_return: float = 10.0,
    ratio: float = 4.0,
    mdd: float = 2.5,
    trades: int = 90,
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
        base = _metrics(trades=40 if "_h" in name else 90)
        policies = {policy: dict(base) for policy in evaluator.POLICY_NAMES}
        for control in ("reverse", "always_long", "always_short"):
            policies[control] = _metrics(
                ratio=1.0,
                trades=40 if "_h" in name else 90,
            )
        windows[name] = policies
    return windows


def test_qualification_gate_is_fixed_and_rejects_half_failure() -> None:
    passing = _passing_windows()
    assert evaluator._qualification(passing)["qualifies"] is True

    failing = _passing_windows()
    failing["select2023_h2"]["cfcf"] = _metrics(
        absolute_return=-0.1,
        trades=29,
    )
    result = evaluator._qualification(failing)
    assert result["qualifies"] is False
    assert "select2023_h2: non-positive absolute return" in result["failures"]
    assert "select2023_h2: fewer than 30 trades" in result["failures"]


def test_qualification_rejects_directional_control_that_generalizes_better() -> None:
    failing = _passing_windows()
    failing["train"]["always_long"] = _metrics(ratio=5.0)
    failing["select2023"]["always_long"] = _metrics(ratio=5.0)
    result = evaluator._qualification(failing)
    assert result["qualifies"] is False
    assert (
        "cfcf: minimum train/select ratio does not beat always_long"
        in result["failures"]
    )
