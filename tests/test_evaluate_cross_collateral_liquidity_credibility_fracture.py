from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import (
    evaluate_cross_collateral_liquidity_credibility_fracture as evaluator,
)


def _market(rows: int = 320) -> pd.DataFrame:
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
    positions = [20, 80, 140, 200, 260]
    sides = [-1, 1, -1, 1, -1]
    return pd.DataFrame(
        {
            "signal_position": positions,
            "entry_position": [value + 1 for value in positions],
            "exit_position": [value + 3 for value in positions],
            "signal_date": [
                str(pd.Timestamp("2023-01-01") + pd.Timedelta(minutes=5 * value))
                for value in positions
            ],
            "entry_date": [
                str(
                    pd.Timestamp("2023-01-01")
                    + pd.Timedelta(minutes=5 * (value + 1))
                )
                for value in positions
            ],
            "exit_date": [
                str(
                    pd.Timestamp("2023-01-01")
                    + pd.Timedelta(minutes=5 * (value + 3))
                )
                for value in positions
            ],
            "side": sides,
            "branch": [
                "bearish_display_firmness_divergence" if side < 0 else
                "bullish_display_firmness_divergence"
                for side in sides
            ],
            "hold_bars": [2] * len(positions),
        }
    )


def test_preregistration_hashes_and_support_are_frozen() -> None:
    result = evaluator.verify_preregistration()
    assert result["protocol"]["outcomes_opened_for_pdf10"] is False
    assert result["protocol"]["price_or_return_loaded"] is False
    assert result["support"]["nonoverlap_total"] == 591
    assert result["independence"]["passes_independence"] is True


def test_evaluator_source_is_frozen_before_outcomes() -> None:
    freeze = evaluator.verify_evaluation_freeze()
    assert freeze["outcomes_opened_for_pdf10"] is False
    assert freeze["opened_windows"] == []
    assert freeze["evaluation_source_commit"] == (
        "4afbf158ff9f2e7cef80ed11d2219e65b843d93c"
    )
    assert freeze["evaluation_source_sha256"] == (
        "513570e06529bd65966e505a2fc005f160417992fa52d36122401419cad9c252"
    )


def test_frozen_signal_replays_without_execution_prices() -> None:
    preregistration = evaluator.verify_preregistration()
    cfg = evaluator.SignalConfig()
    frame, _ = evaluator.load_credibility(cfg)
    assert not {"open", "high", "low", "close"}.intersection(frame.columns)
    signal = evaluator.build_signal(frame, cfg)
    schedule = evaluator.verify_signal_replay(
        frame,
        signal,
        cfg,
        preregistration,
    )
    assert len(schedule) == 591
    assert evaluator._event_clock_sha256(schedule) == evaluator.EVENT_CLOCK_SHA256


def test_action_controls_preserve_the_frozen_annual_clock() -> None:
    reserved = _schedule()
    market = _market()
    expected_positions = reserved["signal_position"].tolist()
    for policy in evaluator.POLICY_NAMES:
        output = evaluator.policy_schedule(
            reserved,
            market,
            policy,
            permutation_seed=7,
            price_momentum_bars=1,
        )
        assert output["signal_position"].tolist() == expected_positions
        assert len(output) == len(reserved)
        assert output["side"].isin([-1, 1]).all()
    reverse = evaluator.policy_schedule(
        reserved,
        market,
        "reverse",
        permutation_seed=7,
        price_momentum_bars=1,
    )
    assert reverse["side"].tolist() == [1, -1, 1, -1, 1]


def test_sign_permutation_is_assigned_once_before_window_slicing() -> None:
    annual = evaluator.policy_schedule(
        _schedule(),
        _market(),
        "permuted_sign",
        permutation_seed=7,
        price_momentum_bars=1,
    )
    first = evaluator._schedule_for_window(
        annual,
        start="2023-01-01",
        end="2023-01-01 18:00:00",
    )
    replay = evaluator.policy_schedule(
        _schedule(),
        _market(),
        "permuted_sign",
        permutation_seed=7,
        price_momentum_bars=1,
    )
    second = evaluator._schedule_for_window(
        replay,
        start="2023-01-01",
        end="2023-01-01 18:00:00",
    )
    pd.testing.assert_frame_equal(first, second)


def test_price_momentum_is_causal_and_ties_preserve_the_clock() -> None:
    market = _market()
    market.loc[19:20, "close"] = 101.0
    output = evaluator.policy_schedule(
        _schedule(),
        market,
        "price_momentum",
        permutation_seed=7,
        price_momentum_bars=1,
    )
    assert len(output) == len(_schedule())
    assert output.loc[0, "side"] == 1
    changed = market.copy()
    changed.loc[21:, "close"] *= 100.0
    replay = evaluator.policy_schedule(
        _schedule().iloc[[0]],
        changed,
        "price_momentum",
        permutation_seed=7,
        price_momentum_bars=1,
    )
    assert replay.loc[0, "side"] == output.loc[0, "side"]


def test_unknown_policy_branch_and_momentum_horizon_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown PDF-10 control"):
        evaluator.policy_schedule(
            _schedule(),
            _market(),
            "repair",
            permutation_seed=1,
            price_momentum_bars=1,
        )
    broken = _schedule()
    broken.loc[0, "branch"] = "unknown"
    with pytest.raises(ValueError, match="unknown branch"):
        evaluator.policy_schedule(
            broken,
            _market(),
            "pdf10",
            permutation_seed=1,
            price_momentum_bars=1,
        )
    with pytest.raises(ValueError, match="one-bar close momentum"):
        evaluator.policy_schedule(
            _schedule(),
            _market(),
            "price_momentum",
            permutation_seed=1,
            price_momentum_bars=2,
        )


def test_window_slice_rejects_a_crossing_trade() -> None:
    annual = _schedule().iloc[[0]].copy()
    annual.loc[annual.index[0], "signal_date"] = "2023-03-31 23:50:00"
    annual.loc[annual.index[0], "entry_date"] = "2023-03-31 23:55:00"
    annual.loc[annual.index[0], "exit_date"] = "2023-04-01 00:05:00"
    with pytest.raises(ValueError, match="crosses an evaluation window"):
        evaluator._schedule_for_window(
            annual,
            start="2023-01-01",
            end="2023-04-01",
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
        base = _metrics(trades=75 if name.startswith("q") else 200)
        policies = {policy: dict(base) for policy in evaluator.POLICY_NAMES}
        for control in evaluator.QUALIFICATION_CONTROLS:
            policies[control] = _metrics(
                ratio=1.0,
                trades=75 if name.startswith("q") else 200,
            )
        windows[name] = policies
    return windows


def test_qualification_rejects_quarter_and_control_failures() -> None:
    passing = _passing_windows()
    assert evaluator._qualification(passing)["qualifies"] is True

    quarter_failure = _passing_windows()
    quarter_failure["q4"]["pdf10"] = _metrics(
        absolute_return=-0.1,
        trades=74,
    )
    result = evaluator._qualification(quarter_failure)
    assert result["qualifies"] is False
    assert "q4: non-positive absolute return" in result["failures"]
    assert "q4: fewer than 75 trades" in result["failures"]

    control_failure = _passing_windows()
    control_failure["train2023_h1"]["price_momentum"] = _metrics(ratio=5.0)
    control_failure["select2023_h2"]["price_momentum"] = _metrics(ratio=5.0)
    result = evaluator._qualification(control_failure)
    assert result["qualifies"] is False
    assert (
        "pdf10: minimum train/select ratio does not beat price_momentum"
        in result["failures"]
    )


def test_evaluation_parameters_are_frozen() -> None:
    cfg = evaluator.EvaluationConfig()
    evaluator._validate_evaluation_config(cfg)
    with pytest.raises(ValueError, match="evaluation config is frozen"):
        evaluator._validate_evaluation_config(replace(cfg, leverage=1.0))


def test_evaluation_freeze_rejects_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "evaluator.py"
    source.write_text("frozen\n")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "outcomes_opened_for_pdf10": False,
                "evaluation_source": str(source),
                "evaluation_source_sha256": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
                "evaluation_source_commit": "a" * 40,
                "preregistration_commit": evaluator.PREREGISTRATION_COMMIT,
                "support_commit": evaluator.SUPPORT_COMMIT,
                "preregistration_result_sha256": (
                    evaluator.PREREGISTRATION_RESULT_SHA256
                ),
                "opened_windows": [],
                "sealed_windows": [
                    *evaluator.WINDOWS,
                    "test2024",
                    "eval2025",
                    "ytd2026",
                ],
            }
        )
    )
    monkeypatch.setattr(evaluator, "EVALUATION_SOURCE", source)
    monkeypatch.setattr(evaluator, "EVALUATION_FREEZE", freeze)
    evaluator.verify_evaluation_freeze()
    source.write_text("mutated\n")
    with pytest.raises(ValueError, match="differs from pre-outcome freeze"):
        evaluator.verify_evaluation_freeze()


def test_frozen_pdf10_result_rejects_and_keeps_2024_sealed() -> None:
    path = Path(
        "results/cross_collateral_liquidity_credibility_fracture_"
        "selection_2026-07-14.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "663d1b4a832fd87cdc92de8569915d5441a6880736b6a46092615eea03822f24"
    )
    result = json.loads(path.read_text())
    assert result["selection"] == {
        "selected_alpha": None,
        "rejected": True,
        "reason": "PDF-10 failed at least one frozen calendar-2023 gate",
    }
    assert result["protocol"]["outcomes_opened_for_pdf10"] is True
    assert result["protocol"]["evaluation_source_sha256"] == (
        "513570e06529bd65966e505a2fc005f160417992fa52d36122401419cad9c252"
    )
    assert result["protocol"]["sealed_windows"] == [
        "test2024",
        "eval2025",
        "ytd2026",
    ]
    assert result["windows"]["train2023_h1"]["pdf10"][
        "absolute_return_pct"
    ] == pytest.approx(-13.09033626874182)
    assert result["windows"]["select2023_h2"]["pdf10"][
        "absolute_return_pct"
    ] == pytest.approx(-20.359545808646907)
    assert all(
        result["windows"][quarter]["pdf10"]["absolute_return_pct"] < 0.0
        for quarter in ("q1", "q2", "q3", "q4")
    )
