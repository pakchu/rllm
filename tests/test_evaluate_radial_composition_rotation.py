from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import (
    evaluate_radial_composition_rotation as evaluator,
)


def _market(rows: int = 1_100) -> pd.DataFrame:
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
    positions = [20, 250, 480, 710, 940]
    sides = [-1, 1, -1, 1, -1]
    return pd.DataFrame(
        {
            "signal_position": positions,
            "entry_position": [value + 1 for value in positions],
            "exit_position": [value + 145 for value in positions],
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
                    + pd.Timedelta(minutes=5 * (value + 145))
                )
                for value in positions
            ],
            "side": sides,
            "branch": [
                "bearish_radial_composition_rotation"
                if side < 0
                else "bullish_radial_composition_rotation"
                for side in sides
            ],
            "hold_bars": [144] * len(positions),
        }
    )


def test_preregistration_hashes_and_support_are_frozen() -> None:
    result = evaluator.verify_preregistration()
    assert result["protocol"]["outcomes_opened_for_rcr144"] is False
    assert result["protocol"]["price_or_return_loaded"] is False
    assert result["support"]["nonoverlap_total"] == 646
    assert result["independence"]["passes_independence"] is True


def test_evaluator_source_is_frozen_before_outcomes() -> None:
    path = Path(
        "results/radial_composition_rotation_evaluator_freeze_2026-07-14.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "488fc9897387fd213ae9eac6ef6bbd1dc7555d7e27c367cc06165954dedcf07d"
    )
    freeze = evaluator.verify_evaluation_freeze()
    assert freeze["outcomes_opened_for_rcr144"] is False
    assert freeze["price_or_return_loaded"] is False
    assert freeze["opened_windows"] == []
    assert freeze["evaluation_source_commit"] == (
        "0065788228f7b09d5d517edb57c914c8c51cbf92"
    )
    assert freeze["evaluation_source_sha256"] == (
        "cdd9534a9002f699e903924a863901eadc2f57000b337c9cf2fdbf03acf0a680"
    )


def test_frozen_signal_replays_without_execution_prices() -> None:
    preregistration = evaluator.verify_preregistration()
    cfg = evaluator.SignalConfig()
    frame, _ = evaluator.load_shells(cfg)
    assert not {"open", "high", "low", "close"}.intersection(frame.columns)
    features = evaluator.build_features(frame, cfg)
    signal = evaluator.build_signal(features, cfg)
    schedule = evaluator.verify_signal_replay(
        frame,
        features,
        signal,
        cfg,
        preregistration,
    )
    assert len(schedule) == 646
    assert evaluator._event_clock_sha256(schedule) == evaluator.EVENT_CLOCK_SHA256


def test_event_clock_hash_binds_positions_sides_and_hold() -> None:
    schedule = _schedule()
    baseline = evaluator._event_clock_sha256(schedule)
    for column, value in (
        ("signal_position", 21),
        ("side", 1),
        ("hold_bars", 143),
    ):
        changed = schedule.copy()
        changed.loc[0, column] = value
        assert evaluator._event_clock_sha256(changed) != baseline


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
        end="2023-01-01 15:00:00",
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
        end="2023-01-01 15:00:00",
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
    with pytest.raises(ValueError, match="unknown RCR-144 control"):
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
            "rcr144",
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
    quarter_failure["q4"]["rcr144"] = _metrics(
        absolute_return=-0.1,
        trades=74,
    )
    result = evaluator._qualification(quarter_failure)
    assert result["qualifies"] is False
    assert "q4: non-positive absolute return" in result["failures"]

    control_failure = _passing_windows()
    control_failure["train2023_h1"]["price_momentum"] = _metrics(ratio=5.0)
    control_failure["select2023_h2"]["price_momentum"] = _metrics(ratio=5.0)
    result = evaluator._qualification(control_failure)
    assert result["qualifies"] is False
    assert (
        "rcr144: minimum train/select ratio does not beat price_momentum"
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
                "outcomes_opened_for_rcr144": False,
                "evaluation_source": str(source),
                "evaluation_source_sha256": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
                "evaluation_source_commit": "a" * 40,
                "preregistration_commit": evaluator.PREREGISTRATION_COMMIT,
                "support_commit": evaluator.SUPPORT_COMMIT,
                "event_clock_commit": evaluator.EVENT_CLOCK_COMMIT,
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


def test_frozen_rcr_result_rejects_and_keeps_2024_sealed() -> None:
    path = Path(
        "results/radial_composition_rotation_selection_2026-07-14.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "e89883bc28bcf3d45f2daffe3f11506ab070ffc68d4d327567a9094fb07988b7"
    )
    result = json.loads(path.read_text())
    assert result["selection"] == {
        "selected_alpha": None,
        "rejected": True,
        "reason": "RCR-144 failed at least one frozen calendar-2023 gate",
    }
    assert result["protocol"]["outcomes_opened_for_rcr144"] is True
    assert result["protocol"]["evaluation_source_sha256"] == (
        "cdd9534a9002f699e903924a863901eadc2f57000b337c9cf2fdbf03acf0a680"
    )
    assert result["protocol"]["sealed_windows"] == [
        "test2024",
        "eval2025",
        "ytd2026",
    ]
    h1 = result["windows"]["train2023_h1"]["rcr144"]
    h2 = result["windows"]["select2023_h2"]["rcr144"]
    assert h1["absolute_return_pct"] == pytest.approx(-4.111172557415355)
    assert h1["strict_mdd_pct"] == pytest.approx(15.826133256768138)
    assert h1["trade_count"] == 300
    assert h2["absolute_return_pct"] == pytest.approx(-12.579577460870972)
    assert h2["strict_mdd_pct"] == pytest.approx(21.45172016842035)
    assert h2["trade_count"] == 346
    assert result["windows"]["q1"]["rcr144"]["absolute_return_pct"] > 0.0
    assert result["windows"]["q2"]["rcr144"]["absolute_return_pct"] < 0.0
    assert result["windows"]["q3"]["rcr144"]["absolute_return_pct"] < 0.0
    assert result["windows"]["q4"]["rcr144"]["absolute_return_pct"] > 0.0
