from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from training import evaluate_metaorder_fragmentation_impact_curvature as evaluate


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
    signal: int,
    entry: int,
    exit: int,
    side: int = 1,
    branch: str = "continuation",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_position": signal,
                "entry_position": entry,
                "exit_position": exit,
                "signal_date": "unused",
                "entry_date": "unused",
                "exit_date": "unused",
                "side": side,
                "branch": branch,
                "hold_bars": exit - entry,
            }
        ]
    )


def _cfg(**changes: object) -> evaluate.EvaluationConfig:
    return replace(
        evaluate.EvaluationConfig(),
        fee_rate=0.0,
        slippage_rate=0.0,
        cluster_permutations=64,
        **changes,
    )


def test_next_open_entry_and_scheduled_open_exit() -> None:
    frame = _market([50.0, 75.0, 100.0, 105.0, 110.0, 200.0])
    stats = evaluate.simulate_schedule(
        frame,
        _schedule(signal=1, entry=2, exit=4),
        start="2023-01-01",
        end="2023-01-02",
        cfg=_cfg(),
    )
    assert stats["absolute_return_pct"] == pytest.approx(5.0)
    assert stats["trade_count"] == 1


def test_frozen_preregistration_hashes_and_config_match() -> None:
    result = evaluate._verify_preregistration()
    assert result["protocol"]["outcomes_opened"] is False
    assert result["all_candidates_pass_support"] is True


def test_same_open_exit_and_reentry_are_executable() -> None:
    frame = _market([100.0] * 8)
    first = _schedule(signal=1, entry=2, exit=4)
    second = _schedule(signal=3, entry=4, exit=6, side=-1, branch="fade")
    schedule = pd.concat([first, second], ignore_index=True)
    stats = evaluate.simulate_schedule(
        frame,
        schedule,
        start="2023-01-01",
        end="2023-01-02",
        cfg=_cfg(),
    )
    assert stats["trade_count"] == 2
    assert stats["long_count"] == 1
    assert stats["short_count"] == 1


def test_exit_bar_extremes_are_not_part_of_held_path() -> None:
    frame = _market(
        [100.0] * 6,
        highs=[100.0, 100.0, 100.0, 100.0, 1_000.0, 100.0],
        lows=[100.0, 100.0, 100.0, 100.0, 1.0, 100.0],
    )
    stats = evaluate.simulate_schedule(
        frame,
        _schedule(signal=1, entry=2, exit=4),
        start="2023-01-01",
        end="2023-01-02",
        cfg=_cfg(),
    )
    assert stats["strict_mdd_pct"] == pytest.approx(0.0)


def test_strict_mdd_uses_complete_favorable_then_adverse_path() -> None:
    frame = _market(
        [100.0] * 7,
        highs=[100.0, 100.0, 120.0, 100.0, 100.0, 100.0, 100.0],
        lows=[100.0, 100.0, 100.0, 100.0, 90.0, 100.0, 100.0],
    )
    stats = evaluate.simulate_schedule(
        frame,
        _schedule(signal=1, entry=2, exit=5),
        start="2023-01-01",
        end="2023-01-02",
        cfg=_cfg(),
    )
    expected = (1.0 - 0.95 / 1.10) * 100.0
    assert stats["strict_mdd_pct"] == pytest.approx(expected)


def test_flat_trade_charges_six_basis_points_per_side_at_half_leverage() -> None:
    frame = _market([100.0] * 6)
    cfg = replace(evaluate.EvaluationConfig(), cluster_permutations=64)
    stats = evaluate.simulate_schedule(
        frame,
        _schedule(signal=1, entry=2, exit=4),
        start="2023-01-01",
        end="2023-01-02",
        cfg=cfg,
    )
    expected = ((1.0 - 0.0006 * 0.5) ** 2 - 1.0) * 100.0
    assert stats["absolute_return_pct"] == pytest.approx(expected)


def test_long_and_short_gross_returns_are_symmetric() -> None:
    long_frame = _market([100.0, 100.0, 100.0, 101.0, 102.0, 102.0])
    short_frame = _market([100.0, 100.0, 100.0, 99.0, 98.0, 98.0])
    long_stats = evaluate.simulate_schedule(
        long_frame,
        _schedule(signal=1, entry=2, exit=4, side=1),
        start="2023-01-01",
        end="2023-01-02",
        cfg=_cfg(),
    )
    short_stats = evaluate.simulate_schedule(
        short_frame,
        _schedule(signal=1, entry=2, exit=4, side=-1, branch="fade"),
        start="2023-01-01",
        end="2023-01-02",
        cfg=_cfg(),
    )
    assert long_stats["absolute_return_pct"] == pytest.approx(1.0)
    assert short_stats["absolute_return_pct"] == pytest.approx(1.0)


def test_cagr_uses_full_wall_clock_split() -> None:
    frame = _market([100.0, 100.0, 100.0, 105.0, 110.0, 110.0])
    frame["date"] = pd.date_range("2020-12-31 23:30", periods=len(frame), freq="5min")
    stats = evaluate.simulate_schedule(
        frame,
        _schedule(signal=1, entry=2, exit=4),
        start="2020-01-01",
        end="2021-01-01",
        cfg=_cfg(),
    )
    years = 366.0 / 365.25
    expected = (1.05 ** (1.0 / years) - 1.0) * 100.0
    assert stats["cagr_pct"] == pytest.approx(expected)


def test_weekly_cluster_sign_flip_is_utc_monday_anchored_and_deterministic() -> None:
    returns = [0.01, 0.02, 0.0]
    dates = [
        "2023-01-01 23:55:00+00:00",
        "2023-01-02 00:00:00+00:00",
        "2023-01-02 00:05:00+00:00",
    ]
    first = evaluate.weekly_cluster_sign_flip(
        returns, dates, permutations=1_000, seed=20_260_714
    )
    second = evaluate.weekly_cluster_sign_flip(
        returns, dates, permutations=1_000, seed=20_260_714
    )
    assert first == second
    assert first["cluster_count"] == 2
    assert first["observed_mean_return"] == pytest.approx(0.01)
    assert 0.0 < first["p_value_one_sided"] < 1.0


def test_weekly_cluster_sign_flip_empty_and_negative_edges() -> None:
    empty = evaluate.weekly_cluster_sign_flip(
        [], [], permutations=32, seed=20_260_714
    )
    negative = evaluate.weekly_cluster_sign_flip(
        [-0.01], ["2023-01-02"], permutations=1_000, seed=20_260_714
    )
    assert empty["p_value_one_sided"] == 1.0
    assert empty["cluster_count"] == 0
    assert negative["p_value_one_sided"] > 0.5


def _window_metrics(
    *,
    absolute_return: float = 10.0,
    ratio: float = 4.0,
    mdd: float = 5.0,
    trades: int = 120,
    p_value: float = 0.05,
) -> dict[str, object]:
    return {
        "absolute_return_pct": absolute_return,
        "cagr_to_strict_mdd": ratio,
        "strict_mdd_pct": mdd,
        "trade_count": trades,
        "weekly_cluster_sign_flip": {"p_value_one_sided": p_value},
    }


def _candidate(name: str, *, ratio: float = 4.0) -> dict[str, object]:
    item: dict[str, object] = {
        "candidate": {"name": name},
        "windows": {
            "train": _window_metrics(ratio=ratio),
            "select2023": _window_metrics(ratio=ratio),
            "select2023_h1": _window_metrics(trades=60),
            "select2023_h2": _window_metrics(trades=60),
        },
    }
    item["qualification"] = evaluate._qualification(item)
    return item


def test_selection_rule_rejects_failures_and_breaks_ties_by_name() -> None:
    failing = _candidate("bad")
    failing["windows"]["select2023"]["cagr_to_strict_mdd"] = 2.99
    failing["qualification"] = evaluate._qualification(failing)
    alpha = _candidate("alpha")
    beta = _candidate("beta")
    result = evaluate._select_candidate([failing, beta, alpha])
    assert failing["qualification"]["qualifies"] is False
    assert result["selected_candidate"] == "alpha"
    assert result["rejected"] is False
