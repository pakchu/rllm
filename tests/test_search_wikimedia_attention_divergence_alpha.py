from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import search_wikimedia_attention_divergence_alpha as wad
from training.preregister_wikimedia_attention_divergence_alpha import Policy


def test_lagged_robust_zscore_is_prefix_invariant_and_excludes_current() -> None:
    values = pd.Series(np.arange(1.0, 101.0))
    baseline = wad.lagged_robust_zscore(values, window=20, minimum=10)
    changed = values.copy()
    changed.iloc[-1] = -1_000_000.0
    replay = wad.lagged_robust_zscore(changed, window=20, minimum=10)
    pd.testing.assert_series_equal(baseline.iloc[:-1], replay.iloc[:-1])
    assert baseline.iloc[-2] == replay.iloc[-2]
    assert baseline.iloc[-1] != replay.iloc[-1]


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03"]),
            "anchor_date": pd.to_datetime(
                ["2022-01-03 12:05", "2022-01-04 12:05", "2022-01-05 12:05"]
            ),
            "broad_attention_z": [2.5, 0.0, 1.2],
            "bitcoin_share_z": [0.0, 0.0, 2.5],
            "price_return_1d": [0.05, -0.05, 0.01],
            "price_return_3d": [0.01, 0.01, -0.09],
        }
    )


def test_policy_families_use_preregistered_direction_logic() -> None:
    features = _feature_frame()
    broad = wad.policy_events(
        features, Policy("broad_attention_reversal", 2.0, 1, 0.04, 1)
    )
    assert broad["side"].tolist() == [-1]
    silent = wad.policy_events(
        features, Policy("silent_impulse_continuation", 0.0, 1, 0.04, 1)
    )
    assert silent["side"].tolist() == [-1]
    share = wad.policy_events(
        features, Policy("bitcoin_share_reversal", 2.0, 3, 0.08, 1)
    )
    assert share["side"].tolist() == [1]


def test_d_plus_two_anchor_enters_at_next_five_minute_open() -> None:
    dates = pd.date_range("2022-01-01", periods=5 * 288, freq="5min")
    market = pd.DataFrame(
        {
            "date": dates,
            "open": np.full(len(dates), 100.0),
            "high": np.full(len(dates), 101.0),
            "low": np.full(len(dates), 99.0),
            "close": np.full(len(dates), 100.0),
        }
    )
    funding = pd.DataFrame(
        {"date": pd.Series(dtype="datetime64[ns]"), "funding_rate": pd.Series(dtype=float)}
    )
    cfg = wad.ExecutionConfig(
        input_csv="",
        metrics_csv="",
        funding_csv="",
        output="",
        manifest_output="",
    )
    engine = wad.ExecutionEngine(market, funding, cfg)
    events = pd.DataFrame(
        {
            "observation_date": [pd.Timestamp("2022-01-01")],
            "anchor_date": [pd.Timestamp("2022-01-03 12:05")],
            "side": [1],
        }
    )
    trades = wad.build_schedule(
        engine,
        events,
        Policy("broad_attention_reversal", 2.0, 1, 0.04, 1),
        start="2022-01-01",
        end="2022-01-06",
    )
    assert len(trades) == 1
    assert engine.dates.iloc[trades[0].entry_position] == pd.Timestamp("2022-01-03 12:10")


def _metric(ret: float = 10.0, ratio: float = 3.0, trades: int = 10, mdd: float = 5.0) -> dict:
    return {
        "absolute_return_pct": ret,
        "cagr_to_strict_mdd": ratio,
        "trades": trades,
        "strict_mdd_pct": mdd,
    }


def test_selection_gates_fail_closed_on_any_bad_year_or_inversion() -> None:
    stats = {
        "fit_2020": _metric(),
        "fit_2021": _metric(),
        "selection_2022": _metric(),
        "combined_2020_2022": _metric(trades=30),
    }
    assert all(wad.selection_gates(stats, _metric(ratio=2.0), _metric(ret=-1.0)).values())
    broken = {name: dict(value) for name, value in stats.items()}
    broken["fit_2021"]["absolute_return_pct"] = -0.01
    assert not all(wad.selection_gates(broken, _metric(ratio=2.0), _metric(ret=-1.0)).values())
    assert not wad.selection_gates(stats, _metric(ratio=2.0), _metric(ret=0.01))[
        "inverted_side_combined_nonpositive"
    ]


def test_rank_key_prioritizes_worst_calendar_year() -> None:
    base = {
        "policy": {
            "family": "broad_attention_reversal",
            "attention_threshold": 2.0,
            "price_horizon_days": 1,
            "price_threshold": 0.04,
            "hold_days": 1,
        },
        "stats": {
            "fit_2020": _metric(ratio=2.0),
            "fit_2021": _metric(ratio=2.0),
            "selection_2022": _metric(ratio=2.0),
            "combined_2020_2022": _metric(ratio=10.0),
        },
    }
    stable = {**base, "stats": {name: dict(value) for name, value in base["stats"].items()}}
    unstable = {**base, "stats": {name: dict(value) for name, value in base["stats"].items()}}
    stable["stats"]["combined_2020_2022"]["cagr_to_strict_mdd"] = 3.0
    unstable["stats"]["fit_2020"]["cagr_to_strict_mdd"] = -1.0
    assert wad.rank_key(stable) < wad.rank_key(unstable)


def test_rank_key_uses_ascending_policy_tuple_after_frozen_metrics() -> None:
    def trial(family: str) -> dict:
        return {
            "policy": {
                "family": family,
                "attention_threshold": 2.0,
                "price_horizon_days": 1,
                "price_threshold": 0.04,
                "hold_days": 1,
            },
            "stats": {
                "fit_2020": _metric(ratio=2.0),
                "fit_2021": _metric(ratio=2.0),
                "selection_2022": _metric(ratio=2.0),
                "combined_2020_2022": _metric(ratio=2.0),
            },
        }

    assert wad.rank_key(trial("a_family")) < wad.rank_key(trial("z_family"))


def test_strict_mdd_includes_entry_and_exit_costs() -> None:
    cfg = wad.ExecutionConfig(
        input_csv="",
        metrics_csv="",
        funding_csv="",
        output="",
        manifest_output="",
        leverage=1.0,
        fee_rate=0.01,
        slippage_rate=0.0,
    )
    trade = wad.Trade(
        signal_position=0,
        entry_position=1,
        exit_position=2,
        side=1,
        gross_return=0.0,
        price_factor=1.0,
        funding_factor=1.0,
        funding_debit_factor=1.0,
        favorable_price_factor=1.0,
        adverse_price_factor=0.9,
        entry_date="2022-01-01",
    )
    stats = wad.strict_equity_stats(
        [trade], start="2022-01-01", end="2023-01-01", cfg=cfg
    )
    assert stats["strict_mdd_pct"] == pytest.approx(
        (1.0 - 0.99 * 0.9 * 0.99) * 100.0
    )
