from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from training import preregister_metaorder_fragmentation_impact_curvature as mfic


def _frame(flows: list[float], responses: list[float]) -> pd.DataFrame:
    rows = len(flows)
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "signed_quote_notional": flows,
            "flow_coherence": np.full(rows, 0.5),
            "normalized_effective_event_count": np.full(rows, 1.0),
            "sign_flip_rate": np.zeros(rows),
            "max_same_sign_run_share": np.ones(rows),
            "signed_price_response": responses,
            "agg_trade_count": np.full(rows, 100),
            "close": 100.0 + np.arange(rows) * 0.1,
            "quarantined": np.zeros(rows, dtype=bool),
        }
    )


def _loose_config() -> mfic.Config:
    return replace(
        mfic.Config(),
        curvature_threshold=1e-6,
        persistence_floor=0.0,
        coherence_floor=0.0,
        hidden_metaorder_quantile=0.0,
        hidden_metaorder_baseline_bars=4,
        hidden_metaorder_baseline_min_periods=1,
        minimum_agg_trade_count=1,
    )


def test_quarantine_extends_forward_without_backfilling() -> None:
    available = pd.Series([True, True, False, True, True, True, True])
    gap_day = pd.Series([False, False, False, False, False, False, True])
    output = mfic.quarantine_mask(available, gap_day, post_gap_bars=2)
    assert output.tolist() == [False, False, True, True, True, False, True]


def test_five_minute_grid_rejects_missing_rows() -> None:
    dates = pd.Series(pd.to_datetime(["2023-01-01 00:00", "2023-01-01 00:10"]))
    with np.testing.assert_raises_regex(ValueError, "complete 5-minute grid"):
        mfic._assert_five_minute_grid(dates, label="test")


def test_positive_impact_curvature_routes_with_metaorder_direction() -> None:
    frame = _frame(
        [10.0] * 12,
        [
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.002,
            0.002,
            0.005,
            0.006,
        ],
    )
    candidate = mfic.Candidate("test", 8, 2, 1, 1)
    signal = mfic.compute_mfic(frame, candidate, _loose_config())
    assert signal.loc[11, "curvature"] > 0.0
    assert signal.loc[11, "branch"] == "continuation"
    assert signal.loc[11, "side"] == 1


def test_negative_curvature_with_extension_routes_as_fade() -> None:
    frame = _frame(
        [10.0] * 12,
        [
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.006,
            0.005,
            0.001,
            0.0001,
        ],
    )
    candidate = mfic.Candidate("test", 8, 2, 1, 2)
    signal = mfic.compute_mfic(frame, candidate, _loose_config())
    assert signal.loc[11, "curvature"] < 0.0
    assert signal.loc[11, "extension"] > 0.0
    assert signal.loc[11, "branch"] == "fade"
    assert signal.loc[11, "side"] == -1


def test_feature_prefix_is_invariant_to_future_changes() -> None:
    frame = _frame([10.0] * 12, np.linspace(0.001, 0.004, 12).tolist())
    candidate = mfic.Candidate("test", 4, 1, 1, 1)
    baseline = mfic.compute_mfic(frame, candidate, _loose_config())
    changed = frame.copy()
    changed.loc[10:, "signed_quote_notional"] = -1_000_000.0
    changed.loc[10:, "signed_price_response"] = -1.0
    replay = mfic.compute_mfic(changed, candidate, _loose_config())
    pd.testing.assert_frame_equal(
        baseline.loc[:9],
        replay.loc[:9],
        check_dtype=True,
    )


def test_baseline_excludes_scores_with_quarantined_lookbacks() -> None:
    frame = _frame([10.0] * 20, np.linspace(0.001, 0.004, 20).tolist())
    frame.loc[4, "quarantined"] = True
    candidate = mfic.Candidate("test", 8, 2, 1, 1)
    cfg = replace(
        _loose_config(),
        hidden_metaorder_quantile=0.5,
        hidden_metaorder_baseline_bars=20,
    )
    baseline = mfic.compute_mfic(frame, candidate, cfg)
    changed = frame.copy()
    changed.loc[4, "signed_quote_notional"] = 1_000_000.0
    replay = mfic.compute_mfic(changed, candidate, cfg)
    assert baseline.loc[13, "hidden_metaorder_baseline"] == replay.loc[
        13, "hidden_metaorder_baseline"
    ]


def test_nonoverlap_schedule_skips_conflicts_and_quarantined_holds() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=10, freq="5min"),
            "quarantined": [False, False, False, False, False, False, False, True, False, False],
        }
    )
    signal = pd.DataFrame(
        {
            "side": [0, 1, 1, 0, 0, -1, 0, 1, 0, 0],
            "hold_bars": [0, 2, 2, 0, 0, 1, 0, 1, 0, 0],
            "branch": [
                "none",
                "continuation",
                "continuation",
                "none",
                "none",
                "fade",
                "none",
                "continuation",
                "none",
                "none",
            ],
        }
    )
    schedule = mfic.nonoverlapping_schedule(signal, frame)
    assert schedule["signal_position"].tolist() == [1]
    assert schedule["side"].tolist() == [1]


def test_nonoverlap_schedule_skips_split_crossing_trades() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-06-30 23:50", periods=6, freq="5min"),
            "quarantined": np.zeros(6, dtype=bool),
        }
    )
    signal = pd.DataFrame(
        {
            "side": [0, 1, 0, 1, 0, 0],
            "hold_bars": [0, 1, 0, 1, 0, 0],
            "branch": ["none", "continuation", "none", "fade", "none", "none"],
        }
    )
    h1 = mfic.nonoverlapping_schedule(
        signal, frame, start="2023-01-01", end="2023-07-01"
    )
    h2 = mfic.nonoverlapping_schedule(
        signal, frame, start="2023-07-01", end="2024-01-01"
    )
    assert h1.empty
    assert h2["signal_position"].tolist() == [3]


def test_source_gap_days_detect_internal_gaps() -> None:
    manifest = {
        "months": [
            {
                "archives": [
                    {
                        "date": "2023-01-01",
                        "first_agg_trade_id": 1,
                        "last_agg_trade_id": 10,
                        "agg_trade_rows": 9,
                    },
                    {
                        "date": "2023-01-02",
                        "first_agg_trade_id": 11,
                        "last_agg_trade_id": 20,
                        "agg_trade_rows": 10,
                    },
                ]
            }
        ]
    }
    assert mfic._source_gap_days(manifest) == {"2023-01-01"}


def test_nonoverlap_schedule_handles_positions_beyond_int16() -> None:
    rows = 40_000
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "quarantined": np.zeros(rows, dtype=bool),
        }
    )
    signal = pd.DataFrame(
        {
            "side": np.zeros(rows, dtype=np.int8),
            "hold_bars": np.zeros(rows, dtype=np.int16),
            "branch": pd.Series("none", index=range(rows), dtype="string"),
        }
    )
    signal.loc[35_000, ["side", "hold_bars", "branch"]] = [1, 6, "continuation"]
    schedule = mfic.nonoverlapping_schedule(signal, frame)
    assert schedule.loc[0, "signal_position"] == 35_000
    assert schedule.loc[0, "exit_position"] == 35_007
