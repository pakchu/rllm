import json

import numpy as np
import pandas as pd

from training import build_high_volatility_dominant_volume_bar_relay_support as support


THRESHOLDS = {"dominant_share_q85": 0.8, "range_vol_q60": 1.0}


def _scores(times, last, block, *, first=None, share=None, volatility=None):
    count = len(times)
    return pd.DataFrame(
        {
            "decision_bar_time": pd.to_datetime(times, utc=True),
            "dominant_share": [0.9] * count if share is None else share,
            "dominant_return_last": last,
            "dominant_return_first": last if first is None else first,
            "block_return": block,
            "range_vol": [2.0] * count if volatility is None else volatility,
        }
    )


def test_score_anchor_uses_last_maximum_volume_tie_and_validates_source():
    dates = pd.date_range("2023-06-01T00:00:00Z", periods=144, freq="5min")
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": np.full(144, 100.0),
            "high": np.full(144, 111.0),
            "low": np.full(144, 99.0),
            "close": np.full(144, 100.0),
            "quote_asset_volume": np.ones(144),
        }
    )
    frame.loc[72 + 10, ["open", "close"]] = [100.0, 101.0]
    frame.loc[72 + 20, ["open", "close"]] = [101.0, 100.0]
    frame.loc[72 + 10, "quote_asset_volume"] = 10.0
    frame.loc[72 + 20, "quote_asset_volume"] = 10.0
    frame.loc[143, "close"] = 110.0

    scored = support._score_anchor(frame)

    assert scored["dominant_share"] == 10.0 / 90.0
    assert scored["dominant_return_first"] > 0
    assert scored["dominant_return_last"] < 0
    assert scored["block_return"] > 0
    assert scored["range_vol"] == 12.0 / 105.0

    frame.loc[100, "quote_asset_volume"] = -1.0
    assert all(np.isnan(value) for value in support._score_anchor(frame).values())


def test_conditions_apply_common_sign_same_side_onset_and_all_controls():
    frame = _scores(
        pd.date_range("2023-07-01T00:55:00Z", periods=6, freq="1h"),
        [1, 1, -1, -1, -1, -1],
        [1, 1, -1, -1, -1, 1],
        first=[1, -1, -1, -1, 1, 1],
        volatility=[2, 2, 2, 0.5, 2, 2],
    )

    active, side = support.conditions(frame, THRESHOLDS)
    assert active.tolist() == [True, False, True, False, True, False]
    assert side[active].tolist() == [1, -1, -1]

    active, _ = support.conditions(frame, THRESHOLDS, "no_volatility_gate")
    assert active.tolist() == [True, False, True, False, False, False]
    active, _ = support.conditions(frame, THRESHOLDS, "first_dominant_tie_break")
    assert active.tolist() == [True, False, True, False, False, True]
    active, side = support.conditions(frame, THRESHOLDS, "block_direction_only")
    assert active.tolist() == [True, False, True, False, True, True]
    active, side = support.conditions(frame, THRESHOLDS, "direction_flip")
    assert side[active].tolist() == [-1, 1, 1]


def test_clock_has_split_safe_global_half_open_twelve_hour_reservation():
    frame = _scores(
        pd.date_range("2023-07-01T00:55:00Z", periods=14, freq="1h"),
        [1, -1] * 7,
        [1, -1] * 7,
    )
    clock = support.build_clock(frame, THRESHOLDS)
    assert clock.entry_time.tolist() == [
        pd.Timestamp("2023-07-01T01:00:00Z"),
        pd.Timestamp("2023-07-01T13:00:00Z"),
    ]
    assert (clock.exit_time - clock.entry_time).eq(pd.Timedelta(hours=12)).all()

    boundary = _scores(
        ["2023-12-31T12:55:00Z", "2024-01-01T00:55:00Z"],
        [1, -1],
        [1, -1],
    )
    clock = support.build_clock(boundary, THRESHOLDS)
    assert clock.entry_time.tolist() == [pd.Timestamp("2024-01-01T01:00:00Z")]
    assert clock.split.tolist() == ["test"]


def test_run_materializes_primary_four_controls_scores_and_support_json(tmp_path, monkeypatch):
    scores = _scores(
        pd.date_range("2023-07-01T00:55:00Z", periods=14, freq="1h"),
        [1, -1] * 7,
        [1, -1] * 7,
    )
    monkeypatch.setattr(support, "SNAPSHOT", tmp_path / "sources" / "scores.csv.gz")
    monkeypatch.setattr(support, "CLOCK", tmp_path / "primary.csv.gz")
    monkeypatch.setattr(support, "CONTROL_DIR", tmp_path / "controls")
    monkeypatch.setattr(support, "RESULT", tmp_path / "support.json")
    monkeypatch.setattr(support, "load_combined_market", lambda: (pd.DataFrame(), {"mode": "test"}))
    monkeypatch.setattr(support, "score_snapshot", lambda _market: (scores, THRESHOLDS))

    first = support.run()
    second = support.run()

    assert first == second
    assert support.SNAPSHOT.exists() and support.CLOCK.exists()
    assert sorted(path.name for path in support.CONTROL_DIR.glob("*.csv.gz")) == [
        f"{name}.csv.gz" for name in sorted(support.CONTROLS)
    ]
    report = json.loads(support.RESULT.read_text())
    assert report["postentry_return_pnl_execution_price_opened"] is False
    assert report["gross9_rows_opened"] is False
    assert report["advance_to_economic_outcomes"] is False
    assert set(report["controls"]) == set(support.CONTROLS)
