import json

import numpy as np
import pandas as pd

from training import build_high_volatility_kospi_asymmetric_transition_relay_support as b


def test_yahoo_native_session_normalization():
    timestamps = [int(pd.Timestamp(d, tz="Asia/Seoul").timestamp()) for d in ("2024-01-02", "2024-01-03")]
    result = {
        "meta": {"symbol": "^KS11", "exchangeTimezoneName": "Asia/Seoul"},
        "timestamp": timestamps,
        "indicators": {"quote": [{
            "volume": [10, 20], "close": [2600, 2610], "open": [2590, 2600],
            "low": [2580, 2595], "high": [2610, 2620],
        }]},
    }
    stable, frame, meta = b.normalize_yahoo_chart(
        json.dumps({"chart": {"result": [result], "error": None}}).encode()
    )
    assert json.loads(stable)["meta"]["symbol"] == "^KS11"
    assert frame.cash_close_time.dt.strftime("%Y-%m-%dT%H:%MZ").tolist() == [
        "2024-01-02T06:30Z", "2024-01-03T06:30Z",
    ]
    assert meta["native_close_local"] == "15:30"


def test_strict_prior_midrank_excludes_current():
    ranks = b.strict_prior_midrank(pd.Series([1.0, 2.0, 3.0]), lookback=2, minimum=2)
    assert ranks.iloc[:2].isna().all()
    assert ranks.iloc[2] == 1.0


def _features():
    return pd.DataFrame({
        "kospi_return": [0.4, -0.5, -0.6, 0.2],
        "prior_kospi_return": [np.nan, 0.4, -0.5, -0.6],
        "kospi_shock_rank": [0.9] * 4,
        "btc_variation_rank": [0.9] * 4,
    })


def test_transition_signal_and_controls_are_frozen():
    frame = _features()
    active, side = b._signal(frame, "primary")
    assert active.tolist() == [False, True, False, True]
    assert side.tolist() == [0, -1, 0, 1]
    stale, stale_side = b._signal(frame, "one_session_stale_transition")
    assert stale.tolist() == [False, False, True, False]
    assert stale_side.tolist() == [0, 0, -1, 0]
    _, flipped = b._signal(frame, "direction_flip")
    assert flipped.tolist() == [0, 1, 0, -1]
    _, forced = b._signal(frame, "same_clock_forced_long")
    assert forced.tolist() == [0, 1, 0, 1]
    assert len(b.CONTROLS) == 6


def test_clock_native_latency_half_open_and_support_fail_closed():
    frame = pd.DataFrame({
        "kospi_session_date": pd.to_datetime(["2023-07-03", "2023-07-04"]),
        "cash_close_time": pd.to_datetime(["2023-07-03T06:30Z", "2023-07-04T06:30Z"]),
        "feature_available_time": pd.to_datetime(["2023-07-03T06:35Z", "2023-07-04T06:35Z"]),
        "kospi_return": [-0.4, 0.5], "prior_kospi_return": [0.2, -0.4],
        "kospi_shock_rank": [0.9, 0.9], "btc_realized_variation": [0.01, 0.02],
        "btc_variation_rank": [0.9, 0.9],
    })
    clock = b.build_clock(frame)
    assert clock.entry_time.dt.strftime("%Y-%m-%dT%H:%MZ").tolist() == [
        "2023-07-03T06:40Z", "2023-07-04T06:40Z",
    ]
    assert b.support_stats(pd.DataFrame(columns=b.CLOCK_COLUMNS), "train")["events"] == 0
    assert b.MINIMUM_EVENTS == {"train": 8, "test": 12, "eval": 12, "final": 8}
