import json

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_spot_execution_activity_leadership_relay_support as s
from training import preregister_high_volatility_spot_execution_activity_leadership_relay as prereg


def synthetic_block() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01T00:00:00Z", periods=96, freq="5min")
    phase = np.arange(96, dtype=float)
    spot_count = 100 + 20 * np.sin(phase / 5) + phase
    perpetual_return = np.r_[0.0001, 0.0002 + spot_count[:-1] * 1e-7]
    perpetual_count = 90 + 5 * np.cos(phase / 7)
    spot_return = 0.0003 + 0.0001 * np.sin(phase / 3)
    frames = []
    for venue, counts, returns in (
        ("spot", spot_count, spot_return),
        ("perpetual", perpetual_count, perpetual_return),
    ):
        opens = 100 + phase * 0.01
        closes = opens * np.exp(returns)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "venue": venue,
                    "open": opens,
                    "high": np.maximum(opens, closes) * 1.0001,
                    "low": np.minimum(opens, closes) * 0.9999,
                    "close": closes,
                    "execution_count": np.rint(counts),
                    "source_rows": 5,
                    "distinct_rows": 5,
                    "first_ts": dates,
                    "last_ts": dates + pd.Timedelta("4m"),
                    "coherent": True,
                }
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values(["date", "venue"]).reset_index(drop=True)


def test_preregistration_and_source_query_are_bound():
    assert s.PREREG_SHA == "f3718e4f6884cc5f972870df53d93e384f9167cd3f8ff048976e5774c2ae201c"
    assert s.sha(prereg.DEFAULT_OUTPUT) == s.PREREG_SHA
    assert json.loads(prereg.DEFAULT_OUTPUT.read_text()) == prereg.build()
    normalized = " ".join(s.QUERY.split()).lower()
    assert "from bars_binance_spot" in normalized
    assert "from bars_binance" in normalized
    assert "number_of_trades" in normalized
    assert all(term not in normalized for term in ("funding", "pnl", "gross9"))


def test_block_metrics_use_bidirectional_predictive_activity_channels():
    block = s.prepare(synthetic_block())
    metrics = s.block_metrics(block)
    spot = block[block.venue.eq("spot")].sort_values("date")
    perpetual = block[block.venue.eq("perpetual")].sort_values("date")
    expected_forward = np.corrcoef(
        np.log1p(spot.execution_count.to_numpy(float)[:-1]),
        perpetual.bar_return.to_numpy(float)[1:],
    )[0, 1]
    expected_reverse = np.corrcoef(
        np.log1p(perpetual.execution_count.to_numpy(float)[:-1]),
        spot.bar_return.to_numpy(float)[1:],
    )[0, 1]
    assert metrics["source_valid"] is True
    assert metrics["spot_to_perpetual_response"] == pytest.approx(expected_forward)
    assert metrics["perpetual_to_spot_response"] == pytest.approx(expected_reverse)
    assert metrics["leadership_margin"] == pytest.approx(abs(expected_forward) - abs(expected_reverse))


def test_source_contract_fails_closed():
    raw = synthetic_block()
    missing = raw.drop(raw.index[0])
    assert s.block_metrics(s.prepare(missing))["source_valid"] is False
    duplicate = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicate or unexpected source key"):
        s.prepare(duplicate)


def test_strict_prior_rank_excludes_current():
    values = pd.Series(range(181), dtype=float)
    ranks = s.strict_prior_midrank(values)
    assert ranks.iloc[:180].isna().all()
    assert ranks.iloc[180] == 1.0


def test_fresh_onset_side_and_diagnostic_controls():
    decisions = pd.date_range("2024-07-01T00:00:00Z", periods=4, freq="8h")
    frame = pd.DataFrame(
        {
            "decision_time": decisions,
            "feature_available_time": decisions,
            "source_valid": True,
            "spot_to_perpetual_response": [0.4] * 4,
            "perpetual_to_spot_response": [0.1] * 4,
            "leadership_margin": [0.3] * 4,
            "leadership_rank": [0.7, 0.8, 0.9, 0.7],
            "perpetual_realized_variation": [0.01] * 4,
            "variation_rank": [0.7] * 4,
            "completed_perpetual_return": [0.01] * 4,
            "direction_side": [1] * 4,
            "eligible": [False, True, True, False],
        },
        columns=s.PANEL_COLUMNS,
    )
    onset, side = s.active_and_side(frame)
    assert onset.tolist() == [False, True, False, False]
    assert side.tolist() == [1, 1, 1, 1]
    assert s.active_and_side(frame, "direction_flip")[1].tolist() == [-1] * 4
    assert s.active_and_side(frame, "forced_long")[1].tolist() == [1] * 4
