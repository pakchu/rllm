import hashlib
import json

import numpy as np
import pandas as pd

from training import (
    build_high_volatility_internal_bar_strength_extreme_reversal_relay_support as support,
)


def test_internal_bar_strength_formula_and_strict_zero_range_policy():
    high = pd.Series([110.0, 100.0, 110.0])
    low = pd.Series([90.0, 100.0, 90.0])
    close = pd.Series([94.0, 100.0, 106.0])
    valid = pd.Series([True, True, False])

    actual = support.internal_bar_strength(high, low, close, valid)

    assert actual.iloc[0] == 0.2
    assert np.isnan(actual.iloc[1])
    assert np.isnan(actual.iloc[2])


def test_prior_rank_excludes_current_caps_history_and_resets(monkeypatch):
    monkeypatch.setitem(support.P, "minimum_variation_history_days", 2)
    monkeypatch.setitem(support.P, "variation_history_days", 3)
    values = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan, 5.0, 6.0, 7.0])
    continuity = pd.Series([True, True, True, True, False, True, True, True])

    ranks = support.prior_rank(values, continuity)

    assert np.isnan(ranks.iloc[1])
    assert ranks.iloc[2] == 1.0
    assert ranks.iloc[3] == 1.0
    assert np.isnan(ranks.iloc[5])
    assert np.isnan(ranks.iloc[6])
    assert ranks.iloc[7] == 1.0


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_valid": [True] * 7,
            "ibs": [0.1, 0.9, 0.5, 0.25, 0.75, 0.1, 0.9],
            "variation_rank": [0.8, 0.8, 0.8, 0.8, 0.4, 0.4, 0.8],
            "feature_available_time": pd.date_range(
                "2024-01-01", periods=7, tz="UTC"
            ),
        }
    )


def test_primary_and_all_frozen_controls():
    panel = _panel()
    primary, side, _ = support.active(panel)
    assert primary.tolist() == [True, True, False, False, False, False, True]
    assert side[primary].tolist() == [1, -1, -1]

    no_gate, _, _ = support.active(panel, "no_variation_gate")
    assert no_gate.tolist() == [True, True, False, False, False, True, True]

    middle, middle_side, _ = support.active(panel, "middle_half_direction")
    assert middle.tolist() == [False, False, False, True, False, False, False]
    assert middle_side[middle].tolist() == [1]

    stale, stale_side, _ = support.active(panel, "one_day_stale_extreme")
    assert stale.tolist() == [False, True, True, False, False, False, False]
    assert stale_side[stale].tolist() == [1, -1]

    follow, follow_side, _ = support.active(panel, "direction_follow")
    assert follow.equals(primary)
    assert follow_side[follow].tolist() == [-1, 1, 1]

    forced, forced_side, _ = support.active(panel, "forced_long")
    assert forced.equals(primary)
    assert forced_side[forced].eq(1).all()


def test_daily_panel_uses_exact_1440_rows_and_completed_ohlc(monkeypatch):
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    end = start + pd.Timedelta("6d")
    monkeypatch.setattr(support, "START", start)
    monkeypatch.setattr(support, "END", end)
    monkeypatch.setitem(support.P, "variation_days", 2)
    monkeypatch.setitem(support.P, "minimum_variation_history_days", 2)
    monkeypatch.setitem(support.P, "variation_history_days", 3)

    rows = []
    closes = [92.0, 108.0, 94.0, 106.0, 93.0, 107.0]
    for day, daily_close in enumerate(closes):
        day_start = start + pd.Timedelta(days=day)
        for minute in range(1440):
            price = 100.0
            low = 90.0 if minute == 1 else price
            high = 110.0 if minute == 2 else price
            close = daily_close if minute == 1439 else price
            rows.append(
                {
                    "ts": day_start + pd.Timedelta(minutes=minute),
                    "open": price,
                    "high": max(high, close),
                    "low": min(low, close),
                    "close": close,
                }
            )
    panel = support.build_panel(pd.DataFrame(rows))

    assert panel.daily_open.eq(100.0).all()
    assert panel.daily_high.eq(110.0).all()
    assert panel.daily_low.eq(90.0).all()
    assert np.allclose(panel.ibs, [(value - 90.0) / 20.0 for value in closes])
    assert panel.feature_available_time.equals(panel.source_day + pd.Timedelta("1d"))
    assert panel.decision_time.equals(panel.feature_available_time)
    assert not panel.source_valid.iloc[:2].any()
    assert panel.source_valid.iloc[2:].all()


def test_source_builder_is_outcome_and_gross9_blind_and_hashes_utf8():
    lowered = support.QUERY.lower()
    assert "open,high,low,close" in lowered
    assert "funding" not in lowered
    assert "gross9" not in lowered
    assert support.PREREG_SHA == hashlib.sha256(
        support.prereg.DEFAULT_OUTPUT.read_bytes()
    ).hexdigest()
    value = {"한글": "알파"}
    expected = hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    assert support.canonical_hash(value) == expected
    assert b"\\u" not in support.json_bytes(value)
