import hashlib
import json

import numpy as np
import pandas as pd

from training import (
    build_high_volatility_relative_daily_volume_continuation_relay_support as support,
)


def _bars(days: int, amount: float = 1.0) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=days * 6, freq="4h", tz="UTC")
    return pd.DataFrame(
        {"valid_bar": True, "bar_volume": amount, "bar_quote_volume": amount * 100},
        index=index,
    )


def test_relative_daily_volume_uses_only_prior_same_slot_days(monkeypatch):
    monkeypatch.setitem(support.P, "relative_daily_volume_days", 2)
    bars = _bars(3)
    bars.loc[bars.index[-6:], "bar_volume"] = [3, 1, 1, 1, 1, 1]

    result = support.relative_daily_volume(bars, "bar_volume")

    assert result.ratio.iloc[:12].isna().all()
    assert result.cumulative.iloc[12] == 3.0
    assert result.average.iloc[12] == 1.0
    assert result.ratio.iloc[12] == 3.0
    assert bool(result.onset.iloc[12])
    assert result.average.iloc[13] == 2.0
    assert result.ratio.iloc[13] == 2.0
    assert not bool(result.onset.iloc[13])


def test_relative_daily_volume_resets_history_after_invalid_slot(monkeypatch):
    monkeypatch.setitem(support.P, "relative_daily_volume_days", 2)
    bars = _bars(4)
    bars.loc[bars.index[13], "valid_bar"] = False

    result = support.relative_daily_volume(bars, "bar_volume")

    assert np.isnan(result.ratio.iloc[13])
    assert result.ratio.iloc[18:].isna().all()


def test_prior_rank_excludes_current_caps_and_resets(monkeypatch):
    monkeypatch.setitem(support.P, "minimum_variation_history_slots", 2)
    monkeypatch.setitem(support.P, "variation_history_slots", 3)
    values = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan, 5.0, 6.0, 7.0])
    valid = pd.Series([True, True, True, True, False, True, True, True])

    ranks = support.prior_rank(values, valid)

    assert np.isnan(ranks.iloc[1])
    assert ranks.iloc[2] == 1.0
    assert ranks.iloc[3] == 1.0
    assert np.isnan(ranks.iloc[5]) and np.isnan(ranks.iloc[6])
    assert ranks.iloc[7] == 1.0


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_valid": [True, True, True, True],
            "day_return": [0.1, -0.1, 0.2, -0.2],
            "rdv": [1.2, 0.9, 1.3, 1.4],
            "quote_rdv": [0.9, 1.2, 1.3, 0.8],
            "onset": [True, False, True, False],
            "quote_onset": [False, True, True, False],
            "variation_rank": [0.8, 0.8, 0.4, 0.8],
        }
    )


def test_primary_and_all_frozen_controls():
    panel = _panel()
    primary, side, _ = support.active(panel)
    assert primary.tolist() == [True, False, False, False]
    assert side[primary].tolist() == [1]
    assert support.active(panel, "no_variation_gate")[0].tolist() == [True, False, True, False]
    assert support.active(panel, "no_rdv_onset_gate")[0].tolist() == [True, True, False, True]
    assert support.active(panel, "quote_volume_rdv")[0].tolist() == [False, True, False, False]
    stale, stale_side, _ = support.active(panel, "one_slot_stale_onset")
    assert stale.tolist() == [False, True, False, False]
    assert stale_side[stale].tolist() == [1]
    flipped, flipped_side, _ = support.active(panel, "direction_flip")
    assert flipped.equals(primary) and flipped_side[flipped].tolist() == [-1]
    forced, forced_side, _ = support.active(panel, "forced_long")
    assert forced.equals(primary) and forced_side[forced].eq(1).all()


def test_source_builder_is_outcome_and_gross9_blind_and_hash_bound():
    lowered = support.QUERY.lower()
    assert "volume,quote_asset_volume" in lowered
    assert "funding" not in lowered and "gross9" not in lowered
    assert support.PREREG_SHA == hashlib.sha256(
        support.prereg.DEFAULT_OUTPUT.read_bytes()
    ).hexdigest()
    value = {"한글": "누적거래량"}
    expected = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    assert support.canonical_hash(value) == expected
    assert b"\\u" not in support.json_bytes(value)
