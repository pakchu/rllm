import hashlib

import numpy as np
import pandas as pd

from training import build_high_volatility_trade_arrival_memory_continuation_support as support


def test_lag_one_correlation_detects_memory_and_rejects_constant_path():
    persistent = np.repeat(np.arange(120, dtype=float), 4)
    assert support.lag_one_correlation(persistent) > 0.99
    assert np.isnan(support.lag_one_correlation(np.ones(480)))
    assert np.isnan(support.lag_one_correlation(np.ones(479)))


def test_prepare_requires_integer_trade_counts_and_coherent_prices():
    frame = pd.DataFrame({
        "ts": ["2023-01-01T00:00:00Z"], "open": [100.0], "high": [101.0],
        "low": [99.0], "close": [100.5], "quote_asset_volume": [10.0],
        "number_of_trades": [3.0],
    })
    assert bool(support.prepare(frame).row_valid.iloc[0])
    frame.loc[0, "number_of_trades"] = 3.5
    assert not bool(support.prepare(frame).row_valid.iloc[0])


def test_prior_rank_excludes_current_and_uses_midrank(monkeypatch):
    monkeypatch.setitem(support.P, "minimum_prior_blocks", 2)
    monkeypatch.setitem(support.P, "prior_blocks", 3)
    ranked = support.prior_rank(pd.Series([1.0, 1.0, 2.0, 0.0]))
    assert np.isnan(ranked.iloc[0]) and np.isnan(ranked.iloc[1])
    assert ranked.iloc[2] == 1.0
    assert ranked.iloc[3] == 0.0


def test_onset_skips_invalid_rows_when_finding_previous_valid_state():
    eligible = pd.Series([False, True, False, True])
    valid = pd.Series([True, True, False, True])
    assert support.previous_valid_onset(eligible, valid).tolist() == [False, True, False, False]


def panel_for_controls():
    times = pd.date_range("2023-07-01", periods=4, freq="8h", tz="UTC")
    return pd.DataFrame({
        "decision_time": times, "feature_available_time": times,
        "source_valid": [True] * 4, "minute_count": [480] * 4,
        "arrival_memory": [0.1, 0.8, 0.8, 0.1], "memory_rank": [0.1, 0.8, 0.8, 0.1],
        "quote_volume_memory": [0.8, 0.1, 0.1, 0.8], "quote_volume_memory_rank": [0.8, 0.1, 0.1, 0.8],
        "realized_variation": [1.0] * 4, "variation_rank": [0.8] * 4,
        "completed_return": [-0.01, 0.02, 0.03, -0.02], "eligible": [False, True, True, False],
        "onset": [False, True, False, False],
    })


def test_primary_and_quote_volume_control_have_frozen_distinct_onsets():
    panel = panel_for_controls()
    primary, side, _ = support.active(panel)
    volume, _, _ = support.active(panel, "quote_volume_memory")
    assert primary.tolist() == [False, True, False, False]
    assert volume.tolist() == [False, False, False, True]
    assert side.tolist() == [-1, 1, 1, -1]


def test_schema_is_outcome_blind_and_gzip_is_deterministic():
    forbidden = {"pnl", "funding", "execution_price", "gross9"}
    assert not {item.lower() for item in (*support.PANEL_COLUMNS, *support.CLOCK_COLUMNS)}.intersection(forbidden)
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    first = support.csv_gz(frame); second = support.csv_gz(frame)
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_evaluator_is_bound_to_frozen_preregistration():
    assert support.sha(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    assert support.P["memory_rank_min"] == 0.75
    assert support.CONTROLS == tuple(support.REG["diagnostic_controls"]["names"])
