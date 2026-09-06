from pathlib import Path

import pandas as pd

from training import build_high_volatility_london_fix_dollar_impulse_relay_support as support


def test_hvlfx_source_builder_is_outcome_blind_and_hash_bound():
    source = Path(support.__file__).read_text()
    assert support.PREREG_SHA == "2908e8cd471c185625e2b21e6a45289ae89b89a3eaa3bfb841eaa732ad6c6a0a"
    assert "bars_polygon" in support.FX_QUERY
    assert "GBPUSD" in support.FX_QUERY and "EURUSD" in support.FX_QUERY
    assert "postentry_return_pnl_execution_price_opened" in source
    assert "gross9_rows_opened" in source


def test_hvlfx_strict_prior_rank_excludes_current_value():
    values = pd.Series(range(130), dtype=float)
    ranks = support.strict_prior_rank(values)
    assert ranks.iloc[:126].isna().all()
    assert ranks.iloc[126] == 1.0


def test_hvlfx_controls_and_clock_are_frozen():
    assert support.CONTROLS == (
        "no_volatility_gate",
        "eurusd_only",
        "one_session_stale_fix_impulse",
        "direction_flip",
    )
    assert support.MINIMUM_EVENTS == {"train": 8, "test": 12, "eval": 12, "final": 8}


def test_primary_requires_common_fix_direction():
    features = pd.DataFrame({
        "gbpusd_return": [0.01, 0.01, -0.01],
        "eurusd_return": [0.02, -0.02, -0.02],
        "common_impulse": [0.015, -0.005, -0.015],
        "btc_variation_rank": [0.8, 0.8, 0.8],
    })
    assert support.signal(features, "primary").tolist() == [1, 0, -1]
    assert support.signal(features, "direction_flip").tolist() == [-1, 0, 1]
