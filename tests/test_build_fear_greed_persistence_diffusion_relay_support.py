import numpy as np
import pandas as pd

from training import build_fear_greed_persistence_diffusion_relay_support as support


def frame():
    return pd.DataFrame(
        {
            "persistent_3d": [True, True, True, True],
            "persistent_2d": [True, True, True, True],
            "signal_valid": [True, True, True, True],
            "btc_valid": [True, True, True, True],
            "cumulative_change_3d": [9.0, -8.0, 7.0, -6.0],
            "persistence_magnitude_rank": [0.8, 0.9, 0.5, 0.8],
            "cumulative_change_2d": [6.0, -5.0, 4.0, -3.0],
            "persistence_2d_rank": [0.8, 0.9, 0.8, 0.8],
            "btc_variation_rank": [0.7, 0.8, 0.9, 0.4],
        }
    )


def test_primary_follows_persistent_sentiment_direction():
    active, side = support.conditions(frame(), "primary")
    assert active.tolist() == [True, True, False, False]
    assert side[active].tolist() == [1.0, -1.0]


def test_controls_are_diagnostic_only():
    data = frame()
    assert support.CONTROLS == (
        "no_volatility_gate",
        "no_persistence_magnitude_tail",
        "two_day_persistence",
        "one_day_stale_persistence",
        "direction_flip",
    )
    assert support.conditions(data, "no_volatility_gate")[0].tolist() == [
        True,
        True,
        False,
        True,
    ]
    assert support.conditions(data, "no_persistence_magnitude_tail")[0].tolist() == [
        True,
        True,
        True,
        False,
    ]
    active, side = support.conditions(data, "direction_flip")
    assert side[active].tolist() == [-1.0, 1.0]


def test_causal_rank_excludes_current_value():
    values = pd.Series(np.arange(91, dtype=float))
    ranks = support.strict_prior_midrank(values, 180, 90)
    assert ranks.iloc[90] == 1.0


def test_builder_binds_sources_and_seals_outcomes():
    assert support.sha(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    assert support.sha(support.SENTIMENT) == support.SENTIMENT_SHA
    assert support.sha(support.PRICE) == support.PRICE_SHA
    source = support.Path(support.__file__).read_text()
    assert '"postentry_return_pnl_execution_price_opened": False' in source
    assert '"gross9_rows_opened": False' in source
