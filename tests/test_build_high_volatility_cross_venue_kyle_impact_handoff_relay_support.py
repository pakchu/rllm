import numpy as np
import pandas as pd

from training import build_high_volatility_cross_venue_kyle_impact_handoff_relay_support as support


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_valid": [True] * 6,
            "spot_impact": [.3, .4, .5, .2, .1, .4],
            "perp_impact": [.2, .2, .2, .3, .2, .2],
            "impact_handoff": [.2, .7, .8, -.4, -.7, .7],
            "handoff_rank": [.5, .85, .9, .15, .1, .85],
            "spot_signed_flow": [1., 2., 3., -2., -1., 2.],
            "perp_signed_flow": [1., 1., 2., -1., 1., 1.],
            "full_variation": [.1] * 6,
            "variation_rank": [.8, .8, .8, .8, .8, .8],
        }
    )


def test_primary_uses_false_to_true_spot_impact_handoff():
    active, side = support.conditions(frame(), "primary")
    assert active.tolist() == [False, True, False, False, False, True]
    assert side[active].tolist() == [1, 1]


def test_controls_are_diagnostic_only():
    candidate = frame()
    candidate.loc[1, "variation_rank"] = .4
    assert support.conditions(candidate, "no_variation_gate")[0].iloc[1]
    active, side = support.conditions(frame(), "direction_flip")
    assert side[active].tolist() == [-1, -1]
    assert support.conditions(frame(), "perpetual_impact_dominance")[0].iloc[3]


def test_venue_impact_uses_completed_five_minute_aggregates():
    minute = pd.DataFrame(
        {
            "open": np.full(480, 100.0),
            "close": np.tile(np.repeat([101.0, 99.0], 5), 48),
            "quote_asset_volume": np.full(480, 10.0),
            "taker_buy_quote": np.tile(np.repeat([7.0, 3.0], 5), 48),
        }
    )
    impact, signed_flow, returns = support._venue_metrics(minute)
    assert impact > 0
    assert abs(signed_flow) < 1e-12
    assert len(returns) == 96


def test_rank_excludes_current_observation():
    values = pd.Series(np.arange(181, dtype=float))
    ranks = support.strict_prior_midrank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0
