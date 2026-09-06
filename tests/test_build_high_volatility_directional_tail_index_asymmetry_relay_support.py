import numpy as np
import pandas as pd

from training import build_high_volatility_directional_tail_index_asymmetry_relay_support as s


def test_tail_statistics_detects_heavier_positive_tail():
    returns = np.zeros(720)
    returns[:30] = np.geomspace(0.001, 0.05, 30)
    returns[30:60] = -np.linspace(0.001, 0.002, 30)
    open_ = np.repeat(100.0, 720)
    close = open_ * np.exp(returns)
    block = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close),
            "low": np.minimum(open_, close),
            "close": close,
        }
    )

    positive_hill, negative_hill, contrast, mass, variation, positives, negatives = (
        s.tail_statistics(block)
    )

    assert positives == 30 and negatives == 30
    assert positive_hill > negative_hill
    assert contrast > 0
    assert mass > 0
    assert variation > 0


def panel():
    return pd.DataFrame(
        {
            "source_valid": [True] * 6,
            "positive_count": [30] * 6,
            "negative_count": [30] * 6,
            "positive_hill": [0.2] * 6,
            "negative_hill": [0.1] * 6,
            "tail_index_contrast": [-0.1, 0.3, 0.4, 0.2, -0.5, -0.4],
            "tail_asymmetry_rank": [0.5, 0.8, 0.9, 0.4, 0.8, 0.9],
            "tail_mass_asymmetry": [0.1, -0.3, -0.4, -0.2, 0.5, 0.4],
            "tail_mass_rank": [0.5, 0.8, 0.9, 0.4, 0.8, 0.9],
            "variation_rank": [0.8] * 6,
            "feature_available_time": pd.date_range(
                "2024-01-01T02:00:00Z", periods=6, freq="12h"
            ),
        }
    )


def test_primary_and_controls():
    active, side, _ = s.active(panel())
    assert active.tolist() == [False, True, False, False, True, False]
    assert side[active].tolist() == [1, -1]

    without_variation = panel()
    without_variation.loc[1, "variation_rank"] = 0.4
    assert s.active(without_variation, "no_variation_gate")[0].iloc[1]

    mass_active, mass_side, _ = s.active(panel(), "fixed_k_tail_mass_asymmetry")
    assert mass_active.tolist() == [False, True, False, False, True, False]
    assert mass_side[mass_active].tolist() == [-1, 1]

    forced_active, forced_side, _ = s.active(panel(), "same_clock_forced_long")
    assert forced_active.equals(active)
    assert forced_side[forced_active].eq(1).all()


def test_prior_rank_excludes_current():
    ranks = s.prior_rank(pd.Series(np.arange(121, dtype=float)))
    assert np.isnan(ranks.iloc[119])
    assert ranks.iloc[120] == 1.0
