import numpy as np
import pandas as pd

from training import build_high_volatility_safehaven_variation_leadership_relay_support as support


def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "signal_valid": [True] * 5,
        "dominant_safehaven_return": [0.4, -0.3, 0.0, 0.2, -0.5],
        "subordinate_safehaven_return": [-0.1, 0.2, 0.3, -0.4, 0.1],
        "leadership_rank": [0.8, 0.9, 0.95, 0.6, 0.8],
        "btc_realized_variation_rank": [0.7, 0.8, 0.9, 0.9, 0.4],
    })


def test_primary_follows_dominant_safehaven_risk_direction() -> None:
    active, side = support.conditions(frame(), "primary")
    assert active.tolist() == [True, True, False, False, False]
    assert side[active].tolist() == [-1.0, 1.0]


def test_controls_are_diagnostic() -> None:
    active, _ = support.conditions(frame(), "no_leadership_tail")
    assert active.tolist() == [True, True, False, True, False]
    active, side = support.conditions(frame(), "subordinate_pair_direction")
    assert side[active].tolist() == [1.0, -1.0, -1.0]
    active, side = support.conditions(frame(), "forced_long")
    assert side[active].tolist() == [1, 1]


def test_causal_rank_excludes_current() -> None:
    values = pd.Series(np.arange(61, dtype=float))
    assert support.strict_prior_midrank(values).iloc[60] == 1.0


def test_safehaven_orientation_and_registration_are_frozen() -> None:
    assert support.RISK_MULTIPLIER == {"USDJPY": -1.0, "USDCHF": -1.0}
    assert support.PREREG_SHA == "21ffaa568d9d79a14a3cbf2e36fa7acf2edcbb2d2a20e320b5b899eb3704129e"
