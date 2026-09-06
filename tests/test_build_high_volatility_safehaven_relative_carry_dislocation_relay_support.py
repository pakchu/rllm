import numpy as np
import pandas as pd

from training import build_high_volatility_safehaven_relative_carry_dislocation_relay_support as support


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_valid": [True] * 5,
            "relative_dislocation": [1.0, -2.0, 0.0, 0.5, -0.7],
            "raw_return_difference": [-0.2, 0.4, 0.1, -0.3, 0.2],
            "dislocation_rank": [0.8, 0.9, 0.95, 0.6, 0.8],
            "btc_realized_variation_rank": [0.7, 0.8, 0.9, 0.9, 0.4],
        }
    )


def test_primary_follows_relative_carry_direction() -> None:
    active, side = support.conditions(frame(), "primary")
    assert active.tolist() == [True, True, False, False, False]
    assert side[active].tolist() == [-1.0, 1.0]


def test_controls_are_diagnostic() -> None:
    active, _ = support.conditions(frame(), "no_dislocation_tail")
    assert active.tolist() == [True, True, False, True, False]
    active, side = support.conditions(frame(), "direction_flip")
    assert side[active].tolist() == [1.0, -1.0]
    active, side = support.conditions(frame(), "forced_long")
    assert side[active].tolist() == [1, 1]


def test_causal_statistics_exclude_current() -> None:
    values = pd.Series(np.arange(61, dtype=float))
    zscore = support.causal_z(values)
    expected = (60 - values.iloc[:60].mean()) / values.iloc[:60].std(ddof=1)
    assert np.isclose(zscore.iloc[60], expected)
    assert support.strict_prior_midrank(values).iloc[60] == 1.0


def test_safehaven_orientation_and_registration_are_frozen() -> None:
    assert support.RISK_MULTIPLIER == {"USDJPY": -1.0, "USDCHF": -1.0}
    assert support.PREREG_SHA == "eff0eb9d1364717c0479e4498180f36f49b4dd66a625cb55452b77eca860e9cd"
