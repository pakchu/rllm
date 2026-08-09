import numpy as np
import pandas as pd

from training import build_high_volatility_realized_skew_reversal_relay_support as support


def frame():
    return pd.DataFrame({"source_valid": [True] * 5, "realized_variation": [1.0] * 5, "realized_skew": [0.2, -0.3, 0.1, 0.4, -0.2], "raw_third_moment": [2.0, -3.0, 1.0, 4.0, -2.0], "variation_rank": [0.8, 0.7, 0.9, 0.5, 0.8], "absolute_skew_rank": [0.8, 0.9, 0.6, 0.9, 0.8]})


def test_primary_fades_extreme_realized_skew_in_high_variation():
    active, side = support.conditions(frame(), "primary")
    assert active.tolist() == [True, True, False, False, True]
    assert side[active].tolist() == [-1, 1, 1]


def test_diagnostic_controls_are_frozen_and_not_needed_by_primary():
    assert support.CONTROLS == ("no_volatility_gate", "no_skew_tail", "raw_third_moment", "one_day_stale_features", "direction_flip", "forced_long")
    assert support.conditions(frame(), "no_volatility_gate")[0].tolist() == [True, True, False, True, True]
    assert support.conditions(frame(), "no_skew_tail")[0].tolist() == [True, True, True, False, True]
    active, side = support.conditions(frame(), "direction_flip")
    assert side[active].tolist() == [1, -1, -1]
    active, side = support.conditions(frame(), "forced_long")
    assert side[active].tolist() == [1, 1, 1]


def test_strict_prior_midrank_excludes_current():
    values = pd.Series(np.arange(61, dtype=float))
    ranks = support.strict_prior_midrank(values)
    assert np.isnan(ranks.iloc[59])
    assert ranks.iloc[60] == 1.0


def test_builder_binds_preregistration_and_seals_outcomes():
    assert support.sha256(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    source = support.Path(support.__file__).read_text()
    assert '"postentry_return_pnl_execution_price_opened": False' in source
    assert '"gross9_rows_opened": False' in source
