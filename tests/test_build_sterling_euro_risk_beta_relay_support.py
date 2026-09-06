from pathlib import Path

import pandas as pd

from training import build_sterling_euro_risk_beta_relay_support as support


def test_serbr_source_builder_is_outcome_blind_and_hash_bound():
    source = Path(support.__file__).read_text()
    assert support.PREREG_SHA == "2ecf95b99acc2e62b96ea717e373eabaa48781e1017803f19e3ba2a2da849550"
    assert "bars_polygon" in support.FX_QUERY
    assert "GBPUSD" in support.FX_QUERY and "EURUSD" in support.FX_QUERY
    assert "postentry_return_pnl_execution_price_opened" in source
    assert "gross9_rows_opened" in source


def test_serbr_strict_prior_rank_excludes_current_value():
    values = pd.Series(range(130), dtype=float)
    ranks = support.strict_prior_rank(values)
    assert ranks.iloc[:126].isna().all()
    assert ranks.iloc[126] == 1.0


def test_serbr_controls_and_clock_are_frozen():
    assert support.CONTROLS == (
        "no_volatility_gate",
        "common_dollar_basket",
        "one_session_stale_relative_return",
        "direction_flip",
    )
    assert support.MINIMUM_EVENTS == {"train": 8, "test": 12, "eval": 12, "final": 8}
