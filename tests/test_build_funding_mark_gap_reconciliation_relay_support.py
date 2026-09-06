from pathlib import Path

import pandas as pd

from training import build_funding_mark_gap_reconciliation_relay_support as support


def test_fmgrr_source_builder_is_outcome_blind_and_hash_bound():
    source = Path(support.__file__).read_text()
    assert support.PREREG_SHA == "0f2c65319401437beae734e6d8e2a9d1cf6aba0205fd0082bf0f336cd74cae11"
    assert support.SETTLEMENT_QUERY.startswith("SELECT funding_time,mark_price")
    assert "postentry_return_pnl_execution_price_opened" in source
    assert "gross9_rows_opened" in source


def test_fmgrr_strict_prior_rank_excludes_current_value():
    values = pd.Series(range(130), dtype=float)
    ranks = support.strict_prior_rank(values)
    assert ranks.iloc[:126].isna().all()
    assert ranks.iloc[126] == 1.0


def test_fmgrr_controls_and_clock_are_frozen():
    assert support.CONTROLS == (
        "no_volatility_gate", "no_gap_tail", "one_day_stale_gap", "direction_flip"
    )
    assert support.MINIMUM_EVENTS == {"train": 8, "test": 12, "eval": 12, "final": 8}
