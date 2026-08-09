import numpy as np
import pandas as pd

from training import build_high_volatility_trade_arrival_backloading_relay_support as support


def test_strict_prior_midrank_excludes_current_and_requires_history(monkeypatch):
    monkeypatch.setattr(support, "HISTORY", 4)
    monkeypatch.setattr(support, "MIN_HISTORY", 3)
    values = pd.Series([1.0, 2.0, 3.0, 2.0, 4.0])
    ranks = support.strict_prior_midrank(values)
    assert np.isnan(ranks.iloc[2])
    assert ranks.iloc[3] == (1.0 + 0.5) / 3.0
    assert ranks.iloc[4] == 1.0


def test_support_contract_keeps_outcomes_and_gross9_sealed():
    source = support.Path(support.__file__).read_text()
    assert "FROM bars_binance" in support.QUERY
    assert "number_of_trades" in support.QUERY
    assert '"postentry_return_pnl_execution_price_opened": False' in source
    assert '"gross9_rows_opened": False' in source
    assert support.CONTROLS == ("no_variation_gate", "no_arrival_tail", "one_block_stale_features", "direction_flip", "forced_long")
