from pathlib import Path

import pandas as pd

from training import materialize_options_crowding_deleveraging_sources as m


def test_materializer_queries_only_frozen_nonprice_tables() -> None:
    source = Path(m.__file__).read_text()
    assert "open_interest_binance" in source
    assert "funding_rates_binance" in source
    assert "bars_binance" not in source
    assert '"btc_price_or_return_opened": False' in source


def test_zero_oi_missing_markers_are_retained_for_downstream_invalidation() -> None:
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2023-07-01", periods=2, freq="5min", tz="UTC"),
            "sum_open_interest": [100.0, 0.0],
            "sum_open_interest_value": [1000.0, 0.0],
        }
    )
    m.validate_oi(frame)
    assert frame["sum_open_interest"].tolist() == [100.0, 0.0]
