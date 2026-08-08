from pathlib import Path

from training import materialize_options_crowding_deleveraging_sources as m


def test_materializer_queries_only_frozen_nonprice_tables() -> None:
    source = Path(m.__file__).read_text()
    assert "open_interest_binance" in source
    assert "funding_rates_binance" in source
    assert "bars_binance" not in source
    assert '"btc_price_or_return_opened": False' in source
