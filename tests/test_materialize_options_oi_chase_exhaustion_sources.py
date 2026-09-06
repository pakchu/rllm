from pathlib import Path
from training import materialize_options_oi_chase_exhaustion_sources as m
def test_materializer_opens_only_completed_feature_hour():
 source=Path(m.__file__).read_text();assert "date_bin('1 hour'" in source;assert 'post_entry_return_pnl_or_execution_price_opened' in source;assert 'funding_rates_binance' not in source
