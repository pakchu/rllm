from training import build_options_crowding_deleveraging_relay_support_v4 as s
import pandas as pd

def test_v4_support_is_outcome_blind_and_uses_v4_identity():
 source=open(s.__file__).read();assert 'OCDR-12C' in source;assert '"advance_to_economic_outcomes": False' in source;assert 'bars_binance' not in source

def test_mixed_exact_and_subsecond_funding_times_parse_without_rounding():
 parsed=pd.to_datetime(pd.Series(['2023-06-20 00:00:00.001000+00:00','2023-06-20 08:00:00+00:00']),utc=True,format='mixed')
 assert parsed.iloc[0].microsecond==1000 and parsed.iloc[1].microsecond==0
