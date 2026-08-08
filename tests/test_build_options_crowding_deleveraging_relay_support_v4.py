from training import build_options_crowding_deleveraging_relay_support_v4 as s

def test_v4_support_is_outcome_blind_and_uses_v4_identity():
 source=open(s.__file__).read();assert 'OCDR-12C' in source;assert '"advance_to_economic_outcomes": False' in source;assert 'bars_binance' not in source
