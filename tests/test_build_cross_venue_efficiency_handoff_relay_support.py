from training import build_cross_venue_efficiency_handoff_relay_support as s
def test_contract():
 assert s.PREREG_SHA=='7d40bb2759b395cd6fd486637b9d706ba08c5524e932857559fc545f195117af';assert 'bars_binance_spot' in s.QUERY.format(table='bars_binance_spot');assert s.CONTROLS==('no_volume_handoff','no_efficiency_asymmetry','spot_efficiency_only','one_boundary_stale_geometry','direction_flip','forced_long','forced_short')
