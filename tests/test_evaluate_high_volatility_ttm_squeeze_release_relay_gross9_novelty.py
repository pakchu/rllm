from pathlib import Path
from training import evaluate_high_volatility_ttm_squeeze_release_relay_gross9_novelty as n

def test_blind_and_bound():
 assert n.POLICY=="HVTTS-24" and n.sha(n.PREREG)==n.PREREG_SHA and n.sha(n.SUPPORT)==n.SUPPORT_SHA and n.sha(n.CLOCK)==n.CLOCK_SHA
 source=Path(n.__file__).read_text();assert '"outcomes_opened": False' in source and "bars_binance" not in source and "funding_rates_binance" not in source

def test_prereg_non_ascii_manifest_uses_frozen_prereg_canonicalization():
 assert n.load_manifest(n.PREREG)["policy_id"] == n.POLICY
