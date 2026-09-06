from pathlib import Path
from training import evaluate_high_volatility_hammer_hanging_man_context_reversal_relay_gross9_novelty as n
def test_blind_and_hash_bound():
 assert n.POLICY=="HVHHM-C10-N5-8" and n.sha(n.PREREG)==n.PREREG_SHA and n.sha(n.SUPPORT)==n.SUPPORT_SHA and n.sha(n.CLOCK)==n.CLOCK_SHA
 source=Path(n.__file__).read_text();assert '"outcomes_opened": False' in source and "bars_binance" not in source and "funding_rates_binance" not in source
