from pathlib import Path
from training import evaluate_high_volatility_chandelier_exit_state_flip_relay_gross9_novelty as n

def test_blind_and_bound():
 assert n.POLICY=="HVCE-24" and n.sha(n.PREREG)==n.PREREG_SHA and n.sha(n.SUPPORT)==n.SUPPORT_SHA and n.sha(n.CLOCK)==n.CLOCK_SHA
 source=Path(n.__file__).read_text();assert '"outcomes_opened": False' in source and "bars_binance" not in source and "funding_rates_binance" not in source
