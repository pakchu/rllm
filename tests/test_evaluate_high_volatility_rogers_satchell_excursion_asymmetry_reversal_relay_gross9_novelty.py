from pathlib import Path
from training import evaluate_high_volatility_rogers_satchell_excursion_asymmetry_reversal_relay_gross9_novelty as n
def test_blind_and_bound():
 assert n.POLICY=="HVRSAR-8" and n.sha(n.PREREG)==n.PREREG_SHA and n.sha(n.SUPPORT)==n.SUPPORT_SHA and n.sha(n.CLOCK)==n.CLOCK_SHA
 s=Path(n.__file__).read_text();assert '"outcomes_opened": False' in s and "bars_binance" not in s and "funding_rates_binance" not in s
