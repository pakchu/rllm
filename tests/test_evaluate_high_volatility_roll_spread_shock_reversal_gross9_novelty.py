from pathlib import Path
from training import evaluate_high_volatility_roll_spread_shock_reversal_gross9_novelty as n
def test_evaluator_is_outcome_blind_and_bound():
 assert n.POLICY=='HVRSSR-8';assert n.sha(n.PREREG)==n.PREREG_SHA;assert n.sha(n.SUPPORT)==n.SUPPORT_SHA;assert n.sha(n.CLOCK)==n.CLOCK_SHA
 source=Path(n.__file__).read_text();assert '"outcomes_opened": False' in source and 'bars_binance' not in source and 'funding_rates_binance' not in source
