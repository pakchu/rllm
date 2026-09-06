from pathlib import Path
from training import evaluate_high_volatility_signed_variance_feedback_relay_gross9_novelty as n
def test_novelty_evaluator_is_blind_and_bound():
 assert n.POLICY=='HVSVF-8' and n.sha(n.PREREG)==n.PREREG_SHA and n.sha(n.SUPPORT)==n.SUPPORT_SHA and n.sha(n.CLOCK)==n.CLOCK_SHA
 s=Path(n.__file__).read_text();assert '"outcomes_opened": False' in s and 'bars_binance' not in s and 'funding_rates_binance' not in s
