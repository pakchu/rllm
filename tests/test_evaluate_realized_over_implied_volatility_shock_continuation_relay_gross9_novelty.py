from pathlib import Path
from training import evaluate_realized_over_implied_volatility_shock_continuation_relay_gross9_novelty as n
def test_rivscr_novelty_evaluator_is_outcome_blind_and_frozen():
 s=Path(n.__file__).read_text();assert n.POLICY=='RIVSCR-6' and n.LIMITS['one_to_one_6h_max_matched_share']==.35 and n.LIMITS['occupied_5m_bar_jaccard']==.25;assert '"outcomes_opened": False' in s and 'bars_binance' not in s and 'funding_rates_binance' not in s
