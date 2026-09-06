from pathlib import Path
from training import evaluate_cross_venue_disagreement_resolution_gross9_novelty as n
def test_novelty_evaluator_is_outcome_blind_and_frozen():
 source=Path(n.__file__).read_text()
 assert n.POLICY_ID=='CVDR-6' and n.LIMITS['exact_entry_jaccard']==.1
 assert "'outcomes_opened':False" in source and "'advance_to_economic_outcomes':advance" in source
 assert 'bars_binance' not in source and 'funding_rates_binance' not in source
