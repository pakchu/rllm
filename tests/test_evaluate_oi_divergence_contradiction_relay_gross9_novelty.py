from pathlib import Path
from training import evaluate_oi_divergence_contradiction_relay_gross9_novelty as novelty
def test_oidcr_novelty_evaluator_is_outcome_blind_and_frozen():
 source=Path(novelty.__file__).read_text();assert novelty.POLICY=="OIDCR-8" and novelty.LIMITS["one_to_one_6h_max_matched_share"]==.35;assert '"outcomes_opened":False' in source and '"advance_to_economic_outcomes":advance' in source and "bars_binance" not in source
