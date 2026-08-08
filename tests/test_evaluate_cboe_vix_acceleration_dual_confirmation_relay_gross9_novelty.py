from pathlib import Path
from training import evaluate_cboe_vix_acceleration_dual_confirmation_relay_gross9_novelty as novelty
def test_cvvdmr_novelty_evaluator_is_outcome_blind_and_frozen():
 source=Path(novelty.__file__).read_text();assert novelty.POLICY=="CVVDMR-6" and novelty.LIMITS["one_to_one_6h_max_matched_share"]==.35 and novelty.LIMITS["occupied_5m_bar_jaccard"]==.25;assert '"outcomes_opened":False' in source and '"advance_to_economic_outcomes":advance' in source;assert "bars_binance" not in source and "funding_rates_binance" not in source
