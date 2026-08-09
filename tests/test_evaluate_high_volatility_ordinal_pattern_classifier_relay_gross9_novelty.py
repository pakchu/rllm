from pathlib import Path
from training import evaluate_high_volatility_ordinal_pattern_classifier_relay_gross9_novelty as novelty

def test_hvitr_novelty_evaluator_is_outcome_blind_and_frozen():
 source=Path(novelty.__file__).read_text();assert novelty.POLICY=="HVOCPR-8" and novelty.LIMITS["one_to_one_6h_max_matched_share"]==.35;assert novelty.SUPPORT.name.endswith("support_2026-08-10.json");assert '"outcomes_opened":False' in source and '"advance_to_economic_outcomes":advance' in source;assert "bars_binance" not in source and "funding_rates_binance" not in source
