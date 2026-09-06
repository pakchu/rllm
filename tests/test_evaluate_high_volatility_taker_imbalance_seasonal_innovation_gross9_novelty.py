from pathlib import Path
from training import evaluate_high_volatility_taker_imbalance_seasonal_innovation_gross9_novelty as novelty
def test_evaluator_is_blind_and_bound():
 assert novelty.POLICY=="HVTISI-8" and novelty.sha(novelty.PREREG)==novelty.PREREG_SHA and novelty.sha(novelty.SUPPORT)==novelty.SUPPORT_SHA and novelty.sha(novelty.CLOCK)==novelty.CLOCK_SHA;source=Path(novelty.__file__).read_text();assert '"outcomes_opened": False' in source and "bars_binance" not in source and "funding_rates_binance" not in source
def test_limits_match_frozen_protocol():
 assert novelty.LIMITS=={"exact_entry_jaccard":.1,"one_to_one_6h_max_matched_share":.35,"occupied_5m_bar_jaccard":.25,"absolute_signed_exposure_pearson":.35}
