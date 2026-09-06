from pathlib import Path
from training import evaluate_treasury_real_term_spread_relay_gross9_novelty as novelty
def test_novelty_evaluator_is_outcome_blind_and_frozen():
 s=Path(novelty.__file__).read_text();assert novelty.POLICY=='TRTSR-24';assert novelty.LIMITS=={'exact_entry_jaccard':.10,'one_to_one_6h_max_matched_share':.35,'occupied_5m_bar_jaccard':.25,'absolute_signed_exposure_pearson':.35};assert '"outcomes_opened": False' in s;assert 'bars_binance' not in s;assert 'funding_rates_binance' not in s
def test_novelty_binds_predecessors():
 assert novelty.sha(novelty.PREREG)==novelty.PREREG_SHA;assert novelty.sha(novelty.SUPPORT)==novelty.SUPPORT_SHA;assert novelty.sha(novelty.CLOCK)==novelty.CLOCK_SHA
