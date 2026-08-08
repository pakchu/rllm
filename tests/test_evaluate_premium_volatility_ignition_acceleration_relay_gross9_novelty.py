from pathlib import Path

from training import evaluate_premium_volatility_ignition_acceleration_relay_gross9_novelty as novelty


def test_pviar_novelty_evaluator_is_outcome_blind_and_frozen():
    source = Path(novelty.__file__).read_text()
    assert novelty.POLICY == "PVIAR-6"
    assert novelty.LIMITS["one_to_one_6h_max_matched_share"] == 0.35
    assert novelty.LIMITS["occupied_5m_bar_jaccard"] == 0.25
    assert '"outcomes_opened": False' in source
    assert '"advance_to_economic_outcomes": advance' in source
    assert "bars_binance" not in source
    assert "funding_rates_binance" not in source
