from pathlib import Path

from training import evaluate_high_volatility_hikkake_pattern_relay_gross9_novelty as novelty


def test_evaluator_is_blind_and_hash_bound() -> None:
    assert novelty.POLICY == "HVHIKKAKE-C3-8"
    assert novelty.sha(novelty.PREREG) == novelty.PREREG_SHA
    assert novelty.sha(novelty.SUPPORT) == novelty.SUPPORT_SHA
    assert novelty.sha(novelty.CLOCK) == novelty.CLOCK_SHA
    source = Path(novelty.__file__).read_text()
    assert '"outcomes_opened": False' in source
    assert "bars_binance" not in source
    assert "funding_rates_binance" not in source
