from pathlib import Path

from training import evaluate_high_volatility_intraday_hour_reversal_gross9_novelty as novelty


def test_novelty_evaluator_is_blind_and_bound():
    assert novelty.POLICY == "HVIHR-1"
    assert novelty.sha(novelty.PREREG) == novelty.PREREG_SHA
    assert novelty.sha(novelty.SUPPORT) == novelty.SUPPORT_SHA
    assert novelty.sha(novelty.CLOCK) == novelty.CLOCK_SHA
    source = Path(novelty.__file__).read_text()
    assert '"outcomes_opened": False' in source
    assert "bars_binance" not in source
    assert "funding_rates_binance" not in source
