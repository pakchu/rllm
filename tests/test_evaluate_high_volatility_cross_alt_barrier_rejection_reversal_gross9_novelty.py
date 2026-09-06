from pathlib import Path
from training import evaluate_high_volatility_cross_alt_barrier_rejection_reversal_gross9_novelty as n

def test_blind_and_bound():
    assert n.POLICY == "HVCABRR-8"
    assert n.sha(n.PREREG) == n.PREREG_SHA
    assert n.sha(n.SUPPORT) == n.SUPPORT_SHA
    assert n.sha(n.CLOCK) == n.CLOCK_SHA
    source = Path(n.__file__).read_text()
    assert '"outcomes_opened": False' in source
    assert "bars_binance" not in source and "funding_rates_binance" not in source
