from pathlib import Path

from training import evaluate_russell_volatility_spread_break_relay_economics as economics


def test_rvsbr_economics_is_sequential_and_frozen():
    source = Path(economics.__file__).read_text()
    assert economics.POLICY_ID == "RVSBR-12"
    assert economics.LEVERAGE == 0.5
    assert economics.BASE_COST == 0.0006
    assert economics.STRESS_COST == 0.001
    assert economics.PREDECESSOR == {"test": "train", "eval": "test", "final": "eval"}
    assert "terminal_reject_no_repair" in source
    assert '"later_stage_outcomes_opened": False' in source


def test_rvsbr_economics_binds_frozen_authorization():
    assert economics.sha256(economics.PREREG) == economics.PREREG_SHA
    assert economics.sha256(economics.SUPPORT) == economics.SUPPORT_SHA
    assert economics.sha256(economics.NOVELTY) == economics.NOVELTY_SHA
    assert economics.sha256(economics.CLOCK) == economics.CLOCK_SHA
    assert economics.CONTROLS == ("no_btc_volatility_gate", "vix_change_only", "one_session_stale_spread", "direction_flip")
