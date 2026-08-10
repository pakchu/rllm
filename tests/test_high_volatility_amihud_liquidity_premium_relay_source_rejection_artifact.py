import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_amihud_liquidity_premium_relay_support_2026-08-11.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvalpr_source_failure_is_terminal_and_outcomes_are_sealed():
    artifact = json.loads(RESULT.read_text())
    assert sha(RESULT) == "e6d88b4bc0ae135dc715dd686afec2c3c509e5d316420ce9a152ff4734b5cb34"
    assert artifact["policy_id"] == "HVALPR-24"
    assert artifact["support_passed"] is False
    assert artifact["advance_to_gross9_novelty"] is False
    assert artifact["advance_to_economic_outcomes"] is False
    assert artifact["decision"] == "terminal_source_support_reject"
    assert artifact["support_checks"]["train_side_balance"] is False
    assert sum(not passed for passed in artifact["support_checks"].values()) == 1
    assert artifact["support"]["train"]["events"] == 22
    assert artifact["support"]["train"]["minority_side_share"] < 0.10
    assert artifact["postentry_return_pnl_execution_price_opened"] is False
    assert artifact["funding_values_opened"] is False
    assert artifact["gross9_rows_opened"] is False
    assert not Path("results/high_volatility_amihud_liquidity_premium_relay_gross9_novelty_2026-08-11.json").exists()
