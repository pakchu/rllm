import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_spot_conditioned_transmission_beta_asymmetry_relay_"
    "support_2026-08-13.json"
)


def test_source_artifact_is_terminal_before_novelty_or_outcomes():
    payload = json.loads(RESULT.read_text())
    assert payload["support_passed"] is False
    assert payload["advance_to_gross9_novelty"] is False
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["support_checks"]["train_side_balance"] is False
    assert payload["support_checks"]["test_side_balance"] is False
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "ac079c607d935e6b6bb3163bd0dcacb6bfa9794f953764cc9da71f7d961b28f8"
    )
