import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_intraday_hour_reversal_support_2026-08-11.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_source_support_artifact():
    artifact = json.loads(RESULT.read_text())
    assert sha(RESULT) == "7c35df808edb23f5861d49d55b08f336e77f270d901bc9cdb419b48a5336f243"
    assert artifact["policy_id"] == "HVIHR-1"
    assert artifact["support_passed"] is True
    assert artifact["advance_to_gross9_novelty"] is True
    assert artifact["advance_to_economic_outcomes"] is False
    assert artifact["decision"] == "pass_to_novelty"
    assert artifact["postentry_return_pnl_execution_price_opened"] is False
    assert artifact["funding_values_opened"] is False
    assert artifact["gross9_rows_opened"] is False
    assert artifact["clock"]["rows"] == 180
    assert artifact["clock"]["sha256"] == "20ca8a21ede858e1607dcc6b164bb5f9b194b7a1e02aa606603492d6e35355a0"
    assert all(artifact["support_checks"].values())
    assert {name: values["events"] for name, values in artifact["support"].items()} == {
        "train": 29,
        "test": 64,
        "eval": 59,
        "final": 28,
    }
