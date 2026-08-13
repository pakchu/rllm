import hashlib
import json
from pathlib import Path

from training import build_high_volatility_alt_modularity_fragmentation_relay_support as s


RESULT = Path("results/high_volatility_alt_modularity_fragmentation_relay_support_2026-08-13.json")
EXPECTED = "e90e54b25d86446e09cb9a0d4d476f4735233b72910063b73ea4728712f5f312"


def test_source_pass_is_immutable_and_outcomes_remain_sealed():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED
    result = json.loads(RESULT.read_text())
    manifest_hash = result.pop("manifest_hash")
    assert s.chash(result) == manifest_hash
    assert result["policy_id"] == "HVAMF-8"
    assert result["support_passed"]
    assert result["decision"] == "pass_to_novelty"
    assert result["advance_to_gross9_novelty"]
    assert not result["advance_to_economic_outcomes"]
    assert not result["postentry_return_pnl_execution_price_opened"]
    assert not result["funding_values_opened"]
    assert not result["gross9_rows_opened"]
    assert {key: value["events"] for key, value in result["support"].items()} == {
        "train": 21,
        "test": 55,
        "eval": 21,
        "final": 20,
    }
    assert all(
        value["minority_side_share"] >= 0.2 and value["max_month_share"] <= 0.45
        for value in result["support"].values()
    )
    assert result["clock"]["rows"] == 117
