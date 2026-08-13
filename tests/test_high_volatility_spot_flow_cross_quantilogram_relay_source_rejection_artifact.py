import hashlib
import json
from pathlib import Path

from training import build_high_volatility_spot_flow_cross_quantilogram_relay_support as s


RESULT = Path("results/high_volatility_spot_flow_cross_quantilogram_relay_support_2026-08-13.json")
EXPECTED = "bf499b5ca6979e7c18a2316e7dc5fca701309bfc81fb2225eec9e5c81b25291f"


def test_terminal_source_rejection_is_immutable_and_outcomes_remain_sealed():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED
    result = json.loads(RESULT.read_text())
    manifest_hash = result.pop("manifest_hash")
    assert s.canonical_hash(result) == manifest_hash
    assert result["policy_id"] == "HVSFCQ-8"
    assert not result["support_passed"]
    assert result["decision"] == "terminal_source_support_reject"
    assert not result["advance_to_gross9_novelty"]
    assert not result["advance_to_economic_outcomes"]
    assert not result["postentry_return_pnl_execution_price_opened"]
    assert not result["funding_values_opened"]
    assert not result["gross9_rows_opened"]
    assert {key: value["events"] for key, value in result["support"].items()} == {
        "train": 0,
        "test": 0,
        "eval": 0,
        "final": 0,
    }
    source = json.loads(Path(result["source_manifest"]["path"]).read_text())
    assert source["output"]["valid_rows"] == 81
