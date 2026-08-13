import hashlib
import json
from pathlib import Path

from training import build_high_volatility_bds_nonlinear_dependence_relay_support as s


RESULT = Path("results/high_volatility_bds_nonlinear_dependence_relay_support_2026-08-13.json")
EXPECTED = "2c4a3b9575176bb45ca71a97df75e4862e4b942da785a151c67d379498fd617e"


def test_source_pass_is_immutable_and_outcomes_remain_sealed():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED
    result = json.loads(RESULT.read_text())
    manifest_hash = result.pop("manifest_hash")
    assert s.canonical_hash(result) == manifest_hash
    assert result["policy_id"] == "HVBDS-8"
    assert result["support_passed"]
    assert result["decision"] == "pass_to_novelty"
    assert result["advance_to_gross9_novelty"]
    assert not result["advance_to_economic_outcomes"]
    assert not result["postentry_return_pnl_execution_price_opened"]
    assert not result["gross9_rows_opened"]
    assert {key: value["events"] for key, value in result["support"].items()} == {
        "train": 20,
        "test": 46,
        "eval": 52,
        "final": 23,
    }
    assert all(
        value["minority_side_share"] >= 0.2 and value["max_month_share"] <= 0.45
        for value in result["support"].values()
    )
    assert result["clock"]["rows"] == 141
