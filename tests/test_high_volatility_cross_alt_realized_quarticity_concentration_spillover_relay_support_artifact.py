import hashlib, json
from pathlib import Path

from training import build_high_volatility_cross_alt_realized_quarticity_concentration_spillover_relay_support as s


RESULT = Path("results/high_volatility_cross_alt_realized_quarticity_concentration_spillover_relay_support_2026-08-13.json")
EXPECTED = "614ea6da1ebf59e962df07dc8c908f1b30aad189e8f69c2986a66a215d282ea4"


def test_source_pass_is_immutable_and_outcomes_remain_sealed():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED
    payload = json.loads(RESULT.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert s.chash(payload) == manifest_hash
    assert payload["policy_id"] == "HVCARQ-6"
    assert payload["support_passed"]
    assert payload["decision"] == "pass_to_novelty"
    assert payload["advance_to_gross9_novelty"]
    assert not payload["advance_to_economic_outcomes"]
    assert not payload["postentry_return_pnl_execution_price_opened"]
    assert not payload["funding_values_opened"]
    assert not payload["gross9_rows_opened"]
    assert {k: v["events"] for k, v in payload["support"].items()} == {
        "train": 28,
        "test": 75,
        "eval": 75,
        "final": 40,
    }
    assert all(v["minority_side_share"] >= 0.2 and v["max_month_share"] <= 0.45 for v in payload["support"].values())
    assert payload["clock"]["rows"] == 218
