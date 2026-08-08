import hashlib
import json

from training import build_volatility_gated_stablecoin_quote_flow_consensus_relay_support as support


def test_vgsqf_source_support_is_frozen_terminal_without_outcome_access():
    assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest() == (
        "8967dc5b4ce33bce7c8e051a81d73e44a73dd8469bc60b9699420f95b07c5cd6"
    )
    assert hashlib.sha256(support.CLOCK.read_bytes()).hexdigest() == (
        "7e6ede88c1796f686ef93fa3ff5b02896c4633c0e17ef41c5b1cb6f7955ffefc"
    )
    payload = json.loads(support.RESULT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == support.canonical_hash(core)
    assert payload["support_passed"] is False
    assert payload["advance_to_gross9_novelty"] is False
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["decision"] == "terminal_source_support_reject"
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert [payload["support"][name]["events"] for name in ("train", "test", "eval", "final")] == [
        14,
        44,
        12,
        17,
    ]
    assert payload["support_checks"]["final_month_concentration"] is False
    assert payload["support"]["final"]["max_month_share"] == 13 / 17
    assert all(not item["promotion_authorized"] for item in payload["controls"].values())
