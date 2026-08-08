import hashlib
import json

from training import build_high_volatility_oi_purge_continuation_relay_support as support


def test_hvopcr_source_support_is_frozen_terminal_without_outcome_access():
    assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest() == (
        "a2550475edfd5388c00bedecf8bd2352cc4c6053bafbf0b79fc762aac01b139e"
    )
    assert hashlib.sha256(support.CLOCK.read_bytes()).hexdigest() == (
        "cabaecb4689cf318c29db18e37e8eacb0df630a4b3d530d6a12493204f3ecf3e"
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
        90,
        99,
        35,
        54,
    ]
    assert payload["support_checks"]["final_month_concentration"] is False
    assert payload["support"]["final"]["max_month_share"] == 29 / 54
    assert all(not item["promotion_authorized"] for item in payload["controls"].values())
