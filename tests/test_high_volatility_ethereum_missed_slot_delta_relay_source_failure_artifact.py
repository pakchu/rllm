import json

from training import record_high_volatility_ethereum_missed_slot_delta_relay_source_failure as record


def test_terminal_failure_artifact_matches_and_keeps_outcomes_sealed() -> None:
    value = json.loads(record.OUTPUT.read_text())
    assert value == record.build()
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == record.canonical_hash(core)
    assert value["status"] == "terminal_source_reject_no_repair"
    assert value["failure"]["http_status"] == 429
    assert value["opened_before_failure"] == {
        "partial_ethereum_header_responses_in_memory": True,
        "complete_daily_boundary_panel": False,
        "daily_missed_slot_counts_or_changes": False,
        "candidate_incidence": False,
        "btc_preentry_variation": False,
        "gross9_rows": False,
        "execution_price": False,
        "postentry_return_or_pnl": False,
        "funding": False,
    }
