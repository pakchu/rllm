import json

from training import preregister_deribit_btc_option_surface_forward_archive as prereg


def test_forward_archive_contract_is_frozen_and_outcome_blind():
    value = prereg.build()
    prereg.validate(value)
    assert value["collection_contract"]["eligible_start"] == "2026-08-16T08:02:00Z"
    assert value["collection_contract"]["historical_backfill"] == "forbidden"
    assert value["collection_contract"]["minimum_joined_share"] == 0.95
    assert all(
        record["economic_or_candidate_eligibility"] is False
        for record in value["diagnostic_snapshots"]
    )
    assert value["research_boundary"]["economic_candidate_authorized"] is False
    assert value["research_boundary"]["post_snapshot_btc_returns_or_pnl_opened"] is False
    json.dumps(value, allow_nan=False)
