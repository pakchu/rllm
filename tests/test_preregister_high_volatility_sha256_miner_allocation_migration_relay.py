import hashlib
import json

from training import preregister_high_volatility_sha256_miner_allocation_migration_relay as prereg


def test_manifest_is_canonical_and_outcome_blind() -> None:
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    expected = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert payload["manifest_hash"] == expected
    assert payload["policy_id"] == "HVSMAM-24"
    assert payload["oos_outcomes_opened"] is False
    assert payload["oos_source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["research_boundary"]["candidate_count"] == 1
    assert payload["research_boundary"]["grid"] is False
    assert payload["research_boundary"]["repair_of_prior_candidate"] is False


def test_frozen_source_clock_and_gates() -> None:
    payload = prereg.build()
    assert payload["source_contract"]["assets"] == ["btc", "bch"]
    assert payload["source_contract"]["metrics"] == [
        "HashRate", "BlkCnt", "AssetEODCompletionTime"
    ]
    assert payload["policy"]["migration_days"] == 3
    assert payload["policy"]["absolute_migration_rank_min"] == 0.80
    assert payload["policy"]["btc_variation_rank_min"] == 0.65
    assert payload["policy"]["hold_hours"] == 24
    assert payload["source_support_gates"]["complete_paired_daily_grid_required"] is True
    assert payload["economic_gates"]["stop_on_first_failure"] is True
    assert "no asset pair" in payload["stopping_rule"]
