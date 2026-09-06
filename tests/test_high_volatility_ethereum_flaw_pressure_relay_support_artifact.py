import json
from training import build_high_volatility_ethereum_flaw_pressure_relay_support as b


def test_source_support_artifact_is_frozen_and_passes():
    value = json.loads(b.RESULT.read_text())
    core = dict(value)
    digest = core.pop("manifest_hash")
    assert digest == b.canonical_hash(core)
    assert value["policy_id"] == "HVEFPR-24"
    assert value["support_passed"] is True
    assert value["advance_to_gross9_novelty"] is True
    assert value["advance_to_economic_outcomes"] is False
    assert all(value["support_checks"].values())
    assert value["postentry_return_pnl_execution_price_opened"] is False
    assert value["gross9_rows_opened"] is False


def test_clock_and_source_manifest_hashes_match():
    value = json.loads(b.RESULT.read_text())
    assert b.sha(b.CLOCK) == value["clock"]["sha256"]
    assert b.sha(b.SOURCE_MANIFEST) == value["source_manifest"]["sha256"]
    assert value["clock"]["rows"] == 205
    assert {k: v["events"] for k, v in value["support"].items()} == {"train": 29, "test": 83, "eval": 53, "final": 40}
