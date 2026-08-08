import hashlib
import json

from training import build_premium_volatility_ignition_acceleration_relay_support as support


def test_pviar_support_artifact_is_hash_bound_and_outcome_blind():
    report = json.loads(support.RESULT.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == support.canonical_hash(core)
    assert report["policy_id"] == "PVIAR-6"
    assert report["support_passed"] is True
    assert report["advance_to_gross9_novelty"] is True
    assert report["advance_to_economic_outcomes"] is False
    assert report["postentry_return_pnl_execution_price_opened"] is False
    assert report["gross9_rows_opened"] is False
    assert all(report["support_checks"].values())
    assert hashlib.sha256(support.CLOCK.read_bytes()).hexdigest() == report["clock"]["sha256"]


def test_pviar_premium_snapshot_retains_missing_minutes_without_imputation():
    report = json.loads(support.PREMIUM_MANIFEST.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == support.canonical_hash(core)
    assert report["no_imputation"] is True
    assert report["outcomes_opened"] is False
    assert report["output"]["missing_minutes"] == 16
    assert report["output"]["valid_rows"] == report["output"]["rows"]
