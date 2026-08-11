import hashlib
import json
from pathlib import Path

from training import build_high_volatility_seismic_stress_rotation_relay_support as support


ARTIFACT = Path("results/high_volatility_seismic_stress_rotation_relay_support_2026-08-12.json")


def test_hvssr_terminal_source_support_is_sealed():
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "f95ee43cb0ba12632209eb848f29aba16fef6cd1e11430389c81b127870e39fb"
    )
    result = json.loads(ARTIFACT.read_text())
    assert result["policy_id"] == "HVSSR-24"
    assert result["support"]["train"] == {
        "events": 30, "longs": 15, "shorts": 15,
        "minority_side_share": 0.5, "max_month_share": 0.4666666666666667,
    }
    assert result["support_checks"]["train_month_concentration"] is False
    assert result["support_passed"] is False
    assert result["advance_to_gross9_novelty"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["decision"] == "terminal_source_support_reject"


def test_hvssr_support_manifest_and_outputs_are_hash_bound():
    result = json.loads(ARTIFACT.read_text())
    assert result["manifest_hash"] == support.canonical_hash(
        {key: value for key, value in result.items() if key != "manifest_hash"}
    )
    manifest_path = Path(result["source_manifest"]["path"])
    assert support.sha(manifest_path) == result["source_manifest"]["sha256"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["outputs"]["event_versions"]["events"] == 6140
    assert manifest["outputs"]["raw_index"]["responses"] == 81
    for output in manifest["outputs"].values():
        assert support.sha(Path(output["path"])) == output["sha256"]
