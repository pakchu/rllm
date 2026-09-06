import hashlib
import json
from pathlib import Path

from training import build_high_volatility_doi_research_attention_relay_support as builder


ARTIFACT = Path("results/high_volatility_doi_research_attention_relay_support_2026-08-12.json")


def test_hvdra_source_support_artifact_is_sealed():
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "f0e03ed636f984253bb11e2bcd0b93dd4731b3ad7128c3585437a73cf13c82ea"
    )
    value = json.loads(ARTIFACT.read_text())
    assert value["policy_id"] == "HVDRA-24"
    assert value["support_passed"] is True
    assert value["decision"] == "pass_to_novelty"
    assert value["postentry_return_pnl_execution_price_opened"] is False
    assert value["gross9_rows_opened"] is False
    assert all(value["support_checks"].values())
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == builder.canonical_hash(core)


def test_hvdra_source_and_clock_hashes_are_bound():
    value = json.loads(ARTIFACT.read_text())
    clock = Path(value["clock"]["path"])
    manifest = Path(value["source_manifest"]["path"])
    assert hashlib.sha256(clock.read_bytes()).hexdigest() == value["clock"]["sha256"]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == value["source_manifest"]["sha256"]
    assert value["clock"]["rows"] == 327
    assert value["support"]["train"]["events"] == 43
    assert value["support"]["final"]["events"] == 72
