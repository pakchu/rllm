import hashlib
import json
from pathlib import Path

from training import build_high_volatility_lottery_sales_risk_appetite_relay_support as builder


ARTIFACT = Path("results/high_volatility_lottery_sales_risk_appetite_relay_support_2026-08-12.json")


def test_hvlsra_source_support_artifact_is_sealed():
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "6c30ce499fe99d3442bdfa0c7a1aa9b89f44d5623af9c5bb5e7d2294e02cc371"
    )
    value = json.loads(ARTIFACT.read_text())
    assert value["policy_id"] == "HVLSRA-24"
    assert value["support_passed"] is True
    assert value["decision"] == "pass_to_novelty"
    assert value["postentry_return_pnl_execution_price_opened"] is False
    assert value["gross9_rows_opened"] is False
    assert all(value["support_checks"].values())
    assert value["manifest_hash"] == builder.canonical_hash({k: v for k, v in value.items() if k != "manifest_hash"})


def test_hvlsra_source_and_clock_hashes_are_bound():
    value = json.loads(ARTIFACT.read_text())
    clock = Path(value["clock"]["path"])
    manifest = Path(value["source_manifest"]["path"])
    assert hashlib.sha256(clock.read_bytes()).hexdigest() == value["clock"]["sha256"]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == value["source_manifest"]["sha256"]
    assert value["clock"]["rows"] == 169
    assert value["support"]["train"]["events"] == 17
    assert value["support"]["final"]["events"] == 36
