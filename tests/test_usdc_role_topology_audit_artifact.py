from __future__ import annotations

import json
from pathlib import Path

from training import audit_usdc_role_topology as topology


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/usdc_role_topology_audit_2026-07-21.json"
ARTIFACT_SHA256 = "103c5f4270faefb8ec4fcd46351b83e1c817b5a2266c07a03a36a66ae576791f"
MANIFEST_HASH = "9fef90b4de9f917be4e6497b705ed2149c2bff328cf20f76f5fa60db27967fc6"


def test_committed_topology_audit_is_hash_bound_and_reproducible() -> None:
    assert topology.sha256_file(ARTIFACT) == ARTIFACT_SHA256
    committed = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    reproduced = topology.build_report()
    assert reproduced == committed
    assert committed["manifest_hash"] == MANIFEST_HASH


def test_topology_artifact_preserves_the_outcome_boundary() -> None:
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    boundary = report["outcome_boundary"]
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["return_or_pnl_fields_read"] == 0
    assert boundary["post_2023_event_rows_read"] == 0
    assert report["decision"]["candidate_clock_opened"] is False
    assert report["decision"]["economic_outcomes_opened"] is False
    assert report["decision"]["candidate_authorized"] is False
    assert report["decision"]["status"] == "retired_before_temporal_pairing"


def test_topology_artifact_records_support_and_concentration_without_addresses() -> (
    None
):
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    graph = report["audit"]["directed_recipient_burner_topology"]
    assert graph["recipient_burner_roles"] == 2
    assert graph["distinct_minter_recipient_edges"] == 2
    assert graph["mint_leg_events"] == 84_406
    assert graph["burn_leg_events"] == 143_076
    assert graph["mint_leg_role_concentration"]["largest_role_share"] > 0.99
    assert graph["burn_leg_role_concentration"]["largest_role_share"] > 0.99
    encoded = ARTIFACT.read_text(encoding="utf-8")
    assert "0x" not in encoded
