from __future__ import annotations

import json
from pathlib import Path

from training import preregister_dollar_collateral_liquidity_bank_relay as p


ARTIFACT = Path(
    "results/dollar_collateral_liquidity_bank_relay_"
    "preregistration_2026-07-24.json"
)
ARTIFACT_SHA256 = (
    "0947513376f36991a1cf4e5dc0a2aae7417f246dd2698f82d19f6dfe09bc67ec"
)
MANIFEST_HASH = (
    "da77218e89ee588f901d719ea9944fad1cfd5f1b88712f679dc1e2de5a3a9e4d"
)


def test_frozen_dclb_preregistration_matches_code_and_hash() -> None:
    assert p.sha256_file(ARTIFACT) == ARTIFACT_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    p.validate_manifest(payload)
    assert payload == p.build_manifest()
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["source_rows_decoded"] is False
    assert payload["comparator_rows_decoded"] is False
    assert all(value == 0 for value in payload["evidence_boundary"].values())
