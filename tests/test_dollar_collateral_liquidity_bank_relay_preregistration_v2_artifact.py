from __future__ import annotations

import json
from pathlib import Path

from training import preregister_dollar_collateral_liquidity_bank_relay_v2 as p


ARTIFACT = Path(
    "results/dollar_collateral_liquidity_bank_relay_"
    "preregistration_v2_2026-07-24.json"
)
ARTIFACT_SHA256 = (
    "423ae3c234e71be4e168a06d5270c7d437ad61118087b2fb548c37b0072269e6"
)
MANIFEST_HASH = (
    "fbee6397da5b891a9e586ca05388dbec64b5f02245324272987670ff25865eea"
)


def test_frozen_dclb_v2_preregistration_matches_code_and_hash() -> None:
    assert p.sha256_file(ARTIFACT) == ARTIFACT_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    p.validate_manifest(payload)
    assert payload == p.build_manifest()
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["source_algebra"]["macro"][
        "control_only_balanced_relation"
    ]["token"] == "MACRO_BALANCED_OPPOSITION"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["source_rows_decoded"] is False
    assert payload["comparator_rows_decoded"] is False
