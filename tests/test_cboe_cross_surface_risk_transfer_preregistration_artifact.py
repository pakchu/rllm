from __future__ import annotations

import json
from pathlib import Path

from training import preregister_cboe_cross_surface_risk_transfer as p


ARTIFACT = Path(
    "results/cboe_cross_surface_risk_transfer_"
    "preregistration_2026-07-24.json"
)
ARTIFACT_SHA256 = (
    "4e26603221e8109c38151873e31110b61f952a29b484266549e480e7c283af52"
)
MANIFEST_HASH = (
    "d6ce5f03a18f47f9e8221f91b3aa7af687754c5c6c45d1e6881e3fa1e9c30123"
)


def test_frozen_cxrt_preregistration_matches_code_and_hash() -> None:
    assert p.sha256_file(ARTIFACT) == ARTIFACT_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    p.validate_manifest(payload)
    assert payload == p.build_manifest()
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["source_rows_decoded"] is False
    assert payload["comparator_rows_decoded"] is False
