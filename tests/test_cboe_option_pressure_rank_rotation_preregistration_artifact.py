from __future__ import annotations

import json
from pathlib import Path

from training import preregister_cboe_option_pressure_rank_rotation as p


ARTIFACT = Path(
    "results/cboe_option_pressure_rank_rotation_"
    "preregistration_2026-07-24.json"
)
ARTIFACT_SHA256 = (
    "76db2b61fe35599acaa9eb52d3406eac891bad3f0c95c17e6ccd212aea719d99"
)
MANIFEST_HASH = (
    "a8f45ab7339eb773650830ed73f541820082de6c9f86a7dfa40a69b430d2fb99"
)


def test_frozen_oprr_preregistration_matches_code_and_hash() -> None:
    assert p.sha256_file(ARTIFACT) == ARTIFACT_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    p.validate_manifest(payload)
    assert payload == p.build_manifest()
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["source_rows_decoded"] is False
    assert payload["comparator_rows_decoded"] is False
