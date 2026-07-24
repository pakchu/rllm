from __future__ import annotations

import json
from pathlib import Path

from training import preregister_paired_intrinsic_venue_orderflow_topology as p


ARTIFACT = Path(
    "results/paired_intrinsic_venue_orderflow_topology_"
    "preregistration_2026-07-24.json"
)
ARTIFACT_SHA256 = (
    "e2a94c3e675addaad4bb0075a27f4155e5f1675bcc1552a66ceb9c30a5ceab28"
)
MANIFEST_HASH = (
    "09faeaffba6d7c88e420c2198dfb536f2b82c3198f210a0e166906bf5c1cb532"
)


def test_frozen_preregistration_artifact_matches_code_and_hash() -> None:
    assert p.sha256_file(ARTIFACT) == ARTIFACT_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    p.validate_manifest(payload)
    assert payload == p.build_manifest()
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["source_rows_decoded"] is False
    assert payload["market_value_rows_decoded"] is False
    assert payload["funding_value_rows_decoded"] is False
    assert payload["comparator_rows_decoded"] is False
    assert payload["post_2023_values_decoded"] is False
