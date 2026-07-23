from __future__ import annotations

import json
from pathlib import Path

from training import preregister_cross_venue_intrinsic_clock_resolution as p


ARTIFACT = Path(
    "results/cross_venue_intrinsic_clock_resolution_"
    "preregistration_2026-07-24.json"
)
ARTIFACT_SHA256 = (
    "f5f293b7b6152d7e28c5bb825d3dd2d8a2626678917960f721e900231c5671f0"
)
MANIFEST_HASH = (
    "75f93cc512ab711936af56834c60c2a416c4bcc43ea672a9076e6763cbff2f1c"
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
    assert payload["comparator_rows_decoded"] is False
