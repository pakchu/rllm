from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import preregister_venue_ticket_migration_shock as vtms


ARTIFACT = Path(vtms.DEFAULT_OUTPUT)
EXPECTED_SHA256 = "04fff22aae17d8cba1a94dc5fb08746c2501013a26a353daa901fa06b548a8ed"
EXPECTED_MANIFEST_HASH = (
    "8ccfc33ddc29f8f60a2ba8c788af22b23c0f5ea8429122657bde5d8cb4321af6"
)


def test_preregistration_artifact_is_frozen_and_outcome_blind() -> None:
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(ARTIFACT.read_text())
    vtms.validate_manifest(payload)
    assert payload["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert payload["outcomes_opened"] is False
    assert (
        payload["research_history_boundary"]["exact_vtms_288_outcomes_opened"] is False
    )
