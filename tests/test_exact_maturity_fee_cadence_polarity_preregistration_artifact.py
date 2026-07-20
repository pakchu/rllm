from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import preregister_exact_maturity_fee_cadence_polarity as prereg


ARTIFACT = Path(
    "results/exact_maturity_fee_cadence_polarity_preregistration_2026-07-20.json"
)
ARTIFACT_SHA256 = (
    "def9e5c4940bfd6a146262668b90e0e6d2efae58c16ac040a6811e2bca77b189"
)
MANIFEST_HASH = "0672c23224cfbf2e4fb9ebcacb08ff9e20f550128523544507fafe49e01c1bd0"
POLICY_HASH = "ecce53d9c17aea05ef78781d67ffea87b7bd27bc864b44624e7e0acaf87d1019"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_emfc_preregistration_artifact_is_exact_and_outcome_blind() -> None:
    assert _sha256(ARTIFACT) == ARTIFACT_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["policy_hash"] == POLICY_HASH
    assert payload["policy_id"] == "EMFC-864"
    assert payload["outcomes_opened"] is False
    assert payload["outcome_boundary"] == prereg.PREREGISTRATION_OUTCOME_BOUNDARY
    assert payload["preregistration_source"]["sha256"] == prereg.sha256_file(
        prereg.PREREGISTRATION_SOURCE
    )
    assert prereg.load_preregistration(ARTIFACT) == payload
