from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import preregister_exact_maturity_fee_cadence_polarity as prereg


ARTIFACT = Path(
    "results/exact_maturity_fee_cadence_polarity_preregistration_2026-07-20.json"
)
ARTIFACT_SHA256 = (
    "43f1505786ad5ddd8a076afebccc26bff65387d8ef9b7a443035136606157ff6"
)
MANIFEST_HASH = "487a4c0dd3aa501605274f0afaacb6714c668078e6fac0506798afa4f9b0d743"
POLICY_HASH = "a264e58f834f2a58dda9ddcf3dcf5035ef941cd2087124b3ea1c8c306559b92f"


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
