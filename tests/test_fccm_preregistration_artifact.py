from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import (
    preregister_funding_currency_custody_mobility_consensus as fccm,
)


ARTIFACT = Path(
    "results/funding_currency_custody_mobility_consensus_"
    "preregistration_2026-07-23.json"
)
ARTIFACT_SHA256 = (
    "90c7cb4e110ddb15466702414a0cfbbac9fed681cba0922095817524560ac204"
)
MANIFEST_HASH = (
    "b33a786aaf9e9c2457e07eaebd0771c9c82971ccdea25676e2a6a9f8bfe2ddf1"
)
POLICY_HASH = (
    "7e22d0b2559ae6c509ee45e6a8c5bb81a71501a305406c5951c295c4e7376ea3"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "b790789cf6d500bdf77e0c2f48fd8fbc2a0e8591689a4625e84acf423c393432"
)


def test_fccm_preregistration_artifact_is_exact_and_reproducible() -> None:
    raw = ARTIFACT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ARTIFACT_SHA256
    payload = json.loads(raw)
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}

    assert payload["manifest_hash"] == MANIFEST_HASH
    assert fccm.canonical_hash(core) == MANIFEST_HASH
    assert payload["policy_hash"] == POLICY_HASH
    assert payload["policy_hash"] == fccm.canonical_hash(payload["policy"])
    assert payload["preregistration_source"] == {
        "path": str(fccm.SCRIPT_PATH),
        "sha256": PREREGISTRATION_SOURCE_SHA256,
    }
    assert fccm.sha256_file(fccm.SCRIPT_PATH) == PREREGISTRATION_SOURCE_SHA256

    reproduced, status = fccm.write_preregistration()
    assert status == "verified_existing"
    assert reproduced == payload


def test_fccm_preregistration_artifact_keeps_every_outcome_boundary_closed() -> None:
    payload = json.loads(ARTIFACT.read_bytes())

    assert payload["candidate"] == "FCCM-72"
    assert payload["artifact_eligible"] is True
    assert payload["verification_mode"] == (
        "verified_hashes_headers_and_commit_guard"
    )
    assert payload["fccm_source_values_or_incidence_opened"] is False
    assert payload["comparator_rows_opened_during_preregistration"] is False
    assert payload["outcomes_opened"] is False
    assert payload["performance_values_opened"] is False
    assert payload["outcome_boundary"] == fccm.EXPECTED_BOUNDARY
    assert all(
        binding["value_rows_read_during_preregistration"] == 0
        for binding in payload["source_bindings"].values()
    )
    assert all(
        binding["value_rows_read_during_preregistration"] == 0
        for binding in payload["comparator_bindings"]
    )
