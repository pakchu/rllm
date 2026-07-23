from __future__ import annotations

import json
from pathlib import Path

from training import preregister_soma_collateral_allocation_fracture as p


ARTIFACT = Path(
    "results/soma_collateral_allocation_fracture_"
    "preregistration_2026-07-24.json"
)
ARTIFACT_SHA256 = (
    "1542ed321e8fc64f49aeea6f2582db64f3cba31bfabf459425427874e37dfaca"
)
MANIFEST_HASH = (
    "fd22849e9a92476fc2e08805bd7e634cc478592238a087ef5a67c121d51f1a44"
)


def test_scaf_preregistration_artifact_is_exact_and_canonical() -> None:
    assert p.sha256_file(ARTIFACT) == ARTIFACT_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload == p.build_manifest()
    assert payload["manifest_hash"] == MANIFEST_HASH
    p.validate_manifest(payload)


def test_scaf_preregistration_artifact_is_preincidence_and_outcome_blind() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["policy"]["policy_id"] == "SCAF-48"
    assert payload["source_incidence_opened"] is False
    assert payload["comparator_rows_decoded"] is False
    assert payload["outcomes_opened"] is False
    assert all(
        value in (0, False)
        for value in payload["evidence_boundary"].values()
    )
    assert payload["strict_sequence"][:4] == [
        "boundary_commit",
        "schema_amendment_commit",
        "mechanism_commit",
        "write_once_preregistration_commit",
    ]
