from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import preregister_intrinsic_volume_price_lag_handoff as ivplh


ARTIFACT = Path(ivplh.DEFAULT_OUTPUT)
ARTIFACT_SHA256 = "942f519f87a86f7c4764b01aca1f2e4b524749888620a3aded51040970163551"
MANIFEST_HASH = "a647a944a65b46fa52799d544acfee1a4c8c72722e05b54c3cf891c2537f0619"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ivplh_preregistration_artifact_is_exact_and_outcome_blind() -> None:
    assert _sha256(ARTIFACT) == ARTIFACT_SHA256
    assert ARTIFACT.read_text(encoding="utf-8") == ivplh._canonical_manifest_text()
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    ivplh.validate_manifest(payload)

    assert payload == ivplh.build_manifest()
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["policy"]["policy_id"] == "IVPLH-72"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["predecessor_rows_decoded"] is False
    assert payload["comparator_rows_decoded"] is False
    assert payload["research_history_boundary"]["source_seen_successor"] is True
    assert payload["strict_sequence"]["stop_at_first_failure"] is True
    assert payload["strict_sequence"]["no_parameter_repair"] is True


def test_ivplh_preregistration_artifact_dependencies_remain_frozen() -> None:
    ivplh.validate_frozen_dependencies()
    assert len(ivplh.frozen_dependencies()) == 14
