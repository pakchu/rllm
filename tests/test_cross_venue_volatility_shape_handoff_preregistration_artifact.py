from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import preregister_cross_venue_volatility_shape_handoff as p


ARTIFACT = Path(
    "results/"
    "cross_venue_volatility_shape_handoff_preregistration_2026-07-30.json"
)
ARTIFACT_SHA256 = (
    "9e6901e67e36d6d9170dae54548419cbfb9178cf84338e3c5652012896ee6604"
)
MANIFEST_HASH = (
    "dd3e334939b24cb5508c66ed9787c2c3e1a0ad006dcadcbfb673e541d6520cad"
)
PRODUCER_COMMIT = "e7f973e305d80dc0f9a1fabbd20ec92f8776d611"


def load() -> tuple[bytes, dict]:
    raw = ARTIFACT.read_bytes()
    return raw, json.loads(raw)


def test_canonical_artifact_has_exact_write_once_identity() -> None:
    raw, payload = load()
    assert hashlib.sha256(raw).hexdigest() == ARTIFACT_SHA256
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["repository"]["commit"] == PRODUCER_COMMIT
    assert payload["repository"]["upstream_commit"] == PRODUCER_COMMIT
    assert payload["repository"]["canonical_remote_commit"] == PRODUCER_COMMIT
    assert payload["repository"]["tracked_clean"] is True
    assert payload["repository"]["upstream_exact"] is True
    assert p.encoded_registration(payload) == raw
    p.validate_registration(payload)


def test_canonical_artifact_proves_source_blind_boundary() -> None:
    _, payload = load()
    boundary = payload["research_boundary"]
    assert boundary["bvol_or_dvol_rows_decoded"] == 0
    assert boundary["candidate_incidence_opened"] is False
    assert boundary["comparator_rows_decoded"] == 0
    assert boundary["gross9_clock_rows_opened"] == 0
    assert boundary["btc_execution_rows_opened"] == 0
    assert boundary["funding_rows_opened"] == 0
    assert boundary["return_pnl_cagr_or_drawdown_opened"] is False
    for artifact in payload["source_contract"]["artifacts"].values():
        binding = artifact["validated_binding"]
        if "header" in binding:
            assert binding["rows_decoded"] == 0
    for comparator in payload["novelty"][
        "prior_volatility_comparators"
    ].values():
        assert comparator["validated_binding"]["rows_decoded"] == 0


def test_canonical_artifact_binds_complete_gross9_and_future_veto() -> None:
    _, payload = load()
    gross9 = payload["gross9"]
    assert gross9["authority_hash"] == p.GROSS9_AUTHORITY_HASH
    assert gross9["reference_preregistration"] == p.ESDI_PREREGISTRATION
    assert gross9["weights"] == p.GROSS9_WEIGHTS
    assert gross9["gross"] == 9.0
    assert sum(gross9["weights"].values()) == 9.0
    assert payload["same_gross"]["candidate_weights"] == ["1/4", "1/2", "3/4"]
    assert payload["same_gross"]["selection_periods_only"] == ["2023H2", "2024"]
    assert payload["same_gross"]["future"][
        "rerank_repair_or_alternate_weight"
    ] is False
    assert payload["gross9_evidence_boundary"][
        "future_rows_used_for_economic_weight_ranking"
    ] is False
    assert payload["gross9_evidence_boundary"][
        "future_rows_used_for_structural_candidate_veto"
    ] is True


def test_manifest_recomputes_from_every_frozen_field() -> None:
    _, payload = load()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert p.canonical_hash(core) == MANIFEST_HASH
