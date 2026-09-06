from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import preregister_blockspace_fee_witness_concordance as prereg


ARTIFACT = Path(
    "results/blockspace_fee_witness_concordance_preregistration_2026-07-30.json"
)
FILE_SHA256 = (
    "c255cccbda22cdc8c43e35f04f5d1792f0a76f88caa966434b5be79bff1f65f7"
)
MANIFEST_HASH = (
    "499bdcd199bfe8ae7dad9bf5e51271f8fb1fd762edbaf4a1e0d026708e9fdf9b"
)


def test_exact_preregistration_artifact_hash_and_build_manifest() -> None:
    raw = ARTIFACT.read_bytes()
    payload = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == FILE_SHA256
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload == prereg.build_manifest()
    prereg.validate_manifest(payload)


def test_preregistration_is_outcome_blind_and_preincidence() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["source_rows_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["candidate_overlap_opened"] is False
    assert payload["economic_rows_opened"] is False
    assert payload["outcomes_opened"] is False
    disclosure = payload["evidence_disclosure"]
    assert disclosure["csv_data_rows_decoded_by_preregistration"] == 0
    assert disclosure["candidate_exact_incidence_opened"] is False
    assert disclosure["candidate_outcomes_opened"] is False


def test_preregistration_freezes_exact_sources_ranks_and_sequence() -> None:
    payload = prereg.build_manifest()
    assert payload["policy"]["rank_history_max"] == 180
    assert payload["policy"]["rank_history_min"] == 120
    assert payload["policy"]["hold_bars_5m"] == 288
    assert payload["support_gates"]["exact_join_gaps"] == 0
    assert payload["novelty"]["opens_only_after_support"] is True
    assert payload["strict_sequence"][3:5] == [
        "dependency_and_exact_header_validation",
        "source_support_and_future_append_invariance",
    ]
