from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ARTIFACT = Path(
    "results/stablecoin_quote_flow_diffusion_preregistration_2026-07-19.json"
)
EXPECTED_FILE_SHA256 = (
    "3fed620146b98e920175445a12e2a8684c2a3431e42b1a784ea0e3076577aee3"
)
EXPECTED_MANIFEST_HASH = (
    "74fd535b8b2256d41e39513466bd697d553ee5c80aece8308e3de637745225b3"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_sqfd_preregistration_is_frozen_and_outcome_blind() -> None:
    assert _sha256(ARTIFACT) == EXPECTED_FILE_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert payload["manifest_hash"] == _canonical_hash(core)
    assert payload["outcomes_opened"] is False
    assert payload["policy"]["policy_id"] == "SQFD-6"
    assert payload["research_history_boundary"][
        "exact_sqfd_post_entry_outcomes_opened"
    ] is False
    assert payload["research_history_boundary"]["threshold_selection_repair"] == {
        "grid_descending": [1.25, 1.0, 0.75, 0.5],
        "selection_window": "train_2023_h2_only",
        "selection_rule": (
            "first threshold with >=50 events, >=35% each side, "
            "<=30% maximum train-month share"
        ),
        "train_event_counts": [13, 28, 55, 104],
        "selected": 0.75,
        "post_2023_support_cannot_change_rule": True,
    }


def test_sqfd_sources_and_comparator_clocks_are_hash_bound() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    source = payload["source_contract"]
    assert _sha256(Path(source["panel"])) == source["panel_sha256"]
    assert _sha256(Path(source["manifest"])) == source["manifest_sha256"]
    assert _sha256(Path(source["builder"])) == source["builder_sha256"]
    for comparator in payload["support_comparators"]["clocks"]:
        assert _sha256(Path(comparator["path"])) == comparator["sha256"]


def test_sqfd_math_reservation_and_statistical_test_are_unambiguous() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    feature = payload["causal_feature_contract"]
    execution = payload["execution_contract"]
    test = payload["outcome_gate"]["statistical_test_contract"]
    assert "shift(1)" in feature["strict_prior_history"]
    assert "interpolation=linear" in feature["strict_prior_history"]
    assert "entry_time >= prior accepted exit_time" in execution["reservation_order"]
    assert test["cluster_key"] == "UTC entry timestamp ISO year/week"
    assert test["seed"] == 20260719
    assert test["draws"] == 20_000
    assert test["empty"] == "p=1.0"
