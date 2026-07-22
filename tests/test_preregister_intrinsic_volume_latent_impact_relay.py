from __future__ import annotations

import json
from pathlib import Path

from training import preregister_intrinsic_volume_latent_impact_relay as p


ARTIFACT = Path(
    "results/intrinsic_volume_latent_impact_relay_preregistration_2026-07-23.json"
)


def test_manifest_is_outcome_and_incidence_blind() -> None:
    payload = p.build_manifest()
    p.validate_manifest(payload)
    assert payload["policy"]["policy_id"] == "IVLIR-72"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["causal_feature_contract"]["future_bar_used_by_signal"] is False


def test_policy_freezes_first_passage_side_and_execution() -> None:
    payload = p.build_manifest()
    policy = payload["policy"]
    assert policy["intrinsic_volume_fraction"] == 0.50
    assert policy["absolute_flow_quantile"] == 0.60
    assert policy["maximum_impact_quantile"] == 0.70
    assert policy["rolling_extrema_bars"] == 2_016
    assert policy["entry_delay_bars"] == 1
    assert policy["hold_bars"] == 72
    assert payload["causal_feature_contract"]["side"].startswith("LONG")
    assert payload["execution_contract"]["maximum_one_candidate_per_utc_day"]


def test_llm_cannot_create_or_retime_trades() -> None:
    payload = p.build_manifest()
    boundary = payload["llm_boundary"]
    assert boundary["allowed_only_after_standalone_train_and_selection_pass"]
    assert boundary["later_action_space"] == ["TRADE_FIXED_SIDE", "ABSTAIN"]
    assert {"candidate clock", "side", "entry", "hold"}.issubset(
        set(boundary["llm_may_not_change"])
    )


def test_write_once_is_reproducible(tmp_path) -> None:
    output = tmp_path / "freeze.json"
    first = p.build_manifest()
    assert p.write_once(output, first) == "created"
    stored = json.loads(output.read_text())
    p.validate_manifest(stored)
    second = p.build_manifest()
    assert first["manifest_hash"] == second["manifest_hash"]
    assert p.write_once(output, second) == "verified_existing"


def test_committed_artifact_matches_frozen_code() -> None:
    stored = json.loads(ARTIFACT.read_text())
    p.validate_manifest(stored)
    current = p.build_manifest()
    assert stored["manifest_hash"] == current["manifest_hash"]
