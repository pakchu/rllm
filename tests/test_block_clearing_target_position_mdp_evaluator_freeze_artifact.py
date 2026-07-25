from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import freeze_block_clearing_target_position_evaluator as freeze


ARTIFACT = Path(
    "results/block_clearing_target_position_mdp_"
    "evaluator_freeze_2026-07-25.json"
)
ARTIFACT_SHA256 = (
    "94eb948d1b83ac9f11cc698b7c6942ccb863cca396929523822291383f6a1633"
)
MANIFEST_HASH = (
    "25f80efcef9e8e5b08b9138dc17cb4c18edcc0b47187ad9b43f999acee0afd87"
)
FREEZE_COMMIT = "ea753aa98e914314c7df744a5c134a2bfb0c7251"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_evaluator_freeze_artifact_identity_is_exact() -> None:
    payload = _payload()
    assert _sha256(ARTIFACT) == ARTIFACT_SHA256
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["freeze_commit"] == FREEZE_COMMIT
    freeze.validate_manifest(payload)


def test_evaluator_freeze_opened_no_outcome_payload() -> None:
    payload = _payload()
    assert payload["outcome_boundary"] == {
        "market_rows_parsed": 0,
        "funding_rows_parsed": 0,
        "market_or_funding_payload_bytes_hashed": False,
        "future_returns_created": 0,
        "rewards_created": 0,
        "models_fit": 0,
        "economic_metrics_computed": 0,
        "bctp_2023_market_outcomes_opened": False,
        "post_2023_source_or_outcomes_opened": False,
    }
    assert payload["mutable_parameters"] == []
    for source in payload["execution_sources"].values():
        assert source["payload_bytes_hashed_during_freeze"] is False


def test_evaluator_freeze_binds_complete_policy_family() -> None:
    payload = _payload()
    assert payload["policy_id"] == freeze.POLICY_ID
    assert payload["family_size"] == 31
    assert tuple(payload["family_ids"]) == freeze.FAMILY_IDS
    assert payload["cheap_gates"]["promotable_primary_ids"] == [
        "categorical_linear_fqi",
        "categorical_ridge_fqi",
        "extra_trees_fqi",
    ]
    assert payload["cheap_gates"]["2022_reselection_allowed"] is False
    assert payload["statistical_test"] == {
        "exact_cluster_max": 20,
        "draws": 100_000,
        "seed": 20_260_725,
        "batch_draws": 2_000,
        "cluster": "monday_00_utc_half_open_week",
        "alternative": "one_sided_positive_studentized_mean",
    }


def test_evaluator_freeze_binds_exact_accounting_and_reward() -> None:
    payload = _payload()
    accounting = payload["accounting"]
    assert accounting["target_absolute_gross"] == 0.5
    assert accounting["base_cost_rate"] == 0.0006
    assert accounting["stress_cost_rate"] == 0.0010
    assert accounting["funding_boundary_rule"] == (
        "min(0,old_cash,new_cash)"
    )
    assert accounting["held_bar_order"] == "favorable_then_adverse"
    assert payload["reward"]["downside_linear_coefficient"] == 1.0 / 3.0
    assert payload["reward"]["target_change_coefficient"] == 0.001
    assert payload["reward"]["future_best_label_used"] is False


def test_evaluator_freeze_keeps_gemma_conditional_and_2023_immutable() -> None:
    payload = _payload()
    gemma = payload["gemma"]
    assert gemma["authorized_now"] is False
    assert gemma["repository"] == "google/gemma-4-E4B"
    assert gemma["revision"] == (
        "9f9f0f28c85251b6616672841d041635e1763f13"
    )
    assert gemma["action_tokens"] == ["A", "B", "C"]
    assert gemma["action_token_ids"] == [236_776, 236_799, 236_780]
    assert payload["stage_protocol"]["2023_may_select_or_repair"] is False
    assert payload["stage_protocol"]["post_2023_opened"] is False
    assert payload["authorized_next_stage"] == (
        "implement_and_verify_frozen_economic_runner"
    )
