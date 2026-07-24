from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd
import pytest

from training import preregister_paired_intrinsic_venue_orderflow_topology as p


def _tokens() -> dict[str, str]:
    return {
        "leader": "SPOT",
        "gap_q": "Q1",
        "early_session": "S18_24",
        "laggard_progress_q": "Q2",
        "spot_early_sign": "POS",
        "um_early_sign": "NEG",
        "spot_late_sign": "ZERO",
        "um_late_sign": "POS",
        "spot_late_abs_flow_q": "Q3",
        "um_late_abs_flow_q": "Q0",
        "gap_change": "WIDEN",
        "leader_change": "SWITCH",
    }


def test_manifest_is_source_incidence_comparator_and_outcome_blind() -> None:
    payload = p.build_manifest()
    p.validate_manifest(payload)
    assert payload["policy"]["policy_id"] == "PIVOT-72"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["source_rows_decoded"] is False
    assert payload["market_value_rows_decoded"] is False
    assert payload["funding_value_rows_decoded"] is False
    assert payload["comparator_rows_decoded"] is False
    assert payload["post_2023_values_decoded"] is False


def test_policy_freezes_all_state_latency_and_hold() -> None:
    payload = p.build_manifest()
    policy = payload["policy"]
    execution = payload["execution_contract"]
    assert policy["reference_calendar_days"] == 28
    assert policy["reference_complete_days_min"] == 21
    assert policy["intrinsic_volume_fraction"] == 0.50
    assert policy["latest_anchor_start_minute_utc"] == 23 * 60 + 50
    assert policy["prior_base_states"] == 180
    assert policy["prior_base_states_min"] == 90
    assert policy["entry_delay_bars_from_late_anchor"] == 3
    assert policy["hold_bars"] == 72
    assert execution["buffer_completion"] == "A_late+10m"
    assert execution["entry"].endswith("A_late+15m")
    assert execution["global_action_independent_reservation"] is True
    assert execution["abstention_releases_reservation"] is False


def test_latest_anchor_keeps_real_inference_window_before_entry() -> None:
    times = p.opportunity_times("2023-06-30T23:50:00Z")
    assert times["state_completion"] == pd.Timestamp("2023-06-30T23:55:00Z")
    assert times["buffer_completion"] == pd.Timestamp("2023-07-01T00:00:00Z")
    assert times["entry"] == pd.Timestamp("2023-07-01T00:05:00Z")
    assert times["exit"] == pd.Timestamp("2023-07-01T06:05:00Z")


def test_exact_zero_is_preserved_and_nonfinite_sign_is_rejected() -> None:
    assert p.sign_token(-1.0) == "NEG"
    assert p.sign_token(-0.0) == "ZERO"
    assert p.sign_token(0.0) == "ZERO"
    assert p.sign_token(1.0) == "POS"
    with pytest.raises(ValueError, match="finite"):
        p.sign_token(float("nan"))


def test_prior_quartile_uses_linear_boundaries_and_maps_ties_upward() -> None:
    prior = np.arange(100, dtype=np.float64)
    assert p.prior_quartile_bucket(24.74, prior) == "Q0"
    assert p.prior_quartile_bucket(24.75, prior) == "Q1"
    assert p.prior_quartile_bucket(49.50, prior) == "Q2"
    assert p.prior_quartile_bucket(74.25, prior) == "Q3"
    with pytest.raises(ValueError, match="history"):
        p.prior_quartile_bucket(1.0, prior[:89])


def test_duplicate_quartile_boundaries_are_not_repaired() -> None:
    prior = np.zeros(90, dtype=np.float64)
    assert p.prior_quartile_bucket(-1.0, prior) == "Q0"
    assert p.prior_quartile_bucket(0.0, prior) == "Q3"


def test_sign_mirror_is_involutive_and_swaps_action() -> None:
    tokens = _tokens()
    mirrored = p.sign_mirror_tokens(tokens)
    assert mirrored["spot_early_sign"] == "NEG"
    assert mirrored["um_early_sign"] == "POS"
    assert mirrored["spot_late_sign"] == "ZERO"
    assert p.sign_mirror_tokens(mirrored) == tokens
    assert p.sign_mirror_action("LONG") == "SHORT"
    assert p.sign_mirror_action("SHORT") == "LONG"
    assert p.sign_mirror_action("ABSTAIN") == "ABSTAIN"


def test_venue_swap_is_token_equivariance_not_action_augmentation() -> None:
    tokens = _tokens()
    swapped = p.venue_swap_tokens(tokens)
    assert swapped["leader"] == "UM"
    assert swapped["spot_early_sign"] == tokens["um_early_sign"]
    assert swapped["um_late_abs_flow_q"] == tokens["spot_late_abs_flow_q"]
    assert p.venue_swap_tokens(swapped) == tokens
    symmetry = p.build_manifest()["gemma_contract"]["training_symmetry"]
    assert symmetry["views"] == ["identity", "sign_mirror"]
    assert symmetry["venue_swap"] == "source-builder equivariance control only"


def test_action_option_orders_are_all_six_and_date_independent() -> None:
    orders = p.action_option_orders()
    assert len(orders) == 6
    assert len(set(orders)) == 6
    assert all(set(order) == {"LONG", "SHORT", "ABSTAIN"} for order in orders)
    assert orders[0] == ("ABSTAIN", "LONG", "SHORT")
    assert orders[-1] == ("SHORT", "LONG", "ABSTAIN")
    prompt = p.build_manifest()["gemma_contract"]["prompt"]
    assert prompt["option_orders"] == [list(order) for order in orders]
    assert prompt["generation"] is False


def test_source_support_floors_match_disclosed_precedent_headroom() -> None:
    support = p.build_manifest()["source_support_gates"]
    assert support["global_opportunities_min"] == 750
    assert support["train_opportunities_min"] == 350
    assert support["selection_opportunities_min"] == 200
    assert support["eval_opportunities_min"] == 200
    assert support["selection_eval_each_half_min"] == 85
    assert support["train_2020_active_months_min"] == 7
    assert support["train_2021_active_months_min"] == 12
    assert support["selection_2022_active_months_min"] == 12
    assert support["eval_2023_active_months_min"] == 12
    assert support["each_quartile_share_range"] == [0.10, 0.40]
    assert support["downstream_levels_must_exist_in_train"] is True
    assert support["required_builder_test_scopes"] == ["synthetic", "real_prefix"]
    assert support["required_builder_tests"] == [
        "venue_swap",
        "sign_mirror",
        "future_append",
        "current_value_exclusion_from_every_prior_quartile",
        "suppressed_state_inclusion_in_prior_history",
        "missing_prefix",
        "exact_anchor_tie",
        "exact_zero_sign_preservation",
        "duplicate_quartile_boundaries",
        "option_order_independence_at_serialization",
        "action_independent_reservation",
    ]
    assert support["failure_action"].startswith("retire PIVOT-72 unchanged")


def test_economics_baselines_and_gemma_are_fully_frozen() -> None:
    payload = p.build_manifest()
    economics = payload["economic_contract"]
    baseline = payload["baseline_contract"]
    gemma = payload["gemma_contract"]
    assert economics["base_cost_notional_per_side"] == 0.0006
    assert economics["stress_cost_notional_per_side"] == 0.0010
    assert economics["stress_replaces_base"] is True
    assert economics["zero_mdd_ratio"]["positive_cagr_cap"] == 1.0e12
    assert "side*(exit_open/entry_open-1)" in economics[
        "mean_gross_underlying_move_bp"
    ]
    assert baseline["policies"]["ridge_contextual_value"]["alpha"] == 100.0
    assert baseline["policies"]["extra_trees_contextual_value"][
        "n_estimators"
    ] == 512
    assert len(baseline["policies"]["shuffled_oracle_label"]["seeds"]) == 32
    assert gemma["model"] == "google/gemma-4-E2B-it"
    assert gemma["revision"] == "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
    assert gemma["quantization"]["double_quant"] is True
    assert gemma["lora"]["targets"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert gemma["sft"]["optimizer_steps"] == 64
    assert gemma["dpo"]["checkpoints"] == [24, 48, 72, 96]


def test_novelty_contract_is_pre2024_and_forbidden_paths_remain_forbidden() -> None:
    novelty = p.build_manifest()["novelty_contract"]
    comparators = {item["id"]: item for item in novelty["static_comparators"]}
    live = comparators["CCHR-live-pre2024"]
    assert live["rows"] == 440
    assert live["declared_coverage"][1] == "2024-01-01T00:00:00Z"
    assert live["post_2023_row_policy"].startswith("hard fail")
    assert comparators["CVTT-V01-V04"][
        "selection_absence_is_not_failure"
    ] is True
    assert novelty["exact_entry_jaccard_max"] == 0.10
    assert novelty["one_bar_tolerant_jaccard_max"] == 0.20
    assert novelty["twelve_bar_tolerant_jaccard_max"] == 0.35
    assert novelty["absolute_signed_occupancy_pearson_max"] == 0.40
    assert novelty["forbidden_paths"] == list(p.FORBIDDEN_COMPARATOR_PATHS)
    assert novelty["carta"]["execution_order"].startswith("only after PIVOT")


def test_frozen_dependency_hashes_and_headers_match_without_decoding_rows() -> None:
    dependencies = p.frozen_dependencies()
    assert dependencies[p.BOUNDARY_DOCUMENT] == p.BOUNDARY_DOCUMENT_SHA256
    assert dependencies[p.MECHANISM_DOCUMENT] == p.MECHANISM_DOCUMENT_SHA256
    assert dependencies[p.SOURCE] == p.SOURCE_SHA256
    assert p.FORBIDDEN_COMPARATOR_PATHS[0] not in dependencies
    assert p.FORBIDDEN_COMPARATOR_PATHS[1] not in dependencies
    p.validate_frozen_dependencies()
    assert p.sha256_csv_header(p.SOURCE) == p.SOURCE_HEADER_SHA256
    assert set(p.SOURCE_ALLOWLIST).issubset(p.csv_header(p.SOURCE))


def test_header_reader_does_not_decode_later_csv_rows(tmp_path) -> None:
    plain = tmp_path / "clock.csv"
    plain.write_bytes(b"a,b\n\xff\xfe\x00not-csv")
    assert p.csv_header(plain) == ["a", "b"]

    compressed = tmp_path / "clock.csv.gz"
    with gzip.open(compressed, "wb") as handle:
        handle.write(b"x,y\n\xff\xfe\x00not-csv")
    assert p.csv_header(compressed) == ["x", "y"]


def test_write_once_is_reproducible_and_rejects_drift(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(
        p,
        "validate_frozen_dependencies",
        lambda: calls.append(True),
    )
    first = p.build_manifest()
    second = p.build_manifest()
    assert first == second
    output = tmp_path / "freeze.json"
    assert p.write_once(output, first) == "created"
    assert calls == [True]
    assert output.read_text(encoding="utf-8") == p._canonical_manifest_text()
    assert p.write_once(output, second) == "verified_existing"
    stored = json.loads(output.read_text())
    stored["policy"]["hold_bars"] = 71
    with pytest.raises(RuntimeError, match="hash mismatch"):
        p.validate_manifest(stored)


def test_write_once_rejects_reformatted_existing_manifest(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(p, "validate_frozen_dependencies", lambda: None)
    payload = p.build_manifest()
    output = tmp_path / "freeze.json"
    output.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="noncanonical existing"):
        p.write_once(output, payload)


def test_validate_manifest_rejects_self_rehashed_non_policy_drift() -> None:
    payload = p.build_manifest()
    payload["novelty_contract"]["one_bar_tolerant_jaccard_max"] = 0.21
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = p.canonical_hash(core)
    with pytest.raises(RuntimeError, match="core differs from code"):
        p.validate_manifest(payload)


def test_strict_sequence_retains_unit_commit_rule() -> None:
    sequence = p.build_manifest()["strict_sequence"]
    assert sequence[-1] == "commit every completed unit with hashes and fresh tests"
