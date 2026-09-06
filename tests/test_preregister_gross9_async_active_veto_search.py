from __future__ import annotations

import copy
import hashlib
import subprocess
from itertools import permutations

import pytest

from training import preregister_gross9_async_active_veto_search as p


def _rows() -> list[dict[str, object]]:
    return [
        {
            "candidate": candidate,
            "source_pass": True,
            "exact_duplicate_pass": True,
            "gross9_pass": True,
            "train_cagr_to_strict_mdd": float(index + 1),
            "train_absolute_return": 0.01 * (index + 1),
            "train_economic_pass": True,
        }
        for index, candidate in enumerate(p.CANDIDATE_FAMILY)
    ]


def _rehash(value: dict[str, object]) -> dict[str, object]:
    core = dict(value)
    core.pop("manifest_hash")
    value["manifest_hash"] = p.canonical_hash(core)
    return value


def test_build_fixes_exact_9_component_all_72_ordered_active_veto_family_and_clock() -> None:
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "G9ASYNCACTIVEVETO-8"
    assert value["component_order"] == list(p.COMPONENT_ORDER)
    assert value["component_count"] == 9
    assert value["candidate_family_size"] == 72
    assert value["candidate_family"] == [
        f"{base}__ASYNC_ACTIVE_OPPOSITE_VETO_6H__{veto}"
        for base, veto in permutations(p.COMPONENT_ORDER, 2)
    ]
    construction = value["construction"]
    assert "all 72 ordered A!=B" in construction["ordered_pair_definition"]
    assert construction["candidate_id_format"] == "A__ASYNC_ACTIVE_OPPOSITE_VETO_6H__B"
    assert "strict lower t-6h < v.entry_time <= t" in construction["base_event_rule"]
    assert "same-side" in construction["veto_rule"]
    assert "opposite-side" in construction["veto_rule"]
    assert "cash/no row" in construction["veto_rule"]
    assert "never reverse side" in construction["veto_rule"]
    assert construction["latest_supersedes_older"] is True
    assert construction["same_timestamp_veto_allowed"] is True
    assert "half-open 8h reservation" in construction["reservation"]
    assert value["clock"]["hold"] == "8 elapsed hours"
    assert value["clock"]["gross_exposure"] == 0.5


def test_binds_same_components_gross9_prior_receipts_source_support_and_builder_placeholder() -> None:
    value = p.build()
    assert set(value["implementation"]) == {"preregister", "train_clock_builder"}
    assert p.sha256_file(value["implementation"]["preregister"]["path"]) == value["implementation"]["preregister"]["sha256"]
    builder = value["implementation"]["train_clock_builder"]
    assert builder["path"] == "training/build_gross9_async_active_veto_train_clocks.py"
    assert builder["sha256"] == p.BUILDER_SHA256
    assert "status" not in builder
    assert set(value["component_artifacts"]) == set(p.COMPONENT_ORDER)
    assert value["gross9_pre2025_clock_manifest"] == p.GROSS9_PRE2025_CLOCK_MANIFEST
    assert value["predecessor_terminal_receipts"] == p.PREDECESSOR_TERMINAL_RECEIPTS
    assert value["prior_source_support_artifacts"] == p.PRIOR_SOURCE_SUPPORT_ARTIFACTS
    preliminary = value["preliminary_source_materialization_receipt"]
    assert preliminary == p.PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT
    assert preliminary["commit"] == "1bfddd3c"
    assert preliminary["sha256"] == "ce95d6373655ded0daba9d6f5635908106337827fbf1a98c978cf41d8231e6e3"
    assert preliminary["manifest_hash"] == "88ce540e6ce329e0d9f763c128b2f431949c191772d207b6a1b6b65ee4fb3e6d"
    assert preliminary["builder"]["sha256"] == "bf8bfaf41d0ca761a2bc0f2db53de5ad05103fe596b880eb0fd8acbbbc6c90df"
    artifact = preliminary["preregistration_artifact_with_placeholder_builder_binding"]
    assert artifact["path"] == "results/gross9_async_active_opposite_veto_search_preregistration_2026-09-02.json"
    assert artifact["sha256"] == "b70dbeea6a6d1bde63ea60c854fcfa09688060bf56c0f5a08f3f21073a5f4cba"
    assert artifact["manifest_hash"] == "8cc95042fe2e76c5193f3679f6d1f073e2e97bd0a69b658c57599b9fba06ba28"
    assert artifact["tracked_at_commit"] is False
    code = preliminary["untracked_preregistration_code_with_placeholder_builder_binding"]
    assert code["path"] == "training/preregister_gross9_async_active_veto_search.py"
    assert code["sha256"] == "f14becdeb93904c581cc89809ec161a692aea6e25b66ad7ae8c718584cc6ec59"
    assert code["tracked_at_commit"] is False
    assert preliminary["placeholder_builder_value"] == "PENDING_G9ASYNCACTIVEVETO_BUILDER_FOLLOWUP"
    assert preliminary["support_count_disclosure"]["passed_candidates"] == 14
    assert preliminary["support_count_disclosure"]["used_to_retune_family_operator_gates_thresholds_or_order"] is False
    triple = value["predecessor_terminal_receipts"][2]
    assert triple["sha256"] == "0b9c9366d0d0d214787e1fdb6f3fad9e604e2dbfad49fedd8f4b84fadbcb5265"
    assert triple["manifest_hash"] == "de32635d2e1853359cfe62ca4ef779442fec0dd29caf680433693f07ce6b6495"
    assert "not tuning inputs" in triple["schedule_scope"]


def test_source_duplicate_gross9_bonferroni_and_boundaries_are_frozen() -> None:
    value = p.build()
    source = value["source_support_gates"]
    assert source["minimum_events"]["train"] == 10
    assert source["minority_side_share_min"] == 0.20
    assert source["max_month_share"] == 0.45
    assert source["distinct_iso_weeks_min"] == 10
    assert source["each_calendar_half_min_events"] == 1
    assert source["opposite_suppressions_min"] == 1
    assert source["source_clock_availability_and_duplicate_drift_hard_fail"] is True
    assert "current 72" in source["current_family_exact_duplicate_gate"]
    assert "nine normalized" in source["normalized_base_control_exact_duplicate_gate"]
    assert "same-side36, handoff36, and triple84" in source["prior_family_exact_duplicate_gate"]
    assert source["empty_schedule_is_duplicate"] is False
    assert source["nonexact_overlap_disclosure_only"] is True
    construction = value["construction"]
    assert "disclosure-only" in construction["normalized_base_controls"]
    assert "reject all members" in construction["current_family_exact_duplicate_gate"]
    assert "nine normalized base controls" in construction["prior_schedule_exact_duplicate_gate"]
    assert "nonexact overlap" in construction["overlap_disclosure"]
    assert "disclosure-only" in construction["overlap_disclosure"]
    assert value["economic_gates"]["weekly_signflip_one_sided_p_max"] == pytest.approx(0.10 / 72)
    assert value["familywise_multiplicity"]["number_of_hypotheses"] == 72
    boundary = value["research_boundary"]
    assert boundary["llm_path_paused"] is True
    assert boundary["design_family_operator_and_gates_fixed_before_preliminary_source_materialization"] is True
    assert boundary["source_incidence_and_support_counts_opened_before_committed_preregistration"] is True
    assert boundary["family_operator_gate_threshold_or_order_changed_after_preliminary_source_materialization"] is False
    assert boundary["preliminary_14_source_passes_used_to_retune"] is False
    assert boundary["gross9_market_funding_or_pnl_opened_by_preregistration"] is False
    assert boundary["triple_source_counts_or_economic_scalars_used_to_tune_operator_or_gates"] is False
    assert boundary["predecessor_economic_scalars_used_to_tune_operator_or_gates"] is False
    assert boundary["active_veto_combination_incidence_opened_by_preregistration"] is False
    assert boundary["active_veto_combination_outcomes_opened_by_preregistration"] is False
    assert boundary["market_or_funding_rows_opened_by_preregistration"] is False
    assert "adaptive exploratory" in boundary["classification"]


def test_select_train_winner_requires_all_72_source_duplicate_gross9_and_no_substitution() -> None:
    rows = _rows()
    rows[-1]["source_pass"] = False
    rows[-2]["exact_duplicate_pass"] = False
    rows[-3]["gross9_pass"] = False
    rows[-4]["train_cagr_to_strict_mdd"] = 99.0
    rows[-4]["train_absolute_return"] = 0.30
    rows[-5]["train_cagr_to_strict_mdd"] = 99.0
    rows[-5]["train_absolute_return"] = 0.30
    winner = p.select_train_winner(rows)
    assert winner["candidate"] == p.CANDIDATE_FAMILY[-5]
    assert winner["frozen_before_test"] is True
    assert winner["substitution_authorized"] is False
    assert winner["rerank_authorized"] is False

    with pytest.raises(ValueError, match="exactly 72 candidate rows"):
        p.select_train_winner(rows[:-1])

    duplicate = _rows()
    duplicate[-1]["candidate"] = duplicate[0]["candidate"]
    with pytest.raises(ValueError, match="candidate IDs must be unique strings"):
        p.select_train_winner(duplicate)

    missing = _rows()
    missing[-1]["candidate"] = "NOT_A_FROZEN_ACTIVE_VETO"
    with pytest.raises(ValueError, match="exact frozen family"):
        p.select_train_winner(missing)


def test_raw_rank_one_failure_is_terminal_without_substitution() -> None:
    rows = _rows()
    rows[-1]["train_economic_pass"] = False
    with pytest.raises(RuntimeError, match="raw rank one failed train; no substitution"):
        p.select_train_winner(rows)


def test_selector_rejects_non_boolean_flags_and_nonfinite_metrics() -> None:
    rows = _rows()
    rows[-1]["source_pass"] = "yes"
    with pytest.raises(ValueError, match="source_pass must be boolean"):
        p.select_train_winner(rows)

    rows = _rows()
    rows[-1]["exact_duplicate_pass"] = 1
    with pytest.raises(ValueError, match="exact_duplicate_pass must be boolean"):
        p.select_train_winner(rows)

    rows = _rows()
    rows[-1]["gross9_pass"] = None
    with pytest.raises(ValueError, match="gross9_pass must be boolean"):
        p.select_train_winner(rows)

    rows = _rows()
    rows[-1]["train_economic_pass"] = "pass"
    with pytest.raises(ValueError, match="train_economic_pass must be boolean"):
        p.select_train_winner(rows)

    for bad in (True, float("nan"), float("inf"), float("-inf")):
        rows = _rows()
        rows[-1]["train_cagr_to_strict_mdd"] = bad
        with pytest.raises(ValueError, match="ranking metrics"):
            p.select_train_winner(rows)


def test_validate_detects_manifest_family_gates_bindings_duplicate_and_boundary_drift() -> None:
    value = p.build()
    drifted = copy.deepcopy(value)
    drifted["clock"]["hold"] = "9 elapsed hours"
    with pytest.raises(RuntimeError, match="preregistration drift"):
        p.validate(drifted)

    drifted = p.build()
    drifted["candidate_family"] = drifted["candidate_family"][:-1]
    with pytest.raises(RuntimeError, match="candidate family drift"):
        p.validate(_rehash(drifted))

    drifted = p.build()
    drifted["source_support_gates"]["opposite_suppressions_min"] = 0
    with pytest.raises(RuntimeError, match="suppression gate drift"):
        p.validate(_rehash(drifted))

    drifted = p.build()
    drifted["construction"]["veto_rule"] = "opposite-side reverses"
    with pytest.raises(RuntimeError, match="veto semantics drift"):
        p.validate(_rehash(drifted))

    drifted = p.build()
    drifted["construction"]["current_family_exact_duplicate_gate"] = "keep first duplicate"
    with pytest.raises(RuntimeError, match="current duplicate gate drift"):
        p.validate(_rehash(drifted))

    drifted = p.build()
    drifted["implementation"]["train_clock_builder"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="builder binding drift"):
        p.validate(_rehash(drifted))

    drifted = p.build()
    drifted["preliminary_source_materialization_receipt"]["manifest_hash"] = "0" * 64
    with pytest.raises(RuntimeError, match="preliminary source materialization binding drift"):
        p.validate(_rehash(drifted))

    drifted = p.build()
    drifted["predecessor_terminal_receipts"][2]["manifest_hash"] = "0" * 64
    with pytest.raises(RuntimeError, match="predecessor terminal binding drift"):
        p.validate(_rehash(drifted))

    drifted = p.build()
    drifted["prior_source_support_artifacts"][0]["manifest_hash"] = "0" * 64
    with pytest.raises(RuntimeError, match="prior source-support binding drift"):
        p.validate(_rehash(drifted))

    drifted = p.build()
    drifted["research_boundary"]["triple_source_counts_or_economic_scalars_used_to_tune_operator_or_gates"] = True
    with pytest.raises(RuntimeError, match="triple tuning boundary drift"):
        p.validate(_rehash(drifted))

    drifted = p.build()
    drifted["research_boundary"]["preliminary_14_source_passes_used_to_retune"] = True
    with pytest.raises(RuntimeError, match="preliminary source-pass retune boundary drift"):
        p.validate(_rehash(drifted))


def test_embedded_hashes_match_current_bound_files() -> None:
    root = p.Path(__file__).resolve().parents[1]
    value = p.build()
    assert p.sha256_file(root / value["gross9_pre2025_clock_manifest"]["path"]) == value["gross9_pre2025_clock_manifest"]["sha256"]
    receipt = value["preliminary_source_materialization_receipt"]
    preliminary_blob = subprocess.run(
        ["git", "show", f"{receipt['commit']}:{receipt['path']}"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    assert hashlib.sha256(preliminary_blob).hexdigest() == receipt["sha256"]
    assert p.sha256_file(root / value["implementation"]["train_clock_builder"]["path"]) == value["implementation"]["train_clock_builder"]["sha256"]
    assert value["preliminary_source_materialization_receipt"]["builder"]["sha256"] == "bf8bfaf41d0ca761a2bc0f2db53de5ad05103fe596b880eb0fd8acbbbc6c90df"
    for predecessor in value["predecessor_terminal_receipts"]:
        assert p.sha256_file(root / predecessor["path"]) == predecessor["sha256"]
    for source_support in value["prior_source_support_artifacts"]:
        assert p.sha256_file(root / source_support["path"]) == source_support["sha256"]
    for artifacts in value["component_artifacts"].values():
        for binding in artifacts.values():
            assert p.sha256_file(root / binding["path"]) == binding["sha256"]
