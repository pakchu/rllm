from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from training import preregister_radial_quote_handoff_relay as rqhr


def test_policy_freezes_exact_mechanism_support_and_sequence() -> None:
    policy = rqhr.policy_payload()
    assert policy["candidate"] == "RQHR-72"
    assert policy["thresholds"]["near_quantile"] == "39/40"
    assert policy["thresholds"]["far_quantile"] == "9/10"
    assert policy["thresholds"]["strict_prior_grid_rows"] == 8_640
    assert policy["thresholds"]["minimum_valid_prior_values"] == 4_032
    assert policy["race"]["state"]["eligible_elapsed_grid_bars"] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert policy["race"]["terminal_rules"]["terminal_bar_consumed"] is True
    assert policy["execution"]["processing_latency_bars"] == 1
    assert policy["execution"]["hold_bars_5m"] == 72
    assert policy["execution"]["chronological_reservation"] is True
    assert policy["execution"]["accept_rule"] == (
        "entry >= previous accepted exit"
    )
    assert policy["execution"]["reservation_reset"] == (
        "only at known UTC quarter boundary"
    )
    assert policy["source_support_gates"]["total_minimum"] == 120
    assert policy["source_support_gates"]["each_side_share_minimum"] == "7/20"
    assert policy["mechanical_nulls"]["maximum_raw_confirmations_each_scenario"] == 0
    assert policy["mechanical_nulls"]["maximum_accepted_events_each_scenario"] == 0
    assert policy["novelty"]["parse_all_raw_rows_before_window_filter"] is True
    assert policy["novelty"]["legacy_display_time_zone_trusted"] is False
    assert policy["strict_economic_gates"]["recent"] == [
        "2026-01-01T00:00:00Z",
        "2026-07-19T00:00:00Z",
    ]
    assert policy["mutable_parameters"] == []


def test_race_contract_is_complete_and_fail_closed() -> None:
    race = rqhr.policy_payload()["race"]
    arm = race["arm"]
    assert arm["current_row_complete"] is True
    assert arm["near_sign_allowed"] == ["+1", "-1"]
    assert arm["near_efficiency_minimum"] == "3/5"
    assert arm["near_intensity_relation"] == ">= current near_threshold"
    assert arm["previous_grid_row_complete"] is True
    assert arm["previous_near_threshold_available"] is True
    assert arm["previous_near_intensity_relation"] == (
        "< previous row's own near_threshold"
    )
    assert arm["same_sign_far_blocks_when"] == {
        "far_sign": "near_sign",
        "far_efficiency_minimum": "1/2",
        "far_intensity_relation": ">= current far_threshold",
    }
    assert arm["no_active_race"] is True
    assert arm["queued_or_replaced_by_later_arm"] is False

    confirmation = race["confirmation"]
    assert confirmation["far_sign"] == "stored arm_sign"
    assert confirmation["far_efficiency_minimum"] == "1/2"
    assert confirmation["cumulative_interval"] == (
        "arm bar through current bar inclusive"
    )
    assert confirmation["cumulative_value"] == (
        "exact sum of skew_2_net+skew_3_net"
    )
    assert confirmation["cumulative_sign"] == "strictly stored arm_sign"
    assert race["cancellation"] == {
        "near_sign": "opposite stored arm_sign",
        "near_efficiency_minimum": "1/2",
        "creates_candidate": False,
    }
    terminal = race["terminal_rules"]
    assert terminal["simultaneous_confirmation_and_cancellation"] == (
        "ambiguous; no event; race immediately retired"
    )
    assert terminal["incomplete_grid_row"] == "cancel immediately; no event"
    assert terminal["no_terminal_by_age_6"] == "timeout; no event"
    assert terminal["rearm_earliest"] == "next elapsed grid row"


def test_every_control_is_machine_frozen() -> None:
    controls = rqhr.policy_payload()["source_controls"]
    assert set(controls) == {
        "common",
        "simultaneous_near_far",
        "far_to_near_reverse_relay",
        "no_efficiency_relay",
        "near_only",
        "far_only",
        "stale_final_signals",
        "quarter_far_triple_permutation",
        "side_controls",
        "primary_must_beat_every_control",
    }
    reverse = controls["far_to_near_reverse_relay"]
    assert reverse["previous_grid_row_complete"] is True
    assert reverse["previous_far_intensity_relation"] == (
        "< previous row's own far_threshold"
    )
    assert reverse["near_already_qualified_blocks_when"]["near_sign"] == (
        "arm_far_sign"
    )
    assert reverse["confirmation"]["cumulative_value"] == (
        "exact sum of skew_4_net+skew_5_net"
    )
    assert reverse["terminal_rules_identical_to_primary"] is True

    stale = controls["stale_final_signals"]
    assert stale["shift_elapsed_grid_rows"] == {
        "one_bar_stale": 1,
        "five_bar_stale": 5,
    }
    assert stale["destination_row_must_be_complete"] is True
    assert stale["outside_2023_or_incomplete_destination"] == (
        "drop; never search forward"
    )

    permutation = controls["quarter_far_triple_permutation"]
    assert len(permutation["tuple"]) == 6
    assert permutation["recipient_availability_preserved"] is True
    assert permutation["recompute"] == [
        "all far features",
        "strict-prior far thresholds",
        "full relay and scheduler",
    ]
    assert controls["side_controls"]["reuse_exact_primary_entries_and_exits"] is True
    assert controls["primary_must_beat_every_control"] is True


def test_null_novelty_economic_and_portfolio_order_is_frozen() -> None:
    policy = rqhr.policy_payload()
    assert policy["economic_sequence"][:3] == [
        "mechanical nulls",
        "2023 source support and controls",
        "2023 comparator novelty",
    ]
    nulls = policy["mechanical_nulls"]
    assert nulls["must_run_before_real_rqhr_column_read"] is True
    assert nulls["scheduled_snapshot_slots_per_bar"] == 10
    assert nulls["snapshot_position"] == "bar_index+snapshot_index/10"
    assert nulls["missing_predicate"] == (
        "bar_index%1009<3 suppresses all ten slots"
    )
    assert nulls["maximum_raw_confirmations_each_scenario"] == 0
    assert nulls["maximum_accepted_events_each_scenario"] == 0

    novelty = policy["novelty"]
    assert novelty["raw_group_checks"] == [
        "artifact protocol and closed flags",
        "declared event count and frozen canonical hash",
        "valid side and positive interval",
        "unique entry and chronological global nonoverlap",
        "quarter containment and position/date/hold consistency",
    ]
    assert novelty["aware_time_reconstruction"] == (
        "2023-01-01T00:00:00Z+position*5 elapsed minutes"
    )
    assert novelty["minimum_fully_contained_rows_each_group"] == 10

    portfolio = policy["portfolio_gate"]
    assert portfolio["stage"] == "after eval pass and before portfolio promotion"
    assert portfolio[
        "maximum_absolute_signed_occupied_exposure_correlation"
    ] == "7/20"
    assert portfolio["window"] == "each exact common OOS window"
    assert portfolio["return_to_mdd_frontier_must_improve"] is True
    assert portfolio["may_rescue_failed_standalone_split"] is False


def test_source_projection_excludes_rncm_fields_and_market_outcomes() -> None:
    policy = rqhr.policy_payload()
    assert tuple(policy["source"]["exact_columns"]) == rqhr.RQHR_COLUMNS
    assert all(
        field not in rqhr.RQHR_COLUMNS
        for field in policy["source"]["forbidden_columns"]
    )
    assert policy["source"]["post_2023_rows_allowed_during_support"] is False
    assert policy["source"]["live_parity_is_separate_production_gate"] is True
    assert policy["production_gate"]["research_outcome_pass_alone_authorizes_live"] is False


def test_comparator_cohort_and_hash_producers_are_locally_frozen() -> None:
    assert [row["group"] for row in rqhr.COMPARATOR_SPECS] == [
        "ccbvfr:primary",
        "pdf10:primary",
        "crrc:primary",
    ]
    assert [row["expected_raw_rows"] for row in rqhr.COMPARATOR_SPECS] == [
        144,
        591,
        156,
    ]
    assert all(isinstance(row["path"], Path) for row in rqhr.COMPARATOR_SPECS)
    assert all(isinstance(row["producer"], Path) for row in rqhr.COMPARATOR_SPECS)
    assert all(len(row["sha256"]) == 64 for row in rqhr.COMPARATOR_SPECS)
    assert all(
        len(row["canonical_clock_sha256"]) == 64
        for row in rqhr.COMPARATOR_SPECS
    )
    assert all(row["protocol"] for row in rqhr.COMPARATOR_SPECS)
    assert all(row["closed_flags"] for row in rqhr.COMPARATOR_SPECS)


def test_preregistration_module_has_no_project_import_dependency() -> None:
    tree = ast.parse(rqhr._repository_path(rqhr.SCRIPT_PATH).read_text())
    imported_roots = {
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imported_roots <= {
        "__future__",
        "argparse",
        "dataclasses",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "tempfile",
        "typing",
    }


def test_static_preregistration_is_ineligible_and_value_blind() -> None:
    payload = rqhr.build_preregistration(verify_sources=False)
    rqhr.validate_preregistration(payload, verify_sources=False)
    assert payload["verification_mode"] == "static_test_fixture"
    assert payload["artifact_eligible"] is False
    assert payload["source_family_values_previously_opened"] is True
    assert payload["rqhr_net_path_efficiency_values_opened"] is False
    assert payload["rqhr_features_arms_confirmations_or_events_opened"] is False
    assert payload["synthetic_nulls_run"] is False
    assert payload["comparator_rows_opened_during_preregistration"] is False
    assert payload["outcomes_opened"] is False
    assert payload["source_binding"]["manifest_metadata_parsed"] is False
    assert payload["outcome_boundary"] == rqhr.STATIC_TEST_OUTCOME_BOUNDARY
    assert all(
        row["value_rows_read_during_preregistration"] == 0
        for row in payload["comparator_bindings"]
    )


def test_verified_preregistration_binds_bytes_without_parsing_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_loads(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("preregistration must not parse any JSON artifact")

    monkeypatch.setattr(json, "loads", forbidden_loads)
    payload = rqhr.build_preregistration(verify_sources=True)
    assert payload["artifact_eligible"] is True
    assert payload["verification_mode"] == "verified_hashes_without_value_parsing"
    assert payload["source_binding"]["panel_sha256"] == rqhr.SOURCE_PANEL_SHA256
    assert payload["source_binding"]["manifest_metadata_parsed"] is False
    assert len(payload["history_bindings"]) == 5
    assert len(payload["comparator_bindings"]) == 3


def test_policy_or_boundary_tampering_fails_closed() -> None:
    payload = rqhr.build_preregistration(verify_sources=False)
    payload["policy"]["thresholds"]["near_quantile"] = "19/20"
    with pytest.raises(RuntimeError, match="policy drift"):
        rqhr.validate_preregistration(payload, verify_sources=False)

    payload = rqhr.build_preregistration(verify_sources=False)
    payload["rqhr_net_path_efficiency_values_opened"] = True
    with pytest.raises(RuntimeError, match="boundary opened"):
        rqhr.validate_preregistration(payload, verify_sources=False)

    payload = rqhr.build_preregistration(verify_sources=False)
    payload["outcome_boundary"]["rqhr_columns_read"] = 1
    with pytest.raises(RuntimeError, match="outcome boundary drift"):
        rqhr.validate_preregistration(payload, verify_sources=False)

    payload = rqhr.build_preregistration(verify_sources=False)
    payload["comparator_bindings"][0]["expected_raw_rows"] = 143
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = rqhr.canonical_hash(core)
    with pytest.raises(RuntimeError, match="differs from frozen build"):
        rqhr.validate_preregistration(payload, verify_sources=False)


def test_hash_drift_fails_before_any_value_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = rqhr.sha256_file

    def drift(path: str | Path) -> str:
        if Path(path) == rqhr.SOURCE_PANEL:
            return "0" * 64
        return original(path)

    monkeypatch.setattr(rqhr, "sha256_file", drift)
    with pytest.raises(RuntimeError, match="source panel hash mismatch"):
        rqhr.build_preregistration(verify_sources=True)


def test_repository_paths_reject_absolute_and_parent_escape() -> None:
    with pytest.raises(RuntimeError, match="repository-relative"):
        rqhr._repository_path("/tmp/outside")
    with pytest.raises(RuntimeError, match="repository-relative"):
        rqhr._repository_path("../outside")


def test_atomic_write_and_preregistration_are_no_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rqhr, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        rqhr,
        "_verify_hash",
        lambda path, expected, label: expected,
    )
    monkeypatch.setattr(rqhr, "sha256_file", lambda path: "f" * 64)
    synced_directories: list[Path] = []
    monkeypatch.setattr(
        rqhr,
        "_fsync_directory",
        lambda path: synced_directories.append(path),
    )

    cfg = rqhr.Config(output="out/prereg.json")
    first, status = rqhr.write_preregistration(cfg)
    assert status == "created"
    second, status = rqhr.write_preregistration(cfg)
    assert status == "verified_existing"
    assert first == second
    assert synced_directories == [tmp_path / "out"]

    output = tmp_path / cfg.output
    changed = json.loads(output.read_text())
    changed["manifest_hash"] = "0" * 64
    output.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="canonical hash mismatch"):
        rqhr.write_preregistration(cfg)

    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text("sentinel\n")
    with pytest.raises(FileExistsError):
        rqhr._atomic_write(sentinel, {"replacement": True})
    assert sentinel.read_text() == "sentinel\n"


def test_cli_static_parser_defaults_to_frozen_output() -> None:
    args = rqhr.parse_args([])
    assert args.output == str(rqhr.DEFAULT_OUTPUT)
