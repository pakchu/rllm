from __future__ import annotations

import gzip
import json

import pytest

from training import preregister_cross_venue_intrinsic_clock_resolution as p


def test_manifest_is_incidence_comparator_and_outcome_blind() -> None:
    payload = p.build_manifest()
    p.validate_manifest(payload)
    assert payload["policy"]["policy_id"] == "CVICR-72"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["source_rows_decoded"] is False
    assert payload["comparator_rows_decoded"] is False
    assert payload["research_history_boundary"][
        "exact_cvicr_anchor_or_candidate_incidence_seen"
    ] is False


def test_policy_freezes_paired_clocks_transition_and_latency() -> None:
    payload = p.build_manifest()
    policy = payload["policy"]
    clock = payload["causal_clock_contract"]
    execution = payload["execution_contract"]
    assert policy["reference_calendar_days"] == 28
    assert policy["reference_complete_days_min"] == 21
    assert policy["intrinsic_volume_fraction"] == 0.50
    assert policy["latest_anchor_start_minute_utc"] == 17 * 60 + 50
    assert policy["gap_reference_pairs"] == 180
    assert policy["gap_reference_pairs_min"] == 90
    assert policy["gap_quantile"] == 0.60
    assert policy["entry_delay_bars_from_late_anchor"] == 2
    assert policy["hold_bars"] == 72
    assert "laggard early sign=-d" in clock["initial_conflict"]
    assert "both equal d" in clock["resolution"]
    assert clock["price_basis_or_return_input"] is None
    assert clock["future_bar_used_by_signal"] is False
    assert execution["signal_available_time"] == "A_late+5m"
    assert execution["decision_order_time"] == "A_late+10m"
    assert execution["entry"].endswith("A_late+10m")
    assert execution["comparator_timestamp"] == "entry_time"


def test_source_contract_is_prefix_causal_and_column_allowlisted() -> None:
    source = p.build_manifest()["source_contract"]
    assert source["allowlist"] == list(p.SOURCE_ALLOWLIST)
    assert source["loader"].startswith("pandas.read_csv(usecols=allowlist)")
    assert "never inspect later-day completeness" in source["current_day"]
    assert "exact 28-calendar-day window" in source["missing_policy"]
    forbidden = {
        "spot_log_return_5m",
        "um_log_return_5m",
        "basis_change_bp",
        "flow_transfer_asymmetry",
    }
    assert forbidden.isdisjoint(source["allowlist"])


def test_controls_support_and_selectivity_are_fail_closed() -> None:
    payload = p.build_manifest()
    controls = payload["source_only_controls"]
    support = payload["source_support_gate"]
    assert controls["ordered"] == list(p.CONTROL_ORDER)
    assert controls["score_bearing"] == list(p.SCORE_BEARING_CONTROLS)
    assert controls["all_emit_side_in"] == [-1, 1]
    assert support["train_events_min"] == 75
    assert support["selection_events_min"] == 24
    assert support["selection_each_half_events_min"] == 10
    selectivity = support["mechanism_selectivity"]
    assert selectivity["primary_over_gap_only_max"] == 0.40
    assert selectivity["fixed_expected_time_entry_jaccard_max"] == 0.10
    assert selectivity["stale_laggard_flow_entry_jaccard_max"] == 0.05
    assert selectivity["undefined_or_empty_required_control"] == "one and fail"


def test_comparator_contracts_freeze_coverage_and_near_time_gates() -> None:
    novelty = p.build_manifest()["novelty_contract"]
    comparators = {item["id"]: item for item in novelty["comparators"]}
    assert list(comparators) == [
        "CATCH-12",
        "CLASP-24",
        "LURI-48",
        "CVTT-V01-V04",
        "IVLIR-primary",
        "IVFHR-primary-and-any-handoff",
        "IVPLH-primary",
    ]
    assert comparators["CVTT-V01-V04"]["declared_coverage"][1] == (
        "2023-01-01T00:00:00Z"
    )
    assert comparators["CVTT-V01-V04"]["selection_absence_is_not_failure"] is True
    assert comparators["IVFHR-primary-and-any-handoff"][
        "compare_groups_separately"
    ] is True
    assert novelty["exact_entry_jaccard_max"] == 0.10
    assert novelty["one_bar_tolerant_jaccard_max"] == 0.20
    assert novelty["twelve_bar_tolerant_jaccard_max"] == 0.35
    assert novelty["six_hour_tolerant_jaccard_intrinsic_family_max"] == 0.60
    assert novelty["absolute_signed_occupancy_pearson_max"] == 0.40


def test_economic_and_llm_non_rescue_contracts_are_frozen() -> None:
    payload = p.build_manifest()
    economics = payload["economic_contract"]
    gates = payload["economic_gates"]
    live = payload["live_parity"]
    sequence = payload["strict_sequence"]
    llm = payload["llm_boundary"]
    assert economics["base_account_cost_per_side"] == 0.0003
    assert economics["stress_replaces_base"] is True
    assert economics["stress_account_cost_per_side"] == 0.0005
    assert economics["cluster_signflip"]["draws"] == 100_000
    assert economics["cluster_signflip"]["seed"] == 20_260_724
    assert gates["train_and_selection_cagr_to_strict_mdd_min"] == 3.0
    assert gates["mean_gross_underlying_bp_min"] == 30.0
    assert gates["primary_ratio_margin_over_controls_min"] == 0.50
    assert gates["score_bearing_controls"] == list(p.SCORE_BEARING_CONTROLS)
    assert sequence["stop_at_first_failure"] is True
    assert sequence["no_parameter_repair"] is True
    assert live["five_minute_finalization_before_accumulation"] is True
    assert live["current_prefix_defect_action"] == "cancel current day and remain flat"
    assert live["computation_buffer_bars"] == 1
    assert "source divergence" in live["fail_flat_on"]
    assert "may not backdate" in live["rest_repair"]
    assert llm["action_space"] == ["TRADE_FIXED_SIDE", "ABSTAIN"]
    assert "side_choice" in llm["forbidden"]
    assert "timestamp" in llm["forbidden"]


def test_frozen_dependency_hashes_and_headers_match_without_decoding_rows() -> None:
    dependencies = p.frozen_dependencies()
    assert dependencies[p.BOUNDARY_DOCUMENT] == p.BOUNDARY_DOCUMENT_SHA256
    assert dependencies[p.MECHANISM_DOCUMENT] == p.MECHANISM_DOCUMENT_SHA256
    assert dependencies[p.SOURCE] == p.SOURCE_SHA256
    assert len(dependencies) == 12
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
