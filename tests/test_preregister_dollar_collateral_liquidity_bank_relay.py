from __future__ import annotations

import gzip
import json

import pytest

from training import preregister_dollar_collateral_liquidity_bank_relay as p


def test_manifest_is_joint_incidence_comparator_and_outcome_blind() -> None:
    payload = p.build_manifest()
    p.validate_manifest(payload)
    assert payload["policy"]["policy_id"] == "DCLB-864"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["source_rows_decoded"] is False
    assert payload["comparator_rows_decoded"] is False
    assert all(value == 0 for value in payload["evidence_boundary"].values())
    history = payload["research_history_boundary"]
    assert history["exact_dclb_joint_state_or_incidence_seen"] is False
    assert history["exact_dclb_market_outcomes_seen"] is False
    assert history["global_pristine_holdout_claimed"] is False


def test_sources_use_exact_allowlists_and_frozen_quarantine_rules() -> None:
    sources = p.build_manifest()["source_contracts"]
    assert sources["h41"]["allowlist"] == list(p.H41_ALLOWLIST)
    assert sources["rrp"]["allowlist"] == list(p.RRP_ALLOWLIST)
    assert sources["h8"]["allowlist"] == list(p.H8_ALLOWLIST)
    assert all(
        source["loader"] == "pandas.read_csv(usecols=exact_allowlist)"
        for source in sources.values()
    )
    assert sources["h41"]["numeric_rule"] == "finite and strictly positive"
    assert "finite and nonnegative" in sources["rrp"]["numeric_rule"]
    assert "blank iff source_complete=false" in sources["rrp"]["numeric_rule"]
    assert "NSA" in sources["h8"]["numeric_rule"]
    assert "nsa_small_cash_assets_latest" in sources["h8"]["allowlist"]


def test_causal_algebra_rank_warmups_and_dst_clock_are_frozen() -> None:
    payload = p.build_manifest()
    algebra = payload["source_algebra"]
    execution = payload["execution_contract"]
    assert algebra["h41"]["strict_prior_midrank_count"] == 104
    assert algebra["h41"]["first_rankable_delta"] == 105
    assert algebra["rrp"]["strict_prior_midrank_count"] == 13
    assert algebra["rrp"]["first_rankable_post_reset_delta"] == 14
    assert "no bridge" in algebra["rrp"]["quarantine"]
    assert algebra["h8"]["robust_z_prior_observations"] == 104
    assert algebra["h8"]["sa_primary_nsa_control_only"] is True
    assert algebra["macro"]["integer"] == (
        "13*h41_center_num-104*rrp_center_num"
    )
    assert algebra["macro"]["side_sign"] == "sign(macro_integer)"
    assert execution["exit"] == "entry_utc + 4,320 elapsed minutes"
    assert execution["exposure"] == "[entry_utc, exit_utc)"
    assert execution["dst_wall_clock_normalization"] is False
    assert execution["global_nonoverlap_before_split"] is True


def test_controls_support_and_composition_are_exact_and_fail_closed() -> None:
    payload = p.build_manifest()
    controls = payload["source_only_controls"]
    support = payload["source_support_gate"]
    assert controls["ordered"] == list(p.CONTROL_ORDER)
    assert len(controls["ordered"]) == 14
    assert controls["all_required"] is True
    assert "same post-quarantine segment" in controls["stale_rrp"]
    assert "YYYY-MM-DDTHH:MM:SSZ" in controls["random_side"]
    assert "exact NSA H8" in controls["nsa_h8"]
    assert support["train"]["events_min"] == 75
    assert support["train"]["each_year_events_min"] == 12
    assert support["train"]["maximum_same_side_run"] == 12
    assert support["selection"]["events_min"] == 20
    assert support["selection"]["each_quarter_events_min"] == 2
    assert support["selection"]["maximum_same_side_run"] == 10
    composition = support["composition_each_split"]
    assert composition["h41_only_same_side_reproduction_max"] == 0.85
    assert composition["rrp_only_same_side_reproduction_max"] == 0.85
    assert composition["random_same_side_reproduction_max"] == 0.60
    assert support["every_required_control_nonempty_each_split"] is True
    assert "before comparator rows or outcomes" in support["failure_action"]


def test_common_window_parsers_groups_and_minimum_counts_are_frozen() -> None:
    novelty = p.build_manifest()["novelty_contract"]
    assert novelty["common_window_policy_sha256"] == (
        p.COMMON_WINDOW_POLICY_SHA256
    )
    assert novelty["window"] == [
        "2020-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
    ]
    assert novelty["full_interval_containment"] is True
    assert novelty["raw_validation_before_filter"] is True
    assert novelty["complete_five_minute_grid"] is True
    comparators = {item["id"]: item for item in novelty["comparators"]}
    assert list(comparators) == ["FLCC", "ORFR", "ORPB", "H8DM", "BDRC"]
    assert comparators["FLCC"]["usecols"] == [
        "candidate_id",
        "clock_name",
        "entry_time",
        "exit_time",
        "side",
    ]
    parser = comparators["FLCC"]["parser"]
    assert parser["group_columns"] == ["candidate_id", "clock_name"]
    assert "exact UTF-8 string equality" in parser["group_filter"]
    assert "RFC3339 timezone suffix" in parser["timestamp_parser"]
    assert parser["side_mapping"] == {"LONG": 1, "SHORT": -1}
    assert parser["unknown_or_blank_side"] == "fail"
    assert {
        group["minimum_contained_rows"]
        for group in comparators["FLCC"]["groups"]
    } == {90}
    assert {
        group["filter"]["control"] for group in comparators["ORFR"]["groups"]
    } == {"primary", "one_day_delta_tail", "one_release_delay"}
    assert comparators["ORPB"]["groups"][0]["minimum_contained_rows"] == 180
    assert comparators["H8DM"]["clock_family"] == "same_h8_anchor"
    assert [
        group["minimum_contained_rows"]
        for group in comparators["BDRC"]["groups"]
    ] == [50, 120]
    assert novelty["same_h8_anchor_thresholds"][
        "absolute_signed_occupancy_pearson_max"
    ] == 0.65
    assert novelty["asynchronous_thresholds"][
        "six_hour_one_to_one_jaccard_max"
    ] == 0.35
    disclosure = novelty["prospective_policy_motivation"]
    assert disclosure["prior_cross_boundary_comparator_timing_row_seen"] is True
    assert "entered in late 2023 and exited in early 2024" in (
        disclosure["disclosed_fact"]
    )
    assert (
        disclosure["dclb_source_incidence_or_overlap_opened_when_disclosed"]
        is False
    )
    assert disclosure["dclb_market_outcomes_opened_when_disclosed"] is False


def test_live_economic_and_rllm_boundaries_are_frozen() -> None:
    payload = p.build_manifest()
    live = payload["live_fail_flat_contract"]
    economic = payload["economic_rllm_sequence"]
    rllm = payload["rllm_boundary"]
    assert live["expected_publication_calendars_predeclared"] is True
    assert live["stale_carry_alternate_endpoint_fill_or_imputation"] is False
    assert "no source update and no event" in (
        live["missing_late_schema_integrity_availability_or_quarantine_mismatch"]
    )
    qualification = economic["sealed_eval_qualification"]
    assert qualification["cagr_to_strict_mdd_min"] == 3.0
    assert qualification["strict_mdd_max"] == 0.15
    assert qualification["executed_trades_min"] == 12
    assert qualification["mean_gross_underlying_bp_strictly_above"] == 20.0
    assert rllm["action_space"] == ["TRADE_FIXED_SIDE", "ABSTAIN"]
    assert "raw_levels_deltas_zscores_ranks_or_rank_numerators" in (
        rllm["forbidden"]
    )
    assert "current_position_state" in rllm["allowed_tokens"]
    assert rllm["no_2022_checkpoint_choice_from_2023"] is True


def test_frozen_hashes_and_headers_match_without_decoding_rows() -> None:
    p.validate_frozen_dependencies()
    assert p.sha256_csv_header(p.H41_SOURCE) == p.H41_HEADER_SHA256
    assert p.sha256_csv_header(p.RRP_SOURCE) == p.RRP_HEADER_SHA256
    assert p.sha256_csv_header(p.H8_SOURCE) == p.H8_HEADER_SHA256
    assert set(p.H41_ALLOWLIST).issubset(p.csv_header(p.H41_SOURCE))
    assert set(p.RRP_ALLOWLIST).issubset(p.csv_header(p.RRP_SOURCE))
    assert set(p.H8_ALLOWLIST).issubset(p.csv_header(p.H8_SOURCE))
    assert len(p.frozen_dependencies()) == 17


def test_header_reader_does_not_decode_later_rows(tmp_path) -> None:
    plain = tmp_path / "clock.csv"
    plain.write_bytes(b"a,b\n\xff\xfe\x00not-csv")
    with pytest.raises(RuntimeError, match="repository-relative"):
        p.csv_header(plain)

    compressed = p.REPOSITORY_ROOT / "results" / ".dclb_header_probe.csv.gz"
    try:
        with gzip.open(compressed, "wb") as handle:
            handle.write(b"x,y\n\xff\xfe\x00not-csv")
        relative = compressed.relative_to(p.REPOSITORY_ROOT)
        assert p.csv_header(relative) == ["x", "y"]
    finally:
        compressed.unlink(missing_ok=True)


def test_write_once_is_reproducible_and_rejects_drift(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        p,
        "validate_frozen_dependencies",
        lambda: calls.append(True),
    )
    payload = p.build_manifest()
    output = "freeze.json"
    assert p.write_once(output, payload) == "created"
    assert calls == [True]
    stored_path = tmp_path / output
    assert stored_path.read_text(encoding="utf-8") == p._canonical_manifest_text()
    assert p.write_once(output, p.build_manifest()) == "verified_existing"
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    stored["policy"]["hold_bars"] = 863
    with pytest.raises(RuntimeError, match="core differs from code"):
        p.validate_manifest(stored)


def test_write_once_rejects_existing_drift_traversal_and_symlinks(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(p, "validate_frozen_dependencies", lambda: None)
    payload = p.build_manifest()

    assert p.write_once("drift.json", payload) == "created"
    (tmp_path / "drift.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="existing manifest hash mismatch"):
        p.write_once("drift.json", payload)

    with pytest.raises(RuntimeError, match="repository-relative"):
        p.write_once("../escape.json", payload)
    with pytest.raises(RuntimeError, match="repository-relative"):
        p.write_once(tmp_path / "absolute.json", payload)

    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "linked_parent").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="parent is missing.*symlinked"):
        p.write_once("linked_parent/freeze.json", payload)

    (tmp_path / "real.json").write_text("not the manifest", encoding="utf-8")
    (tmp_path / "leaf.json").symlink_to(tmp_path / "real.json")
    with pytest.raises(RuntimeError, match="output is symlinked"):
        p.write_once("leaf.json", payload)


def test_write_once_same_content_race_is_verified(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(p, "validate_frozen_dependencies", lambda: None)
    expected = p._canonical_manifest_text().encode("utf-8")

    def race_winner(*args, **kwargs) -> None:
        del args, kwargs
        (tmp_path / "race.json").write_bytes(expected)
        raise FileExistsError

    monkeypatch.setattr(p.os, "link", race_winner)
    assert p.write_once("race.json", p.build_manifest()) == "verified_existing"
    assert not list(tmp_path.glob(".race.json.*.tmp"))


def test_write_once_drift_race_fails_closed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(p, "validate_frozen_dependencies", lambda: None)

    def race_winner(*args, **kwargs) -> None:
        del args, kwargs
        (tmp_path / "race.json").write_bytes(b"drift\n")
        raise FileExistsError

    monkeypatch.setattr(p.os, "link", race_winner)
    with pytest.raises(RuntimeError, match="manifest race drift"):
        p.write_once("race.json", p.build_manifest())
    assert not list(tmp_path.glob(".race.json.*.tmp"))


def test_write_once_fsyncs_file_and_directory(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(p, "validate_frozen_dependencies", lambda: None)
    real_fsync = p.os.fsync
    fsync_modes: list[int] = []

    def tracked_fsync(descriptor: int) -> None:
        fsync_modes.append(p.os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr(p.os, "fsync", tracked_fsync)
    assert p.write_once("durable.json", p.build_manifest()) == "created"
    assert any(p.stat.S_ISREG(mode) for mode in fsync_modes)
    assert any(p.stat.S_ISDIR(mode) for mode in fsync_modes)


def test_validate_manifest_rejects_self_rehashed_drift() -> None:
    payload = p.build_manifest()
    payload["novelty_contract"]["asynchronous_thresholds"][
        "exact_entry_jaccard_max"
    ] = 0.21
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = p.canonical_hash(core)
    with pytest.raises(RuntimeError, match="core differs from code"):
        p.validate_manifest(payload)


def test_dependency_paths_reject_traversal() -> None:
    with pytest.raises(RuntimeError, match="repository-relative"):
        p.csv_header("../outside.csv.gz")
