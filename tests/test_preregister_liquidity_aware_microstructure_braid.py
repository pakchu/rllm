from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from training import preregister_liquidity_aware_microstructure_braid as p


def test_manifest_is_joint_incidence_and_outcome_blind() -> None:
    payload = p.build_manifest()
    p.validate_manifest(payload)
    assert payload["policy"]["policy_id"] == "LAMB-21"
    assert payload["research_history"]["global_pristine_holdout_claimed"] is False
    assert payload["research_history"]["exact_lamb_joint_incidence_seen"] is False
    assert payload["research_history"]["exact_lamb_market_outcomes_seen"] is False
    assert payload["research_history"]["qlcd_primitive_reuse_disclosed"] is True
    assert payload["research_history"]["qlcd_policy_reused"] is False
    assert all(value == 0 for value in payload["evidence_boundary"].values())


def test_source_contracts_are_minimal_hash_bound_projections() -> None:
    sources = p.source_contracts()
    assert list(sources) == ["h41", "rrp", "lattice", "cascade"]
    assert sources["h41"]["allowlist"] == list(p.H41_ALLOWLIST)
    assert sources["rrp"]["allowlist"] == list(p.RRP_ALLOWLIST)
    assert sources["lattice"]["allowlist"] == list(p.LATTICE_ALLOWLIST)
    assert sources["cascade"]["allowlist"] == list(p.CASCADE_ALLOWLIST)
    assert sources["lattice"]["excluded_columns"] == ["qlcd_score"]
    assert sources["cascade"]["excluded_columns"] == ["max_ms_score"]
    assert "qlcd_score" not in sources["lattice"]["allowlist"]
    assert "max_ms_score" not in sources["cascade"]["allowlist"]
    assert sources["lattice"]["cohort_definition"] == {
        "coarse": "quantity_mbtc % 100 == 0",
        "medium": "quantity_mbtc % 10 == 0 and not coarse",
        "fine": "all remaining exact 1 mBTC increments",
    }
    for source in sources.values():
        assert set(source["allowlist"]).issubset(source["physical_header"])
        assert len(source["allowlist"]) == len(set(source["allowlist"]))
        assert source["loader"] == "pandas.read_csv(usecols=exact_allowlist)"
        assert source["load_all_then_drop_forbidden"] is False
        assert source["projection_order_is_frozen"] is True


def test_clock_rank_and_safety_contract_are_frozen() -> None:
    payload = p.build_manifest()
    clock = payload["clock"]
    rank = payload["rank_contract"]
    assert clock["boundaries_utc"] == ["00:00:00", "08:00:00", "16:00:00"]
    assert clock["micro_window"] == "[B-8h,B)"
    assert clock["micro_rows_each_source"] == 96
    assert clock["decision"] == "B+5m"
    assert clock["execution"] == "B+10m at USD-M five-minute open"
    assert clock["invalid_or_unready_target"] == "TARGET_FLAT"
    assert clock["wall_clock_time_compressed"] is False
    assert rank["strictly_prior"] is True
    assert rank["maximum_prior_valid_boundaries"] == 270
    assert rank["minimum_prior_valid_boundaries"] == 180
    assert rank["macro_ranked"] is False
    assert rank["invalid_boundary_enters_reference"] is False
    assert payload["tokens"]["sequence_lines"] == 21
    assert payload["tokens"]["sequence_calendar_span_days"] == 7


def test_token_schema_action_space_and_controls_are_exact() -> None:
    payload = p.build_manifest()
    tokens = payload["tokens"]
    controls = payload["controls"]
    assert tokens["columns"] == list(p.TOKEN_COLUMNS)
    assert len(tokens["columns"]) == 11
    assert tokens["safety_tokens"] == list(p.SAFETY_TOKENS)
    assert tokens["action_space"] == [
        "TARGET_SHORT",
        "TARGET_FLAT",
        "TARGET_LONG",
    ]
    assert tokens["invalid_output_action"] == "TARGET_FLAT"
    assert tokens["raw_numeric_prompt_fields"] == 0
    assert controls["ordered"] == list(p.CONTROL_IDS)
    assert "37 prior five-minute positions" in controls["cascade_delay"]
    assert "first 37 positions control-invalid" in controls["cascade_delay"]
    assert "recompute macro_transition=MIXED" in controls["macro_mask"]
    assert controls["independent_rebuild"] is True
    assert controls["may_replace_primary"] is False


def test_support_gates_and_failure_action_are_conjunctive() -> None:
    gates = p.build_manifest()["source_support_gates"]
    assert gates["all_conjunctive"] is True
    assert gates["source_join_min_each_year"] == 0.99
    assert gates["core_valid_min_each_year"] == 0.95
    assert gates["sequence_ready_min"] == {
        "2020": 750,
        "2021": 1_000,
        "2022": 1_000,
        "2023": 1_000,
    }
    assert gates["quarter_ready_min_after_warmup"] == 225
    assert gates["forced_flat_max_each_full_post_warmup_quarter"] == 0.08
    assert gates["minimum_supported_categories_per_field_each_year"] == 2
    assert gates["macro_support_and_restrict_min_each_year"] == 0.05
    assert gates["micro_buy_and_sell_min_each_year"] == 0.10
    assert gates["cascade_follow_and_absorb_min_each_year"] == 0.075
    assert gates["append_replay_byte_identical"] is True
    assert gates["forbidden_counter_max"] == 0
    assert "safety and current_position excluded" in gates["diversity_denominator"]
    assert gates["failure_action"] == "retire LAMB-21 unchanged before rewards"


def test_economic_stage_is_contingent_and_honest_about_holdouts() -> None:
    payload = p.build_manifest()
    authority = payload["stage_authority"]
    economic = payload["contingent_economic_sequence"]
    assert "token_support" in authority["authorized"]
    assert "future_return" in authority["forbidden"]
    assert "model_training" in authority["forbidden"]
    assert economic["requires_support_pass"] is True
    assert economic["historical_not_realtime_prospective"] is True
    assert economic["live_claim_requires_forward_shadow_or_live_interval"] is True
    assert economic["full_three_calendar_year_claim_before_2026_12_31"] is False
    assert economic["minimum_base_cagr_to_strict_mdd"] == 3.0
    assert economic["maximum_strict_mdd"] == 0.15
    assert economic["lattice_only_and_no_lattice_killer_baselines"] is True


def test_frozen_hashes_headers_gzip_and_commit_match_without_rows() -> None:
    p.assert_boundary_committed()
    p.validate_frozen_dependencies()
    assert p.csv_header(p.H41_SOURCE) == p.H41_PHYSICAL_HEADER
    assert p.csv_header(p.RRP_SOURCE) == p.RRP_PHYSICAL_HEADER
    assert p.csv_header(p.LATTICE_SOURCE) == p.LATTICE_PHYSICAL_HEADER
    assert p.csv_header(p.CASCADE_SOURCE) == p.CASCADE_PHYSICAL_HEADER
    assert p.sha256_csv_header(p.H41_SOURCE) == p.H41_HEADER_SHA256
    assert p.sha256_csv_header(p.RRP_SOURCE) == p.RRP_HEADER_SHA256
    assert p.sha256_csv_header(p.LATTICE_SOURCE) == p.LATTICE_HEADER_SHA256
    assert p.sha256_csv_header(p.CASCADE_SOURCE) == p.CASCADE_HEADER_SHA256
    assert all(
        p.gzip_mtime(path) == 0
        for path in (
            p.H41_SOURCE,
            p.RRP_SOURCE,
            p.LATTICE_SOURCE,
            p.CASCADE_SOURCE,
        )
    )


def test_header_reader_does_not_decode_later_rows() -> None:
    probe = p.REPOSITORY_ROOT / "results" / ".lamb_header_probe.csv.gz"
    try:
        with gzip.GzipFile(filename=probe, mode="wb", mtime=0) as handle:
            handle.write(b"a,b\n\xff\xfe\x00not-a-row")
        relative = probe.relative_to(p.REPOSITORY_ROOT)
        assert p.csv_header(relative) == ("a", "b")
    finally:
        probe.unlink(missing_ok=True)


def test_validate_manifest_rejects_mutation() -> None:
    payload = json.loads(json.dumps(p.build_manifest()))
    payload["policy"]["sequence_lines"] = 20
    with pytest.raises(ValueError, match="differs from frozen code"):
        p.validate_manifest(payload)


def test_json_round_trip_manifest_validates() -> None:
    persisted_shape = json.loads(json.dumps(p.build_manifest()))
    p.validate_manifest(persisted_shape)


def test_persisted_preregistration_validates_against_committed_producer() -> None:
    artifact = p.REPOSITORY_ROOT / p.DEFAULT_OUTPUT
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    p.validate_manifest(payload)
    assert p.sha256_file(p.DEFAULT_OUTPUT) == (
        "4ac8bf8f2d54120130c49a90f3d40a5cfaf141673525cb54df4b5333c01290e6"
    )
    assert payload["producer"] == {
        "head_at_generation": "32f97c8d74e2598c9858da32b7eb203b690da0b4",
        "script": p.PRODUCER_SCRIPT,
        "script_clean_at_generation": True,
        "script_commit": "32f97c8d74e2598c9858da32b7eb203b690da0b4",
        "script_sha256": (
            "1fb3b7f39fe418e9c160a3035cbb63a8f65cb72119a49d36c93fc5528c37e10c"
        ),
        "script_tracked": True,
        "uncommitted_producer": False,
    }


def test_sealed_producer_remains_verifiable_after_later_commits() -> None:
    assert p.producer_binding()["script_commit"] == p.SEALED_PRODUCER_COMMIT
    p.assert_producer_committed(creating=False)
    with pytest.raises(RuntimeError, match="sealed producer HEAD"):
        p.assert_producer_committed(creating=True)


def test_write_once_is_reproducible_and_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = p.build_manifest()
    creation_checks: list[bool] = []
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        p,
        "assert_producer_committed",
        lambda *, creating: creation_checks.append(creating),
    )
    assert p.write_once("results/freeze.json", payload) == "created"
    assert creation_checks == [True]
    target = tmp_path / "results" / "freeze.json"
    assert target.exists()
    assert p.write_once("results/freeze.json", payload) == "verified_existing"
    assert creation_checks == [True]
    target.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact drift"):
        p.write_once("results/freeze.json", payload)


def test_write_once_rejects_unsafe_paths_and_symlink_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = p.build_manifest()
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        p,
        "assert_producer_committed",
        lambda *, creating: None,
    )
    with pytest.raises(RuntimeError, match="repository-relative"):
        p.write_once("../escape.json", payload)
    with pytest.raises(RuntimeError, match="repository-relative"):
        p.write_once(tmp_path / "absolute.json", payload)

    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        p.write_once("linked/freeze.json", payload)
