from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path

import pytest

from training import preregister_soma_maturity_allocation_fracture as p


def test_manifest_is_deterministic_self_hashing_and_closed() -> None:
    first = p.build_manifest()
    second = p.build_manifest()
    assert first == second
    core = {
        key: value for key, value in first.items() if key != "manifest_hash"
    }
    assert first["manifest_hash"] == p.canonical_hash(core)
    p.validate_manifest(first)
    assert first["source_incidence_opened"] is False
    assert first["candidate_comparator_overlap_opened"] is False
    assert first["economic_rows_opened"] is False
    assert first["outcomes_opened"] is False


def test_write_once_artifact_matches_code_exactly() -> None:
    artifact = p.REPOSITORY_ROOT / p.DEFAULT_OUTPUT
    assert artifact.is_file()
    assert not artifact.is_symlink()
    assert artifact.read_bytes() == p.canonical_manifest_bytes(
        p.build_manifest()
    )


def test_frozen_dependencies_and_headers_validate_without_economic_rows() -> None:
    p.validate_frozen_dependencies()
    assert p.sha256_csv_header(p.OPERATIONS) == p.OPERATIONS_HEADER_SHA256
    assert p.sha256_csv_header(p.DETAILS) == p.DETAILS_HEADER_SHA256
    assert p.sha256_csv_header(p.SLCS_CLOCK) == p.SLCS_HEADER_SHA256
    assert p.sha256_csv_header(p.SCAF_CLOCK) == p.SCAF_HEADER_SHA256
    assert set(p.OPERATIONS_USECOLS).issubset(p.csv_header(p.OPERATIONS))
    assert set(p.DETAILS_USECOLS).issubset(p.csv_header(p.DETAILS))
    dependencies = p.active_frozen_dependencies()
    assert str(p.MARKET_DATA) not in dependencies
    assert str(p.FUNDING_DATA) not in dependencies


def test_description_probe_and_exact_parser_contract() -> None:
    payload = p.build_manifest()
    rows = payload["probe_disclosure"]["rows"]
    assert len(rows) == 8
    for _, description in rows:
        parsed = p.parse_security_description(description)
        assert parsed.label == "T"
        assert isinstance(parsed.coupon, Fraction)
    assert p.maturity_distance(
        "2019-01-02",
        "T 08.875 02/15/19",
    ) == 44
    assert p.parse_exact_decimal("0") == 0
    assert p.parse_exact_decimal("12.3400") == Fraction(617, 50)


@pytest.mark.parametrize(
    "description",
    [
        "T 08.875 02/15/19 ",
        " T 08.875 02/15/19",
        "T  08.875 02/15/19",
        "t 08.875 02/15/19",
        "T 100.000 02/15/19",
        "T +8.875 02/15/19",
        "T 08.875 2/15/19",
        "T 08.875 02/30/19",
        "T 08.875 02/15/2019",
    ],
)
def test_description_parser_fails_closed(description: str) -> None:
    with pytest.raises(ValueError, match="SMAF-72"):
        p.parse_security_description(description)


@pytest.mark.parametrize("value", ["", "01", "-1", "+1", "1e3", "1,000"])
def test_exact_decimal_parser_fails_closed(value: str) -> None:
    with pytest.raises(ValueError, match="SMAF-72"):
        p.parse_exact_decimal(value)


def test_maturity_distance_rejects_nonpositive_or_bad_operation_date() -> None:
    with pytest.raises(ValueError, match="outside frozen range"):
        p.maturity_distance("2019-02-15", "T 08.875 02/15/19")
    with pytest.raises(ValueError, match="operation_date"):
        p.maturity_distance("2019-02-30", "T 08.875 03/15/19")


def test_feature_rank_onset_and_execution_are_exact() -> None:
    payload = p.build_manifest()
    feature = payload["feature_contract"]
    assert feature["primary"] == "2*C(S)-C(V)-C(A)"
    assert feature["decomposition"] == "(C(S)-C(V))+(C(S)-C(A))"
    assert "average" in feature["meaning"]
    assert feature["high_side"] == "SHORT"
    assert feature["low_side"] == "LONG"
    rank = payload["rank_and_onset"]
    assert rank["history"] == "latest 126 strictly prior complete operations"
    assert rank["midrank"] == "(2*L+E)/252"
    assert rank["LOW"] == "10*(2*L+E) <= 252"
    assert rank["HIGH"] == "10*(2*L+E) >= 2268"
    assert rank["first_rank_ready_operation"].startswith("baseline only")
    execution = payload["execution"]
    assert execution["already_aligned_still_waits_minutes"] == 5
    assert execution["hold_bars"] == 864
    assert execution["hold_hours"] == 72
    assert execution["global_nonoverlap_before_split"] is True
    controls = payload["controls"]
    assert "same uninterrupted segment" in controls["one_operation_delay"]
    assert controls["delayed_parent_set"] == (
        "raw parent signal IDs and sides unchanged"
    )
    assert "may shrink" in controls["delayed_accepted_set"]


def test_complete_operation_and_causal_batch_fail_closed() -> None:
    payload = p.build_manifest()
    complete = payload["complete_operation_contract"]
    assert complete["operation_cusip_unique"] is True
    assert complete["detail_operation_totals_reconcile_exactly"] == [
        "par_submitted",
        "par_accepted",
    ]
    assert complete["one_invalid_detail_invalidates_operation"] is True
    batch = payload["causal_batch_contract"]
    assert batch["key"] == "exact available_at_utc"
    assert batch["exactly_one_complete_operation_required"] is True
    assert batch["current_batch_mutually_prior"] is False
    assert batch["weekend_holiday_or_no_operation_gap_resets"] is False


def test_support_gates_and_failure_order_are_numeric_and_complete() -> None:
    gates = p.build_manifest()["source_support_gates"]
    assert gates["coverage"] == {
        "each_ratio_exact_in_full_warmup_train_selection": 1.0,
        "train_rank_ready_min": 740,
        "selection_rank_ready_min": 240,
        "each_split_each_raw_tail_share_min": 0.05,
        "each_split_each_raw_tail_share_max": 0.20,
    }
    assert gates["operation_and_batch_window_attribution"] == "available_at_utc"
    assert gates["split_boundary_resets_rank_history"] is False
    assert set(gates["coverage_windows"]) == {
        "full",
        "warmup",
        "train",
        "selection",
    }
    assert gates["coverage_formulas"] == {
        "description_parser_coverage": (
            "valid parsed joined detail rows / all joined detail rows"
        ),
        "complete_operation_share": (
            "complete operations / all operation rows"
        ),
        "single_operation_batch_share": (
            "valid one-complete-operation availability batches / "
            "all distinct available_at_utc batches"
        ),
    }
    assert gates["train"]["events_min"] == 60
    assert gates["train"]["events_max"] == 180
    assert gates["selection"]["events_min"] == 18
    assert gates["selection"]["events_max"] == 70
    internal = gates["internal_component_distinctness"]
    assert internal["controls"] == list(p.SOURCE_CONTROL_ORDER[1:])
    assert internal["exact_entry_jaccard_max"] == 0.70
    assert internal["absolute_signed_occupancy_pearson_max"] == 0.80
    assert gates["failure_order"][-1] == "internal_component_distinctness"


def test_novelty_contract_binds_both_soma_families_and_exact_metrics() -> None:
    novelty = p.build_manifest()["novelty_contract"]
    assert novelty["opens_only_after_all_source_support_passes"] is True
    assert novelty["common_window"] == [
        "2020-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
    ]
    comparators = {item["id"]: item for item in novelty["comparators"]}
    assert set(comparators) == {"SLCS", "SCAF"}
    assert comparators["SLCS"]["allowed_groups"] == list(
        p.SLCS_GROUP_VOCABULARY
    )
    assert comparators["SLCS"]["selected_groups"] == list(p.SLCS_GROUPS)
    assert comparators["SLCS"]["side_map"] == {"1": "LONG", "-1": "SHORT"}
    assert comparators["SCAF"]["allowed_groups"] == list(
        p.SCAF_GROUP_VOCABULARY
    )
    assert comparators["SCAF"]["selected_groups"] == list(p.SCAF_GROUPS)
    assert comparators["SCAF"]["side_map"] == {
        "LONG": "LONG",
        "SHORT": "SHORT",
    }
    assert novelty["thresholds_each_group"] == {
        "exact_entry_jaccard_max": 0.20,
        "same_entry_same_side_reproduction_max": 0.30,
        "candidate_24h_containment_max": 0.40,
        "comparator_24h_containment_max": 0.40,
        "absolute_signed_occupancy_pearson_max": 0.35,
    }
    assert novelty["all_selected_groups_must_pass"] is True


def test_economic_artifacts_and_sequential_gates_are_frozen_but_sealed() -> None:
    payload = p.build_manifest()
    artifacts = payload["sealed_economic_artifacts"]
    assert artifacts["market_data"]["sha256"] == p.MARKET_DATA_SHA256
    assert artifacts["market_data"]["header_sha256"] == p.MARKET_HEADER_SHA256
    assert artifacts["funding_data"]["sha256"] == p.FUNDING_DATA_SHA256
    assert (
        artifacts["funding_data"]["header_sha256"]
        == p.FUNDING_HEADER_SHA256
    )
    economics = payload["economic_contract"]
    assert economics["rows_open_authorized"] is False
    hydration = payload["economic_hydration_contract"]
    assert hydration["materialization_state_at_freeze"] == {
        "market_manifest": False,
        "market_data": False,
        "funding_manifest": True,
        "funding_data": True,
    }
    assert hydration["evaluator_fallback_artifact_root"] is None
    assert hydration["evaluator_refuses_absent_logical_path"] is True
    hydration_manifest = hydration["manifest"]
    assert hydration_manifest["path"] == str(p.HYDRATION_MANIFEST)
    assert hydration_manifest["protocol_version"] == (
        "smaf_72_economic_artifact_hydration_v1"
    )
    assert hydration_manifest["artifact_order"] == "ascending logical_path"
    assert hydration_manifest["portable_source_locator"] == (
        "local-cache:sha256:<artifact sha256>"
    )
    assert hydration_manifest["copied_at_utc"] == (
        "UTC RFC3339 YYYY-MM-DDTHH:MM:SS.ffffffZ"
    )
    assert hydration["host_absolute_source_path"] == {
        "allowed_only_in_uncommitted_gitignored_log": str(
            p.HYDRATION_LOCAL_LOG
        ),
        "forbidden_from_committed_manifest_evaluator_and_results": True,
    }
    assert hydration["later_evaluator_freezes"] == [
        "hydration manifest file SHA256",
        "hydration internal manifest_hash",
    ]
    assert economics["quantity"] == (
        "0.5*pre_entry_equity/entry_open; fixed through exit"
    )
    assert economics["funding_interval"] == (
        "entry_time <= funding_time < exit_time"
    )
    gates = payload["economic_gates"]
    assert gates["train_2020_2022"]["base_cagr_to_strict_mdd_min"] == 3.0
    assert gates["selection_2023"]["base_cagr_to_strict_mdd_min"] == 3.0
    assert gates["train_2020_2022"]["mean_gross_underlying_bp_min"] == 35.0
    assert payload["sequence_rules"]["selection_rows_loaded_during_train"] is False
    assert payload["sequence_rules"]["selection_opens_only_after_train_pass"]
    cluster = economics["weekly_cluster_signflip"]
    assert cluster["draw_indices"] == "0..19999 formatted as five digits"
    assert cluster["stage_tokens"] == [
        "TRAIN_2020_2022",
        "SELECTION_2023",
    ]
    assert cluster["bit"] == (
        "most significant bit of digest byte zero; &0x80"
    )
    assert cluster["bit_one_multiplier"] == -1
    assert cluster["bit_zero_multiplier"] == 1


def test_evidence_boundary_discloses_probe_and_comparator_inventory_only() -> None:
    boundary = p.build_manifest()["evidence_boundary"]
    assert (
        boundary["historical_access_independently_auditable_from_pre_head"]
        is False
    )
    assert boundary["disclosure_is_conservative_contamination_envelope"] is True
    assert boundary["pristine_source_claim"] is False
    assert boundary["description_probe_rows_read"] == 8
    assert boundary["probe_identity_rows_read"] == 8
    assert boundary["slcs_rows_scanned_for_group_inventory"] == 1_685
    assert boundary["scaf_rows_scanned_for_group_inventory"] == 5_809
    assert boundary["source_amount_rate_or_availability_rows_read"] == 0
    assert boundary["smaf_centroids_fractures_ranks_tails_or_events_derived"] == 0
    assert boundary["candidate_overlap_metrics_computed"] == 0
    assert boundary["btc_market_rows_loaded"] == 0
    assert boundary["funding_data_rows_loaded"] == 0
    source = p.SCRIPT_PATH.read_text(encoding="utf-8")
    assert "import pandas" not in source
    assert "csv.DictReader" not in source
    assert ".read_csv(" not in source


def test_manifest_rejects_any_opened_evidence() -> None:
    payload = p.build_manifest()
    payload["evidence_boundary"]["btc_market_rows_loaded"] = 1
    with pytest.raises(RuntimeError, match="differs from code"):
        p.validate_manifest(payload)


def test_dependency_hash_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = p.sha256_file

    def drift(path: str | Path) -> str:
        if str(path) == str(p.PREREGISTRATION_DOCUMENT):
            return "0" * 64
        return original(path)

    monkeypatch.setattr(p, "sha256_file", drift)
    with pytest.raises(RuntimeError, match="frozen dependency changed"):
        p.validate_frozen_dependencies()


def test_write_once_is_repository_confined_and_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(p, "validate_frozen_dependencies", lambda: None)
    (tmp_path / "results").mkdir()
    output = Path("results/preregistration.json")
    payload = p.build_manifest()
    assert p.write_once(output, payload) == "created"
    assert (tmp_path / output).stat().st_mode & 0o777 == 0o444
    assert p.write_once(output, payload) == "verified_existing"
    written = tmp_path / output
    parsed = json.loads(written.read_text(encoding="utf-8"))
    assert parsed == payload
    written.chmod(0o644)
    written.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="noncanonical"):
        p.write_once(output, payload)
    with pytest.raises(RuntimeError, match="repository-relative"):
        p.write_once("../escape.json", payload)


@pytest.mark.parametrize("flag", ["O_NOFOLLOW", "O_DIRECTORY"])
def test_write_once_requires_secure_open_flags(
    flag: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(p, "validate_frozen_dependencies", lambda: None)
    (tmp_path / "results").mkdir()
    monkeypatch.delattr(p.os, flag)
    with pytest.raises(RuntimeError, match=rf"requires nonzero os\.{flag}"):
        p.write_once("results/preregistration.json", p.build_manifest())
    assert list((tmp_path / "results").iterdir()) == []


def test_write_once_requires_directory_relative_link_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(p, "validate_frozen_dependencies", lambda: None)
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(
        p.os,
        "supports_dir_fd",
        {p.os.open, p.os.unlink},
    )
    with pytest.raises(RuntimeError, match="dir_fd support for link"):
        p.write_once("results/preregistration.json", p.build_manifest())
    assert list((tmp_path / "results").iterdir()) == []


def test_write_once_rejects_symlinked_parent_and_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(p, "validate_frozen_dependencies", lambda: None)
    payload = p.build_manifest()
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / "results").symlink_to(external, target_is_directory=True)
    with pytest.raises(RuntimeError, match="parent path is unsafe"):
        p.write_once("results/preregistration.json", payload)
    assert list(external.iterdir()) == []

    (tmp_path / "results").unlink()
    (tmp_path / "results").mkdir()
    external_target = external / "target.json"
    external_target.write_text("unchanged\n", encoding="utf-8")
    (tmp_path / "results" / "preregistration.json").symlink_to(
        external_target
    )
    with pytest.raises(RuntimeError, match="output path is unsafe"):
        p.write_once("results/preregistration.json", payload)
    assert external_target.read_text(encoding="utf-8") == "unchanged\n"


def test_write_once_cleans_temporary_after_file_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(p, "validate_frozen_dependencies", lambda: None)
    results = tmp_path / "results"
    results.mkdir()
    real_fsync = p.os.fsync
    calls = 0

    def fail_first_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("forced file fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(p.os, "fsync", fail_first_fsync)
    with pytest.raises(OSError, match="forced file fsync failure"):
        p.write_once("results/preregistration.json", p.build_manifest())
    assert list(results.iterdir()) == []


def test_write_once_cleans_temporary_after_publication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(p, "validate_frozen_dependencies", lambda: None)
    results = tmp_path / "results"
    results.mkdir()

    def fail_publication(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("forced publication failure")

    monkeypatch.setattr(p, "_publish_temporary", fail_publication)
    with pytest.raises(OSError, match="forced publication failure"):
        p.write_once("results/preregistration.json", p.build_manifest())
    assert list(results.iterdir()) == []
