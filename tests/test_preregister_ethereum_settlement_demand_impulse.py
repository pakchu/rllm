from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

from training import preregister_ethereum_settlement_demand_impulse as p


def fake_repository_identity() -> dict[str, Any]:
    paths = sorted(str(path) for path in p.committed_identity_paths())
    return {
        "git_blobs": {
            path: hashlib.sha1(path.encode("utf-8")).hexdigest()
            for path in paths
        },
        "sha256": {
            path: hashlib.sha256(path.encode("utf-8")).hexdigest()
            for path in paths
        },
        "whole_worktree_clean_required": False,
        "bound_paths_clean_against_HEAD_required": True,
    }


def manifest() -> dict[str, Any]:
    return p.build_manifest(fake_repository_identity())


def test_manifest_is_deterministic_self_hashed_and_evidence_closed() -> None:
    first = manifest()
    assert first == manifest()
    core = {key: value for key, value in first.items() if key != "manifest_hash"}
    assert first["manifest_hash"] == p.canonical_hash(core)
    p.validate_manifest(first)
    assert first["policy_id"] == "ESDI-288"
    assert all(first[name] is False for name in p.EVIDENCE_BOUNDARIES)
    assert first["producer_effects"] == {
        "network_calls": 0,
        "git_metadata_subprocess_calls": 2,
        "data_rows_opened": 0,
        "comparator_or_gross9_artifact_bytes_opened": 0,
        "bound_committed_code_config_and_lock_files_hashed": len(
            p.committed_identity_paths()
        ),
    }


def test_manifest_returns_fresh_nested_structures() -> None:
    first = manifest()
    first["source"]["boundaries"][0]["hash"] = "mutated"
    first["novelty"]["frozen_comparator_artifacts"].clear()
    first["gross9"]["weights"]["cand_rex_veto_7"] = 9.0
    second = manifest()
    assert second["source"]["boundaries"][0]["hash"].startswith("0x")
    assert len(second["novelty"]["frozen_comparator_artifacts"]) == 18
    assert second["gross9"]["weights"]["cand_rex_veto_7"] == 1.6
    assert p.GROSS9_WEIGHTS["cand_rex_veto_7"] == 1.6


def test_document_hash_is_frozen() -> None:
    assert p.sha256_file(p.DOCUMENT_PATH) == p.DOCUMENT_SHA256
    assert p.DOCUMENT_SHA256 == (
        "83fc8b6d83a992e8ecb3077fb5582999"
        "4073e9c7d3eeb783a8bfeb505e3462a8"
    )
    p.validate_frozen_document()


def test_dependency_hash_rejects_symlinked_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    real = tmp_path / "real"
    real.mkdir()
    (real / "file").write_bytes(b"safe")
    blob = hashlib.sha1(b"blob 4\0safe").hexdigest()
    assert p._committed_file_sha256("real/file", blob) == hashlib.sha256(
        b"safe"
    ).hexdigest()
    with pytest.raises(RuntimeError, match="committed blob"):
        p._committed_file_sha256("real/file", "0" * 40)
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="missing or unsafe"):
        p.sha256_file("linked/file")


def test_repository_identity_requires_clean_committed_plain_blobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = fake_repository_identity()
    calls: list[tuple[str, ...]] = []

    def fake_git(*args: str) -> bytes:
        calls.append(args)
        if args[0] == "status":
            return b""
        records = []
        for path, object_id in expected["git_blobs"].items():
            records.append(f"100644 blob {object_id}\t{path}".encode("utf-8"))
        return b"\0".join(records) + b"\0"

    monkeypatch.setattr(p, "_git", fake_git)
    monkeypatch.setattr(
        p,
        "_committed_file_sha256",
        lambda path, _blob: expected["sha256"][str(path)],
    )
    assert p.frozen_repository_identity() == expected
    assert [call[0] for call in calls] == ["status", "ls-tree"]

    monkeypatch.setattr(p, "_git", lambda *args: b"dirty\n")
    with pytest.raises(RuntimeError, match="committed and unchanged"):
        p.frozen_repository_identity()


def test_gross9_runtime_import_closure_and_environment_are_frozen() -> None:
    discovered = p.discover_runtime_code_closure()
    expected = tuple(
        sorted(path for path in p.RUNTIME_CODE_CLOSURE_PATHS if path.suffix == ".py")
    )
    assert discovered == expected
    p.validate_runtime_code_closure()
    assert {"pyproject.toml", "uv.lock"} <= {
        str(path) for path in p.RUNTIME_CODE_CLOSURE_PATHS
    }
    assert set(p.RUNTIME_CODE_ROOTS) <= set(discovered)
    environment = p.current_runtime_environment()
    assert {
        "python": environment["python"],
        "platform": environment["platform"],
        "packages": environment["packages"],
    } == p.FROZEN_RUNTIME_ENVIRONMENT
    assert environment["all_distributions_count"] == (
        p.FROZEN_DISTRIBUTION_INVENTORY_COUNT
    )
    assert environment["all_distributions_sha256"] == (
        p.FROZEN_DISTRIBUTION_INVENTORY_SHA256
    )
    assert len(environment["all_distributions"]) == 108
    assert environment["all_distributions"]["matplotlib"] == "3.10.8"
    assert environment["all_distributions"]["gymnasium"] == "1.2.3"
    assert environment["all_distributions"]["stable-baselines3"] == "2.7.1"
    p.validate_runtime_environment()


def test_epoch_boundaries_count_availability_median_and_vector_hash() -> None:
    source = manifest()["source"]
    assert p.epoch_blocks(4_531) == (16_311_600, 16_315_199, 16_315_263)
    assert p.epoch_blocks(7_004) == (25_214_400, 25_217_999, 25_218_063)
    assert source["epoch_count"] == 2_474
    assert source["first_source_block"] == 16_311_600
    assert source["last_source_block"] == 25_217_999
    assert source["rpc_attempts_per_request"] == 1
    assert source["rpc_retry_backoff_or_resume"] is False
    invariants = source["fail_closed_replay_invariants"]
    assert all(
        value
        for key, value in invariants.items()
        if key != "resume_after_terminal_replay_error_allowed"
    )
    assert invariants["resume_after_terminal_replay_error_allowed"] is False
    values = list(range(1, 3_601))
    assert p.median2(values) == 1_800 + 1_801
    assert p.median2(list(reversed(values))) == 3_601
    assert p.base_fee_vector_sha256(values) == (
        "21f5d847e83ef8925996e7b830691c5e"
        "3184df027d44bc01933a080495707db6"
    )
    with pytest.raises(ValueError, match="3,600"):
        p.median2(values[:-1])
    with pytest.raises(ValueError, match="positive integers"):
        p.median2([0] * 3_600)
    with pytest.raises(ValueError, match="uint256"):
        p.base_fee_vector_sha256([2**256] * 3_600)


def test_exact_rational_midrank_uses_cross_products_and_exact_ties() -> None:
    current = (2, 3)
    prior = [(1, 2)] * 90 + [(4, 6)] * 60 + [(3, 4)] * 30
    assert p.exact_rational_midrank(current, prior) == Fraction(2, 3)
    assert p.compare_rationals((2, 3), (4, 6)) == 0
    assert p.compare_rationals((10**30 + 1, 10**30), (1, 1)) == 1
    threshold_prior = [(1, 2)] * 135 + [(3, 2)] * 45
    assert p.exact_rational_midrank((1, 1), threshold_prior) == Fraction(3, 4)
    with pytest.raises(ValueError, match="exactly 180"):
        p.exact_rational_midrank((1, 1), threshold_prior[:-1])
    with pytest.raises(ValueError, match="positive denominators"):
        p.compare_rationals((1, 0), (1, 1))


def test_entry_clock_and_random_side_helpers_are_pure_and_exact() -> None:
    assert p.ceil_5m_plus_one_bar(0) == 300
    assert p.ceil_5m_plus_one_bar(300) == 600
    assert p.ceil_5m_plus_one_bar(301) == 900
    assert p.ceil_5m_plus_one_bar(599) == 900
    with pytest.raises(ValueError, match="nonnegative"):
        p.ceil_5m_plus_one_bar(-1)
    with pytest.raises(TypeError, match="integer"):
        p.ceil_5m_plus_one_bar(True)
    assert p.canonical_signal_id(5_000) == "ESDI-288|primary|epoch_id=5000"
    digest = hashlib.sha256(
        b"ESDI-288|primary|epoch_id=5000|RANDOM_SIDE"
    ).hexdigest()
    assert digest == (
        "1c0beb9dc023ace340bc0b331e5bdadc"
        "f0456bb3bcad1d7061b49628da72df49"
    )
    assert p.deterministic_random_side(5_000) == "LONG"
    with pytest.raises(ValueError, match="outside"):
        p.deterministic_random_side(4_000)


def test_frozen_novelty_metric_functions_are_exact_and_fail_closed() -> None:
    assert p.entries_in_domain([0, 300, 600, 900], 300, 900) == (300, 600)
    assert p.exact_entry_jaccard([0, 300], [300, 600]) == Fraction(1, 3)
    assert p.bidirectional_entry_containment(
        [0, 1_000],
        [100, 2_000],
        150,
    ) == Fraction(1, 2)
    assert p.fraction_at_most(Fraction(1, 5), 1, 5) is True
    assert p.fraction_at_most(Fraction(1, 5) + Fraction(1, 10_000), 1, 5) is False
    assert p.fraction_below(Fraction(9, 10), 9, 10) is False
    assert p.fraction_below(Fraction(899, 1_000), 9, 10) is True
    with pytest.raises(ValueError, match="strictly increasing"):
        p.exact_entry_jaccard([300, 300], [300])

    left = p.signed_exposure_5m([(0, 600, 1)], 0, 1_200)
    right = p.signed_exposure_5m([(300, 900, -1)], 0, 1_200)
    assert left == (1, 1, 0, 0)
    assert right == (0, -1, -1, 0)
    assert p.occupied_bar_jaccard(left, right) == Fraction(1, 3)
    assert p.squared_signed_exposure_pearson(left, right) == Fraction(0, 1)
    assert p.squared_signed_exposure_pearson(left, left) == Fraction(1, 1)
    with pytest.raises(ValueError, match="zero variance"):
        p.squared_signed_exposure_pearson([0, 0], [0, 1])
    with pytest.raises(ValueError, match="exact integers"):
        p.squared_signed_exposure_pearson([False, 1], [0, 1])
    with pytest.raises(ValueError, match="exact integers"):
        p.occupied_bar_jaccard([0, 2], [0, 1])
    with pytest.raises(ValueError, match="nonoverlapping"):
        p.signed_exposure_5m([(0, 600, 1), (300, 900, -1)], 0, 1_200)


def test_feature_execution_calendars_and_no_parameter_search() -> None:
    payload = manifest()
    feature = payload["feature_and_signal"]
    assert feature["lag_epochs"] == 2
    assert feature["rank_history"].startswith("exactly previous 180")
    assert feature["rank"] == "(2*L+E)/360"
    assert feature["threshold"] == {
        "operator": ">=",
        "numerator": 3,
        "denominator": 4,
    }
    assert feature["signal_id"].startswith("ESDI-288|primary|epoch_id=")
    assert feature["side"] == {
        "positive": "LONG",
        "negative": "SHORT",
        "zero": "ABSTAIN",
    }
    assert feature["parameter_search_or_alternative_rule"] is False
    assert "grid" not in repr(payload).lower()

    execution = payload["execution"]
    assert execution["hold_bars_5m"] == 288
    assert execution["hold_seconds"] == 86_400
    assert execution["leverage"] == 0.5
    assert execution["base_cost_bp_per_notional_side"] == 6
    assert execution["stress_cost_bp_per_notional_side"] == 10
    assert execution["funding"]["interval"] == (
        "entry_time <= funding_time < exit_time"
    )
    assert execution["candidate_order"] == [
        "entry_time",
        "available_at",
        "epoch_id",
        "side",
    ]

    calendars = payload["calendars"]
    assert calendars["full"] == [
        "2023-06-01T00:00:00Z",
        "2026-06-01T00:00:00Z",
    ]
    assert calendars["selection"] == [
        "2023-06-01T00:00:00Z",
        "2025-01-01T00:00:00Z",
    ]
    assert set(calendars["selection_reports"]) == {"2023H2", "2024H1", "2024H2"}


def test_controls_support_and_source_contract() -> None:
    payload = manifest()
    controls = payload["controls"]
    assert controls["own_nonoverlap_clock"] == [
        "base_fee_one_epoch_stale",
        "gas_utilization_only",
        "base_fee_no_tail",
    ]
    assert controls["same_primary_parent_set"] == [
        "exact_direction_flip",
        "deterministic_random_side",
        "constant_long",
        "constant_short",
        "one_bar_delayed_entry",
    ]
    assert "epoch_id=<canonical decimal integer>" in (
        controls["definitions"]["deterministic_random_side"]
    )
    assert controls["controls_cannot_replace_or_repair_primary"] is True

    support = payload["support_gates"]
    assert support["source"]["exact_epochs"] == 2_474
    assert support["selection"] == {
        "total_min": 45,
        "2023H2_min": 12,
        "2024H1_min": 12,
        "2024H2_min": 12,
        "each_side_min": 14,
        "maximum_month_share": 0.20,
    }
    assert support["future25"]["each_side_min"] == 8
    assert support["future26"]["each_side_min"] == 4
    assert support["independent_control_maxima_strict"] == {
        "exact_entry_jaccard": {
            "operator": "<",
            "numerator": 9,
            "denominator": 10,
        },
        "candidate_24h_containment": {
            "operator": "<",
            "numerator": 19,
            "denominator": 20,
        },
    }


def test_comparator_registry_is_exhaustive_capability_aware_and_hash_bound() -> None:
    registry = manifest()["novelty"]["frozen_comparator_artifacts"]
    assert set(registry) == {
        "CAIM",
        "WCTR-288",
        "BFWC-288",
        "BFRT-288",
        "CDLTR-72A",
        "CDLTR-prior-chain-network-bundle",
        "AMTR-48",
        "EBLR-60/30",
        "UGCI-288",
        "WCDR-2016",
        "WTSL-168-SOURCE-SEEN",
        "WSCF-72-SOURCE-FAMILY-SEEN",
        "FCCM-72",
        "URCD-72",
        "SQFD-6",
        "SDDR-12",
        "UCBR-12",
        "BFMWD-primary-variants",
    }
    for artifact in registry.values():
        assert len(artifact["sha256"]) == 64
        assert len(artifact["header_line_sha256"]) == 64
        int(artifact["sha256"], 16)
        int(artifact["header_line_sha256"], 16)

    bundle = registry["CDLTR-prior-chain-network-bundle"]
    assert {"NTB-7", "NWE-8", "chain_activity_impulse_momentum"} <= set(
        bundle["directional_interval_groups"]
    )
    assert "NWE-7" in bundle["timestamp_only_groups"]
    assert bundle["required_metrics_by_capability"]["timestamp_only"] == [
        "exact_entry_jaccard",
        "candidate_24h_containment",
    ]
    assert registry["BFMWD-primary-variants"]["groups"] == [
        "bfmwd_w12_d3_z10_h12",
        "bfmwd_w24_d3_z10_h12",
        "bfmwd_w12_d6_z10_h12",
        "bfmwd_w24_d6_z10_h12",
    ]
    assert registry["AMTR-48"]["exit_column"] == "scheduled_exit"
    assert registry["EBLR-60/30"]["required_columns"] == [
        "candidate",
        "direction",
        "entry_time",
        "planned_exit_time",
    ]
    assert all(len(artifact["comparison_domain"]) == 2 for artifact in registry.values())


def test_novelty_and_gross9_contract_are_fully_bound() -> None:
    payload = manifest()
    novelty = payload["novelty"]
    assert novelty["prior_source_family_thresholds_exact_inclusive"] == {
        "exact_entry_jaccard": {"numerator": 1, "denominator": 5},
        "candidate_24h_containment": {"numerator": 1, "denominator": 2},
        "squared_signed_exposure_pearson": {
            "numerator": 4,
            "denominator": 25,
        },
    }
    assert novelty[
        "gross9_each_positive_weight_sleeve_thresholds_exact_inclusive"
    ] == {
        "exact_entry_jaccard": {"numerator": 1, "denominator": 10},
        "candidate_6h_containment": {"numerator": 7, "denominator": 20},
        "occupied_bar_jaccard": {"numerator": 1, "denominator": 4},
        "squared_signed_exposure_pearson": {
            "numerator": 49,
            "denominator": 400,
        },
    }
    assert novelty["all_frozen_comparator_artifacts_must_be_evaluated"] is True
    assert "never zero-filled" in novelty["metric_applicability"]["timestamp_only"]
    assert novelty["minimum_count_is_after_common_domain_filter"] is True
    assert novelty["metric_definitions"][
        "absolute_signed_exposure_pearson_gate"
    ].startswith("exact squared Pearson")
    assert novelty["gross9_common_domain"] == [
        "2023-06-01T00:00:00Z",
        "2026-06-01T00:00:00Z",
    ]

    gross9 = payload["gross9"]
    assert gross9["weights"] == p.GROSS9_WEIGHTS
    assert sum(gross9["weights"].values()) == 9.0
    assert gross9["candidate_weights"] == [0.25, 0.50, 0.75, 1.00]
    assert gross9["treatment"] == (
        "scale every sleeve by (9-w)/9 and add ESDI at w"
    )
    assert set(gross9["authority"]["sleeves"]) == set(p.GROSS9_WEIGHTS)
    assert set(gross9["authority"]["runtime"]) == {
        "portfolio_live.py",
        "rank7_runtime.py",
        "rex_llm_live.py",
    }
    closure = gross9["authority"]["runtime_code_closure"]
    assert closure["paths"] == [str(path) for path in p.RUNTIME_CODE_CLOSURE_PATHS]
    assert closure["ast_import_closure_must_match_before_artifact_creation"] is True
    assert closure["exact_runtime_environment"] == p.current_runtime_environment()
    assert gross9["authority"]["clock_reconstruction"][
        "five_signed_sleeves_required"
    ] is True
    assert gross9["freeze_rank"] == 1
    assert gross9["future_rerank_or_alternate_weight"] is False


def test_economics_sequence_and_boundary_booleans_are_strict() -> None:
    payload = manifest()
    gates = payload["economic_contract"]["standalone_gate_base_and_stress_each_period"]
    assert gates == {
        "absolute_return": ">0",
        "full_calendar_cagr_to_strict_mdd": ">=3.0",
        "strict_mdd": "<=0.15",
        "mean_gross_underlying_bp": ">=20",
        "calendar_month_clustered_signflip_p": "<=0.10",
    }
    assert len(payload["strict_sequence"]) == 10
    assert "before_full_source_replay" in payload["strict_sequence"][3]
    assert payload["strict_sequence"][7] == (
        "reconstruct_and_run_novelty_stop_on_first_failure"
    )
    assert payload["sequence_rules"] == {
        "stop_at_first_failure": True,
        "later_periods_veto_only": True,
        "parameter_repair_polarity_inversion_or_rank2": False,
        "ordinary_failure_repair_under_policy_identity": False,
    }
    assert all(payload[name] is False for name in p.EVIDENCE_BOUNDARIES)


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/out.json",
        "../results/out.json",
        "results/../out.json",
        "~/out.json",
        "results/another-preregistration.json",
    ],
)
def test_output_path_must_be_the_frozen_singleton(path: str) -> None:
    with pytest.raises(RuntimeError):
        p._output_relative(path)
    assert p._output_relative(p.DEFAULT_OUTPUT) == p.DEFAULT_OUTPUT


def test_write_once_creates_verifies_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = fake_repository_identity()
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        p, "frozen_repository_identity", lambda: copy.deepcopy(identity)
    )
    (tmp_path / "docs").mkdir()
    source_doc = (
        Path(__file__).resolve().parents[1] / p.DOCUMENT_PATH
    ).read_bytes()
    (tmp_path / p.DOCUMENT_PATH).write_bytes(source_doc)
    (tmp_path / "results").mkdir()
    payload = p.build_manifest(identity)

    status, written = p.write_once(p.DEFAULT_OUTPUT, payload)
    assert status == "created"
    assert written == payload
    artifact = tmp_path / p.DEFAULT_OUTPUT
    assert artifact.read_bytes() == p.canonical_manifest_bytes(payload)
    assert p.write_once(p.DEFAULT_OUTPUT, payload)[0] == "verified_existing"

    artifact.chmod(0o600)
    artifact.write_bytes(b"drift\n")
    with pytest.raises(RuntimeError, match="drift"):
        p.write_once(p.DEFAULT_OUTPUT, payload)


def test_write_once_rejects_symlink_parent_output_and_nonregular(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = fake_repository_identity()
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        p, "frozen_repository_identity", lambda: copy.deepcopy(identity)
    )
    (tmp_path / "docs").mkdir()
    real_doc = Path(__file__).resolve().parents[1] / p.DOCUMENT_PATH
    (tmp_path / p.DOCUMENT_PATH).write_bytes(real_doc.read_bytes())

    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "results").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="parent is unsafe"):
        p.write_once()

    (tmp_path / "results").unlink()
    (tmp_path / "results").mkdir()
    artifact = tmp_path / p.DEFAULT_OUTPUT
    artifact.symlink_to(outside / "target.json")
    with pytest.raises(RuntimeError, match="unsafe"):
        p.write_once()

    artifact.unlink()
    os.mkfifo(artifact)
    with pytest.raises(RuntimeError, match="regular file"):
        p.write_once()


def test_default_artifact_path_is_frozen() -> None:
    assert p.DEFAULT_OUTPUT == Path(
        "results/ethereum_settlement_demand_impulse_"
        "preregistration_2026-07-30.json"
    )
