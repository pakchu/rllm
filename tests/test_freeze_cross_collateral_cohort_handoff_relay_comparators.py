from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from training import preregister_cchr_pure_clock_exports as export_prereg
from training import preregister_cross_collateral_cohort_handoff_relay as cchr
from training import (
    freeze_cross_collateral_cohort_handoff_relay_comparators as freeze,
)


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return freeze.build_freeze()


def test_freeze_binds_every_declared_comparator_without_opening_rows(
    payload: dict[str, Any],
) -> None:
    assert payload["comparator_member_count"] == 62
    assert payload["comparator_candidate_map_sha256"] == (
        freeze.clock_common.candidate_map_hash(cchr.comparator_candidate_map())
    )
    assert set(payload["generated_families"]) == set(export_prereg.FAMILIES)
    assert set(payload["legacy_comparators"]) == {"ccpr", "dlpd"}
    assert (
        sum(
            family["required_bindings"]["member_count"]
            for family in payload["generated_families"].values()
        )
        == 55
    )
    assert (
        sum(family["member_count"] for family in payload["legacy_comparators"].values())
        == 7
    )
    assert payload["outcomes_opened"] is False
    assert payload["outcome_boundary"] == freeze.OUTCOME_BOUNDARY
    assert payload["authorization"] == freeze.AUTHORIZATION
    assert payload["authorization"]["outcome_evaluator"] is False
    assert all(
        binding["binding_mode"] == "frozen_legacy_source_projection"
        and binding["generated_export_requirement_applicable"] is False
        for binding in payload["legacy_comparators"].values()
    )


def test_every_generated_family_binds_the_exact_required_fields(
    payload: dict[str, Any],
) -> None:
    required = set(cchr.comparator_freeze_requirement()["must_bind"])
    for family, binding in payload["generated_families"].items():
        assert set(binding["required_bindings"]) == required
        assert binding["clock_rows_read"] == 0
        requirement = cchr.PURE_CLOCK_REQUIREMENTS[family]
        assert binding["paths"]["pure_clock"] == requirement["path"]
        assert binding["paths"]["export_manifest"] == requirement["export_manifest"]
        assert (
            binding["required_bindings"]["member_count"]
            == (requirement["required_member_count"])
        )


def test_zero_incidence_family_is_bound_transparently(
    payload: dict[str, Any],
) -> None:
    far = payload["generated_families"]["far"]
    assert far["clock_metadata"]["rows"] == 0
    assert set(far["clock_metadata"]["rows_by_candidate"].values()) == {0}
    assert far["required_bindings"]["coverage"] == {
        "train": {
            "rows": 0,
            "observed_member_count": 0,
            "first_decision_time": None,
            "last_exit_boundary": None,
        },
        "selection": {
            "rows": 0,
            "observed_member_count": 0,
            "first_decision_time": None,
            "last_exit_boundary": None,
        },
    }


def test_freeze_json_reader_never_receives_clock_or_outcome_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = freeze._read_json
    parsed: list[str] = []

    def tracked(path: str | Path) -> dict[str, Any]:
        text = str(path)
        parsed.append(text)
        if not text.endswith(".json"):
            raise AssertionError(f"non-JSON row artifact was parsed: {text}")
        return original(path)

    monkeypatch.setattr(freeze, "_read_json", tracked)
    freeze.build_freeze()
    expected = {
        str(export_prereg.MASTER_PREREGISTRATION),
        *(str(path) for path in export_prereg.PREREGISTRATION_OUTPUTS.values()),
        *(
            str(cchr.PURE_CLOCK_REQUIREMENTS[family]["export_manifest"])
            for family in export_prereg.FAMILIES
        ),
    }
    assert set(parsed) == expected
    assert len(parsed) == len(expected)
    assert not any("alpha_scan" in path for path in parsed)


def test_freeze_validation_is_order_insensitive_after_sorted_json_roundtrip(
    payload: dict[str, Any],
) -> None:
    roundtrip = json.loads(json.dumps(payload, sort_keys=True))
    freeze.validate_freeze(roundtrip, verify_files=False)


def test_freeze_validation_rejects_tampered_required_binding(
    payload: dict[str, Any],
) -> None:
    tampered = deepcopy(payload)
    del tampered["generated_families"]["pdlh"]["required_bindings"]["pure_clock_sha256"]
    core = {key: value for key, value in tampered.items() if key != "manifest_hash"}
    tampered["manifest_hash"] = freeze.clock_common.canonical_hash(core)
    with pytest.raises(RuntimeError, match="required freeze binding drift"):
        freeze.validate_freeze(tampered, verify_files=False)


def test_non_file_validation_rejects_tampered_family_path(
    payload: dict[str, Any],
) -> None:
    tampered = deepcopy(payload)
    tampered["generated_families"]["dtv"]["paths"]["pure_clock"] = (
        "results/not-the-frozen-clock.csv.gz"
    )
    core = {key: value for key, value in tampered.items() if key != "manifest_hash"}
    tampered["manifest_hash"] = freeze.clock_common.canonical_hash(core)
    with pytest.raises(RuntimeError, match="frozen path drift"):
        freeze.validate_freeze(tampered, verify_files=False)


def test_non_file_validation_rejects_tampered_legacy_member_count(
    payload: dict[str, Any],
) -> None:
    tampered = deepcopy(payload)
    tampered["legacy_comparators"]["ccpr"]["member_count"] = 5
    core = {key: value for key, value in tampered.items() if key != "manifest_hash"}
    tampered["manifest_hash"] = freeze.clock_common.canonical_hash(core)
    with pytest.raises(RuntimeError, match="legacy member-count drift"):
        freeze.validate_freeze(tampered, verify_files=False)


def test_non_file_validation_rejects_invalid_coverage_timestamp(
    payload: dict[str, Any],
) -> None:
    tampered = deepcopy(payload)
    tampered["generated_families"]["pdlh"]["required_bindings"]["coverage"][
        "selection"
    ]["first_decision_time"] = "2023-01-05T01:02:00Z"
    core = {key: value for key, value in tampered.items() if key != "manifest_hash"}
    tampered["manifest_hash"] = freeze.clock_common.canonical_hash(core)
    with pytest.raises(RuntimeError, match="coverage timestamp drift"):
        freeze.validate_freeze(tampered, verify_files=False)


def test_json_reader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"family":"pdlh","family":"dtv"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        freeze._read_json(path)


def test_create_only_publication_preserves_existing_artifact(tmp_path: Path) -> None:
    target = tmp_path / "freeze.json"
    freeze._publish_json_create_only({"version": 1}, target)
    original = target.read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        freeze._publish_json_create_only({"version": 2}, target)
    assert target.read_bytes() == original


def test_publication_rejects_prevalidation_directory_identity_change(
    tmp_path: Path,
) -> None:
    target = tmp_path / "freeze.json"
    with pytest.raises(RuntimeError, match="directory identity changed"):
        freeze._publish_json_create_only(
            {"version": 1},
            target,
            expected_directory_identity=(-1, -1),
        )
    assert not target.exists()
    assert not list(tmp_path.glob(".freeze.json.*.json.tmp"))


def test_publication_rolls_back_owned_target_on_directory_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "freeze.json"
    original_fsync = freeze.os.fsync
    calls = 0

    def fail_directory_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic directory fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(freeze.os, "fsync", fail_directory_fsync)
    with pytest.raises(OSError, match="synthetic directory fsync failure"):
        freeze._publish_json_create_only({"version": 1}, target)
    assert not target.exists()
