from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from training import (
    evaluate_cross_collateral_cohort_handoff_relay_source_gate as gate,
)
from training import (
    freeze_cross_collateral_cohort_handoff_relay_comparators as comparator_freeze,
)


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return gate.build_report()


def test_source_gate_retires_cchr_at_the_earliest_frozen_failure(
    report: dict[str, Any],
) -> None:
    assert report["decision"] == {
        "status": "retired_before_real_incidence",
        "pass": False,
        "reason": (
            "at least one required comparator member has an empty frozen clock, "
            "so novelty exposure is zero-variance and undefined"
        ),
        "failed_family": "far",
        "failed_member_count": 12,
        "cchr_source_incidence_opened": False,
        "economic_outcomes_opened": False,
        "repair_authorized": False,
    }
    assert len(report["failed_required_members"]) == 12
    assert all(
        member.startswith("far:") for member in report["failed_required_members"]
    )
    assert report["family_precheck"]["far"]["clock_rows"] == 0
    assert report["family_precheck"]["far"]["precheck_pass"] is False
    assert all(
        report["family_precheck"][family]["precheck_pass"] is True
        for family in ("pdlh", "dtv", "live")
    )


def test_source_gate_opens_no_source_comparator_or_outcome_rows(
    report: dict[str, Any],
) -> None:
    boundary = report["outcome_boundary"]
    assert boundary["pure_clock_rows_read"] == 0
    assert boundary["legacy_comparator_rows_read"] == 0
    assert boundary["cchr_source_values_read"] == 0
    assert boundary["cchr_incidence_rows_derived"] == 0
    assert boundary["market_rows_loaded"] == 0
    assert boundary["funding_rows_loaded"] == 0
    assert boundary["outcome_artifacts_parsed"] == 0
    assert boundary["return_or_pnl_fields_read"] == 0
    assert report["outcomes_opened"] is False
    assert report["authorization"]["outcome_evaluator"] is False


def test_source_gate_json_parsers_receive_only_json_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master_reader = gate.cchr._read_json
    freeze_reader = comparator_freeze._read_json
    parsed: list[str] = []

    def tracked_master(path: str | Path) -> dict[str, Any]:
        parsed.append(str(path))
        assert str(path).endswith(".json")
        return master_reader(path)

    def tracked_freeze(path: str | Path) -> dict[str, Any]:
        parsed.append(str(path))
        assert str(path).endswith(".json")
        return freeze_reader(path)

    monkeypatch.setattr(gate.cchr, "_read_json", tracked_master)
    monkeypatch.setattr(comparator_freeze, "_read_json", tracked_freeze)
    gate.build_report()
    assert parsed
    assert not any(path.endswith(".csv") or path.endswith(".csv.gz") for path in parsed)
    assert not any("alpha_scan" in path for path in parsed)


def test_source_gate_validation_rejects_repair_authorization(
    report: dict[str, Any],
) -> None:
    tampered = deepcopy(report)
    tampered["decision"]["repair_authorized"] = True
    core = {key: value for key, value in tampered.items() if key != "manifest_hash"}
    tampered["manifest_hash"] = gate.common.canonical_hash(core)
    with pytest.raises(RuntimeError, match="decision drift"):
        gate.validate_report(tampered, verify_files=False)


def test_source_gate_validation_rejects_precheck_failed_member_mismatch(
    report: dict[str, Any],
) -> None:
    tampered = deepcopy(report)
    tampered["family_precheck"]["far"]["zero_row_members"] = []
    tampered["family_precheck"]["far"]["zero_row_member_count"] = 0
    tampered["family_precheck"]["far"]["precheck_pass"] = True
    core = {key: value for key, value in tampered.items() if key != "manifest_hash"}
    tampered["manifest_hash"] = gate.common.canonical_hash(core)
    with pytest.raises(RuntimeError, match="precheck drift"):
        gate.validate_report(tampered, verify_files=False)


def test_family_precheck_names_every_zero_row_member() -> None:
    frozen = comparator_freeze.load_freeze()
    summary, failed = gate._family_precheck(frozen)
    assert len(failed) == 12
    assert failed == summary["far"]["zero_row_members"]
    assert summary["far"]["coverage_rows"] == {"train": 0, "selection": 0}
