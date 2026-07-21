from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.freeze_lvrt_comparator_cohort import (
    REQUIRED_MEMBERS,
    build_freeze,
    canonical_hash,
    publish,
)


def test_real_comparator_cohort_is_complete_before_lvrt_incidence() -> None:
    report = build_freeze()

    assert report["all_required_members_available"] is True
    assert report["required_member_count"] == 7
    assert report["required_members"] == list(REQUIRED_MEMBERS)
    assert set(report["members"]) == set(REQUIRED_MEMBERS)
    assert {key: value["rows"] for key, value in report["members"].items()} == {
        "afcs:afcs_144": 573,
        "bafr:primary": 11248,
        "live:cand_rex_veto_7": 218,
        "live:new_long_minimal_funding_premium": 82,
        "live:oi_upbit_ratio288_low": 140,
        "mfic:mfic_fast": 1566,
        "mfic:mfic_slow": 1635,
    }
    assert report["outcome_boundary"] == {
        "comparator_clock_rows_read": 15462,
        "lvrt_source_rows_read": 0,
        "lvrt_incidence_rows_derived": 0,
        "market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "performance_artifacts_parsed": 0,
        "return_or_pnl_fields_read": 0,
        "network_calls": 0,
        "economic_outcomes_computed": False,
    }
    core = {
        key: value
        for key, value in report.items()
        if key not in {"created_at", "manifest_hash"}
    }
    assert report["manifest_hash"] == canonical_hash(core)


def test_publication_is_create_only(tmp_path: Path) -> None:
    report = build_freeze()
    output = tmp_path / "freeze.json"

    publish(output, report)
    assert json.loads(output.read_text(encoding="utf-8"))["policy_id"] == "LVRT-72"
    with pytest.raises(FileExistsError):
        publish(output, report)
