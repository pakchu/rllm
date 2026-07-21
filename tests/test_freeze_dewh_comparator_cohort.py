from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training.freeze_dewh_comparator_cohort import (
    REQUIRED_MEMBERS,
    build_freeze,
    canonical_hash,
    publish,
)


def test_freeze_contains_all_eight_preincidence_comparators() -> None:
    report = build_freeze()

    assert report["required_member_count"] == 8
    assert report["required_members"] == list(REQUIRED_MEMBERS)
    assert report["all_required_members_available"] is True
    assert report["members"]["dehr:dehr_72_normalized"] == {
        "rows": 159,
        "first_entry": "2020-07-23T09:15:00+00:00",
        "last_exit": "2022-12-17T15:15:00+00:00",
        "year_counts": {"2020": 55, "2021": 50, "2022": 54},
        "longs": 62,
        "shorts": 97,
    }
    assert report["outcome_boundary"] == {
        "comparator_clock_rows_read": 15621,
        "dewh_source_rows_read": 0,
        "dewh_incidence_rows_derived": 0,
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
    implementation = Path(report["implementation_binding"]["path"])
    assert (
        hashlib.sha256(implementation.read_bytes()).hexdigest()
        == report["implementation_binding"]["sha256"]
    )


def test_freeze_publish_is_create_only(tmp_path: Path) -> None:
    report = build_freeze()
    output = tmp_path / "freeze.json"

    publish(output, report)
    assert json.loads(output.read_text())["required_member_count"] == 8
    with pytest.raises(FileExistsError):
        publish(output, report)
