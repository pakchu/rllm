from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from training import build_block_clearing_relational_topology_support as s


REPORT = Path(
    "results/block_clearing_relational_topology_support_2026-07-24.json"
)
CLOCK = Path(
    "data/block_clearing_relational_topology_clocks_2020_2023.csv.gz"
)
REPORT_SHA256 = (
    "9ccccf7a3176fcf86baddacb65c11bbde78ea73ed7ab18d3594b0e6327567055"
)
CLOCK_SHA256 = (
    "c0420c7175410a822455a0d68bf877cba94a2ec17b31f6d9a588244cb893c909"
)
MANIFEST_HASH = (
    "e2b2d7301d204043f2df33f4453da82112fb5db7bfb9aed66a74bee6ec76932b"
)
FRAME_HASH = (
    "d3ae6a2ea133ecc6567d981f0bd5479c439cbb16b1f21d826305425c29c3b3d7"
)


def _report() -> dict[str, object]:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_support_artifacts_are_hash_bound_and_canonical() -> None:
    report = _report()
    assert s.sha256_file(REPORT) == REPORT_SHA256
    assert s.sha256_file(CLOCK) == CLOCK_SHA256
    core = {
        key: value
        for key, value in report.items()
        if key != "manifest_hash"
    }
    assert report["manifest_hash"] == MANIFEST_HASH
    assert s.canonical_hash(core) == MANIFEST_HASH
    assert report["clock"] == {
        "path": str(s.DEFAULT_CLOCK_OUTPUT),
        "sha256": CLOCK_SHA256,
        "frame_hash": FRAME_HASH,
        "rows": 2755,
        "columns": list(s.CLOCK_COLUMNS),
    }

    clock = pd.read_csv(CLOCK, usecols=list(s.CLOCK_COLUMNS))
    assert clock.columns.tolist() == list(s.CLOCK_COLUMNS)
    assert len(clock) == 2755
    assert s._frame_hash(clock) == FRAME_HASH


def test_support_retired_before_any_market_outcome() -> None:
    report = _report()
    assert report["artifact_eligible"] is True
    assert report["decision"] == (
        "retire_BCRT_72_unchanged_before_market_outcomes"
    )
    assert report["authorized_next_stage"] is None
    assert report["first_failing_stage"] == "source_support"
    assert report["first_failing_check"] == (
        "max_entry_gap_days_2020_2022"
    )
    assert report["source_support_passed"] is False
    assert report["token_support_passed"] is False
    failed_source = [
        name
        for name, passed in report["source_support_checks"].items()
        if not passed
    ]
    assert failed_source == ["max_entry_gap_days_2020_2022"]
    assert all(report["token_support_checks"].values())
    assert report["eval_source_report_only"]["boolean_gate"] is False
    assert report["eval_source_report_only"][
        "may_authorize_continue_retire_repair_or_selection"
    ] is False

    boundary = report["outcome_boundary"]
    for field in (
        "BTC_market_rows_decoded",
        "funding_rows_decoded",
        "comparator_rows_decoded",
        "future_return_rows_decoded",
        "return_or_PnL_fields_decoded",
        "PnL_CAGR_MDD_values_decoded",
        "post_2023_rows_decoded",
        "model_labels_created",
        "model_training_runs",
        "network_calls",
    ):
        assert boundary[field] == 0


def test_source_only_funnel_and_year_boundary_gaps_are_exact() -> None:
    report = _report()
    assert report["bucket_audit"]["formed_buckets"] == 2918
    assert report["bucket_audit"]["prefix_replay_buckets_checked"] == 2918
    assert report["bucket_audit"]["prefix_replay_passed"] is True
    assert report["feature_funnel"] == {
        "formed_buckets": 2918,
        "rank_complete_states": 2792,
        "first_rank_complete_predecessor_only": 1,
        "token_ready_states": 2791,
    }
    assert report["reservation_funnel"] == {
        "token_ready": 2791,
        "globally_reserved": 2787,
        "overlap_suppressed": 4,
        "split_suppressed_after_reservation": 32,
        "emitted": 2755,
    }
    stats = report["clock_statistics"]
    assert stats["development"]["events"] == 2035
    assert stats["train"]["events"] == 1314
    assert stats["2020"]["events"] == 595
    assert stats["2021"]["events"] == 719
    assert stats["2022"]["events"] == 721
    assert stats["2023"]["events"] == 720
    assert stats["development"]["maximum_gap_days"] == 5

    clock = pd.read_csv(CLOCK, usecols=["entry_time"])
    entries = pd.to_datetime(clock["entry_time"], utc=True).sort_values()
    gaps = entries.diff().dt.total_seconds().div(86_400)
    observed = gaps[gaps.gt(3.0)].tolist()
    assert observed == [
        5.038194444444445,
        4.777777777777778,
        4.920138888888889,
    ]
