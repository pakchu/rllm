from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from training.evaluate_liquidity_vacuum_replenishment_transition_support import (
    BAR,
    build_outputs,
    build_relay_candidates,
    canonical_hash,
    publish,
    _strict_prior_midrank,
)


@pytest.fixture(scope="module")
def real_outputs() -> tuple[dict[str, Any], bytes | None]:
    return build_outputs()


def test_strict_prior_midrank_excludes_current_and_averages_ties() -> None:
    values = pd.Series([1.0, 2.0, 2.0, 3.0])
    valid = pd.Series([True, True, True, True])

    ranks = _strict_prior_midrank(values, valid, window=2)

    assert np.isnan(ranks.iloc[0])
    assert np.isnan(ranks.iloc[1])
    assert ranks.iloc[2] == pytest.approx(0.75)
    assert ranks.iloc[3] == pytest.approx(1.0)


def test_relay_uses_first_flip_and_inclusive_age_twelve() -> None:
    start = datetime(2020, 2, 1, tzinfo=timezone.utc)
    dates = [start + index * BAR for index in range(14)]
    valid = np.ones(14, dtype=bool)
    setup = np.zeros(14, dtype=bool)
    replenishment = np.zeros(14, dtype=bool)
    flow_sign = np.zeros(14, dtype=np.int8)
    setup[0] = True
    flow_sign[0] = 1
    replenishment[4] = True
    flow_sign[4] = 1
    replenishment[12] = True
    flow_sign[12] = -1

    rows, audit = build_relay_candidates(
        dates,
        valid,
        setup,
        replenishment,
        flow_sign,
        require_flip=True,
    )

    assert len(rows) == 1
    assert rows[0].setup_time == dates[0]
    assert rows[0].confirmation_time == dates[12]
    assert rows[0].side == -1
    assert audit.confirmations == 1
    assert audit.expiries == 0


def test_source_gap_cancels_active_episode() -> None:
    start = datetime(2020, 2, 1, tzinfo=timezone.utc)
    dates = [start + index * BAR for index in range(5)]
    valid = np.ones(5, dtype=bool)
    valid[2] = False
    setup = np.zeros(5, dtype=bool)
    setup[0] = True
    replenishment = np.zeros(5, dtype=bool)
    replenishment[3] = True
    flow_sign = np.asarray([1, 0, 0, -1, 0], dtype=np.int8)

    rows, audit = build_relay_candidates(
        dates,
        valid,
        setup,
        replenishment,
        flow_sign,
        require_flip=True,
    )

    assert rows == []
    assert audit.gap_cancellations == 1


def test_real_lvrt_is_rejected_for_source_sparsity(
    real_outputs: tuple[dict[str, Any], bytes | None],
) -> None:
    report, clock_bytes = real_outputs

    assert clock_bytes is None
    assert report["combined_gate_passed"] is False
    assert report["pure_clock"] is None
    assert report["failure_action"] == "retire_before_economic_evaluation"
    assert report["source_audit"] == {
        "source_rows": 420732,
        "full_grid_rows": 420768,
        "missing_grid_rows": 36,
        "gap_day_rows": 1440,
        "valid_rows": 419302,
        "rank_ready_rows": 399705,
        "gap_days": [
            "2020-04-15",
            "2021-02-09",
            "2021-02-24",
            "2021-05-19",
            "2022-09-06",
        ],
        "source_columns_read": [
            "date",
            "agg_trade_count",
            "event_notional_hhi",
            "normalized_effective_event_count",
            "signed_event_imbalance",
            "max_same_sign_run_share",
            "interarrival_burstiness",
        ],
        "forbidden_source_columns_read": 0,
        "first_timestamp": "2020-01-01T00:00:00+00:00",
        "last_timestamp": "2023-12-31T23:55:00+00:00",
    }
    assert report["primary"]["build_audit"] == {
        "setups": 49,
        "confirmations": 1,
        "expiries": 48,
        "gap_cancellations": 0,
        "active_at_end": False,
    }
    assert report["primary"]["schedule_audit"] == {
        "raw_candidates": 1,
        "split_contained_candidates": 1,
        "split_boundary_drops": 0,
        "overlap_suppressions": 0,
        "accepted_candidates": 1,
    }
    assert report["primary"]["splits"]["train"]["accepted_events"] == 1
    assert report["primary"]["splits"]["selection"]["accepted_events"] == 0
    assert report["novelty_gate"] == {
        "checks": {},
        "passed": False,
        "skipped_reason": "source support failed",
    }
    failed = [
        name for name, passed in report["support_gate"]["checks"].items() if not passed
    ]
    assert "train_total_between_100_and_360" in failed
    assert "selection_total_between_45_and_180" in failed
    assert report["outcome_boundary"]["market_rows_loaded"] == 0
    assert report["outcome_boundary"]["funding_rows_loaded"] == 0
    assert report["outcome_boundary"]["economic_outcomes_computed"] is False
    core = {
        key: value
        for key, value in report.items()
        if key not in {"created_at", "result_hash"}
    }
    assert report["result_hash"] == canonical_hash(core)


def test_report_only_publication_is_create_only(
    tmp_path: Path,
    real_outputs: tuple[dict[str, Any], bytes | None],
) -> None:
    report, clock_bytes = real_outputs
    report_path = tmp_path / "report.json"
    clock_path = tmp_path / "clock.csv.gz"

    publish(report_path, clock_path, report, clock_bytes)
    assert report_path.exists()
    assert not clock_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))["policy_id"] == (
        "LVRT-72"
    )
    with pytest.raises(FileExistsError):
        publish(report_path, clock_path, report, clock_bytes)
