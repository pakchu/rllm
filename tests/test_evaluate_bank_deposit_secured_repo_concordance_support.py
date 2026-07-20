from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from training import evaluate_bank_deposit_secured_repo_concordance_support as support
from training import sofr_rate_dislocation_clock as sofr_clock


def _event(name: str, entry: datetime, *, side: int = 1, hours: int = 72) -> support.Event:
    return support.Event(
        clock_name=name,
        release_date=entry.date().isoformat(),
        decision_time=(entry - timedelta(minutes=5)).isoformat(),
        entry_time=entry.isoformat(),
        exit_time=(entry + timedelta(hours=hours)).isoformat(),
        side=side,
        h8_sign=-side,
        repo_sign=-side,
        repo5_bp=-side,
        sofr_effective_date=entry.date().isoformat(),
        sofr_available_at=(entry - timedelta(hours=2)).isoformat(),
    )


def test_h8_sign_requires_finite_nonzero_score_and_two_confirming_components() -> None:
    assert support._h8_sign(pd.Series({"stress_score": 1.0, "agreement_count": 2})) == 1
    assert support._h8_sign(pd.Series({"stress_score": -1.0, "agreement_count": 3})) == -1
    assert support._h8_sign(pd.Series({"stress_score": 1.0, "agreement_count": 1})) == 0
    assert support._h8_sign(pd.Series({"stress_score": 0.0, "agreement_count": 3})) == 0
    assert support._h8_sign(pd.Series({"stress_score": float("nan"), "agreement_count": 3})) == 0


def test_decision_clock_uses_new_york_dst_and_full_latency_bar() -> None:
    winter_release = datetime(2023, 1, 6, 21, 15, tzinfo=timezone.utc)
    decision, entry, exit_time = support._decision_clock(winter_release)
    assert decision == datetime(2023, 1, 6, 22, 0, tzinfo=timezone.utc)
    assert entry == datetime(2023, 1, 6, 22, 5, tzinfo=timezone.utc)
    assert exit_time - entry == timedelta(hours=72)

    summer_release = datetime(2023, 7, 7, 20, 15, tzinfo=timezone.utc)
    decision, entry, _ = support._decision_clock(summer_release)
    assert decision == datetime(2023, 7, 7, 21, 0, tzinfo=timezone.utc)
    assert entry == datetime(2023, 7, 7, 21, 5, tzinfo=timezone.utc)

    for transition_release in (
        datetime(2023, 3, 10, 21, 15, tzinfo=timezone.utc),
        datetime(2023, 11, 3, 20, 15, tzinfo=timezone.utc),
    ):
        _, transition_entry, transition_exit = support._decision_clock(
            transition_release
        )
        assert transition_exit - transition_entry == timedelta(hours=72)


def test_repo_state_uses_latest_available_exact_observation_lag_and_freshness() -> None:
    start = datetime(2023, 1, 2, 20, 0, tzinfo=timezone.utc)
    rows = [
        sofr_clock.SourceRow(
            effective_date=date(2023, 1, 2) + timedelta(days=index),
            available_at=start + timedelta(days=index),
            rate_bp=100 + index,
        )
        for index in range(8)
    ]
    availability = [row.available_at for row in rows]
    decision = rows[7].available_at + timedelta(hours=2)
    assert support._repo_state(rows, availability, decision) == (1, 5, 7)
    assert support._repo_state(rows, availability, decision, stale_observations=1) == (1, 5, 6)
    assert support._repo_state(
        rows,
        availability,
        rows[7].available_at + timedelta(hours=37),
    ) is None


def test_scheduler_reserves_overlap_and_allows_entry_at_prior_exit() -> None:
    first = datetime(2023, 1, 1, tzinfo=timezone.utc)
    events = [
        _event("primary", first),
        _event("primary", first + timedelta(hours=24)),
        _event("primary", first + timedelta(hours=72)),
    ]
    accepted = support._schedule(events)
    assert [event.entry_time for event in accepted] == [
        first.isoformat(),
        (first + timedelta(hours=72)).isoformat(),
    ]


def test_tolerant_jaccard_uses_maximum_cardinality_ordered_matching() -> None:
    anchor = datetime(2023, 1, 1, tzinfo=timezone.utc)
    left = [anchor + timedelta(hours=10), anchor + timedelta(hours=11)]
    right = [anchor + timedelta(hours=9), anchor + timedelta(hours=10, minutes=30)]
    assert support._tolerant_jaccard(left, right, timedelta(hours=1)) == {
        "matches": 2,
        "left": 2,
        "right": 2,
        "jaccard": 1.0,
    }


def test_atomic_support_run_rejects_without_market_or_funding_access(tmp_path: Path) -> None:
    output = tmp_path / "support.json"
    clocks = tmp_path / "clocks.csv.gz"
    artifact = support.evaluate_support(output, clocks)

    assert artifact["decision"] == "REJECT_SOURCE_SUPPORT_NO_REPAIR"
    assert artifact["support_passed"] is False
    assert artifact["evaluator_authorized"] is False
    assert artifact["outcome_boundary"] == {
        "market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "return_rows_loaded": 0,
        "market_values_read": 0,
        "funding_values_read": 0,
        "return_or_pnl_fields": 0,
        "post_2023_source_rows_loaded": 0,
        "outcomes_opened": False,
    }
    train = artifact["primary"]["train_2020_2022"]
    assert train["events"] == 37
    assert train["long"] == 11
    assert train["short"] == 26
    assert artifact["primary"]["2021"]["events"] == 3
    assert train["repo_abs_le_3bp"] == 26
    diagnostics = artifact["control_clock_diagnostics"]
    assert diagnostics["direction_flip"]["exact_entry_overlap"]["jaccard"] == 1.0
    assert diagnostics["direction_flip"][
        "signed_5m_occupied_exposure_correlation"
    ] == pytest.approx(
        -1.0
    )
    assert output.is_file() and clocks.is_file()
