from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import pytest

from training import build_ofr_repo_mix_shock_resolution_race_support as support
from training import preregister_ofr_repo_mix_shock_resolution_race as prereg


UTC = timezone.utc


def source_rows(day: date, *, available_at: datetime | None = None) -> dict:
    available = available_at or support._expected_availability(day)
    values = {
        "REPO-GCF_AR_AG-P": "5",
        "REPO-GCF_AR_T-P": "3",
        "REPO-TRIV1_AR_AG-P": "4",
        "REPO-TRIV1_AR_T-P": "3",
        "REPO-GCF_TV_AG-P": "60",
        "REPO-GCF_TV_T-P": "40",
        "REPO-TRIV1_TV_AG-P": "50",
        "REPO-TRIV1_TV_T-P": "50",
    }
    return {
        mnemonic: support.SourceRow(
            mnemonic=mnemonic,
            observation_date=day,
            available_at=available,
            value=Fraction(value),
        )
        for mnemonic, value in values.items()
    }


def feature_row(index: int, mix: Fraction, rate: Fraction) -> support.FeatureRow:
    day = date(2019, 1, 1) + timedelta(days=index)
    return support.FeatureRow(
        observation_date=day,
        available_at=datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=index),
        epoch=0,
        decision_allowed=True,
        components={"mix_disagreement": mix, "rate_disagreement": rate},
        dominant_collateral_spread_venue="GCF",
    )


def state_row(
    index: int,
    mix_state: int,
    rate_state: int,
    *,
    epoch: int = 0,
) -> support.StateRow:
    values = {-1: Fraction(-3, 4), 0: Fraction(), 1: Fraction(3, 4)}
    feature = feature_row(index, Fraction(index + 1, 100), Fraction(index + 2, 100))
    day = date(2021, 1, 1) + timedelta(days=index)
    feature = support.replace(
        feature,
        observation_date=day,
        available_at=support._expected_availability(day),
        epoch=epoch,
    )
    rank = support.RankRow(
        feature=feature,
        units={
            "mix_disagreement": values[mix_state],
            "rate_disagreement": values[rate_state],
        },
    )
    return support.StateRow(
        rank=rank,
        epoch=epoch,
        mix_state=mix_state,
        rate_state=rate_state,
    )


def candidate(
    index: int,
    *,
    side: int = 1,
    terminal_type: str = "QUANTITY_ABSORPTION",
) -> support.RaceCandidate:
    terminal = date(2021, 1, 1) + timedelta(days=index)
    signal = datetime.combine(terminal, datetime.min.time(), UTC)
    return support.RaceCandidate(
        precursor_observation_date=terminal - timedelta(days=1),
        terminal_observation_date=terminal,
        signal_time=signal,
        side=side,
        precursor_polarity=side,
        precursor_lag_state=0,
        terminal_type=terminal_type,
        terminal_age_dates=1,
        units={"mix_disagreement": Fraction(), "rate_disagreement": Fraction()},
        dominant_collateral_spread_venue="GCF",
    )


def test_features_are_exact_material_and_equal_availability_is_batched() -> None:
    first = date(2019, 1, 1)
    second = date(2019, 1, 2)
    by_date = {first: source_rows(first), second: source_rows(second)}
    rows, audit = support.build_features(by_date)
    assert rows[0].components["mix_disagreement"] == Fraction(1, 10)
    assert rows[0].components["rate_disagreement"] == Fraction(3, 2)
    assert rows[0].dominant_collateral_spread_venue == "GCF"
    assert rows[0].available_at == support.FEED_FLOOR
    assert rows[0].decision_allowed is False
    assert rows[1].decision_allowed is True
    assert audit.equal_availability_rows_suppressed == 1


def test_missing_or_immaterial_date_breaks_feature_epoch() -> None:
    first = date(2021, 1, 1)
    missing = date(2021, 1, 2)
    immaterial = date(2021, 1, 3)
    last = date(2021, 1, 4)
    rows = source_rows(immaterial)
    rows["REPO-GCF_TV_AG-P"] = support.replace(
        rows["REPO-GCF_TV_AG-P"], value=Fraction(1)
    )
    rows["REPO-GCF_TV_T-P"] = support.replace(
        rows["REPO-GCF_TV_T-P"], value=Fraction(99)
    )
    features, audit = support.build_features(
        {
            first: source_rows(first),
            missing: {},
            immaterial: rows,
            last: source_rows(last),
        }
    )
    assert [row.epoch for row in features] == [0, 2]
    assert audit.invalid_missing_or_null_dates == 1
    assert audit.invalid_materiality_dates == 1


def test_disclosure_marker_invalidates_required_source_date() -> None:
    day = date(2021, 1, 1)
    rows = source_rows(day)
    rows["REPO-GCF_AR_AG-P"] = support.replace(
        rows["REPO-GCF_AR_AG-P"], disclosure_edit="suppressed"
    )
    features, audit = support.build_features({day: rows})
    assert features == []
    assert audit.invalid_missing_or_null_dates == 1


def test_midrank_is_strict_prior_and_current_is_excluded() -> None:
    features = [
        feature_row(index, Fraction(index), Fraction(index)) for index in range(253)
    ]
    rows = support.build_rank_rows(features)
    assert len(rows) == 1
    assert rows[0].feature.observation_date == features[-1].observation_date
    assert rows[0].units == {
        "mix_disagreement": Fraction(1),
        "rate_disagreement": Fraction(1),
    }
    assert support.midrank_unit(Fraction(0), [Fraction(0)] * 252) == Fraction(0)


def test_equal_availability_batch_uses_only_prebatch_rank_history() -> None:
    features = [
        feature_row(index, Fraction(index), Fraction(index)) for index in range(252)
    ]
    shared = datetime(2022, 1, 1, tzinfo=UTC)
    features.extend(
        [
            support.replace(
                feature_row(252, Fraction(1_000), Fraction(1_000)),
                available_at=shared,
                decision_allowed=False,
            ),
            support.replace(
                feature_row(253, Fraction(1_000), Fraction(1_000)),
                available_at=shared,
                decision_allowed=True,
            ),
        ]
    )
    rows = support.build_rank_rows(features)
    assert len(rows) == 2
    assert [row.units["mix_disagreement"] for row in rows] == [
        Fraction(1),
        Fraction(1),
    ]


@pytest.mark.parametrize(
    ("polarity", "terminal", "expected_type", "expected_side"),
    [
        (1, "confirmation", "PRICE_CONFIRMATION", -1),
        (1, "absorption", "QUANTITY_ABSORPTION", 1),
        (-1, "confirmation", "PRICE_CONFIRMATION", 1),
        (-1, "absorption", "QUANTITY_ABSORPTION", -1),
    ],
)
def test_primary_race_freezes_all_four_side_paths(
    polarity: int, terminal: str, expected_type: str, expected_side: int
) -> None:
    rows = [state_row(0, 0, 0), state_row(1, polarity, 0)]
    if terminal == "confirmation":
        rows.append(state_row(2, polarity, polarity))
    else:
        rows.append(state_row(2, 0, 0))
    events, audit = support.derive_race_candidates(rows)
    assert len(events) == 1
    assert events[0].terminal_type == expected_type
    assert events[0].side == expected_side
    assert events[0].terminal_age_dates == 1
    assert audit["armed"] == 1


def test_already_priced_precursor_is_discarded() -> None:
    events, audit = support.derive_race_candidates(
        [state_row(0, 0, 0), state_row(1, 1, 1), state_row(2, 0, 1)]
    )
    assert events == []
    assert audit["already_priced"] == 1
    assert audit["armed"] == 0


def test_same_date_confirmation_and_exit_is_ambiguous_and_cannot_rearm() -> None:
    events, audit = support.derive_race_candidates(
        [
            state_row(0, 0, 0),
            state_row(1, 1, 0),
            state_row(2, -1, 1),
            state_row(3, -1, 0),
        ]
    )
    assert events == []
    assert audit["ambiguous_same_date"] == 1
    assert audit["armed"] == 1


def test_window_accepts_offset_twenty_but_not_twenty_one() -> None:
    through_twenty = [state_row(0, 0, 0), state_row(1, 1, 0)]
    through_twenty.extend(state_row(index, 1, 0) for index in range(2, 21))
    through_twenty.append(state_row(21, 1, 1))
    events, audit = support.derive_race_candidates(through_twenty)
    assert len(events) == 1
    assert events[0].terminal_age_dates == 20
    assert audit["timeouts"] == 0

    after_timeout = [state_row(0, 0, 0), state_row(1, 1, 0)]
    after_timeout.extend(state_row(index, 1, 0) for index in range(2, 22))
    after_timeout.append(state_row(22, 1, 1))
    events, audit = support.derive_race_candidates(after_timeout)
    assert events == []
    assert audit["timeouts"] == 1


def test_continuity_break_cancels_race_and_new_epoch_row_cannot_terminate() -> None:
    events, audit = support.derive_race_candidates(
        [
            state_row(0, 0, 0, epoch=0),
            state_row(1, 1, 0, epoch=0),
            state_row(2, 1, 1, epoch=1),
            state_row(3, 0, 1, epoch=1),
        ]
    )
    assert events == []
    assert audit["continuity_cancellations"] == 1


def test_terminal_date_opposite_transition_is_not_rearmed() -> None:
    events, audit = support.derive_race_candidates(
        [
            state_row(0, 0, 0),
            state_row(1, 1, 0),
            state_row(2, -1, 0),
            state_row(3, -1, -1),
        ]
    )
    assert len(events) == 1
    assert events[0].terminal_type == "QUANTITY_ABSORPTION"
    assert audit["armed"] == 1


def test_terminal_type_controls_schedule_independently() -> None:
    states = [
        state_row(0, 0, 0),
        state_row(1, 1, 0),
        state_row(2, 1, 1),
        state_row(3, -1, 1),
        state_row(4, 0, 1),
    ]
    clocks, _ = support.build_clocks(states)
    assert len(clocks["primary"]) == 1
    assert len(clocks["price_confirmation_only"]) == 1
    assert len(clocks["quantity_absorption_only"]) == 1
    assert clocks["price_confirmation_only"][0].terminal_type == "PRICE_CONFIRMATION"
    assert clocks["quantity_absorption_only"][0].terminal_type == (
        "QUANTITY_ABSORPTION"
    )


def test_schedule_waits_one_bar_and_enforces_global_nonoverlap() -> None:
    first = candidate(0)
    second = candidate(1)
    rows = support.schedule("primary", [first, second])
    assert len(rows) == 1
    assert rows[0].entry_time == first.signal_time + timedelta(minutes=5)
    assert rows[0].exit_time - rows[0].entry_time == timedelta(hours=72)
    assert rows[0].split == "train"


def test_primary_validation_and_support_checks_are_falsifiable() -> None:
    valid = support.schedule("primary", [candidate(0)])[0]
    assert support._primary_race_valid([valid]) is True
    assert support._primary_race_valid(
        [support.replace(valid, terminal_age_dates=21)]
    ) is False
    assert support._primary_race_valid(
        [support.replace(valid, precursor_observation_date=valid.terminal_observation_date)]
    ) is False
    assert support._primary_race_valid(
        [support.replace(valid, precursor_lag_state=valid.precursor_polarity)]
    ) is False
    assert support._primary_race_valid(
        [support.replace(valid, units={"mix_disagreement": 0.0, "rate_disagreement": 0.0})]
    ) is False

    ambiguous = support.replace(valid, terminal_type="AMBIGUOUS_SAME_DATE")
    checks, _, _ = support.source_support(
        [ambiguous],
        prereg.policy_payload(),
        post_2023_source_rows=1,
    )
    assert checks["accepted_ambiguity_count"] is False
    assert checks["post_2023_source_rows"] is False
    assert checks["primary_race_valid"] is False


def test_frozen_comparator_cohort_fails_closed_on_post_2023_interval() -> None:
    assert tuple(prereg.COMPARATOR_SPECS[:-1]) == tuple(
        support.rvfc_support.prereg.COMPARATOR_SPECS
    )
    assert prereg.COMPARATOR_SPECS[-1]["name"] == (
        "ofr_repo_venue_fragmentation_consensus_primary"
    )
    with pytest.raises(support.ComparatorValidationError) as caught:
        support.load_comparator_groups()
    assert caught.value.rows_read > 0
    assert "post-2023 comparator clock" in str(caught.value)


def test_source_failure_short_circuits_comparators_and_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_clocks = {name: [] for name in support.CONTROL_NAMES}
    monkeypatch.setattr(
        support,
        "_load_registration",
        lambda: {"policy": prereg.policy_payload()},
    )
    monkeypatch.setattr(support, "load_source", lambda: ({}, 0, 0))
    monkeypatch.setattr(
        support,
        "build_features",
        lambda *_args, **_kwargs: (
            [],
            support.SourceAudit(0, 0, 0, 0, 0, 0, 0),
        ),
    )
    monkeypatch.setattr(support, "build_rank_rows", lambda _rows: [])
    monkeypatch.setattr(support, "build_state_rows", lambda _rows: [])
    monkeypatch.setattr(
        support,
        "build_clocks",
        lambda _rows: (empty_clocks, {"primary": {}}),
    )
    called = False

    def forbidden_loader() -> tuple[dict, int]:
        nonlocal called
        called = True
        raise AssertionError("comparators must remain closed")

    report = support.build_report(
        write_clock=False,
        load_comparators=forbidden_loader,
    )
    assert called is False
    assert report["source_support_passed"] is False
    assert report["novelty"]["evaluated"] is False
    assert report["outcome_boundary"]["comparator_rows_read"] == 0
    assert report["outcome_boundary"]["btc_market_rows_read"] == 0
    assert report["outcome_boundary"]["funding_rows_read"] == 0
    assert report["outcome_boundary"]["future_return_rows_read"] == 0
    assert report["outcome_boundary"]["pnl_cagr_mdd_opened"] is False


def test_comparator_validation_failure_becomes_deterministic_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_clocks = {name: [] for name in support.CONTROL_NAMES}
    monkeypatch.setattr(
        support,
        "_load_registration",
        lambda: {"policy": prereg.policy_payload()},
    )
    monkeypatch.setattr(support, "load_source", lambda: ({}, 0, 0))
    monkeypatch.setattr(
        support,
        "build_features",
        lambda *_args, **_kwargs: (
            [],
            support.SourceAudit(0, 0, 0, 0, 0, 0, 0),
        ),
    )
    monkeypatch.setattr(support, "build_rank_rows", lambda _rows: [])
    monkeypatch.setattr(support, "build_state_rows", lambda _rows: [])
    monkeypatch.setattr(
        support,
        "build_clocks",
        lambda _rows: (empty_clocks, {"primary": {}}),
    )
    monkeypatch.setattr(
        support,
        "source_support",
        lambda *_args, **_kwargs: (
            {"synthetic_pass": True},
            {},
            {},
        ),
    )

    def invalid_loader() -> tuple[dict, int]:
        raise support.ComparatorValidationError(
            "post-2023 comparator clock: fixture", rows_read=17
        )

    report = support.build_report(
        write_clock=False,
        load_comparators=invalid_loader,
    )
    assert report["source_support_passed"] is True
    assert report["novelty"]["passed"] is False
    assert report["novelty"]["reason"] == "comparator validation failed closed"
    assert report["outcome_boundary"]["comparator_rows_read"] == 17
    assert report["advance_to_evaluator_freeze"] is False


def test_write_or_verify_is_immutable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(support, "REPOSITORY_ROOT", tmp_path)
    path = tmp_path / "artifact.bin"
    assert support._write_or_verify(path, b"first") == "created"
    assert support._write_or_verify(path, b"first") == "verified_existing"
    with pytest.raises(RuntimeError, match="differs"):
        support._write_or_verify(path, b"second")
    assert path.read_bytes() == b"first"
