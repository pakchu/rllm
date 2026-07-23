from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import pytest

from training import build_ofr_repo_venue_fragmentation_consensus_support as support
from training import preregister_ofr_repo_venue_fragmentation_consensus as prereg


UTC = timezone.utc


def source_rows(
    day: date,
    *,
    availability: datetime | None = None,
    gcf_ag: int = 40,
    gcf_t: int = 60,
    tri_ag: int = 30,
    tri_t: int = 70,
) -> dict[str, support.SourceRow]:
    available = availability or support._expected_availability(day)
    values = {
        "REPO-DVP_AR_TOT-P": Fraction(5),
        "REPO-GCF_AR_TOT-P": Fraction(7),
        "REPO-TRIV1_AR_TOT-P": Fraction(6),
        "REPO-DVP_TV_TOT-P": Fraction(100),
        "REPO-GCF_TV_TOT-P": Fraction(80),
        "REPO-TRIV1_TV_TOT-P": Fraction(120),
        "REPO-GCF_AR_AG-P": Fraction(8),
        "REPO-GCF_AR_T-P": Fraction(6),
        "REPO-TRIV1_AR_AG-P": Fraction(5),
        "REPO-TRIV1_AR_T-P": Fraction(6),
        "REPO-GCF_TV_AG-P": Fraction(gcf_ag),
        "REPO-GCF_TV_T-P": Fraction(gcf_t),
        "REPO-TRIV1_TV_AG-P": Fraction(tri_ag),
        "REPO-TRIV1_TV_T-P": Fraction(tri_t),
    }
    return {
        mnemonic: support.SourceRow(mnemonic, day, available, value)
        for mnemonic, value in values.items()
    }


def test_features_are_exact_and_equal_availability_batches_once() -> None:
    floor = support.FEED_FLOOR
    first = date(2019, 1, 2)
    second = date(2019, 1, 3)
    features, audit = support.build_features(
        {
            first: source_rows(first, availability=floor),
            second: source_rows(second, availability=floor),
        }
    )
    assert len(features) == 2
    assert features[0].decision_allowed is False
    assert features[1].decision_allowed is True
    assert audit.equal_availability_rows_suppressed == 1
    row = features[1]
    assert row.components["rate_dispersion"] == 2
    assert row.components["venue_hhi"] == Fraction(77, 225)
    assert row.components["collateral_rate_disagreement"] == Fraction(3, 2)
    assert row.components["collateral_mix_disagreement"] == Fraction(1, 10)
    assert row.dominant_rate_venue == "GCF"
    assert row.dominant_volume_venue == "TRIV1"
    assert row.dominant_collateral_spread_venue == "GCF"


def test_missing_and_immaterial_dates_break_epochs() -> None:
    d1, d2, d3 = date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6)
    missing = source_rows(d2)
    missing.pop("REPO-GCF_AR_TOT-P")
    features, audit = support.build_features(
        {
            d1: source_rows(d1),
            d2: missing,
            d3: source_rows(d3, gcf_ag=1, gcf_t=99),
        }
    )
    assert len(features) == 1
    assert audit.invalid_missing_or_null_dates == 1
    assert audit.invalid_materiality_dates == 1


def test_midrank_and_state_controls_are_exact() -> None:
    prior = [Fraction(value) for value in range(252)]
    assert support.midrank_unit(Fraction(126), prior) == Fraction(1, 252)
    units = {
        "rate_dispersion": Fraction(3, 4),
        "venue_hhi": Fraction(2, 3),
        "collateral_rate_disagreement": Fraction(1, 2),
        "collateral_mix_disagreement": Fraction(-1, 4),
    }
    state, score = support._primary_state(units)
    assert state == 0
    assert score == Fraction(5, 12)
    state, _ = support._state_for_control("rate_family_only", units)
    assert state == 1
    state, _ = support._state_for_control("volume_family_only", units)
    assert state == 0
    state, _ = support._state_for_control(
        "leave_one_collateral_mix_disagreement", units
    )
    assert state == 1


def rank_row(day: date, epoch: int, units: dict[str, Fraction]) -> support.RankRow:
    feature = support.FeatureRow(
        observation_date=day,
        available_at=datetime.combine(day, datetime.min.time(), UTC),
        epoch=epoch,
        decision_allowed=True,
        components={name: Fraction() for name in prereg.COMPONENTS},
        dominant_rate_venue="DVP",
        dominant_volume_venue="GCF",
        dominant_collateral_spread_venue="TRIV1",
    )
    return support.RankRow(feature, units)


def test_transition_continuity_and_nonoverlap() -> None:
    low = {name: Fraction(-1) for name in prereg.COMPONENTS}
    high = {name: Fraction(1) for name in prereg.COMPONENTS}
    rows = [
        support.StatePoint("primary", rank_row(date(2021, 1, 4), 0, low), -1, -1),
        support.StatePoint("primary", rank_row(date(2021, 1, 5), 0, high), 1, 1),
        support.StatePoint("primary", rank_row(date(2021, 1, 6), 0, low), -1, -1),
        support.StatePoint("primary", rank_row(date(2021, 1, 7), 1, high), 1, 1),
        support.StatePoint("primary", rank_row(date(2021, 1, 8), 1, low), -1, -1),
    ]
    candidates = support.candidates_from_states(rows)
    assert [row.rank.feature.observation_date for row in candidates] == [
        date(2021, 1, 5),
        date(2021, 1, 6),
        date(2021, 1, 8),
    ]
    scheduled = support.schedule("primary", candidates)
    assert len(scheduled) == 2
    assert all(
        current.entry_time >= previous.exit_time
        for previous, current in zip(scheduled, scheduled[1:])
    )


def test_suppressed_batch_state_updates_prior_without_emitting() -> None:
    low = {name: Fraction(-1) for name in prereg.COMPONENTS}
    high = {name: Fraction(1) for name in prereg.COMPONENTS}
    first = rank_row(date(2019, 1, 2), 0, low)
    suppressed = rank_row(date(2019, 1, 3), 0, high)
    suppressed = support.RankRow(
        replace(suppressed.feature, decision_allowed=False), high
    )
    decision = rank_row(date(2019, 1, 4), 0, low)
    points = [
        support.StatePoint("primary", first, -1, -1),
        support.StatePoint("primary", suppressed, 1, 1),
        support.StatePoint("primary", decision, -1, -1),
    ]
    candidates = support.candidates_from_states(points)
    assert [row.rank.feature.observation_date for row in candidates] == [
        date(2019, 1, 4)
    ]


def test_stale_control_carries_the_vector_it_used() -> None:
    rows = []
    for offset in range(6):
        units = {
            name: Fraction(offset, 5) for name in prereg.COMPONENTS
        }
        rows.append(rank_row(date(2021, 1, 4) + timedelta(days=offset), 0, units))
    states = support.build_state_points(rows)
    assert states["one_complete_day_stale"][-1].rank.units == rows[-2].units
    assert states["five_complete_day_stale"][-1].rank.units == rows[0].units


def test_one_to_one_novelty_matching_is_not_many_to_one() -> None:
    origin = datetime(2021, 1, 1, tzinfo=UTC)
    left = [origin, origin + timedelta(hours=1)]
    right = [origin + timedelta(minutes=30)]
    assert support.one_to_one_matches(left, right, timedelta(hours=24)) == 1


def test_source_failure_short_circuit_contract() -> None:
    policy = prereg.policy_payload()
    checks, summaries = support.source_support([], policy)
    assert not all(checks.values())
    assert summaries["train"]["events"] == 0


def test_unknown_or_bad_control_fails_closed() -> None:
    units = {name: Fraction() for name in prereg.COMPONENTS}
    with pytest.raises(RuntimeError, match="unknown RVFC source control"):
        support._state_for_control("repair", units)


def test_immutable_output_write_verifies_exact_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(support, "REPOSITORY_ROOT", tmp_path)
    path = tmp_path / "artifact.bin"
    assert support._write_or_verify(path, b"one") == "created"
    assert support._write_or_verify(path, b"one") == "verified_existing"
    with pytest.raises(RuntimeError, match="artifact differs"):
        support._write_or_verify(path, b"two")
