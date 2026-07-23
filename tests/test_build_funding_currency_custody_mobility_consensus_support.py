from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import gzip
import hashlib
from pathlib import Path
import subprocess
from typing import Any

import pytest

from training import (
    build_funding_currency_custody_mobility_consensus_support as support,
)


UTC = timezone.utc


def assert_no_float(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            assert_no_float(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_no_float(child)
    else:
        assert not isinstance(value, float)


def dt(value: str) -> datetime:
    return support.parse_time(value)


def bitfinex_row(
    symbol: str,
    hour: datetime,
    *,
    timestamp_ms: int,
    available_at: datetime | None = None,
    total: int = 100,
    used: int = 50,
    tenor: int = 5,
) -> support.BitfinexRow:
    observation = hour + timedelta(minutes=5)
    return support.BitfinexRow(
        symbol=symbol,
        observation_time=observation,
        hour=hour,
        available_at=available_at or hour + timedelta(minutes=15),
        timestamp_ms=timestamp_ms,
        average_period_days=Fraction(tenor),
        total=Fraction(total),
        used=Fraction(used),
    )


def ranked_feature(
    hour: datetime,
    available_at: datetime,
    units: tuple[Fraction, Fraction, Fraction],
    *,
    identity: str,
) -> support.RankedFeature:
    usd = bitfinex_row("fUSD", hour, timestamp_ms=1, available_at=available_at)
    btc = bitfinex_row("fBTC", hour, timestamp_ms=2, available_at=available_at)
    pair = support.PairAnchor(
        hour=hour,
        available_at=available_at,
        valid=True,
        reason="valid",
        usd=usd,
        btc=btc,
        identity=identity,
    )
    feature = support.FeatureAnchor(
        pair=pair,
        available_at=available_at,
        valid=True,
        reason="valid",
        rotations=units,
    )
    votes = tuple(support.vote(unit) for unit in units)
    return support.RankedFeature(
        feature=feature,
        units=units,
        votes=votes,
        score=sum(units, Fraction()) / 3,
        consensus_state=support.consensus_state(units, votes),
        majority_state=(
            1 if votes.count(1) >= 2 else -1 if votes.count(-1) >= 2 else 0
        ),
    )


def wbtc_state(anchor: datetime, *, active: bool, suffix: str = "0") -> support.WBTCState:
    actors = (
        "0x" + "1" * 40,
        "0x" + "2" * 40,
    )
    return support.WBTCState(
        anchor=anchor,
        valid=True,
        gross_raw=10,
        gross_unit=Fraction(1, 2),
        actors=actors,
        top_share=Fraction(1, 2),
        active=active,
        window_identity=suffix.zfill(64),
    )


def test_exact_rank_vote_and_score_boundaries() -> None:
    positive_prior = [Fraction(index) for index in range(450)] + [
        Fraction(10_000 + index) for index in range(270)
    ]
    assert support.midrank_unit(positive_prior, Fraction(500), 720) == Fraction(1, 4)
    assert support.vote(Fraction(1, 4)) == 1

    tied_prior = [Fraction(index) for index in range(449)] + [Fraction(500)] + [
        Fraction(10_000 + index) for index in range(270)
    ]
    assert support.midrank_unit(tied_prior, Fraction(500), 720) == Fraction(179, 720)
    assert support.vote(Fraction(179, 720)) == 0

    negative_prior = [Fraction(index) for index in range(270)] + [
        Fraction(10_000 + index) for index in range(450)
    ]
    assert support.midrank_unit(negative_prior, Fraction(500), 720) == Fraction(-1, 4)
    assert support.vote(Fraction(-1, 4)) == -1

    accepted = (Fraction(1, 4), Fraction(1, 4), Fraction(1, 2))
    assert support.consensus_state(accepted, (1, 1, 1)) == 1
    rejected = (Fraction(1, 4), Fraction(1, 4), Fraction(179, 720))
    assert support.consensus_state(rejected, (1, 1, 0)) == 0


def test_pair_missing_partial_duplicate_and_exact_lag_fail_closed() -> None:
    start = dt("2021-01-01T00:00:00Z")
    rows = [
        bitfinex_row("fUSD", start, timestamp_ms=1),
        bitfinex_row("fBTC", start, timestamp_ms=2),
        bitfinex_row(
            "fUSD",
            start + timedelta(hours=1),
            timestamp_ms=3,
            available_at=start + timedelta(hours=2, minutes=20),
        ),
        bitfinex_row("fUSD", start + timedelta(hours=2), timestamp_ms=4),
        bitfinex_row("fUSD", start + timedelta(hours=2), timestamp_ms=5),
        bitfinex_row("fBTC", start + timedelta(hours=2), timestamp_ms=6),
    ]
    pairs = support.build_pair_anchors(rows, start=start, end=start + timedelta(hours=4))
    assert pairs[0].valid is True
    assert pairs[1].reason == "partial_pair"
    assert pairs[1].available_at == start + timedelta(hours=2, minutes=20)
    assert pairs[2].reason == "duplicate_pair"
    assert pairs[3].reason == "missing_pair"
    assert pairs[3].available_at == start + timedelta(hours=4, minutes=15)
    features = support.build_feature_anchors(pairs)
    assert all(feature.valid is False for feature in features)
    assert features[0].reason == "invalid_exact_24h_lag"


def test_equal_availability_batch_excludes_current_rows_and_invalid_resets() -> None:
    base = dt("2021-01-01T00:00:00Z")

    def feature(
        hour_offset: int,
        available_offset: int,
        value: int,
        *,
        valid: bool = True,
    ) -> support.FeatureAnchor:
        hour = base + timedelta(hours=hour_offset)
        pair = support.PairAnchor(
            hour,
            base + timedelta(hours=available_offset),
            valid,
            "valid" if valid else "missing_pair",
            identity=f"pair-{hour_offset}" if valid else "",
        )
        return support.FeatureAnchor(
            pair,
            pair.available_at,
            valid,
            pair.reason,
            (Fraction(value), Fraction(value), Fraction(value)) if valid else None,
        )

    features = [
        feature(0, 1, 0),
        feature(1, 2, 10),
        feature(2, 3, 5),
        feature(3, 3, 100),
        feature(4, 4, 7, valid=False),
        feature(5, 4, 6),
        feature(6, 5, 8),
    ]
    batches = support.rank_causal_batches(features, history_size=2)
    shared = next(batch for batch in batches if batch.available_at == base + timedelta(hours=3))
    assert [row.units for row in shared.rows] == [
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(1), Fraction(1)),
    ]
    poisoned = next(batch for batch in batches if batch.available_at == base + timedelta(hours=4))
    assert poisoned.invalid is True
    assert poisoned.eligible is None
    later = next(batch for batch in batches if batch.available_at == base + timedelta(hours=5))
    assert later.eligible is not None
    assert later.eligible.units is not None


def test_state_machine_baseline_transition_and_no_queue_behavior() -> None:
    base = dt("2022-01-01T00:00:00Z")
    states = [
        (Fraction(-1, 2), Fraction(-1, 2), Fraction(-1, 2)),
        (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)),
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)),
    ]
    batches = [
        support.CausalBatch(
            available_at=base + timedelta(days=index),
            rows=(row,),
            invalid=False,
            eligible=row,
        )
        for index, units in enumerate(states)
        for row in (
            ranked_feature(
                base + timedelta(days=index),
                base + timedelta(days=index, minutes=15),
                units,
                identity=f"pair-{index}",
            ),
        )
    ]
    transitions = support.derive_transitions(batches)
    assert [transition.side for transition in transitions] == [1, 1]
    assert [transition.clock.feature.pair.hour.day for transition in transitions] == [
        2,
        5,
    ]

    wbtc = {
        support.floor_day(transitions[0].clock.feature.available_at): wbtc_state(
            support.floor_day(transitions[0].clock.feature.available_at), active=False
        ),
        support.floor_day(transitions[1].clock.feature.available_at): wbtc_state(
            support.floor_day(transitions[1].clock.feature.available_at), active=True
        ),
    }
    opportunities = support.materialize_opportunities(transitions, wbtc)
    assert [row.sponsored for row in opportunities] == [False, True]


def test_component_and_majority_controls_use_their_own_frozen_state() -> None:
    base = dt("2022-01-01T00:00:00Z")
    baseline = ranked_feature(base, base, (Fraction(0),) * 3, identity="baseline")
    majority_only = ranked_feature(
        base + timedelta(hours=1),
        base + timedelta(hours=1),
        (Fraction(1, 4), Fraction(1, 4), Fraction(-1, 2)),
        identity="majority",
    )
    utilization_only = ranked_feature(
        base + timedelta(hours=2),
        base + timedelta(hours=2),
        (Fraction(1, 2), Fraction(0), Fraction(0)),
        identity="utilization",
    )
    batches = [
        support.CausalBatch(row.feature.available_at, (row,), False, row)
        for row in (baseline, majority_only, utilization_only)
    ]
    assert support.derive_transitions(batches, rule="consensus") == []
    assert [row.side for row in support.derive_transitions(batches, rule="majority")] == [
        1
    ]
    assert [
        row.side for row in support.derive_transitions(batches, rule="utilization")
    ] == [1]


def test_invalid_batch_resets_and_next_direction_is_baseline_only() -> None:
    base = dt("2022-01-01T00:00:00Z")
    negative = ranked_feature(base, base, (Fraction(-1, 2),) * 3, identity="n")
    positive = ranked_feature(
        base + timedelta(hours=2),
        base + timedelta(hours=2),
        (Fraction(1, 2),) * 3,
        identity="p1",
    )
    neutral = ranked_feature(
        base + timedelta(hours=3),
        base + timedelta(hours=3),
        (Fraction(0),) * 3,
        identity="z",
    )
    reentry = ranked_feature(
        base + timedelta(hours=4),
        base + timedelta(hours=4),
        (Fraction(1, 2),) * 3,
        identity="p2",
    )
    batches = [
        support.CausalBatch(base, (negative,), False, negative),
        support.CausalBatch(base + timedelta(hours=1), (), True, None),
        support.CausalBatch(base + timedelta(hours=2), (positive,), False, positive),
        support.CausalBatch(base + timedelta(hours=3), (neutral,), False, neutral),
        support.CausalBatch(base + timedelta(hours=4), (reentry,), False, reentry),
    ]
    transitions = support.derive_transitions(batches)
    assert len(transitions) == 1
    assert transitions[0].clock.feature.pair.identity == "p2"


def test_wbtc_available_at_window_rank_and_top_share_boundaries() -> None:
    anchor = dt("2022-07-20T00:00:00Z")
    coverage_start = anchor - timedelta(days=15)
    actor_a = "0x" + "1" * 40
    actor_b = "0x" + "2" * 40

    def event(when: datetime, amount: int, actor: str, index: int) -> support.WBTCEvent:
        return support.WBTCEvent(
            available_at=when,
            amount_raw=amount,
            actor=actor,
            block_hash="0x" + f"{index + 1:064x}",
            transaction_hash="0x" + f"{index + 101:064x}",
            semantic_log_index=index,
        )

    events = [
        event(anchor - timedelta(days=14), 100, actor_a, 0),
        event(anchor - timedelta(days=14) + timedelta(seconds=1), 1, actor_a, 1),
        event(anchor, 16, actor_a, 2),
        event(anchor, 4, actor_b, 3),
        event(anchor + timedelta(seconds=1), 1000, actor_b, 4),
    ]
    states = support.build_wbtc_states(
        events,
        coverage_start=coverage_start,
        end=anchor + timedelta(days=1),
        history_size=1,
    )
    current = states[anchor]
    assert current.gross_raw == 21
    assert current.gross_unit == -1
    assert current.top_share == Fraction(17, 21)
    assert current.active is False

    boundary_events = [
        event(anchor, 4, actor_a, 10),
        event(anchor, 1, actor_b, 11),
    ]
    boundary = support.build_wbtc_states(
        boundary_events,
        coverage_start=coverage_start,
        end=anchor + timedelta(days=1),
        history_size=1,
    )[anchor]
    assert boundary.top_share == Fraction(4, 5)
    assert boundary.gross_unit == 1
    assert boundary.active is True

    one_actor = support.build_wbtc_states(
        [event(anchor, 5, actor_a, 20)],
        coverage_start=coverage_start,
        end=anchor + timedelta(days=1),
        history_size=1,
    )[anchor]
    assert one_actor.gross_raw == 5
    assert one_actor.top_share is None
    assert one_actor.active is False


def test_wbtc_window_identity_is_physical_order_invariant() -> None:
    anchor = dt("2022-01-01T00:00:00Z")
    events = [
        support.WBTCEvent(
            anchor,
            1,
            "0x" + f"{index + 1:040x}",
            "0x" + f"{index + 1:064x}",
            "0x" + f"{index + 11:064x}",
            index,
        )
        for index in range(2)
    ]
    assert support.wbtc_window_identity(anchor, events) == support.wbtc_window_identity(
        anchor, list(reversed(events))
    )


def test_exact_execution_grid_hold_and_split_containment() -> None:
    assert support.execution_entry(dt("2022-01-01T12:10:00Z")) == dt(
        "2022-01-01T12:15:00Z"
    )
    assert support.execution_entry(dt("2022-01-01T12:10:01Z")) == dt(
        "2022-01-01T12:20:00Z"
    )
    entry = dt("2022-12-31T00:00:00Z")
    assert support.contained_split(entry, entry + support.HOLD) is None
    selection_entry = dt("2023-01-01T00:00:00Z")
    assert support.contained_split(selection_entry, selection_entry + support.HOLD) == (
        "selection"
    )


def test_stale_bitfinex_transition_uses_current_hour_clock() -> None:
    source_hour = dt("2022-01-01T00:00:00Z")
    source = ranked_feature(
        source_hour,
        source_hour + timedelta(minutes=15),
        (Fraction(1, 2),) * 3,
        identity="source-pair",
    )
    current = ranked_feature(
        source_hour + timedelta(hours=24),
        source_hour + timedelta(hours=24, minutes=15),
        (Fraction(-1, 2),) * 3,
        identity="current-pair",
    )
    transition = support.RawTransition(source, source, 1)
    batches = [
        support.CausalBatch(current.feature.available_at, (current,), False, current)
    ]
    shifted = support.shift_bitfinex_transitions_24h([transition], batches)
    assert len(shifted) == 1
    assert shifted[0].directional.feature.pair.identity == "source-pair"
    assert shifted[0].clock.feature.pair.identity == "current-pair"
    assert shifted[0].clock.feature.available_at == current.feature.available_at


def test_independent_scheduler_accepts_touching_and_rejects_overlap() -> None:
    base = dt("2022-01-01T00:00:00Z")
    ranked = ranked_feature(
        base,
        base - timedelta(minutes=5),
        (Fraction(1, 2),) * 3,
        identity="pair-a",
    )
    transition = support.RawTransition(ranked, ranked, 1)
    wbtc = wbtc_state(support.floor_day(base), active=True)

    def opportunity(entry: datetime, pair_id: str) -> support.Opportunity:
        row = replace_ranked_identity(ranked, pair_id)
        split = support.contained_split(entry, entry + support.HOLD)
        return support.Opportunity(
            support.RawTransition(row, row, 1),
            wbtc,
            True,
            entry,
            entry + support.HOLD,
            support.entry_split(entry),
            split,
        )

    rows = [
        opportunity(base, "pair-1"),
        opportunity(base + support.HOLD - timedelta(seconds=1), "pair-2"),
        opportunity(base + support.HOLD, "pair-3"),
    ]
    accepted = support.schedule_opportunities(
        "primary", rows, require_sponsorship=True
    )
    assert [row.transition.clock.feature.pair.identity for row in accepted] == [
        "pair-1",
        "pair-3",
    ]
    assert transition.side == 1


def replace_ranked_identity(
    ranked: support.RankedFeature, identity: str
) -> support.RankedFeature:
    pair = replace_pair_identity(ranked.feature.pair, identity)
    return support.RankedFeature(
        feature=support.FeatureAnchor(
            pair,
            ranked.feature.available_at,
            True,
            "valid",
            ranked.feature.rotations,
        ),
        units=ranked.units,
        votes=ranked.votes,
        score=ranked.score,
        consensus_state=ranked.consensus_state,
        majority_state=ranked.majority_state,
    )


def replace_pair_identity(pair: support.PairAnchor, identity: str) -> support.PairAnchor:
    return support.PairAnchor(
        pair.hour,
        pair.available_at,
        pair.valid,
        pair.reason,
        pair.usd,
        pair.btc,
        identity,
    )


def test_random_side_and_placebo_permutations_are_deterministic_and_multiset_safe() -> None:
    entry = dt("2022-01-01T00:00:00Z")
    assert support.deterministic_random_side(entry) == support.deterministic_random_side(
        entry
    )
    base = dt("2022-01-01T00:00:00Z")
    events = [
        support.WBTCEvent(
            base + timedelta(days=index),
            index + 1,
            "0x" + f"{index + 1:040x}",
            "0x" + f"{index + 1:064x}",
            "0x" + f"{index + 101:064x}",
            index,
        )
        for index in range(5)
    ]
    first = support.permute_wbtc_field(events, field="amount_raw")
    second = support.permute_wbtc_field(events, field="amount_raw")
    assert first == second
    assert sorted(event.amount_raw for event in first) == [1, 2, 3, 4, 5]
    assert [event.identity for event in first] == sorted(event.identity for event in first)
    actor_first = support.permute_wbtc_field(events, field="actor_address")
    assert sorted(event.actor for event in actor_first) == sorted(
        event.actor for event in events
    )


def test_one_bar_delay_drops_row_that_leaves_original_split() -> None:
    entry = support.SEALED_FROM - support.HOLD
    ranked = ranked_feature(
        entry,
        entry - timedelta(minutes=5),
        (Fraction(1, 2),) * 3,
        identity="pair",
    )
    primary = support.CandidateClock(
        control="primary",
        transition=support.RawTransition(ranked, ranked, 1),
        wbtc=wbtc_state(support.floor_day(entry), active=True),
        side=1,
        entry_time=entry,
        exit_time=support.SEALED_FROM,
        split="selection",
        row_identity="a" * 64,
        primary_identity="a" * 64,
    )
    assert support._transformed_primary_control(
        "one_bar_delay",
        [primary],
        side_for=lambda row: row.side,
        delay=support.FIVE_MINUTES,
    ) == []


def test_raw_sponsorship_denominator_precedes_split_containment_and_nonoverlap() -> None:
    base = dt("2022-12-31T00:00:00Z")
    ranked = ranked_feature(
        base,
        base - timedelta(minutes=5),
        (Fraction(1, 2),) * 3,
        identity="pair",
    )
    transition = support.RawTransition(ranked, ranked, 1)
    active = wbtc_state(support.floor_day(base), active=True)
    opportunity = support.Opportunity(
        transition,
        active,
        True,
        base,
        base + support.HOLD,
        "train",
        None,
    )
    stats = support.raw_sponsorship_statistics([opportunity], "train")
    assert stats == {
        "raw_directional_transitions_before_split_and_nonoverlap": 1,
        "wbtc_active": 1,
        "wbtc_active_share": "1",
        "split_crossing_transitions": 1,
    }


def test_split_statistics_keep_exact_month_and_component_denominators() -> None:
    def clock(entry: datetime, index: int) -> support.CandidateClock:
        ranked = ranked_feature(
            entry,
            entry - timedelta(minutes=5),
            (Fraction(1, 2),) * 3,
            identity=f"pair-{index}",
        )
        return support.CandidateClock(
            control="primary",
            transition=support.RawTransition(ranked, ranked, 1),
            wbtc=wbtc_state(support.floor_day(entry), active=True, suffix=str(index)),
            side=1,
            entry_time=entry,
            exit_time=entry + support.HOLD,
            split="train",
            row_identity=f"{index:064x}",
            primary_identity=f"{index:064x}",
        )

    entries: list[datetime] = []
    entries.extend(
        dt("2021-01-01T00:00:00Z") + timedelta(hours=4 * index)
        for index in range(10)
    )
    for month in range(2, 12):
        entries.extend(
            datetime(2021, month, 1, tzinfo=UTC) + timedelta(hours=4 * index)
            for index in range(5)
        )
    entries.extend(
        dt("2021-12-01T00:00:00Z") + timedelta(hours=4 * index)
        for index in range(5)
    )
    rows = [clock(entry, index) for index, entry in enumerate(entries)]
    assert len(rows) == 65
    stats = support.split_statistics(rows, "train")
    assert stats["maximum_month_share"] == "2/13"
    assert stats["component_vote_with_side_share"] == {
        "utilization": "1",
        "draw": "1",
        "tenor": "1",
    }


def test_screening_skips_post_seal_values_before_feature_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(support, "REPOSITORY_ROOT", tmp_path)
    path = Path("source.csv.gz")
    header = ("available_at", "amount_raw")
    with gzip.open(tmp_path / path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerow(("2023-12-31T23:59:59Z", "1"))
        writer.writerow(("2024-01-01T00:00:00Z", "not-a-number"))
    digest = hashlib.sha256((tmp_path / path).read_bytes()).hexdigest()
    records, audit = support._screened_gzip_records(
        path,
        expected_hash=digest,
        expected_header=header,
        allowed_columns=("available_at", "amount_raw"),
        sentinel_column="available_at",
        expected_physical_rows=2,
    )
    assert records == [
        {"available_at": "2023-12-31T23:59:59Z", "amount_raw": "1"}
    ]
    assert audit["post_seal_timestamp_sentinels_scanned"] == 1
    assert audit["post_seal_value_rows_loaded"] == 0


def test_bitfinex_screening_uses_causal_availability_before_numeric_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(support, "REPOSITORY_ROOT", tmp_path)
    path = Path("bitfinex.csv.gz")
    header = ("observation_time", "available_at", "funding_amount")
    with gzip.open(tmp_path / path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerow(
            (
                "2023-12-31T23:05:00Z",
                "2024-01-01T00:00:00Z",
                "not-a-number",
            )
        )
    digest = hashlib.sha256((tmp_path / path).read_bytes()).hexdigest()
    records, audit = support._screened_gzip_records(
        path,
        expected_hash=digest,
        expected_header=header,
        allowed_columns=header,
        sentinel_column="observation_time",
        causal_available_column="available_at",
        expected_physical_rows=1,
    )
    assert records == []
    assert audit["post_seal_timestamp_sentinels_scanned"] == 1
    assert audit["post_seal_value_rows_loaded"] == 0


def test_empty_synthetic_pipeline_is_deterministic_and_opens_no_outcome() -> None:
    source_audit = {
        "bitfinex": {"pre_seal_value_rows_loaded": 0},
        "wbtc": {"pre_seal_value_rows_loaded": 0},
    }
    first, first_clock = support.build_support_payload([], [], source_audit)
    second, second_clock = support.build_support_payload([], [], source_audit)
    assert first == second
    assert first_clock == second_clock
    assert first["source_support_passed"] is False
    assert first["artifact_eligible"] is False
    assert first["verification_mode"] == "uncommitted_or_injected_sources"
    assert first["decision"] == (
        "ineligible_uncommitted_or_injected_source_support_payload"
    )
    assert first["advance_to_novelty_evaluator"] is False
    assert first["clock"]["rows"] == 0
    assert first["outcome_boundary"]["comparator_value_rows_read"] == 0
    assert first["outcome_boundary"]["btc_market_rows_read"] == 0
    assert first["outcome_boundary"][
        "git_protocol_subprocess_calls_before_source_read"
    ] == 0
    assert all(
        report["execution_clock_rows_emitted"] == 0
        for report in first["noncausal_placebo_incidence"].values()
    )
    assert all(
        report["within_year_multiset_preserved"] is True
        for report in first["noncausal_placebo_incidence"].values()
    )
    assert_no_float(first)


def test_passing_injected_build_cannot_authorize_novelty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        support,
        "support_report",
        lambda _primary, _raw: ({"synthetic": "pass"}, {"synthetic": True}),
    )
    payload, _ = support.build_support_payload(
        [],
        [],
        {
            "bitfinex": {"pre_seal_value_rows_loaded": 0},
            "wbtc": {"pre_seal_value_rows_loaded": 0},
        },
    )
    assert payload["source_support_passed"] is True
    assert payload["artifact_eligible"] is False
    assert payload["advance_to_novelty_evaluator"] is False
    assert payload["decision"].startswith("ineligible_")


def test_commit_guard_precedes_preregistration_and_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def reject() -> None:
        calls.append("guard")
        raise RuntimeError("support protocol uncommitted")

    monkeypatch.setattr(support, "_assert_protocol_committed", reject)
    monkeypatch.setattr(
        support,
        "_validate_preregistration_artifact",
        lambda: (_ for _ in ()).throw(AssertionError("prereg opened before guard")),
    )
    monkeypatch.setattr(
        support,
        "_load_sources",
        lambda: (_ for _ in ()).throw(AssertionError("source opened before guard")),
    )
    with pytest.raises(RuntimeError, match="uncommitted"):
        support.write_support()
    assert calls == ["guard"]


def test_only_guarded_write_path_can_authorize_passing_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(support, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(support, "_assert_protocol_committed", lambda: None)
    monkeypatch.setattr(support, "_validate_preregistration_artifact", lambda: {})
    monkeypatch.setattr(support, "sha256_file", lambda _path: support.IMPLEMENTATION_CONTRACT_SHA256)
    monkeypatch.setattr(support, "_load_sources", lambda: ([], [], {}))
    direct = {
        "candidate": support.CANDIDATE,
        "artifact_eligible": False,
        "verification_mode": "uncommitted_or_injected_sources",
        "source_support_passed": True,
        "decision": "ineligible_uncommitted_or_injected_source_support_payload",
        "advance_to_novelty_evaluator": False,
        "outcome_boundary": {
            "git_protocol_subprocess_calls_before_source_read": 0,
            "comparator_value_rows_read": 0,
            "btc_market_rows_read": 0,
            "realized_funding_rows_read": 0,
            "future_return_rows_read": 0,
            "pnl_cagr_mdd_values_opened": 0,
        },
        "manifest_hash": "unsealed",
    }
    monkeypatch.setattr(
        support,
        "build_support_payload",
        lambda *_args, **_kwargs: (dict(direct), b"clock\n"),
    )
    payload, status = support.write_support(
        clock_output="out/clock.csv.gz",
        report_output="out/report.json",
    )
    assert status == "created"
    assert payload["artifact_eligible"] is True
    assert payload["verification_mode"] == (
        "committed_protocol_and_verified_source_hashes"
    )
    assert payload["advance_to_novelty_evaluator"] is True
    assert payload["decision"] == "advance_to_committed_novelty_evaluator"
    assert payload["outcome_boundary"][
        "git_protocol_subprocess_calls_before_source_read"
    ] == 2


def test_commit_guard_freezes_script_test_and_contract_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(support, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(support, "SCRIPT_PATH", Path("builder.py"))
    monkeypatch.setattr(support, "TEST_PATH", Path("test_builder.py"))
    monkeypatch.setattr(support, "IMPLEMENTATION_CONTRACT", Path("contract.md"))
    for path in ("builder.py", "test_builder.py", "contract.md"):
        (tmp_path / path).write_text(path + "\n")
    calls: list[tuple[str, ...]] = []
    results = iter(
        [
            subprocess.CompletedProcess((), 0, "", ""),
            subprocess.CompletedProcess((), 0, "", ""),
        ]
    )

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return next(results)

    monkeypatch.setattr(support, "_git_check", git)
    assert support._assert_protocol_committed() is None
    assert calls == [
        (
            "ls-files",
            "--error-unmatch",
            "--",
            "builder.py",
            "test_builder.py",
            "contract.md",
        ),
        (
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "builder.py",
            "test_builder.py",
            "contract.md",
        ),
    ]


def test_implementation_contract_hash_is_frozen() -> None:
    assert support.sha256_file(support.IMPLEMENTATION_CONTRACT) == (
        support.IMPLEMENTATION_CONTRACT_SHA256
    )


def test_atomic_link_never_clobbers_existing_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(support, "REPOSITORY_ROOT", tmp_path)
    output = tmp_path / "artifact.bin"
    support._atomic_link(output, b"first")
    with pytest.raises(FileExistsError):
        support._atomic_link(output, b"second")
    assert output.read_bytes() == b"first"
