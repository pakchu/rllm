from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from fractions import Fraction

import pytest

from training import build_ofr_repo_collateral_routing_efficiency_support as support
from training import preregister_ofr_repo_collateral_routing_efficiency as prereg


UTC = timezone.utc


def _source_rows(
    day: date,
    available_at: datetime,
    *,
    disclosure_edit: bool = False,
) -> dict[str, support.rmsr_support.SourceRow]:
    values = {
        "REPO-GCF_AR_AG-P": Fraction(2),
        "REPO-GCF_AR_T-P": Fraction(1),
        "REPO-TRIV1_AR_AG-P": Fraction(3, 2),
        "REPO-TRIV1_AR_T-P": Fraction(1),
        "REPO-GCF_TV_AG-P": Fraction(60),
        "REPO-GCF_TV_T-P": Fraction(40),
        "REPO-TRIV1_TV_AG-P": Fraction(40),
        "REPO-TRIV1_TV_T-P": Fraction(60),
    }
    return {
        mnemonic: support.rmsr_support.SourceRow(
            mnemonic=mnemonic,
            observation_date=day,
            available_at=available_at,
            value=value,
            disclosure_edit=disclosure_edit and index == 0,
        )
        for index, (mnemonic, value) in enumerate(values.items())
    }


def _feature(
    index: int,
    *,
    quantity: Fraction,
    rate: Fraction,
    available_at: datetime | None = None,
    epoch: int = 0,
    decision_allowed: bool = True,
) -> support.FeatureRow:
    start = date(2019, 1, 1)
    day = start + timedelta(days=index)
    available = available_at or datetime.combine(day, datetime.min.time(), UTC)
    return support.FeatureRow(
        observation_date=day,
        available_at=available,
        epoch=epoch,
        decision_allowed=decision_allowed,
        components=support._components(quantity, rate),
    )


def _rank(
    index: int,
    *,
    quantity: Fraction,
    rate: Fraction,
    product_unit: Fraction,
    quantity_unit: Fraction = Fraction(0),
    rate_unit: Fraction = Fraction(0),
    epoch: int = 0,
    available_at: datetime | None = None,
) -> support.RankRow:
    feature = _feature(
        index,
        quantity=quantity,
        rate=rate,
        epoch=epoch,
        available_at=available_at,
    )
    return support.RankRow(
        feature=feature,
        units={
            "quantity_gap": quantity_unit,
            "rate_gap": rate_unit,
            "routing_pressure": product_unit,
            "absolute_pressure": abs(product_unit),
            "absolute_quantity_gap": abs(quantity_unit),
            "absolute_rate_gap": abs(rate_unit),
        },
    )


def _scheduled_monthly_primary() -> list[support.Scheduled]:
    quadrants = (
        (Fraction(1), Fraction(1)),
        (Fraction(-1), Fraction(-1)),
        (Fraction(1), Fraction(-1)),
        (Fraction(-1), Fraction(1)),
    )
    rows: list[support.Scheduled] = []
    quadrant_index = 0
    for year in (2021, 2022, 2023):
        for month in range(1, 13):
            for day in (5, 15):
                quantity, rate = quadrants[quadrant_index % len(quadrants)]
                quadrant_index += 1
                components = support._components(quantity, rate)
                state = 1 if components["routing_pressure"] > 0 else -1
                entry = datetime(year, month, day, 0, 5, tzinfo=UTC)
                split = "train" if year < 2023 else "selection"
                product_unit = Fraction(3, 4) * state
                units = {
                    "quantity_gap": Fraction(3, 4) * (1 if quantity > 0 else -1),
                    "rate_gap": Fraction(3, 4) * (1 if rate > 0 else -1),
                    "routing_pressure": product_unit,
                    "absolute_pressure": Fraction(3, 4),
                    "absolute_quantity_gap": Fraction(3, 4),
                    "absolute_rate_gap": Fraction(3, 4),
                }
                rows.append(
                    support.Scheduled(
                        control="primary",
                        observation_date=date(year, month, day),
                        signal_time=entry - support.BAR,
                        entry_time=entry,
                        exit_time=entry + support.HOLD,
                        split=split,
                        side=-state,
                        state=state,
                        score=product_unit,
                        components=components,
                        units=units,
                    )
                )
    return rows


def _clock_set(primary: list[support.Scheduled]) -> dict[str, list[support.Scheduled]]:
    clocks = {
        name: [support.replace(row, control=name) for row in primary]
        for name in support.CONTROL_NAMES
    }
    for original_name, swapped_name in (
        ("quantity_gap_original", "quantity_gap_swapped"),
        ("rate_gap_original", "rate_gap_swapped"),
    ):
        original = clocks[original_name]
        swapped: list[support.Scheduled] = []
        for row in original:
            components = dict(row.components)
            components["quantity_gap"] = -components["quantity_gap"]
            components["rate_gap"] = -components["rate_gap"]
            units = dict(row.units)
            units["quantity_gap"] = -units["quantity_gap"]
            units["rate_gap"] = -units["rate_gap"]
            swapped.append(
                support.replace(
                    row,
                    control=swapped_name,
                    side=-row.side,
                    state=-row.state,
                    score=-row.score,
                    components=components,
                    units=units,
                )
            )
        clocks[swapped_name] = swapped
    clocks["exact_direction_flip"] = [
        support.replace(row, control="exact_direction_flip", side=-row.side)
        for row in primary
    ]
    clocks["deterministic_random_side"] = [
        support.replace(
            row,
            control="deterministic_random_side",
            side=support._random_side(row.entry_time),
        )
        for row in primary
    ]
    clocks["constant_long"] = [
        support.replace(row, control="constant_long", side=1) for row in primary
    ]
    clocks["constant_short"] = [
        support.replace(row, control="constant_short", side=-1) for row in primary
    ]
    return clocks


def test_signed_feature_arithmetic_materiality_and_equal_batch_are_exact() -> None:
    available = datetime(2021, 1, 10, tzinfo=UTC)
    first = date(2021, 1, 1)
    second = date(2021, 1, 2)
    invalid = date(2021, 1, 3)
    features, audit = support.build_features(
        {
            first: _source_rows(first, available),
            second: _source_rows(second, available),
            invalid: _source_rows(
                invalid, datetime(2021, 1, 11, tzinfo=UTC), disclosure_edit=True
            ),
        }
    )

    assert len(features) == 2
    assert features[0].components == {
        "quantity_gap": Fraction(1, 5),
        "rate_gap": Fraction(1, 2),
        "routing_pressure": Fraction(1, 10),
        "absolute_pressure": Fraction(1, 10),
        "absolute_quantity_gap": Fraction(1, 5),
        "absolute_rate_gap": Fraction(1, 2),
    }
    assert [row.decision_allowed for row in features] == [False, True]
    assert audit.invalid_missing_or_null_dates == 1
    assert audit.equal_availability_rows_suppressed == 1
    assert audit.venue_swap_dates_checked == 2
    assert audit.venue_swap_identity_failures == 0


def test_rank_rows_use_strict_prebatch_history() -> None:
    features = [
        _feature(index, quantity=Fraction(index + 1), rate=Fraction(index + 1))
        for index in range(support.LOOKBACK)
    ]
    shared = datetime(2020, 1, 1, tzinfo=UTC)
    features.extend(
        [
            _feature(
                support.LOOKBACK,
                quantity=Fraction(1_000),
                rate=Fraction(1_000),
                available_at=shared,
                decision_allowed=False,
            ),
            _feature(
                support.LOOKBACK + 1,
                quantity=Fraction(1),
                rate=Fraction(1),
                available_at=shared,
            ),
        ]
    )

    ranks = support.build_rank_rows(features)
    assert len(ranks) == 2
    assert ranks[0].units["quantity_gap"] == 1
    assert ranks[1].units["quantity_gap"] == Fraction(-251, 252)
    assert ranks[1].units["routing_pressure"] == Fraction(-251, 252)


def test_primary_transition_is_directional_nonpersistent_and_break_safe() -> None:
    base = datetime(2021, 1, 1, tzinfo=UTC)
    rows = [
        _rank(
            0,
            quantity=Fraction(1),
            rate=Fraction(1),
            product_unit=Fraction(0),
            available_at=base,
        ),
        _rank(
            1,
            quantity=Fraction(1),
            rate=Fraction(1),
            product_unit=Fraction(3, 4),
            available_at=base + timedelta(days=4),
        ),
        _rank(
            2,
            quantity=Fraction(1),
            rate=Fraction(1),
            product_unit=Fraction(3, 4),
            available_at=base + timedelta(days=8),
        ),
        _rank(
            3,
            quantity=Fraction(1),
            rate=Fraction(-1),
            product_unit=Fraction(-3, 4),
            available_at=base + timedelta(days=12),
        ),
        _rank(
            4,
            quantity=Fraction(1),
            rate=Fraction(1),
            product_unit=Fraction(3, 4),
            epoch=1,
            available_at=base + timedelta(days=16),
        ),
    ]

    candidates = support.transition_candidates(support._state_points(rows, "primary"))
    assert [(row.state, row.side) for row in candidates] == [(1, -1), (-1, 1)]
    assert [row.rank.feature.observation_date for row in candidates] == [
        rows[1].feature.observation_date,
        rows[3].feature.observation_date,
    ]


def test_venue_swapped_label_pairs_have_exact_flip_and_same_schedule() -> None:
    base = datetime(2021, 1, 1, tzinfo=UTC)
    rows = [
        _rank(
            index,
            quantity=Fraction(1),
            rate=Fraction(1),
            product_unit=Fraction(0),
            quantity_unit=unit,
            rate_unit=unit,
            available_at=base + timedelta(days=index * 4),
        )
        for index, unit in enumerate(
            (Fraction(0), Fraction(3, 4), Fraction(-3, 4), Fraction(3, 4))
        )
    ]
    original = support.schedule(
        "quantity_gap_original",
        support.transition_candidates(
            support._state_points(rows, "quantity_gap_original")
        ),
    )
    swapped = support.schedule(
        "quantity_gap_swapped",
        support.transition_candidates(
            support._state_points(rows, "quantity_gap_swapped")
        ),
    )

    assert len(original) == len(swapped) == 3
    assert support._label_pair_valid(original, swapped)
    assert all(
        left.components["routing_pressure"] == right.components["routing_pressure"]
        for left, right in zip(original, swapped)
    )


def test_permutations_are_deterministic_and_preserve_frozen_marginals() -> None:
    features = [
        _feature(index, quantity=Fraction(index + 1), rate=Fraction(index + 2))
        for index in range(20)
    ]
    rate_first = support.permute_features(features, "year_rate_gap_permutation")
    rate_second = support.permute_features(features, "year_rate_gap_permutation")
    product = support.permute_features(features, "year_product_permutation")

    assert rate_first == rate_second
    assert [row.available_at for row in rate_first] == [
        row.available_at for row in features
    ]
    assert sorted(row.components["rate_gap"] for row in rate_first) == sorted(
        row.components["rate_gap"] for row in features
    )
    assert all(
        row.components["routing_pressure"]
        == row.components["quantity_gap"] * row.components["rate_gap"]
        for row in rate_first
    )
    assert sorted(row.components["routing_pressure"] for row in product) == sorted(
        row.components["routing_pressure"] for row in features
    )


def test_full_control_battery_is_complete_structural_and_pair_exact() -> None:
    features: list[support.FeatureRow] = []
    start = date(2019, 1, 1)
    for index in range(850):
        day = start + timedelta(days=index)
        quantity = Fraction(((index * 7) % 31) - 15, 10)
        rate = Fraction(((index * 11) % 37) - 18, 10)
        quantity = quantity or Fraction(1, 10)
        rate = rate or Fraction(-1, 10)
        features.append(
            support.FeatureRow(
                observation_date=day,
                available_at=datetime.combine(day, datetime.min.time(), UTC)
                + timedelta(days=8),
                epoch=0,
                decision_allowed=True,
                components=support._components(quantity, rate),
            )
        )

    ranks = support.build_rank_rows(features)
    clocks = support.build_clocks(features, ranks)

    assert set(clocks) == set(support.CONTROL_NAMES)
    assert clocks["primary"]
    assert all(support._clock_valid(name, rows) for name, rows in clocks.items())
    assert support._label_pair_valid(
        clocks["quantity_gap_original"], clocks["quantity_gap_swapped"]
    )
    assert support._label_pair_valid(
        clocks["rate_gap_original"], clocks["rate_gap_swapped"]
    )
    assert support._side_control_valid(
        clocks["primary"],
        clocks["exact_direction_flip"],
        lambda row: -row.side,
    )


def test_frozen_source_support_gates_accept_balanced_synthetic_clock() -> None:
    primary = _scheduled_monthly_primary()
    clocks = _clock_set(primary)
    audit = support.SourceAudit(
        normalized_rows_read=0,
        required_rows_read=0,
        source_dates_seen=72,
        valid_feature_dates=72,
        invalid_missing_or_null_dates=0,
        invalid_materiality_dates=0,
        equal_availability_rows_suppressed=0,
        venue_swap_dates_checked=72,
        venue_swap_identity_failures=0,
    )

    checks, summaries = support.source_support(
        clocks, prereg.policy_payload(), audit
    )
    assert all(checks.values()), {name for name, passed in checks.items() if not passed}
    assert summaries["train"]["events"] == 48
    assert summaries["selection"]["events"] == 24
    assert set(summaries["train"]["quadrant_counts"]) == {
        "q+r+",
        "q-r-",
        "q+r-",
        "q-r+",
    }


def test_common_window_excludes_whole_intervals_and_reports_every_bucket() -> None:
    events = [
        support.ComparatorEvent(
            support.TRAIN_START - timedelta(days=2),
            support.TRAIN_START - timedelta(days=1),
            1,
        ),
        support.ComparatorEvent(
            support.TRAIN_START - timedelta(hours=1),
            support.TRAIN_START + timedelta(hours=1),
            -1,
        ),
        support.ComparatorEvent(
            support.TRAIN_START + timedelta(days=1),
            support.TRAIN_START + timedelta(days=2),
            1,
        ),
        support.ComparatorEvent(
            support.SELECTION_END - timedelta(hours=1),
            support.SELECTION_END + timedelta(hours=1),
            -1,
        ),
        support.ComparatorEvent(
            support.SELECTION_END + timedelta(days=1),
            support.SELECTION_END + timedelta(days=2),
            1,
        ),
    ]

    contained, counts = support._filter_required_groups(
        {"example": events}, rows_read=5, raw_group_counts={"example": 5}
    )
    assert contained == {"example": [events[2]]}
    assert counts["example"] == {
        "total_raw_rows_parsed": 5,
        "fully_contained_rows_used": 1,
        "rows_before_window": 1,
        "rows_after_window": 1,
        "rows_crossing_boundary": 2,
    }


def test_raw_comparator_overlap_fails_even_before_common_window() -> None:
    start = support.TRAIN_START - timedelta(days=10)
    events = [
        support.ComparatorEvent(start, start + timedelta(days=2), 1),
        support.ComparatorEvent(
            start + timedelta(days=1), start + timedelta(days=3), -1
        ),
    ]
    with pytest.raises(support.ComparatorValidationError, match="overlapping"):
        support._validate_raw_groups({"example": events}, rows_read=2)


def test_required_comparator_with_zero_contained_rows_fails_closed() -> None:
    event = support.ComparatorEvent(
        support.TRAIN_START - timedelta(days=2),
        support.TRAIN_START - timedelta(days=1),
        1,
    )
    with pytest.raises(
        support.ComparatorValidationError, match="zero contained rows"
    ) as captured:
        support._filter_required_groups(
            {"example": [event]}, rows_read=1, raw_group_counts={"example": 1}
        )
    assert captured.value.window_counts["example"]["rows_before_window"] == 1


def test_clock_schema_contains_no_market_outcome_fields() -> None:
    forbidden = {"open", "high", "low", "close", "return", "pnl", "cagr", "mdd", "funding"}
    assert not forbidden.intersection(name.lower() for name in support.CLOCK_COLUMNS)
