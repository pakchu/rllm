from __future__ import annotations

from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any, cast

import pytest

from training import cross_venue_volatility_shape_handoff as cvvh


UTC = timezone.utc


def candle(
    open_: str,
    high: str,
    low: str,
    close: str,
) -> cvvh.Candle:
    return cvvh.Candle.from_tokens(
        open=open_,
        high=high,
        low=low,
        close=close,
    )


UP_STRONG = candle("100", "112", "99", "110")
DOWN_STRONG = candle("100", "101", "88", "90")
UP_WEAK = candle("100", "103", "99", "102")
DOWN_WEAK = candle("100", "101", "97", "98")
FLAT = candle("100", "101", "99", "100")


def hour(
    offset: int,
    bvol: cvvh.Candle | None,
    dvol: cvvh.Candle | None,
    *,
    start: datetime = datetime(2023, 6, 20, tzinfo=UTC),
    available_delay: timedelta = timedelta(0),
    valid: bool = True,
) -> cvvh.JoinedHour:
    close_time = start + offset * timedelta(hours=1)
    return cvvh.JoinedHour(
        close_time=close_time,
        available_at=close_time + available_delay,
        bvol=bvol,
        dvol=dvol,
        source_valid=valid,
    )


def test_primary_state_uses_exact_opposite_body_and_both_lead_conditions() -> None:
    assert cvvh.state_for_hour(hour(0, UP_STRONG, DOWN_WEAK)) == "SHORT"
    assert cvvh.state_for_hour(hour(0, DOWN_STRONG, UP_WEAK)) == "LONG"
    assert cvvh.state_for_hour(hour(0, UP_WEAK, DOWN_STRONG)) is None
    assert cvvh.state_for_hour(hour(0, FLAT, DOWN_WEAK)) is None


def test_exact_ties_abstain_without_binary_float_arithmetic() -> None:
    body_tied_up = candle("100", "120", "90", "110")
    body_tied_down = candle("200", "210", "180", "180")
    assert cvvh.state_for_hour(hour(0, body_tied_up, body_tied_down)) is None

    range_tied_up = candle("100", "110", "90", "105")
    range_tied_down = candle("200", "220", "180", "198")
    assert cvvh.state_for_hour(hour(0, range_tied_up, range_tied_down)) is None

    with pytest.raises(ValueError, match="exact decimal token"):
        cvvh.Candle.from_tokens(
            open=cast(Any, 100.0),
            high="101",
            low="99",
            close="100",
        )


def test_unequal_opens_use_exact_cross_multiplication_for_both_sides() -> None:
    short = hour(
        0,
        candle("80", "90", "72", "88"),
        candle("200", "210", "185", "190"),
    )
    long = hour(
        0,
        candle("80", "88", "70", "72"),
        candle("200", "215", "190", "210"),
    )
    assert cvvh.state_for_hour(short) == "SHORT"
    assert cvvh.state_for_hour(long) == "LONG"


def test_adjacent_large_cross_products_do_not_round_to_a_tie() -> None:
    n = 10**60
    b_open = n - 1
    d_open = n
    bvol = candle(
        str(b_open),
        str(n + 1),
        str(n - 2),
        str(n),
    )
    dvol = candle(
        str(d_open),
        str(n + 1),
        str(n - 1),
        str(n - 1),
    )
    assert cvvh.state_for_hour(hour(0, bvol, dvol)) == "SHORT"


def test_compact_extreme_decimal_exponent_fails_before_integer_expansion() -> None:
    with pytest.raises(ValueError, match="decimal token bounds"):
        candle("1e1000000000", "1e1000000000", "1e1000000000", "1e1000000000")


@pytest.mark.parametrize(
    "tokens,match",
    [
        (("0", "1", "0.5", "0.7"), "strictly positive"),
        (("100", "99", "98", "101"), "high violates"),
        (("100", "102", "101", "99"), "low violates"),
    ],
)
def test_ohlc_envelope_fails_closed(
    tokens: tuple[str, str, str, str],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        candle(*tokens)


def test_false_to_true_none_and_opposite_side_transitions_emit() -> None:
    rows = [
        hour(0, FLAT, DOWN_WEAK),
        hour(1, UP_STRONG, DOWN_WEAK),
        hour(2, UP_STRONG, DOWN_WEAK),
        hour(3, FLAT, DOWN_WEAK),
        hour(4, DOWN_STRONG, UP_WEAK),
        hour(5, UP_STRONG, DOWN_WEAK),
    ]
    raw = cvvh.raw_candidates(rows)
    assert [(row.signal_time.hour, row.side) for row in raw] == [
        (1, "SHORT"),
        (4, "LONG"),
        (5, "SHORT"),
    ]


def test_gap_or_invalid_hour_suppresses_first_post_boundary_state() -> None:
    start = datetime(2023, 6, 20, tzinfo=UTC)
    gap_rows = [
        hour(0, FLAT, DOWN_WEAK, start=start),
        hour(2, UP_STRONG, DOWN_WEAK, start=start),
        hour(3, UP_STRONG, DOWN_WEAK, start=start),
        hour(4, FLAT, DOWN_WEAK, start=start),
        hour(5, UP_STRONG, DOWN_WEAK, start=start),
    ]
    assert [row.signal_time.hour for row in cvvh.raw_candidates(gap_rows)] == [5]

    invalid_rows = [
        hour(0, FLAT, DOWN_WEAK, start=start),
        hour(1, None, DOWN_WEAK, start=start, valid=False),
        hour(2, UP_STRONG, DOWN_WEAK, start=start),
        hour(3, UP_STRONG, DOWN_WEAK, start=start),
        hour(4, FLAT, DOWN_WEAK, start=start),
        hour(5, UP_STRONG, DOWN_WEAK, start=start),
    ]
    assert [row.signal_time.hour for row in cvvh.raw_candidates(invalid_rows)] == [
        5
    ]


def test_independent_control_definitions_are_exact() -> None:
    primary = hour(0, UP_STRONG, DOWN_WEAK)
    deribit = hour(0, DOWN_WEAK, UP_STRONG)
    body_only = hour(
        0,
        candle("100", "102.1", "99.9", "102"),
        candle("100", "101.5", "98.5", "99"),
    )
    range_only = hour(
        0,
        candle("100", "106", "94", "101"),
        candle("100", "105", "95", "98"),
    )
    assert cvvh.state_for_hour(primary, cvvh.BODY_LEAD_ONLY) == "SHORT"
    assert cvvh.state_for_hour(primary, cvvh.RANGE_LEAD_ONLY) == "SHORT"
    assert cvvh.state_for_hour(deribit, cvvh.DERIBIT_LED) == "SHORT"
    assert cvvh.state_for_hour(body_only, cvvh.BODY_LEAD_ONLY) == "SHORT"
    assert cvvh.state_for_hour(body_only, cvvh.RANGE_LEAD_ONLY) is None
    assert cvvh.state_for_hour(range_only, cvvh.RANGE_LEAD_ONLY) == "SHORT"
    assert cvvh.state_for_hour(range_only, cvvh.BODY_LEAD_ONLY) is None


def test_stale_deribit_requires_exact_three_hour_history() -> None:
    rows = [
        hour(0, FLAT, DOWN_WEAK),
        hour(1, FLAT, DOWN_WEAK),
        hour(2, UP_STRONG, UP_WEAK),
        hour(3, UP_STRONG, DOWN_WEAK),
        hour(4, FLAT, DOWN_WEAK),
        hour(5, UP_STRONG, UP_WEAK),
    ]
    stale = cvvh.raw_candidates(rows, cvvh.STALE_DERIBIT)
    assert [(row.signal_time.hour, row.side) for row in stale] == [
        (2, "SHORT"),
        (5, "SHORT"),
    ]


def test_stale_deribit_resets_on_real_gap_and_recovers_after_three_hours() -> None:
    rows = [
        hour(0, FLAT, DOWN_WEAK),
        hour(1, FLAT, DOWN_WEAK),
        hour(3, UP_STRONG, DOWN_WEAK),
        hour(4, FLAT, DOWN_WEAK),
        hour(5, UP_STRONG, DOWN_WEAK),
        hour(6, FLAT, DOWN_WEAK),
        hour(7, UP_STRONG, DOWN_WEAK),
    ]
    assert [
        row.signal_time.hour
        for row in cvvh.raw_candidates(rows, cvvh.STALE_DERIBIT)
    ] == [5, 7]


def test_stale_deribit_uses_t_minus_two_for_prior_state() -> None:
    prior_short = [
        hour(0, FLAT, DOWN_WEAK),
        hour(1, UP_STRONG, DOWN_WEAK),
        hour(2, UP_STRONG, UP_WEAK),
    ]
    prior_none = [
        hour(0, FLAT, UP_WEAK),
        hour(1, UP_STRONG, DOWN_WEAK),
        hour(2, UP_STRONG, UP_WEAK),
    ]
    assert cvvh.raw_candidates(prior_short, cvvh.STALE_DERIBIT) == ()
    assert [
        row.signal_time.hour
        for row in cvvh.raw_candidates(prior_none, cvvh.STALE_DERIBIT)
    ] == [2]


def test_stale_deribit_uses_current_bvol_previous_dvol_and_current_availability() -> None:
    rows_a = [
        hour(0, FLAT, UP_WEAK),
        hour(1, FLAT, DOWN_WEAK),
        hour(
            2,
            UP_STRONG,
            UP_WEAK,
            available_delay=timedelta(minutes=2, microseconds=1),
        ),
    ]
    rows_b = [
        rows_a[0],
        rows_a[1],
        hour(
            2,
            UP_STRONG,
            DOWN_STRONG,
            available_delay=timedelta(minutes=2, microseconds=1),
        ),
    ]
    event_a = cvvh.raw_candidates(rows_a, cvvh.STALE_DERIBIT)[0]
    event_b = cvvh.raw_candidates(rows_b, cvvh.STALE_DERIBIT)[0]
    assert event_a == event_b
    assert event_a.available_at == rows_a[2].available_at
    assert event_a.entry_time == rows_a[2].close_time + timedelta(minutes=10)


@pytest.mark.parametrize("invalid_index", [0, 1, 2])
def test_stale_deribit_requires_all_three_joined_hours_valid(
    invalid_index: int,
) -> None:
    rows = [
        hour(0, FLAT, UP_WEAK),
        hour(1, FLAT, DOWN_WEAK),
        hour(2, UP_STRONG, DOWN_WEAK),
    ]
    row = rows[invalid_index]
    rows[invalid_index] = cvvh.JoinedHour(
        close_time=row.close_time,
        available_at=row.available_at,
        bvol=row.bvol,
        dvol=row.dvol,
        source_valid=False,
    )
    assert cvvh.raw_candidates(rows, cvvh.STALE_DERIBIT) == ()


@pytest.mark.parametrize(
    ("invalid_index", "missing_venue"),
    [
        (0, "bvol"),
        (0, "dvol"),
        (1, "bvol"),
        (1, "dvol"),
        (2, "bvol"),
        (2, "dvol"),
    ],
)
def test_stale_deribit_requires_both_candles_in_all_three_hours(
    invalid_index: int,
    missing_venue: str,
) -> None:
    rows = [
        hour(0, FLAT, UP_WEAK),
        hour(1, FLAT, DOWN_WEAK),
        hour(2, UP_STRONG, DOWN_WEAK),
    ]
    row = rows[invalid_index]
    rows[invalid_index] = cvvh.JoinedHour(
        close_time=row.close_time,
        available_at=row.available_at,
        bvol=None if missing_venue == "bvol" else row.bvol,
        dvol=None if missing_venue == "dvol" else row.dvol,
        source_valid=True,
    )
    assert cvvh.raw_candidates(rows, cvvh.STALE_DERIBIT) == ()


def test_entry_waits_one_complete_bar_after_ceil_and_hold_is_432_bars() -> None:
    aligned = cvvh.raw_candidates(
        [hour(0, FLAT, DOWN_WEAK), hour(1, UP_STRONG, DOWN_WEAK)]
    )[0]
    assert aligned.entry_time == aligned.available_at + timedelta(minutes=5)
    assert aligned.exit_time - aligned.entry_time == timedelta(hours=36)

    delayed = cvvh.raw_candidates(
        [
            hour(0, FLAT, DOWN_WEAK),
            hour(
                1,
                UP_STRONG,
                DOWN_WEAK,
                available_delay=timedelta(minutes=2, seconds=1),
            ),
        ]
    )[0]
    assert delayed.entry_time == delayed.signal_time + timedelta(minutes=10)


@pytest.mark.parametrize(
    "value,expected",
    [
        (
            datetime(5000, 1, 1, 0, 5, tzinfo=UTC),
            datetime(5000, 1, 1, 0, 5, tzinfo=UTC),
        ),
        (
            datetime(5000, 1, 1, 0, 4, 59, 999999, tzinfo=UTC),
            datetime(5000, 1, 1, 0, 5, tzinfo=UTC),
        ),
        (
            datetime(5000, 1, 1, 0, 5, 0, 1, tzinfo=UTC),
            datetime(5000, 1, 1, 0, 10, tzinfo=UTC),
        ),
    ],
)
def test_five_minute_ceiling_is_integer_exact(
    value: datetime,
    expected: datetime,
) -> None:
    assert cvvh.ceil_to_five_minutes(value) == expected


def test_global_reservation_suppresses_without_queue_and_allows_exit_boundary() -> None:
    first = cvvh.raw_candidates(
        [hour(0, FLAT, DOWN_WEAK), hour(1, UP_STRONG, DOWN_WEAK)]
    )[0]
    inside = cvvh.ScheduledEvent(
        signal_id="CVVH-432|primary|T=2023-06-20T02:00:00Z",
        control=cvvh.PRIMARY,
        signal_time=first.signal_time + timedelta(hours=1),
        available_at=first.available_at + timedelta(hours=1),
        entry_time=first.entry_time + timedelta(hours=1),
        exit_time=first.entry_time + timedelta(hours=37),
        side="LONG",
    )
    boundary = cvvh.ScheduledEvent(
        signal_id="CVVH-432|primary|T=2023-06-21T13:00:00Z",
        control=cvvh.PRIMARY,
        signal_time=first.signal_time + timedelta(hours=36),
        available_at=first.available_at + timedelta(hours=36),
        entry_time=first.exit_time,
        exit_time=first.exit_time + timedelta(hours=36),
        side="SHORT",
    )
    assert cvvh.reserve_globally([inside, boundary, first]) == (first, boundary)


def test_reservation_uses_full_canonical_tie_break_and_rejects_hold_drift() -> None:
    base_entry = datetime(2023, 6, 21, tzinfo=UTC)
    later_signal = cvvh.ScheduledEvent(
        signal_id="CVVH-432|primary|T=2023-06-20T23:00:00Z",
        control=cvvh.PRIMARY,
        signal_time=datetime(2023, 6, 20, 23, tzinfo=UTC),
        available_at=datetime(2023, 6, 20, 23, tzinfo=UTC),
        entry_time=base_entry,
        exit_time=base_entry + timedelta(hours=36),
        side="LONG",
    )
    earlier_signal = cvvh.ScheduledEvent(
        signal_id="CVVH-432|primary|T=2023-06-20T22:00:00Z",
        control=cvvh.PRIMARY,
        signal_time=datetime(2023, 6, 20, 22, tzinfo=UTC),
        available_at=datetime(2023, 6, 20, 22, tzinfo=UTC),
        entry_time=base_entry,
        exit_time=base_entry + timedelta(hours=36),
        side="SHORT",
    )
    assert cvvh.reserve_globally([later_signal, earlier_signal]) == (
        earlier_signal,
    )
    id_b = cvvh.ScheduledEvent(
        **{
            **earlier_signal.__dict__,
            "signal_id": "CVVH-432|z-control|T=2023-06-20T22:00:00Z",
        }
    )
    id_a = cvvh.ScheduledEvent(
        **{
            **earlier_signal.__dict__,
            "signal_id": "CVVH-432|a-control|T=2023-06-20T22:00:00Z",
        }
    )
    assert cvvh.reserve_globally([id_b, id_a]) == (id_a,)
    side_short = cvvh.ScheduledEvent(
        **{**id_a.__dict__, "side": "SHORT"}
    )
    side_long = cvvh.ScheduledEvent(
        **{**id_a.__dict__, "side": "LONG"}
    )
    assert cvvh.reserve_globally([side_short, side_long]) == (side_long,)
    drifted = cvvh.ScheduledEvent(
        **{
            **later_signal.__dict__,
            "exit_time": later_signal.exit_time + timedelta(minutes=5),
        }
    )
    with pytest.raises(ValueError, match="hold drifted"):
        cvvh.reserve_globally([drifted])


def test_parent_set_controls_keep_parent_identity_and_do_not_rereserve() -> None:
    primary = cvvh.build_clock(
        [
            hour(0, FLAT, DOWN_WEAK),
            hour(1, UP_STRONG, DOWN_WEAK),
            hour(2, FLAT, DOWN_WEAK),
            hour(40, FLAT, DOWN_WEAK),
            hour(41, DOWN_STRONG, UP_WEAK),
        ]
    )
    flipped = cvvh.derive_parent_set_control(primary, cvvh.DIRECTION_FLIP)
    random_a = cvvh.derive_parent_set_control(
        primary, cvvh.DETERMINISTIC_RANDOM_SIDE
    )
    random_b = cvvh.derive_parent_set_control(
        primary, cvvh.DETERMINISTIC_RANDOM_SIDE
    )
    constant_long = cvvh.derive_parent_set_control(primary, cvvh.CONSTANT_LONG)
    constant_short = cvvh.derive_parent_set_control(
        primary, cvvh.CONSTANT_SHORT
    )
    delayed = cvvh.derive_parent_set_control(
        primary, cvvh.ONE_BAR_DELAYED_ENTRY
    )
    assert [row.side for row in flipped] == [
        "SHORT" if row.side == "LONG" else "LONG" for row in primary
    ]
    assert random_a == random_b
    assert all(row.side in {"LONG", "SHORT"} for row in random_a)
    assert [row.side for row in constant_long] == ["LONG"] * len(primary)
    assert [row.side for row in constant_short] == ["SHORT"] * len(primary)
    assert [row.entry_time for row in delayed] == [
        row.entry_time + timedelta(minutes=5) for row in primary
    ]
    assert [row.exit_time for row in delayed] == [
        row.exit_time + timedelta(minutes=5) for row in primary
    ]
    assert [row.signal_time for row in delayed] == [
        row.signal_time for row in primary
    ]
    assert [row.available_at for row in delayed] == [
        row.available_at for row in primary
    ]
    assert delayed[0].signal_id == (
        "CVVH-432|one_bar_delayed_entry|T=2023-06-20T01:00:00Z"
    )
    assert constant_long[0].control == "constant_long"
    derived = {
        control: cvvh.derive_parent_set_control(primary, control)
        for control in cvvh.PARENT_SET_CONTROLS
    }
    for control, clock in derived.items():
        assert clock[0].control == control
        assert clock[0].signal_id == (
            f"CVVH-432|{control}|T=2023-06-20T01:00:00Z"
        )
    assert len(delayed) == len(primary)


def test_deterministic_random_side_has_frozen_sha256_vectors() -> None:
    assert (
        cvvh.deterministic_random_side(
            "CVVH-432|primary|T=2023-06-20T01:00:00Z"
        )
        == "LONG"
    )
    assert (
        cvvh.deterministic_random_side(
            "CVVH-432|primary|T=2023-06-21T13:00:00Z"
        )
        == "SHORT"
    )


def test_signal_ids_and_order_are_canonical() -> None:
    event = cvvh.raw_candidates(
        [hour(0, FLAT, DOWN_WEAK), hour(1, UP_STRONG, DOWN_WEAK)]
    )[0]
    assert event.signal_id == "CVVH-432|primary|T=2023-06-20T01:00:00Z"
    assert event.control == "primary"


@pytest.mark.parametrize(
    ("control", "signal_row"),
    [
        (
            cvvh.DERIBIT_LED,
            hour(1, DOWN_WEAK, UP_STRONG),
        ),
        (
            cvvh.BODY_LEAD_ONLY,
            hour(
                1,
                candle("100", "102.1", "99.9", "102"),
                candle("100", "101.5", "98.5", "99"),
            ),
        ),
        (
            cvvh.RANGE_LEAD_ONLY,
            hour(
                1,
                candle("100", "106", "94", "101"),
                candle("100", "105", "95", "98"),
            ),
        ),
    ],
)
def test_build_clock_uses_and_independently_reserves_each_control(
    control: str,
    signal_row: cvvh.JoinedHour,
) -> None:
    repeated = cvvh.JoinedHour(
        close_time=signal_row.close_time + timedelta(hours=2),
        available_at=signal_row.available_at + timedelta(hours=2),
        bvol=signal_row.bvol,
        dvol=signal_row.dvol,
        source_valid=True,
    )
    between = hour(2, FLAT, FLAT)
    rows = [hour(0, FLAT, FLAT), signal_row, between, repeated]
    raw = cvvh.raw_candidates(rows, control)
    clock = cvvh.build_clock(rows, control)
    assert len(raw) == 2
    assert len(clock) == 1
    assert clock[0].control == control
    assert clock[0].signal_id == (
        f"CVVH-432|{control}|T=2023-06-20T01:00:00Z"
    )


def test_naive_availability_and_unsorted_hours_fail_closed() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        cvvh.JoinedHour(
            close_time=datetime(2023, 1, 1),
            available_at=datetime(2023, 1, 1),
            bvol=UP_STRONG,
            dvol=DOWN_WEAK,
        )
    rows = [
        hour(1, FLAT, DOWN_WEAK),
        hour(0, UP_STRONG, DOWN_WEAK),
    ]
    with pytest.raises(ValueError, match="unique and increasing"):
        cvvh.raw_candidates(rows)


class _NullOffsetTimezone(tzinfo):
    def utcoffset(self, dt: datetime | None) -> None:
        return None

    def dst(self, dt: datetime | None) -> None:
        return None

    def tzname(self, dt: datetime | None) -> str:
        return "NULL"


def test_non_null_tzinfo_with_null_offset_is_still_rejected() -> None:
    deceptive = datetime(2023, 1, 1, tzinfo=_NullOffsetTimezone())
    with pytest.raises(ValueError, match="timezone-aware"):
        cvvh.ceil_to_five_minutes(deceptive)


def test_rfc3339_identifier_uses_four_digit_year() -> None:
    start = datetime(1, 1, 1, tzinfo=UTC)
    event = cvvh.raw_candidates(
        [
            hour(0, FLAT, DOWN_WEAK, start=start),
            hour(1, UP_STRONG, DOWN_WEAK, start=start),
        ]
    )[0]
    assert event.signal_id == "CVVH-432|primary|T=0001-01-01T01:00:00Z"
