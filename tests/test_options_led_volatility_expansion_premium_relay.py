from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from training import options_led_volatility_expansion_premium_relay as ovepr


UTC = timezone.utc
START = datetime(2023, 6, 1, tzinfo=UTC)


def candle(open_: str, high: str, low: str, close: str) -> ovepr.VolatilityCandle:
    return ovepr.VolatilityCandle.from_tokens(
        open=open_, high=high, low=low, close=close
    )


FLAT = candle("100", "101", "99", "100")
B_UP = candle("100", "111", "99", "110")
D_UP_LEAD = candle("100", "121", "99", "120")
D_UP_LAG = candle("100", "106", "99", "105")
B_DOWN = candle("100", "101", "89", "90")
D_DOWN_LEAD = candle("100", "101", "79", "80")


def hour(
    offset: int,
    *,
    bvol: ovepr.VolatilityCandle | None = FLAT,
    dvol: ovepr.VolatilityCandle | None = FLAT,
    move: str = "1",
    path_range: str = "2",
    available_delays: tuple[timedelta, timedelta, timedelta] = (
        timedelta(0),
        timedelta(0),
        timedelta(0),
    ),
    valid: bool = True,
) -> ovepr.JoinedHour:
    close_time = START + offset * timedelta(hours=1)
    premium = ovepr.PremiumPathHour.from_tokens(
        move=move, path_range=path_range
    )
    return ovepr.JoinedHour(
        close_time=close_time,
        bvol_available_at=close_time + available_delays[0],
        dvol_available_at=close_time + available_delays[1],
        premium_available_at=close_time + available_delays[2],
        bvol=bvol,
        dvol=dvol,
        premium=premium,
        source_valid=valid,
    )


def warmup() -> list[ovepr.JoinedHour]:
    return [hour(index) for index in range(672)]


def test_primary_uses_strict_normalized_body_lead_and_follow_side() -> None:
    rows = warmup()
    rows.extend(
        [
            hour(672, bvol=B_UP, dvol=D_UP_LEAD, move="2", path_range="4"),
            hour(673, bvol=FLAT, dvol=FLAT),
            hour(674, bvol=B_UP, dvol=D_UP_LEAD, move="-2", path_range="4"),
        ]
    )
    events = ovepr.raw_candidates(rows)
    assert [(event.signal_time, event.side) for event in events] == [
        (START + timedelta(hours=672), "LONG"),
        (START + timedelta(hours=674), "SHORT"),
    ]

    tied = warmup() + [hour(672, bvol=B_UP, dvol=B_UP)]
    assert ovepr.raw_candidates(tied) == ()


def test_efficiency_uses_strictly_prior_720h_median_and_min_672() -> None:
    too_short = [hour(index) for index in range(671)]
    too_short.append(
        hour(671, bvol=B_UP, dvol=D_UP_LEAD, move="3", path_range="4")
    )
    assert ovepr.raw_candidates(too_short) == ()

    rows = warmup()
    rows.append(hour(672, bvol=B_UP, dvol=D_UP_LEAD, move="1", path_range="2"))
    assert ovepr.prior_efficiency_median(rows, 672) == ovepr.Fraction(1, 2)
    assert len(ovepr.raw_candidates(rows)) == 1

    below = warmup()
    below.append(hour(672, bvol=B_UP, dvol=D_UP_LEAD, move="1", path_range="3"))
    assert ovepr.raw_candidates(below) == ()


def test_only_false_to_true_onset_emits_and_gap_or_invalid_resets() -> None:
    rows = warmup()
    rows.extend(
        [
            hour(672, bvol=B_UP, dvol=D_UP_LEAD),
            hour(673, bvol=B_UP, dvol=D_UP_LEAD, move="-1"),
            hour(674),
            hour(675, bvol=B_UP, dvol=D_UP_LEAD),
        ]
    )
    assert [event.signal_time for event in ovepr.raw_candidates(rows)] == [
        START + timedelta(hours=672),
        START + timedelta(hours=675),
    ]

    invalid = warmup()
    invalid[-1] = hour(671, valid=False)
    invalid.append(hour(672, bvol=B_UP, dvol=D_UP_LEAD))
    assert ovepr.raw_candidates(invalid) == ()

    gap = warmup()[:-1]
    gap.append(hour(672, bvol=B_UP, dvol=D_UP_LEAD))
    assert ovepr.raw_candidates(gap) == ()


def test_independent_control_clocks_are_exact() -> None:
    no_lead = warmup() + [hour(672, bvol=B_UP, dvol=D_UP_LAG)]
    assert ovepr.raw_candidates(no_lead) == ()
    assert len(ovepr.raw_candidates(no_lead, ovepr.NO_DERIBIT_LEAD)) == 1

    mirror = warmup() + [hour(672, bvol=B_DOWN, dvol=D_DOWN_LEAD, move="-1")]
    mirror_events = ovepr.raw_candidates(mirror, ovepr.DERIBIT_FALL_MIRROR)
    assert [event.side for event in mirror_events] == ["SHORT"]

    inefficient = warmup() + [
        hour(672, bvol=B_UP, dvol=D_UP_LEAD, move="1", path_range="3")
    ]
    assert ovepr.raw_candidates(inefficient) == ()
    assert len(
        ovepr.raw_candidates(inefficient, ovepr.NO_PREMIUM_EFFICIENCY)
    ) == 1


def test_availability_entry_hold_and_global_nonoverlap_are_fixed() -> None:
    rows = warmup()
    rows.extend(
        [
            hour(
                672,
                bvol=B_UP,
                dvol=D_UP_LEAD,
                available_delays=(
                    timedelta(seconds=1),
                    timedelta(minutes=2),
                    timedelta(seconds=3),
                ),
            ),
            hour(673),
            hour(674, bvol=B_UP, dvol=D_UP_LEAD),
            hour(675),
            *[hour(index) for index in range(676, 697)],
            hour(697, bvol=B_UP, dvol=D_UP_LEAD),
        ]
    )
    raw = ovepr.raw_candidates(rows)
    first = raw[0]
    assert first.available_at == first.signal_time + timedelta(minutes=2)
    assert first.entry_time == first.signal_time + timedelta(minutes=5)
    assert first.exit_time - first.entry_time == timedelta(hours=24)
    accepted = ovepr.reserve_nonoverlap(raw)
    assert [event.signal_time for event in accepted] == [
        START + timedelta(hours=672),
        START + timedelta(hours=697),
    ]


def test_candidate_is_suppressed_when_joint_feature_misses_t_plus_5m() -> None:
    rows = warmup() + [
        hour(
            672,
            bvol=B_UP,
            dvol=D_UP_LEAD,
            available_delays=(
                timedelta(seconds=1),
                timedelta(minutes=5),
                timedelta(seconds=3),
            ),
        )
    ]
    assert ovepr.raw_candidates(rows) == ()


def test_parent_controls_preserve_primary_set_and_only_frozen_dimension() -> None:
    primary = ovepr.primary_schedule(
        warmup() + [hour(672, bvol=B_UP, dvol=D_UP_LEAD)]
    )
    flipped = ovepr.parent_control_schedule(primary, ovepr.DIRECTION_FLIP)
    delayed = ovepr.parent_control_schedule(primary, ovepr.EXTRA_LATENCY_1H)
    random_a = ovepr.parent_control_schedule(primary, ovepr.DETERMINISTIC_RANDOM_SIDE)
    random_b = ovepr.parent_control_schedule(primary, ovepr.DETERMINISTIC_RANDOM_SIDE)
    assert flipped[0].side == "SHORT"
    assert delayed[0].entry_time == primary[0].entry_time + timedelta(hours=1)
    assert delayed[0].exit_time == primary[0].exit_time + timedelta(hours=1)
    assert random_a == random_b
    assert [event.signal_time for event in flipped] == [
        event.signal_time for event in primary
    ]


def test_exact_decimal_and_completed_premium_validation_fail_closed() -> None:
    with pytest.raises(ValueError, match="exact decimal token"):
        ovepr.VolatilityCandle.from_tokens(
            open=100.0, high="101", low="99", close="100"
        )
    incomplete = ovepr.PremiumPathHour.from_tokens(
        move="1", path_range="2", source_rows=59
    )
    assert incomplete.valid is False
    zero_range = ovepr.PremiumPathHour.from_tokens(move="0", path_range="0")
    assert zero_range.valid is False


def test_mechanism_module_has_no_io_or_market_outcome_dependencies() -> None:
    path = Path("training/options_led_volatility_expansion_premium_relay.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported.isdisjoint(
        {"pathlib", "os", "subprocess", "pandas", "numpy", "requests"}
    )
