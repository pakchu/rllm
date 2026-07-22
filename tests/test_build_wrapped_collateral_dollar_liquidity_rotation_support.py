from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import csv
import gzip
import hashlib

from training import (
    build_wrapped_collateral_dollar_liquidity_rotation_support as support,
)


UTC = timezone.utc


def _stablecoin_row(**overrides: str) -> dict[str, str]:
    row = {column: "" for column in support.prereg.STABLECOIN_HEADER}
    row.update(
        {
            "asset": "usdc_eth",
            "event": "mint",
            "event_sign": "1",
            "amount_raw": "10",
            "indexed_address_1": "0x" + "1" * 40,
            "block_number": "1",
            "transaction_index": "0",
            "log_index": "0",
            "available_at": "2023-12-31T23:59:59Z",
        }
    )
    row.update(overrides)
    return row


def _event(
    source: str,
    day: int,
    sign: int,
    amount: int,
    identity: int,
    actor: int = 1,
) -> support.SourceEvent:
    return support.SourceEvent(
        source=source,
        event="mint" if sign == 1 else "burn",
        sign=sign,
        amount_raw=amount,
        available_at=datetime(2021, 1, day, tzinfo=UTC),
        identity=(identity, 0, 0),
        actor="0x" + f"{actor:040x}" if source == "wbtc" else "",
    )


def test_window_is_lower_exclusive_and_cutoff_inclusive() -> None:
    cutoff = datetime(2021, 2, 1, tzinfo=UTC)
    events = [
        support.SourceEvent(
            "usdc", "mint", 1, 10, cutoff - timedelta(days=7), (1, 0, 0)
        ),
        support.SourceEvent(
            "usdc",
            "mint",
            1,
            20,
            cutoff - timedelta(days=7) + timedelta(seconds=1),
            (2, 0, 0),
        ),
        support.SourceEvent("usdc", "burn", -1, 5, cutoff, (3, 0, 0)),
        support.SourceEvent(
            "usdc", "mint", 1, 99, cutoff + timedelta(seconds=1), (4, 0, 0)
        ),
    ]
    state = support.EventIndex(events).aggregate(cutoff, 7, source="usdc")
    assert state.rows == 2
    assert state.net_raw == 15
    assert state.gross_raw == 25


def test_usdc_loader_stops_before_parsing_sealed_event_values(tmp_path) -> None:
    source = tmp_path / "stablecoin.csv.gz"
    rows = [
        _stablecoin_row(),
        _stablecoin_row(
            amount_raw="not-an-integer",
            block_number="2",
            available_at="2024-01-01T00:00:00Z",
        ),
    ]
    with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=support.prereg.STABLECOIN_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    events, audit = support.load_usdc_events(
        source,
        expected_rows_before_seal=1,
        expected_eligible_rows=1,
    )

    assert len(events) == 1
    assert events[0].amount_raw == 10
    assert audit["last_available_at"] == "2023-12-31T23:59:59Z"
    assert audit["boundary_sentinel_timestamp_rows_scanned"] == 1
    assert audit["post_2023_contract_event_value_rows_loaded"] == 0


def test_opposite_sign_mapping_and_component_controls() -> None:
    cutoff = datetime(2021, 2, 1, tzinfo=UTC)
    wbtc = support.WindowState(
        "wbtc", cutoff, 30, -10, 20, -1, 3, ("a", "b"), 0.6, True
    )
    usdc = support.WindowState("usdc", cutoff, 7, 50, 100, 1, 30, (), 0.0, True)
    state = support.DailyState(cutoff + timedelta(hours=6), cutoff, wbtc, usdc)
    assert support.state_side(state, "primary") == 1
    assert support.state_side(state, "wbtc_only_contrarian") == 1
    assert support.state_side(state, "usdc_only_direct") == 1
    assert support.state_side(state, "same_sign_direct") == 0
    assert support.state_side(state, "count_sign_consensus") == 1

    reversed_state = support.replace(
        state,
        wbtc=support.replace(wbtc, net_raw=10, count_net=1),
        usdc=support.replace(usdc, net_raw=-50, count_net=-1),
    )
    assert support.state_side(reversed_state, "primary") == -1


def test_wbtc_validity_requires_breadth_and_actor_cap() -> None:
    cutoff = datetime(2021, 2, 1, tzinfo=UTC)
    events = [
        _event("wbtc", 10, 1, 90, 1, actor=1),
        _event("wbtc", 11, -1, 5, 2, actor=2),
        _event("wbtc", 12, -1, 5, 3, actor=2),
    ]
    state = support.EventIndex(events).aggregate(cutoff, 30, source="wbtc")
    assert state.valid is True
    assert state.top_actor_share == 0.9

    concentrated = [support.replace(events[0], amount_raw=91), *events[1:]]
    state = support.EventIndex(concentrated).aggregate(cutoff, 30, source="wbtc")
    assert state.valid is False


def test_scheduler_is_nonoverlapping_and_skips_split_crossing() -> None:
    valid_wbtc = support.WindowState(
        "wbtc",
        datetime(2021, 1, 1, tzinfo=UTC),
        30,
        -1,
        10,
        -1,
        3,
        ("a", "b"),
        0.5,
        True,
    )
    valid_usdc = support.WindowState(
        "usdc", datetime(2021, 1, 1, tzinfo=UTC), 7, 1, 10, 1, 30, (), 0.0, True
    )
    states = []
    for raw in (
        "2021-01-01T00:00:00Z",
        "2021-01-02T00:00:00Z",
        "2021-01-08T00:00:00Z",
        "2022-12-30T00:00:00Z",
        "2023-01-01T00:00:00Z",
    ):
        decision = support.parse_time(raw)
        states.append(
            support.DailyState(
                decision,
                decision - timedelta(hours=6),
                support.replace(valid_wbtc, cutoff=decision - timedelta(hours=6)),
                support.replace(valid_usdc, cutoff=decision - timedelta(hours=6)),
            )
        )
    scheduled = support.schedule(states, "primary")
    assert [support.format_time(row.entry_time) for row in scheduled] == [
        "2021-01-01T00:05:00Z",
        "2021-01-08T00:05:00Z",
        "2023-01-01T00:05:00Z",
    ]
    assert all(
        later.entry_time >= earlier.exit_time
        for earlier, later in zip(scheduled, scheduled[1:])
    )


def test_amount_permutation_preserves_group_multisets() -> None:
    events = [
        _event("wbtc", day, sign, amount, identity, actor=identity)
        for day, sign, amount, identity in (
            (1, 1, 10, 1),
            (2, 1, 20, 2),
            (3, -1, 30, 3),
            (4, -1, 40, 4),
        )
    ]
    permuted = support.permute_amounts(events)
    before = Counter(
        (event.event, event.available_at.year, event.amount_raw) for event in events
    )
    after = Counter(
        (event.event, event.available_at.year, event.amount_raw)
        for event in permuted
    )
    assert before == after
    assert [event.identity for event in permuted] == [
        event.identity for event in events
    ]


def test_random_side_control_preserves_clock_and_side_counts() -> None:
    base_time = datetime(2021, 1, 1, tzinfo=UTC)
    state = support.WindowState(
        "wbtc", base_time, 30, 1, 1, 1, 3, ("a", "b"), 0.5, True
    )
    usdc = support.WindowState("usdc", base_time, 7, -1, 1, -1, 30, (), 0.0, True)
    primary = [
        support.Candidate(
            "primary",
            base_time + timedelta(days=7 * index),
            base_time + timedelta(days=7 * index, hours=-6),
            base_time + timedelta(days=7 * index, minutes=5),
            base_time + timedelta(days=7 * (index + 1), minutes=5),
            side,
            "train",
            state,
            usdc,
        )
        for index, side in enumerate((1, 1, -1, 1, -1, 1))
    ]
    sides = support.deterministic_random_sides(primary)
    control = support.exact_clock_control(primary, "deterministic_random_side", sides)
    assert Counter(sides) == Counter(candidate.side for candidate in primary)
    assert [row.entry_time for row in control] == [row.entry_time for row in primary]
    assert sides == support.deterministic_random_sides(primary)


def test_deterministic_gzip() -> None:
    row = {column: "0" for column in support.CLOCK_COLUMNS}
    first = support.deterministic_gzip_csv([row])
    second = support.deterministic_gzip_csv([row])
    assert hashlib.sha256(first).digest() == hashlib.sha256(second).digest()
    with gzip.open(io := support.io.BytesIO(first), "rt", encoding="utf-8") as handle:
        assert handle.readline().strip().split(",") == list(support.CLOCK_COLUMNS)
    assert io.closed is False
