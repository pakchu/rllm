from __future__ import annotations

import csv
import gzip
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from training import federal_liquidity_component_concordance_clock as clock


def _rows(count: int = 114, *, shock_index: int = 112) -> list[clock.SourceRow]:
    output: list[clock.SourceRow] = []
    assets = 8_000_000
    tga = 600_000
    rrp = 2_000_000
    start = date(2020, 1, 2)
    for index in range(count):
        assets += 100
        tga -= 10
        rrp -= 5
        if index == shock_index:
            assets += 10_000
            tga -= 2_000
            rrp -= 1_000
        released = start + timedelta(days=7 * index)
        available = datetime.combine(
            released,
            datetime.min.time(),
            tzinfo=timezone.utc,
        ) + timedelta(hours=21, minutes=35)
        output.append(
            clock.SourceRow(
                release_date=released.isoformat(),
                observation_date=(released - timedelta(days=1)).isoformat(),
                available_at_utc=available.isoformat(),
                total_assets_usd_millions=assets,
                treasury_general_account_usd_millions=tga,
                reverse_repurchase_agreements_usd_millions=rrp,
                net_liquidity_usd_millions=assets - tga - rrp,
            )
        )
    return output


def test_midrank_is_exact_and_excludes_current_value() -> None:
    prior = list(range(clock.PRIOR_LOOKBACK))
    assert clock._midrank_numerator(200, prior) == 2 * clock.PRIOR_LOOKBACK
    assert clock._midrank_numerator(-1, prior) == 0
    assert clock._midrank_numerator(50, prior) == 101
    with pytest.raises(ValueError, match="104 prior"):
        clock._midrank_numerator(1, prior[:-1])


def test_component_concordance_builds_long_extreme_from_prior_only_data() -> None:
    rows = _rows()
    spec = clock.CANDIDATE_SPECS[0]
    features = clock.build_features(rows, spec)
    shocked = next(row for row in features if row.source_index == 112)
    assert shocked.net_rank_numerator == 208
    assert shocked.asset_rank_numerator == 208
    assert shocked.tga_release_rank_numerator == 208
    assert shocked.rrp_release_rank_numerator == 208
    assert shocked.component_breadth == 3
    assert shocked.component_tail_breadth == 3
    assert shocked.side == 1

    changed_future = rows.copy()
    future = changed_future[-1]
    changed_future[-1] = clock.SourceRow(
        **{
            **future.__dict__,
            "total_assets_usd_millions": future.total_assets_usd_millions + 999_999,
            "net_liquidity_usd_millions": future.net_liquidity_usd_millions
            + 999_999,
        }
    )
    rebuilt = clock.build_features(changed_future, spec)
    assert next(row for row in rebuilt if row.source_index == 112) == shocked


def test_primary_controls_delay_and_random_side_are_frozen() -> None:
    rows = _rows()
    spec = clock.CANDIDATE_SPECS[0]
    clocks = clock.build_raw_events(rows, spec)
    primary = next(
        event for event in clocks["primary"] if event.feature_release_date == rows[112].release_date
    )
    flipped = next(
        event
        for event in clocks["direction_flip"]
        if event.feature_release_date == rows[112].release_date
    )
    delayed = next(
        event
        for event in clocks["one_release_delay"]
        if event.feature_release_date == rows[112].release_date
    )
    random_event = next(
        event
        for event in clocks["random_side"]
        if event.feature_release_date == rows[112].release_date
    )
    assert primary.side == 1
    assert flipped.side == -1
    assert delayed.signal_release_date == rows[113].release_date
    assert datetime.fromisoformat(primary.entry_time) - datetime.fromisoformat(
        primary.signal_time
    ) == timedelta(minutes=5)
    assert datetime.fromisoformat(primary.exit_time) - datetime.fromisoformat(
        primary.entry_time
    ) == timedelta(days=5)
    assert random_event.side == clock._random_side(spec, clock.build_features(rows, spec)[-2])


def test_nonoverlap_drops_colliding_event() -> None:
    rows = _rows(count=115)
    spec = clock.CANDIDATE_SPECS[0]
    feature = clock.build_features(rows, spec)[-1]
    first = clock._event(spec, "primary", feature, rows[-1], side=1)
    colliding_signal = clock.SourceRow(
        **{
            **rows[-1].__dict__,
            "release_date": (date.fromisoformat(rows[-1].release_date) + timedelta(days=2)).isoformat(),
            "available_at_utc": (
                datetime.fromisoformat(rows[-1].available_at_utc) + timedelta(days=2)
            ).isoformat(),
        }
    )
    second = clock._event(spec, "primary", feature, colliding_signal, side=-1)
    assert clock.reserve_nonoverlap([second, first]) == [first]


def test_event_ledger_is_byte_deterministic(tmp_path: Path) -> None:
    rows = _rows(count=116)
    events = clock.build_all_events(rows)
    path = tmp_path / "events.csv.gz"
    first = clock.write_event_ledger(path, events)
    first_bytes = path.read_bytes()
    second = clock.write_event_ledger(path, events)
    assert second == first
    assert path.read_bytes() == first_bytes
    assert clock.read_event_ledger(path) == events


def test_read_source_rejects_identity_failure_and_2024(tmp_path: Path) -> None:
    path = tmp_path / "source.csv.gz"
    fieldnames = [
        "release_date",
        "observation_date",
        "available_at_utc",
        "total_assets_usd_millions",
        "treasury_general_account_usd_millions",
        "reverse_repurchase_agreements_usd_millions",
        "net_liquidity_usd_millions",
    ]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "release_date": "2024-01-04",
                "observation_date": "2024-01-03",
                "available_at_utc": "2024-01-04T21:35:00+00:00",
                "total_assets_usd_millions": 100,
                "treasury_general_account_usd_millions": 20,
                "reverse_repurchase_agreements_usd_millions": 10,
                "net_liquidity_usd_millions": 71,
            }
        )
    with pytest.raises(ValueError, match="identity failed"):
        clock.read_source(path)

    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "release_date": "2024-01-04",
                "observation_date": "2024-01-03",
                "available_at_utc": "2024-01-04T21:35:00+00:00",
                "total_assets_usd_millions": 100,
                "treasury_general_account_usd_millions": 20,
                "reverse_repurchase_agreements_usd_millions": 10,
                "net_liquidity_usd_millions": 70,
            }
        )
    with pytest.raises(ValueError, match=r"2024\+"):
        clock.read_source(path)
