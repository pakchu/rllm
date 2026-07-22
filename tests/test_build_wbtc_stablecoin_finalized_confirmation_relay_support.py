from __future__ import annotations

from datetime import datetime, timedelta, timezone
import gzip
from pathlib import Path

import pytest

from training import build_wbtc_stablecoin_finalized_confirmation_relay_support as support
from training import preregister_wbtc_stablecoin_finalized_confirmation_relay as prereg


UTC = timezone.utc


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def event(
    *,
    source: str,
    asset: str,
    name: str,
    sign: int,
    amount: int,
    available: str,
    identity: tuple[int, int, int],
    actor: str = "",
) -> support.SourceEvent:
    return support.SourceEvent(
        source=source,
        asset=asset,
        event=name,
        sign=sign,
        amount_raw=amount,
        available_at=dt(available),
        identity=identity,
        actor=actor,
    )


def wbtc_batch(
    available: str = "2021-01-01T00:00:00Z", net: int = 60
) -> support.FlowBatch:
    sign = 1 if net > 0 else -1
    rows = [
        event(
            source="wbtc",
            asset="wbtc_eth",
            name="mint" if sign > 0 else "burn",
            sign=sign,
            amount=abs(net),
            available=available,
            identity=(100, 0, 0),
            actor="0x0000000000000000000000000000000000000001",
        )
    ]
    return support.batch_events(rows, source="wbtc", scope="wbtc")[0]


def stable_batch(available: str, net: int, identity: int) -> support.FlowBatch:
    sign = 1 if net > 0 else -1
    rows = [
        event(
            source="stablecoin",
            asset="usdc_eth",
            name="mint" if sign > 0 else "burn",
            sign=sign,
            amount=abs(net),
            available=available,
            identity=(200 + identity, 0, identity),
        )
    ]
    return support.stablecoin_batches(rows)[0]


def test_atomic_batches_net_same_timestamp_without_intra_batch_passage() -> None:
    rows = [
        event(
            source="wbtc",
            asset="wbtc_eth",
            name="mint",
            sign=1,
            amount=100,
            available="2021-01-01T00:00:00Z",
            identity=(100, 0, 2),
            actor="0x0000000000000000000000000000000000000001",
        ),
        event(
            source="wbtc",
            asset="wbtc_eth",
            name="burn",
            sign=-1,
            amount=40,
            available="2021-01-01T00:00:00Z",
            identity=(100, 0, 1),
            actor="0x0000000000000000000000000000000000000002",
        ),
    ]
    batch = support.batch_events(rows, source="wbtc", scope="wbtc")[0]
    reversed_batch = support.batch_events(
        list(reversed(rows)), source="wbtc", scope="wbtc"
    )[0]
    assert batch.net_raw == 60
    assert batch.gross_raw == 140
    assert batch.rows == 2
    assert batch.actors == (
        "0x0000000000000000000000000000000000000001",
        "0x0000000000000000000000000000000000000002",
    )
    assert batch.identity_hash == reversed_batch.identity_hash

    stable_rows = [
        event(
            source="stablecoin",
            asset="usdc_eth",
            name="mint",
            sign=1,
            amount=30,
            available="2021-01-01T02:00:00Z",
            identity=(200, 0, 0),
        ),
        event(
            source="stablecoin",
            asset="usdt_eth",
            name="redeem",
            sign=-1,
            amount=50,
            available="2021-01-01T02:00:00Z",
            identity=(201, 0, 0),
        ),
    ]
    stable = support.stablecoin_batches(stable_rows)
    assert len(stable) == 1
    assert stable[0].net_raw == -20


def test_first_passage_uses_strict_lower_inclusive_upper_and_cumulative_path() -> None:
    anchor = wbtc_batch()
    stable = [
        stable_batch("2021-01-01T00:00:00Z", 100, 0),
        stable_batch("2021-01-01T02:00:00Z", -30, 1),
        stable_batch("2021-01-01T03:00:00Z", 20, 2),
        stable_batch("2021-01-01T04:00:00Z", 15, 3),
        stable_batch("2021-01-01T12:00:01Z", 1000, 4),
    ]
    raw = support.confirmation_candidates(
        [anchor], stable, control="primary", relation="same"
    )
    assert len(raw) == 1
    assert raw[0].signal_time == dt("2021-01-01T04:00:00Z")
    assert raw[0].cumulative_net_raw == 5
    assert raw[0].stablecoin_batches == 3
    assert raw[0].side == 1

    upper = support.confirmation_candidates(
        [anchor],
        [stable_batch("2021-01-01T12:00:00Z", 1, 5)],
        control="primary",
        relation="same",
    )
    assert upper[0].signal_time == dt("2021-01-01T12:00:00Z")


def test_entry_latency_is_ceil_grid_plus_one_full_bar() -> None:
    assert support.ceil_5m_plus_one_bar(dt("2021-01-01T04:00:00Z")) == dt(
        "2021-01-01T04:05:00Z"
    )
    assert support.ceil_5m_plus_one_bar(dt("2021-01-01T04:00:01Z")) == dt(
        "2021-01-01T04:10:00Z"
    )
    assert support.ceil_5m_plus_one_bar(dt("2021-01-01T04:04:59Z")) == dt(
        "2021-01-01T04:10:00Z"
    )


def _raw(at: str, identity: str, *, side: int = 1) -> support.RawCandidate:
    signal = dt(at)
    entry = support.ceil_5m_plus_one_bar(signal)
    return support.RawCandidate(
        control="primary",
        anchor_time=signal - timedelta(hours=1),
        signal_time=signal,
        entry_time=entry,
        exit_time=entry + timedelta(hours=72),
        side=side,
        wbtc=wbtc_batch((signal - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")),
        confirmation_identity=identity,
        stablecoin_scope="combined",
        cumulative_net_raw=1,
        cumulative_gross_raw=1,
        stablecoin_batches=1,
        usdc_net_raw=1,
        usdc_gross_raw=1,
        usdt_net_raw=0,
        usdt_gross_raw=0,
    )


def test_scheduler_enforces_nonoverlap_reuse_and_split_containment() -> None:
    first = _raw("2021-01-01T04:00:00Z", "a")
    overlapping = _raw("2021-01-03T00:00:00Z", "b")
    exact_exit = _raw("2021-01-04T04:00:00Z", "c")
    reused = _raw("2021-01-08T04:00:00Z", "c")
    crossing = _raw("2022-12-31T23:00:00Z", "d")
    scheduled = support.schedule(
        [crossing, reused, overlapping, exact_exit, first], "primary"
    )
    assert [row.confirmation_identity for row in scheduled] == ["a", "c"]
    assert scheduled[1].entry_time == scheduled[0].exit_time
    assert all(row.window == "train" for row in scheduled)


def test_black_funds_veto_is_causal_and_same_time_veto_wins() -> None:
    anchor = wbtc_batch()
    stable = [stable_batch("2021-01-01T04:00:00Z", 1, 1)]
    before = support.confirmation_candidates(
        [anchor],
        stable,
        control="black_funds_veto",
        relation="same",
        veto_times=[dt("2021-01-01T03:00:00Z")],
    )
    same_time = support.confirmation_candidates(
        [anchor],
        stable,
        control="black_funds_veto",
        relation="same",
        veto_times=[dt("2021-01-01T04:00:00Z")],
    )
    after = support.confirmation_candidates(
        [anchor],
        stable,
        control="black_funds_veto",
        relation="same",
        veto_times=[dt("2021-01-01T05:00:00Z")],
    )
    assert before == []
    assert same_time == []
    assert len(after) == 1


def test_random_side_control_is_deterministic_and_preserves_multiset() -> None:
    rows = [
        support.Candidate(**_raw(f"2021-01-{1 + i * 4:02d}T00:00:00Z", str(i), side=side).__dict__, window="train")
        for i, side in enumerate((1, 1, -1, 1, -1))
    ]
    first = support.deterministic_random_sides(rows)
    second = support.deterministic_random_sides(rows)
    assert first == second
    assert sorted(first) == sorted(row.side for row in rows)


def test_novelty_metrics_are_direction_aware_and_thresholded() -> None:
    primary = [
        support.Candidate(**_raw(time, str(i), side=side).__dict__, window="train")
        for i, (time, side) in enumerate(
            (
                ("2021-01-01T00:00:00Z", 1),
                ("2021-01-05T00:00:00Z", -1),
                ("2021-01-09T00:00:00Z", 1),
            )
        )
    ]
    comparator_entries = tuple(
        support.ComparatorEntry(
            entry_time=(
                primary[index].entry_time
                if index < 3
                else dt(f"2021-02-{index:02d}T00:05:00Z")
            ),
            side=(primary[index].side if index < 2 else -1),
        )
        for index in range(1, 11)
    )
    view = support.ComparatorView(
        name="synthetic",
        start=dt("2021-01-01T00:00:00Z"),
        end=dt("2022-01-01T00:00:00Z"),
        entries=comparator_entries,
    )
    report, checks = support.novelty_report(primary, {"synthetic": view})
    assert report["synthetic"]["gate_eligible"] is True
    assert "same_side" in report["synthetic"]
    assert checks["novelty:synthetic"] is False


def test_stablecoin_loader_stops_at_seal_before_value_decoding(tmp_path: Path) -> None:
    path = tmp_path / "stable.csv.gz"
    valid = {column: "" for column in prereg.STABLECOIN_HEADER}
    valid.update(
        {
            "asset": "usdc_eth",
            "event": "mint",
            "event_sign": "1",
            "amount_raw": "10",
            "decimals": "6",
            "block_number": "200",
            "transaction_index": "0",
            "log_index": "0",
            "confirmation_block_number": "264",
            "available_at": "2023-12-31T23:59:59Z",
        }
    )
    sealed = dict(valid)
    sealed.update(
        {
            "amount_raw": "not-an-int",
            "available_at": "2024-01-01T00:00:00Z",
        }
    )
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = support.csv.DictWriter(handle, fieldnames=prereg.STABLECOIN_HEADER)
        writer.writeheader()
        writer.writerow(valid)
        writer.writerow(sealed)
    directional, veto, audit = support.load_stablecoin_events(
        path,
        expected_rows_before_seal=1,
        expected_directional_rows=1,
        expected_veto_rows=0,
    )
    assert len(directional) == 1
    assert veto == []
    assert audit["boundary_sentinel_timestamp_rows_scanned"] == 1
    assert audit["sealed_non_timestamp_fields_decoded"] == 0
    assert audit["post_2023_contract_event_value_rows_loaded"] == 0


def test_deterministic_gzip_and_write_once_drift(tmp_path: Path) -> None:
    row = {column: "" for column in support.CLOCK_COLUMNS}
    row.update({"candidate": support.CANDIDATE, "control": "primary"})
    first = support.deterministic_gzip_csv([row])
    second = support.deterministic_gzip_csv([row])
    assert first == second
    forbidden = {"price", "return", "pnl", "funding", "cagr", "drawdown"}
    assert not forbidden.intersection(support.CLOCK_COLUMNS)
    output = tmp_path / "clock.csv.gz"
    assert support._write_once(output, first) == "created"
    assert support._write_once(output, first) == "verified_existing"
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        support._write_once(output, first + b"drift")
