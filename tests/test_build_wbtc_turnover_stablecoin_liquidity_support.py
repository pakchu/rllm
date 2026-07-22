from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import gzip
import hashlib

from training import build_wbtc_turnover_stablecoin_liquidity_support as support


UTC = timezone.utc


def _event(
    source: str,
    asset: str,
    event: str,
    sign: int,
    amount: int,
    when: datetime,
    identity: int,
    actor: int = 1,
) -> support.SourceEvent:
    return support.SourceEvent(
        source=source,
        asset=asset,
        event=event,
        sign=sign,
        amount_raw=amount,
        available_at=when,
        identity=(identity, 0, 0),
        actor="0x" + f"{actor:040x}" if source == "wbtc" else "",
    )


def _stablecoin_row(**overrides: str) -> dict[str, str]:
    row = {column: "" for column in support.prereg.STABLECOIN_HEADER}
    row.update(
        {
            "asset": "usdc_eth",
            "event": "mint",
            "event_sign": "1",
            "amount_raw": "10",
            "decimals": "6",
            "block_number": "1",
            "transaction_index": "0",
            "log_index": "0",
            "available_at": "2023-12-31T23:59:59Z",
        }
    )
    row.update(overrides)
    return row


def test_window_is_lower_exclusive_and_cutoff_inclusive() -> None:
    cutoff = datetime(2022, 1, 1, tzinfo=UTC)
    events = [
        _event("stablecoin", "usdc_eth", "mint", 1, 10, cutoff - timedelta(hours=168), 1),
        _event(
            "stablecoin",
            "usdc_eth",
            "mint",
            1,
            20,
            cutoff - timedelta(hours=168) + timedelta(seconds=1),
            2,
        ),
        _event("stablecoin", "usdc_eth", "burn", -1, 5, cutoff, 3),
        _event(
            "stablecoin",
            "usdc_eth",
            "mint",
            1,
            99,
            cutoff + timedelta(seconds=1),
            4,
        ),
    ]
    state = support.FlowIndex(events).basic(cutoff, source="stablecoin")
    assert state.rows == 2
    assert state.net_raw == 15
    assert state.gross_raw == 25


def test_stablecoin_imbalance_and_black_funds_veto() -> None:
    cutoff = datetime(2022, 1, 1, tzinfo=UTC)
    flows = support.FlowIndex(
        [
            _event("stablecoin", "usdc_eth", "mint", 1, 60, cutoff, 1),
            _event("stablecoin", "usdt_eth", "redeem", -1, 40, cutoff, 2),
        ]
    )
    veto = support.VetoIndex(
        [
            _event(
                "stablecoin",
                "usdt_eth",
                "destroyed_black_funds",
                -1,
                1,
                cutoff,
                3,
            )
        ]
    )
    blocked = support._stable_state(
        flows, veto, cutoff, scope="combined", apply_veto=True
    )
    assert blocked.net_raw == 20
    assert blocked.gross_raw == 100
    assert blocked.veto_rows == 1
    assert blocked.valid is False

    allowed = support._stable_state(
        flows, veto, cutoff, scope="combined", apply_veto=False
    )
    assert allowed.valid is True
    assert allowed.veto_rows == 0


def test_stablecoin_direction_never_uses_wbtc_net_sign() -> None:
    cutoff = datetime(2022, 1, 1, tzinfo=UTC)
    wbtc = support.FlowState(
        "wbtc",
        cutoff,
        net_raw=-100,
        gross_raw=100,
        rows=2,
        actors=("a", "b"),
        top_actor_share=0.5,
        valid=True,
    )
    stable = support.FlowState(
        "combined", cutoff, net_raw=10, gross_raw=100, rows=2, valid=True
    )
    empty = support.FlowState("usdc", cutoff, 0, 0, 0)
    state = support.AnchorState(
        cutoff + timedelta(hours=6),
        cutoff,
        "combined",
        wbtc,
        stable,
        empty,
        empty,
    )
    assert support.state_side(state, "primary") == 1
    assert support.state_side(state, "wbtc_signed_placebo") == -1


def test_scheduler_uses_ten_minute_entry_and_global_nonoverlap() -> None:
    base = datetime(2022, 1, 1, tzinfo=UTC)
    valid_wbtc = support.FlowState(
        "wbtc", base, 1, 2, 2, ("a", "b"), 0.5, 0, 0, True
    )
    valid_stable = support.FlowState(
        "combined", base, 1, 2, 2, (), 0.0, 0, 0, True
    )
    empty = support.FlowState("usdc", base, 0, 0, 0)
    states = [
        support.AnchorState(
            base + timedelta(hours=hours),
            base + timedelta(hours=hours - 6),
            "combined",
            valid_wbtc,
            valid_stable,
            empty,
            empty,
        )
        for hours in (0, 6, 24, 30, 48)
    ]
    rows = support.schedule_states(states, "primary")
    assert [row.entry_time for row in rows] == [
        base + timedelta(minutes=10),
        base + timedelta(hours=24, minutes=10),
        base + timedelta(hours=48, minutes=10),
    ]
    assert all(row.exit_time - row.entry_time == timedelta(hours=24) for row in rows)


def test_stablecoin_loader_stops_before_sealed_event_values(tmp_path) -> None:
    source = tmp_path / "stablecoin.csv.gz"
    rows = [
        _stablecoin_row(),
        _stablecoin_row(
            asset="usdt_eth",
            event="destroyed_black_funds",
            event_sign="-1",
            block_number="2",
            log_index="1",
        ),
        _stablecoin_row(
            amount_raw="not-an-integer",
            block_number="3",
            log_index="2",
            available_at="2024-01-01T00:00:00Z",
        ),
    ]
    with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=support.prereg.STABLECOIN_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    directional, veto, audit = support.load_stablecoin_events(
        source,
        expected_rows_before_seal=2,
        expected_directional_rows=1,
        expected_veto_rows=1,
    )
    assert len(directional) == len(veto) == 1
    assert audit["last_available_at"] == "2023-12-31T23:59:59Z"
    assert audit["boundary_sentinel_timestamp_rows_scanned"] == 1
    assert audit["post_2023_contract_event_value_rows_loaded"] == 0


def test_amount_permutation_preserves_group_multisets() -> None:
    base = datetime(2021, 1, 1, tzinfo=UTC)
    events = [
        _event("wbtc", "wbtc_eth", event, sign, amount, base + timedelta(days=i), i, i)
        for i, (event, sign, amount) in enumerate(
            (("mint", 1, 10), ("mint", 1, 20), ("burn", -1, 30), ("burn", -1, 40)),
            1,
        )
    ]
    permuted = support.permute_amounts(events)
    before = Counter((row.event, row.available_at.year, row.amount_raw) for row in events)
    after = Counter((row.event, row.available_at.year, row.amount_raw) for row in permuted)
    assert before == after
    assert [row.identity for row in permuted] == [row.identity for row in events]


def test_deterministic_gzip() -> None:
    row = {column: "0" for column in support.CLOCK_COLUMNS}
    first = support.deterministic_gzip_csv([row])
    second = support.deterministic_gzip_csv([row])
    assert hashlib.sha256(first).digest() == hashlib.sha256(second).digest()
