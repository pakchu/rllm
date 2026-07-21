from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from training import evaluate_authorized_minter_turnaround_relay_support as amtr


UTC = timezone.utc
MINTER_A = "0x" + "11" * 20
MINTER_B = "0x" + "22" * 20
RECIPIENT = "0x" + "33" * 20


def _event(
    number: int,
    *,
    event: str,
    available_at: datetime,
    amount: int = 1_000_000,
    minter: str = MINTER_A,
    occurrence_offset: timedelta = timedelta(minutes=-15),
    ready: bool = True,
    large: bool = True,
) -> amtr.Event:
    return amtr.Event(
        event=event,
        amount_raw=amount,
        minter=minter,
        mint_to=RECIPIENT if event == "mint" else "",
        available_at=available_at,
        block_timestamp=available_at + occurrence_offset,
        block_number=number,
        transaction_index=0,
        log_index=number,
        block_hash="0x" + f"{number + 1000:064x}",
        transaction_hash="0x" + f"{number + 2000:064x}",
        warmup_ready=ready,
        large=large,
    )


def test_nearest_rank_is_the_frozen_order_statistic() -> None:
    values = list(range(1, 257))
    assert amtr.nearest_rank(values, 0.95) == values[243]


def test_tail_excludes_current_and_same_timestamp_rows() -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    history = [
        _event(
            index,
            event="mint",
            available_at=start + timedelta(hours=index),
            amount=index + 1,
            ready=False,
            large=False,
        )
        for index in range(256)
    ]
    timestamp = start + timedelta(hours=300)
    current = [
        _event(300, event="mint", available_at=timestamp, amount=244),
        _event(301, event="mint", available_at=timestamp, amount=10_000),
    ]
    annotated = amtr.annotate_tail([*history, *current])
    tail = annotated[-2:]
    assert all(event.warmup_ready for event in tail)
    assert all(event.large for event in tail)


def test_fenwick_tail_matches_naive_strict_prior_reference() -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    events = [
        _event(
            index,
            event="mint" if index % 2 == 0 else "burn",
            available_at=start + timedelta(hours=index // 3),
            amount=((index * 37) % 113) + 1,
            ready=False,
            large=False,
        )
        for index in range(900)
    ]
    annotated = amtr.annotate_tail(events)
    for current in annotated:
        prior = sorted(
            event.amount_raw
            for event in events
            if event.event == current.event
            and current.available_at - amtr.LOOKBACK
            <= event.available_at
            < current.available_at
        )
        ready = len(prior) >= amtr.MINIMUM_HISTORY
        expected_large = ready and current.amount_raw >= amtr.nearest_rank(
            prior, amtr.TAIL_QUANTILE
        )
        assert current.warmup_ready is ready
        assert current.large is expected_large


def test_pairing_uses_latest_prior_same_minter_and_fixed_direction() -> None:
    base = datetime(2023, 1, 1, tzinfo=UTC)
    old = _event(1, event="burn", available_at=base, amount=1_000)
    latest = _event(
        2,
        event="burn",
        available_at=base + timedelta(hours=1),
        amount=1_100,
    )
    mint = _event(
        3,
        event="mint",
        available_at=base + timedelta(hours=2),
        amount=1_050,
    )
    pairs = amtr.build_pairs([old, latest, mint], control="primary")
    assert len(pairs) == 1
    assert pairs[0].prior.identity == latest.identity
    assert pairs[0].side == 1


def test_pairing_requires_both_causal_and_occurrence_gap() -> None:
    base = datetime(2023, 1, 1, tzinfo=UTC)
    burn = _event(1, event="burn", available_at=base)
    mint = _event(
        2,
        event="mint",
        available_at=base + timedelta(hours=1),
        occurrence_offset=timedelta(minutes=-50),
    )
    assert amtr.build_pairs([burn, mint], control="primary") == []


def test_cross_minter_control_requires_different_actor() -> None:
    base = datetime(2023, 1, 1, tzinfo=UTC)
    burn = _event(1, event="burn", available_at=base, minter=MINTER_A)
    mint = _event(
        2,
        event="mint",
        available_at=base + timedelta(hours=1),
        minter=MINTER_B,
    )
    assert amtr.build_pairs([burn, mint], control="primary") == []
    cross = amtr.build_pairs([burn, mint], control="cross_minter", same_minter=False)
    assert len(cross) == 1


def test_scheduler_waits_one_full_bar_and_enforces_48h_nonoverlap() -> None:
    base = datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC)
    first = amtr.Pair(
        _event(1, event="burn", available_at=base),
        _event(2, event="mint", available_at=base + timedelta(hours=1)),
        "primary",
    )
    second = amtr.Pair(
        _event(3, event="mint", available_at=base + timedelta(hours=2)),
        _event(4, event="burn", available_at=base + timedelta(hours=3)),
        "primary",
    )
    rows = amtr.schedule_pairs([first, second])
    assert len(rows) == 1
    assert rows[0]["entry_time"] == "2023-01-01T01:05:00Z"
    assert rows[0]["scheduled_exit"] == "2023-01-03T01:05:00Z"


def test_scheduler_tie_breaks_same_entry_by_canonical_pair_identity() -> None:
    base = datetime(2023, 1, 1, tzinfo=UTC)
    higher_identity_earlier_completion = amtr.Pair(
        _event(20, event="burn", available_at=base),
        _event(21, event="mint", available_at=base + timedelta(minutes=1)),
        "primary",
    )
    lower_identity_later_completion = amtr.Pair(
        _event(1, event="burn", available_at=base),
        _event(2, event="mint", available_at=base + timedelta(minutes=4)),
        "primary",
    )
    rows = amtr.schedule_pairs(
        [higher_identity_earlier_completion, lower_identity_later_completion]
    )
    assert len(rows) == 1
    assert rows[0]["pair_id"] == lower_identity_later_completion.pair_id


def test_pair_tie_break_uses_numeric_log_index() -> None:
    base = datetime(2023, 1, 1, tzinfo=UTC)
    log_two = _event(2, event="burn", available_at=base)
    log_ten = replace(log_two, log_index=10)
    mint = _event(20, event="mint", available_at=base + timedelta(hours=1))
    pairs = amtr.build_pairs([log_ten, log_two, mint], control="primary")
    assert len(pairs) == 1
    assert pairs[0].prior.log_index == 2


def test_source_contract_requires_n64_and_finalized_coverage() -> None:
    valid = {
        "dual_replay": {"canonical_replay_equal": True},
        "header_materialization": {"event_block_hash_cross_checked": True},
        "outcome_boundary": {"source_only": True},
        "source_contract": {"confirmation_blocks": 64},
        "source_audit": {
            "finalized_coverage": {
                "observed_finalized_block_at_least_required": True,
                "required_through_block": 100,
            }
        },
    }
    amtr._validate_source_contract(valid)
    wrong_delay = deepcopy(valid)
    wrong_delay["source_contract"]["confirmation_blocks"] = 63
    with pytest.raises(RuntimeError, match=r"N\+64"):
        amtr._validate_source_contract(wrong_delay)
    no_finality = deepcopy(valid)
    no_finality["source_audit"]["finalized_coverage"][
        "observed_finalized_block_at_least_required"
    ] = False
    with pytest.raises(RuntimeError, match="finalized coverage"):
        amtr._validate_source_contract(no_finality)


def test_summary_applies_all_frozen_support_gates() -> None:
    rows = []
    start = datetime(2021, 1, 1, tzinfo=UTC)
    minters = [MINTER_A, MINTER_B, "0x" + "44" * 20, "0x" + "55" * 20, "0x" + "66" * 20]
    recipients = [RECIPIENT, "0x" + "77" * 20, "0x" + "88" * 20]
    for index in range(60):
        year = 2021 + index % 3
        entry = start.replace(year=year) + timedelta(days=(index * 17) % 330)
        rows.append(
            {
                "side": 1 if index % 2 == 0 else -1,
                "entry_time": amtr._format_time(entry),
                "minter": minters[index % len(minters)],
                "mint_to": recipients[index % len(recipients)],
            }
        )
    summary = amtr.summarize(rows)
    assert summary["events"] == 60
    assert summary["distinct_minters"] == 5
    assert all(summary["checks"].values())
