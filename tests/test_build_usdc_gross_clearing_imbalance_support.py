from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from training import build_usdc_gross_clearing_imbalance_support as support
from training import preregister_usdc_gross_clearing_imbalance as prereg


def _packet(
    stamp: str,
    *,
    mint: int,
    burn: int,
    threshold: int | None = 100,
    history: int = 720,
) -> support.Packet:
    start = support.parse_time(stamp)
    return support.Packet(
        start=start,
        end=start + timedelta(hours=6),
        mint_raw=mint,
        burn_raw=burn,
        prior_gross_q95=threshold,
        prior_history_packets=history,
    )


def test_nearest_rank_uses_the_frozen_definition() -> None:
    assert support.nearest_rank([1, 2, 3, 4], 0.50) == 2
    assert support.nearest_rank([1, 2, 3, 4], 0.95) == 4
    with pytest.raises(ValueError, match="sample is empty"):
        support.nearest_rank([], 0.95)


def test_build_packets_uses_available_at_and_keeps_zero_packets() -> None:
    events = [
        support.SourceEvent(
            event="mint",
            amount_raw=75,
            block_timestamp=support.parse_time("2020-01-01T05:59:00Z"),
            available_at=support.parse_time("2020-01-01T06:01:00Z"),
            block_number=1,
            transaction_index=0,
            log_index=0,
            block_hash="0x" + "1" * 64,
            transaction_hash="0x" + "2" * 64,
        )
    ]
    cfg = replace(
        prereg.FROZEN_CONFIG,
        warmup_start="2020-01-01T00:00:00Z",
        selection_end_exclusive="2020-01-01T18:00:00Z",
    )
    with pytest.raises(ValueError, match="configuration is frozen"):
        support.build_packets(events, cfg)

    packets = support.build_packets(events)
    assert packets[0].gross_raw == 0
    assert packets[1].start == support.parse_time("2020-01-01T06:00:00Z")
    assert packets[1].mint_raw == 75
    assert packets[2].gross_raw == 0


def test_prior_threshold_excludes_the_current_packet() -> None:
    start = support.parse_time("2020-01-01T00:00:00Z")
    packets = [
        support.Packet(
            start + i * timedelta(hours=6), start + (i + 1) * timedelta(hours=6), i, 0
        )
        for i in range(361)
    ]
    enriched = support.attach_prior_thresholds(packets)
    assert enriched[359].prior_gross_q95 is None
    assert enriched[360].prior_history_packets == 360
    expected = support.nearest_rank(list(range(360)), 0.95)
    assert enriched[360].prior_gross_q95 == expected
    assert enriched[360].gross_raw not in list(range(360))


def test_primary_and_source_controls_have_distinct_conjunctions() -> None:
    primary = _packet("2022-01-01T00:00:00Z", mint=180, burn=20)
    assert support.control_active(primary, "primary", prereg.FROZEN_CONFIG)
    assert support.control_active(primary, "no_gross_tail", prereg.FROZEN_CONFIG)
    assert support.control_active(primary, "no_imbalance_floor", prereg.FROZEN_CONFIG)

    below_tail = _packet("2022-01-01T06:00:00Z", mint=70, burn=10)
    assert not support.control_active(below_tail, "primary", prereg.FROZEN_CONFIG)
    assert support.control_active(below_tail, "no_gross_tail", prereg.FROZEN_CONFIG)

    balanced = _packet("2022-01-01T12:00:00Z", mint=110, burn=90)
    assert not support.control_active(balanced, "primary", prereg.FROZEN_CONFIG)
    assert support.control_active(balanced, "no_imbalance_floor", prereg.FROZEN_CONFIG)


def test_schedule_waits_ten_minutes_and_is_globally_nonoverlapping() -> None:
    packets = [
        _packet("2022-01-01T00:00:00Z", mint=180, burn=20),
        _packet("2022-01-01T06:00:00Z", mint=10, burn=190),
        _packet("2022-01-02T00:00:00Z", mint=10, burn=190),
    ]
    signals = support.schedule_signals(packets, "primary")
    assert len(signals) == 2
    assert signals[0].decision_time == packets[0].end
    assert signals[0].entry_time == packets[0].end + timedelta(minutes=10)
    assert signals[0].exit_time == signals[0].entry_time + timedelta(hours=24)
    assert signals[1].entry_time == signals[0].exit_time


def test_stale_control_delays_the_same_source_state_six_hours() -> None:
    packets = [
        _packet("2022-01-01T00:00:00Z", mint=180, burn=20),
        _packet("2022-01-01T06:00:00Z", mint=0, burn=0),
    ]
    stale = support.schedule_signals(packets, "stale_6h")
    assert len(stale) == 1
    assert stale[0].packet == packets[0]
    assert stale[0].decision_time == packets[1].end
    assert stale[0].entry_time == packets[0].end + timedelta(hours=6, minutes=10)


def test_stale_control_uses_only_accepted_primary_sources() -> None:
    packets = [
        _packet("2020-12-31T12:00:00Z", mint=180, burn=20),
        _packet("2022-01-01T00:00:00Z", mint=180, burn=20),
        _packet("2022-01-01T06:00:00Z", mint=10, burn=190),
        _packet("2022-01-02T00:00:00Z", mint=10, burn=190),
    ]
    primary = support.schedule_signals(packets, "primary")
    stale = support.schedule_signals(packets, "stale_6h")
    assert [signal.packet for signal in stale] == [signal.packet for signal in primary]
    assert all(
        delayed.entry_time == original.entry_time + timedelta(hours=6)
        for original, delayed in zip(primary, stale, strict=True)
    )
    assert all(signal.packet.start.year >= 2021 for signal in stale)


def test_scheduler_excludes_clocks_whose_exit_crosses_selection_end() -> None:
    packets = [
        _packet("2023-12-30T00:00:00Z", mint=180, burn=20),
        _packet("2023-12-31T00:00:00Z", mint=180, burn=20),
    ]
    signals = support.schedule_signals(packets, "primary")
    assert len(signals) == 1
    assert signals[0].exit_time <= support.parse_time("2024-01-01T00:00:00Z")


def test_support_gate_fails_side_and_calendar_concentration() -> None:
    start = support.parse_time("2021-01-01T00:00:00Z")
    packet = _packet("2021-01-01T00:00:00Z", mint=180, burn=20)
    signals = [
        support.Signal(
            control="primary",
            packet=packet,
            decision_time=start + timedelta(hours=i),
            entry_time=start + timedelta(hours=i),
            exit_time=start + timedelta(hours=i + 24),
        )
        for i in range(130)
    ]
    stats = support.support_statistics(signals)
    checks = support.evaluate_support_gates(stats)
    assert checks["side_balance_train"] is False
    assert checks["maximum_entry_month_share"] is False


def test_novelty_math_is_symmetric_and_exact() -> None:
    left = [
        support.parse_time("2023-09-01T00:00:00Z"),
        support.parse_time("2023-09-02T00:00:00Z"),
    ]
    right = [
        support.parse_time("2023-09-01T00:00:00Z"),
        support.parse_time("2023-09-03T00:00:00Z"),
    ]
    assert support.exact_jaccard(left, right) == pytest.approx(1 / 3)
    assert support.near_share(left, right, timedelta(hours=1)) == pytest.approx(0.5)
    assert support.near_share(right, left, timedelta(hours=1)) == pytest.approx(0.5)


def test_sealed_comparator_bundle_never_opens_post_2023_rows() -> None:
    entries, audit = support.load_sealed_comparator_entries()
    assert audit == {
        "sealed_comparator_bundle_files_opened": 1,
        "sealed_comparator_rows_parsed": 780,
        "post_2023_comparator_rows_parsed": 0,
        "original_comparator_files_opened": 0,
    }
    assert entries
    assert all(
        entry < support.parse_time("2024-01-01T00:00:00Z")
        for clocks in entries.values()
        for entry in clocks
    )


def test_preregistration_and_source_files_remain_hash_bound() -> None:
    payload = support.validate_preregistration()
    manifest = support.validate_source_inputs()
    assert payload["candidate"] == "UGCI-288"
    assert manifest["output"]["rows"] == support.EXPECTED_SOURCE_ROWS
