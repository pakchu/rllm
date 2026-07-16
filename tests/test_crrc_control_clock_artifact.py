from __future__ import annotations

import json
from pathlib import Path

from training import qualify_cross_venue_radial_refill_compression as qualify


PATH = Path(
    "results/cross_venue_radial_refill_compression_control_clocks_2026-07-17.json"
)


def test_control_clock_freeze_is_outcome_blind() -> None:
    payload = json.loads(PATH.read_text())
    assert payload["outcomes_opened"] is False
    assert payload["price_funding_return_pnl_or_equity_loaded"] is False
    assert payload["controls_are_diagnostics_not_repair_candidates"] is True
    assert payload["clock"] == {
        "entry_delay_bars": 2,
        "hold_bars": 72,
        "quarter_contained": True,
    }


def test_control_counts_and_hashes_are_frozen() -> None:
    payload = json.loads(PATH.read_text())
    expected = {
        "um_only": (714, 382, 332, "bb38368e4cd5e87436917c394a0416035b541f0494d017a9ca2cfa9adfe9dee2"),
        "cm_only": (681, 360, 321, "70cfc2b6251d299f804113cbdc222a57d41015853fbc13200bceca33c7a78310"),
        "without_credibility": (637, 325, 312, "dddc0514e1e80a947cdda5a1db53e9ddd010687c6a50786a2aab4ec5b7c5a7fe"),
        "inner_add_only": (822, 408, 414, "a5f95711df5da090d564f85a69f77a7108a3251313469c6000c86858d5baefcc"),
        "outer_withdraw_only": (1067, 545, 522, "e755fd11da6ee4b11731648309606b9d71dae73d54ae2b369444f2dabcd53677"),
    }
    for name, (count, longs, shorts, digest) in expected.items():
        control = payload["controls"][name]
        assert control["event_count"] == count
        assert control["side_counts"] == {"long": longs, "short": shorts}
        canonical = [
            {
                "signal_position": int(row["signal_position"]),
                "entry_position": int(row["entry_position"]),
                "exit_position": int(row["exit_position"]),
                "side": int(row["side"]),
                "hold_bars": int(row["hold_bars"]),
            }
            for row in control["events"]
        ]
        assert qualify.canonical_hash(canonical) == digest
        assert control["event_clock_sha256"] == digest


def test_every_control_event_uses_frozen_clock() -> None:
    payload = json.loads(PATH.read_text())
    for control in payload["controls"].values():
        for row in control["events"]:
            assert row["entry_position"] == row["signal_position"] + 2
            assert row["exit_position"] == row["entry_position"] + 72
            assert row["hold_bars"] == 72
