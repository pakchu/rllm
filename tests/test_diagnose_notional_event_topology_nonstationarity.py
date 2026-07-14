from __future__ import annotations

import pandas as pd

from training import diagnose_notional_event_topology_nonstationarity as diagnose


def test_rejected_netf_selection_and_sealed_oos_are_frozen() -> None:
    result = diagnose._verify_rejected_selection()
    assert result["selection"]["rejected"] is True
    assert result["protocol"]["sealed_windows"] == [
        "test2024",
        "eval2025",
        "ytd2026",
    ]


def test_structure_combo_uses_preregistered_mark_order() -> None:
    row = pd.Series(
        {
            "arrival_burst_mark": True,
            "notional_concentration_mark": False,
            "trade_id_span_per_aggregate_event_mark": True,
        }
    )
    assert diagnose.structure_combo(row) == "101"


def test_diagnostic_protocol_cannot_repair_or_open_netf_oos() -> None:
    result = diagnose.run_diagnostic()
    protocol = result["protocol"]
    assert protocol["may_repair_or_promote_netf"] is False
    assert protocol["opened_windows_only"] == ["2020", "2021", "2022", "2023"]
    assert protocol["sealed_windows_still_unopened"] == [
        "test2024",
        "eval2025",
        "ytd2026",
    ]
    assert {
        item["candidate"] for item in result["candidates"]
    } == {"netf_fast", "netf_slow"}
