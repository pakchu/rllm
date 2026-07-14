from __future__ import annotations

import pandas as pd
import pytest

from training import freeze_cash_auction_transfer_catchup_handoff_clock as freeze
from training import preregister_cash_auction_transfer_catchup_handoff as catch


def _schedule() -> pd.DataFrame:
    rows = [
        {
            "signal_position": 0,
            "entry_position": 1,
            "exit_position": 13,
            "signal_date": "2020-01-01 00:00:00",
            "entry_date": "2020-01-01 00:05:00",
            "exit_date": "2020-01-01 01:05:00",
            "side": 1,
            "branch": "catch12",
            "hold_bars": 12,
        },
        {
            "signal_position": 13,
            "entry_position": 14,
            "exit_position": 26,
            "signal_date": "2020-01-01 01:05:00",
            "entry_date": "2020-01-01 01:10:00",
            "exit_date": "2020-01-01 02:10:00",
            "side": -1,
            "branch": "catch12",
            "hold_bars": 12,
        },
    ]
    return pd.DataFrame(rows, columns=catch.SCHEDULE_COLUMNS)


def _support_result(schedule: pd.DataFrame, cfg: catch.Config) -> dict[str, object]:
    support = catch._support(schedule, cfg)
    return {
        "support_decision": "pass",
        "selected_quantile": 0.975,
        "selected_support": {
            "quantile": 0.975,
            "raw_primary": 2,
            "support": support,
        },
    }


def test_validate_schedule_accepts_exact_next_open_fixed_hold_clock() -> None:
    cfg = catch.Config(
        minimum_nonoverlap_total=0,
        minimum_nonoverlap_per_year=0,
        minimum_nonoverlap_per_2023_half=0,
        minimum_nonoverlap_per_2023_quarter=0,
        minimum_side_share=0.0,
        minimum_side_events_per_year=0,
        minimum_active_months=0,
    )
    schedule = _schedule()
    result = freeze._validate_schedule(
        schedule,
        _support_result(schedule, cfg),
        raw_primary=2,
        cfg=cfg,
    )
    assert result["nonoverlap_total"] == 2
    assert not freeze._has_return_column(tuple(schedule.columns))


def test_validate_schedule_rejects_timing_or_support_drift() -> None:
    cfg = catch.Config(
        minimum_nonoverlap_total=0,
        minimum_nonoverlap_per_year=0,
        minimum_nonoverlap_per_2023_half=0,
        minimum_nonoverlap_per_2023_quarter=0,
        minimum_side_share=0.0,
        minimum_side_events_per_year=0,
        minimum_active_months=0,
    )
    schedule = _schedule()
    support = _support_result(schedule, cfg)
    broken = schedule.copy()
    broken.loc[0, "entry_position"] = 2
    with pytest.raises(ValueError, match="next five-minute open"):
        freeze._validate_schedule(broken, support, raw_primary=2, cfg=cfg)
    with pytest.raises(ValueError, match="raw CATCH count"):
        freeze._validate_schedule(schedule, support, raw_primary=3, cfg=cfg)


def test_return_column_detector_uses_tokens_not_fraction_substrings() -> None:
    assert not freeze._has_return_column(("spot_flow_fraction", "entry_date"))
    assert freeze._has_return_column(("future_return",))
