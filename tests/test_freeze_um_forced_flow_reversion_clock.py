from __future__ import annotations

import pandas as pd
import pytest

from training import freeze_um_forced_flow_reversion_clock as freeze
from training import preregister_um_forced_flow_reversion as umfr


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_position": 0,
                "entry_position": 1,
                "exit_position": 37,
                "signal_date": "2020-01-01 00:00:00",
                "entry_date": "2020-01-01 00:05:00",
                "exit_date": "2020-01-01 03:05:00",
                "side": 1,
                "branch": "umfr36",
                "hold_bars": 36,
            },
            {
                "signal_position": 37,
                "entry_position": 38,
                "exit_position": 74,
                "signal_date": "2020-01-01 03:05:00",
                "entry_date": "2020-01-01 03:10:00",
                "exit_date": "2020-01-01 06:10:00",
                "side": -1,
                "branch": "umfr36",
                "hold_bars": 36,
            },
        ],
        columns=umfr.SCHEDULE_COLUMNS,
    )


def _cfg() -> umfr.Config:
    return umfr.Config(
        minimum_nonoverlap_total=0,
        minimum_nonoverlap_per_year=0,
        minimum_nonoverlap_per_2023_half=0,
        minimum_nonoverlap_per_2023_quarter=0,
        minimum_side_share=0.0,
        minimum_side_events_per_year=0,
        minimum_active_months=0,
    )


def _support_result(schedule: pd.DataFrame, cfg: umfr.Config) -> dict[str, object]:
    return {
        "support_decision": "pass",
        "selected_quantile": 0.8,
        "selected_support": {
            "quantile": 0.8,
            "raw_primary": 2,
            "support": umfr._support(schedule, cfg),
        },
    }


def test_validate_schedule_accepts_exact_outcome_free_clock() -> None:
    cfg = _cfg()
    schedule = _schedule()
    result = freeze._validate_schedule(
        schedule, _support_result(schedule, cfg), raw_primary=2, cfg=cfg
    )
    assert result["nonoverlap_total"] == 2
    assert not freeze._has_outcome_column(tuple(schedule.columns))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("entry_position", 2, "next five-minute open"),
        ("exit_position", 38, "fixed hold"),
        ("side", 0, "non-directional"),
        ("branch", "control", "non-primary branch"),
        ("hold_bars", 35, "mutable hold"),
    ],
)
def test_validate_schedule_rejects_execution_drift(
    column: str, value: object, message: str
) -> None:
    cfg = _cfg()
    schedule = _schedule()
    support = _support_result(schedule, cfg)
    broken = schedule.copy()
    broken.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        freeze._validate_schedule(broken, support, raw_primary=2, cfg=cfg)


def test_validate_schedule_rejects_overlap_and_sealed_interval() -> None:
    cfg = _cfg()
    schedule = _schedule()
    support = _support_result(schedule, cfg)
    overlapping = schedule.copy()
    overlapping.loc[1, ["signal_position", "entry_position", "exit_position"]] = [
        35,
        36,
        72,
    ]
    with pytest.raises(ValueError, match="overlapping holds"):
        freeze._validate_schedule(overlapping, support, raw_primary=2, cfg=cfg)
    sealed = schedule.copy()
    sealed.loc[1, "exit_date"] = "2024-01-01 00:00:00"
    with pytest.raises(ValueError, match="sealed interval"):
        freeze._validate_schedule(sealed, support, raw_primary=2, cfg=cfg)


def test_validate_schedule_rejects_support_or_quantile_drift() -> None:
    cfg = _cfg()
    schedule = _schedule()
    support = _support_result(schedule, cfg)
    with pytest.raises(ValueError, match="raw UMFR count"):
        freeze._validate_schedule(schedule, support, raw_primary=3, cfg=cfg)
    support["selected_quantile"] = 0.75
    with pytest.raises(ValueError, match="inconsistent"):
        freeze._validate_schedule(schedule, support, raw_primary=2, cfg=cfg)


def test_outcome_column_detector_uses_tokens_not_fraction_substrings() -> None:
    assert not freeze._has_outcome_column(("spot_flow_fraction", "entry_date"))
    assert freeze._has_outcome_column(("future_return",))
    assert freeze._has_outcome_column(("funding_factor",))
    assert freeze._has_outcome_column(("entry_open",))
