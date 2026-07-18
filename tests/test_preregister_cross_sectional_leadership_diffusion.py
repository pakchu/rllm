from __future__ import annotations

import pandas as pd
import pytest

from training import preregister_cross_sectional_leadership_diffusion as cld


def test_prior_rolling_quantile_excludes_current_value() -> None:
    values = pd.Series([1.0, 2.0, 100.0])
    threshold = cld.prior_rolling_quantile(
        values, quantile=0.5, window=2, minimum=2
    )
    assert pd.isna(threshold.iloc[1])
    assert threshold.iloc[2] == pytest.approx(1.5)


def test_protocol_keeps_post_entry_outcomes_sealed() -> None:
    payload = cld.protocol()
    boundary = payload["evidence_boundary"]
    assert boundary["post_entry_outcomes_opened"] is False
    assert "entry or later OHLC" in boundary["forbidden_before_evaluator_freeze"]
    assert payload["clock"]["entry"] == "one full five-minute bar after that boundary"
    assert "both end at the completed feature boundary" in payload["feature_formula"][
        "btc_lag"
    ]
    assert payload["eventual_execution"]["position_state"].startswith("one BTC position")


def test_support_selector_uses_mechanism_strength_not_event_count() -> None:
    weak = {
        "move_quantile": 0.6,
        "prior_hhi_quantile": 0.6,
        "maximum_hhi_ratio": 0.9,
        "minimum_participation": 4 / 6,
        "minimum_flow_alignment": 0.5,
        "turnover_quantile": 0.5,
        "leader_decline_quantile": 0.5,
        "support": {"passes": True, "nonoverlap_total": 999},
    }
    strong = {
        "move_quantile": 0.6,
        "prior_hhi_quantile": 0.6,
        "maximum_hhi_ratio": 0.9,
        "minimum_participation": 5 / 6,
        "minimum_flow_alignment": 4 / 6,
        "turnover_quantile": 0.5,
        "leader_decline_quantile": 0.6,
        "support": {"passes": True, "nonoverlap_total": 100},
    }
    assert cld.select_support_cell([weak, strong]) is strong


def test_support_selector_rejects_nested_outcome_fields() -> None:
    cell = {
        "move_quantile": 0.6,
        "prior_hhi_quantile": 0.6,
        "maximum_hhi_ratio": 0.9,
        "minimum_participation": 5 / 6,
        "minimum_flow_alignment": 4 / 6,
        "turnover_quantile": 0.5,
        "leader_decline_quantile": 0.6,
        "support": {"passes": True, "cagr": 100.0},
    }
    with pytest.raises(ValueError, match="forbidden outcome"):
        cld.select_support_cell([cell])


def test_quarterly_schedule_delays_entry_and_prevents_overlap() -> None:
    signal = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(
                ["2023-02-01 00:55", "2023-02-01 01:55", "2023-02-01 08:55"]
            ),
            "feature_boundary": pd.to_datetime(
                ["2023-02-01 01:00", "2023-02-01 02:00", "2023-02-01 09:00"]
            ),
            "entry_date": pd.to_datetime(
                ["2023-02-01 01:05", "2023-02-01 02:05", "2023-02-01 09:05"]
            ),
            "side": [1, -1, -1],
            "branch": ["long", "short", "short"],
        }
    )
    schedule = cld.quarterly_schedule(signal, cld.Config())
    assert len(schedule) == 2
    assert schedule["entry_date"].tolist() == [
        pd.Timestamp("2023-02-01 01:05"),
        pd.Timestamp("2023-02-01 09:05"),
    ]
    assert schedule["entry_position"].eq(schedule["signal_position"] + 2).all()
    assert schedule["exit_position"].eq(schedule["entry_position"] + 72).all()
