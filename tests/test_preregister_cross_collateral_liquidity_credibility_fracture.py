from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from training import (
    preregister_cross_collateral_liquidity_credibility_fracture as pdf,
)
from training.preregister_metaorder_fragmentation_impact_curvature import (
    nonoverlapping_schedule,
)


def _venue_input(rows: int = 1) -> pd.DataFrame:
    data: dict[str, object] = {
        "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
        "source_complete": [True] * rows,
        "quarantined": [False] * rows,
    }
    for distance in range(1, 6):
        data[f"um_depth_m{distance}"] = np.ones(rows)
        data[f"um_depth_p{distance}"] = np.full(rows, np.exp(2.0))
        data[f"um_log_net_m{distance}"] = np.full(rows, 2.0)
        data[f"um_log_net_p{distance}"] = np.zeros(rows)
        data[f"um_log_mad_m{distance}"] = np.ones(rows)
        data[f"um_log_step_m{distance}"] = np.ones(rows)
        data[f"um_log_mad_p{distance}"] = np.full(rows, 3.0)
        data[f"um_log_step_p{distance}"] = np.full(rows, 3.0)
    return pd.DataFrame(data)


def test_venue_firmness_sign_is_anchored_by_net_and_penalized_by_churn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pdf,
        "_lagged_z",
        lambda values, clean, cfg: values.where(clean).astype(float),
    )
    output = pdf._venue_features(_venue_input(), "um", pdf.Config())
    assert output.loc[0, "credibility"] == pytest.approx(3.0)
    assert output.loc[0, "display"] == pytest.approx(-2.0)

    high_bid_churn = _venue_input()
    for distance in range(1, 6):
        high_bid_churn.loc[0, f"um_log_mad_m{distance}"] = 7.0
        high_bid_churn.loc[0, f"um_log_step_m{distance}"] = 7.0
    changed = pdf._venue_features(high_bid_churn, "um", pdf.Config())
    assert changed.loc[0, "credibility"] < output.loc[0, "credibility"]


def test_fracture_requires_two_same_side_bars_but_can_remain_tradeable() -> None:
    um = pd.DataFrame(
        {
            "credibility": [1.0, 1.0, 1.0, np.nan, -1.0, -1.0],
            "display": [-1.5, -1.5, -1.5, 0.0, 1.5, 1.5],
        }
    )
    cm = um.copy()
    state = pdf._fracture_state(
        um,
        cm,
        pd.Series([True] * len(um)),
        pdf.Config(),
    )
    assert state["raw_state"].tolist() == [1, 1, 1, 0, -1, -1]
    assert state["confirmed_state"].tolist() == [0, 1, 1, 0, 0, -1]


def test_fracture_rejects_cross_venue_disagreement_and_weak_display() -> None:
    um = pd.DataFrame(
        {"credibility": [1.0, 1.0], "display": [-1.5, -1.5]}
    )
    cm = pd.DataFrame(
        {"credibility": [-1.0, -1.0], "display": [-1.5, -0.5]}
    )
    state = pdf._fracture_state(
        um,
        cm,
        pd.Series([True, True]),
        pdf.Config(),
    )
    assert state["raw_state"].eq(0).all()
    assert state["confirmed_state"].eq(0).all()


def test_lagged_normalization_does_not_revise_the_past() -> None:
    cfg = replace(
        pdf.Config(),
        robust_baseline_bars=4,
        robust_min_periods=3,
    )
    values = pd.Series([1.0, 2.0, 4.0, 2.0, 1.0, 3.0, 2.0, 5.0])
    clean = pd.Series([True] * len(values))
    original = pdf._lagged_z(values, clean, cfg)
    changed = values.copy()
    changed.iloc[-1] = 500.0
    replay = pdf._lagged_z(changed, clean, cfg)
    pd.testing.assert_series_equal(original.iloc[:-1], replay.iloc[:-1])


def test_pdf10_clock_enters_next_open_and_exits_at_t_plus_three() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=8, freq="5min"),
            "quarantined": [False] * 8,
        }
    )
    signal = pd.DataFrame(
        {
            "side": [0, 1, 1, 0, 0, 0, 0, 0],
            "hold_bars": [0, 2, 2, 0, 0, 0, 0, 0],
            "branch": ["none", "bull", "bull", "none", "none", "none", "none", "none"],
        }
    )
    schedule = nonoverlapping_schedule(
        signal,
        frame,
        start="2023-01-01",
        end="2023-01-02",
    )
    assert len(schedule) == 1
    assert schedule.loc[0, "signal_position"] == 1
    assert schedule.loc[0, "entry_position"] == 2
    assert schedule.loc[0, "exit_position"] == 4


def test_pdf10_v1_parameters_are_not_a_search_grid() -> None:
    cfg = pdf.Config()
    assert cfg.credibility_entry_z == 0.75
    assert cfg.display_entry_z == 1.00
    assert cfg.confirmation_bars == 2
    assert cfg.hold_bars == 2
    assert cfg.minimum_nonoverlap_total == 500
    assert cfg.minimum_nonoverlap_per_half == 180
    assert cfg.minimum_nonoverlap_per_quarter == 75
    assert cfg.minimum_side_share == 0.35
    pdf._validate_frozen_config(cfg)
    with pytest.raises(ValueError, match="config is frozen"):
        pdf._validate_frozen_config(replace(cfg, display_entry_z=0.75))
    with pytest.raises(ValueError, match="config is frozen"):
        pdf._validate_frozen_config(
            replace(cfg, credibility_manifest="results/alternate.json")
        )


def test_event_clock_hash_binds_positions_and_sides() -> None:
    schedule = pd.DataFrame(
        {"signal_position": [10, 20], "side": [1, -1]}
    )
    baseline = pdf._event_clock_sha256(schedule)
    changed_position = schedule.copy()
    changed_position.loc[1, "signal_position"] = 21
    changed_side = schedule.copy()
    changed_side.loc[1, "side"] = 1
    assert pdf._event_clock_sha256(changed_position) != baseline
    assert pdf._event_clock_sha256(changed_side) != baseline


def test_empty_support_clock_fails_closed() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range(
                "2023-01-01", "2024-01-01", freq="5min", inclusive="left"
            ),
            "quarantined": False,
        }
    )
    signal = pd.DataFrame(
        {
            "side": np.zeros(len(frame), dtype=np.int8),
            "hold_bars": np.zeros(len(frame), dtype=np.int16),
            "branch": "none",
        }
    )
    schedule = pdf._quarterly_schedule(signal, frame)
    result = pdf.support_summary(
        signal,
        frame,
        pdf.Config(),
        schedule=schedule,
    )
    assert schedule.empty
    assert result["nonoverlap_total"] == 0
    assert result["by_quarter"] == {"q1": 0, "q2": 0, "q3": 0, "q4": 0}
    assert result["passes_support"] is False
