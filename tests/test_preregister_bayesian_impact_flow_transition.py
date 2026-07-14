from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from training import preregister_bayesian_impact_flow_transition as bift


def _small_cfg() -> bift.Config:
    return replace(
        bift.Config(),
        change_quantile=0.50,
        robust_baseline_hours=4,
        robust_min_periods=2,
        change_baseline_hours=4,
        change_min_periods=2,
        hazard_lambda_hours=4.0,
        max_run_length_hours=8,
        confirmation_hours=3,
        hold_bars=2,
        minimum_nonoverlap_total=1,
        minimum_nonoverlap_per_year=0,
        minimum_nonoverlap_per_2023_half=0,
        minimum_side_share=0.0,
        minimum_branch_share=0.0,
    )


def _diagnostics(*, aligned: bool = True) -> pd.DataFrame:
    rows = 10
    close = np.full(rows, 100.0)
    close[5:8] = [101.0, 102.0, 103.0] if aligned else [99.0, 98.0, 97.0]
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2023-01-01 00:55", periods=rows, freq="h"
            ),
            "clean": np.ones(rows, dtype=bool),
            "detector_available": np.ones(rows, dtype=bool),
            "flow_imbalance": np.full(rows, 0.10),
            "close": close,
            "run_drop": [1.0, 1.0, 1.0, 1.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "surprise": [1.0, 1.0, 1.0, 1.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


def test_lagged_robust_score_is_prefix_invariant() -> None:
    values = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    clean = pd.Series(True, index=values.index)
    baseline = bift.lagged_robust_zscore(
        values, clean, window=4, minimum=2
    )
    changed = values.copy()
    changed.loc[6] = -1_000_000.0
    replay = bift.lagged_robust_zscore(
        changed, clean, window=4, minimum=2
    )
    pd.testing.assert_series_equal(baseline.loc[:5], replay.loc[:5])


def test_segmented_bocpd_resets_after_unavailable_hour() -> None:
    observations = np.array(
        [[0.0], [0.2], [0.4], [999.0], [-1.0], [-0.8], [-0.6]],
        dtype=float,
    )
    available = np.array([True, True, True, False, True, True, True])
    cfg = _small_cfg()
    segmented = bift.segmented_bocpd(observations, available, cfg)
    suffix = bift.bocpd_student_t(
        observations[4:],
        hazard_lambda=cfg.hazard_lambda_hours,
        max_run_length=cfg.max_run_length_hours,
        short_run_horizon=cfg.short_run_horizon_hours,
    )
    assert np.isnan(segmented["run_drop"][3])
    for name in suffix:
        np.testing.assert_allclose(segmented[name][4:], suffix[name])


def test_candidate_waits_three_hours_and_uses_fixed_branch_rule() -> None:
    cfg = _small_cfg()
    propagation = bift.classify_candidates(_diagnostics(), cfg)
    assert propagation.loc[4, "setup"]
    assert not propagation.loc[4, "candidate"]
    assert propagation.loc[7, "candidate"]
    assert propagation.loc[7, "origin_hour_position"] == 4
    assert propagation.loc[7, "branch"] == "propagation"
    assert propagation.loc[7, "side"] == 1

    absorption = bift.classify_candidates(_diagnostics(aligned=False), cfg)
    assert absorption.loc[7, "candidate"]
    assert absorption.loc[7, "branch"] == "absorption"
    assert absorption.loc[7, "side"] == -1

    unresolved = _diagnostics()
    unresolved.loc[5:7, "close"] = 100.0
    unresolved_state = bift.classify_candidates(unresolved, cfg)
    assert not unresolved_state.loc[7, "candidate"]
    assert unresolved_state.loc[7, "side"] == 0


def test_candidate_prefix_is_invariant_to_later_changes() -> None:
    cfg = _small_cfg()
    baseline = bift.classify_candidates(_diagnostics(), cfg)
    changed = _diagnostics()
    changed.loc[8:, "flow_imbalance"] = -1.0
    changed.loc[8:, "close"] = 1.0
    changed.loc[8:, ["run_drop", "surprise"]] = 1_000_000.0
    replay = bift.classify_candidates(changed, cfg)
    pd.testing.assert_frame_equal(baseline.loc[:7], replay.loc[:7])


def test_schedule_requires_setup_origin_inside_split() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range(
                "2023-06-30 23:45", periods=12, freq="5min"
            ),
            "quarantined": np.zeros(12, dtype=bool),
        }
    )
    signal = pd.DataFrame(
        {
            "side": [0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0],
            "hold_bars": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            "branch": [
                "none",
                "none",
                "none",
                "propagation",
                "none",
                "absorption",
                "none",
                "none",
                "none",
                "none",
                "none",
                "none",
            ],
            "origin_position": [-1, -1, -1, 1, -1, 3, -1, -1, -1, -1, -1, -1],
        }
    )
    schedule = bift.nonoverlapping_bift_schedule(
        signal,
        frame,
        start="2023-07-01",
        end="2024-01-01",
    )
    assert schedule["signal_position"].tolist() == [5]
    assert schedule["origin_position"].tolist() == [3]


def test_support_stopping_rule_selects_strictest_passing_quantile() -> None:
    trials = [
        {"change_quantile": 0.90, "passes_support": True},
        {"change_quantile": 0.925, "passes_support": True},
        {"change_quantile": 0.95, "passes_support": False},
    ]
    assert bift._selected_support_quantile(trials) == 0.925


def test_frozen_support_artifact_keeps_outcomes_sealed() -> None:
    result = json.loads(
        Path(
            "results/bayesian_impact_flow_transition_support_2026-07-14.json"
        ).read_text()
    )
    assert result["protocol"]["outcomes_opened_for_bift"] is False
    assert result["all_support_gates_pass"] is True
    assert result["support_calibration"]["selected_change_quantile"] == 0.925
    assert result["support"]["nonoverlap_total"] == 272
    assert result["support"]["by_year"] == {
        "2020": 64,
        "2021": 46,
        "2022": 76,
        "2023": 86,
    }
    assert result["support"]["2023_h1"] == 51
    assert result["support"]["2023_h2"] == 34
    assert result["scheduled_branch_counts"] == {
        "propagation": 190,
        "absorption": 82,
    }
    for key in ("preregistration_source", "preregistration_document"):
        path = Path(result["frozen_artifacts"][key])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == result["frozen_artifacts"][f"{key}_sha256"]
