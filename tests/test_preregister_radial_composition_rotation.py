from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import preregister_radial_composition_rotation as rcr


def _share_frame(rows: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(20260714)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "source_complete": [True] * rows,
            "quarantined": [False] * rows,
        }
    )
    for venue in ("um", "cm"):
        for side in ("m", "p"):
            values = rng.lognormal(mean=0.0, sigma=0.5, size=(rows, 5))
            for shell in range(1, 6):
                frame[f"{venue}_shell_share_median_{side}{shell}"] = values[
                    :, shell - 1
                ]
    return frame


def _swap(frame: pd.DataFrame, first: str, second: str) -> pd.DataFrame:
    output = frame.copy()
    if first.endswith("_"):
        first_columns = [column for column in frame if column.startswith(first)]
        pairs = [
            (first_column, second + first_column[len(first) :])
            for first_column in first_columns
        ]
    else:
        pairs = [
            (
                column,
                column[: -len(f"{first}{shell}")] + f"{second}{shell}",
            )
            for column in frame
            for shell in range(1, 6)
            if column.endswith(f"{first}{shell}")
        ]
    for first_column, second_column in pairs:
        output[first_column] = frame[second_column]
        output[second_column] = frame[first_column]
    return output


def test_radial_barycenter_renormalizes_median_shell_shares() -> None:
    frame = pd.DataFrame(
        {
            "um_shell_share_median_m1": [5.0, 0.0],
            "um_shell_share_median_m2": [0.0, 0.0],
            "um_shell_share_median_m3": [0.0, 0.0],
            "um_shell_share_median_m4": [0.0, 0.0],
            "um_shell_share_median_m5": [0.0, 7.0],
        }
    )
    result = rcr._radial_barycenter(frame, "um", "m")
    assert result.tolist() == pytest.approx([1.0, 5.0])

    scaled = frame * 11.0
    pd.testing.assert_series_equal(
        rcr._radial_barycenter(scaled, "um", "m"),
        result,
    )


def test_rotation_is_bid_ask_antisymmetric() -> None:
    frame = pd.DataFrame(
        {
            "source_complete": [True, True],
            **{
                f"um_shell_share_median_{side}{shell}": [
                    float(shell == 3),
                    float(shell == (2 if side == "m" else 4)),
                ]
                for side in ("m", "p")
                for shell in range(1, 6)
            },
        }
    )
    rotation = rcr._venue_rotation(frame, "um")
    assert rotation.loc[1, "bid_inward"] == pytest.approx(1.0)
    assert rotation.loc[1, "ask_inward"] == pytest.approx(-1.0)
    assert rotation.loc[1, "polarization"] == pytest.approx(np.sqrt(2.0))

    swapped = _swap(frame, "_m", "_p")
    inverse = rcr._venue_rotation(swapped, "um")
    assert inverse.loc[1, "polarization"] == pytest.approx(
        -rotation.loc[1, "polarization"]
    )


def test_missing_or_false_source_completeness_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rcr,
        "_lagged_z",
        lambda values, clean, cfg: values.where(clean).astype(float),
    )
    frame = _share_frame(5)
    frame["source_complete"] = pd.Series(
        [True, True, np.nan, False, True], dtype="object"
    )
    rotation = rcr._venue_rotation(frame, "um")
    assert rotation["clean"].tolist() == [False, True, False, False, False]
    features = rcr.build_features(frame, rcr.Config())
    assert features["available"].tolist() == [False, True, False, False, False]


def test_cross_venue_score_is_venue_invariant_and_side_antisymmetric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rcr,
        "_lagged_z",
        lambda values, clean, cfg: values.where(clean).astype(float),
    )
    frame = _share_frame()
    original = rcr.build_features(frame, rcr.Config())

    venue_swap = _swap(frame, "um_", "cm_")
    venue_result = rcr.build_features(venue_swap, rcr.Config())
    pd.testing.assert_series_equal(
        venue_result["score"], original["score"], check_names=False
    )

    side_swap = _swap(frame, "_m", "_p")
    side_result = rcr.build_features(side_swap, rcr.Config())
    np.testing.assert_allclose(
        side_result.loc[original["available"], "score"],
        -original.loc[original["available"], "score"],
    )


def test_future_share_change_does_not_revise_prior_scores() -> None:
    cfg = replace(
        rcr.Config(),
        robust_baseline_bars=6,
        robust_min_periods=4,
    )
    frame = _share_frame(30)
    original = rcr.build_features(frame, cfg)
    changed = frame.copy()
    for shell in range(1, 6):
        changed.loc[29, f"um_shell_share_median_m{shell}"] *= 1.0 + shell
    replay = rcr.build_features(changed, cfg)
    pd.testing.assert_series_equal(
        original["score"].iloc[:-1], replay["score"].iloc[:-1]
    )


def test_actual_robust_normalization_preserves_side_antisymmetry() -> None:
    cfg = replace(
        rcr.Config(),
        robust_baseline_bars=6,
        robust_min_periods=4,
    )
    frame = _share_frame(30)
    original = rcr.build_features(frame, cfg)
    inverse = rcr.build_features(_swap(frame, "_m", "_p"), cfg)
    paired = original["available"] & inverse["available"]
    np.testing.assert_allclose(
        inverse.loc[paired, "score"],
        -original.loc[paired, "score"],
        atol=1e-12,
    )


def test_signal_is_threshold_symmetric_and_holds_144_bars() -> None:
    features = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=6, freq="5min"),
            "score": [-2.1, -2.0, -1.99, 1.99, 2.0, 2.1],
            "available": [True, True, True, True, True, False],
        }
    )
    signal = rcr.build_signal(features, rcr.Config())
    assert signal["side"].tolist() == [-1, -1, 0, 0, 1, 0]
    assert signal["hold_bars"].tolist() == [144, 144, 0, 0, 144, 0]


def test_rcr_clock_enters_next_open_and_exits_after_144_bars() -> None:
    rows = 160
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "quarantined": False,
        }
    )
    signal = pd.DataFrame(
        {
            "side": np.zeros(rows, dtype=np.int8),
            "hold_bars": np.zeros(rows, dtype=np.int16),
            "branch": "none",
        }
    )
    signal.loc[5, ["side", "hold_bars", "branch"]] = [
        1,
        144,
        "bullish_radial_composition_rotation",
    ]
    schedule = rcr.pdf._quarterly_schedule(signal, frame)
    assert len(schedule) == 1
    assert schedule.loc[0, "signal_position"] == 5
    assert schedule.loc[0, "entry_position"] == 6
    assert schedule.loc[0, "exit_position"] == 150


def test_empty_support_fails_closed() -> None:
    schedule = pd.DataFrame(columns=["signal_date", "side"])
    result = rcr.support_summary(schedule, rcr.Config())
    assert result["nonoverlap_total"] == 0
    assert result["by_quarter"] == {"q1": 0, "q2": 0, "q3": 0, "q4": 0}
    assert result["passes_support"] is False


def test_support_boundaries_enforce_side_and_quarter_balance() -> None:
    dates = pd.concat(
        [
            pd.Series(pd.date_range(start, periods=30, freq="12h"))
            for start in (
                "2023-01-01",
                "2023-04-01",
                "2023-07-01",
                "2023-10-01",
            )
        ],
        ignore_index=True,
    )
    schedule = pd.DataFrame(
        {
            "signal_date": dates.astype(str),
            "side": np.tile([1, -1], 60),
        }
    )
    assert rcr.support_summary(schedule, rcr.Config())["passes_support"] is True

    one_sided = schedule.copy()
    one_sided["side"] = 1
    assert rcr.support_summary(one_sided, rcr.Config())["passes_support"] is False

    concentrated = schedule.copy()
    concentrated["signal_date"] = pd.date_range(
        "2023-01-01", periods=120, freq="12h"
    ).astype(str)
    assert (
        rcr.support_summary(concentrated, rcr.Config())["passes_support"]
        is False
    )


def test_independence_gates_reject_prior_clock_and_feature_relabeling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = pd.DataFrame({"signal_position": [10, 20, 30, 40]})
    same_clock = schedule.copy()
    disjoint_clock = pd.DataFrame({"signal_position": [100, 200]})
    monkeypatch.setattr(
        rcr,
        "_prior_schedules",
        lambda frame: (
            pd.DataFrame(),
            same_clock,
            pd.DataFrame(),
            disjoint_clock,
        ),
    )
    features = pd.DataFrame(
        {"score": np.arange(1.0, 9.0), "available": [True] * 8}
    )
    monkeypatch.setattr(
        rcr,
        "_feature_comparators",
        lambda frame, credibility: pd.DataFrame(
            {
                "cclh_cross_pressure": np.arange(1.0, 9.0),
                "cclh_cross_elasticity": np.arange(8.0, 0.0, -1.0),
                "pdf10_credibility": np.arange(1.0, 9.0),
                "pdf10_display": np.arange(8.0, 0.0, -1.0),
            }
        ),
    )
    result = rcr.independence_summary(
        schedule,
        pd.DataFrame(index=features.index),
        features,
        rcr.Config(),
    )
    assert result["event_overlap"]["cclh"]["jaccard"] == 1.0
    assert result["event_overlap"]["cclh"]["current_event_match_share"] == 1.0
    assert result["passes_event_independence"] is False
    assert result["maximum_absolute_feature_spearman"] == pytest.approx(1.0)
    assert result["passes_feature_independence"] is False
    assert result["passes_independence"] is False


def test_rcr_v1_parameters_are_not_a_search_grid() -> None:
    cfg = rcr.Config()
    assert cfg.score_threshold == 2.0
    assert cfg.hold_bars == 144
    assert cfg.minimum_available_total == 90_000
    assert cfg.minimum_available_per_quarter == 15_000
    assert cfg.minimum_strong_per_quarter == 500
    assert cfg.minimum_nonoverlap_total == 120
    assert cfg.maximum_prior_feature_spearman == 0.60
    rcr._validate_frozen_config(cfg)
    with pytest.raises(ValueError, match="config is frozen"):
        rcr._validate_frozen_config(replace(cfg, score_threshold=1.75))


def test_barycenter_requires_only_outcome_blind_share_columns() -> None:
    source = rcr.PREREGISTRATION_SOURCE.read_text()
    assert "shell_share_median" in source
    assert "market_manifest" not in source
    assert "future_return" not in source
    assert "raw_return" not in source


def test_frozen_rcr_support_passes_without_opening_outcomes() -> None:
    path = Path("results/radial_composition_rotation_support_2026-07-14.json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "0e801542c29a964ea969ac4cc4317f98f89d95639683687fb605b9799fcd2d2e"
    )
    result = json.loads(path.read_text())
    assert result["protocol"]["outcomes_opened_for_rcr144"] is False
    assert result["protocol"]["price_or_return_loaded"] is False
    assert result["protocol"]["support_rejected"] is False
    assert result["availability"]["passes_availability"] is True
    assert result["support"]["nonoverlap_total"] == 646
    assert result["support"]["by_quarter"] == {
        "q1": 132,
        "q2": 168,
        "q3": 170,
        "q4": 176,
    }
    assert result["independence"]["passes_independence"] is True
    assert result["all_support_gates_pass"] is True


def test_frozen_rcr_event_clock_binds_all_execution_fields() -> None:
    path = Path(
        "results/radial_composition_rotation_event_clock_2026-07-14.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "14924e4273f9cf54c406c4d69291e1a8315c6ea8fde2bcb102ef84cebbd1dbb0"
    )
    result = json.loads(path.read_text())
    assert result["outcomes_opened_for_rcr144"] is False
    assert result["price_or_return_loaded"] is False
    assert result["event_count"] == 646
    assert result["side_counts"] == {"long": 321, "short": 325}
    assert result["quarter_counts"] == {
        "q1": 132,
        "q2": 168,
        "q3": 170,
        "q4": 176,
    }
    assert result["event_clock_sha256"] == (
        "67a1223201078578fb0406faee1f954fcc0698060e588626c5b1928351685665"
    )
