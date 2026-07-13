from __future__ import annotations

import json

import numpy as np
import pandas as pd

from training.search_funding_premium_external_state_gate_alpha import _manifest_core_hash
from training.search_online_rls_price_impact_alpha import (
    OnlineRlsImpactConfig,
    _policy_specs,
    _rls_path,
    _signals,
    run,
)


def test_rls_prefix_is_unchanged_by_future_observation() -> None:
    returns = np.array([0.01, -0.02, 0.03, 0.01], dtype=float)
    imbalance = np.array([0.2, -0.4, 0.5, 0.1], dtype=float)
    baseline = _rls_path(returns, imbalance, half_life=2)
    extended = _rls_path(np.r_[returns, 99.0], np.r_[imbalance, -1.0], half_life=2)
    for key in baseline:
        np.testing.assert_allclose(baseline[key], extended[key][: len(returns)], equal_nan=True)


def test_rls_emits_preupdate_slope_and_prediction() -> None:
    path = _rls_path(np.array([0.10, 0.20]), np.array([1.0, 1.0]), half_life=1)
    assert path["slope"][0] == 0.0
    assert path["prediction"][0] == 0.0
    assert path["slope"][1] > 0.0
    assert path["prediction"][1] < 0.20


def test_current_residual_cannot_enter_its_own_z_denominator() -> None:
    size = 6050
    returns = np.zeros(size, dtype=float)
    imbalance = np.zeros(size, dtype=float)
    returns[-1] = 1.0
    path = _rls_path(returns, imbalance, half_life=576)
    # Prior residual history is effectively zero, so a current shock must not
    # be damped by updating variance before z-score emission.
    forgetting = 2.0 ** (-1.0 / 576.0)
    expected = 1.0 / np.sqrt(max(1e-6 * forgetting ** (size - 1), 1e-12))
    assert np.isclose(path["residual_z"][-1], expected)


def test_policy_family_is_exactly_eight_fixed_mappings() -> None:
    rows = 21_000
    features = pd.DataFrame(
        {
            "rls_slope_576": np.linspace(-1.0, 1.0, rows),
            "rls_slope_2016": np.linspace(-2.0, 2.0, rows),
        }
    )
    specs = _policy_specs(features, np.ones(rows, dtype=bool))
    assert len(specs) == 8
    assert {(s["half_life"], s["slope_state"], s["direction"]) for s in specs} == {
        (half_life, state, direction)
        for half_life in (576, 2016)
        for state in ("high", "low")
        for direction in ("continuation", "fade")
    }


def test_signal_direction_and_flip_are_exact_opposites() -> None:
    features = pd.DataFrame(
        {
            "rls_slope_576": [2.0, 2.0],
            "rls_residual_z_576": [2.0, -2.0],
        }
    )
    spec = {
        "half_life": 576,
        "slope_state": "high",
        "direction": "fade",
        "stride": 1,
        "hold": 24,
        "slope_lower": -1.0,
        "slope_upper": 1.0,
        "residual_z_threshold": 1.5,
    }
    active, side = _signals(features, spec)
    flipped_active, flipped = _signals(features, spec, flip=True)
    assert active.tolist() == [True, True]
    assert flipped_active.tolist() == active.tolist()
    assert side.tolist() == [-1, 1]
    assert flipped.tolist() == [1, -1]


def test_frozen_preflight_report_never_opens_oos(tmp_path) -> None:
    core = {
        "protocol": {},
        "source_prefix_hashes": {},
        "feature_hash": "unused",
        "search_space": {"raw_specs": 8, "eligible_unique_paths": 0},
        "preflight_diagnostics": [],
        "selected": [],
    }
    manifest = {"as_of": "test", "sha256": _manifest_core_hash(core), **core}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    report = run(
        OnlineRlsImpactConfig(
            input_csv="later-data-must-not-be-opened",
            output=str(tmp_path / "report.json"),
            manifest_output=str(manifest_path),
            docs_output=str(tmp_path / "report.md"),
        )
    )
    assert report["preflight_only"] is True
    assert report["oos_opened"] is False
    assert report["selected"] == []
