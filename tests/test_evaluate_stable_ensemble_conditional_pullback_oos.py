import inspect
import json
from dataclasses import replace

import numpy as np

from training.evaluate_stable_ensemble_conditional_pullback_oos import (
    EXPECTED_MANIFEST_HASH,
    FUTURE_WINDOWS,
    OOS_GATE,
    Config,
    frozen_activation,
    passes_oos_gate,
    run,
    validate_frozen_manifest,
    validate_oos_horizon,
)


def _stats() -> dict:
    return {
        name: {
            "absolute_return_pct": 1.0,
            "cagr_to_strict_mdd": 3.1,
            "strict_mdd_pct": 10.0,
            "trades": gate["min_trades"],
        }
        for name, gate in OOS_GATE.items()
    }


def test_committed_manifest_matches_evaluator_pin_and_remains_sealed():
    manifest = validate_frozen_manifest(Config())
    assert manifest["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert manifest["oos_opened"] is False
    assert manifest["future_windows"] == {key: list(value) for key, value in FUTURE_WINDOWS.items()}


def test_oos_gate_requires_every_frozen_window_constraint():
    stats = _stats()
    assert passes_oos_gate(stats)
    for name in OOS_GATE:
        failed = json.loads(json.dumps(stats))
        failed[name]["cagr_to_strict_mdd"] = 2.99
        assert not passes_oos_gate(failed)
    failed = json.loads(json.dumps(stats))
    failed["holdout_2026h1"]["trades"] -= 1
    assert not passes_oos_gate(failed)


def test_oos_horizon_cannot_open_beyond_or_short_of_frozen_windows():
    cfg = Config()
    manifest = validate_frozen_manifest(cfg)
    validate_oos_horizon(cfg, manifest)
    for cutoff in ("2026-05-01", "2027-01-01"):
        try:
            validate_oos_horizon(replace(cfg, exclude_from=cutoff), manifest)
        except RuntimeError as error:
            assert "frozen OOS horizon" in str(error)
        else:
            raise AssertionError("non-frozen cutoff was accepted")


def test_frozen_activation_uses_pinned_thresholds_not_future_quantiles():
    from training.search_stable_ensemble_conditional_pullback_alpha import FEATURE_COLUMNS

    matrix = np.zeros((4, len(FEATURE_COLUMNS)), dtype=float)
    width = FEATURE_COLUMNS.index("rex_2016_range_width_pct")
    pullback = FEATURE_COLUMNS.index("htf_1d_range_pos")
    matrix[:, width] = [0.2, 0.05, 0.05, 0.0]
    matrix[:, pullback] = [0.0, -0.5, 0.1, 0.0]
    context = {
        "market": [None] * 4,
        "matrix": matrix,
        "funding_leg": np.array([True, True, True, False]),
    }
    model = {
        "anchor_positions": np.arange(4),
        "anchor_predictions": np.array([0.9, 0.9, 0.9, 0.9]),
    }
    manifest = {
        "candidate_spec": {
            "thresholds": {
                "funding_threshold": 0.8,
                "premium_threshold": 0.8,
                "width_threshold": 0.1,
                "pullback_threshold": -0.2,
            }
        }
    }
    assert frozen_activation(context, model, manifest).tolist() == [True, True, False, True]


def test_run_contract_replays_prefix_before_future_builder():
    source = inspect.getsource(run)
    assert source.index("_replay_pre_oos") < source.index("build_full_design")
    assert source.index("validate_frozen_manifest") < source.index("_replay_pre_oos")
    assert source.index("validate_oos_horizon") < source.index("_replay_pre_oos")
    assert "pre_integrity" in source
