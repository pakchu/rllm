import json

import numpy as np
import pytest

from training.audit_weak_feature_responsibility_stability import (
    CANDIDATE_SPEC,
    EXPECTED_CANDIDATE_STATS,
    STABILITY_SPEC,
    _audit_hash,
    _implementation_hash,
    _spec_hash,
    assert_expected_candidate_stats,
    causal_shift,
    recent_side,
    stability_decision,
    validate_audit,
)


def test_causal_shift_excludes_current_row():
    assert causal_shift(np.array([10, 20, 30]), fill=-1).tolist() == [-1, 10, 20]
    shifted = causal_shift(np.array([1.5, np.nan, 3.0]), fill=np.nan)
    assert np.isnan(shifted[0])
    assert shifted[1] == 1.5
    assert np.isnan(shifted[2])


def test_recent_side_uses_shifted_completed_events_only():
    raw_side = np.array([0, 1, 0, 0, -1, 0], dtype=np.int8)
    shifted = causal_shift(raw_side)
    side, age = recent_side(shifted, bars=2)
    assert side.tolist() == [0, 0, 1, 1, 1, -1]
    assert age.tolist() == [9999, 9999, 0, 1, 2, 0]
    expired, _ = recent_side(shifted, bars=1)
    assert expired.tolist() == [0, 0, 1, 1, 0, -1]


def test_candidate_and_stability_specs_encode_seed_audit():
    assert CANDIDATE_SPEC["n_estimators"] == 160
    assert CANDIDATE_SPEC["max_depth"] == 3
    assert CANDIDATE_SPEC["min_samples_leaf"] == 16
    assert CANDIDATE_SPEC["max_features"] == 0.7
    assert CANDIDATE_SPEC["random_state"] == 715
    assert CANDIDATE_SPEC["anchor_cooldown_bars"] == 144
    assert CANDIDATE_SPEC["activation_quantile"] == 0.50
    assert CANDIDATE_SPEC["activation_threshold"] == pytest.approx(0.00365087256140527)
    assert CANDIDATE_SPEC["leverage"] == 0.50
    assert CANDIDATE_SPEC["funding_exit"] == {"hold_bars": 576, "take_bps": 400, "stop_bps": 1_000_000}
    assert CANDIDATE_SPEC["premium_exit"] == {"hold_bars": 144, "take_bps": 1_000_000, "stop_bps": 300}
    assert CANDIDATE_SPEC["selection_end_exclusive"] == "2024-01-01"
    assert STABILITY_SPEC["seeds"] == [7, 71, 715, 2026, 71515]
    assert STABILITY_SPEC["tree_counts"] == [160, 2000]
    assert STABILITY_SPEC["minimum_large_forest_seed_passes"] == 3
    assert STABILITY_SPEC["require_large_forest_ensemble_pass"] is True


def test_stability_decision_rejects_single_seed_luck():
    assert not stability_decision(candidate_pass=True, large_seed_passes=0, ensemble_pass=False)
    assert not stability_decision(candidate_pass=True, large_seed_passes=1, ensemble_pass=True)
    assert not stability_decision(candidate_pass=True, large_seed_passes=3, ensemble_pass=False)
    assert stability_decision(candidate_pass=True, large_seed_passes=3, ensemble_pass=True)


def test_audit_gate_and_expected_stats_helper():
    stats = {name: {**values, "mean_net_bps": 1.0, "win_rate": 0.5} for name, values in EXPECTED_CANDIDATE_STATS.items()}
    stats.update(
        {
            "train_2020h2": {"absolute_return_pct": 1.0},
            "train_2021": {"absolute_return_pct": 1.0},
            "train_2022": {"absolute_return_pct": 1.0},
        }
    )
    assert_expected_candidate_stats(stats)
    payload = {
        "oos_opened": False,
        "sealed_windows": ["2024+"],
        "selection_end_exclusive": "2024-01-01",
        "candidate_spec": CANDIDATE_SPEC,
        "stability_spec": STABILITY_SPEC,
        "spec_hash": _spec_hash(),
        "implementation_hash": _implementation_hash(),
        "source_prefix_hashes": {},
        "feature_prefix_hash": "x",
        "base_activation_hash": "x",
        "nested_side_hash": "x",
        "braid_side_hash": "x",
        "candidate": {"stats": stats},
        "small_forest_seed_runs": [],
        "large_forest_seed_runs": [],
        "large_forest_ensemble": {},
        "stability_summary": {"decision": "reject", "promotable": False},
    }
    payload["audit_hash"] = _audit_hash(payload)
    validate_audit(json.loads(json.dumps(payload)))
    payload["oos_opened"] = True
    with pytest.raises(RuntimeError, match="sealed"):
        validate_audit(payload)
