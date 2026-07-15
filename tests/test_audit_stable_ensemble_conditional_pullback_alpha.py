import json

import numpy as np
import pytest

from training.audit_stable_ensemble_conditional_pullback_alpha import (
    AUDIT_SPEC,
    DELAY_BARS,
    FEATURE_COLUMNS,
    _audit_hash,
    _spec_hash,
    delayed_feature_context,
    interaction_mask,
    stability_decision,
)


def test_delay_shifts_predictors_but_preserves_current_source_identity():
    matrix = np.arange(5 * len(FEATURE_COLUMNS), dtype=float).reshape(5, len(FEATURE_COLUMNS))
    funding = FEATURE_COLUMNS.index("funding_leg")
    premium = FEATURE_COLUMNS.index("premium_leg")
    matrix[:, funding] = [0, 1, 0, 1, 0]
    matrix[:, premium] = [1, 0, 1, 0, 1]
    delayed = delayed_feature_context({"matrix": matrix}, bars=2)["matrix"]
    other = FEATURE_COLUMNS.index("rex_2016_range_width_pct")
    assert delayed[3, other] == matrix[1, other]
    assert delayed[:, funding].tolist() == matrix[:, funding].tolist()
    assert delayed[:, premium].tolist() == matrix[:, premium].tolist()


def test_interaction_ablation_truth_tables():
    width = np.array([0.2, 0.05, 0.05])
    pullback = np.array([0.0, -0.5, 0.1])
    kwargs = {
        "width": width,
        "pullback": pullback,
        "width_threshold": 0.1,
        "pullback_threshold": -0.2,
    }
    assert interaction_mask("conditional", **kwargs).tolist() == [True, True, False]
    assert interaction_mask("source_only", **kwargs).tolist() == [True, True, True]
    assert interaction_mask("unconditional_pullback", **kwargs).tolist() == [False, True, False]
    assert interaction_mask("width_only", **kwargs).tolist() == [True, False, False]
    assert interaction_mask("reversed_pullback", **kwargs).tolist() == [True, False, True]
    with pytest.raises(ValueError, match="unknown"):
        interaction_mask("bad", **kwargs)


def test_stability_decision_requires_every_audit_axis():
    good = {
        "individual_passes": {1000: 4, 2000: 3},
        "ensembles_pass": {1000: True, 2000: True},
        "delayed_individual_passes": 5,
        "delayed_ensemble_pass": True,
        "ablation_passes": {
            "conditional": True,
            "source_only": False,
            "unconditional_pullback": False,
            "width_only": False,
            "reversed_pullback": False,
        },
    }
    assert stability_decision(**good)
    assert not stability_decision(**{**good, "individual_passes": {1000: 4, 2000: 2}})
    assert not stability_decision(**{**good, "delayed_ensemble_pass": False})
    bad_ablation = {**good["ablation_passes"], "source_only": True}
    assert not stability_decision(**{**good, "ablation_passes": bad_ablation})


def test_audit_spec_is_pre_oos_and_hash_is_deterministic():
    assert AUDIT_SPEC["selection_end_exclusive"] == "2024-01-01"
    assert AUDIT_SPEC["tree_counts"] == [1_000, 2_000]
    assert AUDIT_SPEC["delay_control"]["bars"] == DELAY_BARS == 12
    assert _spec_hash() == _spec_hash()
    payload = {
        "phase": "x",
        "oos_opened": False,
        "sealed_windows": ["2024+"],
        "selection_end_exclusive": "2024-01-01",
        "audit_spec": AUDIT_SPEC,
        "spec_hash": _spec_hash(),
        "implementation_hash": "x",
        "source_prefix_hashes": {},
        "feature_prefix_hash": "x",
        "forest_size_runs": {},
        "delay_control": {},
        "ablations": [],
        "stability_summary": {},
    }
    assert _audit_hash(payload) == _audit_hash(json.loads(json.dumps(payload)))
