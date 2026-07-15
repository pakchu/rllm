import json

from training.audit_delayed_conditional_pullback_test2024 import (
    AUDIT_SPEC,
    TEST_END,
    _audit_hash,
    _spec_hash,
    audit_decision,
    test_2024_passes as passes_test_2024,
)


def test_2024_gate_checks_return_ratio_mdd_and_support():
    good = {
        "absolute_return_pct": 1.0,
        "cagr_to_strict_mdd": 3.0,
        "strict_mdd_pct": 15.0,
        "trades": 12,
    }
    assert passes_test_2024(good)
    for key, value in (
        ("absolute_return_pct", 0.0),
        ("cagr_to_strict_mdd", 2.99),
        ("strict_mdd_pct", 15.01),
        ("trades", 11),
    ):
        assert not passes_test_2024({**good, key: value})


def test_audit_decision_requires_seed_and_ensemble_stability():
    good = {
        "individual_passes": {1000: 3, 2000: 4},
        "mean_three_passes": {1000: True, 2000: True},
        "mean_five_passes": {1000: True, 2000: True},
    }
    assert audit_decision(**good)
    assert not audit_decision(**{**good, "individual_passes": {1000: 2, 2000: 4}})
    assert not audit_decision(**{**good, "mean_five_passes": {1000: True, 2000: False}})


def test_audit_spec_physically_seals_2025_plus_and_hashes_deterministically():
    assert TEST_END == "2025-01-01"
    assert AUDIT_SPEC["opened_window"] == ["2024-01-01", "2025-01-01"]
    assert AUDIT_SPEC["sealed_windows"] == ["2025+"]
    assert AUDIT_SPEC["information_delay_bars"] == 12
    assert _spec_hash() == _spec_hash()
    payload = {
        "phase": "x",
        "test_opened": True,
        "eval_opened": False,
        "sealed_windows": ["2025+"],
        "test_end_exclusive": TEST_END,
        "audit_spec": AUDIT_SPEC,
        "spec_hash": _spec_hash(),
        "implementation_hash": "x",
        "source_hashes_through_2024": {},
        "feature_hash_through_2024": "x",
        "forest_runs": {},
        "selected_candidate": {},
        "audit_summary": {},
    }
    assert _audit_hash(payload) == _audit_hash(json.loads(json.dumps(payload)))
