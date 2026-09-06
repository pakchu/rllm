import pytest

from training import audit_high_volatility_geomagnetic_activity_rotation_source_contract as a


def test_status_contract_rejects_definitive_substitution():
    audit = a.validate_status_contract({"datetime": ["a", "b"], "Kp": [1.0, 2.0], "status": ["def", "def"]})
    assert audit == {
        "rows": 2,
        "status_counts": {"def": 2},
        "all_rows_preserve_preregistered_nowcast_status": False,
    }


def test_status_contract_accepts_only_exact_nowcast_status():
    audit = a.validate_status_contract({"datetime": ["a"], "Kp": [1.0], "status": ["nowcast"]})
    assert audit["all_rows_preserve_preregistered_nowcast_status"] is True


def test_status_contract_rejects_mismatched_cardinality():
    with pytest.raises(RuntimeError, match="cardinality"):
        a.validate_status_contract({"datetime": ["a"], "Kp": [1.0], "status": []})
