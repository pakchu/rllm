import json

from training import preregister_cross_alt_breadth_exhaustion_reversal as prereg


def test_caber_preregistration_is_singleton_and_outcome_blind():
    result = prereg.build()
    prereg.validate(result)
    assert result["policy_id"] == "CABER-12"
    assert result["singleton"] is True
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["policy"]["breadth_min"] == 5
    assert result["policy"]["variation_rank_min"] == 0.65


def test_caber_round_trip(tmp_path):
    result = prereg.build()
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(result, allow_nan=False))
    prereg.validate(json.loads(path.read_text()))
