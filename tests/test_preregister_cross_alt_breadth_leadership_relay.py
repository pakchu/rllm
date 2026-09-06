import json

from training import preregister_cross_alt_breadth_leadership_relay as prereg


def test_cablr_preregistration_is_singleton_and_sealed():
    result = prereg.build()
    prereg.validate(result)
    assert result["policy_id"] == "CABLR-8"
    assert result["singleton"] is True
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["policy"]["breadth_min"] == 4
    assert result["policy"]["hold_hours"] == 8


def test_cablr_round_trip(tmp_path):
    result = prereg.build()
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(result, allow_nan=False))
    prereg.validate(json.loads(path.read_text()))
