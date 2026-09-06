import json
from training import preregister_cross_alt_flow_leadership_relay as prereg


def test_caflr_registration_is_singleton_and_sealed():
    result = prereg.build(); prereg.validate(result)
    assert result["policy_id"] == "CAFLR-6"
    assert result["singleton"] is True
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["policy"]["breadth_min"] == 4
    assert result["policy"]["hold_hours"] == 6


def test_caflr_round_trip(tmp_path):
    result = prereg.build(); path = tmp_path / "registration.json"
    path.write_text(json.dumps(result, allow_nan=False)); prereg.validate(json.loads(path.read_text()))
