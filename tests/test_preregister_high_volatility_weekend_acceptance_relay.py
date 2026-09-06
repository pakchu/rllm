import json

from training import preregister_high_volatility_weekend_acceptance_relay as prereg


def test_manifest_is_deterministic_and_outcome_blind():
    first, second = prereg.build(), prereg.build()
    assert first == second
    prereg.validate(first)
    assert not first["source_incidence_opened"]
    assert not first["current_candidate_outcomes_opened"]
    assert not first["gross9_rows_opened"]
    assert first["policy"]["range_rank_min"] == .65
    assert first["clock"]["hold"] == "24 elapsed hours"
    assert set(first["diagnostic_controls"]["definitions"]) == {
        "no_range_gate", "no_acceptance_gate", "central_close_rejection",
        "sunday_only_geometry", "direction_flip", "same_clock_forced_long",
    }


def test_serialized_manifest_round_trip(tmp_path):
    payload = prereg.build()
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(payload, allow_nan=False))
    loaded = json.loads(path.read_text())
    prereg.validate(loaded)
    assert loaded["diagnostic_controls"]["cannot_be_promoted"]
