import json

from training import preregister_high_volatility_thursday_macro_reaction_fade as prereg


def test_preregistration_is_deterministic_outcome_blind_and_bound():
    first, second = prereg.build(), prereg.build()
    assert first == second
    prereg.validate(first)
    assert not first["outcomes_opened"]
    assert not first["source_incidence_opened"]
    assert not first["gross9_rows_opened"]
    assert first["policy"]["variation_rank_min"] == .65
    assert first["clock"]["hold"] == "6 elapsed hours"


def test_serialization_and_control_freeze(tmp_path):
    payload = prereg.build(); path = tmp_path / "prereg.json"
    path.write_text(json.dumps(payload, allow_nan=False))
    loaded = json.loads(path.read_text()); prereg.validate(loaded)
    assert set(loaded["diagnostic_controls"]["definitions"]) == {"no_variation_gate", "reaction_continuation", "half_hour_reaction_fade", "one_week_stale_reaction", "direction_flip", "same_clock_forced_long"}
    assert loaded["diagnostic_controls"]["cannot_be_promoted"]
