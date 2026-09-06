import json

from training import preregister_high_volatility_alt_breadth_diffusion_slope_relay as p


def test_manifest_is_canonical_and_blind():
    payload = p.build()
    p.validate(payload)
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == p.canonical_hash(core)
    assert payload["policy_id"] == "HVABDS-8"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["policy"]["slope_rank_min"] == 0.80
    assert payload["policy"]["variation_rank_min"] == 0.65
    assert payload["clock"]["hold"] == "8 elapsed hours"
    assert json.dumps(payload, allow_nan=False)


def test_stopping_rule_forbids_repairs():
    payload = p.build()
    assert payload["research_boundary"]["grid"] is False
    assert payload["research_boundary"]["repair_of_prior_candidate"] is False
    assert payload["diagnostic_controls"]["cannot_be_promoted"] is True
    assert "no universe" in payload["stopping_rule"]
