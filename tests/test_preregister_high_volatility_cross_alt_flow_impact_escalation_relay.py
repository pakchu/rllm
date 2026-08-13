from training import preregister_high_volatility_cross_alt_flow_impact_escalation_relay as s


def test_blind_canonical_registration():
    payload = s.build()
    s.validate(payload)
    assert payload["manifest_hash"] == s.canonical_hash({k: v for k, v in payload.items() if k != "manifest_hash"})
    assert not payload["outcomes_opened"]
    assert not payload["source_incidence_opened"]
    assert not payload["gross9_rows_opened"]


def test_fixed_escalation_spillover_policy():
    payload = s.build()
    assert len(s.ALTS) == 6
    assert payload["policy"]["path_bars"] == 96
    assert payload["policy"]["escalation_rank_min"] == 0.70
    assert payload["policy"]["minimum_selected_alts"] == 4
    assert payload["policy"]["decision_hours_utc"] == [3, 11, 19]
    assert payload["clock"]["hold"] == "8 elapsed hours"
    assert not payload["research_boundary"]["repair_of_prior_candidate"]
    assert not payload["research_boundary"]["promoted_prior_control"]
