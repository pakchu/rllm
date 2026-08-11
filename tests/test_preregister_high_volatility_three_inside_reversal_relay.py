from training import preregister_high_volatility_three_inside_reversal_relay as prereg

def test_boundary() -> None:
    value=prereg.build()
    assert value["policy_id"]=="HV3INSIDE-R10-8" and value["as_of_date"]=="2026-08-11"
    assert value["outcomes_opened"] is False and value["source_incidence_opened"] is False and value["gross9_rows_opened"] is False
    assert value["policy"]["indicator_period_hours"]==13 and value["policy"]["hold_hours"]==8
    assert value["research_boundary"]["candidate_count"]==1 and value["research_boundary"]["grid"] is False and value["research_boundary"]["repair_of_prior_candidate"] is False

def test_hash() -> None:
    value=prereg.build(); core={key:item for key,item in value.items() if key!="manifest_hash"}
    assert value["manifest_hash"]==prereg.canonical_hash(core)
