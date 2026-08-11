from training import preregister_high_volatility_cross_alt_order_flow_tail_asymmetry_relay as prereg


def test_boundary() -> None:
    contract = prereg.build()
    assert contract["policy_id"] == "HVCAOFTAR-8"
    assert contract["outcomes_opened"] is False
    assert contract["source_incidence_opened"] is False
    assert contract["gross9_rows_opened"] is False
    assert contract["singleton"] is True
    assert contract["research_boundary"]["candidate_count"] == 1
    assert contract["research_boundary"]["grid"] is False
    assert contract["research_boundary"]["repair_of_prior_candidate"] is False
    assert contract["policy"]["flow_asymmetry_rank_min"] == 0.90
    assert contract["policy"]["flow_intensity_rank_min"] == 0.65
    assert contract["policy"]["hold_hours"] == 8


def test_hash() -> None:
    contract = prereg.build()
    core = {key: value for key, value in contract.items() if key != "manifest_hash"}
    assert contract["manifest_hash"] == prereg.canonical_hash(core)
