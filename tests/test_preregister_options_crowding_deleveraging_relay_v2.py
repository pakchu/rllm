from training import preregister_options_crowding_deleveraging_relay_v2 as p


def test_v2_changes_only_historical_oi_availability_contract() -> None:
    report = p.build()
    assert report["policy"]["policy_id"] == "OCDR-12A"
    assert report["research_boundary"]["mechanism_threshold_side_hold_changed_from_v1"] is False
    assert "ts+5m" in report["causal_clock"]["oi_archive_availability"]
    assert report["research_boundary"]["v2_candidate_incidence_opened"] is False
    assert report["outcomes_opened"] is False
