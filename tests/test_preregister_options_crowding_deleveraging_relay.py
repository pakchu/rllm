from training import preregister_options_crowding_deleveraging_relay as p


def test_preregistration_is_outcome_blind_and_singleton() -> None:
    report = p.build()
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == p.canonical_hash(core)
    assert report["outcomes_opened"] is False
    assert report["research_boundary"]["candidate_count"] == 1
    assert report["research_boundary"]["threshold_hold_direction_grid"] is False


def test_candidate_targets_crowded_volatility_unwind_without_premium_direction() -> None:
    report = p.build()
    clock = report["causal_clock"]
    assert "negative sign" in report["mechanism"]["side"]
    assert "premium" not in clock["oi_change"].lower()
    assert report["policy"]["hold_hours"] == 12
    assert report["economic_gates"]["stop_on_first_failure"] is True
