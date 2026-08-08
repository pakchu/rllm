from training import preregister_quarter_hour_opening_imbalance_relay as prereg


def test_qhoir_is_outcome_blind_singleton() -> None:
    report = prereg.build()
    boundary = report["research_boundary"]
    assert report["policy_id"] == "QHOIR-8"
    assert report["outcomes_opened"] is False
    assert report["source_incidence_opened"] is False
    assert report["singleton"] is True
    assert boundary["candidate_count"] == 1
    assert boundary["grid"] is False
    assert boundary["repair_of_prior_candidate"] is False


def test_qhoir_freezes_quarter_hour_causal_clock() -> None:
    report = prereg.build()
    policy = report["policy"]
    assert policy["quarter_hour_minutes"] == [0, 15, 30, 45]
    assert policy["imbalance_abs_min"] == 0.0
    assert policy["feature_delay_minutes"] == 1
    assert policy["entry_delay_minutes_from_boundary"] == 5
    assert policy["hold_hours"] == 8
    assert report["clock"]["side"].startswith("sign")


def test_qhoir_discloses_full_minute_adaptation() -> None:
    report = prereg.build()
    support = report["mechanism"]["external_support"]
    assert support["arxiv"] == "2607.09426v2"
    assert support["implementation_is_not_a_replication"] is True
    assert "full first minute" in support["untested_adaptation"]


def test_qhoir_freezes_high_vol_and_terminal_gates() -> None:
    report = prereg.build()
    policy = report["policy"]
    assert policy["variation_prior_observations"] == 8640
    assert policy["variation_prior_min_observations"] == 5760
    assert policy["variation_midrank_min"] == 0.65
    assert report["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert report["diagnostic_controls"]["phase_controls_are_falsification_only"] is True


def test_qhoir_hash_binds_core() -> None:
    report = prereg.build()
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == prereg.canonical_hash(core)
