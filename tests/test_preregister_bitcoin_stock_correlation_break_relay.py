from training import preregister_bitcoin_stock_correlation_break_relay as prereg


def test_bscbr_is_outcome_blind_singleton() -> None:
    report = prereg.build()
    boundary = report["research_boundary"]
    assert report["policy_id"] == "BSCBR-24"
    assert report["outcomes_opened"] is False
    assert report["source_incidence_opened"] is False
    assert report["singleton"] is True
    assert boundary["candidate_count"] == 1
    assert boundary["grid"] is False
    assert boundary["repair_of_prior_candidate"] is False
    assert boundary["promoted_prior_control"] is False


def test_bscbr_freezes_causal_correlation_break_clock() -> None:
    report = prereg.build()
    policy = report["policy"]
    assert policy["correlation_model"] == "bivariate_gaussian_garch11_dcc11"
    assert policy["correlation_change_abs_min"] == 0.02
    assert policy["variation_midrank_min"] == 0.65
    assert policy["feature_delay_minutes_after_cash_close"] == 5
    assert policy["entry_delay_minutes_after_feature"] == 5
    assert policy["hold_hours"] == 24
    assert report["clock"]["side"].startswith("-sign")
    assert report["clock"]["no_imputation"] is True
    assert report["estimator"]["fit_window"] == ["2020-01-01", "2023-01-01"]
    assert report["estimator"]["optimizer"]["randomness"] is False


def test_bscbr_discloses_research_source_and_nonreplication() -> None:
    report = prereg.build()
    support = report["mechanism"]["external_support"]
    assert support["doi"] == "10.1016/j.jfs.2024.101285"
    assert support["implementation_is_not_a_replication"] is True
    assert "research only" in report["source_plan"]["spy"]


def test_bscbr_freezes_strict_gates_and_controls() -> None:
    report = prereg.build()
    assert report["source_support_gates"]["minimum_events"] == {
        "train": 8,
        "test": 12,
        "eval": 12,
        "final": 8,
    }
    assert report["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert report["economic_gates"]["stress_cagr_to_strict_mdd_min"] == 2.5
    assert report["diagnostic_controls"]["diagnostic_controls_cannot_be_promoted"] is True


def test_bscbr_hash_binds_core() -> None:
    report = prereg.build()
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == prereg.canonical_hash(core)
