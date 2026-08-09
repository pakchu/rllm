from training import preregister_high_volatility_initial_claims_labor_relay as prereg


def test_preregistration_is_outcome_blind_before_vintage_download(tmp_path, monkeypatch):
    policy = prereg.build()
    monkeypatch.setattr(prereg, "SOURCE", tmp_path / "not_downloaded.csv")
    prereg.validate(policy)
    assert policy["policy_id"] == "HVICLR-72"
    assert policy["singleton"]
    assert not policy["outcomes_opened"]
    assert not policy["source_incidence_opened"]
    assert not policy["gross9_rows_opened"]
    assert policy["features"]["no_latest_vintage_backfill"]
    assert policy["source_plan"]["claims"]["download_after_preregistration_commit"]
    assert policy["diagnostic_controls"]["cannot_be_promoted"]
