from training import preregister_high_volatility_nfci_weekend_relay as prereg
def test_preregistration_is_outcome_blind_before_vintage_download(tmp_path,monkeypatch):
 p=prereg.build();monkeypatch.setattr(prereg,"SOURCE",tmp_path/"not_downloaded.csv");prereg.validate(p);assert p["policy_id"]=="HVNFCI-72";assert p["singleton"];assert not p["outcomes_opened"];assert not p["source_incidence_opened"];assert not p["gross9_rows_opened"];assert p["features"]["no_latest_vintage_backfill"];assert p["source_plan"]["nfci"]["download_after_preregistration_commit"]
