from training import preregister_high_volatility_qqq_gold_rotation_relay as prereg
def test_outcome_blind_before_download(tmp_path,monkeypatch):
 p=prereg.build();monkeypatch.setattr(prereg,"SOURCE",tmp_path/"missing.csv.gz");prereg.validate(p);assert p["policy_id"]=="HVQGR-12" and p["singleton"];assert not p["outcomes_opened"] and not p["source_incidence_opened"] and not p["gross9_rows_opened"]
