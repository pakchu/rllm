from training import preregister_high_volatility_risk_parity_consensus_relay as prereg
def test_outcome_blind_before_download(tmp_path,monkeypatch):
 report=prereg.build();monkeypatch.setattr(prereg,"SOURCE",tmp_path/"missing.csv.gz");prereg.validate(report)
 assert report["policy_id"]=="HVRPC-24" and report["singleton"]
 assert not report["outcomes_opened"] and not report["source_incidence_opened"] and not report["gross9_rows_opened"]
