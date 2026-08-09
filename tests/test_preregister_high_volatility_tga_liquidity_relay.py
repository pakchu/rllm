from training import preregister_high_volatility_tga_liquidity_relay as prereg


def test_outcome_blind_singleton_before_source(tmp_path, monkeypatch):
    report = prereg.build()
    monkeypatch.setattr(prereg, "SOURCE", tmp_path / "missing.csv.gz")
    prereg.validate(report)
    assert report["policy_id"] == "HVTGAL-24"
    assert report["singleton"] is True
    assert report["outcomes_opened"] is False
    assert report["source_incidence_opened"] is False
    assert report["gross9_rows_opened"] is False
