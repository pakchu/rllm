import hashlib,json
from training import evaluate_fx_volatility_sponsored_momentum_relay_gross9_novelty as novelty


def test_fvsmr_novelty_pass_is_outcome_sealed():
    assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest()=="5cbd2e7634d2739c29cb92d516199e5b2120244dbbd1a12889688f5ef6d1aefc"
    result=json.loads(novelty.OUTPUT.read_text())
    assert result["policy_id"]=="FVSMR-12"
    assert result["every_gross9_sleeve_passed"] is True
    assert result["gross9_novelty_status"]=="passed"
    assert result["advance_to_economic_outcomes"] is True
    boundary=result["evidence_boundary"]
    assert boundary["btc_execution_rows_opened"]==0
    assert boundary["funding_rows_opened"]==0
    assert boundary["economic_outcome_rows_opened"]==0
    assert boundary["outcomes_opened"] is False


def test_fvsmr_passes_every_metric():
    result=json.loads(novelty.OUTPUT.read_text())
    for sleeve in result["gross9_sleeves"].values():
        assert sleeve["passed"] is True
        assert all(sleeve["checks"].values())
        assert sleeve["metrics"]["exact_entry_jaccard"]==0.0
