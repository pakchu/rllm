import hashlib,json
from training import evaluate_dollar_factor_shock_relay_economics as economics


def test_dfsr_train_rejection_is_terminal_and_sequential():
    path=economics.OUTPUTS["train"]
    assert hashlib.sha256(path.read_bytes()).hexdigest()=="31156f4afeba3ea6cd956fe8f48e997ff1d608fb3eb26a3eb702002455b10d31"
    result=json.loads(path.read_text())
    assert result["policy_id"]=="DFSR-12"
    assert result["stage"]=="train"
    assert result["passed"] is False
    assert result["decision"]=="terminal_reject_no_repair"
    assert result["later_stage_outcomes_opened"] is False
    assert not economics.OUTPUTS["test"].exists()
    assert not economics.OUTPUTS["eval"].exists()
    assert not economics.OUTPUTS["final"].exists()


def test_dfsr_train_passes_six_of_eight_gates_but_fails_stress_and_halves():
    result=json.loads(economics.OUTPUTS["train"].read_text())
    assert result["manifest_hash"]==economics.canonical_hash({key:value for key,value in result.items() if key!="manifest_hash"})
    base=result["primary"]["base"]
    assert base["cagr_to_strict_mdd"]==3.50039759121863
    assert base["mean_gross_underlying_bp"]==45.260509937927736
    assert result["primary"]["cluster_signflip"]["pvalue"]==0.04964950350496495
    assert result["primary"]["stress"]["cagr_to_strict_mdd"]==2.4447965150777438
    assert result["primary"]["calendar_halves"]["first"]["absolute_return_pct"]==-0.3363719010393451
    assert sum(result["checks"].values())==6
