import hashlib,json
from training import evaluate_equity_commodity_volatility_residual_shock_relay_economics as economics


def test_ecvrs_train_rejection_is_terminal_and_sequential():
    path=economics.OUTPUTS["train"]
    assert hashlib.sha256(path.read_bytes()).hexdigest()=="7755434a6ebc1cde88d2b4716d0b832f7639903e4c79e9304887cab6d070b6c4"
    result=json.loads(path.read_text())
    assert result["policy_id"]=="ECVRS-12"
    assert result["stage"]=="train"
    assert result["passed"] is False
    assert result["decision"]=="terminal_reject_no_repair"
    assert result["later_stage_outcomes_opened"] is False
    assert not economics.OUTPUTS["test"].exists()
    assert not economics.OUTPUTS["eval"].exists()
    assert not economics.OUTPUTS["final"].exists()


def test_ecvrs_train_metrics_and_manifest_are_frozen():
    result=json.loads(economics.OUTPUTS["train"].read_text())
    assert result["manifest_hash"]==economics.canonical_hash({key:value for key,value in result.items() if key!="manifest_hash"})
    base=result["primary"]["base"]
    assert base["trades"]==9
    assert base["absolute_return_pct"]==-2.778170302364047
    assert base["mean_gross_underlying_bp"]==-50.2899879781186
    assert result["primary"]["cluster_signflip"]["pvalue"]==0.8599314006859932
    assert result["checks"]["strict_mdd_max_15"] is True
    assert sum(result["checks"].values())==1
