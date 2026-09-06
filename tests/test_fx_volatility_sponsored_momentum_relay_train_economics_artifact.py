import hashlib
import json

from training import evaluate_fx_volatility_sponsored_momentum_relay_economics as economics


def test_fvsmr_train_rejection_is_terminal_and_sequential():
    path = economics.OUTPUTS["train"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "bda38e9fe4050584c9a67a457f21a2498c14cb0df16160d6c50ae394b7366fe0"
    result = json.loads(path.read_text())
    assert result["policy_id"] == "FVSMR-12"
    assert result["stage"] == "train"
    assert result["passed"] is False
    assert result["decision"] == "terminal_reject_no_repair"
    assert result["later_stage_outcomes_opened"] is False
    assert not economics.OUTPUTS["test"].exists()
    assert not economics.OUTPUTS["eval"].exists()
    assert not economics.OUTPUTS["final"].exists()


def test_fvsmr_train_fails_seven_of_eight_economic_gates():
    result = json.loads(economics.OUTPUTS["train"].read_text())
    assert result["manifest_hash"] == economics.canonical_hash(
        {key: value for key, value in result.items() if key != "manifest_hash"}
    )
    base = result["primary"]["base"]
    assert base["absolute_return_pct"] == -1.170175195758938
    assert base["cagr_to_strict_mdd"] == -1.0104325813538875
    assert base["strict_mdd_pct"] == 2.2856279272223
    assert base["mean_gross_underlying_bp"] == -5.561627117798443
    assert result["primary"]["cluster_signflip"]["pvalue"] == 0.7732922670773292
    assert result["primary"]["stress"]["absolute_return_pct"] == -1.7226298058452771
    assert result["primary"]["calendar_halves"]["first"]["absolute_return_pct"] == -0.3363719010393451
    assert result["primary"]["calendar_halves"]["second"]["absolute_return_pct"] == -0.8366174407093196
    assert sum(result["checks"].values()) == 1
