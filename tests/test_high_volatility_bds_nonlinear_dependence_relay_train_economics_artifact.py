import hashlib
import json
from pathlib import Path

from training import evaluate_high_volatility_bds_nonlinear_dependence_relay_economics as e


RESULT = Path("results/high_volatility_bds_nonlinear_dependence_relay_train_economics_2026-08-13.json")
EXPECTED = "8897d8f8765cd9395c5f4d7ebe9564b98b05f858036e97c6595088a06f106ba3"


def test_terminal_train_rejection_is_immutable_and_later_stages_absent():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED
    result = json.loads(RESULT.read_text())
    manifest_hash = result.pop("manifest_hash")
    assert e.canonical_hash(result) == manifest_hash
    assert result["policy_id"] == "HVBDS-8"
    assert result["stage"] == "train"
    assert not result["passed"]
    assert result["decision"] == "terminal_reject_no_repair"
    assert not result["later_stage_outcomes_opened"]
    base = result["primary"]["base"]
    assert base["trades"] == 20
    assert base["absolute_return_pct"] < 0
    assert base["mean_gross_underlying_bp"] < 20
    assert base["cagr_to_strict_mdd"] < 3
    assert result["primary"]["stress"]["absolute_return_pct"] < 0
    assert result["primary"]["cluster_signflip"]["pvalue"] > 0.1
    assert all(
        half["absolute_return_pct"] < 0
        for half in result["primary"]["calendar_halves"].values()
    )
    for stage in ("test", "eval", "final"):
        assert not Path(
            f"results/high_volatility_bds_nonlinear_dependence_relay_{stage}_economics_2026-08-13.json"
        ).exists()
