import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_intraday_hour_reversal_train_economics_2026-08-11.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvihr_train_failure_is_terminal():
    artifact = json.loads(RESULT.read_text())
    assert sha(RESULT) == "50bee8771f69ff7f7bd2c6cde80482cfd9f5367a243b592d6383d1972c5ef3da"
    assert artifact["policy_id"] == "HVIHR-1"
    assert artifact["stage"] == "train"
    assert artifact["passed"] is False
    assert artifact["advance_to_next_stage"] is False
    assert artifact["advance_to_post_stage_volatility_audit"] is False
    assert artifact["decision"] == "terminal_reject_no_repair"
    assert artifact["later_stage_outcomes_opened"] is False
    assert artifact["primary"]["base"]["absolute_return_pct"] < 0
    assert artifact["primary"]["base"]["mean_gross_underlying_bp"] < 0
    assert artifact["primary"]["stress"]["absolute_return_pct"] < 0
    assert artifact["primary"]["cluster_signflip"]["pvalue"] > 0.9
    assert artifact["primary"]["calendar_halves"]["second"]["absolute_return_pct"] < 0
    assert not Path("results/high_volatility_intraday_hour_reversal_test_economics_2026-08-11.json").exists()
