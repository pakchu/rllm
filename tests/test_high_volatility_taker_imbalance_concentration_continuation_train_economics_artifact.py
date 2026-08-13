import hashlib
import json
from pathlib import Path

from training import evaluate_high_volatility_taker_imbalance_concentration_continuation_economics as economics


RESULT = Path("results/high_volatility_taker_imbalance_concentration_continuation_train_economics_2026-08-13.json")


def test_terminal_train_rejection_is_sealed_and_later_stages_absent() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == "91a06ce5f554ac7c16aad4995dc04700a2c36e241ee1c49ae3e76ee5df5c8c0f"
    report = json.loads(RESULT.read_text())
    manifest_hash = report.pop("manifest_hash")
    assert economics.canonical_hash(report) == manifest_hash
    assert report["policy_id"] == "HVTICC-8" and report["stage"] == "train"
    assert not report["passed"] and report["decision"] == "terminal_reject_no_repair"
    assert not report["later_stage_outcomes_opened"]
    base = report["primary"]["base"]
    assert base["trades"] == 62
    assert base["absolute_return_pct"] < 0
    assert base["mean_gross_underlying_bp"] < 20
    assert base["cagr_to_strict_mdd"] < 3
    assert report["primary"]["stress"]["absolute_return_pct"] < 0
    assert report["primary"]["cluster_signflip"]["pvalue"] > 0.1
    assert report["primary"]["calendar_halves"]["first"]["absolute_return_pct"] < 0
    assert report["primary"]["calendar_halves"]["second"]["absolute_return_pct"] < 0
    for stage in ("test", "eval", "final"):
        assert not Path(
            f"results/high_volatility_taker_imbalance_concentration_continuation_{stage}_economics_2026-08-13.json"
        ).exists()
