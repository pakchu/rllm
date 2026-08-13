import hashlib
import json
from pathlib import Path

from training import evaluate_high_volatility_taker_imbalance_seasonal_innovation_economics as economics


RESULT = Path("results/high_volatility_taker_imbalance_seasonal_innovation_train_economics_2026-08-13.json")


def test_terminal_train_rejection_is_sealed_and_later_stages_absent() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == "bf45ff10fdde2a06fce58e9b25abc3de412882b1dc377c84e35fd16e3b6bbadd"
    report = json.loads(RESULT.read_text())
    manifest_hash = report.pop("manifest_hash")
    assert economics.canonical_hash(report) == manifest_hash
    assert report["policy_id"] == "HVTISI-8"
    assert report["stage"] == "train"
    assert not report["passed"]
    assert report["decision"] == "terminal_reject_no_repair"
    assert not report["later_stage_outcomes_opened"]
    base = report["primary"]["base"]
    assert base["trades"] == 48
    assert base["absolute_return_pct"] > 0
    assert base["mean_gross_underlying_bp"] >= 20
    assert base["cagr_to_strict_mdd"] < 3
    assert report["primary"]["stress"]["cagr_to_strict_mdd"] < 2.5
    assert report["primary"]["cluster_signflip"]["pvalue"] > 0.1
    assert report["primary"]["calendar_halves"]["first"]["absolute_return_pct"] < 0
    for stage in ("test", "eval", "final"):
        assert not Path(
            f"results/high_volatility_taker_imbalance_seasonal_innovation_{stage}_economics_2026-08-13.json"
        ).exists()
