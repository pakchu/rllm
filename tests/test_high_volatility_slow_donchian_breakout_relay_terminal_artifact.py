import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_slow_donchian_breakout_relay_train_economics_2026-08-10.json"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvsdbr_train_rejection_is_terminal_and_exact():
    report = json.loads(RESULT.read_text())
    assert report["stage"] == "train"
    assert report["passed"] is False
    assert report["decision"] == "terminal_reject_no_repair"
    base = report["primary"]["base"]
    stress = report["primary"]["stress"]
    assert base["absolute_return_pct"] == 0.9898893456946611
    assert base["mean_gross_underlying_bp"] == 28.469652956240044
    assert base["cagr_to_strict_mdd"] == 0.19441037701594432
    assert stress["absolute_return_pct"] == 0.42693196138019474
    assert stress["cagr_to_strict_mdd"] == 0.08117721879213209
    assert report["primary"]["cluster_signflip"]["pvalue"] == 0.42265577344226557
    assert report["checks"]["mean_gross_move_min_20bp"] is True
    assert report["checks"]["stress_absolute_return_positive"] is True
    assert report["checks"]["each_calendar_half_positive"] is False
    assert report["later_stage_outcomes_opened"] is False
    assert sha256(RESULT) == "f95b46a30c150a1afc1ac1dcd5a8f08a5abee74cad9e27cccc78bc9da52efd1b"


def test_hvsdbr_later_stages_remain_sealed():
    for stage in ("test", "eval", "final"):
        path = Path(
            f"results/high_volatility_slow_donchian_breakout_relay_{stage}_economics_2026-08-10.json"
        )
        assert not path.exists()
