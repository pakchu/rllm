import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_range_turnover_coupling_relay_train_economics_2026-08-10.json"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvrtcr_train_rejection_is_terminal_and_exact():
    report = json.loads(RESULT.read_text())
    assert report["stage"] == "train"
    assert report["passed"] is False
    assert report["decision"] == "terminal_reject_no_repair"
    base = report["primary"]["base"]
    stress = report["primary"]["stress"]
    assert base["absolute_return_pct"] == -6.618024307604243
    assert base["mean_gross_underlying_bp"] == -20.402931657017724
    assert base["cagr_to_strict_mdd"] == -1.4774928594978924
    assert stress["absolute_return_pct"] == -8.177211949613106
    assert report["primary"]["cluster_signflip"]["pvalue"] == 0.9804401955980441
    assert report["checks"]["absolute_return_positive"] is False
    assert report["checks"]["mean_gross_move_min_20bp"] is False
    assert report["checks"]["each_calendar_half_positive"] is False
    assert report["later_stage_outcomes_opened"] is False
    assert sha256(RESULT) == "e1d2e64bda068c4cc528c570ffae11baf287dc7f4b2319328f8ff05c24ac729c"


def test_hvrtcr_later_stages_remain_sealed():
    for stage in ("test", "eval", "final"):
        path = Path(
            f"results/high_volatility_range_turnover_coupling_relay_{stage}_economics_2026-08-10.json"
        )
        assert not path.exists()
