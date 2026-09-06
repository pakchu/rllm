import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_turnover_lead_response_relay_train_economics_2026-08-10.json"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvtlrr_train_rejection_is_terminal_and_exact():
    report = json.loads(RESULT.read_text())
    assert report["stage"] == "train"
    assert report["passed"] is False
    assert report["decision"] == "terminal_reject_no_repair"
    base = report["primary"]["base"]
    stress = report["primary"]["stress"]
    assert base["absolute_return_pct"] > 0
    assert base["mean_gross_underlying_bp"] >= 20
    assert base["strict_mdd_pct"] <= 15
    assert base["cagr_to_strict_mdd"] == 0.2400875977904739
    assert stress["cagr_to_strict_mdd"] == 0.021180521323003014
    assert report["primary"]["cluster_signflip"]["pvalue"] == 0.4366256337436626
    assert report["checks"]["cagr_to_strict_mdd_min_3"] is False
    assert report["checks"]["cluster_signflip_p_max_0_1"] is False
    assert report["checks"]["stress_cagr_to_strict_mdd_min_2_5"] is False
    assert report["checks"]["each_calendar_half_positive"] is False
    assert report["later_stage_outcomes_opened"] is False
    assert sha256(RESULT) == "c3543b4bd6e39270115f47ed74fb49d99626ebfb5b5edc759ac4ceee0328e372"


def test_hvtlrr_later_stages_remain_sealed():
    for stage in ("test", "eval", "final"):
        path = Path(
            f"results/high_volatility_turnover_lead_response_relay_{stage}_economics_2026-08-10.json"
        )
        assert not path.exists()
