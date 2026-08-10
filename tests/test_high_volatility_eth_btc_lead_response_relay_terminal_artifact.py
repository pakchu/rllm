import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_eth_btc_lead_response_relay_train_economics_2026-08-10.json"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hveblr_train_rejection_is_terminal_and_exact():
    report = json.loads(RESULT.read_text())
    assert report["stage"] == "train"
    assert report["passed"] is False
    assert report["decision"] == "terminal_reject_no_repair"
    base = report["primary"]["base"]
    stress = report["primary"]["stress"]
    assert base["absolute_return_pct"] == 0.907124103555379
    assert base["mean_gross_underlying_bp"] == 17.21408327585538
    assert base["cagr_to_strict_mdd"] == 0.22035762305673268
    assert stress["absolute_return_pct"] == -0.8168892994701249
    assert report["primary"]["cluster_signflip"]["pvalue"] == 0.45708542914570854
    assert report["checks"]["mean_gross_move_min_20bp"] is False
    assert report["checks"]["stress_absolute_return_positive"] is False
    assert report["checks"]["each_calendar_half_positive"] is True
    assert report["later_stage_outcomes_opened"] is False
    assert sha256(RESULT) == "63f6783fd274461c119e03edc1d5bf684b0e1f18de0e854832660f98dcb3c490"


def test_hveblr_later_stages_remain_sealed():
    for stage in ("test", "eval", "final"):
        path = Path(
            f"results/high_volatility_eth_btc_lead_response_relay_{stage}_economics_2026-08-10.json"
        )
        assert not path.exists()
