import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_wick_response_transfer_relay_gross9_novelty_2026-08-10.json")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvwrtr_novelty_rejection_is_frozen_and_terminal():
    report = json.loads(RESULT.read_text())
    assert report["every_gross9_sleeve_passed"] is False
    assert report["advance_to_economic_outcomes"] is False
    assert report["evidence_boundary"]["outcomes_opened"] is False
    failed = {name for name, value in report["gross9_sleeves"].items() if not value["passed"]}
    assert failed == {"cand_rex_veto_7", "fresh_kimchi_fx", "frozen_annual_rank7"}
    assert report["gross9_sleeves"]["frozen_annual_rank7"]["metrics"]["one_to_one_6h_max_matched_share"] == 0.4482758620689655
    assert sha256(RESULT) == "22f1fae6a4f39796ef4f3a1fd354d2707a66649f03190b96ef28b334c5427306"


def test_hvwrtr_economics_remain_sealed():
    for stage in ("train", "test", "eval", "final"):
        assert not Path(f"results/high_volatility_wick_response_transfer_relay_{stage}_economics_2026-08-10.json").exists()
