import hashlib
import json
from pathlib import Path

from training import evaluate_high_volatility_alt_modularity_fragmentation_relay_economics as e


RESULT = Path("results/high_volatility_alt_modularity_fragmentation_relay_train_economics_2026-08-13.json")
EXPECTED = "3e1318e8b56941574b828d239c1db0e4ee3c12056dcdfbe7671364046b5c8366"


def test_terminal_train_rejection_is_immutable_and_later_stages_absent():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED
    result = json.loads(RESULT.read_text())
    manifest_hash = result.pop("manifest_hash")
    assert e.canonical_hash(result) == manifest_hash
    assert result["policy_id"] == "HVAMF-8"
    assert result["stage"] == "train"
    assert not result["passed"]
    assert result["decision"] == "terminal_reject_no_repair"
    assert not result["later_stage_outcomes_opened"]
    base = result["primary"]["base"]
    assert base["trades"] == 21
    assert base["absolute_return_pct"] > 0
    assert base["mean_gross_underlying_bp"] >= 20
    assert base["cagr_to_strict_mdd"] < 3
    assert result["primary"]["stress"]["absolute_return_pct"] > 0
    assert result["primary"]["stress"]["cagr_to_strict_mdd"] < 2.5
    assert result["primary"]["cluster_signflip"]["pvalue"] <= 0.1
    assert result["primary"]["calendar_halves"]["first"]["absolute_return_pct"] < 0
    assert result["primary"]["calendar_halves"]["second"]["absolute_return_pct"] > 0
    for stage in ("test", "eval", "final"):
        assert not Path(
            f"results/high_volatility_alt_modularity_fragmentation_relay_{stage}_economics_2026-08-13.json"
        ).exists()
