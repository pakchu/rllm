import hashlib
import json
from pathlib import Path

from training import evaluate_high_volatility_sample_entropy_collapse_continuation_economics as economics


RESULT = Path("results/high_volatility_sample_entropy_collapse_continuation_train_economics_2026-08-13.json")
EXPECTED = "d5f982146e4b5b345be9da875b3d2cf93f193d91d68f094745a163c4b1787504"


def test_terminal_train_rejection_is_immutable_and_later_stages_absent() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED
    payload = json.loads(RESULT.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert economics.canonical_hash(payload) == manifest_hash
    assert payload["policy_id"] == "HVSENC-8"
    assert payload["stage"] == "train"
    assert payload["passed"] is False
    assert payload["decision"] == "terminal_reject_no_repair"
    assert payload["later_stage_outcomes_opened"] is False
    base = payload["primary"]["base"]
    assert base["trades"] == 29
    assert base["absolute_return_pct"] < 0
    assert base["mean_gross_underlying_bp"] < 20
    assert base["cagr_to_strict_mdd"] < 3
    assert payload["primary"]["stress"]["absolute_return_pct"] < 0
    assert payload["primary"]["cluster_signflip"]["pvalue"] > 0.1
    assert payload["primary"]["calendar_halves"]["first"]["absolute_return_pct"] < 0
    for stage in ("test", "eval", "final"):
        assert not Path(
            f"results/high_volatility_sample_entropy_collapse_continuation_{stage}_economics_2026-08-13.json"
        ).exists()
