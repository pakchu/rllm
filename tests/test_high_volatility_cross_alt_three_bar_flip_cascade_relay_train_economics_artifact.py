import hashlib, json
from pathlib import Path

from training import evaluate_high_volatility_cross_alt_three_bar_flip_cascade_relay_economics as e


RESULT = Path("results/high_volatility_cross_alt_three_bar_flip_cascade_relay_train_economics_2026-08-13.json")
EXPECTED = "fe19cf40c7c19ec003ff2deae031f1bf1caf30f979357ddfa3e7c785616b57d7"


def test_terminal_train_rejection_is_immutable_and_later_stages_absent():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED
    payload = json.loads(RESULT.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert e.canonical_hash(payload) == manifest_hash
    assert payload["policy_id"] == "HVCAFC-6"
    assert payload["stage"] == "train"
    assert not payload["passed"]
    assert payload["decision"] == "terminal_reject_no_repair"
    assert not payload["later_stage_outcomes_opened"]
    base = payload["primary"]["base"]
    assert base["trades"] == 33
    assert base["absolute_return_pct"] < 0
    assert base["mean_gross_underlying_bp"] < 20
    assert payload["primary"]["stress"]["absolute_return_pct"] < 0
    assert payload["primary"]["cluster_signflip"]["pvalue"] > 0.1
    assert all(v["absolute_return_pct"] < 0 for v in payload["primary"]["calendar_halves"].values())
    for stage in ("test", "eval", "final"):
        assert not Path(
            f"results/high_volatility_cross_alt_three_bar_flip_cascade_relay_{stage}_economics_2026-08-13.json"
        ).exists()
