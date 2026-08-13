import hashlib, json
from pathlib import Path

from training import evaluate_high_volatility_cross_alt_realized_quarticity_concentration_spillover_relay_economics as e


RESULT = Path("results/high_volatility_cross_alt_realized_quarticity_concentration_spillover_relay_train_economics_2026-08-13.json")
EXPECTED = "4ca33fdac67e6b18e09e5c99772742bd8265ee02be84739cc61441256cb40210"


def test_terminal_train_rejection_is_immutable_and_later_stages_absent():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED
    payload = json.loads(RESULT.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert e.canonical_hash(payload) == manifest_hash
    assert payload["policy_id"] == "HVCARQ-6"
    assert payload["stage"] == "train" and not payload["passed"]
    assert payload["decision"] == "terminal_reject_no_repair"
    assert not payload["later_stage_outcomes_opened"]
    base = payload["primary"]["base"]
    assert base["trades"] == 28
    assert base["absolute_return_pct"] > 0 and base["mean_gross_underlying_bp"] >= 20
    assert base["cagr_to_strict_mdd"] < 3
    assert payload["primary"]["stress"]["absolute_return_pct"] > 0
    assert payload["primary"]["stress"]["cagr_to_strict_mdd"] < 2.5
    assert payload["primary"]["cluster_signflip"]["pvalue"] > 0.1
    assert all(v["absolute_return_pct"] > 0 for v in payload["primary"]["calendar_halves"].values())
    for stage in ("test", "eval", "final"):
        assert not Path(f"results/high_volatility_cross_alt_realized_quarticity_concentration_spillover_relay_{stage}_economics_2026-08-13.json").exists()
