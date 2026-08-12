import hashlib
import json

from training import evaluate_high_volatility_large_ticket_temporal_clustering_relay_economics as economics


EXPECTED_SHA256 = "0e1430fc96d3f31cfc2404e628394dc6307ef68b57df00c353b219e98b2fd7af"


def test_train_rejection_is_immutable_and_later_stages_remain_sealed() -> None:
    path = economics.OUTPUTS["train"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(path.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert economics.canonical_hash(payload) == manifest_hash
    assert payload["stage"] == "train"
    assert payload["passed"] is False
    assert payload["decision"] == "terminal_reject_no_repair"
    assert payload["later_stage_outcomes_opened"] is False
    assert payload["primary"]["base"]["absolute_return_pct"] < 0
    assert payload["primary"]["base"]["mean_gross_underlying_bp"] < 20
    assert payload["primary"]["cluster_signflip"]["pvalue"] > 0.1
    assert payload["primary"]["calendar_halves"]["second"]["absolute_return_pct"] < 0
    assert not economics.OUTPUTS["test"].exists()
    assert not economics.OUTPUTS["eval"].exists()
    assert not economics.OUTPUTS["final"].exists()
