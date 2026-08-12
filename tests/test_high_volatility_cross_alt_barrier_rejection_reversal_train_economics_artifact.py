import hashlib
import json

from training import evaluate_high_volatility_cross_alt_barrier_rejection_reversal_economics as e


EXPECTED = "bcfcadd5a215f8dbc3ec3f22f220a85ba183246f4ea7a530195ef668a6a228f9"


def test_train_rejection_is_immutable_and_later_stages_sealed():
    path = e.OUTPUTS["train"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED
    result = json.loads(path.read_text())
    manifest_hash = result.pop("manifest_hash")
    assert e.canonical_hash(result) == manifest_hash
    assert result["stage"] == "train"
    assert not result["passed"]
    assert result["decision"] == "terminal_reject_no_repair"
    assert not result["later_stage_outcomes_opened"]
    assert not result["advance_to_next_stage"]
    assert not result["advance_to_post_stage_volatility_audit"]
    assert result["primary"]["base"]["absolute_return_pct"] < 0
    assert result["primary"]["base"]["mean_gross_underlying_bp"] < 0
    assert result["primary"]["stress"]["absolute_return_pct"] < 0
    assert not e.OUTPUTS["test"].exists()
    assert not e.OUTPUTS["eval"].exists()
    assert not e.OUTPUTS["final"].exists()
