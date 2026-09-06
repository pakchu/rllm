import hashlib
import json

from training import evaluate_deribit_expansion_partial_absorption_economics as evaluator


RESULT = evaluator.OUTPUTS["train"]


def test_depar_train_economics_is_frozen_terminal_without_future_open():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "be10360651ad94f997b58966073675d0293203ae875f36c1dccbd2f1f2f62c0a"
    )
    payload = json.loads(RESULT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}

    assert payload["manifest_hash"] == evaluator.canonical_hash(core)
    assert payload["passed"] is False
    assert payload["decision"] == "terminal_reject_no_repair"
    assert payload["later_stage_outcomes_opened"] is False

    assert payload["primary"]["base"]["absolute_return_pct"] > 0
    assert payload["primary"]["base"]["cagr_to_strict_mdd"] < 3
    assert payload["primary"]["cluster_signflip"]["pvalue"] > 0.1
    assert payload["primary"]["calendar_halves"]["second"]["absolute_return_pct"] < 0
    assert payload["primary"]["stress"]["cagr_to_strict_mdd"] < 2.5

    assert not evaluator.OUTPUTS["test"].exists()
    assert not evaluator.OUTPUTS["eval"].exists()
    assert not evaluator.OUTPUTS["final"].exists()
