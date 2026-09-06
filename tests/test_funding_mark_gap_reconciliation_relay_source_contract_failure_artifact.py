import hashlib
import json
from pathlib import Path

from training import build_funding_mark_gap_reconciliation_relay_support as support


ARTIFACT = Path(
    "results/funding_mark_gap_reconciliation_relay_source_contract_failure_2026-08-09.json"
)


def test_fmgrr_source_contract_failure_is_terminal_and_outcome_sealed():
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "1a06dc27ce04036390a4ec4d437a059e0a188828f12bea4e06f5fbe8e615dfcf"
    )
    result = json.loads(ARTIFACT.read_text())
    assert result["policy_id"] == "FMGRR-6"
    assert result["failure_stage"] == "source_contract_validation"
    assert result["target_0800_nonpositive_mark_price_rows"] == 303
    assert result["support_passed"] is False
    assert result["advance_to_gross9_novelty"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["decision"] == "terminal_source_contract_reject_no_repair"
    assert result["candidate_incidence_opened"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert not support.RESULT.exists()
    assert not support.CLOCK.exists()


def test_fmgrr_source_failure_manifest_is_hash_bound():
    result = json.loads(ARTIFACT.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == support.chash(core)
    evaluator = Path(result["source_evaluator"]["path"])
    assert result["source_evaluator"]["sha256"] == support.sha(evaluator)
