import hashlib
import json
from pathlib import Path

from training import build_high_volatility_three_inside_reversal_relay_support as support


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_rejection_is_terminal_and_outcome_sealed() -> None:
    result = json.loads(support.RESULT.read_text())
    assert result["policy_id"] == "HV3INSIDE-R10-8"
    assert result["support_passed"] is False
    assert result["decision"] == "terminal_source_support_reject"
    assert result["advance_to_gross9_novelty"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["funding_values_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert [result["support"][stage]["events"] for stage in ("train", "test", "eval", "final")] == [7, 11, 10, 9]
    failed = {name for name, passed in result["support_checks"].items() if not passed}
    assert failed == {"train_minimum_events", "test_minimum_events", "eval_minimum_events", "final_month_concentration"}


def test_rejection_artifacts_are_hash_bound() -> None:
    result = json.loads(support.RESULT.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == support.canonical_hash(core)
    assert result["clock"]["sha256"] == sha(Path(result["clock"]["path"]))
    assert sha(support.RESULT) == "707281673dd7a312443d63e1dffb2d8bd7eee0992c5761e8171d4e75b6499cf4"
    for item in result["controls"].values():
        assert item["promotion_authorized"] is False
        assert item["sha256"] == sha(Path(item["path"]))
