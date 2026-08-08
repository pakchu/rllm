import hashlib
import json
from pathlib import Path

from training import evaluate_quarter_hour_opening_imbalance_relay_gross9_novelty as novelty


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_qhoir_novelty_failure_is_terminal_and_outcome_sealed():
    result = json.loads(novelty.OUTPUT.read_text())
    assert result["policy_id"] == "QHOIR-8"
    assert result["source_support_passed"] is True
    assert result["every_gross9_sleeve_passed"] is False
    assert result["gross9_novelty_status"] == "failed"
    assert result["advance_to_economic_outcomes"] is False
    assert result["failure_action"] == "reject QHOIR-8 unchanged before economic outcomes"
    boundary = result["evidence_boundary"]
    assert boundary["btc_execution_rows_opened"] == 0
    assert boundary["btc_price_or_return_rows_opened"] == 0
    assert boundary["funding_rows_opened"] == 0
    assert boundary["economic_outcome_rows_opened"] == 0
    assert boundary["portfolio_return_or_pnl_metrics_computed"] is False
    assert boundary["outcomes_opened"] is False


def test_qhoir_fails_only_the_frozen_near_six_hour_gate():
    result = json.loads(novelty.OUTPUT.read_text())
    for sleeve in result["gross9_sleeves"].values():
        failed = {name for name, passed in sleeve["checks"].items() if not passed}
        assert failed == {"one_to_one_6h_max_matched_share"}
        assert sleeve["metrics"]["one_to_one_6h_max_matched_share"] > 0.35
        assert sleeve["metrics"]["exact_entry_jaccard"] == 0.0


def test_qhoir_novelty_artifact_is_hash_bound():
    result = json.loads(novelty.OUTPUT.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == novelty.canonical_hash(core)
    assert result["preregistration"]["sha256"] == sha(novelty.PREREG)
    assert result["source_support"]["sha256"] == sha(novelty.SUPPORT)
    gross9 = result["gross9_structural_clocks"]
    assert gross9["sha256"] == sha(Path(gross9["path"]))
