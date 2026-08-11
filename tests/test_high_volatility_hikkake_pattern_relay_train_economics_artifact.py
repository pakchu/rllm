import hashlib
import json
from pathlib import Path

from training import evaluate_high_volatility_hikkake_pattern_relay_economics as economics


RESULT = economics.OUTPUTS["train"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_train_failure_is_terminal_and_later_outcomes_remain_sealed() -> None:
    report = json.loads(RESULT.read_text())
    assert report["stage"] == "train"
    assert report["passed"] is False
    assert report["decision"] == "terminal_reject_no_repair"
    assert report["later_stage_outcomes_opened"] is False
    assert report["primary"]["base"]["absolute_return_pct"] == 0.7179654165765603
    assert report["primary"]["base"]["cagr_to_strict_mdd"] == 0.16900969199841523
    assert report["primary"]["base"]["mean_gross_underlying_bp"] == 16.332920357583816
    assert report["primary"]["stress"]["absolute_return_pct"] == -0.9224167550362816
    assert report["primary"]["cluster_signflip"]["pvalue"] == 0.45958540414595855
    assert report["primary"]["calendar_halves"]["first"]["absolute_return_pct"] > 0
    assert report["primary"]["calendar_halves"]["second"]["absolute_return_pct"] > 0
    assert sha(RESULT) == "e2f7381c2ceadfebb3beb7bd506cd12d6ee70ddeb5b25b31a7cb781474e197ea"
    assert not economics.OUTPUTS["test"].exists()
    assert not economics.OUTPUTS["eval"].exists()
    assert not economics.OUTPUTS["final"].exists()


def test_train_report_manifest_is_hash_bound() -> None:
    report = json.loads(RESULT.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == economics.canonical_hash(core)
