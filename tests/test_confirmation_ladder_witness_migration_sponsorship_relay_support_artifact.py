import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/confirmation_ladder_witness_migration_sponsorship_relay_support_2026-08-13.json"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_support_pass_and_closed_outcomes():
    report = json.loads(RESULT.read_text())
    assert report["policy_id"] == "CLWMSR-6"
    assert report["support_passed"] is True
    assert report["advance_to_gross9_novelty"] is True
    assert report["advance_to_economic_outcomes"] is False
    assert report["execution_prices_opened"] is False
    assert report["postentry_return_or_pnl_opened"] is False
    assert report["gross9_rows_opened"] is False
    assert report["rv20_opened"] is False
    assert all(report["support_checks"].values())


def test_bound_artifact_hashes_and_counts():
    report = json.loads(RESULT.read_text())
    assert sha(Path(report["clock"]["path"])) == report["clock"]["sha256"]
    assert report["clock"]["rows"] == 125
    assert {name: values["events"] for name, values in report["support"].items()} == {
        "train": 24, "test": 50, "eval": 32, "final": 19,
    }
    for control in report["controls"].values():
        assert sha(Path(control["path"])) == control["sha256"]
        assert control["promotion_authorized"] is False
