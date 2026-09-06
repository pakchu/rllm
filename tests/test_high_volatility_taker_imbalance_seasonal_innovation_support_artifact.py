import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_taker_imbalance_seasonal_innovation_support_2026-08-13.json")
CLOCK = Path("data/high_volatility_taker_imbalance_seasonal_innovation_clocks_2023_2026.csv.gz")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_pass_is_sealed_and_outcome_blind() -> None:
    report = json.loads(RESULT.read_text())
    assert sha(RESULT) == "e88b526a95360c59682962c5024c10853c2ce736d038fc5f191c9c234ec1bce9"
    assert report["policy_id"] == "HVTISI-8"
    assert report["support_passed"]
    assert report["decision"] == "pass_to_novelty"
    assert report["advance_to_gross9_novelty"]
    assert not report["advance_to_economic_outcomes"]
    assert not report["postentry_return_pnl_execution_price_opened"]
    assert not report["gross9_rows_opened"]


def test_all_source_gates_and_hashes_pass() -> None:
    report = json.loads(RESULT.read_text())
    assert all(report["support_checks"].values())
    assert {name: values["events"] for name, values in report["support"].items()} == {
        "train": 48,
        "test": 95,
        "eval": 61,
        "final": 43,
    }
    assert report["clock"] == {
        "path": str(CLOCK),
        "sha256": "14411e102623f0ff489e6d38359fe289fc47f11b48333857353c3d4eb68ac68e",
        "rows": 247,
    }
    assert report["source_manifest"]["sha256"] == "1c18ecdbc0e40c9f40cf27cef5299b04ca069e7a2d477701a3b658e021d3dac6"
