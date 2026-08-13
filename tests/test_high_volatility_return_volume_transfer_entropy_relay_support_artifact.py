import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_return_volume_transfer_entropy_relay_support_2026-08-13.json"
)
CLOCK = Path(
    "data/high_volatility_return_volume_transfer_entropy_relay_clocks_2023_2026.csv.gz"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_support_pass_is_sealed_and_outcome_blind() -> None:
    report = json.loads(RESULT.read_text())
    assert sha(RESULT) == "42548442c88874040f73108a4938a87415e4d81d080846b88ca369e7a57df08a"
    assert report["policy_id"] == "HVRVTE-8"
    assert report["support_passed"]
    assert report["decision"] == "pass_to_novelty"
    assert report["advance_to_gross9_novelty"]
    assert not report["advance_to_economic_outcomes"]
    assert not report["postentry_return_pnl_execution_price_opened"]
    assert not report["gross9_rows_opened"]


def test_all_frozen_support_checks_pass_and_hashes_are_bound() -> None:
    report = json.loads(RESULT.read_text())
    assert all(report["support_checks"].values())
    assert {name: values["events"] for name, values in report["support"].items()} == {
        "train": 31,
        "test": 76,
        "eval": 71,
        "final": 35,
    }
    assert report["clock"] == {
        "path": str(CLOCK),
        "sha256": "e27f50cfe2a9d3eb90a41e23affa75c33c4a06360ce2627fcebe8647dc0b87c6",
        "rows": 213,
    }
    assert report["source_manifest"]["sha256"] == (
        "80510996fdd15bab51b7b155f305a548505a2f2f67946f2ff754034f4c15438e"
    )
