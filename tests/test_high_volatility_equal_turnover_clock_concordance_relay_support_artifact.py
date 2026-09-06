import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_equal_turnover_clock_concordance_relay_support_2026-08-13.json"
)
CLOCK = Path(
    "data/high_volatility_equal_turnover_clock_concordance_relay_clocks_2023_2026.csv.gz"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_terminal_source_rejection_is_sealed_and_outcome_blind() -> None:
    report = json.loads(RESULT.read_text())
    assert sha(RESULT) == "2cd1628b8ca1f7ec25c192f87b02bb107db746594fcfd8baab5fba1df3d1d9e7"
    assert report["policy_id"] == "HVETCC-8"
    assert not report["support_passed"]
    assert report["decision"] == "terminal_source_support_reject"
    assert not report["advance_to_gross9_novelty"]
    assert not report["advance_to_economic_outcomes"]
    assert not report["postentry_return_pnl_execution_price_opened"]
    assert not report["gross9_rows_opened"]


def test_only_frozen_final_month_concentration_fails_and_hashes_are_bound() -> None:
    report = json.loads(RESULT.read_text())
    assert {name: values["events"] for name, values in report["support"].items()} == {
        "train": 18,
        "test": 46,
        "eval": 43,
        "final": 18,
    }
    failed = {name for name, passed in report["support_checks"].items() if not passed}
    assert failed == {"final_month_concentration"}
    assert report["support"]["final"]["max_month_share"] == 0.5
    assert report["clock"] == {
        "path": str(CLOCK),
        "sha256": "4daf19af8df6abe3631be3ea67b98d9fee3bb4452858b7740dd3256531c1517e",
        "rows": 125,
    }
    assert report["source_manifest"]["sha256"] == (
        "cb6ae478148260bdbaeac135915cee5ea7cfc401c08b4f6a16a0a1be28947c5e"
    )
    assert not Path(
        "results/high_volatility_equal_turnover_clock_concordance_relay_gross9_novelty_2026-08-13.json"
    ).exists()
