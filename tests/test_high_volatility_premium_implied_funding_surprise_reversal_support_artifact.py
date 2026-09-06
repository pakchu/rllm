import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_premium_implied_funding_surprise_reversal_support_2026-08-13.json")
CLOCK = Path("data/high_volatility_premium_implied_funding_surprise_reversal_clocks_2023_2026.csv.gz")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_support_artifact_is_terminal_and_outcome_blind() -> None:
    payload = json.loads(RESULT.read_text())
    assert sha(RESULT) == "dd55a7e370ca2f354b1e0a0dbe130a3e588cf7746fbf697df46c34e8c60cfb20"
    assert payload["policy_id"] == "HVPIFSR-8"
    assert payload["support_passed"] is False
    assert payload["decision"] == "terminal_source_support_reject"
    assert payload["advance_to_gross9_novelty"] is False
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False


def test_only_final_month_concentration_fails() -> None:
    payload = json.loads(RESULT.read_text())
    failed = {name for name, passed in payload["support_checks"].items() if not passed}
    assert failed == {"final_month_concentration"}
    assert payload["support"]["final"] == {
        "events": 17,
        "longs": 9,
        "shorts": 8,
        "minority_side_share": 8 / 17,
        "max_month_share": 8 / 17,
    }
    assert payload["clock"] == {
        "path": str(CLOCK),
        "sha256": "16a924563881f4f8e4f4928bc6993d586c57660380063f20d9c99252c25df7fa",
        "rows": 130,
    }


def test_source_bindings_are_hash_sealed() -> None:
    payload = json.loads(RESULT.read_text())
    assert payload["source_manifest"]["sha256"] == (
        "b85f1e1bd6a1bcac91af6b0e6753ba6348c8df1fa6e0b7d1bf15b3845d9368d1"
    )
    assert payload["source_manifest"]["sha256"] == sha(Path(payload["source_manifest"]["path"]))
    assert payload["clock"]["sha256"] == sha(CLOCK)
