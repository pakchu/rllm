import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_crypto_market_mode_ignition_relay_support_2026-08-10.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_support_passes_without_opening_outcomes():
    payload = json.loads(RESULT.read_text())
    assert payload["support_passed"] is True
    assert payload["advance_to_gross9_novelty"] is True
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["funding_values_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert all(payload["support_checks"].values())
    assert payload["clock"]["rows"] == 261
    assert sha(Path(payload["clock"]["path"])) == payload["clock"]["sha256"]


def test_source_support_counts_and_hashes_are_frozen():
    payload = json.loads(RESULT.read_text())
    assert {name: values["events"] for name, values in payload["support"].items()} == {
        "train": 57, "test": 80, "eval": 83, "final": 41,
    }
    assert sha(RESULT) == "ca2e22d1dd96d90a015fdb3024b64571edeb94e38d0aed02ae91bb28359b382c"
    assert payload["source_manifest"]["sha256"] == "1a596983cc26a825af61db3d5b46a65fc4b12eb88c98709cb4f7fbb18e765dd4"
    assert payload["clock"]["sha256"] == "faa4f8c09b107c6846f0d8143aa740d9f5e665c4c628696af7fc535f572c3aab"
    assert all(not item["promotion_authorized"] for item in payload["controls"].values())
