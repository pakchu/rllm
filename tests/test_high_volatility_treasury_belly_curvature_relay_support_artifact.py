import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_treasury_belly_curvature_relay_support_2026-08-10.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_support_artifact_passes_without_opening_outcomes():
    payload = json.loads(RESULT.read_text())
    assert payload["support_passed"] is True
    assert payload["advance_to_gross9_novelty"] is True
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert all(payload["support_checks"].values())
    assert payload["clock"]["rows"] == 256
    assert sha(Path(payload["clock"]["path"])) == payload["clock"]["sha256"]


def test_source_support_split_counts_and_hashes_are_frozen():
    payload = json.loads(RESULT.read_text())
    assert {name: values["events"] for name, values in payload["support"].items()} == {
        "train": 32, "test": 98, "eval": 82, "final": 44
    }
    assert sha(RESULT) == "cdbbfff8b9be04e6e11a261f5bdb6cf2eb6d2d7586dbef53858c0c9a3f726b87"
    assert payload["source_state"]["sha256"] == "46d64f7c9d42f9e25eab406da2e29454855ff3cd867ff54eecf8cb7a4b4dce6b"
    assert payload["clock"]["sha256"] == "4d458f6ae879d15e3184c057f6c85bd089c7a8117e7c35420644fa39e3c18886"
    assert all(not item["promotion_authorized"] for item in payload["controls"].values())
