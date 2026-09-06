import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_sample_entropy_collapse_continuation_support_2026-08-13.json")
CLOCK = Path("data/high_volatility_sample_entropy_collapse_continuation_clocks_2023_2026.csv.gz")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_support_pass_is_hash_sealed_and_outcome_blind() -> None:
    payload = json.loads(RESULT.read_text())
    assert sha(RESULT) == "3e84b5b54d336622bd7728356882c47daca902d8cb8dc37213d81fe412ac3298"
    assert payload["policy_id"] == "HVSENC-8"
    assert payload["support_passed"] is True
    assert payload["decision"] == "pass_to_novelty"
    assert payload["advance_to_gross9_novelty"] is True
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False


def test_all_stage_source_gates_pass() -> None:
    payload = json.loads(RESULT.read_text())
    assert all(payload["support_checks"].values())
    assert {name: values["events"] for name, values in payload["support"].items()} == {
        "train": 29, "test": 84, "eval": 90, "final": 32,
    }
    assert payload["clock"] == {
        "path": str(CLOCK),
        "sha256": "3a7a71d53ffc460ea975819194ec102fae155f0b5a2c8cb2866ac07cd510d10f",
        "rows": 235,
    }


def test_source_manifest_and_clock_match_bound_hashes() -> None:
    payload = json.loads(RESULT.read_text())
    assert payload["source_manifest"]["sha256"] == (
        "b65c075520a63915e4fe3e1c7b12a8bed32df875e678ac2046b613c1fa47f4f2"
    )
    assert payload["source_manifest"]["sha256"] == sha(Path(payload["source_manifest"]["path"]))
    assert payload["clock"]["sha256"] == sha(CLOCK)
