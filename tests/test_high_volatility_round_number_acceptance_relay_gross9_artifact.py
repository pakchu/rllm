import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_round_number_acceptance_relay_gross9_novelty_2026-08-12.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvrnar_gross9_artifact_is_terminal_and_hash_bound():
    report = json.loads(RESULT.read_text())
    assert report["policy_id"] == "HVRNAR-8"
    assert report["source_support_passed"] is True
    assert report["every_gross9_sleeve_passed"] is False
    assert report["gross9_novelty_status"] == "failed"
    assert report["advance_to_economic_outcomes"] is False
    assert report["evidence_boundary"]["outcomes_opened"] is False
    assert report["evidence_boundary"]["economic_outcome_rows_opened"] == 0
    assert report["gross9_sleeves"]["fresh_kimchi_fx"]["passed"] is False
    assert report["gross9_sleeves"]["markov_transition_long"]["passed"] is False
    assert sha256(Path(report["source_support"]["path"])) == report["source_support"]["sha256"]
    assert sha256(Path(report["preregistration"]["path"])) == report["preregistration"]["sha256"]
