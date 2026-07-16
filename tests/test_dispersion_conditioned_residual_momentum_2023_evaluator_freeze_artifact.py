from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import evaluate_dispersion_conditioned_residual_momentum_2023 as evaluate
from training import freeze_dispersion_conditioned_residual_momentum_2023_evaluator as freeze


ARTIFACT = Path(
    "results/dispersion_conditioned_residual_momentum_2023_evaluator_freeze_2026-07-17.json"
)
EXPECTED_SHA256 = "22768578d646fdc252677d7c9a56cc0fc5ed182fa1219b2f9d498f7d94c5366d"


def test_evaluator_freeze_artifact_is_exact_and_outcome_blind() -> None:
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(ARTIFACT.read_text())
    freeze.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["opened_windows"] == []
    assert payload["sealed_windows"] == ["2023", "2024", "2025", "2026"]
    assert payload["market_rows_parsed_during_freeze"] == 0
    assert payload["funding_rows_loaded_during_freeze"] == 0
    assert payload["source_prefix_contract"]["2024_rows_permitted"] == 0


def test_evaluator_accepts_only_the_commit_bound_freeze() -> None:
    payload = evaluate.verify_evaluation_freeze()
    assert payload["evaluation_source_commit"] == "aa236cf052f0a1894eee8d75a9e83cff747238b4"
    assert payload["evaluation_source_sha256"] == hashlib.sha256(
        evaluate.EVALUATION_SOURCE.read_bytes()
    ).hexdigest()
    assert payload["freeze_source_sha256"] == hashlib.sha256(
        evaluate.FREEZE_SOURCE.read_bytes()
    ).hexdigest()
