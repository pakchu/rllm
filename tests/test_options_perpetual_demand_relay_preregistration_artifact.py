from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import preregister_options_perpetual_demand_relay as prereg


ARTIFACT = Path(
    "results/options_perpetual_demand_relay_preregistration_2026-07-19.json"
)


def test_opdr_preregistration_artifact_is_frozen_and_outcome_blind() -> None:
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "9673fe0fc0cc929514c730a56157f6ed409dd1063486c7df082c215e459ba696"
    )
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    prereg.validate_manifest(report, verify_sources=False)
    assert report["manifest_hash"] == (
        "c5f61217324c51faeb46324ff31906205e2fd71b84fbb1c39b067b2e4ce4cf6c"
    )
    assert report["outcomes_opened"] is False
    assert report["policy"] == prereg.asdict(prereg.Policy())
    assert report["research_history_boundary"]["sealed_candidate_windows"] == [
        "train_2023_h2",
        "test_2024",
        "eval_2025",
        "final_2026_h1",
    ]
