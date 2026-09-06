from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import preregister_options_led_volatility_expansion_premium_relay as p


ARTIFACT = Path(
    "results/options_led_volatility_expansion_premium_relay_"
    "preregistration_2026-08-08.json"
)


def test_ovepr_preregistration_artifact_is_canonical_frozen_and_outcome_blind() -> None:
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "180e6be7f6889024896303be511a07b3a95b44dc225f4566a2edab7127022dd6"
    )
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    p.validate_manifest(report)
    assert report["manifest_hash"] == (
        "0433453fd6895045e031f62e4c20f5b3591b2b74a9d36aad2ea6b18f2a0f932f"
    )
    assert report == p.build_manifest()
    assert report["outcomes_opened"] is False
    assert report["research_boundary"]["outcome_files_opened"] == 0
