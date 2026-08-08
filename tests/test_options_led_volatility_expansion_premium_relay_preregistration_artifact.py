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
        "871e96d99454648dac1f0dacb0bf9c3c6cf06602198d3e1d5d55ac68c243a482"
    )
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    p.validate_manifest(report)
    assert report["manifest_hash"] == (
        "0f6070ef618e3506ac3fe955cdcffea8ff220b6c09720af599933a70df1eda49"
    )
    assert report == p.build_manifest()
    assert report["outcomes_opened"] is False
    assert report["research_boundary"]["outcome_files_opened"] == 0
