from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import preregister_premium_compression_breakout_relay as prereg


ARTIFACT = Path(
    "results/premium_compression_breakout_relay_preregistration_2026-07-19.json"
)


def test_pcbr_preregistration_artifact_is_frozen_and_outcome_blind() -> None:
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "df3a45cfaf36503793159cb969524803519ef95fd4421d19bfb58d266405bcce"
    )
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    prereg.validate_manifest(report, verify_feature_source=False)
    assert report["manifest_hash"] == (
        "26cc00df2baa24a9df32c37e904155104c52928da89eda2dd9975a252c964dd2"
    )
    assert report["outcomes_opened"] is False
    assert report["policy"] == prereg.asdict(prereg.Policy())
