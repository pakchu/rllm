from __future__ import annotations

import json
from pathlib import Path

from training.develop_causal_residual_expert_switcher_pre2026 import (
    OUTPUT,
    SCRIPT_PATH,
    TEST_PATH,
    sha256_file,
)


def test_committed_cres_development_artifact_is_bound_to_frozen_harness() -> None:
    result = json.loads(Path(OUTPUT).read_text())
    assert result["research_status"] == (
        "development_only_2023_2025_seen_2026_outcomes_unopened"
    )
    assert result["development_gate"]["passes"] is True
    assert result["attestation"]["script_sha256"] == sha256_file(SCRIPT_PATH)
    assert result["attestation"]["test_sha256"] == sha256_file(TEST_PATH)
    assert result["primary"]["combined_2024_2025"]["cagr_to_strict_mdd"] >= 3.0
    assert result["primary"]["combined_2024_2025"]["trades"] >= 40
    assert all(row["entry_time"] < "2026-01-01" for row in result["selected_trade_rows"])
