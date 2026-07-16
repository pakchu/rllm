from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.preregister_post_funding_crowding_release_episode_v2 import canonical_hash


RESULT = Path(
    "results/post_funding_crowding_release_episode_v2_selection_2023_2024_2026-07-17.json"
)
EXPECTED_SHA256 = "5f2400aec0d78903be28624a20e34306ade7d06df7271a928c45b518f1d8d3a4"


def test_pfcr2_selection_artifact_is_locked_and_rejected() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(RESULT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert canonical_hash(core) == payload["manifest_hash"]
    assert payload["outcomes_opened"] is True
    assert payload["opened_windows"] == ["2023", "2024"]
    assert payload["sealed_windows"] == ["2025", "2026"]
    assert payload["decision"] == "rejected_before_2025_no_outcome_repair"
    assert payload["evaluation"]["passes_2023_2024_selection"] is False


def test_pfcr2_failed_both_profitability_and_robustness() -> None:
    evaluation = json.loads(RESULT.read_text())["evaluation"]
    combined = evaluation["primary"]["combined_2023_2024"]
    assert combined["absolute_return_pct"] < 0.0
    assert combined["cagr_to_strict_mdd"] < 0.0
    assert combined["trades"] == 82
    assert evaluation["ten_bp_notional_side_cost_stress"]["absolute_return_pct"] < 0.0
    assert evaluation["entry_delay_plus_5m"]["absolute_return_pct"] < 0.0
    assert evaluation["weekly_cluster_signflip"]["raw_p_value"] > 0.10
