from __future__ import annotations

import json
from pathlib import Path

from training.download_bitmex_trollbox_attention import canonical_hash, sha256_file


RESULT = Path(
    "results/bitmex_trollbox_semantic_synthetic_controls_2026-07-20.json"
)
RESULT_FILE_SHA256 = (
    "c34abd3e9e38d52c8e98a75793b5c215e70021c7c451ce3e832ddc9e43539c5e"
)


def test_semantic_synthetic_pass_is_hash_bound_and_market_blind() -> None:
    assert sha256_file(RESULT) == RESULT_FILE_SHA256
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in result.items()
        if key not in {"result_hash", "created_at"}
    }

    assert result["result_hash"] == canonical_hash(core)
    assert result["contract_hash"] == (
        "4515018d89bdbad48c44092073ebf2e2dd882c6ecfee2ba993eb7d681e4405f7"
    )
    assert result["passed"] is True
    assert result["private_text_opened"] is False
    assert result["market_or_outcomes_opened"] is False
    assert result["contract"]["market_or_outcomes_opened"] is False
    assert result["contract"]["semantic_revision"] == (
        "v3_direction_neutral_meta_instruction_guard"
    )
    assert result["contract"]["meta_instruction_guard"][
        "directional_output"
    ] is False
    assert len(result["controls"]) == 13
    assert all(control["passed"] for control in result["controls"])
    assert sum(
        control["meta_instruction_guarded"]
        for control in result["controls"]
    ) == 3
    assert sum(
        control["decision_source"] == "gemma2"
        for control in result["controls"]
    ) == 10
    assert all(result["numeric_controls"].values())
