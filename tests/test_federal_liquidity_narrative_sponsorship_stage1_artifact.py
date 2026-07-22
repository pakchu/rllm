from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import evaluate_federal_liquidity_narrative_sponsorship_relay as evaluator


STAGE1_FILE_SHA256 = (
    "efbc4eb91d5662d082d6f35a8cca14a366d96b8ba9f0f4994328024d36e4ef0d"
)
STAGE1_MANIFEST_HASH = (
    "ea26f41c20f21b633e088ae77f595bf974da9ffb60e2ee506195ec489f3702a6"
)
FREEZE_MANIFEST_HASH = (
    "09dade9c6e5198465a8480d8559c31f703d5517d9f2b0a58a1c6a87e8c427f50"
)


def _stage1() -> dict[str, object]:
    return json.loads(Path(evaluator.STAGE1_OUTPUT).read_text())


def test_stage1_artifact_is_hash_valid_terminal_rejection() -> None:
    payload = _stage1()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert evaluator._sha256(evaluator.STAGE1_OUTPUT) == STAGE1_FILE_SHA256
    assert payload["manifest_hash"] == STAGE1_MANIFEST_HASH
    assert payload["manifest_hash"] == evaluator._canonical_hash(core)
    assert payload["evaluator_freeze_manifest_hash"] == FREEZE_MANIFEST_HASH
    evaluator._validate_stage1_identity(
        payload, expected_freeze_hash=FREEZE_MANIFEST_HASH
    )
    assert payload["stage1_passed"] is False
    assert payload["advance_to_stage2"] is False
    assert payload["disposition"] == "REJECT_STAGE1_KEEP_2023_AND_LATER_SEALED"


def test_stage1_primary_failed_every_frozen_gate_and_2023_stayed_sealed() -> None:
    payload = _stage1()
    candidate = payload["candidate"]
    primary = candidate["primary"]
    assert primary["absolute_return_pct"] == pytest.approx(-14.097833709311624)
    assert primary["cagr_pct"] == pytest.approx(-4.938121959927755)
    assert primary["strict_mdd_pct"] == pytest.approx(51.54454975059733)
    assert primary["cagr_to_strict_mdd"] == pytest.approx(-0.09580298952694857)
    assert primary["trades"] == 67
    assert primary["monthly_cluster_signflip_p"] == pytest.approx(
        0.5712714364281786
    )
    assert candidate["gates"]
    assert not any(candidate["gates"].values())
    assert payload["2023_outcomes_opened"] is False
    assert payload["2023_execution_rows_parsed"] == 0
    assert payload["2023_funding_rows_parsed"] == 0
    assert payload["sealed_windows"] == ["stage2_2023", "2024", "2025", "2026_ytd"]


def test_stage1_prefixes_match_freeze_and_selection_artifact_does_not_exist() -> None:
    payload = _stage1()
    diagnostics = payload["physical_source_diagnostics"]
    assert diagnostics["market"]["window_line_sha256"] == (
        evaluator.STAGE1_MARKET_WINDOW_LINE_SHA256
    )
    assert diagnostics["funding"]["window_line_sha256"] == (
        evaluator.STAGE1_FUNDING_WINDOW_LINE_SHA256
    )
    assert diagnostics["funding"]["sealed_boundary_values_parsed"] is False
    assert not evaluator.STAGE2_OUTPUT.exists()
