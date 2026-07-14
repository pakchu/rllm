from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import diagnose_carta_transfer_failure as diagnose


def _row(value: str, follow: float, fade: float) -> dict[str, object]:
    tokens = {
        field: value
        for field in diagnose.TOKEN_COLUMNS
    }
    return {
        "tokens": tokens,
        "action_outcomes": {
            "FOLLOW": {"utility": follow},
            "FADE": {"utility": fade},
        },
    }


def test_frozen_carta_rejection_keeps_gemma_and_oos_closed() -> None:
    result = diagnose._verify_rejection()
    assert result["selection"]["rejected"] is True
    assert result["selection"]["gemma_stage_allowed"] is False
    assert result["protocol"]["sealed_windows"] == [
        "test2024",
        "eval2025",
        "ytd2026",
    ]


def test_identical_supported_effects_transfer_with_unit_correlation() -> None:
    rows = [_row("A", 0.02, -0.01), _row("B", -0.01, 0.02)] * 3
    effects = diagnose.token_action_effects(rows, minimum_count=1)
    transfer = diagnose.effect_transfer(effects, effects)
    assert transfer["shared_supported_token_action_cells"] > 1
    assert transfer["pearson_effect_correlation"] == pytest.approx(1.0)
    assert transfer["effect_sign_agreement"] == pytest.approx(1.0)


def test_diagnostic_result_does_not_promote_recent_history_variants() -> None:
    result = json.loads(
        Path("results/carta_transfer_failure_diagnostic_2026-07-14.json").read_text()
    )
    assert result["protocol"]["may_repair_or_promote_carta"] is False
    assert result["protocol"]["sealed_windows_still_unopened"] == [
        "test2024",
        "eval2025",
        "ytd2026",
    ]
    for item in result["recent_history_model_transfer_to_2023"].values():
        assert item["relational_ridge"]["absolute_return_pct"] < 0.0
        assert item["naive_bayes"]["absolute_return_pct"] < 0.0
