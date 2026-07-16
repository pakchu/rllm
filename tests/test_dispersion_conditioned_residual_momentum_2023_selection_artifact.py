from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training.preregister_dispersion_conditioned_residual_momentum import canonical_hash


RESULT = Path("results/dispersion_conditioned_residual_momentum_selection_2023_2026-07-17.json")
EXPECTED_SHA256 = "f7975edef28a9b361c8fd2d7f392639eff223d5711fb2f040452f639aa32f2e0"


def test_2023_result_is_locked_and_future_windows_stay_sealed() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(RESULT.read_text())
    body = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    assert canonical_hash(body) == payload["manifest_hash"]
    assert payload["opened_window"] == ["2023-01-01", "2024-01-01"]
    assert payload["2024_test_opened"] is False
    assert payload["2025_eval_opened"] is False
    assert payload["2026_holdout_opened"] is False
    assert payload["decision"] == "rejected_before_2024"


def test_dcrm_is_rejected_on_strict_risk_and_significance() -> None:
    payload = json.loads(RESULT.read_text())
    evaluation = payload["evaluation"]
    primary = evaluation["primary"]["2023"]
    assert primary["absolute_return_pct"] == pytest.approx(2.225243533158916)
    assert primary["cagr_pct"] == pytest.approx(2.226784517682767)
    assert primary["strict_mdd_pct"] == pytest.approx(24.619124301374363)
    assert primary["cagr_to_strict_mdd"] == pytest.approx(0.09044937953209232)
    assert primary["trades"] == 38
    assert evaluation["weekly_cluster_signflip"]["raw_p_value"] == pytest.approx(
        0.4615269236538173
    )
    assert evaluation["passes_2023_selection"] is False
    assert evaluation["selection_gates"]["2023_h1_absolute_return_positive"] is False
    assert evaluation["selection_gates"]["ten_bp_stress_absolute_return_positive"] is False
