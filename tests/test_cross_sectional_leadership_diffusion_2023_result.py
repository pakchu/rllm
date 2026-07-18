from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_cross_sectional_leadership_diffusion as cld


RESULT = Path(
    "results/cross_sectional_leadership_diffusion_selection_2023_2026-07-18.json"
)


def test_cld_2023_result_is_rejected_without_opening_later_windows() -> None:
    payload = json.loads(RESULT.read_text())
    body = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    assert cld.canonical_hash(body) == payload["manifest_hash"]
    assert payload["outcomes_opened"] is True
    assert payload["decision"] == "rejected_before_2024"
    assert payload["2024_test_opened"] is False
    assert payload["2025_eval_opened"] is False
    assert payload["2026_holdout_opened"] is False
    assert payload["maximum_loaded_timestamp_exclusive"] == "2024-01-01 00:00:00"


def test_cld_primary_failed_economically_and_statistically() -> None:
    payload = json.loads(RESULT.read_text())
    evaluation = payload["evaluation"]
    annual = evaluation["primary"]["2023"]
    assert annual["absolute_return_pct"] == pytest.approx(-11.523451109168171)
    assert annual["cagr_pct"] == pytest.approx(-11.523451109168171)
    assert annual["strict_mdd_pct"] == pytest.approx(13.420211502405277)
    assert annual["cagr_to_strict_mdd"] == pytest.approx(-0.8586638971452012)
    assert annual["trades"] == 106
    assert annual["mean_net_bps"] == pytest.approx(-11.439478577983882)
    assert all(
        evaluation["primary"][name]["absolute_return_pct"] < 0.0
        for name in ("h1", "h2", "q1", "q2", "q3", "q4")
    )
    assert evaluation["direction_flip"]["absolute_return_pct"] < 0.0
    assert evaluation["weekly_cluster_signflip"]["p_value_one_sided"] > 0.99
    assert evaluation["passes_2023_selection"] is False
