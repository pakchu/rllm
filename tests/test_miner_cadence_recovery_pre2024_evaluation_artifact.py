from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training import evaluate_miner_cadence_recovery_pre2024 as evaluate


RESULT = Path("results/miner_cadence_recovery_pre2024_evaluation_2026-07-17.json")
EXPECTED_SHA256 = "c7c3100847b3318fb0b2976a985042594be2b30086ab13ca032c77bc3c41e74f"
EXPECTED_MANIFEST_HASH = (
    "c5e9f3ecb7d7d5b24e01bc7ebcb4cce1c463e97d83bb722035466109d6b03e09"
)


def test_result_is_frozen_and_rejected_before_orthogonality() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(RESULT.read_text())
    evaluate.validate_result_hash(payload)
    assert payload["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert payload["qualification"]["qualifies"] is False
    assert payload["selection"] == {
        "selected_alpha": None,
        "performance_candidate": None,
        "rejected": True,
        "status": "rejected_before_orthogonality",
        "orthogonality_evaluated": False,
        "promotion_ready": False,
    }


def test_primary_failure_and_absolute_returns_are_frozen() -> None:
    payload = json.loads(RESULT.read_text())
    primary = payload["windows"]["primary"]
    expected = {
        "train": (-37.28171817773881, -22.426358460055518, 46.257781493634354),
        "train2021": (2.0031239839019888, 2.3955958648729236, 32.75344880804758),
        "train2022": (-36.53627339234256, -36.556035426466984, 39.99444760716065),
        "select2023": (8.870100523058566, 8.876437935181979, 14.558246875458824),
        "select2023_h1": (12.417345474399589, 26.642479577627643, 7.734667473089996),
        "select2023_h2": (-4.199330740474427, -8.16345316166396, 12.085847196246513),
    }
    for name, (absolute, cagr, strict_mdd) in expected.items():
        metrics = primary[name]["base_6bp"]
        assert metrics["absolute_return_pct"] == pytest.approx(absolute)
        assert metrics["cagr_pct"] == pytest.approx(cagr)
        assert metrics["strict_mdd_pct"] == pytest.approx(strict_mdd)


def test_2024_was_never_parsed_or_opened() -> None:
    payload = json.loads(RESULT.read_text())
    protocol = payload["protocol"]
    market = payload["source"]["market"]
    assert protocol["sealed_windows"] == ["2024", "2025", "2026_ytd"]
    assert market["physical_parse_boundary"] == (
        "stop before parsing first date >= 2024-01-01"
    )
    assert market["last_date"] == "2023-12-31 23:55:00"
    assert payload["selection"]["orthogonality_evaluated"] is False


def test_random_and_constant_long_diagnostics_expose_regime_beta() -> None:
    payload = json.loads(RESULT.read_text())
    windows = payload["windows"]
    assert windows["random_clock"]["train"]["base_6bp"][
        "absolute_return_pct"
    ] == pytest.approx(29.27678888969172)
    assert windows["random_clock"]["select2023"]["base_6bp"][
        "absolute_return_pct"
    ] == pytest.approx(28.245948152248257)
    assert windows["constant_long"]["select2023"]["base_6bp"][
        "absolute_return_pct"
    ] == pytest.approx(52.179217111844764)
