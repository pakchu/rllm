from __future__ import annotations

import json
from pathlib import Path

from training import evaluate_funding_adjusted_delivery_carry as evaluate


ARTIFACT = Path("results/funding_adjusted_delivery_carry_stage1_2021_2022_2026-07-17.json")
EXPECTED_FILE_SHA256 = "313e55175ae797eca0103871aaa777734f113962af9eea38943022e6bb3c2898"
EXPECTED_MANIFEST_HASH = "9c6043f579cb2e3678f53d543d821a7d520706df7a8cf7f3c817dcaecc591038"


def test_stage1_result_is_hash_bound_and_rejected() -> None:
    report = json.loads(ARTIFACT.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert evaluate._sha256(ARTIFACT) == EXPECTED_FILE_SHA256
    assert evaluate._canonical_hash(core) == EXPECTED_MANIFEST_HASH
    assert report["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert report["evaluation_source_sha256"] == evaluate._sha256(
        evaluate.EVALUATOR_SOURCE
    )
    assert report["gate_passed"] is False
    assert report["disposition"] == (
        "REJECT_STAGE1_KEEP_2023_AND_2024_PLUS_SEALED"
    )
    assert report["opened_windows"] == ["stage1_2021_2022"]
    assert report["sealed_windows"] == ["stage2_2023", "2024", "2025", "2026_ytd"]
    assert not evaluate.STAGE2_OUTPUT.exists()


def test_stage1_statistics_and_physical_cutoff_are_frozen() -> None:
    report = json.loads(ARTIFACT.read_text())
    base = report["base_cost"]
    assert base["absolute_return_pct"] == -4.9901975579324525
    assert base["cagr_pct"] == -2.6481651815023155
    assert base["strict_mdd_pct"] == 27.547325303976923
    assert base["cagr_to_strict_mdd"] == -0.09613148108865611
    assert base["trades"] == 30
    assert report["stress_cost"]["absolute_return_pct"] == -7.253794191675089
    assert report["yearly"]["2021_partial"]["absolute_return_pct"] < 0.0
    assert report["yearly"]["2022"]["absolute_return_pct"] < 0.0
    assert all(value < 0.0 for value in report["entry_cohort_half_returns_pct"].values())
    assert all(value is False for value in report["gates"].values())
    diagnostics = report["execution_diagnostics"]
    assert diagnostics["cutoff"] == "2023-01-01T00:00:00+00:00"
    assert diagnostics["last_market_time"] == "2022-12-31T23:55:00+00:00"


def test_controls_falsify_the_funding_adjusted_direction_mapping() -> None:
    report = json.loads(ARTIFACT.read_text())
    controls = report["controls"]
    assert controls["direction_flip"]["absolute_return_pct"] < 0.0
    assert controls["one_funding_event_delay"]["absolute_return_pct"] < 0.0
    assert controls["zero_funding"]["absolute_return_pct"] < 0.0
    assert controls["basis_only"]["absolute_return_pct"] > 0.0
    assert controls["basis_only"]["cagr_to_strict_mdd"] < 0.10
    assert controls["constant_long_perp_short_quarter"]["absolute_return_pct"] > 0.0
    assert controls["constant_long_perp_short_quarter"]["cagr_to_strict_mdd"] < 0.10
