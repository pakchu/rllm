from __future__ import annotations

import json

import pandas as pd

from training import evaluate_aggregate_fill_compression_sweep as evaluate


EXPECTED_FILE_SHA256 = (
    "8bd769a58cc9dc0b23226e752a7804d8f51ca24f3db9710a0293c55a53ccf771"
)
EXPECTED_MANIFEST_HASH = (
    "44d4ba1f8e512b08a0d8b9b44ab91d774bb2907503f029c413cb84d5872ce02a"
)


def _report() -> dict:
    return json.loads(evaluate.STAGE1_OUTPUT.read_text())


def test_stage1_result_is_hash_bound_and_rejected_before_2023() -> None:
    report = _report()
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert evaluate._sha256(evaluate.STAGE1_OUTPUT) == EXPECTED_FILE_SHA256
    assert evaluate._canonical_hash(core) == EXPECTED_MANIFEST_HASH
    assert report["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert report["candidate_id"] == "AFCS-144"
    assert report["stage"] == "stage1_2020_2022"
    assert report["stage1_qualifies"] is False
    assert report["next_action"] == "reject_keep_2023_and_2024plus_sealed"
    assert report["sealed_after_run"] == ["2023", "2024", "2025", "2026_ytd"]
    assert not evaluate.STAGE2_OUTPUT.exists()


def test_stage1_statistics_and_strict_accounting_are_frozen() -> None:
    report = _report()
    base = report["base"]
    assert base["absolute_return_pct"] == -4.033628474302464
    assert base["cagr_pct"] == -1.362728331383356
    assert base["strict_mdd_pct"] == 21.835849769209915
    assert base["cagr_to_strict_mdd"] == -0.062407845162266086
    assert base["trade_count"] == 421
    assert base["long_count"] == 217
    assert base["short_count"] == 204
    assert base["mean_gross_underlying_move_bp"] == 11.199090677721856
    assert report["stress_10bp"]["absolute_return_pct"] == -18.907851456536996
    assert report["annual"]["2020"]["absolute_return_pct"] > 0.0
    assert report["annual"]["2021"]["absolute_return_pct"] < 0.0
    assert report["annual"]["2022"]["absolute_return_pct"] < 0.0
    assert report["half_year"]["2022_h2"]["absolute_return_pct"] < -10.0
    assert sum(report["gate"].values()) == 2


def test_stage1_parser_stopped_physically_before_2023() -> None:
    source = _report()["source"]
    cutoff = pd.Timestamp("2023-01-01")
    assert source["cutoff"] == cutoff.isoformat()
    assert source["market_rows_parsed"] == 315_648
    assert source["funding_rows_parsed"] == 3_288
    assert pd.Timestamp(source["last_market_time"]) < cutoff
    assert pd.Timestamp(source["last_funding_time"]) < cutoff


def test_component_controls_do_not_support_the_frozen_mechanism() -> None:
    report = _report()
    primary = report["base"]["cagr_to_strict_mdd"]
    controls = report["controls"]
    assert controls["no_aligned_response"]["cagr_to_strict_mdd"] > primary
    assert controls["direction_flip"]["absolute_return_pct"] < 0.0
    assert controls["one_hour_signal_delay"]["absolute_return_pct"] < 0.0
    assert controls["one_day_shifted_clock"]["absolute_return_pct"] < 0.0
    assert controls["random_side"]["absolute_return_pct"] < 0.0
    assert report["passing_rejection_placebos"] == []
