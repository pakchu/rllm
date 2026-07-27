from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tests.test_backtest_all_alpha_month import EXPECTED_ALPHAS

RESULT = Path(
    "results/all_alpha_july_2026_performance_2026-07-27.json"
)


def test_all_alpha_july_artifact_is_complete_and_comparable() -> None:
    report = json.loads(RESULT.read_text())
    assert report["mode"] == "all_frozen_atomic_alpha_completed_bar_monthly_replay"
    assert report["accounting_version"] == "same_btc_low_high_v1"
    assert report["retrospective_not_pristine_oos"] is True
    assert report["config"]["env_path"] == "<redacted>"
    assert report["window"] == {
        "requested_start": "2026-07-01 00:00:00",
        "requested_end_exclusive": "2026-08-01 00:00:00",
        "start": "2026-07-01 00:00:00",
        "end_exclusive": "2026-07-27 15:00:00",
        "last_completed_bar": "2026-07-27 14:55:00",
        "bars": 7668,
        "calendar_days": 26.625,
    }
    assert (
        report["data_quality"]["window_market_hash"]
        == "d4dcd50aeabe600eb8a1bb37f5348e019ebc1c469db074ae11c3176bb88f70e2"
    )
    assert set(report["metrics"]) == EXPECTED_ALPHAS
    assert report["inventory"]["scored_atomic_alphas"] == 24
    assert report["inventory"]["unaccounted_atomic_files"] == []
    assert report["data_quality"]["asof_cap"] == {
        "asof": "2026-07-27 15:03:00+00:00",
        "completed_end_exclusive": "2026-07-27 15:00:00+00:00",
        "market_rows_discarded_after_asof": 0,
        "funding_rows_discarded_after_asof": 0,
    }
    activity = report["activity_flow_frozen_fit"]
    assert activity["fit_window"] == "2020-01-01/2024-01-01"
    assert activity["fit_rows"] == 420768
    assert max(abs(value) for value in activity["threshold_drift"].values()) < 5e-4


def test_promoted_gross8_exactly_matches_the_frozen_monthly_replay() -> None:
    report = json.loads(RESULT.read_text())
    portfolio = report["promoted_gross8"]
    assert np.isclose(portfolio["absolute_return_pct"], -1.0403933929998055)
    assert np.isclose(portfolio["strict_mdd_pct"], 2.0359030824293245)
    assert portfolio["trades"] == 6
    assert portfolio["longs"] == 0
    assert portfolio["shorts"] == 6
    assert portfolio["trades_by_sleeve"] == {
        "fresh_kimchi_fx": 0,
        "frozen_annual_rank7": 0,
        "rex_taker_low_range_position": 5,
        "cand_rex_veto_7": 1,
        "markov_transition_long": 0,
    }


def test_atomic_metrics_and_signal_diagnostics_are_consistent() -> None:
    report = json.loads(RESULT.read_text())
    for name, metric in report["metrics"].items():
        diagnostic = report["signal_diagnostics"][name]
        assert metric["trades"] == diagnostic["accepted_trades"]
        assert metric["trades"] == metric["longs"] + metric["shorts"]
        assert np.isfinite(metric["absolute_return_pct"])
        assert np.isfinite(metric["strict_mdd_pct"])
        assert metric["strict_mdd_pct"] >= 0.0

    high = report["metrics"]["oi_divergence_highfreq"]
    selector = report["metrics"]["oi_divergence_highfreq_selector"]
    assert {
        key: value for key, value in high.items() if key != "trades_by_sleeve"
    } == {
        key: value
        for key, value in selector.items()
        if key != "trades_by_sleeve"
    }
    assert [
        "oi_divergence_highfreq",
        "oi_divergence_highfreq_selector",
    ] in report["duplicate_groups"]["exact_signal"]
    assert report["duplicate_groups"]["exact_path"] == [
        [
            "oi_divergence_highfreq",
            "oi_divergence_highfreq_selector",
        ]
    ]


def test_legacy_rex_weekend_exception_does_not_change_july_signals() -> None:
    report = json.loads(RESULT.read_text())
    for row in report["legacy_rex_availability_sensitivity"].values():
        assert row["weekend_fallback_signal_rows"] == 0


def test_all_alpha_artifact_pins_every_replay_source() -> None:
    report = json.loads(RESULT.read_text())
    for raw_path, expected in report["source_sha256"].items():
        path = Path(raw_path)
        assert path.is_file(), raw_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
