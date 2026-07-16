import numpy as np
import pandas as pd
import pytest

from training.audit_portfolio_shadow_signal_parity import (
    REQUIRED_HISTORY_BARS,
    REQUIRED_LOOKBACK_MINUTES,
    _resolve_existing,
    compare_integer_vectors,
    compare_signals,
    historical_schedule,
    interval_slots,
    vector_gate_pass,
)


def test_timestamp_stride_matches_historical_anchor_inside_eligible_domain():
    dates = pd.date_range("2019-12-31 15:00:00", periods=800, freq="5min")
    expected, domain = historical_schedule(len(dates), hold_bars=288, stride_bars=6)
    actual = interval_slots(dates, stride_bars=6, stride_offset_bars=5)
    report = compare_signals(dates, expected, actual, domain)

    assert report["passed"] is True
    assert report["mismatch_count"] == 0
    assert report["expected"]["sha256"] == report["actual"]["sha256"]


def test_vector_gate_fails_closed_when_oi_value_is_stale():
    frame = pd.DataFrame(
        {
            "oi_zscore": [0.0, 0.0, 2.0, np.nan],
            "open_interest_available": [1.0, 0.0, 1.0, 1.0],
        }
    )
    actual = vector_gate_pass(
        frame,
        [{"feature": "oi_zscore", "op": "<=", "threshold": 1.0}],
    )
    np.testing.assert_array_equal(actual, np.asarray([True, False, False, False]))


def test_comparison_reports_first_signal_and_integer_state_mismatch():
    dates = pd.date_range("2026-01-01", periods=4, freq="5min")
    domain = np.ones(4, dtype=bool)
    signal_report = compare_signals(
        dates,
        np.asarray([0, 1, -1, 0]),
        np.asarray([0, -1, -1, 0]),
        domain,
    )
    integer_report = compare_integer_vectors(
        dates,
        np.asarray([-1, 52, 143, 26]),
        np.asarray([-1, 52, 39, 26]),
        domain,
    )

    assert signal_report["passed"] is False
    assert signal_report["first_mismatches"][0]["position"] == 1
    assert integer_report["passed"] is False
    assert integer_report["first_mismatches"][0]["expected"] == 143


def test_shadow_history_contract_covers_longest_completed_htf_guard():
    assert REQUIRED_HISTORY_BARS == 17_280
    assert REQUIRED_LOOKBACK_MINUTES == 86_400


def test_missing_external_artifact_requires_explicit_root(tmp_path):
    with pytest.raises(FileNotFoundError, match="--artifact-root"):
        _resolve_existing("data/not-in-this-checkout.jsonl")
    source = tmp_path / "data/not-in-this-checkout.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("{}\n")
    assert _resolve_existing("data/not-in-this-checkout.jsonl", artifact_root=tmp_path) == source
