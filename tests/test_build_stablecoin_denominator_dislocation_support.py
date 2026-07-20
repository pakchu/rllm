from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from training import build_stablecoin_denominator_dislocation_support as support


def _event_rows(*, count_per_month: int = 10) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sequence = 0
    for month in ("2023-09", "2023-10", "2023-11", "2023-12"):
        start = pd.Timestamp(f"{month}-01T00:00:00Z")
        for index in range(count_per_month):
            entry = start + pd.Timedelta(days=index * 2, minutes=5)
            rows.append(
                {
                    "entry_time": entry,
                    "exit_time": entry + pd.Timedelta("1h"),
                    "side": 1 if sequence % 2 == 0 else -1,
                }
            )
            sequence += 1
    return pd.DataFrame(rows)


def test_real_preregistration_is_hash_bound_and_outcome_blind() -> None:
    payload = support.load_preregistration()
    assert payload["outcomes_opened"] is False
    assert payload["candidate"] == "SDDR-12"
    assert support.sha256_file(support.PREREGISTRATION) == (
        support.PREREGISTRATION_SHA256
    )


def test_build_clocks_retains_only_source_state() -> None:
    dates = pd.date_range("2023-09-01", periods=674, freq="1h", tz="UTC")
    phase = np.arange(len(dates)) % 4
    usdc = np.choose(phase, [-0.002, -0.001, 0.001, 0.002]).astype(float)
    fdusd = np.choose(phase, [-0.0022, -0.0011, 0.0011, 0.0022]).astype(float)
    disagreement = np.full(len(dates), 0.0001)
    usdc[672] = 0.02
    fdusd[672] = 0.021
    source = pd.DataFrame(
        {
            "date": dates,
            "source_available_at": dates + pd.Timedelta("1h"),
            "usdc_vs_usdt": usdc,
            "fdusd_vs_usdt": fdusd,
            "alt_consensus": (usdc + fdusd) / 2.0,
            "alt_disagreement": disagreement,
            "source_complete": True,
        }
    )
    clocks = support.build_clocks(source)
    assert set(clocks["control"]) == set(support.CONTROLS)
    assert tuple(clocks.columns) == support.sddr.EVENT_COLUMNS
    assert not any(
        token in column.lower()
        for token in support.FORBIDDEN_CLOCK_TOKENS
        for column in clocks.columns
    )


def test_support_gate_requires_count_balance_dispersion_and_every_month() -> None:
    prereg = support.load_preregistration()
    summary = support.support_summary(_event_rows())
    clean_novelty = {
        control: {
            "exact_jaccard": 0.0,
            "max_bidirectional_near_share": 0.0,
        }
        for control in support.sddr.COMPARATOR_CONTROLS
    }
    checks, failures = support.support_checks(summary, clean_novelty, prereg)
    assert all(checks.values())
    assert failures == []

    sparse = support.support_summary(_event_rows(count_per_month=4))
    _, failures = support.support_checks(sparse, clean_novelty, prereg)
    assert "minimum_events" in failures
    assert "2023-09_minimum_events" in failures


def test_novelty_is_exact_and_bidirectional_with_common_coverage() -> None:
    primary = pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2023-09-01T01:05:00Z",
                "2023-09-10T01:05:00Z",
                "2024-01-02T01:05:00Z",
            ],
            utc=True,
        )
    )
    comparator = pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2023-09-01T01:05:00Z",
                "2023-09-10T05:05:00Z",
                "2023-12-20T01:05:00Z",
            ],
            utc=True,
        )
    )
    metrics = support.novelty_metrics(primary, comparator, hours=6)
    assert metrics["primary_events"] == 2
    assert metrics["comparator_events"] == 3
    assert metrics["exact_jaccard"] == pytest.approx(0.25)
    assert metrics["primary_near_share"] == 1.0
    assert metrics["comparator_near_share"] == pytest.approx(2 / 3)
    assert metrics["max_bidirectional_near_share"] == 1.0


def test_frozen_writers_are_idempotent_and_refuse_repair(tmp_path) -> None:
    path = tmp_path / "freeze.json"
    support.write_frozen_json(path, {"value": 1})
    first = path.read_bytes()
    support.write_frozen_json(path, {"value": 1})
    assert path.read_bytes() == first
    with pytest.raises(FileExistsError):
        support.write_frozen_json(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}


def test_clock_encoding_is_byte_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "entry_time": [pd.Timestamp("2023-09-01T01:05:00Z")],
            "side": [1],
            "z": [1.23456789012345],
        }
    )
    assert support._clock_bytes(frame) == support._clock_bytes(frame)
