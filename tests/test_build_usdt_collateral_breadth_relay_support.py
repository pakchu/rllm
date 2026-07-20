from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from training import build_usdt_collateral_breadth_relay_support as support


def _source(rows: int = 674) -> pd.DataFrame:
    dates = pd.date_range("2023-08-01", periods=rows, freq="1h", tz="UTC")
    phase = np.arange(rows) % 4
    base = np.choose(phase, [-0.002, -0.001, 0.001, 0.002]).astype(float)
    data: dict[str, object] = {
        "date": dates,
        "source_available_at": dates + pd.Timedelta("1h"),
    }
    for index, (log_column, valid_column) in enumerate(
        zip(support.ucbr.LOG_COLUMNS, support.ucbr.VALID_COLUMNS, strict=True)
    ):
        values = base * (1.0 + index * 0.05)
        values[672] = 0.02
        values[673] = 0.0
        data[log_column] = values
        data[valid_column] = True
    data["valid_breadth"] = 4
    data["source_complete"] = True
    return pd.DataFrame(data)


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
                    "exit_time": entry + pd.Timedelta("12h"),
                    "side": 1 if sequence % 2 == 0 else -1,
                }
            )
            sequence += 1
    return pd.DataFrame(rows)


def test_real_preregistration_is_hash_bound_and_unopened() -> None:
    payload = support.load_preregistration()
    assert payload["outcomes_opened"] is False
    assert payload["real_event_incidence_opened"] is False
    assert support.sha256_file(support.PREREGISTRATION) == support.PREREGISTRATION_SHA256


def test_build_clocks_contains_every_frozen_control_and_no_outcome() -> None:
    clocks = support.build_clocks(_source())
    assert set(clocks["control"]) == set(support.CONTROLS)
    assert tuple(clocks.columns) == support.ucbr.EVENT_COLUMNS
    assert not any(
        token in column.lower()
        for token in support.FORBIDDEN_CLOCK_TOKENS
        for column in clocks.columns
    )
    for _, group in clocks.groupby("control"):
        entries = pd.to_datetime(group["entry_time"], utc=True).sort_values()
        exits = pd.to_datetime(group["exit_time"], utc=True).sort_values()
        assert (entries.iloc[1:].to_numpy() >= exits.iloc[:-1].to_numpy()).all()


def test_support_gate_requires_count_balance_dispersion_and_every_month() -> None:
    prereg = support.load_preregistration()
    summary = support.support_summary(_event_rows())
    clean_novelty = {
        name: {"exact_jaccard": 0.0, "max_bidirectional_near_share": 0.0}
        for name in (
            "SDDR-12:primary",
            "SQFD-6:primary",
            "SQFD-6:no_usdt_lag",
            "SQFD-6:no_participation",
        )
    }
    checks, failures = support.support_checks(summary, clean_novelty, prereg)
    assert all(checks.values())
    assert failures == []
    sparse = support.support_summary(_event_rows(count_per_month=4))
    _, failures = support.support_checks(sparse, clean_novelty, prereg)
    assert "minimum_events" in failures
    assert "2023-09_minimum_events" in failures


def test_novelty_is_exact_and_bidirectional_inside_common_window() -> None:
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


def test_comparator_loader_uses_only_frozen_source_clocks() -> None:
    prereg = support.load_preregistration()
    clocks, rows = support.load_comparator_clocks(prereg)
    assert set(clocks) == {
        "SDDR-12:primary",
        "SQFD-6:primary",
        "SQFD-6:no_usdt_lag",
        "SQFD-6:no_participation",
    }
    assert rows > 0


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
