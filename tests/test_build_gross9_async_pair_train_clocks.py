from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from training import build_gross9_async_pair_train_clocks as builder


def _clock(component: str, rows: list[tuple[str, int, str, str]]) -> pd.DataFrame:
    values = []
    for entry_raw, side, decision_raw, available_raw in rows:
        entry = pd.Timestamp(entry_raw)
        values.append(
            {
                "candidate": component,
                "control": "primary",
                "split": "train",
                "decision_time": pd.Timestamp(decision_raw),
                "feature_available_time": pd.Timestamp(available_raw),
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta("8h"),
                "side": side,
            }
        )
    return pd.DataFrame(values, columns=builder.COMMON_CLOCK_FIELDS)


def _write_clock(path: Path, component: str, entries: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=builder.COMMON_CLOCK_FIELDS)
        writer.writeheader()
        for entry_raw, side in entries:
            entry = pd.Timestamp(entry_raw)
            writer.writerow(
                {
                    "candidate": component,
                    "control": "primary",
                    "split": "train" if entry < builder.TRAIN_END else "test",
                    "decision_time": str(entry - pd.Timedelta("5m")),
                    "feature_available_time": str(entry - pd.Timedelta("3m")),
                    "entry_time": str(entry),
                    "exit_time": str(entry + pd.Timedelta("8h")),
                    "side": side,
                }
            )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_async_pair_uses_later_same_side_latest_other_within_window() -> None:
    left = _clock(
        "LEFT-8",
        [
            ("2023-07-01T00:00:00Z", 1, "2023-06-30T23:50:00Z", "2023-06-30T23:55:00Z"),
            ("2023-07-01T03:00:00Z", -1, "2023-07-01T02:50:00Z", "2023-07-01T02:55:00Z"),
        ],
    )
    right = _clock(
        "RIGHT-8",
        [
            ("2023-07-01T05:59:00Z", 1, "2023-07-01T05:40:00Z", "2023-07-01T05:45:00Z"),
            ("2023-07-01T06:01:00Z", -1, "2023-07-01T05:50:00Z", "2023-07-01T05:55:00Z"),
        ],
    )
    result = builder.build_async_pair_clock("LEFT-8", "RIGHT-8", left, right)
    assert result["entry_time"].tolist() == [pd.Timestamp("2023-07-01T05:59:00Z")]
    row = result.iloc[0]
    assert row["side"] == 1
    assert row["trigger_component_id"] == "RIGHT-8"
    assert row["confirming_component_id"] == "LEFT-8"
    assert row["confirming_entry_time"] == pd.Timestamp("2023-07-01T00:00:00Z")
    assert row["decision_time"] == pd.Timestamp("2023-07-01T05:40:00Z")
    assert row["feature_available_time"] == pd.Timestamp("2023-07-01T05:45:00Z")
    assert row["exit_time"] == row["entry_time"] + pd.Timedelta("8h")


def test_exact_simultaneous_entries_are_deduped_once() -> None:
    left = _clock("LEFT-8", [("2023-07-01T00:00:00Z", 1, "2023-06-30T23:00:00Z", "2023-06-30T23:30:00Z")])
    right = _clock("RIGHT-8", [("2023-07-01T00:00:00Z", 1, "2023-06-30T23:15:00Z", "2023-06-30T23:45:00Z")])
    result = builder.build_async_pair_clock("LEFT-8", "RIGHT-8", left, right)
    assert len(result) == 1
    assert result.loc[0, "trigger_component_id"] == "LEFT-8"
    assert result.loc[0, "confirming_component_id"] == "RIGHT-8"
    assert result.loc[0, "decision_time"] == pd.Timestamp("2023-06-30T23:15:00Z")
    assert result.loc[0, "feature_available_time"] == pd.Timestamp("2023-06-30T23:45:00Z")


def test_global_half_open_reservation_drops_overlaps_and_allows_abutment() -> None:
    left = _clock(
        "LEFT-8",
        [
            ("2023-07-01T00:00:00Z", 1, "2023-06-30T23:50:00Z", "2023-06-30T23:55:00Z"),
            ("2023-07-01T09:00:00Z", -1, "2023-07-01T08:50:00Z", "2023-07-01T08:55:00Z"),
        ],
    )
    right = _clock(
        "RIGHT-8",
        [
            ("2023-07-01T01:00:00Z", 1, "2023-07-01T00:50:00Z", "2023-07-01T00:55:00Z"),
            ("2023-07-01T02:00:00Z", 1, "2023-07-01T01:50:00Z", "2023-07-01T01:55:00Z"),
            ("2023-07-01T09:00:00Z", -1, "2023-07-01T08:40:00Z", "2023-07-01T08:45:00Z"),
        ],
    )
    result = builder.build_async_pair_clock("LEFT-8", "RIGHT-8", left, right)
    assert result["entry_time"].tolist() == [
        pd.Timestamp("2023-07-01T01:00:00Z"),
        pd.Timestamp("2023-07-01T09:00:00Z"),
    ]
    assert result.iloc[1]["entry_time"] >= result.iloc[0]["exit_time"]


def test_support_stats_and_gates() -> None:
    frame = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                ["2023-07-01T00:00:00Z", "2023-07-02T00:00:00Z", "2023-08-01T00:00:00Z"],
                utc=True,
            ),
            "side": [1, -1, 1],
        }
    )
    stats = builder.support_stats(frame)
    assert stats["events"] == 3
    assert stats["minority_side_share"] == pytest.approx(1 / 3)
    assert stats["max_month_share"] == pytest.approx(2 / 3)
    checks = builder.support_checks({"events": 8, "minority_side_share": 0.25, "max_month_share": 0.40})
    assert checks == {"minimum_events": True, "side_balance": True, "month_concentration": True}


def test_load_train_prefix_clock_stops_before_oos_and_validates_hash_boundary(tmp_path: Path) -> None:
    path = tmp_path / "left.csv.gz"
    _write_clock(
        path,
        "LEFT-8",
        [
            ("2023-06-30T16:00:00Z", 1),
            ("2023-07-01T00:00:00Z", 1),
            ("2023-12-31T18:00:00Z", -1),
            ("2024-01-01T00:00:00Z", 1),
            ("2024-01-02T00:00:00Z", -1),
        ],
    )
    artifacts = {"LEFT-8": {"clock": {"path": str(path), "sha256": _sha(path)}}}
    frame = builder.load_train_prefix_clock("LEFT-8", artifacts)
    assert frame["entry_time"].tolist() == [pd.Timestamp("2023-07-01T00:00:00Z")]
    assert frame.attrs["stopped_at_train_end"] is True
    assert frame["entry_time"].lt(builder.TRAIN_END).all()
    assert frame["exit_time"].le(builder.TRAIN_END).all()


def test_candidate_family_is_exact_36_pairs_from_nine_components() -> None:
    family = builder.candidate_family()
    assert len(builder.COMPONENT_ORDER) == 9
    assert len(family) == 36
    assert len(set(family)) == 36
    assert family[0] == builder.pair_id(builder.COMPONENT_ORDER[0], builder.COMPONENT_ORDER[1])


def test_verify_bound_component_artifacts_checks_hashes_and_pass_scalars(tmp_path: Path) -> None:
    component_order = tuple(f"C{i}-8" for i in range(9))
    artifacts = {}
    for component in component_order:
        prereg = tmp_path / f"{component}_prereg.json"
        support = tmp_path / f"{component}_support.json"
        gross9 = tmp_path / f"{component}_gross9.json"
        train_economics = tmp_path / f"{component}_train.json"
        clock = tmp_path / f"{component}_clock.csv.gz"
        prereg.write_text(json.dumps({"policy_id": component}) + "\n")
        support.write_text(json.dumps({"policy_id": component, "support_passed": True}) + "\n")
        gross9.write_text(json.dumps({"policy_id": component, "source_support_passed": True, "every_gross9_sleeve_passed": True, "gross9_novelty_status": "passed"}) + "\n")
        train_economics.write_text(
            json.dumps(
                {
                    "policy_id": component,
                    "stage": "train",
                    "passed": True,
                    "decision": "pass",
                    "later_stage_outcomes_opened": False,
                }
            )
            + "\n"
        )
        _write_clock(clock, component, [("2023-07-01T00:00:00Z", 1)])
        artifacts[component] = {
            "train_economics": {"path": str(train_economics), "sha256": _sha(train_economics)},
            "preregistration": {"path": str(prereg), "sha256": _sha(prereg)},
            "source_support": {"path": str(support), "sha256": _sha(support)},
            "gross9": {"path": str(gross9), "sha256": _sha(gross9)},
            "clock": {"path": str(clock), "sha256": _sha(clock)},
        }
    verified = builder.verify_bound_component_artifacts(artifacts, component_order)
    assert tuple(verified) == component_order
    assert all(
        value["source_support_passed"]
        and value["gross9_passed"]
        and value["train_economics_passed"]
        for value in verified.values()
    )
