from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pandas as pd
import pytest

from training import build_gross9_async_opposition_handoff_train_clocks as builder


def _clock(component: str, rows: list[tuple[str, int, str | None, str | None]]) -> pd.DataFrame:
    values = []
    for entry_raw, side, decision_raw, available_raw in rows:
        entry = pd.Timestamp(entry_raw)
        values.append(
            {
                "candidate": component,
                "control": "primary",
                "split": "train",
                "decision_time": pd.Timestamp(decision_raw) if decision_raw else entry - pd.Timedelta("10m"),
                "feature_available_time": pd.Timestamp(available_raw) if available_raw else entry - pd.Timedelta("5m"),
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta("8h"),
                "side": side,
            }
        )
    return pd.DataFrame(values, columns=builder.COMMON_CLOCK_FIELDS)


def test_opposition_handoff_uses_later_trigger_and_latest_opposite_strict_window() -> None:
    left = _clock(
        "LEFT-8",
        [
            ("2023-07-01T00:00:00Z", -1, "2023-06-30T23:50:00Z", "2023-06-30T23:55:00Z"),
            ("2023-07-01T02:00:00Z", -1, "2023-07-01T01:50:00Z", "2023-07-01T01:55:00Z"),
        ],
    )
    right = _clock("RIGHT-8", [("2023-07-01T05:59:00Z", 1, "2023-07-01T05:50:00Z", "2023-07-01T05:45:00Z")])
    result, diag = builder.build_async_opposition_handoff_clock("LEFT-8", "RIGHT-8", left, right)
    assert result["entry_time"].tolist() == [pd.Timestamp("2023-07-01T05:59:00Z")]
    row = result.iloc[0]
    assert row["side"] == 1
    assert row["trigger_component_id"] == "RIGHT-8"
    assert row["confirming_component_id"] == "LEFT-8"
    assert row["confirming_entry_time"] == pd.Timestamp("2023-07-01T02:00:00Z")
    assert row["decision_time"] == pd.Timestamp("2023-07-01T05:50:00Z")
    assert row["feature_available_time"] == pd.Timestamp("2023-07-01T05:50:00Z")
    assert row["exit_time"] == row["entry_time"] + pd.Timedelta("8h")
    assert diag["pre_reservation_rows"] == 1


def test_strict_window_excludes_boundary_and_simultaneous_events() -> None:
    left = _clock(
        "LEFT-8",
        [
            ("2023-07-01T00:00:00Z", -1, None, None),  # exactly t-6h: included
            ("2023-07-01T06:00:00Z", -1, None, None),  # simultaneous: excluded as trigger and not confirmer
        ],
    )
    right = _clock("RIGHT-8", [("2023-07-01T06:00:00Z", 1, None, None)])
    result, diag = builder.build_async_opposition_handoff_clock("LEFT-8", "RIGHT-8", left, right)
    assert result.empty
    assert diag["simultaneous_component_event_exclusions"] == 2

    left2 = _clock("LEFT-8", [("2023-06-30T23:59:59Z", -1, None, None)])
    right2 = _clock("RIGHT-8", [("2023-07-01T06:00:00Z", 1, None, None)])
    result2, diag2 = builder.build_async_opposition_handoff_clock("LEFT-8", "RIGHT-8", left2, right2)
    assert result2.empty
    assert diag2["no_other_strict_window_rejections"] >= 1


def test_zero_same_side_in_other_strict_window_is_required() -> None:
    left = _clock(
        "LEFT-8",
        [
            ("2023-07-01T01:00:00Z", -1, None, None),
            ("2023-07-01T02:00:00Z", 1, None, None),
        ],
    )
    right = _clock("RIGHT-8", [("2023-07-01T03:00:00Z", 1, None, None)])
    result, diag = builder.build_async_opposition_handoff_clock("LEFT-8", "RIGHT-8", left, right)
    assert result.empty
    assert diag["same_side_strict_window_rejections"] == 1


def test_pair_local_half_open_reservation_drops_overlaps_and_allows_abutment() -> None:
    left = _clock(
        "LEFT-8",
        [
            ("2023-07-01T00:00:00Z", -1, None, None),
            ("2023-07-01T09:00:00Z", -1, None, None),
        ],
    )
    right = _clock(
        "RIGHT-8",
        [
            ("2023-07-01T01:00:00Z", 1, None, None),
            ("2023-07-01T02:00:00Z", 1, None, None),
            ("2023-07-01T04:00:00Z", 1, None, None),
        ],
    )
    result, diag = builder.build_async_opposition_handoff_clock("LEFT-8", "RIGHT-8", left, right)
    assert result["entry_time"].tolist() == [
        pd.Timestamp("2023-07-01T01:00:00Z"),
        pd.Timestamp("2023-07-01T09:00:00Z"),
    ]
    assert diag["reservation_dropped_rows"] == 2


def test_same_side_pre_reservation_intersection_is_forbidden() -> None:
    left = _clock("LEFT-8", [("2023-07-01T00:00:00Z", -1, None, None)])
    right = _clock("RIGHT-8", [("2023-07-01T01:00:00Z", 1, None, None)])
    with pytest.raises(RuntimeError, match="same-side entry intersection"):
        builder.build_async_opposition_handoff_clock(
            "LEFT-8",
            "RIGHT-8",
            left,
            right,
            same_side_pre_reservation_keys={(pd.Timestamp("2023-07-01T01:00:00Z"), 1)},
        )


def test_support_stats_and_gates_include_distinct_iso_weeks() -> None:
    entries = pd.to_datetime(["2023-07-01T00:00:00Z", "2023-07-08T00:00:00Z", "2023-07-15T00:00:00Z", "2023-08-05T00:00:00Z", "2023-08-12T00:00:00Z", "2023-08-19T00:00:00Z", "2023-09-02T00:00:00Z", "2023-09-09T00:00:00Z", "2023-09-16T00:00:00Z"], utc=True)
    frame = pd.DataFrame({"entry_time": entries, "side": [1, -1, 1, -1, 1, -1, 1, -1, 1]})
    stats = builder.support_stats(frame)
    assert stats["events"] == 9
    assert stats["minority_side_share"] == pytest.approx(4 / 9)
    assert stats["distinct_iso_weeks"] == 9
    checks = builder.support_checks(stats)
    assert checks == {"minimum_events": True, "side_balance": True, "month_concentration": True, "distinct_iso_weeks": True}


def test_candidate_family_is_exact_36_pairs_from_nine_components() -> None:
    family = builder.candidate_family()
    assert len(builder.COMPONENT_ORDER) == 9
    assert len(family) == 36
    assert len(set(family)) == 36
    assert family[0] == builder.pair_id(builder.COMPONENT_ORDER[0], builder.COMPONENT_ORDER[1])


def test_run_requires_and_records_validated_preregistration_without_outcome_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    component_order = tuple(f"C{i}-8" for i in range(9))
    clocks = {
        component: _clock(
            component,
            [
                ("2023-07-01T00:00:00Z", -1 if idx == 0 else 1, None, None),
                ("2023-07-01T01:00:00Z", 1 if idx == 0 else -1, None, None),
            ],
        )
        for idx, component in enumerate(component_order)
    }
    monkeypatch.setattr(builder, "COMPONENT_ORDER", component_order)
    monkeypatch.setattr(builder, "verify_bound_component_artifacts", lambda: {component: {"verified": True} for component in component_order})
    monkeypatch.setattr(builder, "load_train_prefix_clock", lambda component: clocks[component])
    prereg_value = builder.prereg.build()
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(__import__("json").dumps(prereg_value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(builder.prereg, "DEFAULT_OUTPUT", prereg_path)
    monkeypatch.setattr(builder, "reconstruct_same_side_pre_reservation_keys", lambda left, right, left_clock, right_clock: set())
    monkeypatch.setattr(builder, "load_same_side_post_reservation_keys", lambda left, right, clock_dir: set())
    result = builder.run(tmp_path / "clocks", tmp_path / "result.json", tmp_path / "same")
    assert result["preregistration"]["path"] == str(prereg_path)
    assert result["preregistration"]["manifest_hash"] == prereg_value["manifest_hash"]
    assert result["candidate_family_size"] == 36
    assert len(result["pairs"]) == 36
    assert len(list((tmp_path / "clocks").glob("*.csv.gz"))) == 36
    assert result["evidence_boundary"]["market_rows_opened"] is False
    assert result["evidence_boundary"]["funding_opened"] is False
    assert result["evidence_boundary"]["gross9_rows_opened"] is False
    assert result["evidence_boundary"]["pair_combination_economic_outcomes_opened"] is False
    assert (tmp_path / "result.json").is_file()


def test_reconstruct_same_side_pre_reservation_keeps_overlap_hidden_by_same_side_reservation() -> None:
    left = _clock(
        "LEFT-8",
        [
            ("2023-07-01T00:00:00Z", 1, None, None),
            ("2023-07-01T00:30:00Z", 1, None, None),
        ],
    )
    right = _clock(
        "RIGHT-8",
        [
            ("2023-07-01T01:00:00Z", 1, None, None),
            ("2023-07-01T01:30:00Z", 1, None, None),
        ],
    )
    raw_keys = builder.reconstruct_same_side_pre_reservation_keys("LEFT-8", "RIGHT-8", left, right)
    post_reserved, _ = builder.build_async_opposition_handoff_clock(
        "LEFT-8",
        "RIGHT-8",
        left,
        right,
        same_side_pre_reservation_keys=set(),
    )
    # Same-side raw reconstruction sees both overlapping later entries; a
    # post-reservation same-side artifact would keep only the first 8h slot.
    assert (pd.Timestamp("2023-07-01T01:00:00Z"), 1) in raw_keys
    assert (pd.Timestamp("2023-07-01T01:30:00Z"), 1) in raw_keys
    assert post_reserved.empty

    handoff_left = _clock("LEFT-8", [("2023-07-01T00:30:00Z", -1, None, None)])
    handoff_right = _clock("RIGHT-8", [("2023-07-01T01:30:00Z", 1, None, None)])
    with pytest.raises(RuntimeError, match="pre-reservation same-side entry intersection"):
        builder.build_async_opposition_handoff_clock(
            "LEFT-8",
            "RIGHT-8",
            handoff_left,
            handoff_right,
            same_side_pre_reservation_keys=raw_keys,
            same_side_post_reservation_keys={(pd.Timestamp("2023-07-01T01:00:00Z"), 1)},
        )


def test_run_rejects_missing_and_drifted_preregistration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing_prereg.json"
    monkeypatch.setattr(builder.prereg, "DEFAULT_OUTPUT", missing)
    with pytest.raises(RuntimeError, match="missing committed preregistration artifact"):
        builder.load_validated_preregistration()

    value = builder.prereg.build()
    value["policy_id"] = "DRIFT"
    drifted = tmp_path / "drifted_prereg.json"
    drifted.write_text(__import__("json").dumps(value) + "\n", encoding="utf-8")
    monkeypatch.setattr(builder.prereg, "DEFAULT_OUTPUT", drifted)
    with pytest.raises(RuntimeError):
        builder.load_validated_preregistration()


def test_candidate_family_default_must_match_preregistration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builder.prereg, "CANDIDATE_FAMILY", ("wrong",))
    with pytest.raises(RuntimeError, match="candidate family differs from preregistration"):
        builder.candidate_family()


def test_load_same_side_post_reservation_keys_reads_only_entry_time_and_side(tmp_path: Path) -> None:
    left, right = "LEFT-8", "RIGHT-8"
    path = tmp_path / f"{builder.same_side_pair_id(left, right)}.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["entry_time", "side", "pnl"])
        writer.writeheader()
        writer.writerow({"entry_time": "2023-07-01T01:00:00Z", "side": 1, "pnl": "999"})
    assert builder.load_same_side_post_reservation_keys(left, right, tmp_path) == {(pd.Timestamp("2023-07-01T01:00:00Z"), 1)}
