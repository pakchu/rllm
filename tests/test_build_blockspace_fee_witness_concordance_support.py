from __future__ import annotations

import ast
import gzip
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_blockspace_fee_witness_concordance_support as s


def _sources(
    rows: int = 200,
    *,
    start: str = "2023-06-01T00:00:00Z",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    starts = pd.date_range(start, periods=rows, freq=s.BUCKET)
    ends = starts + s.BUCKET
    available = ends + pd.Timedelta(hours=48)
    phase = np.arange(rows, dtype=float)
    fee_p10 = np.expm1(2.0 + 0.010 * phase)
    fee_p25 = np.expm1(3.0 + 0.006 * phase)
    fee_p75 = np.expm1(4.0 + 0.002 * phase)
    fee_p90 = np.expm1(5.0 + 0.001 * phase)
    size = 1_000_000.0 + 5_000.0 * phase
    weight = 2_000_000.0 + 6_000.0 * phase
    bfrt = pd.DataFrame(
        {
            "bucket_start_utc": starts,
            "bucket_end_utc": ends,
            "available_at_utc": available,
            "fee_p10": fee_p10,
            "fee_p25": fee_p25,
            "fee_p75": fee_p75,
            "fee_p90": fee_p90,
        },
        columns=s.BFRT_USECOLS,
    )
    wctr = pd.DataFrame(
        {
            "bucket_start_utc": starts,
            "bucket_end_utc": ends,
            "available_at_utc": available,
            "avg_size": size,
            "avg_weight": weight,
        },
        columns=s.WCTR_USECOLS,
    )
    return bfrt, wctr


def _clock(
    entry: str,
    *,
    side: str = "LONG",
    control: str = "primary",
    hold: pd.Timedelta = s.HOLD,
) -> dict[str, object]:
    entry_time = pd.Timestamp(entry)
    bucket_start = entry_time.floor("12h") - s.BUCKET
    return {
        "policy_id": s.POLICY_ID,
        "control": control,
        "window": "selection",
        "signal_id": hashlib.sha256(
            f"{entry}|{side}|{control}".encode()
        ).hexdigest(),
        "bucket_start_utc": bucket_start,
        "bucket_end_utc": bucket_start + s.BUCKET,
        "source_available_at_utc": entry_time - s.BAR,
        "entry_time_utc": entry_time,
        "exit_time_utc": entry_time + hold,
        "side": side,
        "rank_L": 100,
        "rank_E": 0,
        "rank_n": 120,
        "rank": 100 / 120,
    }


def _clock_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=s.CLOCK_COLUMNS)


def test_formula_polarity_and_joint_availability() -> None:
    bfrt, wctr = _sources(4)
    features, audit = s.build_joint_features(
        bfrt, wctr, exact_domain=False
    )
    row = features.iloc[-1]
    assert row["R"] > 0.0
    assert row["W"] > 0.0
    assert row["U"] > 0.0
    assert row["Q"] == pytest.approx(0.5 * (abs(row["W"]) + abs(row["U"])))
    assert row["joint_available_at_utc"] == wctr["available_at_utc"].iloc[-1]
    assert audit["exact_join_gaps"] == 0
    reversed_bfrt = bfrt.copy()
    reversed_wctr = wctr.copy()
    reversed_bfrt.loc[:, s.BFRT_USECOLS[3:]] = bfrt.loc[
        ::-1, s.BFRT_USECOLS[3:]
    ].to_numpy()
    reversed_wctr.loc[:, s.WCTR_USECOLS[3:]] = wctr.loc[
        ::-1, s.WCTR_USECOLS[3:]
    ].to_numpy()
    negative, _ = s.build_joint_features(
        reversed_bfrt, reversed_wctr, exact_domain=False
    )
    assert negative.iloc[-1]["R"] < 0.0
    assert negative.iloc[-1]["W"] < 0.0
    assert negative.iloc[-1]["U"] < 0.0


def test_exact_midrank_ties_current_exclusion_and_180_truncation() -> None:
    values = np.asarray([0.0] * 181 + [1.0, 1.0], dtype=float)
    valid = np.ones(len(values), dtype=bool)
    lower, equal, count, rank = s._rank_feature(values, valid)
    assert count[181] == 180
    assert lower[181] == 180
    assert equal[181] == 0
    assert rank[181] == 1.0
    assert count[182] == 180
    assert lower[182] == 179
    assert equal[182] == 1
    assert rank[182] == pytest.approx(179.5 / 180)


def test_source_validation_rejects_schema_time_domain_and_join_drift() -> None:
    bfrt, wctr = _sources(5)
    leaked = bfrt.assign(close=1.0)
    with pytest.raises(RuntimeError, match="schema drift"):
        s.validate_source_frame(leaked, "BFRT", exact_domain=False)
    bad_time = bfrt.copy()
    bad_time.loc[0, "bucket_end_utc"] += pd.Timedelta(minutes=1)
    with pytest.raises(RuntimeError, match="duration drift"):
        s.validate_source_frame(bad_time, "BFRT", exact_domain=False)
    bad_availability = bfrt.copy()
    bad_availability.loc[0, "available_at_utc"] += pd.Timedelta(minutes=1)
    with pytest.raises(RuntimeError, match="availability clock drift"):
        s.validate_source_frame(
            bad_availability, "BFRT", exact_domain=False
        )
    bad_order = bfrt.copy()
    bad_order.loc[0, "fee_p10"] = bad_order.loc[0, "fee_p90"] + 1.0
    with pytest.raises(RuntimeError, match="ordering drift"):
        s.validate_source_frame(bad_order, "BFRT", exact_domain=False)
    bad_domain = wctr.copy()
    bad_domain.loc[0, "avg_weight"] = 4_000_001.0
    with pytest.raises(RuntimeError, match="domain drift"):
        s.validate_source_frame(bad_domain, "WCTR", exact_domain=False)
    with pytest.raises(RuntimeError, match="join gap"):
        s.build_joint_features(
            bfrt,
            wctr.drop(index=2).reset_index(drop=True),
            exact_domain=False,
        )


def test_requires_t_t1_t2_consecutive() -> None:
    bfrt, wctr = _sources(5)
    bfrt = bfrt.drop(index=1).reset_index(drop=True)
    wctr = wctr.drop(index=1).reset_index(drop=True)
    features, _ = s.build_joint_features(bfrt, wctr, exact_domain=False)
    assert features["base_valid"].tolist() == [False, False, False, True]


def test_primary_and_each_independent_control_semantics() -> None:
    rows = 123
    starts = pd.date_range(
        "2023-06-01", periods=rows, freq=s.BUCKET, tz="UTC"
    )
    feature = pd.DataFrame(
        {
            "bucket_start_utc": starts,
            "bucket_end_utc": starts + s.BUCKET,
            "joint_available_at_utc": starts + s.BUCKET,
            "base_valid": True,
            "R": 1.0,
            "W": 1.0,
            "U": 1.0,
            "Q": 1.0,
            "rank_L": 100,
            "rank_E": 0,
            "rank_n": 120,
            "rank": 0.8,
            "q_rank_L": 100,
            "q_rank_E": 0,
            "q_rank_n": 120,
            "q_rank": 0.8,
        },
        columns=s.FEATURE_COLUMNS,
    )
    assert len(s.raw_candidates(feature, "primary")) == rows
    assert len(s.raw_candidates(feature, "fee_rotation_only")) == rows
    assert len(s.raw_candidates(feature, "witness_fullness_only")) == rows
    assert len(s.raw_candidates(feature, "drop_witness")) == rows
    assert len(s.raw_candidates(feature, "drop_fullness")) == rows
    stale = s.raw_candidates(feature, "one_bucket_stale_witness_fullness")
    assert len(stale) == rows - 1
    feature.loc[122, "W"] = -1.0
    assert len(s.raw_candidates(feature.tail(1), "primary")) == 0
    assert len(s.raw_candidates(feature.tail(1), "drop_witness")) == 1
    assert len(s.raw_candidates(feature.tail(1), "drop_fullness")) == 0


def test_stale_uses_prior_components_but_current_availability_and_rank() -> None:
    starts = pd.date_range(
        "2023-06-01", periods=2, freq=s.BUCKET, tz="UTC"
    )
    frame = pd.DataFrame(
        {
            "bucket_start_utc": starts,
            "bucket_end_utc": starts + s.BUCKET,
            "joint_available_at_utc": starts + s.BUCKET + pd.Timedelta(minutes=1),
            "base_valid": True,
            "R": [1.0, 1.0],
            "W": [1.0, -1.0],
            "U": [1.0, -1.0],
            "Q": [1.0, 1.0],
            "rank_L": [90, 100],
            "rank_E": [0, 0],
            "rank_n": [120, 120],
            "rank": [0.75, 0.8],
            "q_rank_L": [90, 90],
            "q_rank_E": [0, 0],
            "q_rank_n": [120, 120],
            "q_rank": [0.75, 0.75],
        },
        columns=s.FEATURE_COLUMNS,
    )
    stale = s.raw_candidates(frame, "one_bucket_stale_witness_fullness")
    assert len(stale) == 1
    assert stale.iloc[0]["source_available_at_utc"] == frame.iloc[1][
        "joint_available_at_utc"
    ]
    assert stale.iloc[0]["rank_L"] == 100


def test_split_containment_precedes_global_reservation_and_boundary_is_inclusive_exit() -> None:
    crossing = _clock_frame(
        [_clock("2024-12-31T12:05:00Z", hold=pd.Timedelta(hours=13))]
    )
    valid = _clock_frame([_clock("2025-01-01T00:05:00Z")])
    reserved = s.reserve_nonoverlap(pd.concat([crossing, valid], ignore_index=True))
    assert len(reserved) == 1
    assert reserved.iloc[0]["entry_time_utc"] == pd.Timestamp(
        "2025-01-01T00:05:00Z"
    )
    boundary = _clock_frame(
        [_clock("2024-12-31T00:00:00Z", hold=pd.Timedelta(days=1))]
    )
    accepted = s.reserve_nonoverlap(boundary)
    assert len(accepted) == 1
    assert accepted.iloc[0]["window"] == "selection"


def test_same_parent_side_controls_and_delayed_no_rerun() -> None:
    primary = s.reserve_nonoverlap(
        _clock_frame(
            [
                _clock("2024-01-01T00:00:00Z", side="LONG"),
                _clock("2024-01-02T00:00:00Z", side="SHORT"),
            ]
        )
    )
    assert s._parent_control(primary, "exact_direction_flip")["side"].tolist() == [
        "SHORT",
        "LONG",
    ]
    assert s._parent_control(primary, "constant_long")["side"].tolist() == [
        "LONG",
        "LONG",
    ]
    random = s._parent_control(primary, "deterministic_random_side")
    assert random["side"].tolist() == [
        s.prereg.deterministic_random_side(value)
        for value in primary["signal_id"]
    ]
    delayed = s._parent_control(primary, "one_bar_delayed_entry")
    assert delayed["entry_time_utc"].tolist() == (
        primary["entry_time_utc"] + s.BAR
    ).tolist()
    assert delayed["signal_id"].tolist() == primary["signal_id"].tolist()
    ending_at_split = s.reserve_nonoverlap(
        _clock_frame([_clock("2024-12-31T00:00:00Z")])
    )
    assert len(ending_at_split) == 1
    assert s._parent_control(
        ending_at_split, "one_bar_delayed_entry"
    ).empty


def test_support_floors_fail_closed_on_empty_or_concentrated_rows() -> None:
    empty = pd.DataFrame(columns=s.CLOCK_COLUMNS)
    _, checks = s.support_checks(empty)
    assert not all(checks.values())
    concentrated = _clock_frame(
        [
            _clock(
                f"2024-01-{1 + (index % 20):02d}T00:00:00Z",
                side="LONG" if index % 2 else "SHORT",
            )
            for index in range(45)
        ]
    )
    concentrated["window"] = "selection"
    _, checks = s.support_checks(concentrated)
    assert checks["selection_maximum_month_share"] is False


def test_support_floors_pass_at_frozen_period_incidence() -> None:
    dates: list[pd.Timestamp] = []
    months = (
        "2023-11",
        "2023-12",
        "2024-01",
        "2024-02",
        "2024-03",
        "2024-04",
        "2024-05",
        "2024-06",
        "2024-07",
        "2024-08",
        "2024-09",
        "2024-10",
        "2024-11",
        "2024-12",
        "2025-01",
        "2025-02",
        "2025-03",
        "2025-04",
        "2025-05",
        "2025-06",
        "2025-07",
        "2025-08",
        "2025-09",
        "2025-10",
        "2025-11",
        "2025-12",
        "2026-01",
        "2026-02",
        "2026-03",
        "2026-04",
        "2026-05",
    )
    for month_index, month in enumerate(months):
        days = (1, 5, 10, 20) if month_index < 3 else (1, 10, 20)
        dates.extend(
            pd.Timestamp(f"{month}-{day:02d}T00:00:00Z")
            for day in days
        )
    primary = s.reserve_nonoverlap(
        _clock_frame(
            [
                _clock(
                    timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    side="LONG" if index % 2 else "SHORT",
                )
                for index, timestamp in enumerate(dates)
            ]
        )
    )
    _, checks = s.support_checks(primary)
    assert all(checks.values())


def test_future_append_invariance_rebuilds_completed_prefixes() -> None:
    bfrt, wctr = _sources(200)
    passed, report = s.future_append_invariance(bfrt, wctr)
    assert passed
    assert set(report) == {
        "2025-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
    }
    assert all(row["passed"] for row in report.values())
    assert all(
        "feature_rank_state_sha256" in row["completed_prefix"]
        for row in report.values()
    )


def test_overlap_metrics_exact_containment_and_signed_exposure() -> None:
    candidate = _clock_frame(
        [
            _clock("2024-01-01T00:00:00Z", side="LONG"),
            _clock("2024-01-03T00:00:00Z", side="SHORT"),
        ]
    )
    comparator = _clock_frame(
        [
            _clock("2024-01-01T00:00:00Z", side="LONG"),
            _clock("2024-01-03T06:00:00Z", side="SHORT"),
        ]
    )
    metrics = s.novelty_metrics(candidate, comparator)
    assert metrics["exact_entry_jaccard"] == pytest.approx(1 / 3)
    assert metrics["candidate_6h_containment"] == 1.0
    assert 0.0 <= metrics["absolute_signed_exposure_pearson"] <= 1.0


def test_prior_comparator_numeric_sides_are_normalized_and_horizon_bounded() -> None:
    frame = pd.DataFrame(
        {
            "clock": ["primary", "primary", "primary"],
            "entry_time_utc": [
                "2023-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
                "2024-01-03T00:00:00Z",
            ],
            "exit_time_utc": [
                "2023-01-02T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-04T00:00:00Z",
            ],
            "side": ["1", "1", "-1"],
        }
    )
    prepared = s._prepare_comparator_frame(frame)
    assert prepared["side"].tolist() == ["LONG", "SHORT"]
    assert prepared["entry_time_utc"].min() >= s.FULL_START
    bad = frame.iloc[1:].copy()
    bad.loc[bad.index[0], "side"] = "0"
    with pytest.raises(RuntimeError, match="side drift"):
        s._prepare_comparator_frame(bad)


def test_support_failure_never_opens_comparators() -> None:
    bfrt, wctr = _sources(125)

    def forbidden_loader(
        _: object,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, object], int]:
        raise AssertionError("comparator loader must remain unopened")

    report, _, _ = s.build_support_from_frames(
        bfrt,
        wctr,
        artifact_eligible=True,
        comparator_loader=forbidden_loader,
    )
    assert report["support_passed"] is False
    assert report["novelty_status"] == "not_opened"
    assert report["rows_loaded"]["comparator_total"] == 0
    assert report["decision"] == "retire_BFWC_288_unchanged"
    assert report["outcomes_opened"] is False


def test_deterministic_gzip_and_atomic_write_once(tmp_path: Path) -> None:
    rows = _clock_frame([_clock("2024-01-01T00:00:00Z")])
    first = s.deterministic_clock_bytes(rows)
    second = s.deterministic_clock_bytes(rows)
    assert first == second
    assert gzip.decompress(first).decode().splitlines()[0] == ",".join(
        s.CLOCK_COLUMNS
    )
    output = tmp_path / "clock.csv.gz"
    assert s.write_once(output, first) == "created"
    assert s.write_once(output, first) == "verified_existing"
    with pytest.raises(RuntimeError, match="noncanonical"):
        s.write_once(output, b"different")
    artifact = {"z": 1, "a": [True, False]}
    assert s._json_bytes(artifact) == s._json_bytes(artifact)
    assert s._json_bytes(artifact).endswith(b"\n")


def test_clock_schema_contains_no_source_values_or_outcomes() -> None:
    assert not {"R", "W", "U", "Q"}.intersection(s.CLOCK_COLUMNS)
    assert not any(
        token in column.lower()
        for column in s.CLOCK_COLUMNS
        for token in s.FORBIDDEN_TOKENS
    )


def test_static_source_has_no_market_funding_or_outcome_loads() -> None:
    source_path = Path(s.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(
        token in name.lower()
        for name in imports
        for token in ("market", "funding", "premium", "gross9")
    )
    read_csv_functions = {
        node.func.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_csv"
        and isinstance(node.func.value, ast.Name)
    }
    assert read_csv_functions == {"pd"}
    text = source_path.read_text(encoding="utf-8")
    assert "cache_market_ext_5m" not in text
    assert "BTCUSDT_funding" not in text
    assert "BTCUSDT_premium" not in text
