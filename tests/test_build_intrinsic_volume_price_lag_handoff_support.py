from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_intrinsic_volume_price_lag_handoff_support as s


def _anchors(
    count: int,
    *,
    start: str = "2021-01-01",
    sides: list[str] | None = None,
    returns: list[float] | None = None,
) -> pd.DataFrame:
    days = pd.date_range(start, periods=count, freq="D", tz="UTC")
    side_values = sides or ["LONG"] * count
    return_values = returns or [0.01] * count
    signs = [1 if side == "LONG" else -1 for side in side_values]
    return pd.DataFrame(
        {
            "source_day": days,
            "anchor_index": np.arange(count, dtype=int),
            "anchor_time": days + pd.Timedelta(hours=8),
            "side_sign": signs,
            "side": side_values,
            "cumulative_flow": np.asarray(signs, dtype=float) * 0.1,
            "anchor_return": return_values,
            "anchor_minute_utc": [480] * count,
            "target_quote_volume": [100.0] * count,
            "cumulative_quote_volume": [100.0] * count,
        }
    )


def _raw_row(
    source_day: str,
    entry_time: str,
    *,
    control: str = "primary",
    side: str = "LONG",
) -> dict[str, object]:
    entry = pd.Timestamp(entry_time)
    day = pd.to_datetime(source_day, utc=True)
    return s._candidate_row(
        control,
        day,
        entry - s.BAR,
        entry,
        entry + 72 * s.BAR,
        side,
    )


def _primary_and_predecessor(count: int = 66) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        _raw_row(
            f"2021-{1 + index // 28:02d}-{1 + index % 28:02d}",
            (
                pd.Timestamp("2021-01-01T08:05:00Z")
                + index * pd.Timedelta(days=1)
            ).isoformat(),
            side="LONG" if index % 2 == 0 else "SHORT",
        )
        for index in range(count)
    ]
    primary = pd.DataFrame(rows, columns=s.CLOCK_COLUMNS)
    predecessor = pd.DataFrame(
        {
            "clock_name": ["any_handoff"] * count,
            "source_day": primary["source_day"],
            "decision_time": primary["decision_time"],
            "entry_time": primary["decision_time"],
            "exit_time": primary["exit_time"] - s.BAR,
            "side": primary["side"],
        }
    )
    return primary, predecessor


def test_state_uses_strictly_prior_reference_and_consecutive_handoff() -> None:
    sides = ["LONG"] * 91 + ["SHORT"]
    returns = [0.01] * 92
    result = s.annotate_state(_anchors(92, sides=sides, returns=returns))
    event = result.iloc[-1]

    assert event["reference_count"] == 91
    assert bool(event["reference_ready"])
    assert bool(event["calendar_consecutive"])
    assert bool(event["handoff"])
    assert event["directional_return"] == pytest.approx(-0.01)
    assert bool(event["price_lag"])
    assert bool(event["primary"])


def test_calendar_gap_resets_handoff_without_resetting_reference_history() -> None:
    frame = _anchors(92)
    frame.loc[91, "source_day"] += pd.Timedelta(days=1)
    frame.loc[91, "anchor_time"] += pd.Timedelta(days=1)
    frame.loc[91, ["side", "side_sign"]] = ["SHORT", -1]
    result = s.annotate_state(frame)
    event = result.iloc[-1]

    assert event["reference_count"] == 91
    assert bool(event["reference_ready"])
    assert not bool(event["calendar_consecutive"])
    assert not bool(event["handoff"])
    assert not bool(event["primary"])


def test_raw_candidate_freezes_complete_bar_latency_and_hold() -> None:
    features = s.annotate_state(
        _anchors(
            92,
            sides=["LONG"] * 91 + ["SHORT"],
            returns=[0.01] * 92,
        )
    )
    candidate = s.raw_candidates(
        features,
        "primary",
        features["primary"],
    ).iloc[0]
    anchor = features.iloc[-1]["anchor_time"]

    assert candidate["decision_time"] == anchor + s.BAR
    assert candidate["entry_time"] == anchor + 2 * s.BAR
    assert candidate["exit_time"] == candidate["entry_time"] + 72 * s.BAR
    assert candidate["signal_id"] == s.signal_id(
        "primary",
        candidate["source_day"],
        candidate["decision_time"],
        candidate["side"],
    )


def test_control_masks_remove_only_the_named_condition() -> None:
    sides = ["LONG"] * 91 + ["SHORT", "SHORT"]
    returns = [0.01] * 91 + [-0.01, 0.01]
    features = s.annotate_state(_anchors(93, sides=sides, returns=returns))

    primary = s.raw_candidates(features, "primary", features["primary"])
    handoff_only = s.raw_candidates(
        features,
        "handoff_without_price_lag",
        features["reference_ready"] & features["handoff"],
    )
    lag_only = s.raw_candidates(
        features,
        "price_lag_without_handoff",
        features["reference_ready"] & features["price_lag"],
    )

    assert len(primary) == 0
    assert len(handoff_only) == 1
    assert len(lag_only) == 1


def test_containment_precedes_independent_split_reservation() -> None:
    boundary = _raw_row(
        "2022-12-31T00:00:00Z",
        "2022-12-31T18:00:00Z",
    )
    crossing = _raw_row(
        "2022-12-31T00:00:00Z",
        "2022-12-31T23:00:00Z",
    )
    selection = _raw_row(
        "2023-01-01T00:00:00Z",
        "2023-01-01T00:05:00Z",
        side="SHORT",
    )
    raw = pd.DataFrame(
        [boundary, crossing, selection],
        columns=s.CLOCK_COLUMNS,
    )
    scheduled = s.schedule_candidates(raw)

    assert list(scheduled["entry_time"]) == [
        pd.Timestamp("2022-12-31T18:00:00Z"),
        pd.Timestamp("2023-01-01T00:05:00Z"),
    ]
    assert scheduled.iloc[0]["exit_time"] == s.CALIBRATION_END


def test_entry_equal_prior_exit_is_accepted() -> None:
    first = _raw_row("2021-01-01", "2021-01-01T08:00:00Z")
    second = _raw_row("2021-01-01", "2021-01-01T14:00:00Z")
    scheduled = s.schedule_candidates(
        pd.DataFrame([first, second], columns=s.CLOCK_COLUMNS)
    )
    assert len(scheduled) == 2


@pytest.mark.parametrize(
    ("control", "field"),
    [
        ("anchor_side_year_permutation", "side_sign"),
        ("anchor_return_year_permutation", "anchor_return"),
    ],
)
def test_year_permutations_are_deterministic_bijections_and_recompute_state(
    control: str,
    field: str,
) -> None:
    base = _anchors(
        500,
        start="2021-01-01",
        sides=["LONG" if index % 3 else "SHORT" for index in range(500)],
        returns=[float(index) / 10_000 for index in range(500)],
    )
    annotated = s.annotate_state(base)
    first = s.permute_anchor_field(annotated, control)
    second = s.permute_anchor_field(annotated, control)

    pd.testing.assert_frame_equal(first, second)
    for year in (2021, 2022):
        original = base.loc[base["source_day"].dt.year.eq(year), field]
        permuted = first.loc[first["source_day"].dt.year.eq(year), field]
        assert sorted(original.tolist()) == sorted(permuted.tolist())
    assert list(first.columns).count("primary") == 1
    expected_directional = (
        first["side_sign"].astype(int) * first["anchor_return"].astype(float)
    )
    assert np.allclose(first["directional_return"], expected_directional)


def test_duplicate_permutation_source_day_fails_closed() -> None:
    frame = _anchors(2)
    frame.loc[1, "source_day"] = frame.loc[0, "source_day"]
    with pytest.raises(RuntimeError, match="source days duplicated"):
        s.permute_anchor_field(frame, "anchor_side_year_permutation")


def test_random_side_is_derived_before_control_signal_identity() -> None:
    primary, _ = _primary_and_predecessor(1)
    result = s.deterministic_random_side_candidates(primary)
    row = result.iloc[0]
    side_free = {
        "control": "deterministic_random_side",
        "decision_time": s._format_time(primary.iloc[0]["decision_time"]),
        "policy_id": "IVPLH-72",
        "primary_entry_time": s._format_time(primary.iloc[0]["entry_time"]),
        "source_day": s._format_day(primary.iloc[0]["source_day"]),
        "source_panel_sha256": s.prereg.MARKET_SOURCE_SHA256,
    }
    digest = hashlib.sha256(
        json.dumps(
            side_free,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).digest()
    expected = "LONG" if digest[0] < 128 else "SHORT"

    assert row["side"] == expected
    assert row["signal_id"] == s.signal_id(
        "deterministic_random_side",
        row["source_day"],
        row["decision_time"],
        expected,
    )


def test_predecessor_identity_requires_all_66_rows_and_exact_time_shifts() -> None:
    primary, predecessor = _primary_and_predecessor()
    assert all(s.validate_predecessor_identity(primary, predecessor).values())

    shifted = predecessor.copy()
    shifted.loc[0, "exit_time"] += s.BAR
    checks = s.validate_predecessor_identity(primary, shifted)
    assert checks["predecessor_identity_exact"]
    assert checks["predecessor_entry_shift_exact"]
    assert not checks["predecessor_exit_shift_exact"]

    assert not all(
        s.validate_predecessor_identity(
            primary.iloc[:-1],
            predecessor.iloc[:-1],
        ).values()
    )


def test_predecessor_loader_decodes_only_selected_rows_and_rejects_unknown(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"
    controls = {
        name: {"events": 1 if name == "any_handoff" else 0}
        for name in (
            "primary",
            "any_handoff",
            "no_price_lag",
            "no_flow_strength",
            "persistence_level",
            "fixed_noon_handoff",
            "exact_side_flip",
            "deterministic_random_side",
        )
    }
    report_path.write_text(
        json.dumps({"outcomes_opened": False, "controls": controls})
    )
    header = [
        "clock_name",
        "source_day",
        "decision_time",
        "entry_time",
        "exit_time",
        "side",
    ]
    clock_path = tmp_path / "clock.csv.gz"

    def write_clock(first_name: str) -> None:
        with gzip.open(clock_path, "wt", encoding="utf-8", newline="") as handle:
            handle.write(",".join(header) + "\n")
            handle.write(f"{first_name},bad,bad,bad,bad,bad\n")
            handle.write(
                "any_handoff,2021-01-01T00:00:00Z,"
                "2021-01-01T08:00:00Z,2021-01-01T08:00:00Z,"
                "2021-01-01T14:00:00Z,LONG\n"
            )

    lineage = {
        "support_report": {"path": str(report_path)},
        "clock": {"path": str(clock_path), "header": header},
        "known_clock_names": list(controls),
        "selected_clock_name": "any_handoff",
        "disclosed_global_rows": 1,
        "identity_key": ["source_day", "side", "decision_time"],
    }
    write_clock("primary")
    selected, audit = s.load_predecessor({"predecessor_lineage": lineage})
    assert len(selected) == 1
    assert audit["physical_clock_rows_scanned"] == 2
    assert audit["clock_rows_decoded"] == 1

    write_clock("unknown")
    with pytest.raises(RuntimeError, match="unknown control"):
        s.load_predecessor({"predecessor_lineage": lineage})


def test_empty_permutation_denominators_are_one_and_fail() -> None:
    empty = pd.DataFrame(columns=s.CLOCK_COLUMNS)
    assert s.exact_entry_jaccard(empty, empty, "train") == 1.0
    assert s.same_side_reproduction(empty, empty, "selection") == 1.0


def test_clock_statistics_use_split_local_month_denominator() -> None:
    rows = pd.DataFrame(
        [
            _raw_row("2021-01-01", "2021-01-01T08:00:00Z"),
            _raw_row("2021-01-02", "2021-01-02T08:00:00Z"),
            _raw_row("2021-02-01", "2021-02-01T08:00:00Z"),
            _raw_row("2021-03-01", "2021-03-01T08:00:00Z"),
        ],
        columns=s.CLOCK_COLUMNS,
    )
    stats = s.clock_stats(rows)
    assert stats["events"] == 4
    assert stats["maximum_month_share"] == 0.5


def test_deterministic_clock_bytes_have_fixed_schema_and_gzip_header() -> None:
    primary, _ = _primary_and_predecessor(3)
    controls = {
        name: (
            primary.assign(control=name)
            if name == "primary"
            else pd.DataFrame(columns=s.CLOCK_COLUMNS)
        )
        for name in s.CONTROL_ORDER
    }
    first = s.deterministic_clock_bytes(controls)
    second = s.deterministic_clock_bytes(controls)

    assert first == second
    with gzip.GzipFile(fileobj=io.BytesIO(first), mode="rb") as zipped:
        header = zipped.readline().decode()
        row = zipped.readline().decode().rstrip("\n").split(",")
    assert header == ",".join(s.CLOCK_COLUMNS) + "\n"
    values = dict(zip(s.CLOCK_COLUMNS, row, strict=True))
    assert values["source_day"] == "2021-01-01"
    assert values["decision_time"] == "2021-01-01T08:00:00Z"
    assert values["entry_time"] == "2021-01-01T08:05:00Z"
    assert values["exit_time"] == "2021-01-01T14:05:00Z"


def test_synthetic_report_keeps_comparators_and_outcomes_closed() -> None:
    features = s.annotate_state(_anchors(100))
    predecessor = pd.DataFrame(
        columns=(
            "clock_name",
            "source_day",
            "decision_time",
            "entry_time",
            "exit_time",
            "side",
        )
    )
    payload, clock_bytes = s.build_support_from_anchors(
        features,
        features,
        predecessor,
    )

    assert payload["artifact_eligible"] is False
    assert payload["outcomes_opened"] is False
    assert payload["post_entry_return_computed"] is False
    assert payload["funding_loaded"] is False
    assert payload["comparator_rows_decoded"] is False
    assert payload["source_support_passed"] is False
    assert payload["advance_to_comparator_novelty_freeze"] is False
    assert payload["comparator_status"] == "not_opened_source_support_stage"
    assert payload["outcome_boundary"]["comparator_rows_decoded"] == 0
    assert payload["outcome_boundary"]["post_entry_price_rows_decoded"] == 0
    assert payload["outcome_boundary"]["funding_rows_decoded"] == 0
    assert payload["outcome_boundary"]["future_return_rows_decoded"] == 0
    assert payload["clock"]["sha256"] == hashlib.sha256(clock_bytes).hexdigest()


def test_protocol_commit_guard_rejects_untracked_or_dirty_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        def __init__(self, returncode: int):
            self.returncode = returncode

    monkeypatch.setattr(s, "_git_check", lambda *_args: Result(1))
    with pytest.raises(RuntimeError, match="not committed"):
        s._assert_protocol_committed()

    outcomes = iter((Result(0), Result(1)))
    monkeypatch.setattr(s, "_git_check", lambda *_args: next(outcomes))
    with pytest.raises(RuntimeError, match="differs from HEAD"):
        s._assert_protocol_committed()
