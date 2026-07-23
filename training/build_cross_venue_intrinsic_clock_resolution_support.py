"""Build outcome-blind CVICR-72 support clocks and novelty diagnostics."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_cross_venue_intrinsic_clock_resolution as prereg


PROTOCOL_VERSION = "cross_venue_intrinsic_clock_resolution_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/build_cross_venue_intrinsic_clock_resolution_support.py"
)
TEST_PATH = Path(
    "tests/test_build_cross_venue_intrinsic_clock_resolution_support.py"
)
IMPLEMENTATION_CONTRACT = Path(
    "docs/cvicr-source-support-implementation-contract-2026-07-24.md"
)
IMPLEMENTATION_CONTRACT_SHA256 = (
    "abcb4f5c38a6abd84036287a8e2f16ef0313b1b7cae580433093c5c7bb429f4c"
)
PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = (
    "f5f293b7b6152d7e28c5bb825d3dd2d8a2626678917960f721e900231c5671f0"
)
PREREGISTRATION_MANIFEST_HASH = (
    "75f93cc512ab711936af56834c60c2a416c4bcc43ea672a9076e6763cbff2f1c"
)
DEFAULT_CLOCK_OUTPUT = Path(
    "data/cross_venue_intrinsic_clock_resolution_clocks_2020_2023.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/cross_venue_intrinsic_clock_resolution_support_2026-07-24.json"
)

BAR = pd.Timedelta(minutes=5)
DAY = pd.Timedelta(days=1)
SOURCE_START = pd.Timestamp("2020-01-01T00:00:00Z")
SOURCE_END = pd.Timestamp("2024-01-01T00:00:00Z")
TRAIN_START = SOURCE_START
TRAIN_END = pd.Timestamp("2023-01-01T00:00:00Z")
SELECTION_START = TRAIN_END
SELECTION_END = SOURCE_END
ROWS_PER_DAY = 288
LATEST_ANCHOR_INDEX = (17 * 60 + 50) // 5

CLOCK_COLUMNS = (
    "control",
    "signal_id",
    "source_day",
    "causal_origin_time",
    "resolution_time",
    "signal_available_time",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
    "leader",
)
FORBIDDEN_CLOCK_TOKENS = (
    "open",
    "high",
    "low",
    "close",
    "price",
    "return",
    "basis",
    "future",
    "label",
    "funding",
    "pnl",
    "reward",
    "cagr",
    "mdd",
)
INDEPENDENT_CONTROLS = (
    "primary",
    "gap_only",
    "initial_conflict_only",
    "late_alignment_only",
    "no_leader_persistence",
    "no_gap_tail",
    "fixed_expected_time_clocks",
    "stale_laggard_flow_24h",
)
WINDOWS = {
    "train": (TRAIN_START, TRAIN_END),
    "selection": (SELECTION_START, SELECTION_END),
    "2020": (TRAIN_START, pd.Timestamp("2021-01-01T00:00:00Z")),
    "2021": (
        pd.Timestamp("2021-01-01T00:00:00Z"),
        pd.Timestamp("2022-01-01T00:00:00Z"),
    ),
    "2022": (
        pd.Timestamp("2022-01-01T00:00:00Z"),
        TRAIN_END,
    ),
    "2023_h1": (
        SELECTION_START,
        pd.Timestamp("2023-07-01T00:00:00Z"),
    ),
    "2023_h2": (
        pd.Timestamp("2023-07-01T00:00:00Z"),
        SELECTION_END,
    ),
    "2023_q1": (
        SELECTION_START,
        pd.Timestamp("2023-04-01T00:00:00Z"),
    ),
    "2023_q2": (
        pd.Timestamp("2023-04-01T00:00:00Z"),
        pd.Timestamp("2023-07-01T00:00:00Z"),
    ),
    "2023_q3": (
        pd.Timestamp("2023-07-01T00:00:00Z"),
        pd.Timestamp("2023-10-01T00:00:00Z"),
    ),
    "2023_q4": (
        pd.Timestamp("2023-10-01T00:00:00Z"),
        SELECTION_END,
    ),
}


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _format_time(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise RuntimeError("CVICR timestamp must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp.microsecond or timestamp.nanosecond:
        raise RuntimeError("CVICR timestamp must be whole-second")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_day(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise RuntimeError("CVICR source day must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp != timestamp.floor("D"):
        raise RuntimeError("CVICR source day must be UTC midnight")
    return timestamp.strftime("%Y-%m-%d")


def _git_check(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_protocol_committed() -> None:
    paths = (str(SCRIPT_PATH), str(TEST_PATH), str(IMPLEMENTATION_CONTRACT))
    tracked = _git_check("ls-files", "--error-unmatch", "--", *paths)
    if tracked.returncode:
        raise RuntimeError("CVICR source-support protocol is not committed")
    clean = _git_check("diff", "--quiet", "HEAD", "--", *paths)
    if clean.returncode:
        raise RuntimeError("CVICR source-support protocol differs from HEAD")


def validate_preregistration() -> Mapping[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("CVICR preregistration artifact hash drift")
    payload = json.loads(_path(PREREGISTRATION).read_text(encoding="utf-8"))
    prereg.validate_manifest(payload)
    if payload != prereg.build_manifest():
        raise RuntimeError("CVICR preregistration differs from frozen builder")
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("CVICR preregistration manifest hash drift")
    for field in (
        "outcomes_opened",
        "source_incidence_opened",
        "source_rows_decoded",
        "comparator_rows_decoded",
    ):
        if payload.get(field) is not False:
            raise RuntimeError(f"CVICR preregistration boundary opened: {field}")
    if tuple(payload["source_only_controls"]["ordered"]) != prereg.CONTROL_ORDER:
        raise RuntimeError("CVICR control order drift")
    return payload


def verify_pre_source_bindings(
    payload: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    prereg.validate_frozen_dependencies()
    bindings: list[tuple[str | Path, str, str]] = [
        (
            IMPLEMENTATION_CONTRACT,
            IMPLEMENTATION_CONTRACT_SHA256,
            "implementation_contract",
        ),
        (
            PREREGISTRATION,
            PREREGISTRATION_SHA256,
            "preregistration",
        ),
    ]
    for path, expected in prereg.frozen_dependencies().items():
        bindings.append((path, expected, str(path)))
    audit: dict[str, dict[str, str]] = {}
    for path, expected, label in bindings:
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"CVICR frozen binding changed: {label}")
        audit[label] = {"path": str(path), "sha256": actual}
    if payload["source_contract"]["allowlist"] != list(prereg.SOURCE_ALLOWLIST):
        raise RuntimeError("CVICR source allowlist drift")
    return audit


def _strict_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    if not normalized.isin(("true", "false")).all():
        raise RuntimeError("CVICR source_complete is not strict boolean")
    return normalized.eq("true")


def validate_source_frame(
    frame: pd.DataFrame,
    *,
    exact_grid: bool = True,
) -> pd.DataFrame:
    if list(frame.columns) != list(prereg.SOURCE_ALLOWLIST):
        raise RuntimeError("CVICR source loader did not preserve exact allowlist")
    validated = frame.copy()
    for column in (
        "date",
        "feature_available_time_utc",
        "trade_earliest_time_utc",
    ):
        validated[column] = pd.to_datetime(
            validated[column],
            utc=True,
            errors="raise",
        )
    for column in (
        "spot_quote_notional",
        "um_quote_notional",
        "spot_signed_quote_notional",
        "um_signed_quote_notional",
    ):
        validated[column] = pd.to_numeric(
            validated[column],
            errors="coerce",
        ).astype(float)
    validated["source_complete"] = _strict_bool(
        validated["source_complete"]
    )
    if validated["date"].duplicated().any():
        raise RuntimeError("CVICR source timestamps duplicated")
    if not validated["date"].is_monotonic_increasing:
        raise RuntimeError("CVICR source timestamps not monotonic")
    expected_availability = validated["date"] + BAR
    if not validated["feature_available_time_utc"].equals(
        expected_availability
    ):
        raise RuntimeError("CVICR feature availability drift")
    if not validated["trade_earliest_time_utc"].equals(
        expected_availability
    ):
        raise RuntimeError("CVICR trade availability drift")
    if exact_grid:
        expected = pd.date_range(
            SOURCE_START,
            SOURCE_END,
            freq=BAR,
            inclusive="left",
        )
        if len(validated) != 420_768 or not validated["date"].equals(
            pd.Series(expected, name="date")
        ):
            raise RuntimeError("CVICR source five-minute grid drift")
    quote = validated[["spot_quote_notional", "um_quote_notional"]].to_numpy(
        dtype=float
    )
    signed = validated[
        ["spot_signed_quote_notional", "um_signed_quote_notional"]
    ].to_numpy(dtype=float)
    numeric_valid = (
        np.isfinite(quote).all(axis=1)
        & np.isfinite(signed).all(axis=1)
        & (quote > 0.0).all(axis=1)
        & (np.abs(signed) <= quote).all(axis=1)
    )
    validated["_row_valid"] = (
        validated["source_complete"].to_numpy(dtype=bool) & numeric_valid
    )
    return validated


def load_source(path: str | Path = prereg.SOURCE) -> pd.DataFrame:
    if str(path) != prereg.SOURCE:
        raise RuntimeError("CVICR support must use the frozen source path")
    frame = pd.read_csv(
        _path(path),
        usecols=list(prereg.SOURCE_ALLOWLIST),
        dtype={
            "date": "string",
            "feature_available_time_utc": "string",
            "trade_earliest_time_utc": "string",
            "spot_quote_notional": "float64",
            "um_quote_notional": "float64",
            "spot_signed_quote_notional": "float64",
            "um_signed_quote_notional": "float64",
            "source_complete": "string",
        },
    )
    frame = frame.loc[:, prereg.SOURCE_ALLOWLIST]
    return validate_source_frame(frame)


def _linear_quantile(values: Iterable[float], quantile: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise RuntimeError("CVICR quantile input invalid")
    return float(np.quantile(array, quantile, method="linear"))


def _lower_median(values: Iterable[int]) -> int:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        raise RuntimeError("CVICR lower median input empty")
    return ordered[(len(ordered) - 1) // 2]


def _first_passage(cumulative: np.ndarray, target: float) -> int | None:
    if not np.isfinite(target) or target <= 0.0:
        return None
    indices = np.flatnonzero(cumulative >= target)
    if not len(indices):
        return None
    value = int(indices[0])
    return value if value <= LATEST_ANCHOR_INDEX else None


def _sign(numerator: float, denominator: float) -> int:
    if (
        not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or denominator <= 0.0
    ):
        return 0
    return int(np.sign(numerator / denominator))


FEATURE_COLUMNS = (
    "source_day",
    "spot_anchor_index",
    "um_anchor_index",
    "early_index",
    "late_index",
    "leader",
    "side_sign",
    "gap_bars",
    "gap_reference_count",
    "gap_threshold",
    "gap_pass",
    "initial_conflict",
    "late_alignment",
    "leader_persistence",
    "laggard_resolution",
    "primary",
    "gap_only",
    "initial_conflict_only",
    "late_alignment_only",
    "no_leader_persistence",
    "no_gap_tail",
    "fixed_valid",
    "fixed_early_index",
    "fixed_late_index",
    "fixed_leader",
    "fixed_side_sign",
    "stale_valid",
)


def build_daily_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(frame) % ROWS_PER_DAY:
        raise RuntimeError("CVICR source cannot reshape to UTC days")
    days = frame["date"].iloc[::ROWS_PER_DAY].reset_index(drop=True)
    if not days.eq(days.dt.floor("D")).all():
        raise RuntimeError("CVICR day blocks do not start at UTC midnight")
    count = len(days)
    spot_quote = frame["spot_quote_notional"].to_numpy(dtype=float).reshape(
        count, ROWS_PER_DAY
    )
    um_quote = frame["um_quote_notional"].to_numpy(dtype=float).reshape(
        count, ROWS_PER_DAY
    )
    spot_signed = frame[
        "spot_signed_quote_notional"
    ].to_numpy(dtype=float).reshape(count, ROWS_PER_DAY)
    um_signed = frame["um_signed_quote_notional"].to_numpy(
        dtype=float
    ).reshape(count, ROWS_PER_DAY)
    valid = frame["_row_valid"].to_numpy(dtype=bool).reshape(
        count, ROWS_PER_DAY
    )
    complete = valid.all(axis=1)
    spot_totals = np.where(
        complete,
        np.sum(spot_quote, axis=1),
        np.nan,
    )
    um_totals = np.where(
        complete,
        np.sum(um_quote, axis=1),
        np.nan,
    )
    spot_anchor_history: list[int | None] = [None] * count
    um_anchor_history: list[int | None] = [None] * count
    gap_history: list[int] = []
    records: list[dict[str, Any]] = []
    funnel = {
        "utc_days": count,
        "complete_reference_days": int(complete.sum()),
        "expected_volume_ready": 0,
        "paired_non_tied_prefix_valid": 0,
        "gap_reference_ready": 0,
        "gap_pass": 0,
        "initial_conflict": 0,
        "late_alignment": 0,
        "raw_primary": 0,
        "raw_fixed_expected_time": 0,
        "raw_stale_laggard": 0,
    }

    for index, day in enumerate(days):
        start = max(0, index - prereg.Policy().reference_calendar_days)
        spot_reference = spot_totals[start:index]
        um_reference = um_totals[start:index]
        spot_reference = spot_reference[np.isfinite(spot_reference)]
        um_reference = um_reference[np.isfinite(um_reference)]
        if (
            len(spot_reference)
            < prereg.Policy().reference_complete_days_min
            or len(um_reference)
            < prereg.Policy().reference_complete_days_min
        ):
            continue
        funnel["expected_volume_ready"] += 1
        spot_target = prereg.Policy().intrinsic_volume_fraction * float(
            np.median(spot_reference)
        )
        um_target = prereg.Policy().intrinsic_volume_fraction * float(
            np.median(um_reference)
        )
        sq = np.where(np.isfinite(spot_quote[index]), spot_quote[index], 0.0)
        uq = np.where(np.isfinite(um_quote[index]), um_quote[index], 0.0)
        ss = np.where(np.isfinite(spot_signed[index]), spot_signed[index], 0.0)
        us = np.where(np.isfinite(um_signed[index]), um_signed[index], 0.0)
        cumulative_spot_quote = np.cumsum(sq, dtype=np.float64)
        cumulative_um_quote = np.cumsum(uq, dtype=np.float64)
        cumulative_spot_signed = np.cumsum(ss, dtype=np.float64)
        cumulative_um_signed = np.cumsum(us, dtype=np.float64)
        spot_anchor = _first_passage(cumulative_spot_quote, spot_target)
        um_anchor = _first_passage(cumulative_um_quote, um_target)
        if spot_anchor is not None and valid[index, : spot_anchor + 1].all():
            spot_anchor_history[index] = spot_anchor
        if um_anchor is not None and valid[index, : um_anchor + 1].all():
            um_anchor_history[index] = um_anchor
        if spot_anchor is None or um_anchor is None or spot_anchor == um_anchor:
            continue
        early = min(spot_anchor, um_anchor)
        late = max(spot_anchor, um_anchor)
        if late + 1 >= ROWS_PER_DAY or not valid[index, : late + 2].all():
            continue
        funnel["paired_non_tied_prefix_valid"] += 1
        leader = "spot" if spot_anchor < um_anchor else "um"
        laggard = "um" if leader == "spot" else "spot"
        gap = late - early
        reference_gaps = gap_history[-prereg.Policy().gap_reference_pairs :]
        gap_ready = (
            len(reference_gaps) >= prereg.Policy().gap_reference_pairs_min
        )
        gap_threshold = (
            _linear_quantile(reference_gaps, prereg.Policy().gap_quantile)
            if gap_ready
            else np.nan
        )
        gap_pass = bool(gap_ready and gap >= gap_threshold)
        if gap_ready:
            funnel["gap_reference_ready"] += 1
        if gap_pass:
            funnel["gap_pass"] += 1

        quote_cumulative = {
            "spot": cumulative_spot_quote,
            "um": cumulative_um_quote,
        }
        signed_cumulative = {
            "spot": cumulative_spot_signed,
            "um": cumulative_um_signed,
        }

        def flow_sign(venue: str, at: int) -> int:
            return _sign(
                float(signed_cumulative[venue][at]),
                float(quote_cumulative[venue][at]),
            )

        side_sign = flow_sign(leader, early)
        early_laggard = flow_sign(laggard, early)
        late_leader = flow_sign(leader, late)
        late_laggard = flow_sign(laggard, late)
        initial_conflict = bool(
            side_sign != 0 and early_laggard == -side_sign
        )
        leader_persistence = bool(
            side_sign != 0 and late_leader == side_sign
        )
        laggard_resolution = bool(
            side_sign != 0 and late_laggard == side_sign
        )
        late_alignment = bool(leader_persistence and laggard_resolution)
        primary = bool(gap_pass and initial_conflict and late_alignment)
        if initial_conflict:
            funnel["initial_conflict"] += 1
        if late_alignment:
            funnel["late_alignment"] += 1
        if primary:
            funnel["raw_primary"] += 1

        anchor_start = max(
            0, index - prereg.Policy().reference_calendar_days
        )
        prior_spot_anchors = [
            value
            for value in spot_anchor_history[anchor_start:index]
            if value is not None
        ]
        prior_um_anchors = [
            value
            for value in um_anchor_history[anchor_start:index]
            if value is not None
        ]
        fixed_valid = False
        fixed_early = -1
        fixed_late = -1
        fixed_leader = ""
        fixed_side_sign = 0
        if (
            gap_ready
            and len(prior_spot_anchors)
            >= prereg.Policy().reference_complete_days_min
            and len(prior_um_anchors)
            >= prereg.Policy().reference_complete_days_min
        ):
            fixed_spot = _lower_median(prior_spot_anchors)
            fixed_um = _lower_median(prior_um_anchors)
            if fixed_spot != fixed_um:
                fixed_early = min(fixed_spot, fixed_um)
                fixed_late = max(fixed_spot, fixed_um)
                if (
                    fixed_late + 1 < ROWS_PER_DAY
                    and valid[index, : fixed_late + 2].all()
                ):
                    fixed_leader = (
                        "spot" if fixed_spot < fixed_um else "um"
                    )
                    fixed_laggard = (
                        "um" if fixed_leader == "spot" else "spot"
                    )
                    fixed_side_sign = flow_sign(
                        fixed_leader, fixed_early
                    )
                    fixed_valid = bool(
                        fixed_late - fixed_early >= gap_threshold
                        and fixed_side_sign != 0
                        and flow_sign(fixed_laggard, fixed_early)
                        == -fixed_side_sign
                        and flow_sign(fixed_leader, fixed_late)
                        == fixed_side_sign
                        and flow_sign(fixed_laggard, fixed_late)
                        == fixed_side_sign
                    )
        if fixed_valid:
            funnel["raw_fixed_expected_time"] += 1

        stale_valid = False
        if index > 0 and complete[index - 1] and gap_pass and side_sign != 0:
            previous_quote = {
                "spot": np.cumsum(
                    spot_quote[index - 1], dtype=np.float64
                ),
                "um": np.cumsum(um_quote[index - 1], dtype=np.float64),
            }
            previous_signed = {
                "spot": np.cumsum(
                    spot_signed[index - 1], dtype=np.float64
                ),
                "um": np.cumsum(um_signed[index - 1], dtype=np.float64),
            }

            def stale_sign(at: int) -> int:
                return _sign(
                    float(previous_signed[laggard][at]),
                    float(previous_quote[laggard][at]),
                )

            stale_valid = bool(
                stale_sign(early) == -side_sign
                and stale_sign(late) == side_sign
                and leader_persistence
            )
        if stale_valid:
            funnel["raw_stale_laggard"] += 1

        records.append(
            {
                "source_day": pd.Timestamp(day),
                "spot_anchor_index": spot_anchor,
                "um_anchor_index": um_anchor,
                "early_index": early,
                "late_index": late,
                "leader": leader,
                "side_sign": side_sign,
                "gap_bars": gap,
                "gap_reference_count": len(reference_gaps),
                "gap_threshold": gap_threshold,
                "gap_pass": gap_pass,
                "initial_conflict": initial_conflict,
                "late_alignment": late_alignment,
                "leader_persistence": leader_persistence,
                "laggard_resolution": laggard_resolution,
                "primary": primary,
                "gap_only": bool(gap_pass and side_sign != 0),
                "initial_conflict_only": bool(
                    gap_pass and initial_conflict
                ),
                "late_alignment_only": bool(
                    gap_pass and side_sign != 0 and late_alignment
                ),
                "no_leader_persistence": bool(
                    gap_pass and initial_conflict and laggard_resolution
                ),
                "no_gap_tail": bool(initial_conflict and late_alignment),
                "fixed_valid": fixed_valid,
                "fixed_early_index": fixed_early,
                "fixed_late_index": fixed_late,
                "fixed_leader": fixed_leader,
                "fixed_side_sign": fixed_side_sign,
                "stale_valid": stale_valid,
            }
        )
        gap_history.append(gap)
    return pd.DataFrame(records, columns=FEATURE_COLUMNS), funnel


def signal_id(
    control: str,
    source_day: Any,
    causal_origin: Any,
    resolution: Any,
    signal_available: Any,
    decision: Any,
    entry: Any,
    exit_time: Any,
    side: str,
    leader: str,
) -> str:
    identity = {
        "causal_origin_time": _format_time(causal_origin),
        "control": control,
        "decision_time": _format_time(decision),
        "entry_time": _format_time(entry),
        "exit_time": _format_time(exit_time),
        "leader": leader,
        "policy_id": prereg.Policy().policy_id,
        "policy": asdict(prereg.Policy()),
        "resolution_time": _format_time(resolution),
        "signal_available_time": _format_time(signal_available),
        "side": side,
        "source_day": _format_day(source_day),
        "source_sha256": prereg.SOURCE_SHA256,
    }
    return canonical_hash(identity)


def _candidate_row(
    control: str,
    source_day: Any,
    early_index: int,
    late_index: int,
    side_sign: int,
    leader: str,
    *,
    entry_delay_bars: int = 0,
) -> dict[str, Any]:
    if side_sign not in (-1, 1) or leader not in ("spot", "um"):
        raise RuntimeError("CVICR candidate side or leader invalid")
    day = pd.Timestamp(source_day)
    causal_origin = day + int(early_index) * BAR
    resolution = day + int(late_index) * BAR
    signal_available = resolution + BAR
    decision = resolution + 2 * BAR
    entry = decision + entry_delay_bars * BAR
    exit_time = entry + prereg.Policy().hold_bars * BAR
    side = "LONG" if side_sign == 1 else "SHORT"
    return {
        "control": control,
        "signal_id": signal_id(
            control,
            day,
            causal_origin,
            resolution,
            signal_available,
            decision,
            entry,
            exit_time,
            side,
            leader,
        ),
        "source_day": day,
        "causal_origin_time": causal_origin,
        "resolution_time": resolution,
        "signal_available_time": signal_available,
        "decision_time": decision,
        "entry_time": entry,
        "exit_time": exit_time,
        "side": side,
        "leader": leader,
    }


def raw_candidates(
    features: pd.DataFrame,
    control: str,
    *,
    entry_delay_bars: int = 0,
) -> pd.DataFrame:
    if control == "fixed_expected_time_clocks":
        selected = features.loc[features["fixed_valid"]]
        rows = [
            _candidate_row(
                control,
                row.source_day,
                row.fixed_early_index,
                row.fixed_late_index,
                row.fixed_side_sign,
                row.fixed_leader,
                entry_delay_bars=entry_delay_bars,
            )
            for row in selected.itertuples(index=False)
        ]
    else:
        mask_name = (
            "stale_valid"
            if control == "stale_laggard_flow_24h"
            else "primary"
            if control in (
                "primary",
                "one_bar_execution_delay",
                "one_hour_execution_delay",
            )
            else control
        )
        selected = features.loc[features[mask_name]]
        rows = [
            _candidate_row(
                control,
                row.source_day,
                row.early_index,
                row.late_index,
                row.side_sign,
                row.leader,
                entry_delay_bars=entry_delay_bars,
            )
            for row in selected.itertuples(index=False)
        ]
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def reserve_nonoverlap(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    selected: list[dict[str, Any]] = []
    previous_exit: pd.Timestamp | None = None
    ordered = rows.sort_values(
        ["entry_time", "signal_id"],
        kind="mergesort",
    )
    for row in ordered.itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        exit_time = pd.Timestamp(row.exit_time)
        if previous_exit is not None and entry < previous_exit:
            continue
        selected.append({column: getattr(row, column) for column in CLOCK_COLUMNS})
        previous_exit = exit_time
    return pd.DataFrame(selected, columns=CLOCK_COLUMNS)


def _same_clock_variant(
    primary: pd.DataFrame,
    control: str,
    sides: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row, side in zip(primary.itertuples(index=False), sides, strict=True):
        side_sign = 1 if side == "LONG" else -1
        candidate = {
            column: getattr(row, column) for column in CLOCK_COLUMNS
        }
        candidate["control"] = control
        candidate["side"] = side
        candidate["signal_id"] = signal_id(
            control,
            row.source_day,
            row.causal_origin_time,
            row.resolution_time,
            row.signal_available_time,
            row.decision_time,
            row.entry_time,
            row.exit_time,
            "LONG" if side_sign == 1 else "SHORT",
            row.leader,
        )
        rows.append(candidate)
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def build_controls(
    features: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    raw: dict[str, pd.DataFrame] = {
        name: raw_candidates(features, name)
        for name in INDEPENDENT_CONTROLS
    }
    controls = {
        name: reserve_nonoverlap(raw[name])
        for name in INDEPENDENT_CONTROLS
    }
    primary = controls["primary"]
    controls["exact_direction_flip"] = _same_clock_variant(
        primary,
        "exact_direction_flip",
        [
            "SHORT" if side == "LONG" else "LONG"
            for side in primary["side"]
        ],
    )
    random_sides = []
    for entry in primary["entry_time"]:
        digest = hashlib.sha256(
            f"CVICR-72|{_format_time(entry)}".encode("ascii")
        ).digest()
        random_sides.append("LONG" if digest[0] < 128 else "SHORT")
    controls["deterministic_random_side"] = _same_clock_variant(
        primary,
        "deterministic_random_side",
        random_sides,
    )
    delayed = {
        "one_bar_execution_delay": 1,
        "one_hour_execution_delay": 12,
    }
    for name, delay in delayed.items():
        raw[name] = raw_candidates(
            features,
            name,
            entry_delay_bars=delay,
        )
        controls[name] = reserve_nonoverlap(raw[name])
    if set(controls) != set(prereg.CONTROL_ORDER):
        raise RuntimeError("CVICR constructed control set drift")
    ordered = {name: controls[name] for name in prereg.CONTROL_ORDER}
    return ordered, {name: len(raw.get(name, controls[name])) for name in ordered}


def _contained(
    rows: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    mask = (
        rows["source_day"].ge(start)
        & rows["causal_origin_time"].ge(start)
        & rows["resolution_time"].ge(start)
        & rows["signal_available_time"].ge(start)
        & rows["decision_time"].ge(start)
        & rows["entry_time"].ge(start)
        & rows["causal_origin_time"].lt(end)
        & rows["resolution_time"].lt(end)
        & rows["signal_available_time"].lt(end)
        & rows["decision_time"].lt(end)
        & rows["entry_time"].lt(end)
        & rows["exit_time"].le(end)
    )
    return rows.loc[mask].copy()


def _window(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    start, end = WINDOWS[name]
    return _contained(rows, start, end).sort_values(
        ["entry_time", "signal_id"],
        kind="mergesort",
    )


def _longest_run(values: Iterable[str]) -> int:
    best = 0
    current = 0
    previous: str | None = None
    for value in values:
        if value == previous:
            current += 1
        else:
            current = 1
            previous = value
        best = max(best, current)
    return best


def clock_stats(rows: pd.DataFrame) -> dict[str, Any]:
    ordered = rows.sort_values(
        ["entry_time", "signal_id"],
        kind="mergesort",
    )
    total = len(ordered)
    if not total:
        return {
            "events": 0,
            "long": 0,
            "short": 0,
            "long_share": None,
            "short_share": None,
            "spot_led": 0,
            "um_led": 0,
            "spot_led_share": None,
            "um_led_share": None,
            "active_months": 0,
            "maximum_month_share": None,
            "maximum_quarter_share": None,
            "maximum_gap_days": None,
            "maximum_same_side_run": 0,
            "maximum_same_leader_run": 0,
        }
    side = ordered["side"].value_counts().to_dict()
    leader = ordered["leader"].value_counts().to_dict()
    entry = ordered["entry_time"].dt.tz_convert(None)
    month_counts = entry.dt.to_period("M").astype(str).value_counts()
    quarter_counts = entry.dt.to_period("Q").astype(str).value_counts()
    gaps = ordered["entry_time"].diff().dropna()
    return {
        "events": total,
        "long": int(side.get("LONG", 0)),
        "short": int(side.get("SHORT", 0)),
        "long_share": float(side.get("LONG", 0) / total),
        "short_share": float(side.get("SHORT", 0) / total),
        "spot_led": int(leader.get("spot", 0)),
        "um_led": int(leader.get("um", 0)),
        "spot_led_share": float(leader.get("spot", 0) / total),
        "um_led_share": float(leader.get("um", 0) / total),
        "active_months": int(len(month_counts)),
        "maximum_month_share": float(month_counts.max() / total),
        "maximum_quarter_share": float(quarter_counts.max() / total),
        "maximum_gap_days": (
            float(gaps.max() / DAY) if not gaps.empty else None
        ),
        "maximum_same_side_run": _longest_run(ordered["side"]),
        "maximum_same_leader_run": _longest_run(ordered["leader"]),
    }


def exact_entry_jaccard(
    primary: pd.DataFrame,
    control: pd.DataFrame,
    split: str,
) -> float:
    left = set(_window(primary, split)["entry_time"])
    right = set(_window(control, split)["entry_time"])
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _count_ratio(
    primary: pd.DataFrame,
    control: pd.DataFrame,
    split: str,
) -> float | None:
    numerator = len(_window(primary, split))
    denominator = len(_window(control, split))
    return float(numerator / denominator) if denominator else None


def _timing_integrity(rows: pd.DataFrame, control: str) -> bool:
    if rows.empty:
        return True
    delay = (
        1
        if control == "one_bar_execution_delay"
        else 12
        if control == "one_hour_execution_delay"
        else 0
    )
    return bool(
        rows["signal_available_time"].eq(rows["resolution_time"] + BAR).all()
        and rows["decision_time"].eq(rows["resolution_time"] + 2 * BAR).all()
        and rows["entry_time"]
        .eq(rows["decision_time"] + delay * BAR)
        .all()
        and rows["exit_time"]
        .eq(rows["entry_time"] + prereg.Policy().hold_bars * BAR)
        .all()
        and rows["side"].isin(("LONG", "SHORT")).all()
        and rows["leader"].isin(("spot", "um")).all()
        and not rows["signal_id"].duplicated().any()
        and all(
            signal_id(
                control,
                row.source_day,
                row.causal_origin_time,
                row.resolution_time,
                row.signal_available_time,
                row.decision_time,
                row.entry_time,
                row.exit_time,
                row.side,
                row.leader,
            )
            == row.signal_id
            for row in rows.itertuples(index=False)
        )
    )


def _reservation_integrity(rows: pd.DataFrame) -> bool:
    ordered = rows.sort_values(
        ["entry_time", "signal_id"],
        kind="mergesort",
    )
    if len(ordered) < 2:
        return True
    return bool(
        ordered["entry_time"]
        .iloc[1:]
        .reset_index(drop=True)
        .ge(ordered["exit_time"].iloc[:-1].reset_index(drop=True))
        .all()
    )


def support_checks(
    controls: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, Any], dict[str, bool], dict[str, Any]]:
    gate = prereg.build_manifest()["source_support_gate"]
    primary = controls["primary"]
    statistics = {name: clock_stats(_window(primary, name)) for name in WINDOWS}
    train = statistics["train"]
    selection = statistics["selection"]

    selectivity: dict[str, dict[str, dict[str, float | None]]] = {}
    for control in prereg.SCORE_BEARING_CONTROLS:
        selectivity[control] = {}
        for split in ("train", "selection"):
            selectivity[control][split] = {
                "primary_over_control": _count_ratio(
                    primary, controls[control], split
                ),
                "exact_entry_jaccard": exact_entry_jaccard(
                    primary, controls[control], split
                ),
            }

    def shares_ok(stats: Mapping[str, Any], key: str, minimum: float) -> bool:
        left = stats[f"{key}_share"]
        right_key = "short" if key == "long" else "um_led"
        right = stats[f"{right_key}_share"]
        return bool(
            left is not None
            and right is not None
            and left >= minimum
            and right >= minimum
        )

    mechanism = gate["mechanism_selectivity"]
    ratio_limits = {
        "gap_only": mechanism["primary_over_gap_only_max"],
        "initial_conflict_only": mechanism[
            "primary_over_initial_conflict_only_max"
        ],
        "late_alignment_only": mechanism[
            "primary_over_late_alignment_only_max"
        ],
        "no_gap_tail": mechanism["primary_over_no_gap_tail_max"],
    }
    checks = {
        "train_events_min": train["events"] >= gate["train_events_min"],
        "each_train_year_events_min": all(
            statistics[str(year)]["events"]
            >= gate["each_train_year_events_min"]
            for year in (2020, 2021, 2022)
        ),
        "train_active_months_min": (
            train["active_months"] >= gate["train_active_months_min"]
        ),
        "train_side_support": shares_ok(
            train, "long", gate["train_each_side_share_min"]
        ),
        "train_leader_support": shares_ok(
            train, "spot_led", gate["train_each_leader_share_min"]
        ),
        "train_max_month_share": bool(
            train["maximum_month_share"] is not None
            and train["maximum_month_share"] <= gate["train_max_month_share"]
        ),
        "train_max_quarter_share": bool(
            train["maximum_quarter_share"] is not None
            and train["maximum_quarter_share"]
            <= gate["train_max_quarter_share"]
        ),
        "train_max_entry_gap_days": bool(
            train["maximum_gap_days"] is not None
            and train["maximum_gap_days"] <= gate["train_max_entry_gap_days"]
        ),
        "train_max_same_side_run": (
            train["maximum_same_side_run"] <= gate["train_max_same_side_run"]
        ),
        "train_max_same_leader_run": (
            train["maximum_same_leader_run"]
            <= gate["train_max_same_leader_run"]
        ),
        "selection_events_min": (
            selection["events"] >= gate["selection_events_min"]
        ),
        "selection_each_half_events_min": all(
            statistics[name]["events"]
            >= gate["selection_each_half_events_min"]
            for name in ("2023_h1", "2023_h2")
        ),
        "selection_each_quarter_events_min": all(
            statistics[name]["events"]
            >= gate["selection_each_quarter_events_min"]
            for name in ("2023_q1", "2023_q2", "2023_q3", "2023_q4")
        ),
        "selection_active_months_min": (
            selection["active_months"]
            >= gate["selection_active_months_min"]
        ),
        "selection_side_support": shares_ok(
            selection, "long", gate["selection_each_side_share_min"]
        ),
        "selection_leader_support": shares_ok(
            selection, "spot_led", gate["selection_each_leader_share_min"]
        ),
        "selection_max_month_share": bool(
            selection["maximum_month_share"] is not None
            and selection["maximum_month_share"]
            <= gate["selection_max_month_share"]
        ),
        "selection_max_entry_gap_days": bool(
            selection["maximum_gap_days"] is not None
            and selection["maximum_gap_days"]
            <= gate["selection_max_entry_gap_days"]
        ),
        "selection_max_same_side_run": (
            selection["maximum_same_side_run"]
            <= gate["selection_max_same_side_run"]
        ),
        "selection_max_same_leader_run": (
            selection["maximum_same_leader_run"]
            <= gate["selection_max_same_leader_run"]
        ),
        "mechanism_count_ratio_selectivity": all(
            metrics["primary_over_control"] is not None
            and metrics["primary_over_control"] <= ratio_limits[control]
            for control in ratio_limits
            for metrics in selectivity[control].values()
        ),
        "fixed_expected_time_entry_selectivity": all(
            selectivity["fixed_expected_time_clocks"][split][
                "exact_entry_jaccard"
            ]
            <= mechanism["fixed_expected_time_entry_jaccard_max"]
            and len(_window(controls["fixed_expected_time_clocks"], split)) > 0
            for split in ("train", "selection")
        ),
        "stale_laggard_entry_selectivity": all(
            selectivity["stale_laggard_flow_24h"][split][
                "exact_entry_jaccard"
            ]
            <= mechanism["stale_laggard_flow_entry_jaccard_max"]
            and len(_window(controls["stale_laggard_flow_24h"], split)) > 0
            for split in ("train", "selection")
        ),
        "all_controls_timing_and_identity": all(
            _timing_integrity(controls[name], name)
            for name in prereg.CONTROL_ORDER
        ),
        "all_controls_global_nonoverlap": all(
            _reservation_integrity(controls[name])
            for name in prereg.CONTROL_ORDER
        ),
        "clock_has_no_outcome_columns": not any(
            token in column.lower()
            for column in CLOCK_COLUMNS
            for token in FORBIDDEN_CLOCK_TOKENS
        ),
    }
    return statistics, checks, selectivity


SELECTIVITY_CHECKS = (
    "mechanism_count_ratio_selectivity",
    "fixed_expected_time_entry_selectivity",
    "stale_laggard_entry_selectivity",
)


def first_failure(
    source_checks: Mapping[str, bool],
    novelty_checks: Mapping[str, bool],
    *,
    artifact_eligible: bool,
) -> tuple[str, str | None]:
    source_failures = [
        name
        for name, passed in source_checks.items()
        if not passed and name not in SELECTIVITY_CHECKS
    ]
    if source_failures:
        return "source_support", source_failures[0]
    selectivity_failures = [
        name
        for name in SELECTIVITY_CHECKS
        if not source_checks.get(name, False)
    ]
    if selectivity_failures:
        return "mechanism_selectivity", selectivity_failures[0]
    if not artifact_eligible:
        return "artifact_eligibility", "synthetic_or_injected_build"
    novelty_failures = [
        name for name, passed in novelty_checks.items() if not passed
    ]
    if novelty_failures:
        return "novelty", novelty_failures[0]
    if not novelty_checks:
        return "novelty", "required_comparator_checks_missing"
    return "none", None


def _combined_clock(controls: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    nonempty = [
        controls[name]
        for name in prereg.CONTROL_ORDER
        if not controls[name].empty
    ]
    if not nonempty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    return (
        pd.concat(nonempty, ignore_index=True)
        .sort_values(
            ["entry_time", "control", "signal_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def deterministic_clock_bytes(
    controls: Mapping[str, pd.DataFrame],
) -> bytes:
    combined = _combined_clock(controls)
    if list(combined.columns) != list(CLOCK_COLUMNS):
        raise RuntimeError("CVICR clock schema drift")
    serialized = combined.copy()
    serialized["source_day"] = serialized["source_day"].map(_format_day)
    for column in (
        "causal_origin_time",
        "resolution_time",
        "signal_available_time",
        "decision_time",
        "entry_time",
        "exit_time",
    ):
        serialized[column] = serialized[column].map(_format_time)
    text = serialized.to_csv(
        index=False,
        columns=CLOCK_COLUMNS,
        lineterminator="\n",
    ).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(
        fileobj=buffer,
        mode="wb",
        filename="",
        mtime=0,
    ) as zipped:
        zipped.write(text)
    return buffer.getvalue()


def maximum_tolerant_matches(
    left: Iterable[pd.Timestamp],
    right: Iterable[pd.Timestamp],
    tolerance: pd.Timedelta,
) -> int:
    first = sorted(pd.Timestamp(value) for value in left)
    second = sorted(pd.Timestamp(value) for value in right)
    i = 0
    j = 0
    matched = 0
    while i < len(first) and j < len(second):
        if first[i] < second[j] - tolerance:
            i += 1
        elif second[j] < first[i] - tolerance:
            j += 1
        else:
            matched += 1
            i += 1
            j += 1
    return matched


def tolerant_entry_jaccard(
    left: pd.DataFrame,
    right: pd.DataFrame,
    tolerance: pd.Timedelta,
) -> float:
    matched = maximum_tolerant_matches(
        left["entry_time"],
        right["entry_time"],
        tolerance,
    )
    denominator = len(left) + len(right) - matched
    return float(matched / denominator) if denominator else 1.0


def _signed_occupancy(
    rows: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> np.ndarray:
    grid_size = int((end - start) / BAR)
    occupancy = np.zeros(grid_size, dtype=np.int8)
    for row in rows.sort_values("entry_time", kind="mergesort").itertuples(
        index=False
    ):
        entry = pd.Timestamp(row.entry_time)
        exit_time = pd.Timestamp(row.exit_time)
        if (entry - start) % BAR or (exit_time - start) % BAR:
            raise RuntimeError("CVICR comparator interval is off five-minute grid")
        left = int((entry - start) / BAR)
        right = int((exit_time - start) / BAR)
        if left < 0 or right > grid_size or left >= right:
            raise RuntimeError("CVICR comparator interval outside coverage")
        if np.any(occupancy[left:right] != 0):
            raise RuntimeError("CVICR comparator selected group overlaps itself")
        side = (
            int(row.side_sign)
            if hasattr(row, "side_sign")
            else 1
            if row.side == "LONG"
            else -1
        )
        occupancy[left:right] = side
    return occupancy


def occupancy_metrics(
    left: pd.DataFrame,
    right: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[float | None, float]:
    first = _signed_occupancy(left, start, end)
    second = _signed_occupancy(right, start, end)
    left_active = first != 0
    right_active = second != 0
    union = left_active | right_active
    position_jaccard = (
        float(np.sum(left_active & right_active) / np.sum(union))
        if np.any(union)
        else 1.0
    )
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        return None, position_jaccard
    correlation = float(np.corrcoef(first, second)[0, 1])
    if not np.isfinite(correlation):
        return None, position_jaccard
    return abs(correlation), position_jaccard


def _read_comparator_groups(
    payload: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], int]:
    groups: dict[str, dict[str, Any]] = {}
    decoded_rows = 0
    for contract in payload["novelty_contract"]["comparators"]:
        if sha256_file(contract["path"]) != contract["sha256"]:
            raise RuntimeError(f"CVICR comparator hash drift: {contract['id']}")
        if prereg.csv_header(contract["path"]) != contract["header"]:
            raise RuntimeError(f"CVICR comparator header drift: {contract['id']}")
        usecols = [
            contract["group_column"],
            contract["entry_column"],
            contract["side_column"],
        ]
        if contract.get("exit_column"):
            usecols.append(contract["exit_column"])
        else:
            usecols.append("hold_bars")
        usecols = list(dict.fromkeys(usecols))
        raw = pd.read_csv(_path(contract["path"]), usecols=usecols, dtype="string")
        decoded_rows += len(raw)
        for selected_group in contract["selected_groups"]:
            selected = raw.loc[
                raw[contract["group_column"]].eq(str(selected_group))
            ].copy()
            key = (
                f"{contract['id']}:{selected_group}"
                if len(contract["selected_groups"]) > 1
                else contract["id"]
            )
            if selected.empty:
                raise RuntimeError(f"CVICR comparator group empty: {key}")
            entry = pd.to_datetime(
                selected[contract["entry_column"]],
                utc=True,
                errors="raise",
            )
            if contract.get("exit_column"):
                exit_time = pd.to_datetime(
                    selected[contract["exit_column"]],
                    utc=True,
                    errors="raise",
                )
            else:
                hold = pd.to_numeric(
                    selected["hold_bars"],
                    errors="raise",
                ).astype(int)
                exit_time = entry + hold * BAR
            encoding = {
                str(name): int(value)
                for name, value in contract["side_encoding"].items()
            }
            side_sign = selected[contract["side_column"]].map(
                lambda value: encoding.get(str(value))
            )
            if side_sign.isna().any() or not side_sign.isin((-1, 1)).all():
                raise RuntimeError(f"CVICR comparator side invalid: {key}")
            rows = pd.DataFrame(
                {
                    "entry_time": entry,
                    "exit_time": exit_time,
                    "side_sign": side_sign.astype(int),
                }
            ).sort_values("entry_time", kind="mergesort")
            if rows["entry_time"].duplicated().any():
                raise RuntimeError(f"CVICR comparator entries duplicated: {key}")
            if not rows["exit_time"].gt(rows["entry_time"]).all():
                raise RuntimeError(f"CVICR comparator interval invalid: {key}")
            start, end = (
                pd.Timestamp(contract["declared_coverage"][0]),
                pd.Timestamp(contract["declared_coverage"][1]),
            )
            common_start = max(SOURCE_START, start)
            common_end = min(SOURCE_END, end)
            contained = rows.loc[
                rows["entry_time"].ge(common_start)
                & rows["exit_time"].le(common_end)
            ].reset_index(drop=True)
            if contained.empty:
                raise RuntimeError(
                    f"CVICR comparator empty in common coverage: {key}"
                )
            groups[key] = {
                "rows": contained,
                "start": common_start,
                "end": common_end,
                "six_hour_gate": bool(contract["six_hour_tolerant_gate"]),
                "artifact_id": contract["id"],
                "selected_group": str(selected_group),
            }
    return groups, decoded_rows


def evaluate_novelty(
    primary: pd.DataFrame,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool], int]:
    groups, decoded_rows = _read_comparator_groups(payload)
    contract = payload["novelty_contract"]
    report: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for key, group in groups.items():
        start = group["start"]
        end = group["end"]
        candidate = _contained(primary, start, end)
        comparator = group["rows"]
        if candidate.empty:
            raise RuntimeError(f"CVICR primary empty in comparator coverage: {key}")
        exact = tolerant_entry_jaccard(candidate, comparator, pd.Timedelta(0))
        one_bar = tolerant_entry_jaccard(candidate, comparator, BAR)
        twelve_bar = tolerant_entry_jaccard(candidate, comparator, 12 * BAR)
        six_hour = tolerant_entry_jaccard(candidate, comparator, 72 * BAR)
        correlation, position = occupancy_metrics(
            candidate,
            comparator,
            start,
            end,
        )
        row = {
            "artifact_id": group["artifact_id"],
            "selected_group": group["selected_group"],
            "common_coverage": [_format_time(start), _format_time(end)],
            "candidate_rows": len(candidate),
            "comparator_rows": len(comparator),
            "exact_entry_jaccard": exact,
            "one_bar_tolerant_jaccard": one_bar,
            "twelve_bar_tolerant_jaccard": twelve_bar,
            "six_hour_tolerant_jaccard": six_hour,
            "absolute_signed_occupancy_pearson": correlation,
            "position_time_jaccard_report_only": position,
            "six_hour_gate_applied": group["six_hour_gate"],
        }
        report[key] = row
        checks[f"{key}:exact_entry_jaccard"] = (
            exact <= contract["exact_entry_jaccard_max"]
        )
        checks[f"{key}:one_bar_tolerant_jaccard"] = (
            one_bar <= contract["one_bar_tolerant_jaccard_max"]
        )
        checks[f"{key}:twelve_bar_tolerant_jaccard"] = (
            twelve_bar <= contract["twelve_bar_tolerant_jaccard_max"]
        )
        checks[f"{key}:signed_occupancy_pearson"] = bool(
            correlation is not None
            and correlation
            <= contract["absolute_signed_occupancy_pearson_max"]
        )
        if group["six_hour_gate"]:
            checks[f"{key}:six_hour_tolerant_jaccard"] = (
                six_hour
                <= contract[
                    "six_hour_tolerant_jaccard_intrinsic_family_max"
                ]
            )
    return report, checks, decoded_rows


def _control_report(
    controls: Mapping[str, pd.DataFrame],
    raw_counts: Mapping[str, int],
) -> dict[str, Any]:
    primary = controls["primary"]
    return {
        name: {
            "raw_rows": int(raw_counts[name]),
            "globally_reserved_rows": len(controls[name]),
            "train": clock_stats(_window(controls[name], "train")),
            "selection": clock_stats(_window(controls[name], "selection")),
            "exact_entry_jaccard_to_primary": {
                split: exact_entry_jaccard(primary, controls[name], split)
                for split in ("train", "selection")
            },
        }
        for name in prereg.CONTROL_ORDER
    }


def _core_payload(
    features: pd.DataFrame,
    feature_funnel: Mapping[str, Any],
    controls: Mapping[str, pd.DataFrame],
    raw_counts: Mapping[str, int],
    source_audit: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    clock_bytes: bytes,
    *,
    artifact_eligible: bool,
) -> dict[str, Any]:
    statistics, source_checks, selectivity = support_checks(controls)
    source_passed = all(source_checks.values())
    novelty_report: dict[str, Any] = {}
    novelty_checks: dict[str, bool] = {}
    comparator_rows_decoded = 0
    comparator_status = "not_opened_source_support_failed"
    if source_passed and artifact_eligible:
        novelty_report, novelty_checks, comparator_rows_decoded = (
            evaluate_novelty(controls["primary"], preregistration)
        )
        comparator_status = "opened_after_source_support_pass"
    elif source_passed:
        comparator_status = "synthetic_build_not_authorized"
    novelty_passed = bool(
        source_passed
        and artifact_eligible
        and novelty_checks
        and all(novelty_checks.values())
    )
    first_failing_stage, first_failing_check = first_failure(
        source_checks,
        novelty_checks,
        artifact_eligible=artifact_eligible,
    )
    if not source_passed:
        decision = "retire_CVICR_72_unchanged_before_comparators_and_outcomes"
    elif not artifact_eligible:
        decision = "synthetic_build_cannot_authorize_comparators_or_outcomes"
    elif not novelty_passed:
        decision = "retire_CVICR_72_unchanged_before_outcomes"
    else:
        decision = "advance_to_separately_frozen_strict_economic_evaluator"
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": prereg.Policy().policy_id,
        "artifact_eligible": artifact_eligible,
        "outcomes_opened": False,
        "post_entry_return_computed": False,
        "funding_loaded": False,
        "source_incidence_opened": True,
        "comparator_rows_decoded": comparator_rows_decoded,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "implementation": {
            "source": str(SCRIPT_PATH),
            "source_sha256": sha256_file(SCRIPT_PATH),
            "tests": str(TEST_PATH),
            "tests_sha256": sha256_file(TEST_PATH),
            "contract": str(IMPLEMENTATION_CONTRACT),
            "contract_sha256": IMPLEMENTATION_CONTRACT_SHA256,
        },
        "source_audit": dict(source_audit),
        "feature_funnel": dict(feature_funnel),
        "primary_statistics": statistics,
        "control_report": _control_report(controls, raw_counts),
        "mechanism_selectivity": selectivity,
        "source_support_checks": source_checks,
        "source_support_passed": source_passed,
        "comparator_status": comparator_status,
        "novelty_report": novelty_report,
        "novelty_checks": novelty_checks,
        "novelty_passed": novelty_passed,
        "first_failing_stage": first_failing_stage,
        "first_failing_check": first_failing_check,
        "clock": {
            "path": str(DEFAULT_CLOCK_OUTPUT),
            "sha256": hashlib.sha256(clock_bytes).hexdigest(),
            "rows": len(_combined_clock(controls)),
            "columns": list(CLOCK_COLUMNS),
            "control_counts": {
                name: len(controls[name]) for name in prereg.CONTROL_ORDER
            },
        },
        "decision": decision,
        "authorized_next_stage": (
            "freeze_strict_economic_evaluator" if novelty_passed else None
        ),
        "outcome_boundary": {
            "source_rows_decoded": int(
                source_audit.get("source_rows_decoded", 0)
            ),
            "comparator_rows_decoded": comparator_rows_decoded,
            "post_entry_price_rows_decoded": 0,
            "funding_rows_decoded": 0,
            "future_return_rows_decoded": 0,
            "return_or_pnl_fields_decoded": 0,
            "pnl_cagr_mdd_values_decoded": 0,
            "network_calls": 0,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def build_support_from_features(
    features: pd.DataFrame,
    *,
    feature_funnel: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bytes]:
    controls, raw_counts = build_controls(features)
    clock_bytes = deterministic_clock_bytes(controls)
    payload = validate_preregistration()
    report = _core_payload(
        features,
        feature_funnel or {"synthetic_or_injected": True},
        controls,
        raw_counts,
        {"source_rows_decoded": 0, "synthetic_or_injected": True},
        payload,
        clock_bytes,
        artifact_eligible=False,
    )
    return report, clock_bytes


def build_real_support_payload() -> tuple[dict[str, Any], bytes]:
    _assert_protocol_committed()
    payload = validate_preregistration()
    bindings = verify_pre_source_bindings(payload)
    source = load_source()
    features, funnel = build_daily_features(source)
    controls, raw_counts = build_controls(features)
    clock_bytes = deterministic_clock_bytes(controls)
    source_audit = {
        "path": prereg.SOURCE,
        "sha256": prereg.SOURCE_SHA256,
        "header_sha256": prereg.SOURCE_HEADER_SHA256,
        "source_rows_decoded": len(source),
        "allowlist": list(prereg.SOURCE_ALLOWLIST),
        "first_timestamp": _format_time(source["date"].iloc[0]),
        "last_timestamp": _format_time(source["date"].iloc[-1]),
        "valid_rows": int(source["_row_valid"].sum()),
        "invalid_rows": int((~source["_row_valid"]).sum()),
        "pre_source_bindings": bindings,
        "synthetic_or_injected": False,
    }
    return (
        _core_payload(
            features,
            funnel,
            controls,
            raw_counts,
            source_audit,
            payload,
            clock_bytes,
            artifact_eligible=True,
        ),
        clock_bytes,
    )


def _write_once(path: str | Path, payload: bytes) -> str:
    output = _path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != payload:
            raise RuntimeError(f"CVICR noncanonical existing artifact: {path}")
        return "verified_existing"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            if output.read_bytes() != payload:
                raise RuntimeError(f"CVICR artifact race drift: {path}")
            return "verified_existing"
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def write_support(
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
) -> dict[str, Any]:
    if Path(clock_output) != DEFAULT_CLOCK_OUTPUT:
        raise RuntimeError("CVICR real clock output path is frozen")
    report, clock_bytes = build_real_support_payload()
    clock_status = _write_once(clock_output, clock_bytes)
    report_bytes = (
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    report_status = _write_once(report_output, report_bytes)
    return {
        "report_status": report_status,
        "clock_status": clock_status,
        "report": str(report_output),
        "clock": str(clock_output),
        "source_support_passed": report["source_support_passed"],
        "novelty_passed": report["novelty_passed"],
        "decision": report["decision"],
        "manifest_hash": report["manifest_hash"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_REPORT_OUTPUT))
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK_OUTPUT))
    args = parser.parse_args()
    print(
        json.dumps(
            write_support(args.output, args.clock_output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
