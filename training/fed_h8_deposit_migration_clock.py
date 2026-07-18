"""Outcome-blind event clocks for Fed H.8 Deposit Migration (H8DM-1)."""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Literal, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from training import build_fed_h8_deposit_migration_panel as source_builder


SOURCE = Path(
    "data/fed_h8_deposit_migration_2017_2023/"
    "fed_h8_deposit_migration_2017_2023.csv.gz"
)
BUILD_MANIFEST = Path("data/fed_h8_deposit_migration_2017_2023/build_manifest.json")
SOURCE_SHA256 = "c8d1bfb0bbd13ef6d35f09ad7367ef8d2d5bb28981376223b735746ade68a572"
BUILD_MANIFEST_SHA256 = (
    "1f0a194e628ab9c44c23fc4a923145dcf89a62bface745cc36872eeee919eda9"
)
DEFAULT_OUTPUT = "results/fed_h8_deposit_migration_preregistered_clock_2026-07-18.csv.gz"
NEW_YORK = ZoneInfo("America/New_York")
ROBUST_WINDOW_RELEASES = 104
TAIL_WINDOW_RELEASES = 52
STRUCTURAL_EXCLUSION_RELEASES = frozenset(
    {
        "2020-10-02",  # large/small panel-shift method revision
        "2023-03-31",  # FDIC bridge-bank treatment revision
        "2023-06-30",  # weekly seasonal-adjustment outlier method revision
    }
)

ClockMode = Literal[
    "primary",
    "migration_only",
    "borrowings_only",
    "cash_only",
    "no_agreement",
    "nsa_primary",
]


@dataclass(frozen=True)
class Event:
    release_date: str
    signal_time: str
    entry_time: str
    exit_time: str
    side: int
    clock_mode: str
    adjustment: str
    tail_quantile: float
    migration_bp: float
    small_borrowings_bp: float
    small_cash_bp: float
    migration_z: float
    borrowings_z: float
    cash_stress_z: float
    stress_score: float
    agreement_count: int
    threshold_abs: float


EVENT_COLUMNS = tuple(Event.__dataclass_fields__)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source() -> None:
    for path, expected in {
        SOURCE: SOURCE_SHA256,
        BUILD_MANIFEST: BUILD_MANIFEST_SHA256,
    }.items():
        if not expected or sha256_file(path) != expected:
            raise ValueError(f"H8DM-1 source hash mismatch: {path}")


def load_source(path: str | Path = SOURCE) -> pd.DataFrame:
    source = Path(path)
    if source == SOURCE:
        verify_source()
    frame = pd.read_csv(source, dtype=str)
    if tuple(frame.columns) != source_builder.PANEL_COLUMNS:
        raise ValueError("H8DM-1 source schema changed")
    if len(frame) != source_builder.FROZEN_COVERAGE[0]:
        raise ValueError("H8DM-1 source row count changed")
    if frame["release_date"].duplicated().any():
        raise ValueError("H8DM-1 source release dates are duplicated")
    release = pd.to_datetime(frame["release_time_utc"], utc=True, errors="raise")
    if not release.is_monotonic_increasing:
        raise ValueError("H8DM-1 release clock is not increasing")
    if release.iloc[-1] >= pd.Timestamp("2024-01-01", tz="UTC"):
        raise ValueError("H8DM-1 source escaped the frozen pre-2024 horizon")
    for column in source_builder.LEVEL_COLUMNS:
        parsed = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(parsed).all() or not parsed.gt(0).all():
            raise ValueError(f"H8DM-1 invalid source level: {column}")
        frame[column] = parsed
    frame["release_time"] = release
    return frame


def _log_change_bp(latest: pd.Series, prior: pd.Series) -> pd.Series:
    result = 10_000.0 * np.log(latest.astype(float) / prior.astype(float))
    if not np.isfinite(result).all():
        raise ValueError("H8DM-1 log change is non-finite")
    return result


def _rolling_mad(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _robust_z(
    series: pd.Series, *, window: int = ROBUST_WINDOW_RELEASES
) -> pd.Series:
    prior = series.astype(float).shift(1)
    median = prior.rolling(window=window, min_periods=window).median()
    mad = prior.rolling(window=window, min_periods=window).apply(
        _rolling_mad, raw=True
    )
    denominator = 1.4826 * mad
    zscore = (series.astype(float) - median) / denominator
    return zscore.where(denominator.gt(0) & np.isfinite(zscore))


def _tail_threshold(
    score: pd.Series,
    *,
    quantile: float,
    window: int = TAIL_WINDOW_RELEASES,
) -> pd.Series:
    if not 0.5 <= quantile < 1.0:
        raise ValueError("H8DM-1 tail quantile must be in [0.5, 1.0)")
    return (
        score.abs()
        .shift(1)
        .rolling(window=window, min_periods=window)
        .quantile(quantile)
    )


def build_features(
    frame: pd.DataFrame,
    *,
    adjustment: Literal["sa", "nsa"] = "sa",
    tail_quantile: float = 0.75,
) -> pd.DataFrame:
    if adjustment not in {"sa", "nsa"}:
        raise ValueError("H8DM-1 adjustment must be sa or nsa")
    result = frame.copy()

    def change(group: str, metric: str) -> pd.Series:
        prefix = f"{adjustment}_{group}_{metric}"
        return _log_change_bp(
            result[f"{prefix}_latest"], result[f"{prefix}_prior"]
        )

    large_other = change("large", "other_deposits")
    small_other = change("small", "other_deposits")
    result["migration_bp"] = large_other - small_other
    result["small_borrowings_bp"] = change("small", "borrowings")
    result["small_cash_bp"] = change("small", "cash_assets")
    result["migration_z"] = _robust_z(result["migration_bp"])
    result["borrowings_z"] = _robust_z(result["small_borrowings_bp"])
    result["cash_stress_z"] = -_robust_z(result["small_cash_bp"])
    result["stress_score"] = result[
        ["migration_z", "borrowings_z", "cash_stress_z"]
    ].mean(axis=1, skipna=False)
    signs = np.sign(
        result[["migration_z", "borrowings_z", "cash_stress_z"]].to_numpy(float)
    )
    composite_sign = np.sign(result["stress_score"].to_numpy(float))[:, None]
    result["agreement_count"] = np.sum(signs == composite_sign, axis=1)
    for name in (
        "stress_score",
        "migration_z",
        "borrowings_z",
        "cash_stress_z",
    ):
        result[f"{name}_threshold_abs"] = _tail_threshold(
            result[name], quantile=tail_quantile
        )
    result["tail_quantile"] = tail_quantile
    return result


def _execution_times(
    release: datetime, *, delay_weeks: int = 0
) -> tuple[datetime, datetime]:
    if release.tzinfo is None:
        raise ValueError("H8DM-1 release timestamp must be timezone-aware")
    if delay_weeks not in {0, 1, 4}:
        raise ValueError("H8DM-1 delay must be zero, one, or four weeks")
    local = release.astimezone(NEW_YORK)
    execution_date = local.date() + timedelta(weeks=delay_weeks)
    # H.8 is scheduled for "generally" 16:15 ET rather than an exchange-grade
    # timestamp.  A fixed 45-minute buffer prevents us from claiming access at
    # the nominal publication instant.
    entry = datetime.combine(execution_date, time(17, 0), tzinfo=NEW_YORK)
    exit_time = entry + timedelta(hours=48)
    if delay_weeks == 0 and entry <= release:
        raise ValueError("H8DM-1 entry must follow release")
    return entry.astimezone(timezone.utc), exit_time.astimezone(timezone.utc)


def _mode_score(
    features: pd.DataFrame, mode: ClockMode
) -> tuple[pd.Series, pd.Series, bool, str]:
    if mode in {"primary", "no_agreement", "nsa_primary"}:
        return (
            features["stress_score"],
            features["stress_score_threshold_abs"],
            mode != "no_agreement",
            "nsa" if mode == "nsa_primary" else "sa",
        )
    mapping = {
        "migration_only": "migration_z",
        "borrowings_only": "borrowings_z",
        "cash_only": "cash_stress_z",
    }
    column = mapping[mode]
    return features[column], features[f"{column}_threshold_abs"], False, "sa"


def build_events(
    frame: pd.DataFrame,
    *,
    mode: ClockMode = "primary",
    tail_quantile: float = 0.75,
    delay_weeks: int = 0,
) -> list[Event]:
    adjustment: Literal["sa", "nsa"] = "nsa" if mode == "nsa_primary" else "sa"
    features = build_features(
        frame,
        adjustment=adjustment,
        tail_quantile=tail_quantile,
    )
    score, threshold, require_agreement, adjustment_label = _mode_score(features, mode)
    events: list[Event] = []
    for index, row in features.iterrows():
        value = float(score.loc[index])
        cutoff = float(threshold.loc[index])
        if not math.isfinite(value) or not math.isfinite(cutoff):
            continue
        if abs(value) < cutoff:
            continue
        if require_agreement and int(row["agreement_count"]) < 2:
            continue
        if str(row["release_date"]) in STRUCTURAL_EXCLUSION_RELEASES:
            continue
        if str(row["release_weekday"]) not in {"Thursday", "Friday"}:
            continue
        side = -1 if value > 0 else 1 if value < 0 else 0
        if side == 0:
            continue
        release = pd.Timestamp(row["release_time"]).to_pydatetime()
        entry, exit_time = _execution_times(release, delay_weeks=delay_weeks)
        clock_mode = (
            f"{mode}_delay_{delay_weeks}w" if delay_weeks else mode
        )
        events.append(
            Event(
                release_date=str(row["release_date"]),
                signal_time=release.astimezone(timezone.utc).isoformat(),
                entry_time=entry.isoformat(),
                exit_time=exit_time.isoformat(),
                side=side,
                clock_mode=clock_mode,
                adjustment=adjustment_label,
                tail_quantile=float(tail_quantile),
                migration_bp=float(row["migration_bp"]),
                small_borrowings_bp=float(row["small_borrowings_bp"]),
                small_cash_bp=float(row["small_cash_bp"]),
                migration_z=float(row["migration_z"]),
                borrowings_z=float(row["borrowings_z"]),
                cash_stress_z=float(row["cash_stress_z"]),
                stress_score=float(row["stress_score"]),
                agreement_count=int(row["agreement_count"]),
                threshold_abs=cutoff,
            )
        )
    events_frame(events)
    return events


def events_frame(events: list[Event]) -> pd.DataFrame:
    records = [asdict(event) for event in events]
    frame = (
        pd.DataFrame.from_records(records)
        if records
        else pd.DataFrame(
            {column: pd.Series(dtype="object") for column in EVENT_COLUMNS}
        )
    )
    if frame.empty:
        return frame
    for column in ("signal_time", "entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    frame = frame.sort_values("entry_time").reset_index(drop=True)
    if not frame["side"].isin((-1, 1)).all():
        raise ValueError("H8DM-1 emitted invalid side")
    if not frame["entry_time"].gt(frame["signal_time"]).all():
        raise ValueError("H8DM-1 entry no longer follows signal")
    if not frame["exit_time"].gt(frame["entry_time"]).all():
        raise ValueError("H8DM-1 exit no longer follows entry")
    if len(frame) > 1:
        entries = frame["entry_time"].iloc[1:].reset_index(drop=True)
        exits = frame["exit_time"].iloc[:-1].reset_index(drop=True)
        if not entries.ge(exits).all():
            raise ValueError("H8DM-1 events overlap")
    return frame


def cast_mode(value: str) -> ClockMode:
    allowed = {
        "primary",
        "migration_only",
        "borrowings_only",
        "cash_only",
        "no_agreement",
        "nsa_primary",
    }
    if value not in allowed:
        raise ValueError(f"unknown H8DM-1 clock mode: {value}")
    return cast(ClockMode, value)
