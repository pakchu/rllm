"""Evaluate CDLTR-72A source support and clock novelty without market outcomes.

The evaluator is intentionally outcome-blind.  It reads only the three source
panels and the sanitized comparator clock bound by the immutable CDLTR-72A
preregistration.  It never imports a market simulator or reads BTC prices,
funding, returns, equity, PnL, or post-2023 observations.

The pure derivation functions are kept separate from ``run_evaluation`` so the
relay, support gates, controls, and novelty metrics can be frozen against
synthetic inputs before any real source-value row is opened.
"""

# Pandas' static stubs expose schema-validated DataFrame selections as broad
# Series/DataFrame/NaT unions. Runtime schema and null checks below narrow them.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportOperatorIssue=false, reportReturnType=false, reportAssignmentType=false

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from training import preregister_cross_domain_liquidity_transmission_relay as prereg


POLICY_ID = "CDLTR-72A"
PROTOCOL_VERSION = "cross_domain_liquidity_transmission_relay_support_v3"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_SOURCE = Path(
    "training/evaluate_cross_domain_liquidity_transmission_relay_support.py"
)
PREREGISTRATION_ARTIFACT = prereg.DEFAULT_OUTPUT
PREREGISTRATION_ARTIFACT_SHA256 = (
    "a70329eb292aff8a334e986450959661f633cc61deb232874c883c3d1b5982e0"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "85d81e605afde5c94219ab752c9d52a4e7f8ed8a5b7a97ff1dc725cf6e1c5021"
)
PREREGISTRATION_MANIFEST_HASH = (
    "7b094ece1c86ca84081476a1fb4b5035df1149a0eb1075aa5d657fd7eca799c7"
)

DEFAULT_OUTPUT_REPORT = Path(
    "results/cross_domain_liquidity_transmission_relay_support_2026-07-21.json"
)
DEFAULT_OUTPUT_CLOCK = Path(
    "results/cross_domain_liquidity_transmission_relay_support_clock_2026-07-21.csv.gz"
)

UTC = ZoneInfo("UTC")
NEW_YORK = ZoneInfo("America/New_York")
FIVE_MINUTES = pd.Timedelta(minutes=5)
MACRO_TTL = pd.Timedelta(hours=36)
RELAY_DEADLINE = pd.Timedelta(hours=36)
HOLD = pd.Timedelta(hours=72)
EVALUATION_START = pd.Timestamp("2021-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2023-01-01T00:00:00Z")
EVALUATION_END = pd.Timestamp("2024-01-01T00:00:00Z")

SOURCE_COLUMNS = {
    "rrp": tuple(prereg.SOURCE_BINDINGS["rrp"]["allowed_columns"]),
    "cboe": tuple(prereg.SOURCE_BINDINGS["cboe"]["allowed_columns"]),
    "network": tuple(prereg.SOURCE_BINDINGS["network"]["allowed_columns"]),
}
VOTE_COLUMNS = (
    "source",
    "observation_date",
    "available_at",
    "side",
    "valid",
)
CLOCK_COLUMNS = (
    "clock",
    "window",
    "decision_time_utc",
    "entry_time_utc",
    "exit_time_utc",
    "side",
)
PRIMARY_CLOCK = "primary"
CONTROL_NAMES = (
    "macro_only",
    "network_only",
    "reverse_order",
    "one_network_report_delay",
    "direction_flip",
    "deterministic_random_side",
)
ALL_CLOCKS = (PRIMARY_CLOCK, *CONTROL_NAMES)

SUPPORT_LIMITS = {
    "train_total_minimum": 60,
    "each_train_year_minimum": 25,
    "each_train_half_year_minimum": 12,
    "selection_total_minimum": 30,
    "each_selection_half_year_minimum": 12,
    "train_each_side_minimum": 18,
    "selection_each_side_minimum": 8,
    "maximum_month_share": 0.20,
    "maximum_weekday_share": 0.35,
}
NOVELTY_LIMITS = {
    "decision_date_jaccard_maximum": 0.30,
    "cdltr_dates_within_one_utc_day_fraction_maximum": 0.50,
    "signed_occupied_exposure_absolute_pearson_maximum": 0.40,
}


@dataclass(frozen=True)
class Config:
    output_report: str = str(DEFAULT_OUTPUT_REPORT)
    output_clock: str = str(DEFAULT_OUTPUT_CLOCK)


def _repository_path(path: str | Path) -> Path:
    raw = str(path)
    candidate = Path(path)
    if raw.startswith("~") or candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError("CDLTR support path must be repository-relative")
    root = REPOSITORY_ROOT.resolve(strict=True)
    current = REPOSITORY_ROOT
    for part in candidate.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError("CDLTR support path contains a symlink")
        if not current.exists():
            break
    target = REPOSITORY_ROOT / candidate
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise RuntimeError("CDLTR support path escapes the repository") from error
    return target


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    return _sha256_path(_repository_path(path))


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} columns missing: {missing}")


def _utc_series(values: pd.Series, label: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, utc=True, errors="raise")
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} timestamps are invalid") from error
    if bool(parsed.isna().any()):
        raise RuntimeError(f"{label} timestamps contain nulls")
    return parsed


def _date_series(values: pd.Series, label: str) -> pd.Series:
    text = values.astype("string")
    date_only = text.str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    midnight = text.str.fullmatch(r"\d{4}-\d{2}-\d{2} 00:00:00")
    if not bool((date_only | midnight).fillna(False).all()):
        raise RuntimeError(
            f"{label} dates must be canonical dates or midnight timestamps"
        )
    canonical_dates = text.str.slice(0, 10)
    try:
        parsed = pd.to_datetime(canonical_dates, format="%Y-%m-%d", errors="raise")
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} dates are invalid") from error
    if bool(parsed.isna().any()):
        raise RuntimeError(f"{label} dates contain nulls")
    normalized = parsed.dt.strftime("%Y-%m-%d")
    if not normalized.equals(canonical_dates.astype(str).reset_index(drop=True)):
        raise RuntimeError(f"{label} dates are not valid calendar dates")
    return parsed.dt.date


def _strict_bool_series(values: pd.Series, label: str) -> pd.Series:
    def parse(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        raise RuntimeError(f"{label} contains a non-boolean value")

    return values.map(parse).astype(bool)


def _validate_ordered_dates(frame: pd.DataFrame, label: str) -> None:
    dates = frame["observation_date"]
    if bool(dates.duplicated().any()):
        raise RuntimeError(f"{label} contains duplicate observation dates")
    if not bool(dates.is_monotonic_increasing):
        raise RuntimeError(f"{label} observation dates are not increasing")


def _validate_ordered_availability(frame: pd.DataFrame, label: str) -> None:
    available = frame["available_at"]
    if bool(available.duplicated().any()):
        raise RuntimeError(f"{label} contains duplicate availability timestamps")
    if not bool(available.is_monotonic_increasing):
        raise RuntimeError(f"{label} availability timestamps are not increasing")


def _vote_frame(
    rows: list[dict[str, Any]],
    label: str,
    *,
    collapse_simultaneous_availability: bool = False,
) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(rows, columns=VOTE_COLUMNS)
    if frame.empty:
        raise RuntimeError(f"{label} produced no vote rows")
    frame["available_at"] = _utc_series(frame["available_at"], label)
    frame["side"] = pd.to_numeric(frame["side"], errors="raise").astype(np.int8)
    frame["valid"] = frame["valid"].astype(bool)
    if not bool(frame["side"].isin((-1, 0, 1)).all()):
        raise RuntimeError(f"{label} produced an invalid side")
    if not bool(frame.loc[~frame["valid"], "side"].eq(0).all()):
        raise RuntimeError(f"{label} invalid rows must be neutral")
    _validate_ordered_dates(frame, label)
    if collapse_simultaneous_availability:
        if not bool(frame["available_at"].is_monotonic_increasing):
            raise RuntimeError(
                f"{label} availability regresses before simultaneous-batch collapse"
            )
        frame = (
            frame.sort_values(["available_at", "observation_date"], kind="mergesort")
            .groupby("available_at", sort=False, as_index=False)
            .tail(1)
            .sort_values("available_at", kind="mergesort")
            .reset_index(drop=True)
        )
    _validate_ordered_availability(frame, label)
    if any(value >= date(2024, 1, 1) for value in frame["observation_date"]):
        raise RuntimeError(f"{label} contains a post-2023 observation")
    frame = frame.loc[frame["available_at"].lt(EVALUATION_END)]
    if frame.empty:
        raise RuntimeError(f"{label} has no causally available pre-2024 votes")
    return frame.loc[:, list(VOTE_COLUMNS)].reset_index(drop=True)


def derive_rrp_votes(source: pd.DataFrame) -> pd.DataFrame:
    """Derive exact fifth-prior normal-operation-slot RRP votes."""
    _require_columns(source, SOURCE_COLUMNS["rrp"], "RRP source")
    frame = source.loc[:, list(SOURCE_COLUMNS["rrp"])].copy().reset_index(drop=True)
    frame["observation_date"] = _date_series(frame["operation_date"], "RRP")
    frame["available_at"] = _utc_series(frame["result_available_at_utc"], "RRP")
    frame["complete"] = _strict_bool_series(frame["source_complete"], "RRP")
    reason = frame["quarantine_reason"].fillna("").astype(str)
    if not bool(reason.loc[frame["complete"]].eq("").all()):
        raise RuntimeError("complete RRP rows may not carry quarantine reasons")
    if not bool(reason.loc[~frame["complete"]].ne("").all()):
        raise RuntimeError("incomplete RRP rows must carry quarantine reasons")
    accepted = pd.to_numeric(frame["total_amount_accepted_usd"], errors="coerce")
    if bool((~np.isfinite(accepted.loc[frame["complete"]])).any()):
        raise RuntimeError("complete RRP rows require finite accepted amounts")
    if bool((accepted.loc[frame["complete"]] < 0).any()):
        raise RuntimeError("RRP accepted amounts must be nonnegative")
    if bool(accepted.loc[~frame["complete"]].notna().any()):
        raise RuntimeError("quarantined RRP rows may not expose accepted amounts")
    _validate_ordered_dates(frame, "RRP")

    rows: list[dict[str, Any]] = []
    for position, row in enumerate(frame.itertuples(index=False)):
        valid = position >= 5 and bool(
            frame.loc[position - 5 : position, "complete"].all()
        )
        side = 0
        if valid:
            delta = float(accepted.iloc[position] - accepted.iloc[position - 5])
            side = 1 if delta < 0 else -1 if delta > 0 else 0
        rows.append(
            {
                "source": "rrp",
                "observation_date": row.observation_date,
                "available_at": row.available_at,
                "side": side,
                "valid": valid,
            }
        )
    return _vote_frame(rows, "RRP")


def derive_cboe_votes(source: pd.DataFrame) -> pd.DataFrame:
    """Make each intersection-date close usable at the next date's 09:35 ET."""
    _require_columns(source, SOURCE_COLUMNS["cboe"], "Cboe source")
    frame = source.loc[:, list(SOURCE_COLUMNS["cboe"])].copy().reset_index(drop=True)
    frame["observation_date"] = _date_series(frame["observation_date"], "Cboe")
    _validate_ordered_dates(frame, "Cboe")
    short = pd.to_numeric(frame["VIX9D_close"], errors="coerce")
    long = pd.to_numeric(frame["VIX3M_close"], errors="coerce")

    rows: list[dict[str, Any]] = []
    for position in range(1, len(frame)):
        observation_date = cast(date, frame.at[position - 1, "observation_date"])
        release_date = cast(date, frame.at[position, "observation_date"])
        available = datetime.combine(release_date, time(9, 35), tzinfo=NEW_YORK)
        finite_positive = all(
            math.isfinite(float(value)) and float(value) > 0
            for value in (short.iloc[position - 1], long.iloc[position - 1])
        )
        side = 0
        if finite_positive:
            side = (
                1
                if short.iloc[position - 1] < long.iloc[position - 1]
                else -1
                if short.iloc[position - 1] > long.iloc[position - 1]
                else 0
            )
        rows.append(
            {
                "source": "cboe",
                "observation_date": observation_date,
                "available_at": available.astimezone(UTC),
                "side": side,
                "valid": finite_positive,
            }
        )
    return _vote_frame(rows, "Cboe")


def derive_network_votes(source: pd.DataFrame) -> pd.DataFrame:
    """Derive 7-calendar-day network breadth votes over exact 8-day runs."""
    _require_columns(source, SOURCE_COLUMNS["network"], "network source")
    frame = source.loc[:, list(SOURCE_COLUMNS["network"])].copy().reset_index(drop=True)
    frame["observation_date"] = _date_series(frame["observation_date"], "network")
    frame["available_at"] = _utc_series(frame["available_at"], "network")
    _validate_ordered_dates(frame, "network")
    metrics = ("AdrActCnt", "TxCnt", "TxTfrCnt")
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    by_date = {
        cast(date, row.observation_date): row for row in frame.itertuples(index=False)
    }

    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        current_date = cast(date, row.observation_date)
        dates = [
            current_date - pd.Timedelta(days=offset) for offset in range(7, -1, -1)
        ]
        consecutive = all(cast(date, item) in by_date for item in dates)
        values_positive = consecutive
        if consecutive:
            for item in dates:
                historical = by_date[cast(date, item)]
                values_positive = values_positive and all(
                    math.isfinite(float(getattr(historical, metric)))
                    and float(getattr(historical, metric)) > 0
                    for metric in metrics
                )
        side = 0
        valid = bool(consecutive and values_positive)
        if valid:
            prior = by_date[cast(date, current_date - pd.Timedelta(days=7))]
            signs = [
                int(
                    np.sign(
                        math.log(
                            float(getattr(row, metric)) / float(getattr(prior, metric))
                        )
                    )
                )
                for metric in metrics
            ]
            positive = sum(value > 0 for value in signs)
            negative = sum(value < 0 for value in signs)
            side = 1 if positive >= 2 else -1 if negative >= 2 else 0
        rows.append(
            {
                "source": "network",
                "observation_date": current_date,
                "available_at": row.available_at,
                "side": side,
                "valid": valid,
            }
        )
    return _vote_frame(
        rows,
        "network",
        collapse_simultaneous_availability=True,
    )


def _events_by_time(frame: pd.DataFrame) -> dict[pd.Timestamp, list[dict[str, Any]]]:
    output: dict[pd.Timestamp, list[dict[str, Any]]] = defaultdict(list)
    for row in frame.to_dict("records"):
        output[cast(pd.Timestamp, row["available_at"])].append(row)
    return output


def _macro_side(
    states: Mapping[str, tuple[int, pd.Timestamp] | None],
    timestamp: pd.Timestamp,
) -> int:
    live: list[int] = []
    for source in ("rrp", "cboe"):
        state = states.get(source)
        if state is None:
            return 0
        side, available = state
        if side == 0 or timestamp - available >= MACRO_TTL:
            return 0
        live.append(side)
    return live[0] if live[0] == live[1] else 0


def _relay_candidates(
    rrp_votes: pd.DataFrame,
    cboe_votes: pd.DataFrame,
    network_votes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return primary confirmation candidates and every macro episode onset."""
    macro = pd.concat([rrp_votes, cboe_votes], ignore_index=True).sort_values(
        ["available_at", "source"], kind="mergesort"
    )
    macro_events = _events_by_time(macro)
    network_events = _events_by_time(network_votes)
    expiry_times = {
        cast(pd.Timestamp, value) + MACRO_TTL for value in macro["available_at"]
    }
    timeline = sorted(set(macro_events) | set(network_events) | expiry_times)

    states: dict[str, tuple[int, pd.Timestamp] | None] = {
        "rrp": None,
        "cboe": None,
    }
    macro_side = 0
    active: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    onsets: list[dict[str, Any]] = []
    outcomes: Counter[str] = Counter()

    for timestamp in timeline:
        previous_side = macro_side
        for row in macro_events.get(timestamp, []):
            source = str(row["source"])
            side = int(row["side"])
            if bool(row["valid"]) and side != 0:
                states[source] = (side, timestamp)
        macro_side = _macro_side(states, timestamp)

        if active is not None and active["pending"]:
            if timestamp > active["deadline"]:
                active["pending"] = False
                outcomes["network_deadline_missed"] += 1

        if macro_side != previous_side:
            if active is not None and active["pending"]:
                active["pending"] = False
                outcomes["macro_left_before_confirmation"] += 1
            active = None
            if macro_side != 0:
                active = {
                    "onset": timestamp,
                    "side": macro_side,
                    "deadline": timestamp + RELAY_DEADLINE,
                    "pending": True,
                }
                onsets.append(
                    {
                        "decision_time": timestamp,
                        "side": macro_side,
                        "origin": "macro_episode_onset",
                    }
                )
                outcomes["macro_episodes"] += 1

        for report in network_events.get(timestamp, []):
            if active is None or not active["pending"]:
                continue
            if timestamp <= active["onset"]:
                continue
            active["pending"] = False
            report_side = int(report["side"])
            if timestamp > active["deadline"]:
                outcomes["late_first_network_report"] += 1
            elif not bool(report["valid"]):
                outcomes["invalid_first_network_report"] += 1
            elif report_side == 0:
                outcomes["neutral_first_network_report"] += 1
            elif report_side != active["side"]:
                outcomes["opposite_first_network_report"] += 1
            elif macro_side != active["side"]:
                outcomes["macro_not_live_at_confirmation"] += 1
            else:
                candidates.append(
                    {
                        "decision_time": timestamp,
                        "side": active["side"],
                        "origin": "network_confirmation",
                    }
                )
                outcomes["confirmed"] += 1

    if active is not None and active["pending"]:
        outcomes["unresolved_at_source_end"] += 1
    candidate = pd.DataFrame.from_records(
        candidates, columns=("decision_time", "side", "origin")
    )
    onset = pd.DataFrame.from_records(
        onsets, columns=("decision_time", "side", "origin")
    )
    return (
        candidate,
        onset,
        {key: int(value) for key, value in sorted(outcomes.items())},
    )


def _network_onset_candidates(network_votes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    previous_side = 0
    for row in network_votes.itertuples(index=False):
        side = int(row.side) if bool(row.valid) else 0
        if side != previous_side and side != 0:
            rows.append(
                {
                    "decision_time": row.available_at,
                    "side": side,
                    "origin": "network_vote_onset",
                }
            )
        previous_side = side
    return pd.DataFrame.from_records(rows, columns=("decision_time", "side", "origin"))


def _reverse_order_candidates(
    rrp_votes: pd.DataFrame,
    cboe_votes: pd.DataFrame,
    network_votes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Confirm a network onset with the first strictly later macro update."""
    macro = pd.concat([rrp_votes, cboe_votes], ignore_index=True).sort_values(
        ["available_at", "source"], kind="mergesort"
    )
    macro_events = _events_by_time(macro)
    network_events = _events_by_time(network_votes)
    expiry_times = {
        cast(pd.Timestamp, value) + MACRO_TTL for value in macro["available_at"]
    }
    timeline = sorted(set(macro_events) | set(network_events) | expiry_times)
    states: dict[str, tuple[int, pd.Timestamp] | None] = {
        "rrp": None,
        "cboe": None,
    }
    network_side = 0
    pending: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    audit: Counter[str] = Counter()

    for timestamp in timeline:
        updates = macro_events.get(timestamp, [])
        for row in updates:
            side = int(row["side"])
            if bool(row["valid"]) and side != 0:
                states[str(row["source"])] = (side, timestamp)
        current_macro = _macro_side(states, timestamp)

        if pending is not None and timestamp > pending["deadline"]:
            audit["macro_deadline_missed"] += 1
            pending = None
        if updates and pending is not None and timestamp > pending["onset"]:
            audit["first_macro_update_inspected"] += 1
            if current_macro == pending["side"]:
                rows.append(
                    {
                        "decision_time": timestamp,
                        "side": pending["side"],
                        "origin": "reverse_order_confirmation",
                    }
                )
                audit["confirmed"] += 1
            else:
                audit["first_macro_update_disagreed"] += 1
            pending = None

        for report in network_events.get(timestamp, []):
            current_network = int(report["side"]) if bool(report["valid"]) else 0
            if current_network != network_side:
                if pending is not None:
                    audit["network_left_before_confirmation"] += 1
                    pending = None
                if current_network != 0:
                    pending = {
                        "onset": timestamp,
                        "side": current_network,
                        "deadline": timestamp + RELAY_DEADLINE,
                    }
                    audit["network_episodes"] += 1
            network_side = current_network

    if pending is not None:
        audit["unresolved_at_source_end"] += 1
    return (
        pd.DataFrame.from_records(rows, columns=("decision_time", "side", "origin")),
        {key: int(value) for key, value in sorted(audit.items())},
    )


def next_entry_time(decision_time: Any) -> pd.Timestamp:
    decision = pd.Timestamp(decision_time)
    if decision.tzinfo is None:
        raise RuntimeError("CDLTR decision time must be timezone-aware")
    decision = decision.tz_convert("UTC")
    return decision.ceil("5min") + FIVE_MINUTES


def _window_for(
    timestamp: pd.Timestamp,
) -> tuple[str, pd.Timestamp, pd.Timestamp] | None:
    if EVALUATION_START <= timestamp < TRAIN_END:
        return "train", EVALUATION_START, TRAIN_END
    if TRAIN_END <= timestamp < EVALUATION_END:
        return "selection", TRAIN_END, EVALUATION_END
    return None


def schedule_candidates(
    candidates: pd.DataFrame,
    *,
    clock: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    _require_columns(candidates, ("decision_time", "side"), f"{clock} candidates")
    accepted: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    last_exit: pd.Timestamp | None = None
    ordered = candidates.copy()
    ordered["decision_time"] = _utc_series(
        ordered["decision_time"], f"{clock} candidates"
    )
    ordered = ordered.sort_values("decision_time", kind="mergesort")
    for row in ordered.itertuples(index=False):
        decision = cast(pd.Timestamp, row.decision_time)
        side = int(row.side)
        if side not in (-1, 1):
            raise RuntimeError(f"{clock} candidate has an invalid side")
        window = _window_for(decision)
        if window is None:
            dropped["outside_evaluation_windows"] += 1
            continue
        window_name, start, end = window
        entry = next_entry_time(decision)
        exit_time = entry + HOLD
        if not start <= decision <= entry < exit_time <= end:
            dropped["split_crossing"] += 1
            continue
        if last_exit is not None and entry < last_exit:
            dropped["global_overlap"] += 1
            continue
        accepted.append(
            {
                "clock": clock,
                "window": window_name,
                "decision_time_utc": decision,
                "entry_time_utc": entry,
                "exit_time_utc": exit_time,
                "side": side,
            }
        )
        last_exit = exit_time
    return (
        pd.DataFrame.from_records(accepted, columns=CLOCK_COLUMNS),
        {key: int(value) for key, value in sorted(dropped.items())},
    )


def _exact_primary_control(
    primary: pd.DataFrame,
    *,
    clock: str,
    sides: Iterable[int],
) -> pd.DataFrame:
    output = primary.copy()
    output["clock"] = clock
    output["side"] = list(sides)
    if not bool(output["side"].isin((-1, 1)).all()):
        raise RuntimeError(f"{clock} produced an invalid side")
    return output.loc[:, list(CLOCK_COLUMNS)].reset_index(drop=True)


def _random_side(entry_time: Any) -> int:
    timestamp = cast(pd.Timestamp, pd.Timestamp(entry_time)).tz_convert("UTC")
    key = f"CDLTR-72|20260721|{timestamp.isoformat()}"
    return 1 if hashlib.sha256(key.encode("ascii")).digest()[0] < 128 else -1


def _delayed_network_candidates(
    primary: pd.DataFrame, network_votes: pd.DataFrame
) -> pd.DataFrame:
    valid = network_votes.loc[network_votes["valid"]].sort_values(
        "available_at", kind="mergesort"
    )
    available = list(valid["available_at"])
    rows: list[dict[str, Any]] = []
    for event in primary.itertuples(index=False):
        later = next(
            (value for value in available if value > event.decision_time_utc), None
        )
        if later is not None:
            rows.append(
                {
                    "decision_time": later,
                    "side": int(event.side),
                    "origin": "next_valid_network_report",
                }
            )
    return pd.DataFrame.from_records(rows, columns=("decision_time", "side", "origin"))


def build_clocks(
    rrp_votes: pd.DataFrame,
    cboe_votes: pd.DataFrame,
    network_votes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    primary_candidates, macro_onsets, relay_audit = _relay_candidates(
        rrp_votes, cboe_votes, network_votes
    )
    primary, primary_drops = schedule_candidates(
        primary_candidates, clock=PRIMARY_CLOCK
    )

    macro_only, macro_drops = schedule_candidates(macro_onsets, clock="macro_only")
    network_onsets = _network_onset_candidates(network_votes)
    network_only, network_drops = schedule_candidates(
        network_onsets, clock="network_only"
    )
    reverse_candidates, reverse_audit = _reverse_order_candidates(
        rrp_votes, cboe_votes, network_votes
    )
    reverse_order, reverse_drops = schedule_candidates(
        reverse_candidates, clock="reverse_order"
    )
    delayed_candidates = _delayed_network_candidates(primary, network_votes)
    delayed, delayed_drops = schedule_candidates(
        delayed_candidates, clock="one_network_report_delay"
    )
    direction_flip = _exact_primary_control(
        primary,
        clock="direction_flip",
        sides=(-int(value) for value in primary["side"]),
    )
    random_side = _exact_primary_control(
        primary,
        clock="deterministic_random_side",
        sides=(_random_side(value) for value in primary["entry_time_utc"]),
    )
    clocks = [
        primary,
        macro_only,
        network_only,
        reverse_order,
        delayed,
        direction_flip,
        random_side,
    ]
    combined = pd.concat(clocks, ignore_index=True)
    combined = combined.sort_values(
        ["clock", "entry_time_utc", "decision_time_utc"], kind="mergesort"
    ).reset_index(drop=True)
    if set(combined["clock"].unique()) - set(ALL_CLOCKS):
        raise RuntimeError("CDLTR clock family drift")
    audit = {
        "primary_candidates": int(len(primary_candidates)),
        "macro_episode_onsets": int(len(macro_onsets)),
        "network_vote_onsets": int(len(network_onsets)),
        "reverse_order_candidates": int(len(reverse_candidates)),
        "relay": relay_audit,
        "reverse_order": reverse_audit,
        "drops": {
            "primary": primary_drops,
            "macro_only": macro_drops,
            "network_only": network_drops,
            "reverse_order": reverse_drops,
            "one_network_report_delay": delayed_drops,
            "direction_flip": {},
            "deterministic_random_side": {},
        },
    }
    return combined, audit


def _period_count(
    frame: pd.DataFrame, start: str, end: str, *, column: str = "entry_time_utc"
) -> int:
    values = frame[column]
    return int(((values >= pd.Timestamp(start)) & (values < pd.Timestamp(end))).sum())


def _maximum_share(frame: pd.DataFrame, period: str) -> float:
    if frame.empty:
        return 1.0
    entry = frame["entry_time_utc"]
    if period == "month":
        buckets = entry.dt.strftime("%Y-%m")
    elif period == "weekday":
        buckets = entry.dt.weekday
    else:
        raise ValueError(f"unsupported concentration period: {period}")
    return float(buckets.value_counts(normalize=True).max())


def support_summary(primary: pd.DataFrame) -> dict[str, Any]:
    _require_columns(primary, CLOCK_COLUMNS, "primary clock")
    train = primary.loc[primary["window"].eq("train")]
    selection = primary.loc[primary["window"].eq("selection")]
    year_counts = {
        "2021": _period_count(primary, "2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z"),
        "2022": _period_count(primary, "2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z"),
    }
    half_counts = {
        "2021H1": _period_count(
            primary, "2021-01-01T00:00:00Z", "2021-07-01T00:00:00Z"
        ),
        "2021H2": _period_count(
            primary, "2021-07-01T00:00:00Z", "2022-01-01T00:00:00Z"
        ),
        "2022H1": _period_count(
            primary, "2022-01-01T00:00:00Z", "2022-07-01T00:00:00Z"
        ),
        "2022H2": _period_count(
            primary, "2022-07-01T00:00:00Z", "2023-01-01T00:00:00Z"
        ),
        "2023H1": _period_count(
            primary, "2023-01-01T00:00:00Z", "2023-07-01T00:00:00Z"
        ),
        "2023H2": _period_count(
            primary, "2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"
        ),
    }
    side_counts = {
        "train": {
            "long": int(train["side"].eq(1).sum()),
            "short": int(train["side"].eq(-1).sum()),
        },
        "selection": {
            "long": int(selection["side"].eq(1).sum()),
            "short": int(selection["side"].eq(-1).sum()),
        },
    }
    concentrations = {
        window: {
            "maximum_month_share": _maximum_share(frame, "month"),
            "maximum_weekday_share": _maximum_share(frame, "weekday"),
        }
        for window, frame in (("train", train), ("selection", selection))
    }
    checks = {
        "train_total_minimum": len(train) >= SUPPORT_LIMITS["train_total_minimum"],
        "each_train_year_minimum": all(
            value >= SUPPORT_LIMITS["each_train_year_minimum"]
            for value in year_counts.values()
        ),
        "each_train_half_year_minimum": all(
            half_counts[name] >= SUPPORT_LIMITS["each_train_half_year_minimum"]
            for name in ("2021H1", "2021H2", "2022H1", "2022H2")
        ),
        "selection_total_minimum": len(selection)
        >= SUPPORT_LIMITS["selection_total_minimum"],
        "each_selection_half_year_minimum": all(
            half_counts[name] >= SUPPORT_LIMITS["each_selection_half_year_minimum"]
            for name in ("2023H1", "2023H2")
        ),
        "train_each_side_minimum": min(side_counts["train"].values())
        >= SUPPORT_LIMITS["train_each_side_minimum"],
        "selection_each_side_minimum": min(side_counts["selection"].values())
        >= SUPPORT_LIMITS["selection_each_side_minimum"],
        "maximum_month_share": all(
            value["maximum_month_share"] <= SUPPORT_LIMITS["maximum_month_share"]
            for value in concentrations.values()
        ),
        "maximum_weekday_share": all(
            value["maximum_weekday_share"] <= SUPPORT_LIMITS["maximum_weekday_share"]
            for value in concentrations.values()
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "limits": dict(SUPPORT_LIMITS),
        "checks": checks,
        "counts": {
            "train": int(len(train)),
            "selection": int(len(selection)),
            **year_counts,
            **half_counts,
        },
        "side_counts": side_counts,
        "concentration": concentrations,
        "clock_field": "entry_time_utc",
    }


def control_calendar_summary(clocks: pd.DataFrame) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for name in CONTROL_NAMES:
        frame = clocks.loc[clocks["clock"].eq(name)].sort_values(
            "entry_time_utc", kind="mergesort"
        )
        interval_order = bool(
            (
                (frame["decision_time_utc"] <= frame["entry_time_utc"])
                & (frame["entry_time_utc"] < frame["exit_time_utc"])
            ).all()
        )
        exact_hold = bool(
            (frame["exit_time_utc"] - frame["entry_time_utc"]).eq(HOLD).all()
        )
        side_valid = bool(frame["side"].isin((-1, 1)).all())
        within_windows = True
        for row in frame.itertuples(index=False):
            window = _window_for(cast(pd.Timestamp, row.decision_time_utc))
            if window is None:
                within_windows = False
                break
            expected, start, end = window
            within_windows = within_windows and bool(
                row.window == expected
                and start
                <= row.decision_time_utc
                <= row.entry_time_utc
                < row.exit_time_utc
                <= end
            )
        previous_exit = frame["exit_time_utc"].shift(1)
        nonoverlap = bool(
            (
                frame["entry_time_utc"].iloc[1:].array >= previous_exit.iloc[1:].array
            ).all()
        )
        checks = {
            "interval_order": interval_order,
            "exact_72h_hold": exact_hold,
            "side_valid": side_valid,
            "train_or_selection_contained": within_windows,
            "globally_nonoverlapping": nonoverlap,
            "post_2023_rows_absent": bool(
                frame["decision_time_utc"].lt(EVALUATION_END).all()
                and frame["exit_time_utc"].le(EVALUATION_END).all()
            ),
        }
        support_checks = support_summary(frame)
        summaries[name] = {
            "passed": bool(all(checks.values()) and support_checks["passed"]),
            "rows": int(len(frame)),
            "calendar_and_containment_checks": checks,
            "support": support_checks,
        }
    return {
        "passed": bool(all(value["passed"] for value in summaries.values())),
        "controls": summaries,
    }


def _as_utc_date_set(values: pd.Series) -> set[date]:
    timestamps = _utc_series(values, "novelty decision")
    return set(timestamps.dt.date)


def decision_date_overlap(
    candidate_decisions: pd.Series,
    comparator_decisions: pd.Series,
) -> dict[str, Any]:
    candidate_dates = _as_utc_date_set(candidate_decisions)
    comparator_dates = _as_utc_date_set(comparator_decisions)
    union = candidate_dates | comparator_dates
    intersection = candidate_dates & comparator_dates
    nearby = sum(
        any(abs((candidate - reference).days) <= 1 for reference in comparator_dates)
        for candidate in candidate_dates
    )
    jaccard = float(len(intersection) / len(union)) if union else 1.0
    nearby_fraction = float(nearby / len(candidate_dates)) if candidate_dates else 1.0
    return {
        "candidate_unique_dates": int(len(candidate_dates)),
        "comparator_unique_dates": int(len(comparator_dates)),
        "intersection_dates": int(len(intersection)),
        "decision_date_jaccard": jaccard,
        "candidate_dates_within_one_utc_day": int(nearby),
        "candidate_dates_within_one_utc_day_fraction": nearby_fraction,
    }


def _validate_intervals(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    checked = frame.copy()
    for column in ("entry_time", "exit_time"):
        checked[column] = _utc_series(checked[column], f"{label} {column}")
    checked["side"] = pd.to_numeric(checked["side"], errors="raise").astype(np.int8)
    if not bool(checked["side"].isin((-1, 1)).all()):
        raise RuntimeError(f"{label} has an invalid directional side")
    if not bool((checked["entry_time"] < checked["exit_time"]).all()):
        raise RuntimeError(f"{label} has an invalid interval")
    epoch_ns = checked["entry_time"].astype("int64")
    exit_ns = checked["exit_time"].astype("int64")
    step_ns = int(FIVE_MINUTES.value)
    if bool(((epoch_ns % step_ns) != 0).any()) or bool(
        ((exit_ns % step_ns) != 0).any()
    ):
        raise RuntimeError(f"{label} intervals are not 5m aligned")
    checked = checked.sort_values("entry_time", kind="mergesort").reset_index(drop=True)
    if len(checked) > 1 and bool(
        (
            checked["entry_time"].iloc[1:].array
            < checked["exit_time"].shift(1).iloc[1:].array
        ).any()
    ):
        raise RuntimeError(f"{label} directional intervals overlap")
    return checked


def signed_occupied_exposure_correlation(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
) -> dict[str, Any]:
    left = candidate.rename(
        columns={"entry_time_utc": "entry_time", "exit_time_utc": "exit_time"}
    )[["entry_time", "exit_time", "side"]]
    right = comparator[["entry_time", "exit_time", "side"]]
    left = _validate_intervals(left, "CDLTR")
    right = _validate_intervals(right, "comparator")
    if left.empty or right.empty:
        return {
            "defined": False,
            "failure_reason": "directional_clock_empty",
            "signed_occupied_exposure_pearson": None,
            "absolute_signed_occupied_exposure_pearson": None,
        }
    common_start = EVALUATION_START
    common_end = EVALUATION_END
    periods = int((common_end - common_start) // FIVE_MINUTES)
    if periods <= 1:
        return {
            "defined": False,
            "failure_reason": "common_5m_span_too_short",
            "signed_occupied_exposure_pearson": None,
            "absolute_signed_occupied_exposure_pearson": None,
        }

    def exposure(frame: pd.DataFrame) -> np.ndarray:
        values = np.zeros(periods, dtype=np.int8)
        for row in frame.itertuples(index=False):
            start = max(cast(pd.Timestamp, row.entry_time), common_start)
            end = min(cast(pd.Timestamp, row.exit_time), common_end)
            if start >= end:
                continue
            first = int((start - common_start) // FIVE_MINUTES)
            last = int((end - common_start) // FIVE_MINUTES)
            if bool(np.any(values[first:last] != 0)):
                raise RuntimeError("directional exposure is not flat/long/short")
            values[first:last] = int(row.side)
        return values

    left_values = exposure(left)
    right_values = exposure(right)
    left_variance = float(np.var(left_values.astype(np.float64)))
    right_variance = float(np.var(right_values.astype(np.float64)))
    if left_variance == 0.0 or right_variance == 0.0:
        return {
            "defined": False,
            "failure_reason": "zero_variance",
            "common_start_utc": common_start.isoformat(),
            "common_end_utc": common_end.isoformat(),
            "grid_rows_5m": periods,
            "candidate_nonflat_rows": int(np.count_nonzero(left_values)),
            "comparator_nonflat_rows": int(np.count_nonzero(right_values)),
            "candidate_variance": left_variance,
            "comparator_variance": right_variance,
            "signed_occupied_exposure_pearson": None,
            "absolute_signed_occupied_exposure_pearson": None,
        }
    correlation = float(
        np.corrcoef(left_values.astype(np.float64), right_values.astype(np.float64))[
            0, 1
        ]
    )
    if not math.isfinite(correlation):
        raise RuntimeError("directional occupied exposure correlation is non-finite")
    return {
        "defined": True,
        "failure_reason": None,
        "common_start_utc": common_start.isoformat(),
        "common_end_utc": common_end.isoformat(),
        "grid_rows_5m": periods,
        "candidate_nonflat_rows": int(np.count_nonzero(left_values)),
        "comparator_nonflat_rows": int(np.count_nonzero(right_values)),
        "candidate_variance": left_variance,
        "comparator_variance": right_variance,
        "signed_occupied_exposure_pearson": correlation,
        "absolute_signed_occupied_exposure_pearson": abs(correlation),
    }


def novelty_summary(primary: pd.DataFrame, comparators: pd.DataFrame) -> dict[str, Any]:
    _require_columns(comparators, prereg.COMPARATOR_HEADER, "comparator clock")
    frame = comparators.loc[:, list(prereg.COMPARATOR_HEADER)].copy()
    frame["decision_time"] = _utc_series(frame["decision_time"], "comparator")
    frame["entry_time"] = _utc_series(frame["entry_time"], "comparator")
    if bool(frame["decision_time"].ge(EVALUATION_END).any()):
        raise RuntimeError("sanitized comparator contains post-2023 decisions")
    if not bool((frame["decision_time"] <= frame["entry_time"]).all()):
        raise RuntimeError("sanitized comparator decision occurs after entry")
    frame = frame.loc[
        frame["decision_time"].ge(EVALUATION_START)
        & frame["decision_time"].lt(EVALUATION_END)
    ].reset_index(drop=True)
    expected = {
        *prereg.DIRECTIONAL_COMPARATORS,
        *prereg.TIMESTAMP_ONLY_COMPARATORS,
    }
    if set(frame["comparator"].unique()) != expected:
        raise RuntimeError("CDLTR comparator identity set drift")

    summaries: dict[str, Any] = {}
    for name in sorted(expected):
        reference = frame.loc[frame["comparator"].eq(name)].copy()
        if reference.empty:
            raise RuntimeError(f"CDLTR comparator is empty: {name}")
        capability = set(reference["capability"])
        if len(capability) != 1:
            raise RuntimeError(f"CDLTR comparator capability drift: {name}")
        overlap = decision_date_overlap(
            primary["decision_time_utc"], reference["decision_time"]
        )
        checks = {
            "decision_date_jaccard": overlap["decision_date_jaccard"]
            <= NOVELTY_LIMITS["decision_date_jaccard_maximum"],
            "candidate_dates_within_one_utc_day_fraction": overlap[
                "candidate_dates_within_one_utc_day_fraction"
            ]
            <= NOVELTY_LIMITS["cdltr_dates_within_one_utc_day_fraction_maximum"],
        }
        exposure: dict[str, Any] | None = None
        observed_capability = next(iter(capability))
        if observed_capability == "directional_interval":
            if name not in prereg.DIRECTIONAL_COMPARATORS:
                raise RuntimeError(f"unexpected directional comparator: {name}")
            exit_blank = reference["exit_time"].astype("string").fillna("").eq("")
            side_blank = reference["side"].astype("string").fillna("").eq("")
            if bool(exit_blank.any()) or bool(side_blank.any()):
                raise RuntimeError(f"directional comparator is incomplete: {name}")
            exposure = signed_occupied_exposure_correlation(primary, reference)
            correlation = exposure["absolute_signed_occupied_exposure_pearson"]
            checks["signed_occupied_exposure_defined"] = bool(exposure["defined"])
            checks["signed_occupied_exposure_pearson"] = bool(
                exposure["defined"]
                and correlation
                <= NOVELTY_LIMITS["signed_occupied_exposure_absolute_pearson_maximum"]
            )
        elif observed_capability == "timestamp_only":
            if name not in prereg.TIMESTAMP_ONLY_COMPARATORS:
                raise RuntimeError(f"unexpected timestamp-only comparator: {name}")
            exit_blank = reference["exit_time"].astype("string").fillna("").eq("")
            side_blank = reference["side"].astype("string").fillna("").eq("")
            source_clock = reference["source_clock"].astype("string").fillna("")
            direction_leak = source_clock.str.upper().str.contains(
                r"(?:^|[^A-Z])(?:LONG|SHORT)(?:[^A-Z]|$)", regex=True
            )
            if not bool(exit_blank.all()) or not bool(side_blank.all()):
                raise RuntimeError(
                    f"timestamp-only comparator invents direction or exit: {name}"
                )
            if bool(direction_leak.any()):
                raise RuntimeError(
                    f"timestamp-only comparator source clock leaks direction: {name}"
                )
        else:
            raise RuntimeError(f"unknown comparator capability: {name}")
        summaries[name] = {
            "passed": bool(all(checks.values())),
            "capability": observed_capability,
            "rows": int(len(reference)),
            "checks": checks,
            "decision_date_overlap": overlap,
            "signed_occupied_exposure": exposure,
        }
    return {
        "passed": bool(all(value["passed"] for value in summaries.values())),
        "limits": dict(NOVELTY_LIMITS),
        "comparators": summaries,
        "flcc_candidates_evaluated_independently": [
            name for name in sorted(summaries) if name.startswith("FLCC-1:")
        ],
    }


def _require_bound_file(path: str | Path, expected_sha: str, label: str) -> Path:
    target = _repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"{label} is missing or symlinked")
    if _sha256_path(target) != expected_sha:
        raise RuntimeError(f"{label} SHA drift")
    return target


def _load_preregistration() -> dict[str, Any]:
    path = _require_bound_file(
        PREREGISTRATION_ARTIFACT,
        PREREGISTRATION_ARTIFACT_SHA256,
        "CDLTR preregistration artifact",
    )
    _require_bound_file(
        prereg.PREREGISTRATION_SOURCE,
        PREREGISTRATION_SOURCE_SHA256,
        "CDLTR preregistration source",
    )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("CDLTR preregistration manifest hash drift")
    core = {key: value for key, value in artifact.items() if key != "manifest_hash"}
    if canonical_hash(core) != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("CDLTR preregistration canonical hash drift")
    if artifact.get("candidate") != POLICY_ID:
        raise RuntimeError("CDLTR preregistration candidate drift")
    if artifact.get("outcomes_opened") is not False:
        raise RuntimeError("CDLTR preregistration outcome boundary is open")
    support_gates = artifact.get("policy", {}).get("support_gates", {})
    for key, value in SUPPORT_LIMITS.items():
        if support_gates.get(key) != value:
            raise RuntimeError("CDLTR support limits drift from preregistration")
    if support_gates.get("all_controls_must_pass_calendar_and_containment") is not True:
        raise RuntimeError("CDLTR control gate drift from preregistration")
    novelty = artifact.get("policy", {}).get("novelty_gates", {})
    for key, value in NOVELTY_LIMITS.items():
        if novelty.get(key) != value:
            raise RuntimeError("CDLTR novelty limits drift from preregistration")
    return cast(dict[str, Any], artifact)


def _load_bound_csv(
    binding: Mapping[str, Any], columns: Sequence[str], label: str
) -> pd.DataFrame:
    path = _require_bound_file(
        cast(Path, binding["source"]), str(binding["source_sha256"]), label
    )
    frame = cast(pd.DataFrame, pd.read_csv(path, usecols=list(columns)))
    if tuple(frame.columns) != tuple(columns):
        raise RuntimeError(f"{label} loaded-column order drift")
    return frame


def load_bound_inputs() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Open only preregistered source values and sanitized comparator events."""
    _load_preregistration()
    rrp = _load_bound_csv(
        prereg.SOURCE_BINDINGS["rrp"], SOURCE_COLUMNS["rrp"], "RRP source"
    )
    cboe = _load_bound_csv(
        prereg.SOURCE_BINDINGS["cboe"], SOURCE_COLUMNS["cboe"], "Cboe source"
    )
    network = _load_bound_csv(
        prereg.SOURCE_BINDINGS["network"],
        SOURCE_COLUMNS["network"],
        "network source",
    )
    comparator_path = _require_bound_file(
        prereg.COMPARATOR_CLOCK,
        prereg.COMPARATOR_CLOCK_SHA256,
        "sanitized comparator clock",
    )
    comparators = cast(pd.DataFrame, pd.read_csv(comparator_path))
    if tuple(comparators.columns) != prereg.COMPARATOR_HEADER:
        raise RuntimeError("sanitized comparator clock header drift")
    if len(comparators) != 9_985:
        raise RuntimeError("sanitized comparator clock row-count drift")
    return rrp, cboe, network, comparators


def _vote_audit(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "valid_rows": int(frame["valid"].sum()),
        "invalid_rows": int((~frame["valid"]).sum()),
        "long_rows": int(frame["side"].eq(1).sum()),
        "short_rows": int(frame["side"].eq(-1).sum()),
        "neutral_rows": int(frame["side"].eq(0).sum()),
        "first_observation_date": str(frame["observation_date"].min()),
        "last_observation_date": str(frame["observation_date"].max()),
        "first_available_at_utc": cast(
            pd.Timestamp, frame["available_at"].min()
        ).isoformat(),
        "last_available_at_utc": cast(
            pd.Timestamp, frame["available_at"].max()
        ).isoformat(),
    }


def _source_vote_audit(source: pd.DataFrame, votes: pd.DataFrame) -> dict[str, Any]:
    return {
        "source_rows_read": int(len(source)),
        "source_rows_without_emitted_pre2024_vote": int(len(source) - len(votes)),
        **_vote_audit(votes),
    }


def evaluate_source_only(
    rrp_source: pd.DataFrame,
    cboe_source: pd.DataFrame,
    network_source: pd.DataFrame,
    comparators: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rrp_votes = derive_rrp_votes(rrp_source)
    cboe_votes = derive_cboe_votes(cboe_source)
    network_votes = derive_network_votes(network_source)
    clocks, relay_audit = build_clocks(rrp_votes, cboe_votes, network_votes)
    primary = clocks.loc[clocks["clock"].eq(PRIMARY_CLOCK)].reset_index(drop=True)
    support = support_summary(primary)
    controls = control_calendar_summary(clocks)
    novelty = novelty_summary(primary, comparators)
    passed = bool(support["passed"] and controls["passed"] and novelty["passed"])
    failed_stages = [
        name
        for name, stage_passed in (
            ("support", support["passed"]),
            ("control_support_calendar_and_containment", controls["passed"]),
            ("novelty", novelty["passed"]),
        )
        if not stage_passed
    ]
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": POLICY_ID,
        "source_only": True,
        "market_outcomes_opened": False,
        "performance_values_opened": False,
        "source_audit": {
            "rrp": _source_vote_audit(rrp_source, rrp_votes),
            "cboe": _source_vote_audit(cboe_source, cboe_votes),
            "network": _source_vote_audit(network_source, network_votes),
        },
        "relay_audit": relay_audit,
        "support": support,
        "control_support_calendar_and_containment": controls,
        "novelty": novelty,
        "verdict": {
            "passed": passed,
            "status": "PASS" if passed else "REJECT",
            "failed_stages": failed_stages,
            "strict_economic_train_authorized": passed,
            "repair_allowed_under_candidate_identity": False,
        },
        "outcome_boundary": {
            "source_value_rows_read": int(
                len(rrp_source) + len(cboe_source) + len(network_source)
            ),
            "source_feature_rows_derived": int(
                len(rrp_votes) + len(cboe_votes) + len(network_votes)
            ),
            "real_event_incidence_rows_derived": int(len(primary)),
            "comparator_event_rows_read": int(len(comparators)),
            "comparator_manifest_values_parsed": 0,
            "btc_market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "return_rows_loaded": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_rows_read": 0,
            "network_calls": 0,
            "subprocess_calls": 0,
        },
    }
    return report, clocks


def _clock_csv_bytes(clock: pd.DataFrame) -> bytes:
    formatted = clock.copy()
    for column in ("decision_time_utc", "entry_time_utc", "exit_time_utc"):
        formatted[column] = _utc_series(formatted[column], "clock output").dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    raw = (
        formatted.loc[:, list(CLOCK_COLUMNS)]
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    )
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(raw)
    return buffer.getvalue()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _protected_paths() -> set[Path]:
    paths = {
        _repository_path(EVALUATOR_SOURCE),
        _repository_path(PREREGISTRATION_ARTIFACT),
        _repository_path(prereg.PREREGISTRATION_SOURCE),
        _repository_path(prereg.COMPARATOR_CLOCK),
        _repository_path(prereg.COMPARATOR_MANIFEST),
    }
    for binding in prereg.SOURCE_BINDINGS.values():
        for key in ("source", "manifest", "builder"):
            paths.add(_repository_path(cast(Path, binding[key])))
    return paths


def run_evaluation(cfg: Config | None = None) -> dict[str, Any]:
    frozen_cfg = Config() if cfg is None else cfg
    output_report = _repository_path(frozen_cfg.output_report)
    output_clock = _repository_path(frozen_cfg.output_clock)
    if output_report == output_clock:
        raise ValueError("CDLTR support outputs must be distinct")
    if output_report in _protected_paths() or output_clock in _protected_paths():
        raise ValueError("CDLTR support output aliases a protected input")
    if output_report.exists() or output_clock.exists():
        raise FileExistsError("CDLTR support artifacts are immutable")

    rrp, cboe, network, comparators = load_bound_inputs()
    report, clock = evaluate_source_only(rrp, cboe, network, comparators)
    clock_payload = _clock_csv_bytes(clock)
    report.update(
        {
            "config": asdict(frozen_cfg),
            "evaluator_source": {
                "path": str(EVALUATOR_SOURCE),
                "sha256": sha256_file(EVALUATOR_SOURCE),
            },
            "preregistration": {
                "path": str(PREREGISTRATION_ARTIFACT),
                "sha256": PREREGISTRATION_ARTIFACT_SHA256,
                "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
            },
            "clock_artifact": {
                "path": frozen_cfg.output_clock,
                "sha256": hashlib.sha256(clock_payload).hexdigest(),
                "rows": int(len(clock)),
                "primary_rows": int(clock["clock"].eq(PRIMARY_CLOCK).sum()),
                "control_rows": int(clock["clock"].ne(PRIMARY_CLOCK).sum()),
                "columns": list(CLOCK_COLUMNS),
            },
        }
    )
    report["manifest_hash"] = canonical_hash(report)
    report_payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write(output_clock, clock_payload)
    try:
        _atomic_write(output_report, report_payload)
    except BaseException:
        output_clock.unlink(missing_ok=True)
        raise
    return report


def parse_args(argv: Sequence[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-report", default=str(DEFAULT_OUTPUT_REPORT))
    parser.add_argument("--output-clock", default=str(DEFAULT_OUTPUT_CLOCK))
    args = parser.parse_args(argv)
    return Config(output_report=args.output_report, output_clock=args.output_clock)


def main(argv: Sequence[str] | None = None) -> int:
    report = run_evaluation(parse_args(argv))
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "status": report["verdict"]["status"],
                "primary_rows": report["clock_artifact"]["primary_rows"],
                "manifest_hash": report["manifest_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
