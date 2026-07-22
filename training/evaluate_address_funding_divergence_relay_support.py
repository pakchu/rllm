"""Evaluate AFDR-864 source support and novelty without BTC outcomes.

The normal run may parse only the frozen Coin Metrics address fields, the four
allowlisted completed-funding signal fields, and hash-bound comparator clocks.
It must never load BTC prices, settlement-mark values, returns, or PnL.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from training import preregister_address_funding_divergence_relay as prereg


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "address_funding_divergence_relay_support_v1"
AS_OF_DATE = "2026-07-20"
PREREGISTRATION = Path(
    "results/address_funding_divergence_relay_preregistration_2026-07-20.json"
)
PREREGISTRATION_SHA256 = (
    "f76a82881aab7560015c15b181e73b17d09e173fdfbb7a3001b1234e8be66220"
)
EVALUATOR_SOURCE = Path(
    "training/evaluate_address_funding_divergence_relay_support.py"
)
PROTOCOL_DOCUMENT = Path(
    "docs/address-funding-divergence-relay-support-protocol-2026-07-20.md"
)
SOURCE_ACCESS_SEAL = Path(
    "results/address_funding_divergence_relay_source_access_seal_2026-07-20.json"
)
DEFAULT_CLOCKS = Path(
    "data/address_funding_divergence_relay_clocks_2021_2023.csv.gz"
)
DEFAULT_RESULT = Path(
    "results/address_funding_divergence_relay_support_2026-07-20.json"
)

FUNDING_STEP_MS = 8 * 60 * 60 * 1_000
FUNDING_PUBLICATION_DELAY = pd.Timedelta(
    minutes=prereg.FROZEN_CONFIG.funding_publication_delay_minutes
)
FIVE_MINUTES = pd.Timedelta(minutes=5)

SPLITS = {
    "train": (
        pd.Timestamp(prereg.FROZEN_CONFIG.train_start),
        pd.Timestamp(prereg.FROZEN_CONFIG.train_end_exclusive),
    ),
    "selection": (
        pd.Timestamp(prereg.FROZEN_CONFIG.selection_start),
        pd.Timestamp(prereg.FROZEN_CONFIG.selection_end_exclusive),
    ),
}

STATE_CONTROLS = ("primary", "balance_only", "activity_only", "funding_only")
DERIVED_CONTROLS = (
    "one_address_report_delay",
    "direction_flip",
    "deterministic_random_side",
)
CONTROL_ORDER = (*STATE_CONTROLS, *DERIVED_CONTROLS)

CLOCK_COLUMNS = (
    "candidate",
    "control",
    "split",
    "side",
    "observation_date",
    "decision_time",
    "entry_time",
    "exit_time",
)


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(repository_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"AFDR-864 JSON must be an object: {path}")
    return payload


def _verify_internal_hash(payload: Mapping[str, Any], label: str) -> None:
    unhashed = dict(payload)
    observed = unhashed.pop("manifest_hash", None)
    if observed != canonical_hash(unhashed):
        raise ValueError(f"AFDR-864 {label} internal hash mismatch")


def _timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp is pd.NaT:
        raise ValueError("AFDR-864 timestamp must not be NaT")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return cast(pd.Timestamp, timestamp)


def _explicit_utc_timestamp(value: Any, label: str) -> pd.Timestamp:
    """Parse a comparator timestamp only when its raw value carries a timezone."""
    timestamp = pd.Timestamp(value)
    if timestamp is pd.NaT:
        raise ValueError(f"AFDR-864 {label} timestamp is missing")
    if timestamp.tzinfo is None:
        raise ValueError(f"AFDR-864 {label} timestamp is timezone-less")
    return cast(pd.Timestamp, timestamp.tz_convert("UTC"))


def _explicit_utc_series(values: pd.Series, label: str) -> pd.Series:
    if bool(values.isna().any()):
        raise ValueError(f"AFDR-864 {label} timestamp is missing")
    parsed = values.map(lambda value: _explicit_utc_timestamp(value, label))
    return cast(pd.Series, pd.to_datetime(parsed, utc=True, errors="raise"))


def validate_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("AFDR-864 preregistration file hash mismatch")
    payload = _load_json(PREREGISTRATION)
    _verify_internal_hash(payload, "preregistration")
    expected = prereg.preregistration_payload()
    if payload != expected:
        raise ValueError("AFDR-864 preregistration contract drifted")
    if payload.get("candidate") != prereg.CANDIDATE:
        raise ValueError("AFDR-864 candidate identity drifted")
    boundary = payload.get("outcome_boundary", {})
    if boundary.get("outcomes_opened") is not False:
        raise ValueError("AFDR-864 preregistration opened outcomes")
    return payload


def access_seal_payload() -> dict[str, Any]:
    """Bind evaluator bytes before any numeric source/comparator row is read."""
    validate_preregistration()
    prereg.validate_contract()
    payload: dict[str, Any] = {
        "protocol_version": "address_funding_divergence_relay_source_access_v1",
        "candidate": prereg.CANDIDATE,
        "as_of_date": AS_OF_DATE,
        "bindings": {
            "preregistration": {
                "path": str(PREREGISTRATION),
                "sha256": PREREGISTRATION_SHA256,
            },
            "evaluator_source": {
                "path": str(EVALUATOR_SOURCE),
                "sha256": sha256_file(EVALUATOR_SOURCE),
            },
            "protocol_document": {
                "path": str(PROTOCOL_DOCUMENT),
                "sha256": sha256_file(PROTOCOL_DOCUMENT),
            },
            "address_source": {
                "path": str(prereg.ADDRESS_SOURCE),
                "sha256": prereg.ADDRESS_SOURCE_SHA256,
                "manifest": str(prereg.ADDRESS_MANIFEST),
                "manifest_sha256": prereg.ADDRESS_MANIFEST_SHA256,
            },
            "funding_source": {
                "path": str(prereg.FUNDING_SOURCE),
                "sha256": prereg.FUNDING_SOURCE_SHA256,
                "manifest": str(prereg.FUNDING_MANIFEST),
                "manifest_sha256": prereg.FUNDING_MANIFEST_SHA256,
            },
            "comparators": [dict(item) for item in prereg.COMPARATORS],
        },
        "feature_values_inspected_before_seal": False,
        "comparator_rows_inspected_before_seal": False,
        "market_outcomes_opened_before_seal": False,
        "row_counters": {
            "address_numeric_rows": 0,
            "funding_numeric_rows": 0,
            "comparator_rows": 0,
            "btc_market_rows": 0,
            "return_or_pnl_rows": 0,
            "post_2023_rows": 0,
        },
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def write_once_json(payload: dict[str, Any], output: str | Path) -> None:
    destination = repository_path(output)
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    if destination.exists():
        if destination.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(
                f"refusing to overwrite frozen AFDR-864 JSON: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def validate_access_seal() -> dict[str, Any]:
    payload = _load_json(SOURCE_ACCESS_SEAL)
    _verify_internal_hash(payload, "source-access seal")
    expected = access_seal_payload()
    if payload != expected:
        raise ValueError("AFDR-864 source-access seal drifted")
    if payload.get("feature_values_inspected_before_seal") is not False:
        raise ValueError("AFDR-864 source values were inspected before seal")
    if payload.get("market_outcomes_opened_before_seal") is not False:
        raise ValueError("AFDR-864 outcomes were opened before seal")
    return payload


def validate_header(path: str | Path, expected: tuple[str, ...]) -> None:
    columns = tuple(pd.read_csv(repository_path(path), nrows=0).columns)
    if columns != expected:
        raise ValueError(f"AFDR-864 source header drifted: {path}")


def load_address_source() -> pd.DataFrame:
    if sha256_file(prereg.ADDRESS_SOURCE) != prereg.ADDRESS_SOURCE_SHA256:
        raise ValueError("AFDR-864 address source hash mismatch")
    if sha256_file(prereg.ADDRESS_MANIFEST) != prereg.ADDRESS_MANIFEST_SHA256:
        raise ValueError("AFDR-864 address manifest hash mismatch")
    validate_header(prereg.ADDRESS_SOURCE, prereg.ADDRESS_COLUMNS)
    frame = pd.read_csv(
        repository_path(prereg.ADDRESS_SOURCE),
        dtype=str,
        usecols=cast(Any, list(prereg.ADDRESS_COLUMNS)),
    )
    if tuple(frame.columns) != prereg.ADDRESS_COLUMNS:
        raise ValueError("AFDR-864 address parser escaped its allowlist")
    if len(frame) != 1_826:
        raise ValueError("AFDR-864 address row count changed")
    frame["observation_date"] = pd.to_datetime(
        frame["observation_date"], utc=True, errors="raise"
    )
    frame["available_at"] = pd.to_datetime(
        frame["available_at"], utc=True, errors="raise"
    )
    expected_dates = pd.date_range(
        "2019-01-01", "2023-12-31", freq="1D", tz="UTC"
    )
    observed_dates = pd.DatetimeIndex(frame["observation_date"])
    if not observed_dates.equals(expected_dates):
        raise ValueError("AFDR-864 address daily grid changed")
    if (
        frame["available_at"]
        < frame["observation_date"] + pd.Timedelta(days=1)
    ).any():
        raise ValueError("AFDR-864 address availability precedes D+1")
    if frame["observation_date"].max() >= pd.Timestamp("2024-01-01", tz="UTC"):
        raise ValueError("AFDR-864 address source crossed 2024")
    for column in ("AdrBalCnt", "AdrActCnt"):
        text = frame[column]
        if not text.str.fullmatch(r"[1-9]\d*").all():
            raise ValueError(f"AFDR-864 address value is not positive: {column}")
        frame[column] = text.map(int).astype(np.int64)
    return frame


def load_funding_source() -> pd.DataFrame:
    if sha256_file(prereg.FUNDING_SOURCE) != prereg.FUNDING_SOURCE_SHA256:
        raise ValueError("AFDR-864 funding source hash mismatch")
    if sha256_file(prereg.FUNDING_MANIFEST) != prereg.FUNDING_MANIFEST_SHA256:
        raise ValueError("AFDR-864 funding manifest hash mismatch")
    validate_header(prereg.FUNDING_SOURCE, prereg.FUNDING_PHYSICAL_COLUMNS)
    frame = pd.read_csv(
        repository_path(prereg.FUNDING_SOURCE),
        dtype=str,
        usecols=cast(Any, list(prereg.FUNDING_SIGNAL_COLUMNS)),
    )
    if tuple(frame.columns) != prereg.FUNDING_SIGNAL_COLUMNS:
        raise ValueError("AFDR-864 funding parser escaped its signal allowlist")
    if len(frame) != 4_383:
        raise ValueError("AFDR-864 funding row count changed")
    if not frame["funding_time_ms"].str.fullmatch(r"\d+").all():
        raise ValueError("AFDR-864 funding millisecond timestamp is invalid")
    frame["funding_time_ms"] = frame["funding_time_ms"].astype(np.int64)
    frame["funding_time_utc"] = pd.to_datetime(
        frame["funding_time_utc"], utc=True, errors="raise"
    )
    from_ms = pd.to_datetime(frame["funding_time_ms"], unit="ms", utc=True)
    if not pd.DatetimeIndex(from_ms).equals(
        pd.DatetimeIndex(frame["funding_time_utc"])
    ):
        raise ValueError("AFDR-864 funding timestamp representations disagree")
    if not bool(_series(frame, "symbol").eq("BTCUSDT").all()):
        raise ValueError("AFDR-864 funding source contains another symbol")
    frame["funding_rate"] = pd.to_numeric(
        frame["funding_rate"], errors="raise"
    )
    if not np.isfinite(frame["funding_rate"].to_numpy(float)).all():
        raise ValueError("AFDR-864 funding rate is nonfinite")
    times_ms = frame["funding_time_ms"].to_numpy(np.int64)
    if len(np.unique(times_ms)) != len(times_ms) or np.any(np.diff(times_ms) <= 0):
        raise ValueError("AFDR-864 funding timestamps are duplicate or unordered")
    slots = times_ms // FUNDING_STEP_MS * FUNDING_STEP_MS
    offsets = times_ms - slots
    if (offsets < 0).any() or (
        offsets > prereg.FROZEN_CONFIG.maximum_funding_slot_offset_ms
    ).any():
        raise ValueError("AFDR-864 funding timestamp escaped its canonical slot")
    if np.any(np.diff(slots) != FUNDING_STEP_MS):
        raise ValueError("AFDR-864 funding canonical grid is incomplete")
    frame["canonical_slot_ms"] = slots
    frame["funding_available_at"] = (
        frame["funding_time_utc"] + FUNDING_PUBLICATION_DELAY
    )
    if frame["funding_time_utc"].min() < pd.Timestamp("2020-01-01", tz="UTC"):
        raise ValueError("AFDR-864 funding source starts before its frozen window")
    if frame["funding_time_utc"].max() >= pd.Timestamp("2024-01-01", tz="UTC"):
        raise ValueError("AFDR-864 funding source crossed 2024")
    return frame


def strict_prior_midrank(
    values: np.ndarray,
    observation_dates: np.ndarray,
    feature_available_at: np.ndarray,
    decision_available_at: np.ndarray,
    *,
    lookback_days: int,
    minimum: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Tie-midrank against causal, strictly prior, fixed-calendar references."""
    if lookback_days < 1 or minimum < 1:
        raise ValueError("AFDR-864 rank configuration is invalid")
    ranks = np.full(len(values), np.nan, dtype=float)
    counts = np.zeros(len(values), dtype=np.int64)
    lookback = np.timedelta64(lookback_days, "D")
    for index in range(len(values)):
        if not np.isfinite(values[index]):
            continue
        prior = np.arange(index)
        eligible = prior[
            np.isfinite(values[prior])
            & ~np.isnat(feature_available_at[prior])
            & (observation_dates[prior] >= observation_dates[index] - lookback)
            & (observation_dates[prior] < observation_dates[index])
            & (feature_available_at[prior] < decision_available_at[index])
        ]
        counts[index] = len(eligible)
        if len(eligible) < minimum:
            continue
        reference = values[eligible]
        less = int(np.count_nonzero(reference < values[index]))
        equal = int(np.count_nonzero(reference == values[index]))
        ranks[index] = (less + 0.5 * equal) / len(reference)
    return ranks, counts


def funding_pressure_at(
    decision_time: pd.Timestamp,
    funding: pd.DataFrame,
) -> tuple[float, Any, bool]:
    available = funding.loc[funding["funding_available_at"].le(decision_time)]
    required = prereg.FROZEN_CONFIG.required_funding_settlements
    if len(available) < required:
        return math.nan, pd.NaT, False
    tail = available.tail(required)
    slots = tail["canonical_slot_ms"].to_numpy(np.int64)
    if len(np.unique(slots)) != required or np.any(np.diff(slots) != FUNDING_STEP_MS):
        return math.nan, pd.NaT, False
    latest = _timestamp(tail["funding_available_at"].iloc[-1])
    age = decision_time - latest
    if age < pd.Timedelta(0) or age > pd.Timedelta(
        hours=prereg.FROZEN_CONFIG.maximum_latest_funding_age_hours
    ):
        return math.nan, pd.NaT, False
    rates = tail["funding_rate"].to_numpy(float)
    if not np.isfinite(rates).all():
        return math.nan, pd.NaT, False
    return float(rates.sum()), latest, True


def _state_onsets(
    states: np.ndarray,
    valid: np.ndarray,
    observation_dates: np.ndarray,
) -> np.ndarray:
    events = np.zeros(len(states), dtype=bool)
    one_day = np.timedelta64(1, "D")
    for index in range(1, len(states)):
        events[index] = bool(
            valid[index]
            and states[index] != 0
            and observation_dates[index] - observation_dates[index - 1] == one_day
            and valid[index - 1]
            and states[index - 1] == 0
        )
    return events


def build_features(
    address: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    minimum_prior: int | None = None,
) -> pd.DataFrame:
    cfg = prereg.FROZEN_CONFIG
    minimum = cfg.minimum_prior_observations if minimum_prior is None else minimum_prior
    frame = cast(
        pd.DataFrame,
        address.loc[:, ["observation_date", "available_at"]]
        .copy()
        .reset_index(drop=True),
    )
    observation = _series(frame, "observation_date").to_numpy(
        dtype="datetime64[ns]"
    )
    current_available = _series(frame, "available_at").to_numpy(
        dtype="datetime64[ns]"
    )
    balance = _series(address, "AdrBalCnt").to_numpy(float)
    activity = _series(address, "AdrActCnt").to_numpy(float)
    n = len(frame)
    balance_growth = np.full(n, np.nan)
    activity_growth = np.full(n, np.nan)
    address_feature_available = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    funding_pressure = np.full(n, np.nan)
    funding_feature_available = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    funding_valid = np.zeros(n, dtype=bool)

    index_by_date = {value: index for index, value in enumerate(observation)}
    lag_delta = np.timedelta64(cfg.address_change_days, "D")
    for index, date in enumerate(observation):
        lag_index = index_by_date.get(date - lag_delta)
        if lag_index is not None:
            balance_growth[index] = math.log(balance[index] / balance[lag_index])
            activity_growth[index] = math.log(activity[index] / activity[lag_index])
            address_feature_available[index] = max(
                current_available[index], current_available[lag_index]
            )
        pressure, latest, valid = funding_pressure_at(
            _timestamp(frame.loc[index, "available_at"]), funding
        )
        if valid:
            funding_pressure[index] = pressure
            funding_feature_available[index] = latest.to_datetime64()
            funding_valid[index] = True

    balance_rank, balance_reference_count = strict_prior_midrank(
        balance_growth,
        observation,
        address_feature_available,
        current_available,
        lookback_days=cfg.reference_lookback_days,
        minimum=minimum,
    )
    activity_rank, activity_reference_count = strict_prior_midrank(
        activity_growth,
        observation,
        address_feature_available,
        current_available,
        lookback_days=cfg.reference_lookback_days,
        minimum=minimum,
    )
    funding_rank, funding_reference_count = strict_prior_midrank(
        funding_pressure,
        observation,
        funding_feature_available,
        current_available,
        lookback_days=cfg.reference_lookback_days,
        minimum=minimum,
    )

    source_age = cast(
        pd.Series,
        _series(frame, "available_at") - _series(frame, "observation_date"),
    )
    source_age_days = source_age.dt.total_seconds().to_numpy(float) / 86_400.0
    current_fresh = (source_age_days >= 1.0) & (
        source_age_days <= cfg.maximum_address_lag_days
    )
    address_causal = ~np.isnat(address_feature_available) & (
        address_feature_available <= current_available
    )
    balance_valid = current_fresh & address_causal & np.isfinite(balance_rank)
    activity_valid = current_fresh & address_causal & np.isfinite(activity_rank)
    funding_control_valid = current_fresh & funding_valid & np.isfinite(funding_rank)
    primary_valid = balance_valid & activity_valid & funding_control_valid
    network_rank = (balance_rank + activity_rank) / 2.0

    states: dict[str, np.ndarray] = {}
    states["primary"] = np.where(
        primary_valid
        & (network_rank >= cfg.upper_rank)
        & (funding_rank <= cfg.lower_rank),
        1,
        np.where(
            primary_valid
            & (network_rank <= cfg.lower_rank)
            & (funding_rank >= cfg.upper_rank),
            -1,
            0,
        ),
    )
    states["balance_only"] = np.where(
        balance_valid & (balance_rank >= cfg.upper_rank),
        1,
        np.where(balance_valid & (balance_rank <= cfg.lower_rank), -1, 0),
    )
    states["activity_only"] = np.where(
        activity_valid & (activity_rank >= cfg.upper_rank),
        1,
        np.where(activity_valid & (activity_rank <= cfg.lower_rank), -1, 0),
    )
    states["funding_only"] = np.where(
        funding_control_valid & (funding_rank <= cfg.lower_rank),
        1,
        np.where(
            funding_control_valid & (funding_rank >= cfg.upper_rank), -1, 0
        ),
    )
    validity = {
        "primary": primary_valid,
        "balance_only": balance_valid,
        "activity_only": activity_valid,
        "funding_only": funding_control_valid,
    }

    frame["balance_growth_7d"] = balance_growth
    frame["activity_growth_7d"] = activity_growth
    frame["funding_pressure_72h"] = funding_pressure
    frame["address_feature_available_at"] = pd.to_datetime(
        address_feature_available, utc=True
    )
    frame["funding_feature_available_at"] = pd.to_datetime(
        funding_feature_available, utc=True
    )
    frame["balance_growth_rank"] = balance_rank
    frame["activity_growth_rank"] = activity_rank
    frame["funding_pressure_rank"] = funding_rank
    frame["network_rank"] = network_rank
    frame["balance_reference_count"] = balance_reference_count
    frame["activity_reference_count"] = activity_reference_count
    frame["funding_reference_count"] = funding_reference_count
    frame["source_age_days"] = source_age_days
    frame["primary_valid"] = primary_valid
    for control in STATE_CONTROLS:
        frame[f"{control}_state"] = states[control]
        frame[f"{control}_event"] = _state_onsets(
            states[control], validity[control], observation
        )
    return cast(pd.DataFrame, frame)


def _raw_candidates(features: pd.DataFrame, control: str) -> pd.DataFrame:
    candidates = features.loc[features[f"{control}_event"]].copy()
    candidates["candidate"] = prereg.CANDIDATE
    candidates["control"] = control
    candidates["split"] = ""
    candidates["side"] = candidates[f"{control}_state"].astype(int)
    candidates["decision_time"] = candidates["available_at"]
    candidates["entry_time"] = (
        _series(candidates, "decision_time").dt.ceil("5min")
        + pd.Timedelta(minutes=prereg.FROZEN_CONFIG.entry_delay_minutes)
    )
    candidates["exit_time"] = candidates["entry_time"] + pd.Timedelta(
        minutes=prereg.FROZEN_CONFIG.hold_bars
        * prereg.FROZEN_CONFIG.bar_minutes
    )
    return candidates[list(CLOCK_COLUMNS)]


def _schedule(candidates: pd.DataFrame, control: str) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for split, (start, end) in SPLITS.items():
        contained = candidates.loc[
            candidates["entry_time"].ge(start) & candidates["exit_time"].le(end)
        ].sort_values(["entry_time", "observation_date"], kind="mergesort")
        accepted: list[int] = []
        next_entry: pd.Timestamp | None = None
        for index, row in contained.iterrows():
            entry = _timestamp(row["entry_time"])
            if next_entry is not None and entry < next_entry:
                continue
            accepted.append(index)
            next_entry = _timestamp(row["exit_time"])
        subset = contained.loc[accepted].copy()
        subset["split"] = split
        subset["control"] = control
        selected.append(subset)
    if not selected:
        return pd.DataFrame(columns=pd.Index(CLOCK_COLUMNS))
    scheduled = cast(
        pd.DataFrame,
        pd.concat(selected, ignore_index=True).loc[:, list(CLOCK_COLUMNS)],
    )
    scheduled = cast(
        pd.DataFrame,
        scheduled.sort_values(
            by=cast(Any, ["entry_time", "control"]), kind="mergesort"
        ).reset_index(drop=True),
    )
    return cast(pd.DataFrame, scheduled)


def _random_side(entry_time: Any) -> int:
    entry = _timestamp(entry_time).strftime("%Y-%m-%dT%H:%M:%SZ")
    material = f"AFDR-864|20260720|{entry}".encode()
    return 1 if hashlib.sha256(material).digest()[0] < 128 else -1


def build_clocks(features: pd.DataFrame) -> pd.DataFrame:
    clocks: dict[str, pd.DataFrame] = {
        control: _schedule(_raw_candidates(features, control), control)
        for control in STATE_CONTROLS
    }
    primary = clocks["primary"]

    flipped = primary.copy()
    flipped["control"] = "direction_flip"
    flipped["side"] = -flipped["side"].astype(int)
    clocks["direction_flip"] = flipped

    randomized = primary.copy()
    randomized["control"] = "deterministic_random_side"
    randomized["side"] = randomized["entry_time"].map(_random_side)
    clocks["deterministic_random_side"] = randomized

    delayed_rows: list[dict[str, Any]] = []
    for row in primary.to_dict(orient="records"):
        _, origin_end = SPLITS[str(row["split"])]
        future = features.loc[
            features["observation_date"].gt(row["observation_date"])
            & features["primary_valid"]
            & features["available_at"].lt(origin_end)
        ].sort_values("observation_date")
        if future.empty:
            continue
        report = future.iloc[0]
        decision = _timestamp(report["available_at"])
        entry = decision.ceil("5min") + pd.Timedelta(
            minutes=prereg.FROZEN_CONFIG.entry_delay_minutes
        )
        delayed_rows.append(
            {
                "candidate": prereg.CANDIDATE,
                "control": "one_address_report_delay",
                "split": "",
                "side": int(row["side"]),
                "observation_date": report["observation_date"],
                "decision_time": decision,
                "entry_time": entry,
                "exit_time": entry
                + pd.Timedelta(
                    minutes=prereg.FROZEN_CONFIG.hold_bars
                    * prereg.FROZEN_CONFIG.bar_minutes
                ),
            }
        )
    delayed = pd.DataFrame(delayed_rows, columns=pd.Index(CLOCK_COLUMNS))
    clocks["one_address_report_delay"] = _schedule(
        delayed, "one_address_report_delay"
    )

    combined = pd.concat(
        [clocks[control] for control in CONTROL_ORDER], ignore_index=True
    )
    if combined.empty:
        return pd.DataFrame(columns=pd.Index(CLOCK_COLUMNS))
    combined["side"] = combined["side"].astype(int)
    return cast(
        pd.DataFrame,
        combined.sort_values(
            by=cast(Any, ["entry_time", "control", "side"]), kind="mergesort"
        ).reset_index(drop=True),
    )


def _rolling_30day_share(entry: pd.DatetimeIndex) -> float:
    if len(entry) == 0:
        return 1.0
    values = np.sort(entry.asi8)
    width = int(pd.Timedelta(days=30).value)
    maximum = 0
    for left, value in enumerate(values):
        right = int(np.searchsorted(values, value + width, side="left"))
        maximum = max(maximum, right - left)
    return maximum / len(values)


def split_support_summary(clock: pd.DataFrame) -> dict[str, Any]:
    entry = pd.DatetimeIndex(pd.to_datetime(clock["entry_time"], utc=True))
    if len(clock) == 0:
        return {
            "events": 0,
            "year_counts": {},
            "half_counts": {},
            "side_counts": {"long": 0, "short": 0},
            "maximum_month_share": 1.0,
            "maximum_weekday_share": 1.0,
            "maximum_rolling_30day_share": 1.0,
        }
    year_counts = pd.Series([time.year for time in entry]).value_counts().sort_index()
    half_labels = [f"{time.year}-H{1 if time.month <= 6 else 2}" for time in entry]
    half_counts = pd.Series(half_labels).value_counts().sort_index()
    month_counts = pd.Series(entry.strftime("%Y-%m")).value_counts()
    weekday_counts = pd.Series([time.weekday() for time in entry]).value_counts()
    return {
        "events": len(clock),
        "year_counts": {str(key): int(value) for key, value in year_counts.items()},
        "half_counts": {str(key): int(value) for key, value in half_counts.items()},
        "side_counts": {
            "long": int(clock["side"].eq(1).sum()),
            "short": int(clock["side"].eq(-1).sum()),
        },
        "maximum_month_share": float(month_counts.max() / len(clock)),
        "maximum_weekday_share": float(weekday_counts.max() / len(clock)),
        "maximum_rolling_30day_share": float(_rolling_30day_share(entry)),
    }


def support_checks(
    train: Mapping[str, Any], selection: Mapping[str, Any]
) -> tuple[dict[str, bool], list[str]]:
    gates = prereg.SUPPORT_GATES
    checks = {
        "minimum_train_events": train["events"] >= gates["minimum_train_events"],
        "minimum_2021_events": train["year_counts"].get("2021", 0)
        >= gates["minimum_events_each_train_year"],
        "minimum_2022_events": train["year_counts"].get("2022", 0)
        >= gates["minimum_events_each_train_year"],
        "minimum_selection_events": selection["events"]
        >= gates["minimum_selection_events"],
        "minimum_2023_h1_events": selection["half_counts"].get("2023-H1", 0)
        >= gates["minimum_events_each_selection_half"],
        "minimum_2023_h2_events": selection["half_counts"].get("2023-H2", 0)
        >= gates["minimum_events_each_selection_half"],
        "minimum_train_long_events": train["side_counts"]["long"]
        >= gates["minimum_train_events_each_side"],
        "minimum_train_short_events": train["side_counts"]["short"]
        >= gates["minimum_train_events_each_side"],
        "minimum_selection_long_events": selection["side_counts"]["long"]
        >= gates["minimum_selection_events_each_side"],
        "minimum_selection_short_events": selection["side_counts"]["short"]
        >= gates["minimum_selection_events_each_side"],
        "maximum_train_month_share": train["maximum_month_share"]
        <= gates["maximum_month_share_each_split"],
        "maximum_selection_month_share": selection["maximum_month_share"]
        <= gates["maximum_month_share_each_split"],
        "maximum_train_weekday_share": train["maximum_weekday_share"]
        <= gates["maximum_weekday_share_each_split"],
        "maximum_selection_weekday_share": selection["maximum_weekday_share"]
        <= gates["maximum_weekday_share_each_split"],
        "maximum_train_rolling_30day_share": train["maximum_rolling_30day_share"]
        <= gates["maximum_rolling_30day_share_each_split"],
        "maximum_selection_rolling_30day_share": selection[
            "maximum_rolling_30day_share"
        ]
        <= gates["maximum_rolling_30day_share_each_split"],
    }
    failures = [name for name, passed in checks.items() if not passed]
    return checks, failures


def _parse_comparator_csv(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    required = {
        str(spec["entry_column"]),
        *map(str, spec["filters"].keys()),
    }
    group_column = spec["group_column"]
    if group_column is not None:
        required.add(str(group_column))
    if spec["capability"] == "directional_interval":
        required.update({str(spec["side_column"]), str(spec["exit_column"])})
    path = repository_path(str(spec["path"]))
    header = set(pd.read_csv(path, nrows=0).columns)
    if not required.issubset(header):
        raise ValueError(f"AFDR-864 comparator schema drifted: {spec['candidate']}")
    frame = pd.read_csv(path, usecols=cast(Any, sorted(required)), dtype=str)
    for column, allowed in spec["filters"].items():
        frame = frame.loc[frame[column].isin(allowed)]
    if frame.empty:
        raise ValueError(
            f"AFDR-864 comparator is empty after frozen filters: {spec['candidate']}"
        )
    groups: list[tuple[str, pd.DataFrame]]
    if group_column is None:
        groups = [(str(spec["candidate"]), frame)]
    else:
        group_values = _series(frame, str(group_column))
        if bool(group_values.isna().any()) or bool(
            group_values.astype(str).str.strip().eq("").any()
        ):
            raise ValueError(
                f"AFDR-864 comparator group identity is missing: {spec['candidate']}"
            )
        groups = [
            (f"{spec['candidate']}:{name}", group.copy())
            for name, group in frame.groupby(group_column, sort=True, dropna=False)
        ]
    if not groups:
        raise ValueError(
            f"AFDR-864 comparator produced no independent member: {spec['candidate']}"
        )
    output: list[dict[str, Any]] = []
    for name, group in groups:
        entry_time = _explicit_utc_series(
            _series(group, str(spec["entry_column"])),
            f"comparator {name} entry",
        )
        normalized = pd.DataFrame(
            {
                "entry_time": entry_time,
            }
        )
        if spec["capability"] == "directional_interval":
            exit_time = _explicit_utc_series(
                _series(group, str(spec["exit_column"])),
                f"comparator {name} exit",
            )
            normalized["exit_time"] = exit_time
            side = cast(
                pd.Series,
                pd.to_numeric(group[str(spec["side_column"])], errors="raise"),
            )
            if side.isna().any() or not side.isin([-1, 1]).all():
                raise ValueError(f"AFDR-864 comparator side is invalid: {name}")
            normalized["side"] = side.astype(int)
        elif spec["capability"] != "timestamp_only":
            raise ValueError(
                f"AFDR-864 comparator capability is unknown: {spec['candidate']}"
            )
        output.append(
            {
                "member": name,
                "capability": str(spec["capability"]),
                "start": _timestamp(spec["comparison_start"]),
                "end": _timestamp(spec["comparison_end_exclusive"]),
                "events": normalized,
            }
        )
    return output


def _parse_microstructure_bundle(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = _load_json(str(spec["path"]))
    protocol = payload.get("protocol", {})
    if protocol.get("post_entry_outcomes_computed") is not False:
        raise ValueError("AFDR-864 microstructure comparator opened outcomes")
    if protocol.get("output_fields") != ["signal_date", "side"]:
        raise ValueError("AFDR-864 microstructure comparator fields drifted")
    comparators = payload.get("comparators")
    if not isinstance(comparators, dict) or not comparators:
        raise ValueError("AFDR-864 microstructure comparator bundle is empty")
    members: list[dict[str, Any]] = []
    base_start = pd.Timestamp("2021-01-01", tz="UTC")
    base_end = pd.Timestamp("2024-01-01", tz="UTC")
    for name in sorted(comparators):
        member = comparators[name]
        if not isinstance(member, dict):
            raise ValueError("AFDR-864 microstructure member is malformed")
        events = member.get("events")
        if not isinstance(events, list) or not events:
            raise ValueError("AFDR-864 microstructure events are malformed")
        if any(not isinstance(event, dict) or set(event) != {"signal_date", "side"} for event in events):
            raise ValueError("AFDR-864 microstructure event schema drifted")
        if any(event["side"] not in {-1, 1} for event in events):
            raise ValueError("AFDR-864 microstructure event side is invalid")
        entry_time = _explicit_utc_series(
            pd.Series([event["signal_date"] for event in events]),
            f"microstructure {name} entry",
        )
        frame = pd.DataFrame(
            {
                "entry_time": entry_time,
            }
        )
        coverage_start = _explicit_utc_timestamp(
            member["coverage_start_inclusive"],
            f"microstructure {name} coverage start",
        )
        coverage_end = _explicit_utc_timestamp(
            member["coverage_end_exclusive"],
            f"microstructure {name} coverage end",
        )
        start = cast(pd.Timestamp, max(base_start, coverage_start))
        end = cast(pd.Timestamp, min(base_end, coverage_end))
        if start >= end:
            raise ValueError("AFDR-864 microstructure coverage is empty")
        members.append(
            {
                "member": f"prior_microstructure:{name}",
                "capability": "timestamp_only",
                "start": start,
                "end": end,
                "events": frame,
            }
        )
    return members


def _comparator_registry_failure(
    spec: Mapping[str, Any], error: Exception
) -> dict[str, Any]:
    return {
        "member": f"{spec.get('candidate', 'unknown')}:__registry_failure__",
        "capability": str(spec.get("capability", "unknown")),
        "start": str(spec.get("comparison_start", "unknown")),
        "end": str(spec.get("comparison_end_exclusive", "unknown")),
        "events": pd.DataFrame(),
        "contract_failure": f"{type(error).__name__}: {error}",
    }


def load_comparator_members() -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for spec in prereg.COMPARATORS:
        try:
            if sha256_file(spec["path"]) != spec["sha256"]:
                raise ValueError(
                    f"AFDR-864 comparator hash mismatch: {spec['candidate']}"
                )
            if spec["format"] == "csv":
                parsed = _parse_comparator_csv(spec)
            elif spec["format"] == "json_comparator_event_bundle":
                parsed = _parse_microstructure_bundle(spec)
            else:
                raise ValueError("AFDR-864 comparator format is unknown")
            if not parsed:
                raise ValueError(
                    f"AFDR-864 comparator produced no member: {spec['candidate']}"
                )
            members.extend(parsed)
        except (KeyError, OSError, TypeError, ValueError) as error:
            members.append(_comparator_registry_failure(spec, error))
    if not members:
        error = ValueError("AFDR-864 comparator registry produced no members")
        return [
            {
                "member": "__registry_failure__",
                "capability": "unknown",
                "start": "unknown",
                "end": "unknown",
                "events": pd.DataFrame(),
                "contract_failure": f"ValueError: {error}",
            }
        ]
    identities = [str(member["member"]) for member in members]
    if len(set(identities)) != len(identities):
        error = ValueError("AFDR-864 comparator member identity is duplicated")
        members.append(
            {
                "member": "__registry_identity_failure__",
                "capability": "unknown",
                "start": "unknown",
                "end": "unknown",
                "events": pd.DataFrame(),
                "contract_failure": f"ValueError: {error}",
            }
        )
    return members


def _near_share(left: np.ndarray, right: np.ndarray, tolerance_ns: int) -> float:
    if len(left) == 0 or len(right) == 0:
        return 0.0
    right = np.sort(right)
    near = 0
    for value in np.sort(left):
        position = int(np.searchsorted(right, value))
        distances: list[int] = []
        if position < len(right):
            distances.append(abs(int(right[position] - value)))
        if position:
            distances.append(abs(int(right[position - 1] - value)))
        near += int(min(distances) <= tolerance_ns)
    return near / len(left)


def timestamp_novelty_metrics(
    candidate: pd.DatetimeIndex,
    comparator: pd.DatetimeIndex,
) -> dict[str, float | int]:
    candidate_ns = np.unique(candidate.asi8)
    comparator_ns = np.unique(comparator.asi8)
    intersection = np.intersect1d(candidate_ns, comparator_ns, assume_unique=True)
    union = np.union1d(candidate_ns, comparator_ns)
    tolerance = int(
        pd.Timedelta(hours=prereg.SUPPORT_GATES["novelty_containment_hours"]).value
    )
    return {
        "candidate_events": len(candidate_ns),
        "comparator_events": len(comparator_ns),
        "exact_jaccard": float(len(intersection) / len(union)) if len(union) else 1.0,
        "candidate_near_share": float(
            _near_share(candidate_ns, comparator_ns, tolerance)
        ),
        "comparator_near_share": float(
            _near_share(comparator_ns, candidate_ns, tolerance)
        ),
    }


def _exposure_vector(
    intervals: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> np.ndarray:
    step_ns = int(FIVE_MINUTES.value)
    start_ns = int(start.value)
    end_ns = int(end.value)
    if (end_ns - start_ns) % step_ns or end_ns <= start_ns:
        raise ValueError("AFDR-864 novelty scope is not a five-minute grid")
    exposure = np.zeros((end_ns - start_ns) // step_ns, dtype=np.int8)
    ordered = intervals.sort_values("entry_time", kind="mergesort")
    previous_exit: int | None = None
    for row in ordered.to_dict(orient="records"):
        entry = int(_timestamp(row["entry_time"]).value)
        exit_time = int(_timestamp(row["exit_time"]).value)
        side = int(row["side"])
        if side not in {-1, 1} or exit_time <= entry:
            raise ValueError("AFDR-864 directional comparator interval is invalid")
        if entry % step_ns or exit_time % step_ns:
            raise ValueError("AFDR-864 directional interval is not five-minute aligned")
        if previous_exit is not None and entry < previous_exit:
            raise ValueError("AFDR-864 directional comparator intervals overlap")
        previous_exit = exit_time
        clipped_entry = max(entry, start_ns)
        clipped_exit = min(exit_time, end_ns)
        if clipped_entry >= clipped_exit:
            continue
        left = (clipped_entry - start_ns) // step_ns
        right = (clipped_exit - start_ns) // step_ns
        exposure[left:right] = side
    return exposure


def signed_exposure_correlation(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    left = _exposure_vector(candidate, start, end).astype(float)
    right = _exposure_vector(comparator, start, end).astype(float)
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        raise ValueError("AFDR-864 novelty exposure has zero variance")
    correlation = float(np.corrcoef(left, right)[0, 1])
    if not math.isfinite(correlation):
        raise ValueError("AFDR-864 novelty correlation is nonfinite")
    return correlation


def evaluate_novelty(
    primary: pd.DataFrame,
    members: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    gates = prereg.SUPPORT_GATES
    results: list[dict[str, Any]] = []
    all_passed = True
    for member in members:
        start_label = str(member.get("start", "unknown"))
        end_label = str(member.get("end", "unknown"))
        timestamp_metrics: dict[str, float | int] | None = None
        correlation: float | None = None
        checks: dict[str, bool] = {}
        contract_failure = member.get("contract_failure")
        try:
            if contract_failure is not None:
                raise ValueError(str(contract_failure))
            start = _explicit_utc_timestamp(
                member["start"], f"comparator {member['member']} scope start"
            )
            end = _explicit_utc_timestamp(
                member["end"], f"comparator {member['member']} scope end"
            )
            if start >= end:
                raise ValueError("AFDR-864 comparator comparison scope is empty")
            start_label = start.isoformat()
            end_label = end.isoformat()
            candidate_rows = primary.loc[
                primary["entry_time"].ge(start) & primary["entry_time"].lt(end)
            ].copy()
            comparator_rows = member["events"].copy()
            comparator_entries = _explicit_utc_series(
                _series(comparator_rows, "entry_time"),
                f"comparator {member['member']} entry",
            )
            comparator_rows["entry_time"] = comparator_entries
            event_mask = comparator_entries.ge(start) & comparator_entries.lt(end)
            timestamp_metrics = timestamp_novelty_metrics(
                pd.DatetimeIndex(candidate_rows["entry_time"]),
                pd.DatetimeIndex(comparator_entries[event_mask]),
            )
            checks = {
                "comparator_contract_valid": True,
                "minimum_candidate_events": timestamp_metrics["candidate_events"]
                >= gates["minimum_common_candidate_events"],
                "minimum_comparator_events": timestamp_metrics["comparator_events"]
                >= gates["minimum_common_comparator_events"],
                "maximum_exact_jaccard": timestamp_metrics["exact_jaccard"]
                <= gates["maximum_exact_entry_jaccard"],
                "maximum_bidirectional_containment": max(
                    timestamp_metrics["candidate_near_share"],
                    timestamp_metrics["comparator_near_share"],
                )
                <= gates["maximum_bidirectional_novelty_containment"],
            }
            if member["capability"] == "directional_interval":
                comparator_exits = _explicit_utc_series(
                    _series(comparator_rows, "exit_time"),
                    f"comparator {member['member']} exit",
                )
                comparator_rows["exit_time"] = comparator_exits
                comparator_sides = cast(
                    pd.Series,
                    pd.to_numeric(comparator_rows["side"], errors="raise"),
                )
                if comparator_sides.isna().any() or not comparator_sides.isin(
                    [-1, 1]
                ).all():
                    raise ValueError("AFDR-864 comparator side is invalid")
                comparator_rows["side"] = comparator_sides.astype(int)
                overlap_mask = comparator_rows["exit_time"].gt(start) & comparator_rows[
                    "entry_time"
                ].lt(end)
                candidate_overlap = primary.loc[
                    primary["exit_time"].gt(start) & primary["entry_time"].lt(end),
                    ["entry_time", "exit_time", "side"],
                ]
                correlation = signed_exposure_correlation(
                    candidate_overlap,
                    comparator_rows.loc[
                        overlap_mask, ["entry_time", "exit_time", "side"]
                    ],
                    start,
                    end,
                )
                checks["maximum_absolute_signed_exposure_correlation"] = (
                    abs(correlation)
                    <= gates["maximum_absolute_signed_exposure_correlation"]
                )
            elif member["capability"] != "timestamp_only":
                raise ValueError("AFDR-864 comparator member capability is unknown")
        except (KeyError, TypeError, ValueError) as error:
            contract_failure = f"{type(error).__name__}: {error}"
            checks["comparator_contract_valid"] = False
        passed = contract_failure is None and bool(all(checks.values()))
        all_passed &= passed
        results.append(
            {
                "member": member["member"],
                "capability": member["capability"],
                "comparison_start": start_label,
                "comparison_end_exclusive": end_label,
                "timestamp_metrics": timestamp_metrics,
                "signed_exposure_correlation": correlation,
                "contract_failure": contract_failure,
                "checks": checks,
                "passed": passed,
            }
        )
    return results, all_passed


def _clock_bytes(clock: pd.DataFrame) -> bytes:
    safe = cast(pd.DataFrame, clock.loc[:, list(CLOCK_COLUMNS)].copy())
    for column in (
        "observation_date",
        "decision_time",
        "entry_time",
        "exit_time",
    ):
        parsed = pd.Series(
            pd.to_datetime(_series(safe, column), utc=True), index=safe.index
        )
        safe[column] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    safe = cast(
        pd.DataFrame,
        safe.sort_values(
            by=cast(Any, ["entry_time", "control", "side"]), kind="mergesort"
        ).reset_index(drop=True),
    )
    raw = safe.to_csv(index=False, lineterminator="\n").encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as stream:
        stream.write(raw)
    return buffer.getvalue()


def write_once_bytes(data: bytes, output: str | Path) -> None:
    destination = repository_path(output)
    if destination.exists():
        if destination.read_bytes() != data:
            raise FileExistsError(
                f"refusing to overwrite frozen AFDR-864 bytes: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_support(
    *,
    clock_output: str | Path = DEFAULT_CLOCKS,
    result_output: str | Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    seal = validate_access_seal()
    address = load_address_source()
    funding = load_funding_source()
    features = build_features(address, funding)
    clock = build_clocks(features)
    primary = clock.loc[clock["control"].eq("primary")].copy()
    train_summary = split_support_summary(primary.loc[primary["split"].eq("train")])
    selection_summary = split_support_summary(
        primary.loc[primary["split"].eq("selection")]
    )
    checks, support_failures = support_checks(train_summary, selection_summary)
    members = load_comparator_members()
    novelty, novelty_passed = evaluate_novelty(primary, members)

    clock_bytes = _clock_bytes(clock)
    write_once_bytes(clock_bytes, clock_output)
    clock_sha256 = hashlib.sha256(clock_bytes).hexdigest()

    controls = {
        control: {
            split: int(
                clock["control"].eq(control).mul(clock["split"].eq(split)).sum()
            )
            for split in SPLITS
        }
        for control in CONTROL_ORDER
    }
    passed = bool(not support_failures and novelty_passed)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": prereg.CANDIDATE,
        "as_of_date": AS_OF_DATE,
        "decision": "PASS_SOURCE_SUPPORT" if passed else "REJECT_NO_REPAIR",
        "source_access_seal": {
            "path": str(SOURCE_ACCESS_SEAL),
            "manifest_hash": seal["manifest_hash"],
            "sha256": sha256_file(SOURCE_ACCESS_SEAL),
        },
        "source_quality": {
            "address_rows": len(address),
            "address_first": address["observation_date"].min().isoformat(),
            "address_last": address["observation_date"].max().isoformat(),
            "funding_rows": len(funding),
            "funding_first": funding["funding_time_utc"].min().isoformat(),
            "funding_last": funding["funding_time_utc"].max().isoformat(),
            "signal_columns": {
                "address": list(prereg.ADDRESS_COLUMNS),
                "funding": list(prereg.FUNDING_SIGNAL_COLUMNS),
            },
        },
        "support": {
            "train": train_summary,
            "selection": selection_summary,
            "checks": checks,
            "failures": support_failures,
            "passed": not support_failures,
        },
        "novelty": {
            "members": novelty,
            "passed": novelty_passed,
        },
        "control_event_counts": controls,
        "clock_artifact": {
            "path": str(clock_output),
            "sha256": clock_sha256,
            "rows": len(clock),
            "columns": list(CLOCK_COLUMNS),
        },
        "outcome_boundary": {
            "outcomes_opened": False,
            "btc_market_rows_read": 0,
            "settlement_mark_values_read": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_rows_read": 0,
            "address_numeric_rows_read": len(address),
            "funding_signal_rows_read": len(funding),
            "comparator_members_read": len(members),
        },
        "next_action": (
            "freeze strict economic evaluator before train transport"
            if passed
            else "retire AFDR-864 without sign, threshold, hold, LLM, or RL repair"
        ),
    }
    payload["manifest_hash"] = canonical_hash(payload)
    write_once_json(payload, result_output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-access", action="store_true")
    parser.add_argument("--clock-output", type=Path, default=DEFAULT_CLOCKS)
    parser.add_argument("--result-output", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    if args.freeze_access:
        payload = access_seal_payload()
        write_once_json(payload, SOURCE_ACCESS_SEAL)
    else:
        payload = run_support(
            clock_output=args.clock_output,
            result_output=args.result_output,
        )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
