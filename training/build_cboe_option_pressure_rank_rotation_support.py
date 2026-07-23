"""Build outcome-blind OPRR-288 source-support and novelty evidence."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_cboe_option_pressure_rank_rotation as prereg


PROTOCOL_VERSION = "cboe_option_pressure_rank_rotation_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/build_cboe_option_pressure_rank_rotation_support.py"
)
TEST_PATH = Path(
    "tests/test_build_cboe_option_pressure_rank_rotation_support.py"
)
IMPLEMENTATION_CONTRACT = Path(
    "docs/oprr-source-support-implementation-contract-2026-07-24.md"
)
IMPLEMENTATION_CONTRACT_SHA256 = (
    "9e779fc2014fc9af97df6c1943d39e2ffdb726715697556726596e04ac37a9ff"
)
PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = (
    "76db2b61fe35599acaa9eb52d3406eac891bad3f0c95c17e6ccd212aea719d99"
)
PREREGISTRATION_MANIFEST_HASH = (
    "a8f45ab7339eb773650830ed73f541820082de6c9f86a7dfa40a69b430d2fb99"
)
DEFAULT_CLOCK_OUTPUT = Path(
    "data/cboe_option_pressure_rank_rotation_clocks_2020_2023.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/cboe_option_pressure_rank_rotation_support_2026-07-24.json"
)

BAR = pd.Timedelta(minutes=5)
DAY = pd.Timedelta(days=1)
SOURCE_END = pd.Timestamp("2024-01-01T00:00:00Z")
TRAIN_START = pd.Timestamp("2021-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2023-01-01T00:00:00Z")
SELECTION_START = TRAIN_END
SELECTION_END = SOURCE_END
NOVELTY_START = TRAIN_START
NOVELTY_END = SELECTION_END
NY_TZ = "America/New_York"

CLOCK_COLUMNS = (
    "control",
    "signal_id",
    "source_date",
    "signal_available_time",
    "entry_time",
    "exit_time",
    "side",
    "sponsor_surface",
    "prior_sponsor_position",
    "current_sponsor_position",
    "rotation_direction",
    "rotation_magnitude",
    "option_own_change_agreement",
    "term_confirmation",
    "tail_confirmation",
    "term_tail_order_relation",
    "term_tail_order_changed",
    "calendar_gap_bucket",
)
FORBIDDEN_CLOCK_TOKENS = (
    "open",
    "high",
    "low",
    "close",
    "price",
    "raw",
    "rank",
    "return",
    "basis",
    "future",
    "label",
    "funding",
    "pnl",
    "reward",
    "cagr",
    "mdd",
    "pressure",
)
INDEPENDENT_CONTROLS = (
    "primary",
    "rank_rotation_only",
    "option_own_confirmed",
    "non_option_pair_only",
    "term_sponsor_rotation",
    "tail_sponsor_rotation",
    "one_common_date_stale",
    "one_day_execution_delay",
)
SURFACES = ("term", "tail", "option")
POSITION_TOKENS = {0: "BELOW", 1: "MIDDLE", 2: "ABOVE"}
CLOSURES = frozenset(date.fromisoformat(value) for value in prereg.SESSION_CLOSURES)
_COMPARATOR_OPEN_TOKEN = object()
WINDOWS = {
    "train": (TRAIN_START, TRAIN_END),
    "selection": (SELECTION_START, SELECTION_END),
    "2021": (TRAIN_START, pd.Timestamp("2022-01-01T00:00:00Z")),
    "2022": (pd.Timestamp("2022-01-01T00:00:00Z"), TRAIN_END),
    "2023_h1": (SELECTION_START, pd.Timestamp("2023-07-01T00:00:00Z")),
    "2023_h2": (pd.Timestamp("2023-07-01T00:00:00Z"), SELECTION_END),
    "2023_q1": (SELECTION_START, pd.Timestamp("2023-04-01T00:00:00Z")),
    "2023_q2": (
        pd.Timestamp("2023-04-01T00:00:00Z"),
        pd.Timestamp("2023-07-01T00:00:00Z"),
    ),
    "2023_q3": (
        pd.Timestamp("2023-07-01T00:00:00Z"),
        pd.Timestamp("2023-10-01T00:00:00Z"),
    ),
    "2023_q4": (pd.Timestamp("2023-10-01T00:00:00Z"), SELECTION_END),
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
        raise RuntimeError("OPRR timestamp must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp.microsecond or timestamp.nanosecond:
        raise RuntimeError("OPRR timestamp must be whole-second")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_date(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise RuntimeError("OPRR source date must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp != timestamp.floor("D"):
        raise RuntimeError("OPRR source date must be UTC midnight")
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
        raise RuntimeError("OPRR source-support protocol is not committed")
    clean = _git_check("diff", "--quiet", "HEAD", "--", *paths)
    if clean.returncode:
        raise RuntimeError("OPRR source-support protocol differs from HEAD")


def validate_preregistration() -> Mapping[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("OPRR preregistration artifact hash drift")
    payload = json.loads(_path(PREREGISTRATION).read_text(encoding="utf-8"))
    prereg.validate_manifest(payload)
    if payload != prereg.build_manifest():
        raise RuntimeError("OPRR preregistration differs from frozen builder")
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("OPRR preregistration manifest hash drift")
    for field in (
        "outcomes_opened",
        "source_incidence_opened",
        "source_rows_decoded",
        "comparator_rows_decoded",
    ):
        if payload.get(field) is not False:
            raise RuntimeError(f"OPRR preregistration boundary opened: {field}")
    if tuple(payload["source_only_controls"]["ordered"]) != prereg.CONTROL_ORDER:
        raise RuntimeError("OPRR control order drift")
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
        (PREREGISTRATION, PREREGISTRATION_SHA256, "preregistration"),
    ]
    for path, expected in prereg.frozen_dependencies().items():
        bindings.append((path, expected, str(path)))
    audit: dict[str, dict[str, str]] = {}
    for path, expected, label in bindings:
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"OPRR frozen binding changed: {label}")
        audit[label] = {"path": str(path), "sha256": actual}
    expected_allowlists = {
        "term": list(prereg.TERM_ALLOWLIST),
        "tail": list(prereg.TAIL_ALLOWLIST),
        "option": list(prereg.OPTION_ALLOWLIST),
    }
    for name, allowlist in expected_allowlists.items():
        if payload["source_contracts"][name]["allowlist"] != allowlist:
            raise RuntimeError(f"OPRR {name} source allowlist drift")
    return audit


def validate_source_frame(
    frame: pd.DataFrame,
    *,
    allowlist: Sequence[str],
    source_name: str,
) -> pd.DataFrame:
    if list(frame.columns) != list(allowlist):
        raise RuntimeError(f"OPRR {source_name} loader did not preserve allowlist")
    validated = frame.copy()
    validated["observation_date"] = pd.to_datetime(
        validated["observation_date"], utc=True, errors="raise"
    )
    dates = validated["observation_date"]
    if dates.duplicated().any():
        raise RuntimeError(f"OPRR {source_name} dates duplicated")
    if not dates.is_monotonic_increasing:
        raise RuntimeError(f"OPRR {source_name} dates not increasing")
    if not dates.eq(dates.dt.floor("D")).all():
        raise RuntimeError(f"OPRR {source_name} dates not UTC midnight")
    if dates.ge(SOURCE_END).any():
        raise RuntimeError(f"OPRR {source_name} includes 2024-or-later data")
    numeric_columns = [column for column in allowlist if column != "observation_date"]
    for column in numeric_columns:
        validated[column] = pd.to_numeric(
            validated[column], errors="coerce"
        ).astype(float)
    values = validated[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not (values > 0.0).all():
        raise RuntimeError(f"OPRR {source_name} primitive invalid")
    return validated


def _load_panel(
    path: str | Path,
    *,
    expected_path: str,
    allowlist: Sequence[str],
    source_name: str,
) -> pd.DataFrame:
    if str(path) != expected_path:
        raise RuntimeError(f"OPRR {source_name} path differs from frozen source")
    frame = pd.read_csv(
        _path(path),
        usecols=list(allowlist),
        dtype={column: "string" for column in allowlist},
    )
    frame = frame.loc[:, list(allowlist)]
    return validate_source_frame(
        frame, allowlist=allowlist, source_name=source_name
    )


def load_term_source(path: str | Path = prereg.TERM_SOURCE) -> pd.DataFrame:
    return _load_panel(
        path,
        expected_path=prereg.TERM_SOURCE,
        allowlist=prereg.TERM_ALLOWLIST,
        source_name="term",
    )


def load_tail_source(path: str | Path = prereg.TAIL_SOURCE) -> pd.DataFrame:
    return _load_panel(
        path,
        expected_path=prereg.TAIL_SOURCE,
        allowlist=prereg.TAIL_ALLOWLIST,
        source_name="tail",
    )


def load_option_source(path: str | Path = prereg.OPTION_SOURCE) -> pd.DataFrame:
    return _load_panel(
        path,
        expected_path=prereg.OPTION_SOURCE,
        allowlist=prereg.OPTION_ALLOWLIST,
        source_name="option",
    )


def strict_prior_midranks(
    values: Iterable[float],
    *,
    lookback: int | None = None,
    minimum: int | None = None,
) -> np.ndarray:
    policy = prereg.Policy()
    width = policy.rank_lookback_observations if lookback is None else int(lookback)
    required = (
        policy.rank_minimum_prior_observations
        if minimum is None
        else int(minimum)
    )
    if width <= 0 or required <= 0 or required > width:
        raise RuntimeError("OPRR rank window invalid")
    history: list[float] = []
    result: list[float] = []
    for raw in values:
        value = float(raw)
        if not np.isfinite(value):
            result.append(np.nan)
            continue
        prior = np.asarray(history[-width:], dtype=np.float64)
        if len(prior) < required:
            result.append(np.nan)
        else:
            below = int(np.sum(prior < value))
            equal = int(np.sum(prior == value))
            result.append(float((below + 0.5 * equal) / len(prior)))
        history.append(value)
    return np.asarray(result, dtype=np.float64)


def _surface_frame(
    dates: pd.Series,
    first: np.ndarray,
    second: np.ndarray,
    *,
    name: str,
    vix: np.ndarray | None = None,
) -> pd.DataFrame:
    first_rank = strict_prior_midranks(first)
    second_rank = strict_prior_midranks(second)
    pressure = (first_rank + second_rank) / 2.0
    payload: dict[str, Any] = {
        "observation_date": dates.to_numpy(),
        f"{name}_pressure": pressure,
    }
    if vix is not None:
        payload[f"{name}_vix"] = vix
    return pd.DataFrame(payload)


def build_term_features(frame: pd.DataFrame) -> pd.DataFrame:
    vix9d = frame["VIX9D_close"].to_numpy(dtype=np.float64)
    vix = frame["VIX_close"].to_numpy(dtype=np.float64)
    vix3m = frame["VIX3M_close"].to_numpy(dtype=np.float64)
    return _surface_frame(
        frame["observation_date"],
        np.log(vix9d / vix),
        np.log(vix / vix3m),
        name="term",
        vix=vix,
    )


def build_tail_features(frame: pd.DataFrame) -> pd.DataFrame:
    skew = frame["SKEW_close"].to_numpy(dtype=np.float64)
    vvix = frame["VVIX_close"].to_numpy(dtype=np.float64)
    vix = frame["VIX_close"].to_numpy(dtype=np.float64)
    return _surface_frame(
        frame["observation_date"],
        np.log(skew / 100.0),
        np.log(vvix / vix),
        name="tail",
        vix=vix,
    )


def build_option_features(frame: pd.DataFrame) -> pd.DataFrame:
    total = frame["total_volume"].to_numpy(dtype=np.float64)
    index_call = frame["index_call_volume"].to_numpy(dtype=np.float64)
    index_put = frame["index_put_volume"].to_numpy(dtype=np.float64)
    index_volume = frame["index_volume"].to_numpy(dtype=np.float64)
    equity_call = frame["equity_call_volume"].to_numpy(dtype=np.float64)
    equity_put = frame["equity_put_volume"].to_numpy(dtype=np.float64)
    vix_call = frame["vix_call_volume"].to_numpy(dtype=np.float64)
    vix_put = frame["vix_put_volume"].to_numpy(dtype=np.float64)
    institutional_gap = np.log((index_put + 0.5) / (index_call + 0.5)) - np.log(
        (equity_put + 0.5) / (equity_call + 0.5)
    )
    vix_call_pressure = np.log((vix_call + 0.5) / (vix_put + 0.5))
    index_share = np.log((index_volume + 1.0) / (total + 1.0))
    deltas = [
        np.concatenate(([np.nan], np.diff(values)))
        for values in (institutional_gap, vix_call_pressure, index_share)
    ]
    ranks = [strict_prior_midranks(values) for values in deltas]
    return pd.DataFrame(
        {
            "observation_date": frame["observation_date"].to_numpy(),
            "option_pressure": np.mean(np.vstack(ranks), axis=0),
        }
    )


def ordinal_position(term: float, tail: float, option: float, sponsor: str) -> int | None:
    values = {"term": float(term), "tail": float(tail), "option": float(option)}
    if sponsor not in values:
        raise RuntimeError(f"OPRR sponsor invalid: {sponsor}")
    array = np.asarray(list(values.values()), dtype=np.float64)
    if not np.isfinite(array).all():
        return None
    comparisons = [
        value for name, value in values.items() if name != sponsor
    ]
    chosen = values[sponsor]
    if any(chosen == value for value in comparisons):
        return None
    if sponsor == "option" and comparisons[0] == comparisons[1]:
        return None
    return int(sum(value < chosen for name, value in values.items() if name != sponsor))


def _strict_sign(value: float) -> int:
    if not np.isfinite(value):
        return 0
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _gap_bucket(previous: pd.Timestamp, current: pd.Timestamp) -> str:
    days = (current.date() - previous.date()).days
    if days <= 0:
        raise RuntimeError("OPRR common dates not increasing")
    if days == 1:
        return "1D"
    if days <= 3:
        return "2_3D"
    return "4P_D"


def _term_tail_order(term: float, tail: float) -> str:
    if term < tail:
        return "TERM_BELOW_TAIL"
    if term > tail:
        return "TERM_ABOVE_TAIL"
    return "TERM_TAIL_TIE"


def build_common_states(
    term: pd.DataFrame,
    tail: pd.DataFrame,
    option: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    term_features = build_term_features(term)
    tail_features = build_tail_features(tail)
    option_features = build_option_features(option)
    common = term_features.merge(
        tail_features, on="observation_date", how="inner", validate="one_to_one"
    ).merge(
        option_features, on="observation_date", how="inner", validate="one_to_one"
    )
    if common.empty:
        raise RuntimeError("OPRR exact common-date panel empty")
    if not np.equal(
        common["term_vix"].to_numpy(dtype=np.float64),
        common["tail_vix"].to_numpy(dtype=np.float64),
    ).all():
        raise RuntimeError("OPRR term/tail VIX cross-panel mismatch")
    complete = common.dropna(
        subset=["term_pressure", "tail_pressure", "option_pressure"]
    ).copy()
    complete = complete.sort_values(
        "observation_date", kind="mergesort"
    ).reset_index(drop=True)
    records: list[dict[str, Any]] = []
    for row in complete.itertuples(index=False):
        pressures = {
            "term": float(row.term_pressure),
            "tail": float(row.tail_pressure),
            "option": float(row.option_pressure),
        }
        record: dict[str, Any] = {
            "observation_date": pd.Timestamp(row.observation_date),
            **{f"{name}_pressure": value for name, value in pressures.items()},
            "term_tail_order": _term_tail_order(
                pressures["term"], pressures["tail"]
            ),
        }
        for sponsor in SURFACES:
            record[f"{sponsor}_position"] = ordinal_position(
                pressures["term"], pressures["tail"], pressures["option"], sponsor
            )
        records.append(record)
    states = pd.DataFrame(records)
    if not states.empty:
        for column in (f"{surface}_position" for surface in SURFACES):
            states[column] = pd.array(states[column], dtype="Int64")
    funnel = {
        "term_rows": len(term),
        "tail_rows": len(tail),
        "option_rows": len(option),
        "term_rank_complete_rows": int(term_features["term_pressure"].notna().sum()),
        "tail_rank_complete_rows": int(tail_features["tail_pressure"].notna().sum()),
        "option_rank_complete_rows": int(option_features["option_pressure"].notna().sum()),
        "exact_common_dates": len(common),
        "rank_complete_common_dates": len(states),
        "pairwise_distinct_common_dates": int(
            states["option_position"].notna().sum() if not states.empty else 0
        ),
        "adjacent_rank_complete_transitions": max(0, len(states) - 1),
    }
    return states, funnel


def build_transitions(states: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for index in range(1, len(states)):
        previous = states.iloc[index - 1]
        current = states.iloc[index]
        previous_date = pd.Timestamp(previous["observation_date"])
        current_date = pd.Timestamp(current["observation_date"])
        record: dict[str, Any] = {
            "previous_date": previous_date,
            "source_date": current_date,
            "calendar_gap_bucket": _gap_bucket(previous_date, current_date),
            "term_tail_order_relation": str(current["term_tail_order"]),
            "term_tail_order_changed": (
                "CHANGED"
                if str(previous["term_tail_order"]) != str(current["term_tail_order"])
                else "UNCHANGED"
            ),
        }
        for surface in SURFACES:
            delta = float(current[f"{surface}_pressure"]) - float(
                previous[f"{surface}_pressure"]
            )
            record[f"{surface}_delta_sign"] = _strict_sign(delta)
            prior_position = previous[f"{surface}_position"]
            current_position = current[f"{surface}_position"]
            if pd.isna(prior_position) or pd.isna(current_position):
                record[f"prior_{surface}_position"] = None
                record[f"current_{surface}_position"] = None
                record[f"{surface}_rotation"] = None
            else:
                prior_integer = int(prior_position)
                current_integer = int(current_position)
                record[f"prior_{surface}_position"] = prior_integer
                record[f"current_{surface}_position"] = current_integer
                record[f"{surface}_rotation"] = current_integer - prior_integer
        records.append(record)
    return pd.DataFrame(records)


def _next_session_date(source_date: Any) -> date:
    timestamp = pd.Timestamp(source_date)
    if timestamp.tzinfo is None:
        raise RuntimeError("OPRR source date must be timezone-aware")
    current = timestamp.tz_convert("UTC").date() + timedelta(days=1)
    while current.weekday() >= 5 or current in CLOSURES:
        current += timedelta(days=1)
    if not date(2020, 1, 1) <= current < date(2025, 1, 1):
        raise RuntimeError("OPRR prospective session outside frozen calendar")
    return current


def _ny_time(session_date: date, hour: int, minute: int) -> pd.Timestamp:
    naive = pd.Timestamp(session_date) + pd.Timedelta(hours=hour, minutes=minute)
    return naive.tz_localize(NY_TZ).tz_convert("UTC")


def _position_token(value: Any) -> str:
    if value is None or pd.isna(value):
        return "UNAVAILABLE"
    integer = int(value)
    if integer not in POSITION_TOKENS:
        raise RuntimeError("OPRR ordinal position invalid")
    return POSITION_TOKENS[integer]


def _transition_tokens(
    transition: Mapping[str, Any] | Any,
    sponsor: str,
) -> dict[str, str]:
    getter = (
        transition.get
        if isinstance(transition, Mapping)
        else lambda key: getattr(transition, key)
    )
    if sponsor not in SURFACES:
        raise RuntimeError("OPRR token sponsor invalid")
    rotation = getter(f"{sponsor}_rotation")
    rotation_integer = None if rotation is None or pd.isna(rotation) else int(rotation)
    direction = (
        "UP" if rotation_integer is not None and rotation_integer > 0
        else "DOWN" if rotation_integer is not None and rotation_integer < 0
        else "NONE" if rotation_integer == 0
        else "UNAVAILABLE"
    )
    magnitude = (
        "ONE_STEP" if rotation_integer is not None and abs(rotation_integer) == 1
        else "TWO_STEP" if rotation_integer is not None and abs(rotation_integer) == 2
        else "NONE" if rotation_integer == 0
        else "UNAVAILABLE"
    )
    rotation_sign = 0 if rotation_integer is None else _strict_sign(rotation_integer)

    def agreement(surface: str) -> str:
        delta_sign = int(getter(f"{surface}_delta_sign"))
        if rotation_sign == 0 or delta_sign == 0:
            return "ZERO_OR_UNAVAILABLE"
        return "AGREE" if delta_sign == rotation_sign else "DISAGREE"

    return {
        "sponsor_surface": sponsor.upper(),
        "prior_sponsor_position": _position_token(
            getter(f"prior_{sponsor}_position")
        ),
        "current_sponsor_position": _position_token(
            getter(f"current_{sponsor}_position")
        ),
        "rotation_direction": direction,
        "rotation_magnitude": magnitude,
        "option_own_change_agreement": agreement("option"),
        "term_confirmation": agreement("term"),
        "tail_confirmation": agreement("tail"),
        "term_tail_order_relation": str(getter("term_tail_order_relation")),
        "term_tail_order_changed": str(getter("term_tail_order_changed")),
        "calendar_gap_bucket": str(getter("calendar_gap_bucket")),
    }


def _non_option_tokens(transition: Mapping[str, Any] | Any) -> dict[str, str]:
    getter = (
        transition.get
        if isinstance(transition, Mapping)
        else lambda key: getattr(transition, key)
    )
    return {
        "sponsor_surface": "NONE",
        "prior_sponsor_position": "NOT_USED",
        "current_sponsor_position": "NOT_USED",
        "rotation_direction": "NOT_USED",
        "rotation_magnitude": "NOT_USED",
        "option_own_change_agreement": "NOT_REQUIRED",
        "term_confirmation": "AGREE",
        "tail_confirmation": "AGREE",
        "term_tail_order_relation": str(getter("term_tail_order_relation")),
        "term_tail_order_changed": str(getter("term_tail_order_changed")),
        "calendar_gap_bucket": str(getter("calendar_gap_bucket")),
    }


def signal_id(row: Mapping[str, Any]) -> str:
    identity: dict[str, Any] = {
        "policy_id": prereg.Policy().policy_id,
        "policy": asdict(prereg.Policy()),
        "preregistration_manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        "source_hashes": {
            "term": prereg.TERM_SOURCE_SHA256,
            "tail": prereg.TAIL_SOURCE_SHA256,
            "option": prereg.OPTION_SOURCE_SHA256,
        },
    }
    for column in CLOCK_COLUMNS:
        if column == "signal_id":
            continue
        value = row[column]
        if column == "source_date":
            value = _format_date(value)
        elif column.endswith("_time"):
            value = _format_time(value)
        identity[column] = value
    return canonical_hash(identity)


def _candidate_row(
    control: str,
    transition: Mapping[str, Any] | Any,
    direction: int,
    *,
    sponsor: str | None,
    token_transition: Mapping[str, Any] | Any | None = None,
    source_transition: Mapping[str, Any] | Any | None = None,
    entry_delay_bars: int = 0,
) -> dict[str, Any]:
    if direction not in (-1, 1):
        raise RuntimeError("OPRR candidate direction invalid")
    source_getter = (
        (source_transition or transition).get
        if isinstance(source_transition or transition, Mapping)
        else lambda key: getattr(source_transition or transition, key)
    )
    source_date = pd.Timestamp(source_getter("source_date"))
    session = _next_session_date(source_date)
    policy = prereg.Policy()
    signal_available = _ny_time(
        session,
        policy.entry_local_hour,
        policy.entry_local_minute - policy.signal_buffer_minutes,
    )
    entry = _ny_time(
        session, policy.entry_local_hour, policy.entry_local_minute
    ) + entry_delay_bars * BAR
    tokens_from = transition if token_transition is None else token_transition
    tokens = (
        _transition_tokens(tokens_from, sponsor)
        if sponsor is not None
        else _non_option_tokens(tokens_from)
    )
    row: dict[str, Any] = {
        "control": control,
        "source_date": source_date,
        "signal_available_time": signal_available,
        "entry_time": entry,
        "exit_time": entry + policy.hold_bars * BAR,
        "side": "SHORT" if direction > 0 else "LONG",
        **tokens,
    }
    row["signal_id"] = signal_id(row)
    return {column: row[column] for column in CLOCK_COLUMNS}


def _sponsor_direction(transition: Mapping[str, Any] | Any, sponsor: str) -> int:
    getter = (
        transition.get
        if isinstance(transition, Mapping)
        else lambda key: getattr(transition, key)
    )
    rotation = getter(f"{sponsor}_rotation")
    if rotation is None or pd.isna(rotation) or int(rotation) == 0:
        return 0
    direction = _strict_sign(int(rotation))
    return direction if all(
        int(getter(f"{surface}_delta_sign")) == direction for surface in SURFACES
    ) else 0


def raw_candidates(transitions: pd.DataFrame, control: str) -> pd.DataFrame:
    if control not in INDEPENDENT_CONTROLS:
        raise RuntimeError(f"OPRR independent control unsupported: {control}")
    rows: list[dict[str, Any]] = []
    transition_records = list(transitions.itertuples(index=False))
    for index, transition in enumerate(transition_records):
        direction = 0
        sponsor: str | None = "option"
        token_transition: Any | None = None
        source_transition: Any | None = None
        delay = 0
        if control in ("primary", "one_day_execution_delay"):
            direction = _sponsor_direction(transition, "option")
            delay = (
                prereg.Policy().hold_bars
                if control == "one_day_execution_delay"
                else 0
            )
        elif control == "rank_rotation_only":
            rotation = transition.option_rotation
            direction = 0 if pd.isna(rotation) else _strict_sign(int(rotation))
        elif control == "option_own_confirmed":
            rotation = transition.option_rotation
            direction = 0 if pd.isna(rotation) else _strict_sign(int(rotation))
            if direction == 0 or transition.option_delta_sign != direction:
                direction = 0
        elif control == "non_option_pair_only":
            sponsor = None
            if (
                transition.term_delta_sign != 0
                and transition.term_delta_sign == transition.tail_delta_sign
            ):
                direction = int(transition.term_delta_sign)
        elif control == "term_sponsor_rotation":
            sponsor = "term"
            direction = _sponsor_direction(transition, sponsor)
        elif control == "tail_sponsor_rotation":
            sponsor = "tail"
            direction = _sponsor_direction(transition, sponsor)
        elif control == "one_common_date_stale":
            if index == 0:
                continue
            stale = transition_records[index - 1]
            direction = _sponsor_direction(stale, "option")
            token_transition = stale
            source_transition = transition
        if direction == 0:
            continue
        rows.append(
            _candidate_row(
                control,
                transition,
                direction,
                sponsor=sponsor,
                token_transition=token_transition,
                source_transition=source_transition,
                entry_delay_bars=delay,
            )
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def reserve_nonoverlap(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    selected: list[dict[str, Any]] = []
    previous_exit: pd.Timestamp | None = None
    ordered = rows.sort_values(["entry_time", "signal_id"], kind="mergesort")
    for row in ordered.itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        exit_time = pd.Timestamp(row.exit_time)
        if previous_exit is not None and entry < previous_exit:
            continue
        selected.append({column: getattr(row, column) for column in CLOCK_COLUMNS})
        previous_exit = exit_time
    return pd.DataFrame(selected, columns=CLOCK_COLUMNS)


def _replace_control_and_side(row: Any, control: str, side: str) -> dict[str, Any]:
    candidate = {column: getattr(row, column) for column in CLOCK_COLUMNS}
    candidate["control"] = control
    candidate["side"] = side
    candidate["signal_id"] = signal_id(candidate)
    return candidate


def _same_clock_variant(
    primary: pd.DataFrame,
    control: str,
    sides: Iterable[str],
) -> pd.DataFrame:
    rows = [
        _replace_control_and_side(row, control, side)
        for row, side in zip(primary.itertuples(index=False), sides, strict=True)
    ]
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def build_controls(
    transitions: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    raw = {name: raw_candidates(transitions, name) for name in INDEPENDENT_CONTROLS}
    controls = {name: reserve_nonoverlap(raw[name]) for name in INDEPENDENT_CONTROLS}
    primary = controls["primary"]
    controls["exact_direction_flip"] = _same_clock_variant(
        primary,
        "exact_direction_flip",
        ["SHORT" if side == "LONG" else "LONG" for side in primary["side"]],
    )
    random_sides: list[str] = []
    for entry in primary["entry_time"]:
        message = b"OPRR-288|" + _format_time(entry).encode("ascii")
        digest = hashlib.sha256(message).digest()
        random_sides.append("LONG" if digest[0] < 128 else "SHORT")
    controls["deterministic_random_side"] = _same_clock_variant(
        primary, "deterministic_random_side", random_sides
    )
    raw["exact_direction_flip"] = raw["primary"].copy()
    raw["deterministic_random_side"] = raw["primary"].copy()
    if set(controls) != set(prereg.CONTROL_ORDER):
        raise RuntimeError("OPRR constructed control set drift")
    if set(raw) != set(prereg.CONTROL_ORDER):
        raise RuntimeError("OPRR raw control set drift")
    return (
        {name: controls[name] for name in prereg.CONTROL_ORDER},
        {name: raw[name] for name in prereg.CONTROL_ORDER},
    )


def _contained(rows: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    return rows.loc[
        rows["entry_time"].ge(start) & rows["exit_time"].le(end)
    ].copy()


def _entry_window(
    rows: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    return rows.loc[
        rows["entry_time"].ge(start) & rows["entry_time"].lt(end)
    ].copy()


def _intersecting_window(
    rows: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    return rows.loc[
        rows["exit_time"].gt(start) & rows["entry_time"].lt(end)
    ].copy()


def _window(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    start, end = WINDOWS[name]
    return _contained(rows, start, end).sort_values(
        ["entry_time", "signal_id"], kind="mergesort"
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
    ordered = rows.sort_values(["entry_time", "signal_id"], kind="mergesort")
    total = len(ordered)
    if not total:
        return {
            "events": 0,
            "long": 0,
            "short": 0,
            "long_share": None,
            "short_share": None,
            "active_months": 0,
            "maximum_month_share": None,
            "maximum_quarter_share": None,
            "maximum_gap_days": None,
            "maximum_same_side_run": 0,
        }
    side = ordered["side"].value_counts().to_dict()
    local = ordered["entry_time"].dt.tz_convert(NY_TZ).dt.tz_localize(None)
    month_counts = local.dt.to_period("M").astype(str).value_counts()
    quarter_counts = local.dt.to_period("Q").astype(str).value_counts()
    local_dates = [value.date() for value in local]
    gaps = [
        (right - left).days for left, right in zip(local_dates, local_dates[1:])
    ]
    return {
        "events": total,
        "long": int(side.get("LONG", 0)),
        "short": int(side.get("SHORT", 0)),
        "long_share": float(side.get("LONG", 0) / total),
        "short_share": float(side.get("SHORT", 0) / total),
        "active_months": int(len(month_counts)),
        "maximum_month_share": float(month_counts.max() / total),
        "maximum_quarter_share": float(quarter_counts.max() / total),
        "maximum_gap_days": float(max(gaps)) if gaps else None,
        "maximum_same_side_run": _longest_run(ordered["side"]),
    }


def same_side_reproduction(
    primary: pd.DataFrame,
    control: pd.DataFrame,
    split: str,
) -> float | None:
    left = _window(primary, split)
    if left.empty:
        return None
    right = _window(control, split)
    lookup = {
        (pd.Timestamp(row.entry_time), str(row.side))
        for row in right.itertuples(index=False)
    }
    matched = sum(
        (pd.Timestamp(row.entry_time), str(row.side)) in lookup
        for row in left.itertuples(index=False)
    )
    return float(matched / len(left))


def exact_entry_jaccard(
    left: pd.DataFrame, right: pd.DataFrame
) -> float | None:
    first = set(pd.Timestamp(value) for value in left["entry_time"])
    second = set(pd.Timestamp(value) for value in right["entry_time"])
    if not first or not second:
        return None
    return float(len(first & second) / len(first | second))


def _timing_integrity(rows: pd.DataFrame, control: str) -> bool:
    if rows.empty:
        return True
    delay = (
        prereg.Policy().hold_bars * BAR
        if control == "one_day_execution_delay"
        else pd.Timedelta(0)
    )
    allowed_sponsors = {"OPTION", "TERM", "TAIL", "NONE"}
    return bool(
        rows["entry_time"].eq(rows["signal_available_time"] + BAR + delay).all()
        and rows["exit_time"].eq(
            rows["entry_time"] + prereg.Policy().hold_bars * BAR
        ).all()
        and rows["source_date"].lt(rows["signal_available_time"]).all()
        and rows["side"].isin(("LONG", "SHORT")).all()
        and rows["sponsor_surface"].isin(allowed_sponsors).all()
        and not rows["signal_id"].duplicated().any()
        and all(
            signal_id({column: getattr(row, column) for column in CLOCK_COLUMNS})
            == row.signal_id
            for row in rows.itertuples(index=False)
        )
    )


def _schedule_integrity(rows: pd.DataFrame, control: str) -> bool:
    if rows.empty:
        return True
    delay = (
        prereg.Policy().hold_bars * BAR
        if control == "one_day_execution_delay"
        else pd.Timedelta(0)
    )
    for row in rows.itertuples(index=False):
        session = _next_session_date(row.source_date)
        expected_signal = _ny_time(session, 9, 30)
        expected_entry = _ny_time(session, 9, 35) + delay
        if row.signal_available_time != expected_signal or row.entry_time != expected_entry:
            return False
    return True


def _reservation_integrity(rows: pd.DataFrame) -> bool:
    ordered = rows.sort_values(["entry_time", "signal_id"], kind="mergesort")
    if len(ordered) < 2:
        return True
    return bool(
        ordered["entry_time"]
        .iloc[1:]
        .reset_index(drop=True)
        .ge(ordered["exit_time"].iloc[:-1].reset_index(drop=True))
        .all()
    )


def _raw_retention(
    primary_raw: pd.DataFrame,
    control_raw: pd.DataFrame,
    split: str,
) -> float | None:
    left = _window(primary_raw, split)
    right = _window(control_raw, split)
    if right.empty:
        return None
    primary_dates = set(pd.Timestamp(value) for value in left["source_date"])
    control_dates = set(pd.Timestamp(value) for value in right["source_date"])
    if len(control_dates) != len(right):
        raise RuntimeError("OPRR raw control source dates duplicated")
    return float(len(primary_dates & control_dates) / len(control_dates))


def _position_share(rows: pd.DataFrame, column: str, token: str) -> float | None:
    return float(rows[column].eq(token).mean()) if len(rows) else None


def _composition_metrics(
    controls: Mapping[str, pd.DataFrame],
    raw: Mapping[str, pd.DataFrame],
    split: str,
) -> dict[str, Any]:
    primary = controls["primary"]
    rows = _window(primary, split)
    total = len(rows)
    metrics: dict[str, Any] = {"events": total}
    metrics["one_step_share"] = (
        float(rows["rotation_magnitude"].eq("ONE_STEP").mean()) if total else None
    )
    metrics["two_step_share"] = (
        float(rows["rotation_magnitude"].eq("TWO_STEP").mean()) if total else None
    )
    position_number = {"BELOW": 0, "MIDDLE": 1, "ABOVE": 2}
    family_counts = {"0<->1": 0, "1<->2": 0, "0<->2": 0}
    for row in rows.itertuples(index=False):
        prior = position_number.get(str(row.prior_sponsor_position))
        current = position_number.get(str(row.current_sponsor_position))
        if prior is None or current is None or prior == current:
            continue
        family = f"{min(prior, current)}<->{max(prior, current)}"
        family_counts[family] += 1
    for family, count in family_counts.items():
        metrics[f"transition_family_{family}_share"] = (
            float(count / total) if total else None
        )
    for token in ("BELOW", "MIDDLE", "ABOVE"):
        metrics[f"prior_position_{token.lower()}_share"] = _position_share(
            rows, "prior_sponsor_position", token
        )
        metrics[f"current_position_{token.lower()}_share"] = _position_share(
            rows, "current_sponsor_position", token
        )
    metrics["raw_primary_retention_within_option_own"] = _raw_retention(
        raw["primary"], raw["option_own_confirmed"], split
    )
    metrics["raw_primary_retention_within_non_option_pair"] = _raw_retention(
        raw["primary"], raw["non_option_pair_only"], split
    )
    for control in ("term_sponsor_rotation", "tail_sponsor_rotation"):
        metrics[f"{control}_exact_entry_jaccard"] = exact_entry_jaccard(
            rows, _window(controls[control], split)
        )
    metrics["one_common_date_stale_same_side_reproduction"] = (
        same_side_reproduction(primary, controls["one_common_date_stale"], split)
    )
    metrics["deterministic_random_side_same_side_reproduction"] = (
        same_side_reproduction(
            primary, controls["deterministic_random_side"], split
        )
    )
    return metrics


def support_checks(
    states: pd.DataFrame,
    transitions: pd.DataFrame,
    controls: Mapping[str, pd.DataFrame],
    raw: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, Any], dict[str, bool], dict[str, Any], dict[str, bool]]:
    payload = prereg.build_manifest()
    gate = payload["source_support_gate"]
    composition_gate = payload["rotation_composition_gate"]
    primary = controls["primary"]
    statistics = {name: clock_stats(_window(primary, name)) for name in WINDOWS}
    source_checks: dict[str, bool] = {
        "states_nonempty": not states.empty,
        "transitions_nonempty": not transitions.empty,
        "state_dates_unique_increasing": bool(
            not states["observation_date"].duplicated().any()
            and states["observation_date"].is_monotonic_increasing
        ),
        "clock_schema_outcome_safe": bool(
            not any(
                token in column.lower()
                for column in CLOCK_COLUMNS
                for token in FORBIDDEN_CLOCK_TOKENS
            )
        ),
    }
    for name in prereg.CONTROL_ORDER:
        source_checks[f"{name}:timing_integrity"] = _timing_integrity(
            controls[name], name
        )
        source_checks[f"{name}:prospective_schedule_integrity"] = (
            _schedule_integrity(controls[name], name)
        )
        source_checks[f"{name}:reservation_integrity"] = _reservation_integrity(
            controls[name]
        )
    train = statistics["train"]
    selection = statistics["selection"]
    source_checks.update(
        {
            "train_events_min": train["events"] >= gate["train_events_min"],
            "2021_events_min": (
                statistics["2021"]["events"] >= gate["each_train_year_events_min"]
            ),
            "2022_events_min": (
                statistics["2022"]["events"] >= gate["each_train_year_events_min"]
            ),
            "train_active_months_min": (
                train["active_months"] >= gate["train_active_months_min"]
            ),
            "train_long_share_min": bool(
                train["long_share"] is not None
                and train["long_share"] >= gate["train_each_side_share_min"]
            ),
            "train_short_share_min": bool(
                train["short_share"] is not None
                and train["short_share"] >= gate["train_each_side_share_min"]
            ),
            "train_max_month_share": bool(
                train["maximum_month_share"] is not None
                and train["maximum_month_share"] <= gate["train_max_month_share"]
            ),
            "train_max_quarter_share": bool(
                train["maximum_quarter_share"] is not None
                and train["maximum_quarter_share"] <= gate["train_max_quarter_share"]
            ),
            "train_max_entry_gap_days": bool(
                train["maximum_gap_days"] is not None
                and train["maximum_gap_days"] <= gate["train_max_entry_gap_days"]
            ),
            "train_max_same_side_run": (
                train["maximum_same_side_run"] <= gate["train_max_same_side_run"]
            ),
            "selection_events_min": (
                selection["events"] >= gate["selection_events_min"]
            ),
            "2023_h1_events_min": (
                statistics["2023_h1"]["events"]
                >= gate["selection_each_half_events_min"]
            ),
            "2023_h2_events_min": (
                statistics["2023_h2"]["events"]
                >= gate["selection_each_half_events_min"]
            ),
            **{
                f"{quarter}_events_min": (
                    statistics[quarter]["events"]
                    >= gate["selection_each_quarter_events_min"]
                )
                for quarter in ("2023_q1", "2023_q2", "2023_q3", "2023_q4")
            },
            "selection_active_months_min": (
                selection["active_months"] >= gate["selection_active_months_min"]
            ),
            "selection_long_share_min": bool(
                selection["long_share"] is not None
                and selection["long_share"] >= gate["selection_each_side_share_min"]
            ),
            "selection_short_share_min": bool(
                selection["short_share"] is not None
                and selection["short_share"] >= gate["selection_each_side_share_min"]
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
        }
    )
    composition = {
        split: _composition_metrics(controls, raw, split)
        for split in ("train", "selection")
    }
    composition_checks: dict[str, bool] = {}
    for split, metrics in composition.items():
        composition_checks[f"{split}:one_step_share"] = bool(
            metrics["one_step_share"] is not None
            and metrics["one_step_share"] >= composition_gate["one_step_share_min"]
        )
        composition_checks[f"{split}:two_step_share"] = bool(
            metrics["two_step_share"] is not None
            and metrics["two_step_share"] >= composition_gate["two_step_share_min"]
        )
        for family in composition_gate["transition_families"]:
            value = metrics[f"transition_family_{family}_share"]
            composition_checks[f"{split}:transition_family_{family}_share"] = bool(
                value is not None
                and value
                >= composition_gate["each_undirected_transition_family_share_min"]
            )
        for position in ("below", "middle", "above"):
            prior = metrics[f"prior_position_{position}_share"]
            current = metrics[f"current_position_{position}_share"]
            composition_checks[f"{split}:prior_position_{position}_share"] = bool(
                prior is not None
                and prior >= composition_gate["each_prior_position_share_min"]
            )
            composition_checks[f"{split}:current_position_{position}_share"] = bool(
                current is not None
                and current >= composition_gate["each_current_position_share_min"]
            )
        option_retention = metrics["raw_primary_retention_within_option_own"]
        composition_checks[f"{split}:option_own_raw_retention"] = bool(
            option_retention is not None
            and option_retention
            <= composition_gate["raw_primary_retention_within_option_own_max"]
        )
        pair_retention = metrics["raw_primary_retention_within_non_option_pair"]
        composition_checks[f"{split}:non_option_pair_raw_retention"] = bool(
            pair_retention is not None
            and pair_retention
            <= composition_gate["raw_primary_retention_within_non_option_pair_max"]
        )
        for control in ("term_sponsor_rotation", "tail_sponsor_rotation"):
            jaccard = metrics[f"{control}_exact_entry_jaccard"]
            composition_checks[f"{split}:{control}_exact_entry_jaccard"] = bool(
                jaccard is not None
                and jaccard <= composition_gate["sponsor_exact_entry_jaccard_max"]
            )
        stale = metrics["one_common_date_stale_same_side_reproduction"]
        composition_checks[f"{split}:stale_same_side_reproduction"] = bool(
            stale is not None
            and stale <= composition_gate["stale_same_side_reproduction_max"]
        )
        random = metrics["deterministic_random_side_same_side_reproduction"]
        composition_checks[f"{split}:random_same_side_reproduction"] = bool(
            random is not None
            and random <= composition_gate["random_same_side_reproduction_max"]
        )
    return statistics, source_checks, composition, composition_checks


def first_failure(
    source_checks: Mapping[str, bool],
    composition_checks: Mapping[str, bool],
    novelty_checks: Mapping[str, bool],
    *,
    artifact_eligible: bool,
) -> tuple[str, str | None]:
    for name, passed in source_checks.items():
        if not passed:
            return "source_support", name
    for name, passed in composition_checks.items():
        if not passed:
            return "rotation_composition", name
    if not artifact_eligible:
        return "artifact_eligibility", "synthetic_or_injected_build"
    for name, passed in novelty_checks.items():
        if not passed:
            return "comparator_novelty", name
    if not novelty_checks:
        return "comparator_novelty", "required_comparator_checks_missing"
    return "none", None


def _combined_clock(controls: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    nonempty = [
        controls[name] for name in prereg.CONTROL_ORDER if not controls[name].empty
    ]
    if not nonempty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    return (
        pd.concat(nonempty, ignore_index=True)
        .sort_values(["entry_time", "control", "signal_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def deterministic_clock_bytes(controls: Mapping[str, pd.DataFrame]) -> bytes:
    combined = _combined_clock(controls)
    if list(combined.columns) != list(CLOCK_COLUMNS):
        raise RuntimeError("OPRR clock schema drift")
    if any(
        token in column.lower()
        for column in combined.columns
        for token in FORBIDDEN_CLOCK_TOKENS
    ):
        raise RuntimeError("OPRR clock contains forbidden outcome/raw field")
    serialized = combined.copy()
    serialized["source_date"] = serialized["source_date"].map(_format_date)
    for column in ("signal_available_time", "entry_time", "exit_time"):
        serialized[column] = serialized[column].map(_format_time)
    text = serialized.to_csv(
        index=False, columns=CLOCK_COLUMNS, lineterminator="\n"
    ).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as zipped:
        zipped.write(text)
    return buffer.getvalue()


def maximum_tolerant_matches(
    left: Iterable[pd.Timestamp], right: Iterable[pd.Timestamp]
) -> int:
    first = sorted(pd.Timestamp(value) for value in left)
    second = sorted(pd.Timestamp(value) for value in right)
    first_dates = [value.tz_convert(NY_TZ).date() for value in first]
    second_dates = [value.tz_convert(NY_TZ).date() for value in second]
    previous = [0] * (len(second) + 1)
    for left_date in first_dates:
        current = [0] * (len(second) + 1)
        for index, right_date in enumerate(second_dates, start=1):
            best = max(previous[index], current[index - 1])
            if abs((left_date - right_date).days) <= 1:
                best = max(best, previous[index - 1] + 1)
            current[index] = best
        previous = current
    return previous[-1]


def tolerant_entry_jaccard(
    left: pd.DataFrame, right: pd.DataFrame
) -> float | None:
    if left.empty or right.empty:
        return None
    matched = maximum_tolerant_matches(left["entry_time"], right["entry_time"])
    denominator = len(left) + len(right) - matched
    return float(matched / denominator) if denominator else None


def _side_sign(row: Any) -> int:
    if hasattr(row, "side_sign"):
        return int(row.side_sign)
    side = str(row.side)
    if side not in ("LONG", "SHORT"):
        raise RuntimeError("OPRR side invalid")
    return 1 if side == "LONG" else -1


def _signed_occupancy(
    rows: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> np.ndarray:
    grid_size = int((end - start) / BAR)
    occupancy = np.zeros(grid_size, dtype=np.int8)
    ordered = rows.sort_values("entry_time", kind="mergesort")
    for row in ordered.itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        exit_time = pd.Timestamp(row.exit_time)
        if (entry - start) % BAR or (exit_time - start) % BAR:
            raise RuntimeError("OPRR comparator interval off five-minute grid")
        clipped_entry = max(entry, start)
        clipped_exit = min(exit_time, end)
        if clipped_entry >= clipped_exit:
            continue
        left = int((clipped_entry - start) / BAR)
        right = int((clipped_exit - start) / BAR)
        if np.any(occupancy[left:right] != 0):
            raise RuntimeError("OPRR selected clock overlaps itself")
        occupancy[left:right] = _side_sign(row)
    return occupancy


def occupancy_metrics(
    left: pd.DataFrame,
    right: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[float | None, float | None]:
    first = _signed_occupancy(left, start, end)
    second = _signed_occupancy(right, start, end)
    first_active = first != 0
    second_active = second != 0
    union = first_active | second_active
    position_jaccard = (
        float(np.sum(first_active & second_active) / np.sum(union))
        if np.any(union)
        else None
    )
    first_centered = first.astype(np.float64) - float(np.mean(first))
    second_centered = second.astype(np.float64) - float(np.mean(second))
    denominator = float(
        np.sqrt(np.sum(first_centered**2) * np.sum(second_centered**2))
    )
    if denominator == 0.0 or not np.isfinite(denominator):
        return None, position_jaccard
    correlation = float(np.sum(first_centered * second_centered) / denominator)
    if not np.isfinite(correlation):
        return None, position_jaccard
    return abs(correlation), position_jaccard


def _authorize_comparator_open(
    *,
    source_support_passed: bool,
    composition_passed: bool,
    artifact_eligible: bool,
) -> object:
    if not (
        source_support_passed
        and composition_passed
        and artifact_eligible
    ):
        raise RuntimeError(
            "OPRR comparator opening requires committed source and "
            "composition pass"
        )
    return _COMPARATOR_OPEN_TOKEN


def _read_comparator_groups(
    payload: Mapping[str, Any],
    *,
    authorization: object,
) -> tuple[dict[str, dict[str, Any]], int]:
    if authorization is not _COMPARATOR_OPEN_TOKEN:
        raise RuntimeError("OPRR comparator authorization missing")
    groups: dict[str, dict[str, Any]] = {}
    decoded_rows = 0
    for contract in payload["novelty_contract"]["comparators"]:
        if sha256_file(contract["path"]) != contract["sha256"]:
            raise RuntimeError(f"OPRR comparator hash drift: {contract['id']}")
        if prereg.sha256_csv_header(contract["path"]) != contract["header_sha256"]:
            raise RuntimeError(f"OPRR comparator header hash drift: {contract['id']}")
        if prereg.csv_header(contract["path"]) != contract["header"]:
            raise RuntimeError(f"OPRR comparator header drift: {contract['id']}")
        usecols = list(
            dict.fromkeys(
                [
                    contract["group_column"],
                    contract["entry_column"],
                    contract["exit_column"],
                    contract["side_column"],
                ]
            )
        )
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
                raise RuntimeError(f"OPRR comparator group empty: {key}")
            entry = pd.to_datetime(
                selected[contract["entry_column"]], utc=True, errors="raise"
            )
            exit_time = pd.to_datetime(
                selected[contract["exit_column"]], utc=True, errors="raise"
            )
            encoding = {
                str(name): int(value)
                for name, value in contract["side_encoding"].items()
            }
            side_sign = selected[contract["side_column"]].map(
                lambda value: encoding.get(str(value))
            )
            if side_sign.isna().any() or not side_sign.isin((-1, 1)).all():
                raise RuntimeError(f"OPRR comparator side invalid: {key}")
            rows = pd.DataFrame(
                {
                    "entry_time": entry,
                    "exit_time": exit_time,
                    "side_sign": side_sign.astype(int),
                }
            ).sort_values("entry_time", kind="mergesort")
            if rows["entry_time"].duplicated().any():
                raise RuntimeError(f"OPRR comparator entries duplicated: {key}")
            if not rows["exit_time"].gt(rows["entry_time"]).all():
                raise RuntimeError(f"OPRR comparator interval invalid: {key}")
            start = max(
                NOVELTY_START, pd.Timestamp(contract["declared_coverage"][0])
            )
            end = min(NOVELTY_END, pd.Timestamp(contract["declared_coverage"][1]))
            exact_rows = _entry_window(rows, start, end).reset_index(drop=True)
            occupancy_rows = _intersecting_window(rows, start, end).reset_index(
                drop=True
            )
            if exact_rows.empty or occupancy_rows.empty:
                raise RuntimeError(f"OPRR comparator empty in coverage: {key}")
            _signed_occupancy(occupancy_rows, start, end)
            groups[key] = {
                "exact_rows": exact_rows,
                "occupancy_rows": occupancy_rows,
                "start": start,
                "end": end,
                "artifact_id": contract["id"],
                "selected_group": str(selected_group),
            }
    return groups, decoded_rows


def _same_entry_same_side_candidate_share(
    candidate: pd.DataFrame, comparator: pd.DataFrame
) -> float | None:
    if candidate.empty or comparator.empty:
        return None
    lookup = {
        (pd.Timestamp(row.entry_time), int(row.side_sign))
        for row in comparator.itertuples(index=False)
    }
    matched = sum(
        (pd.Timestamp(row.entry_time), _side_sign(row)) in lookup
        for row in candidate.itertuples(index=False)
    )
    return float(matched / len(candidate))


def evaluate_novelty(
    primary: pd.DataFrame,
    payload: Mapping[str, Any],
    *,
    authorization: object,
) -> tuple[dict[str, Any], dict[str, bool], int]:
    if authorization is not _COMPARATOR_OPEN_TOKEN:
        raise RuntimeError("OPRR novelty authorization missing")
    groups, decoded_rows = _read_comparator_groups(
        payload, authorization=authorization
    )
    contract = payload["novelty_contract"]
    report: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for key, group in groups.items():
        start, end = group["start"], group["end"]
        candidate_exact = _entry_window(primary, start, end).reset_index(drop=True)
        candidate_occupancy = _intersecting_window(primary, start, end).reset_index(
            drop=True
        )
        comparator_exact = group["exact_rows"]
        if candidate_exact.empty or candidate_occupancy.empty:
            raise RuntimeError(f"OPRR primary empty in comparator coverage: {key}")
        exact = exact_entry_jaccard(candidate_exact, comparator_exact)
        reproduction = _same_entry_same_side_candidate_share(
            candidate_exact, comparator_exact
        )
        correlation, position = occupancy_metrics(
            candidate_occupancy, group["occupancy_rows"], start, end
        )
        tolerant = tolerant_entry_jaccard(candidate_exact, comparator_exact)
        report[key] = {
            "artifact_id": group["artifact_id"],
            "selected_group": group["selected_group"],
            "common_coverage": [_format_time(start), _format_time(end)],
            "candidate_entry_rows": len(candidate_exact),
            "comparator_entry_rows": len(comparator_exact),
            "exact_entry_jaccard": exact,
            "same_entry_same_side_reproduction": reproduction,
            "absolute_signed_occupancy_pearson": correlation,
            "one_local_calendar_day_tolerant_jaccard_report_only": tolerant,
            "position_time_jaccard_report_only": position,
        }
        checks[f"{key}:exact_entry_jaccard"] = bool(
            exact is not None and exact <= contract["exact_entry_jaccard_max"]
        )
        checks[f"{key}:same_entry_same_side_reproduction"] = bool(
            reproduction is not None
            and reproduction <= contract["same_entry_same_side_reproduction_max"]
        )
        checks[f"{key}:signed_occupancy_pearson"] = bool(
            correlation is not None
            and correlation <= contract["absolute_signed_occupancy_pearson_max"]
        )
    return report, checks, decoded_rows


def _control_report(
    controls: Mapping[str, pd.DataFrame], raw: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    primary = controls["primary"]
    return {
        name: {
            "raw_rows": len(raw[name]),
            "globally_reserved_rows": len(controls[name]),
            "train": clock_stats(_window(controls[name], "train")),
            "selection": clock_stats(_window(controls[name], "selection")),
            "same_side_reproduction_to_primary": {
                split: same_side_reproduction(primary, controls[name], split)
                for split in ("train", "selection")
            },
        }
        for name in prereg.CONTROL_ORDER
    }


def _core_payload(
    states: pd.DataFrame,
    transitions: pd.DataFrame,
    feature_funnel: Mapping[str, Any],
    controls: Mapping[str, pd.DataFrame],
    raw: Mapping[str, pd.DataFrame],
    source_audit: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    clock_bytes: bytes,
    *,
    artifact_eligible: bool,
) -> dict[str, Any]:
    statistics, source_checks, composition, composition_checks = support_checks(
        states, transitions, controls, raw
    )
    source_passed = bool(source_checks and all(source_checks.values()))
    composition_passed = bool(
        source_passed and composition_checks and all(composition_checks.values())
    )
    novelty_report: dict[str, Any] = {}
    novelty_checks: dict[str, bool] = {}
    comparator_rows_decoded = 0
    comparator_status = "not_opened_source_support_or_composition_failed"
    if composition_passed and artifact_eligible:
        authorization = _authorize_comparator_open(
            source_support_passed=source_passed,
            composition_passed=composition_passed,
            artifact_eligible=artifact_eligible,
        )
        novelty_report, novelty_checks, comparator_rows_decoded = evaluate_novelty(
            controls["primary"],
            preregistration,
            authorization=authorization,
        )
        comparator_status = "opened_after_complete_source_and_composition_pass"
    elif composition_passed:
        comparator_status = "synthetic_build_not_authorized"
    novelty_passed = bool(
        composition_passed
        and artifact_eligible
        and novelty_checks
        and all(novelty_checks.values())
    )
    first_stage, first_check = first_failure(
        source_checks,
        composition_checks,
        novelty_checks,
        artifact_eligible=artifact_eligible,
    )
    if not source_passed or not composition_passed:
        decision = "retire_OPRR_288_unchanged_before_comparators_and_outcomes"
    elif not artifact_eligible:
        decision = "synthetic_build_cannot_authorize_comparators_or_outcomes"
    elif not novelty_passed:
        decision = "retire_OPRR_288_unchanged_before_outcomes"
    else:
        decision = "advance_to_separately_frozen_strict_economic_RLLM_evaluator"
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
        "control_report": _control_report(controls, raw),
        "composition_report": composition,
        "source_support_checks": source_checks,
        "source_support_passed": source_passed,
        "composition_checks": composition_checks,
        "composition_passed": composition_passed,
        "comparator_status": comparator_status,
        "novelty_report": novelty_report,
        "novelty_checks": novelty_checks,
        "novelty_passed": novelty_passed,
        "first_failing_stage": first_stage,
        "first_failing_check": first_check,
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
            "freeze_strict_economic_RLLM_evaluator" if novelty_passed else None
        ),
        "outcome_boundary": {
            "term_source_rows_decoded": int(source_audit.get("term_rows_decoded", 0)),
            "tail_source_rows_decoded": int(source_audit.get("tail_rows_decoded", 0)),
            "option_source_rows_decoded": int(source_audit.get("option_rows_decoded", 0)),
            "comparator_rows_decoded": comparator_rows_decoded,
            "BTC_market_rows_decoded": 0,
            "funding_rows_decoded": 0,
            "future_return_rows_decoded": 0,
            "return_or_PnL_fields_decoded": 0,
            "PnL_CAGR_MDD_values_decoded": 0,
            "network_calls": 0,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def build_support_from_states(
    states: pd.DataFrame,
    *,
    feature_funnel: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bytes]:
    transitions = build_transitions(states)
    controls, raw = build_controls(transitions)
    clock_bytes = deterministic_clock_bytes(controls)
    payload = validate_preregistration()
    report = _core_payload(
        states,
        transitions,
        feature_funnel or {"synthetic_or_injected": True},
        controls,
        raw,
        {
            "term_rows_decoded": 0,
            "tail_rows_decoded": 0,
            "option_rows_decoded": 0,
            "synthetic_or_injected": True,
        },
        payload,
        clock_bytes,
        artifact_eligible=False,
    )
    return report, clock_bytes


def build_real_support_payload() -> tuple[dict[str, Any], bytes]:
    _assert_protocol_committed()
    payload = validate_preregistration()
    bindings = verify_pre_source_bindings(payload)
    term = load_term_source()
    tail = load_tail_source()
    option = load_option_source()
    states, funnel = build_common_states(term, tail, option)
    transitions = build_transitions(states)
    controls, raw = build_controls(transitions)
    clock_bytes = deterministic_clock_bytes(controls)
    source_audit = {
        "term": {
            "path": prereg.TERM_SOURCE,
            "sha256": prereg.TERM_SOURCE_SHA256,
            "header_sha256": prereg.TERM_HEADER_SHA256,
            "allowlist": list(prereg.TERM_ALLOWLIST),
            "first_date": _format_date(term["observation_date"].iloc[0]),
            "last_date": _format_date(term["observation_date"].iloc[-1]),
        },
        "tail": {
            "path": prereg.TAIL_SOURCE,
            "sha256": prereg.TAIL_SOURCE_SHA256,
            "header_sha256": prereg.TAIL_HEADER_SHA256,
            "allowlist": list(prereg.TAIL_ALLOWLIST),
            "first_date": _format_date(tail["observation_date"].iloc[0]),
            "last_date": _format_date(tail["observation_date"].iloc[-1]),
        },
        "option": {
            "path": prereg.OPTION_SOURCE,
            "sha256": prereg.OPTION_SOURCE_SHA256,
            "header_sha256": prereg.OPTION_HEADER_SHA256,
            "allowlist": list(prereg.OPTION_ALLOWLIST),
            "first_date": _format_date(option["observation_date"].iloc[0]),
            "last_date": _format_date(option["observation_date"].iloc[-1]),
        },
        "term_rows_decoded": len(term),
        "tail_rows_decoded": len(tail),
        "option_rows_decoded": len(option),
        "pre_source_bindings": bindings,
        "synthetic_or_injected": False,
    }
    return (
        _core_payload(
            states,
            transitions,
            funnel,
            controls,
            raw,
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
            raise RuntimeError(f"OPRR noncanonical existing artifact: {path}")
        return "verified_existing"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
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
                raise RuntimeError(f"OPRR artifact race drift: {path}")
            return "verified_existing"
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def write_support(
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
) -> dict[str, Any]:
    if Path(clock_output) != DEFAULT_CLOCK_OUTPUT:
        raise RuntimeError("OPRR real clock output path is frozen")
    if Path(report_output) != DEFAULT_REPORT_OUTPUT:
        raise RuntimeError("OPRR real report output path is frozen")
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
        "composition_passed": report["composition_passed"],
        "novelty_passed": report["novelty_passed"],
        "decision": report["decision"],
        "manifest_hash": report["manifest_hash"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT_OUTPUT))
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK_OUTPUT))
    args = parser.parse_args()
    result = write_support(args.report_output, args.clock_output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
