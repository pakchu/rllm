"""Build outcome-blind CXRT-288 source-support clocks and novelty evidence."""
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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_cboe_cross_surface_risk_transfer as prereg


PROTOCOL_VERSION = "cboe_cross_surface_risk_transfer_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/build_cboe_cross_surface_risk_transfer_support.py")
TEST_PATH = Path("tests/test_build_cboe_cross_surface_risk_transfer_support.py")
IMPLEMENTATION_CONTRACT = Path(
    "docs/cxrt-source-support-implementation-contract-2026-07-24.md"
)
IMPLEMENTATION_CONTRACT_SHA256 = (
    "92741a24c7e3560a0fbf046f6c749f5b4b16c272464d56a285827346dafe1865"
)
PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = (
    "4e26603221e8109c38151873e31110b61f952a29b484266549e480e7c283af52"
)
PREREGISTRATION_MANIFEST_HASH = (
    "d6ce5f03a18f47f9e8221f91b3aa7af687754c5c6c45d1e6881e3fa1e9c30123"
)
DEFAULT_CLOCK_OUTPUT = Path(
    "data/cboe_cross_surface_risk_transfer_clocks_2020_2023.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/cboe_cross_surface_risk_transfer_support_2026-07-24.json"
)

BAR = pd.Timedelta(minutes=5)
DAY = pd.Timedelta(days=1)
SOURCE_END = pd.Timestamp("2024-01-01T00:00:00Z")
TRAIN_START = pd.Timestamp("2021-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2023-01-01T00:00:00Z")
SELECTION_START = TRAIN_END
SELECTION_END = SOURCE_END

CLOCK_COLUMNS = (
    "control",
    "signal_id",
    "source_date",
    "signal_available_time",
    "entry_time",
    "exit_time",
    "side",
    "term_vote",
    "tail_vote",
    "option_vote",
    "vote_relation",
    "minority_surface",
    "term_bucket",
    "tail_bucket",
    "option_bucket",
    "term_transition",
    "tail_transition",
    "option_transition",
    "prior_majority_transition",
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
)
INDEPENDENT_CONTROLS = (
    "primary",
    "term_only",
    "tail_only",
    "option_only",
    "term_tail_agreement",
    "term_option_agreement",
    "tail_option_agreement",
    "one_day_execution_delay",
)
SURFACES = ("term", "tail", "option")
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
        raise RuntimeError("CXRT timestamp must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp.microsecond or timestamp.nanosecond:
        raise RuntimeError("CXRT timestamp must be whole-second")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_date(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise RuntimeError("CXRT source date must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp != timestamp.floor("D"):
        raise RuntimeError("CXRT source date must be UTC midnight")
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
        raise RuntimeError("CXRT source-support protocol is not committed")
    clean = _git_check("diff", "--quiet", "HEAD", "--", *paths)
    if clean.returncode:
        raise RuntimeError("CXRT source-support protocol differs from HEAD")


def validate_preregistration() -> Mapping[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("CXRT preregistration artifact hash drift")
    payload = json.loads(_path(PREREGISTRATION).read_text(encoding="utf-8"))
    prereg.validate_manifest(payload)
    if payload != prereg.build_manifest():
        raise RuntimeError("CXRT preregistration differs from frozen builder")
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("CXRT preregistration manifest hash drift")
    for field in (
        "outcomes_opened",
        "source_incidence_opened",
        "source_rows_decoded",
        "comparator_rows_decoded",
    ):
        if payload.get(field) is not False:
            raise RuntimeError(f"CXRT preregistration boundary opened: {field}")
    if tuple(payload["source_only_controls"]["ordered"]) != prereg.CONTROL_ORDER:
        raise RuntimeError("CXRT control order drift")
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
            raise RuntimeError(f"CXRT frozen binding changed: {label}")
        audit[label] = {"path": str(path), "sha256": actual}
    source_contracts = payload["source_contracts"]
    expected_allowlists = {
        "term": list(prereg.TERM_ALLOWLIST),
        "tail": list(prereg.TAIL_ALLOWLIST),
        "option": list(prereg.OPTION_ALLOWLIST),
    }
    for name, allowlist in expected_allowlists.items():
        if source_contracts[name]["allowlist"] != allowlist:
            raise RuntimeError(f"CXRT {name} source allowlist drift")
    return audit


def validate_source_frame(
    frame: pd.DataFrame,
    *,
    allowlist: Sequence[str],
    source_name: str,
) -> pd.DataFrame:
    if list(frame.columns) != list(allowlist):
        raise RuntimeError(f"CXRT {source_name} loader did not preserve allowlist")
    validated = frame.copy()
    validated["observation_date"] = pd.to_datetime(
        validated["observation_date"], utc=True, errors="raise"
    )
    dates = validated["observation_date"]
    if dates.duplicated().any():
        raise RuntimeError(f"CXRT {source_name} dates duplicated")
    if not dates.is_monotonic_increasing:
        raise RuntimeError(f"CXRT {source_name} dates not increasing")
    if not dates.eq(dates.dt.floor("D")).all():
        raise RuntimeError(f"CXRT {source_name} dates not UTC midnight")
    if dates.ge(SOURCE_END).any():
        raise RuntimeError(f"CXRT {source_name} includes 2024-or-later data")
    numeric_columns = [column for column in allowlist if column != "observation_date"]
    for column in numeric_columns:
        validated[column] = pd.to_numeric(validated[column], errors="coerce").astype(float)
    values = validated[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not (values > 0.0).all():
        raise RuntimeError(f"CXRT {source_name} primitive invalid")
    return validated


def _load_panel(
    path: str | Path,
    *,
    expected_path: str,
    allowlist: Sequence[str],
    source_name: str,
) -> pd.DataFrame:
    if str(path) != expected_path:
        raise RuntimeError(f"CXRT {source_name} path differs from frozen source")
    frame = pd.read_csv(
        _path(path),
        usecols=list(allowlist),
        dtype={column: "string" for column in allowlist},
    )
    frame = frame.loc[:, list(allowlist)]
    return validate_source_frame(
        frame,
        allowlist=allowlist,
        source_name=source_name,
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
        policy.rank_minimum_prior_observations if minimum is None else int(minimum)
    )
    if width <= 0 or required <= 0 or required > width:
        raise RuntimeError("CXRT rank window invalid")
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
        f"{name}_first_rank": first_rank,
        f"{name}_second_rank": second_rank,
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
    pressure = np.mean(np.vstack(ranks), axis=0)
    return pd.DataFrame(
        {
            "observation_date": frame["observation_date"].to_numpy(),
            "option_first_rank": ranks[0],
            "option_second_rank": ranks[1],
            "option_third_rank": ranks[2],
            "option_pressure": pressure,
        }
    )


def pressure_vote(value: float) -> int:
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise RuntimeError("CXRT pressure outside [0,1]")
    if value < 0.5:
        return 1
    if value > 0.5:
        return -1
    return 0


def pressure_bucket(value: float) -> str:
    vote = pressure_vote(value)
    if vote == 0:
        return "NEUTRAL"
    if value < 0.25:
        return "RELIEF_STRONG"
    if value < 0.5:
        return "RELIEF_WEAK"
    if value <= 0.75:
        return "STRESS_WEAK"
    return "STRESS_STRONG"


def majority_vote(votes: Sequence[int]) -> int:
    normalized = tuple(int(value) for value in votes)
    if any(value not in (-1, 0, 1) for value in normalized):
        raise RuntimeError("CXRT vote invalid")
    total = sum(normalized)
    nonzero = sum(value != 0 for value in normalized)
    if nonzero < 2 or total == 0:
        return 0
    return 1 if total > 0 else -1


def vote_relation(votes: Sequence[int]) -> str:
    majority = majority_vote(votes)
    if majority == 0:
        return "INELIGIBLE"
    normalized = tuple(int(value) for value in votes)
    if normalized == (majority, majority, majority):
        return "UNANIMOUS"
    if 0 in normalized:
        return "NEUTRAL_SUPPORTED"
    return "SPLIT_MAJORITY"


def minority_surface(votes: Sequence[int]) -> str:
    majority = majority_vote(votes)
    if majority == 0:
        return "NONE"
    normalized = tuple(int(value) for value in votes)
    for index, surface in enumerate(SURFACES):
        others = [normalized[j] for j in range(3) if j != index]
        if others == [majority, majority] and normalized[index] != majority:
            return surface.upper()
    return "NONE"


def vote_transition(previous: int | None, current: int) -> str:
    if previous is None:
        return "NO_PRIOR"
    if previous == 0 and current == 0:
        return "NEUTRAL_PERSIST"
    if previous == 0:
        return "FROM_NEUTRAL"
    if current == 0:
        return "TO_NEUTRAL"
    if previous == current:
        return "PERSIST"
    return "FLIP"


def _gap_bucket(previous: pd.Timestamp | None, current: pd.Timestamp) -> str:
    if previous is None:
        return "NO_PRIOR"
    days = int((current - previous) / DAY)
    if days <= 0:
        raise RuntimeError("CXRT common dates not increasing")
    if days == 1:
        return "1D"
    if days <= 3:
        return "2_3D"
    return "4P_D"


def build_common_states(
    term: pd.DataFrame,
    tail: pd.DataFrame,
    option: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    term_features = build_term_features(term)
    tail_features = build_tail_features(tail)
    option_features = build_option_features(option)
    common = term_features.merge(
        tail_features,
        on="observation_date",
        how="inner",
        validate="one_to_one",
    ).merge(
        option_features,
        on="observation_date",
        how="inner",
        validate="one_to_one",
    )
    if common.empty:
        raise RuntimeError("CXRT exact common-date panel empty")
    if not np.equal(
        common["term_vix"].to_numpy(dtype=np.float64),
        common["tail_vix"].to_numpy(dtype=np.float64),
    ).all():
        raise RuntimeError("CXRT term/tail VIX cross-panel mismatch")
    complete = common.dropna(
        subset=["term_pressure", "tail_pressure", "option_pressure"]
    ).copy()
    complete = complete.sort_values("observation_date", kind="mergesort").reset_index(
        drop=True
    )
    records: list[dict[str, Any]] = []
    previous_votes: tuple[int, int, int] | None = None
    previous_date: pd.Timestamp | None = None
    for row in complete.itertuples(index=False):
        date = pd.Timestamp(row.observation_date)
        votes = (
            pressure_vote(row.term_pressure),
            pressure_vote(row.tail_pressure),
            pressure_vote(row.option_pressure),
        )
        majority = majority_vote(votes)
        record = {
            "observation_date": date,
            "term_pressure": float(row.term_pressure),
            "tail_pressure": float(row.tail_pressure),
            "option_pressure": float(row.option_pressure),
            "term_vote": votes[0],
            "tail_vote": votes[1],
            "option_vote": votes[2],
            "majority_vote": majority,
            "eligible": majority != 0,
            "side": "LONG" if majority == 1 else "SHORT" if majority == -1 else "NONE",
            "vote_relation": vote_relation(votes),
            "minority_surface": minority_surface(votes),
            "term_bucket": pressure_bucket(row.term_pressure),
            "tail_bucket": pressure_bucket(row.tail_pressure),
            "option_bucket": pressure_bucket(row.option_pressure),
            "term_transition": vote_transition(
                None if previous_votes is None else previous_votes[0], votes[0]
            ),
            "tail_transition": vote_transition(
                None if previous_votes is None else previous_votes[1], votes[1]
            ),
            "option_transition": vote_transition(
                None if previous_votes is None else previous_votes[2], votes[2]
            ),
            "prior_majority_transition": vote_transition(
                None if previous_votes is None else majority_vote(previous_votes),
                majority,
            ),
            "calendar_gap_bucket": _gap_bucket(previous_date, date),
        }
        records.append(record)
        previous_votes = votes
        previous_date = date
    states = pd.DataFrame(records)
    funnel = {
        "term_rows": len(term),
        "tail_rows": len(tail),
        "option_rows": len(option),
        "term_rank_complete_rows": int(term_features["term_pressure"].notna().sum()),
        "tail_rank_complete_rows": int(tail_features["tail_pressure"].notna().sum()),
        "option_rank_complete_rows": int(option_features["option_pressure"].notna().sum()),
        "exact_common_dates": len(common),
        "rank_complete_common_dates": len(states),
        "schedulable_common_dates": max(0, len(states) - 1),
    }
    return states, funnel


def _ny_time(source_date: Any, hour: int, minute: int) -> pd.Timestamp:
    date = pd.Timestamp(source_date)
    if date.tzinfo is None:
        raise RuntimeError("CXRT schedule date must be timezone-aware")
    utc_date = date.tz_convert("UTC")
    naive = pd.Timestamp(utc_date.date()) + pd.Timedelta(hours=hour, minutes=minute)
    return naive.tz_localize("America/New_York").tz_convert("UTC")


def _state_tokens(state: Mapping[str, Any] | Any) -> dict[str, Any]:
    getter = state.get if isinstance(state, Mapping) else lambda key: getattr(state, key)
    return {
        "term_vote": int(getter("term_vote")),
        "tail_vote": int(getter("tail_vote")),
        "option_vote": int(getter("option_vote")),
        "vote_relation": str(getter("vote_relation")),
        "minority_surface": str(getter("minority_surface")),
        "term_bucket": str(getter("term_bucket")),
        "tail_bucket": str(getter("tail_bucket")),
        "option_bucket": str(getter("option_bucket")),
        "term_transition": str(getter("term_transition")),
        "tail_transition": str(getter("tail_transition")),
        "option_transition": str(getter("option_transition")),
        "prior_majority_transition": str(getter("prior_majority_transition")),
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
        elif isinstance(value, np.integer):
            value = int(value)
        identity[column] = value
    return canonical_hash(identity)


def _candidate_row(
    control: str,
    current_state: Mapping[str, Any] | Any,
    next_date: Any,
    side_vote: int,
    *,
    token_state: Mapping[str, Any] | Any | None = None,
    entry_delay_bars: int = 0,
) -> dict[str, Any]:
    if side_vote not in (-1, 1):
        raise RuntimeError("CXRT candidate side invalid")
    source_date = pd.Timestamp(
        current_state["observation_date"]
        if isinstance(current_state, Mapping)
        else current_state.observation_date
    )
    policy = prereg.Policy()
    signal_available = _ny_time(
        next_date,
        policy.entry_local_hour,
        policy.entry_local_minute - policy.signal_buffer_minutes,
    )
    entry = _ny_time(
        next_date,
        policy.entry_local_hour,
        policy.entry_local_minute,
    ) + entry_delay_bars * BAR
    exit_time = entry + policy.hold_bars * BAR
    row: dict[str, Any] = {
        "control": control,
        "source_date": source_date,
        "signal_available_time": signal_available,
        "entry_time": entry,
        "exit_time": exit_time,
        "side": "LONG" if side_vote == 1 else "SHORT",
        **_state_tokens(current_state if token_state is None else token_state),
    }
    row["signal_id"] = signal_id(row)
    return {column: row[column] for column in CLOCK_COLUMNS}


def raw_candidates(states: pd.DataFrame, control: str) -> pd.DataFrame:
    if control not in INDEPENDENT_CONTROLS:
        raise RuntimeError(f"CXRT independent control unsupported: {control}")
    rows: list[dict[str, Any]] = []
    for index in range(max(0, len(states) - 1)):
        state = states.iloc[index]
        next_date = states.iloc[index + 1]["observation_date"]
        votes = {
            "term": int(state["term_vote"]),
            "tail": int(state["tail_vote"]),
            "option": int(state["option_vote"]),
        }
        side_vote = 0
        if control in ("primary", "one_day_execution_delay"):
            side_vote = int(state["majority_vote"])
        elif control.endswith("_only"):
            side_vote = votes[control.removesuffix("_only")]
        else:
            left, right = control.removesuffix("_agreement").split("_")
            if votes[left] != 0 and votes[left] == votes[right]:
                side_vote = votes[left]
        if side_vote == 0:
            continue
        rows.append(
            _candidate_row(
                control,
                state,
                next_date,
                side_vote,
                entry_delay_bars=(
                    prereg.Policy().hold_bars
                    if control == "one_day_execution_delay"
                    else 0
                ),
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


def _stale_control(primary: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    by_date = {pd.Timestamp(row.observation_date): index for index, row in states.iterrows()}
    rows: list[dict[str, Any]] = []
    for primary_row in primary.itertuples(index=False):
        index = by_date[pd.Timestamp(primary_row.source_date)]
        if index <= 0:
            continue
        stale_state = states.iloc[index - 1]
        stale_vote = int(stale_state["majority_vote"])
        if stale_vote == 0:
            continue
        current_state = states.iloc[index]
        next_date = states.iloc[index + 1]["observation_date"]
        candidate = _candidate_row(
            "one_common_date_stale",
            current_state,
            next_date,
            stale_vote,
            token_state=stale_state,
        )
        if (
            candidate["entry_time"] != primary_row.entry_time
            or candidate["exit_time"] != primary_row.exit_time
        ):
            raise RuntimeError("CXRT stale control did not preserve primary clock")
        rows.append(candidate)
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def build_controls(
    states: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    raw = {name: raw_candidates(states, name) for name in INDEPENDENT_CONTROLS}
    controls = {name: reserve_nonoverlap(raw[name]) for name in INDEPENDENT_CONTROLS}
    primary = controls["primary"]
    controls["one_common_date_stale"] = _stale_control(primary, states)
    controls["exact_direction_flip"] = _same_clock_variant(
        primary,
        "exact_direction_flip",
        ["SHORT" if side == "LONG" else "LONG" for side in primary["side"]],
    )
    random_sides: list[str] = []
    for entry in primary["entry_time"]:
        digest = hashlib.sha256(
            f"CXRT-288|{_format_time(entry)}".encode("ascii")
        ).digest()
        random_sides.append("LONG" if digest[0] < 128 else "SHORT")
    controls["deterministic_random_side"] = _same_clock_variant(
        primary,
        "deterministic_random_side",
        random_sides,
    )
    if set(controls) != set(prereg.CONTROL_ORDER):
        raise RuntimeError("CXRT constructed control set drift")
    ordered = {name: controls[name] for name in prereg.CONTROL_ORDER}
    raw_counts = {
        name: len(raw[name]) if name in raw else len(ordered[name])
        for name in prereg.CONTROL_ORDER
    }
    return ordered, raw_counts


def _contained(rows: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    mask = (
        rows["source_date"].ge(start)
        & rows["signal_available_time"].ge(start)
        & rows["entry_time"].ge(start)
        & rows["source_date"].lt(end)
        & rows["signal_available_time"].lt(end)
        & rows["entry_time"].lt(end)
        & rows["exit_time"].le(end)
    )
    return rows.loc[mask].copy()


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
        "active_months": int(len(month_counts)),
        "maximum_month_share": float(month_counts.max() / total),
        "maximum_quarter_share": float(quarter_counts.max() / total),
        "maximum_gap_days": float(gaps.max() / DAY) if not gaps.empty else None,
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


def exact_entry_jaccard(left: pd.DataFrame, right: pd.DataFrame) -> float:
    first = set(pd.Timestamp(value) for value in left["entry_time"])
    second = set(pd.Timestamp(value) for value in right["entry_time"])
    union = first | second
    return float(len(first & second) / len(union)) if union else 1.0


def _timing_integrity(rows: pd.DataFrame, control: str) -> bool:
    if rows.empty:
        return True
    delay = prereg.Policy().hold_bars * BAR if control == "one_day_execution_delay" else pd.Timedelta(0)
    return bool(
        rows["entry_time"].eq(rows["signal_available_time"] + BAR + delay).all()
        and rows["exit_time"].eq(rows["entry_time"] + prereg.Policy().hold_bars * BAR).all()
        and rows["source_date"].lt(rows["signal_available_time"].dt.floor("D")).all()
        and rows["side"].isin(("LONG", "SHORT")).all()
        and rows["term_vote"].isin((-1, 0, 1)).all()
        and rows["tail_vote"].isin((-1, 0, 1)).all()
        and rows["option_vote"].isin((-1, 0, 1)).all()
        and not rows["signal_id"].duplicated().any()
        and all(
            signal_id({column: getattr(row, column) for column in CLOCK_COLUMNS})
            == row.signal_id
            for row in rows.itertuples(index=False)
        )
    )


def _schedule_integrity(rows: pd.DataFrame, states: pd.DataFrame, control: str) -> bool:
    if rows.empty:
        return True
    next_dates = {
        pd.Timestamp(states.iloc[index]["observation_date"]): pd.Timestamp(
            states.iloc[index + 1]["observation_date"]
        )
        for index in range(max(0, len(states) - 1))
    }
    delay = prereg.Policy().hold_bars * BAR if control == "one_day_execution_delay" else pd.Timedelta(0)
    for row in rows.itertuples(index=False):
        source = pd.Timestamp(row.source_date)
        if source not in next_dates:
            return False
        expected_signal = _ny_time(next_dates[source], 9, 30)
        expected_entry = _ny_time(next_dates[source], 9, 35) + delay
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


def _composition_metrics(
    primary: pd.DataFrame,
    controls: Mapping[str, pd.DataFrame],
    split: str,
) -> dict[str, Any]:
    rows = _window(primary, split)
    total = len(rows)
    metrics: dict[str, Any] = {"events": total}
    for surface in SURFACES:
        votes = rows[f"{surface}_vote"] if total else pd.Series(dtype=int)
        metrics[f"{surface}_relief_share"] = (
            float(votes.eq(1).sum() / total) if total else None
        )
        metrics[f"{surface}_stress_share"] = (
            float(votes.eq(-1).sum() / total) if total else None
        )
    nonunanimous = rows.loc[rows["vote_relation"].ne("UNANIMOUS")]
    metrics["nonunanimous_events"] = len(nonunanimous)
    for surface in SURFACES:
        metrics[f"{surface}_unique_minority_share"] = (
            float(nonunanimous["minority_surface"].eq(surface.upper()).sum() / len(nonunanimous))
            if len(nonunanimous)
            else None
        )
    metrics["unanimous_share"] = (
        float(rows["vote_relation"].eq("UNANIMOUS").sum() / total) if total else None
    )
    for control in (
        "term_only",
        "tail_only",
        "option_only",
        "one_common_date_stale",
        "deterministic_random_side",
    ):
        metrics[f"{control}_same_side_reproduction"] = same_side_reproduction(
            primary, controls[control], split
        )
    return metrics


def support_checks(
    states: pd.DataFrame,
    controls: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, Any], dict[str, bool], dict[str, Any], dict[str, bool]]:
    gate = prereg.build_manifest()["source_support_gate"]
    primary = controls["primary"]
    statistics = {name: clock_stats(_window(primary, name)) for name in WINDOWS}
    train = statistics["train"]
    selection = statistics["selection"]
    source_checks: dict[str, bool] = {
        "train_events_min": train["events"] >= gate["train_events_min"],
        "each_train_year_events_min": all(
            statistics[str(year)]["events"] >= gate["each_train_year_events_min"]
            for year in (2021, 2022)
        ),
        "train_active_months_min": train["active_months"] >= gate["train_active_months_min"],
        "train_side_support": bool(
            train["long_share"] is not None
            and train["short_share"] is not None
            and train["long_share"] >= gate["train_each_side_share_min"]
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
        "train_max_same_side_run": train["maximum_same_side_run"] <= gate["train_max_same_side_run"],
        "selection_events_min": selection["events"] >= gate["selection_events_min"],
        "selection_each_half_events_min": all(
            statistics[name]["events"] >= gate["selection_each_half_events_min"]
            for name in ("2023_h1", "2023_h2")
        ),
        "selection_each_quarter_events_min": all(
            statistics[name]["events"] >= gate["selection_each_quarter_events_min"]
            for name in ("2023_q1", "2023_q2", "2023_q3", "2023_q4")
        ),
        "selection_active_months_min": selection["active_months"] >= gate["selection_active_months_min"],
        "selection_side_support": bool(
            selection["long_share"] is not None
            and selection["short_share"] is not None
            and selection["long_share"] >= gate["selection_each_side_share_min"]
            and selection["short_share"] >= gate["selection_each_side_share_min"]
        ),
        "selection_max_month_share": bool(
            selection["maximum_month_share"] is not None
            and selection["maximum_month_share"] <= gate["selection_max_month_share"]
        ),
        "selection_max_entry_gap_days": bool(
            selection["maximum_gap_days"] is not None
            and selection["maximum_gap_days"] <= gate["selection_max_entry_gap_days"]
        ),
        "selection_max_same_side_run": selection["maximum_same_side_run"] <= gate["selection_max_same_side_run"],
        "all_controls_timing_identity_and_schedule": all(
            _timing_integrity(controls[name], name)
            and _schedule_integrity(controls[name], states, name)
            for name in prereg.CONTROL_ORDER
        ),
        "all_controls_global_nonoverlap": all(
            _reservation_integrity(controls[name]) for name in prereg.CONTROL_ORDER
        ),
        "clock_has_no_outcome_or_raw_columns": not any(
            token in column.lower()
            for column in CLOCK_COLUMNS
            for token in FORBIDDEN_CLOCK_TOKENS
        ),
    }
    composition_gate = gate["composition"]
    composition = {
        split: _composition_metrics(primary, controls, split)
        for split in ("train", "selection")
    }
    composition_checks: dict[str, bool] = {}
    for split, metrics in composition.items():
        for control in prereg.CONTROL_ORDER:
            composition_checks[f"{split}:required_control:{control}"] = len(
                _window(controls[control], split)
            ) > 0
        for surface in SURFACES:
            for token in ("relief", "stress"):
                value = metrics[f"{surface}_{token}_share"]
                composition_checks[f"{split}:{surface}_{token}_share"] = bool(
                    value is not None
                    and value >= composition_gate["each_surface_each_vote_share_min"]
                )
            minority = metrics[f"{surface}_unique_minority_share"]
            composition_checks[f"{split}:{surface}_unique_minority_share"] = bool(
                minority is not None
                and minority >= composition_gate["each_surface_unique_minority_share_min"]
            )
        unanimous = metrics["unanimous_share"]
        lower, upper = composition_gate["unanimous_share_range"]
        composition_checks[f"{split}:unanimous_share"] = bool(
            unanimous is not None and lower <= unanimous <= upper
        )
        for control in ("term_only", "tail_only", "option_only"):
            value = metrics[f"{control}_same_side_reproduction"]
            composition_checks[f"{split}:{control}_same_side_reproduction"] = bool(
                value is not None
                and value <= composition_gate["single_surface_same_side_reproduction_max"]
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
            return "relational_composition", name
    if not artifact_eligible:
        return "artifact_eligibility", "synthetic_or_injected_build"
    for name, passed in novelty_checks.items():
        if not passed:
            return "comparator_novelty", name
    if not novelty_checks:
        return "comparator_novelty", "required_comparator_checks_missing"
    return "none", None


def _combined_clock(controls: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    nonempty = [controls[name] for name in prereg.CONTROL_ORDER if not controls[name].empty]
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
        raise RuntimeError("CXRT clock schema drift")
    serialized = combined.copy()
    serialized["source_date"] = serialized["source_date"].map(_format_date)
    for column in ("signal_available_time", "entry_time", "exit_time"):
        serialized[column] = serialized[column].map(_format_time)
    text = serialized.to_csv(index=False, columns=CLOCK_COLUMNS, lineterminator="\n").encode(
        "utf-8"
    )
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as zipped:
        zipped.write(text)
    return buffer.getvalue()


def maximum_tolerant_matches(
    left: Iterable[pd.Timestamp],
    right: Iterable[pd.Timestamp],
    tolerance: pd.Timedelta,
) -> int:
    first = sorted(pd.Timestamp(value) for value in left)
    second = sorted(pd.Timestamp(value) for value in right)
    i = j = matched = 0
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
    matched = maximum_tolerant_matches(left["entry_time"], right["entry_time"], tolerance)
    denominator = len(left) + len(right) - matched
    return float(matched / denominator) if denominator else 1.0


def _side_sign(row: Any) -> int:
    if hasattr(row, "side_sign"):
        return int(row.side_sign)
    return 1 if row.side == "LONG" else -1


def _signed_occupancy(
    rows: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> np.ndarray:
    grid_size = int((end - start) / BAR)
    occupancy = np.zeros(grid_size, dtype=np.int8)
    for row in rows.sort_values("entry_time", kind="mergesort").itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        exit_time = pd.Timestamp(row.exit_time)
        if (entry - start) % BAR or (exit_time - start) % BAR:
            raise RuntimeError("CXRT comparator interval off five-minute grid")
        left = int((entry - start) / BAR)
        right = int((exit_time - start) / BAR)
        if left < 0 or right > grid_size or left >= right:
            raise RuntimeError("CXRT comparator interval outside coverage")
        if np.any(occupancy[left:right] != 0):
            raise RuntimeError("CXRT selected clock overlaps itself")
        occupancy[left:right] = _side_sign(row)
    return occupancy


def occupancy_metrics(
    left: pd.DataFrame,
    right: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[float | None, float]:
    first = _signed_occupancy(left, start, end)
    second = _signed_occupancy(right, start, end)
    first_active = first != 0
    second_active = second != 0
    union = first_active | second_active
    position_jaccard = (
        float(np.sum(first_active & second_active) / np.sum(union))
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
            raise RuntimeError(f"CXRT comparator hash drift: {contract['id']}")
        if prereg.sha256_csv_header(contract["path"]) != contract["header_sha256"]:
            raise RuntimeError(f"CXRT comparator header hash drift: {contract['id']}")
        if prereg.csv_header(contract["path"]) != contract["header"]:
            raise RuntimeError(f"CXRT comparator header drift: {contract['id']}")
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
                raise RuntimeError(f"CXRT comparator group empty: {key}")
            entry = pd.to_datetime(selected[contract["entry_column"]], utc=True, errors="raise")
            exit_time = pd.to_datetime(selected[contract["exit_column"]], utc=True, errors="raise")
            encoding = {str(name): int(value) for name, value in contract["side_encoding"].items()}
            side_sign = selected[contract["side_column"]].map(
                lambda value: encoding.get(str(value))
            )
            if side_sign.isna().any() or not side_sign.isin((-1, 1)).all():
                raise RuntimeError(f"CXRT comparator side invalid: {key}")
            rows = pd.DataFrame(
                {
                    "entry_time": entry,
                    "exit_time": exit_time,
                    "side_sign": side_sign.astype(int),
                }
            ).sort_values("entry_time", kind="mergesort")
            if rows["entry_time"].duplicated().any():
                raise RuntimeError(f"CXRT comparator entries duplicated: {key}")
            if not rows["exit_time"].gt(rows["entry_time"]).all():
                raise RuntimeError(f"CXRT comparator interval invalid: {key}")
            coverage_start = pd.Timestamp(contract["declared_coverage"][0])
            coverage_end = pd.Timestamp(contract["declared_coverage"][1])
            common_start = max(TRAIN_START, coverage_start)
            common_end = min(SELECTION_END, coverage_end)
            contained = rows.loc[
                rows["entry_time"].ge(common_start) & rows["exit_time"].le(common_end)
            ].reset_index(drop=True)
            if contained.empty:
                raise RuntimeError(f"CXRT comparator empty in common coverage: {key}")
            _signed_occupancy(contained, common_start, common_end)
            groups[key] = {
                "rows": contained,
                "start": common_start,
                "end": common_end,
                "artifact_id": contract["id"],
                "selected_group": str(selected_group),
            }
    return groups, decoded_rows


def _same_entry_same_side_candidate_share(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
) -> float:
    lookup = {
        (pd.Timestamp(row.entry_time), int(row.side_sign))
        for row in comparator.itertuples(index=False)
    }
    matched = sum(
        (pd.Timestamp(row.entry_time), _side_sign(row)) in lookup
        for row in candidate.itertuples(index=False)
    )
    return float(matched / len(candidate)) if len(candidate) else 1.0


def evaluate_novelty(
    primary: pd.DataFrame,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool], int]:
    groups, decoded_rows = _read_comparator_groups(payload)
    contract = payload["novelty_contract"]
    report: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for key, group in groups.items():
        start, end = group["start"], group["end"]
        candidate = _contained(primary, start, end)
        comparator = group["rows"]
        if candidate.empty:
            raise RuntimeError(f"CXRT primary empty in comparator coverage: {key}")
        exact = exact_entry_jaccard(candidate, comparator)
        reproduction = _same_entry_same_side_candidate_share(candidate, comparator)
        correlation, position = occupancy_metrics(candidate, comparator, start, end)
        tolerant = tolerant_entry_jaccard(candidate, comparator, DAY)
        report[key] = {
            "artifact_id": group["artifact_id"],
            "selected_group": group["selected_group"],
            "common_coverage": [_format_time(start), _format_time(end)],
            "candidate_rows": len(candidate),
            "comparator_rows": len(comparator),
            "exact_entry_jaccard": exact,
            "same_entry_same_side_reproduction": reproduction,
            "absolute_signed_occupancy_pearson": correlation,
            "one_calendar_day_tolerant_jaccard_report_only": tolerant,
            "position_time_jaccard_report_only": position,
        }
        checks[f"{key}:exact_entry_jaccard"] = exact <= contract["exact_entry_jaccard_max"]
        checks[f"{key}:same_entry_same_side_reproduction"] = (
            reproduction <= contract["same_entry_same_side_reproduction_max"]
        )
        checks[f"{key}:signed_occupancy_pearson"] = bool(
            correlation is not None
            and correlation <= contract["absolute_signed_occupancy_pearson_max"]
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
            "same_side_reproduction_to_primary": {
                split: same_side_reproduction(primary, controls[name], split)
                for split in ("train", "selection")
            },
        }
        for name in prereg.CONTROL_ORDER
    }


def _core_payload(
    states: pd.DataFrame,
    feature_funnel: Mapping[str, Any],
    controls: Mapping[str, pd.DataFrame],
    raw_counts: Mapping[str, int],
    source_audit: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    clock_bytes: bytes,
    *,
    artifact_eligible: bool,
) -> dict[str, Any]:
    statistics, source_checks, composition, composition_checks = support_checks(
        states, controls
    )
    source_passed = all(source_checks.values())
    composition_passed = bool(source_passed and all(composition_checks.values()))
    novelty_report: dict[str, Any] = {}
    novelty_checks: dict[str, bool] = {}
    comparator_rows_decoded = 0
    comparator_status = "not_opened_source_support_or_composition_failed"
    if composition_passed and artifact_eligible:
        novelty_report, novelty_checks, comparator_rows_decoded = evaluate_novelty(
            controls["primary"], preregistration
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
        decision = "retire_CXRT_288_unchanged_before_comparators_and_outcomes"
    elif not artifact_eligible:
        decision = "synthetic_build_cannot_authorize_comparators_or_outcomes"
    elif not novelty_passed:
        decision = "retire_CXRT_288_unchanged_before_outcomes"
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
        "control_report": _control_report(controls, raw_counts),
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
    controls, raw_counts = build_controls(states)
    clock_bytes = deterministic_clock_bytes(controls)
    payload = validate_preregistration()
    report = _core_payload(
        states,
        feature_funnel or {"synthetic_or_injected": True},
        controls,
        raw_counts,
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
    controls, raw_counts = build_controls(states)
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
            raise RuntimeError(f"CXRT noncanonical existing artifact: {path}")
        return "verified_existing"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
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
                raise RuntimeError(f"CXRT artifact race drift: {path}")
            return "verified_existing"
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def write_support(
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
) -> dict[str, Any]:
    if Path(clock_output) != DEFAULT_CLOCK_OUTPUT:
        raise RuntimeError("CXRT real clock output path is frozen")
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
