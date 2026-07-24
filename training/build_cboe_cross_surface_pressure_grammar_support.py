"""Build outcome-blind CSPG-288 source states, clocks, and support evidence."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import tempfile
from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_cboe_cross_surface_pressure_grammar as prereg


PROTOCOL_VERSION = "cboe_cross_surface_pressure_grammar_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/build_cboe_cross_surface_pressure_grammar_support.py"
)
TEST_PATH = Path(
    "tests/test_build_cboe_cross_surface_pressure_grammar_support.py"
)
IMPLEMENTATION_CONTRACT = Path(
    "docs/cspg-source-support-implementation-contract-2026-07-24.md"
)
IMPLEMENTATION_CONTRACT_SHA256 = (
    "3ad52be04d251a217cf365f4fd3032f5d0fd86825922480f398a58387ae759ef"
)
PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = (
    "aa22b964af179dad2daf617496344eb7c335b2f63cfbf4a32f893c065e58d229"
)
PREREGISTRATION_MANIFEST_HASH = (
    "2145ff3cda4632cbc1bd824bc0ecefcc95cca12e201443c3c8e1401689ef02ef"
)
DEFAULT_CLOCK_OUTPUT = Path(
    "data/cboe_cross_surface_pressure_grammar_clocks_2020_2023.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/cboe_cross_surface_pressure_grammar_support_2026-07-24.json"
)

SOURCE_START = pd.Timestamp("2020-01-01T00:00:00Z")
SOURCE_END = pd.Timestamp("2024-01-01T00:00:00Z")
WINDOWS = {
    "global": (SOURCE_START, SOURCE_END),
    "train": (SOURCE_START, pd.Timestamp("2022-01-01T00:00:00Z")),
    "2020": (SOURCE_START, pd.Timestamp("2021-01-01T00:00:00Z")),
    "2021": (
        pd.Timestamp("2021-01-01T00:00:00Z"),
        pd.Timestamp("2022-01-01T00:00:00Z"),
    ),
    "2022": (
        pd.Timestamp("2022-01-01T00:00:00Z"),
        pd.Timestamp("2023-01-01T00:00:00Z"),
    ),
    "2023": (
        pd.Timestamp("2023-01-01T00:00:00Z"),
        SOURCE_END,
    ),
}
LEVEL_COLUMNS = ("term_level", "tail_level", "option_level")
CHANGE_COLUMNS = ("term_change", "tail_change", "option_change")
LEADER_COLUMNS = ("stress_leader", "relief_leader")
CLOCK_COLUMNS = (
    "signal_id",
    "source_date",
    "signal_available_time",
    "entry_time",
    "exit_time",
    *prereg.TOKEN_COLUMNS,
)
FORBIDDEN_CLOCK_TOKENS = (
    "open",
    "high",
    "low",
    "close",
    "price",
    "raw",
    "rank",
    "action",
    "side",
    "return",
    "future",
    "market",
    "funding",
    "label",
    "reward",
    "pnl",
    "cagr",
    "mdd",
)


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
        raise RuntimeError("CSPG timestamp must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp.microsecond or timestamp.nanosecond:
        raise RuntimeError("CSPG timestamp must be whole-second")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_date(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise RuntimeError("CSPG source date must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp != timestamp.floor("D"):
        raise RuntimeError("CSPG source date must be UTC midnight")
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
        raise RuntimeError("CSPG source-support protocol is not committed")
    clean = _git_check("diff", "--quiet", "HEAD", "--", *paths)
    if clean.returncode:
        raise RuntimeError("CSPG source-support protocol differs from HEAD")


def validate_preregistration() -> Mapping[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("CSPG preregistration artifact hash drift")
    payload = json.loads(_path(PREREGISTRATION).read_text(encoding="utf-8"))
    prereg.validate_manifest(payload)
    if payload != prereg.build_manifest():
        raise RuntimeError("CSPG preregistration differs from frozen builder")
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("CSPG preregistration manifest hash drift")
    boundary = payload["outcome_boundary"]
    for field in (
        "source_values_decoded",
        "cspg_pressures_derived",
        "cspg_tokens_derived",
        "cspg_incidence_rows_derived",
        "market_rows_loaded",
        "funding_rows_loaded",
        "comparator_rows_decoded",
        "future_return_rows_loaded",
        "return_or_pnl_fields",
        "post_2023_rows_loaded",
        "model_labels_created",
        "model_training_runs",
    ):
        if boundary[field] != 0:
            raise RuntimeError(f"CSPG preregistration boundary opened: {field}")
    return payload


def _source_dependencies() -> dict[str, str]:
    return {
        str(IMPLEMENTATION_CONTRACT): IMPLEMENTATION_CONTRACT_SHA256,
        str(PREREGISTRATION): PREREGISTRATION_SHA256,
        prereg.BOUNDARY_DOCUMENT: prereg.BOUNDARY_DOCUMENT_SHA256,
        prereg.MECHANISM_DOCUMENT: prereg.MECHANISM_DOCUMENT_SHA256,
        prereg.TERM_SOURCE: prereg.TERM_SOURCE_SHA256,
        prereg.TERM_MANIFEST: prereg.TERM_MANIFEST_SHA256,
        prereg.TAIL_SOURCE: prereg.TAIL_SOURCE_SHA256,
        prereg.TAIL_MANIFEST: prereg.TAIL_MANIFEST_SHA256,
        prereg.OPTION_SOURCE: prereg.OPTION_SOURCE_SHA256,
        prereg.OPTION_MANIFEST: prereg.OPTION_MANIFEST_SHA256,
    }


def verify_pre_source_bindings(
    payload: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    audit: dict[str, dict[str, str]] = {}
    for path, expected in _source_dependencies().items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"CSPG frozen source binding changed: {path}")
        audit[path] = {"path": path, "sha256": actual}
    source_contracts = payload["source_contracts"]
    contracts = (
        (
            "term",
            prereg.TERM_SOURCE,
            prereg.TERM_HEADER_SHA256,
            prereg.TERM_ALLOWLIST,
        ),
        (
            "tail",
            prereg.TAIL_SOURCE,
            prereg.TAIL_HEADER_SHA256,
            prereg.TAIL_ALLOWLIST,
        ),
        (
            "option",
            prereg.OPTION_SOURCE,
            prereg.OPTION_HEADER_SHA256,
            prereg.OPTION_ALLOWLIST,
        ),
    )
    for name, path, expected_header, allowlist in contracts:
        if prereg.sha256_csv_header(path) != expected_header:
            raise RuntimeError(f"CSPG {name} source header hash drift")
        header = prereg.csv_header(path)
        if [column for column in header if column in allowlist] != list(allowlist):
            raise RuntimeError(f"CSPG {name} source allowlist/order drift")
        if source_contracts[name]["allowlist"] != list(allowlist):
            raise RuntimeError(f"CSPG {name} preregistered allowlist drift")
    return audit


def validate_source_frame(
    frame: pd.DataFrame,
    *,
    allowlist: Sequence[str],
    source_name: str,
) -> pd.DataFrame:
    if list(frame.columns) != list(allowlist):
        raise RuntimeError(f"CSPG {source_name} loader did not preserve allowlist")
    validated = frame.copy()
    validated["observation_date"] = pd.to_datetime(
        validated["observation_date"],
        utc=True,
        errors="raise",
    )
    dates = validated["observation_date"]
    if dates.duplicated().any():
        raise RuntimeError(f"CSPG {source_name} dates duplicated")
    if not dates.is_monotonic_increasing:
        raise RuntimeError(f"CSPG {source_name} dates not increasing")
    if not dates.eq(dates.dt.floor("D")).all():
        raise RuntimeError(f"CSPG {source_name} dates not UTC midnight")
    if dates.ge(SOURCE_END).any():
        raise RuntimeError(f"CSPG {source_name} includes 2024-or-later data")
    numeric = [column for column in allowlist if column != "observation_date"]
    for column in numeric:
        validated[column] = pd.to_numeric(
            validated[column],
            errors="coerce",
        ).astype(float)
    values = validated[numeric].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not (values > 0.0).all():
        raise RuntimeError(f"CSPG {source_name} primitive invalid")
    return validated


def _load_panel(
    path: str | Path,
    *,
    expected_path: str,
    allowlist: Sequence[str],
    source_name: str,
) -> pd.DataFrame:
    if str(path) != expected_path:
        raise RuntimeError(f"CSPG {source_name} path differs from frozen source")
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
        policy.rank_minimum_prior_observations
        if minimum is None
        else int(minimum)
    )
    if width <= 0 or required <= 0 or required > width:
        raise RuntimeError("CSPG rank window invalid")
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
            below = int(np.count_nonzero(prior < value))
            equal = int(np.count_nonzero(prior == value))
            result.append(float((below + 0.5 * equal) / len(prior)))
        history.append(value)
    return np.asarray(result, dtype=np.float64)


def _surface_frame(
    dates: pd.Series,
    primitives: Sequence[np.ndarray],
    *,
    name: str,
    vix: np.ndarray | None = None,
) -> pd.DataFrame:
    ranks = [strict_prior_midranks(values) for values in primitives]
    pressure = np.mean(np.vstack(ranks), axis=0)
    payload: dict[str, Any] = {
        "observation_date": dates.to_numpy(),
        **{
            f"{name}_primitive_{index}_rank": rank
            for index, rank in enumerate(ranks, start=1)
        },
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
        (np.log(vix9d / vix), np.log(vix / vix3m)),
        name="term",
        vix=vix,
    )


def build_tail_features(frame: pd.DataFrame) -> pd.DataFrame:
    skew = frame["SKEW_close"].to_numpy(dtype=np.float64)
    vvix = frame["VVIX_close"].to_numpy(dtype=np.float64)
    vix = frame["VIX_close"].to_numpy(dtype=np.float64)
    return _surface_frame(
        frame["observation_date"],
        (np.log(skew / 100.0), np.log(vvix / vix)),
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
    institutional_gap = np.log(
        (index_put + 0.5) / (index_call + 0.5)
    ) - np.log((equity_put + 0.5) / (equity_call + 0.5))
    vix_call_pressure = np.log((vix_call + 0.5) / (vix_put + 0.5))
    index_share = np.log((index_volume + 1.0) / (total + 1.0))
    deltas = tuple(
        np.concatenate(([np.nan], np.diff(values)))
        for values in (institutional_gap, vix_call_pressure, index_share)
    )
    return _surface_frame(
        frame["observation_date"],
        deltas,
        name="option",
    )


def _pressure_mapping(
    term: float,
    tail: float,
    option: float,
) -> OrderedDict[str, float]:
    return OrderedDict(
        (
            ("TERM", float(term)),
            ("TAIL", float(tail)),
            ("OPTION", float(option)),
        )
    )


def _level_mapping(pressures: Mapping[str, float]) -> OrderedDict[str, str]:
    return OrderedDict(
        (surface, prereg.pressure_level(pressures[surface]))
        for surface in prereg.SURFACES
    )


def _tokens(
    current_pressures: Mapping[str, float],
    previous_pressures: Mapping[str, float],
) -> dict[str, str]:
    current_levels = _level_mapping(current_pressures)
    previous_levels = _level_mapping(previous_pressures)
    changes = OrderedDict(
        (
            surface,
            prereg.change_token(
                current_levels[surface],
                previous_levels[surface],
            ),
        )
        for surface in prereg.SURFACES
    )
    current_topology = {
        "term_level": current_levels["TERM"],
        "tail_level": current_levels["TAIL"],
        "option_level": current_levels["OPTION"],
        "stress_leader": prereg.extreme_leader(
            current_pressures,
            highest=True,
        ),
        "relief_leader": prereg.extreme_leader(
            current_pressures,
            highest=False,
        ),
    }
    previous_topology = {
        "term_level": previous_levels["TERM"],
        "tail_level": previous_levels["TAIL"],
        "option_level": previous_levels["OPTION"],
        "stress_leader": prereg.extreme_leader(
            previous_pressures,
            highest=True,
        ),
        "relief_leader": prereg.extreme_leader(
            previous_pressures,
            highest=False,
        ),
    }
    tokens = OrderedDict(
        (
            ("term_level", current_levels["TERM"]),
            ("tail_level", current_levels["TAIL"]),
            ("option_level", current_levels["OPTION"]),
            ("term_change", changes["TERM"]),
            ("tail_change", changes["TAIL"]),
            ("option_change", changes["OPTION"]),
            ("stress_leader", current_topology["stress_leader"]),
            ("relief_leader", current_topology["relief_leader"]),
            ("dispersion", prereg.dispersion_token(current_pressures)),
            ("agreement", prereg.agreement_token(current_levels)),
            (
                "topology_transition",
                prereg.topology_transition(
                    current_topology,
                    previous_topology,
                ),
            ),
            ("pressure_breadth", prereg.pressure_breadth(changes)),
        )
    )
    return prereg.validate_tokens(tokens)


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
        raise RuntimeError("CSPG exact common-date panel empty")
    if not np.equal(
        common["term_vix"].to_numpy(dtype=np.float64),
        common["tail_vix"].to_numpy(dtype=np.float64),
    ).all():
        raise RuntimeError("CSPG term/tail VIX cross-panel mismatch")
    pressure_columns = (
        "term_pressure",
        "tail_pressure",
        "option_pressure",
    )
    complete = (
        common.dropna(subset=list(pressure_columns))
        .sort_values("observation_date", kind="mergesort")
        .reset_index(drop=True)
    )
    records: list[dict[str, Any]] = []
    previous: OrderedDict[str, float] | None = None
    for row in complete.itertuples(index=False):
        current = _pressure_mapping(
            row.term_pressure,
            row.tail_pressure,
            row.option_pressure,
        )
        if previous is not None:
            records.append(
                {
                    "observation_date": pd.Timestamp(row.observation_date),
                    **{
                        "term_pressure": current["TERM"],
                        "tail_pressure": current["TAIL"],
                        "option_pressure": current["OPTION"],
                    },
                    **_tokens(current, previous),
                }
            )
        previous = current
    states = pd.DataFrame(
        records,
        columns=(
            "observation_date",
            *pressure_columns,
            *prereg.TOKEN_COLUMNS,
        ),
    )
    funnel = {
        "term_rows": len(term),
        "tail_rows": len(tail),
        "option_rows": len(option),
        "term_rank_complete_rows": int(
            term_features["term_pressure"].notna().sum()
        ),
        "tail_rank_complete_rows": int(
            tail_features["tail_pressure"].notna().sum()
        ),
        "option_rank_complete_rows": int(
            option_features["option_pressure"].notna().sum()
        ),
        "exact_common_dates": len(common),
        "rank_complete_common_states": len(complete),
        "token_ready_common_states": len(states),
    }
    return states, funnel


def prefix_invariance(
    term: pd.DataFrame,
    tail: pd.DataFrame,
    option: pd.DataFrame,
    *,
    trim_rows: int = 32,
) -> bool:
    if min(len(term), len(tail), len(option)) <= trim_rows:
        raise RuntimeError("CSPG real-prefix invariance input too short")
    full, _ = build_common_states(term, tail, option)
    prefix_sources = tuple(
        frame.iloc[:-trim_rows].reset_index(drop=True)
        for frame in (term, tail, option)
    )
    prefix, _ = build_common_states(*prefix_sources)
    cutoff = min(
        pd.Timestamp(frame["observation_date"].iloc[-1])
        for frame in prefix_sources
    )
    expected = full.loc[
        full["observation_date"].le(cutoff),
        ["observation_date", *prereg.TOKEN_COLUMNS],
    ].reset_index(drop=True)
    actual = prefix.loc[
        prefix["observation_date"].le(cutoff),
        ["observation_date", *prereg.TOKEN_COLUMNS],
    ].reset_index(drop=True)
    return expected.equals(actual)


def signal_id(row: Mapping[str, Any]) -> str:
    payload = {
        "policy_id": prereg.POLICY_ID,
        "source_date": _format_date(row["source_date"]),
        "entry_time": _format_time(row["entry_time"]),
        "tokens": {column: str(row[column]) for column in prereg.TOKEN_COLUMNS},
    }
    return f"CSPG-{canonical_hash(payload)[:24]}"


def raw_candidates(states: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in states.itertuples(index=False):
        source_date = pd.Timestamp(row.observation_date)
        times = prereg.opportunity_times(source_date.date())
        record: dict[str, Any] = {
            "source_date": source_date,
            "signal_available_time": pd.Timestamp(times["signal_available"]),
            "entry_time": pd.Timestamp(times["entry"]),
            "exit_time": pd.Timestamp(times["exit"]),
            **{
                column: str(getattr(row, column))
                for column in prereg.TOKEN_COLUMNS
            },
        }
        record["signal_id"] = signal_id(record)
        records.append(record)
    return pd.DataFrame(records, columns=CLOCK_COLUMNS)


def reserve_nonoverlap(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    ordered = rows.sort_values(
        ["entry_time", "exit_time", "signal_id"],
        kind="mergesort",
    )
    accepted: list[dict[str, Any]] = []
    previous_exit: pd.Timestamp | None = None
    for row in ordered.itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        exit_time = pd.Timestamp(row.exit_time)
        if entry.tzinfo is None or exit_time.tzinfo is None or exit_time <= entry:
            raise RuntimeError("CSPG reservation interval invalid")
        if previous_exit is None or entry >= previous_exit:
            accepted.append(row._asdict())
            previous_exit = exit_time
    return pd.DataFrame(accepted, columns=CLOCK_COLUMNS).reset_index(drop=True)


def _contained(
    rows: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    mask = (
        rows["source_date"].ge(start)
        & rows["signal_available_time"].ge(start)
        & rows["entry_time"].ge(start)
        & rows["exit_time"].le(end)
    )
    return rows.loc[mask].copy().reset_index(drop=True)


def _window(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    start, end = WINDOWS[name]
    return _contained(rows, start, end)


def _month_counts(rows: pd.DataFrame) -> Counter[str]:
    return Counter(
        pd.Timestamp(value).strftime("%Y-%m") for value in rows["entry_time"]
    )


def _maximum_gap_days(rows: pd.DataFrame) -> int | None:
    if len(rows) < 2:
        return None
    local_dates = [
        pd.Timestamp(value).tz_convert("America/New_York").date()
        for value in rows["entry_time"].sort_values(kind="mergesort")
    ]
    return max(
        (current - previous).days
        for previous, current in zip(local_dates, local_dates[1:])
    )


def clock_stats(rows: pd.DataFrame) -> dict[str, Any]:
    total = len(rows)
    months = _month_counts(rows)
    return {
        "events": total,
        "first_entry": (
            _format_time(rows["entry_time"].min()) if total else None
        ),
        "last_exit": _format_time(rows["exit_time"].max()) if total else None,
        "active_months": len(months),
        "maximum_month_share": (
            float(max(months.values()) / total) if total else None
        ),
        "maximum_gap_days": _maximum_gap_days(rows),
    }


def _partition_counts(rows: pd.DataFrame, year: int) -> dict[str, int]:
    boundaries = (
        (f"{year}_h1", f"{year}-01-01", f"{year}-07-01"),
        (f"{year}_h2", f"{year}-07-01", f"{year + 1}-01-01"),
        (f"{year}_q1", f"{year}-01-01", f"{year}-04-01"),
        (f"{year}_q2", f"{year}-04-01", f"{year}-07-01"),
        (f"{year}_q3", f"{year}-07-01", f"{year}-10-01"),
        (f"{year}_q4", f"{year}-10-01", f"{year + 1}-01-01"),
    )
    return {
        name: len(
            _contained(
                rows,
                pd.Timestamp(start, tz="UTC"),
                pd.Timestamp(end, tz="UTC"),
            )
        )
        for name, start, end in boundaries
    }


def _shares(rows: pd.DataFrame, column: str) -> dict[str, float]:
    total = len(rows)
    if not total:
        return {}
    counts = rows[column].value_counts(dropna=False)
    return {
        str(value): float(count / total)
        for value, count in sorted(counts.items(), key=lambda item: str(item[0]))
    }


def token_stats(rows: pd.DataFrame) -> dict[str, Any]:
    total = len(rows)
    token_shares = {
        column: _shares(rows, column) for column in prereg.TOKEN_COLUMNS
    }
    signature_share = None
    if total:
        signatures = rows.loc[:, list(prereg.TOKEN_COLUMNS)].astype(str).agg(
            "|".join,
            axis=1,
        )
        signature_share = float(signatures.value_counts().max() / total)
    return {
        "events": total,
        "token_shares": token_shares,
        "maximum_exact_signature_share": signature_share,
    }


def _timing_integrity(rows: pd.DataFrame) -> bool:
    for row in rows.itertuples(index=False):
        expected = prereg.opportunity_times(
            pd.Timestamp(row.source_date).date()
        )
        if (
            pd.Timestamp(row.signal_available_time)
            != pd.Timestamp(expected["signal_available"])
            or pd.Timestamp(row.entry_time) != pd.Timestamp(expected["entry"])
            or pd.Timestamp(row.exit_time) != pd.Timestamp(expected["exit"])
        ):
            return False
    return True


def _reservation_integrity(rows: pd.DataFrame) -> bool:
    if rows.empty:
        return True
    ordered = rows.sort_values("entry_time", kind="mergesort")
    entries = ordered["entry_time"].iloc[1:].reset_index(drop=True)
    prior_exits = ordered["exit_time"].iloc[:-1].reset_index(drop=True)
    return bool(entries.ge(prior_exits).all())


def _token_checks(
    rows: pd.DataFrame,
    split: str,
    gate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    report = token_stats(rows)
    shares = report["token_shares"]
    checks: dict[str, bool] = {}
    for column in LEVEL_COLUMNS:
        for level in ("LOW", "HIGH"):
            checks[f"{split}:{column}:{level}_share_min"] = (
                shares[column].get(level, 0.0)
                >= gate["pressure_low_high_each_share_min"]
            )
        checks[f"{split}:{column}:max_share"] = bool(
            shares[column]
            and max(shares[column].values()) <= gate["max_level_or_change_share"]
        )
    for column in CHANGE_COLUMNS:
        for level in ("DOWN", "UP"):
            checks[f"{split}:{column}:{level}_share_min"] = (
                shares[column].get(level, 0.0)
                >= gate["change_down_up_each_share_min"]
            )
        checks[f"{split}:{column}:max_share"] = bool(
            shares[column]
            and max(shares[column].values()) <= gate["max_level_or_change_share"]
        )
    for column in LEADER_COLUMNS:
        non_tie = rows.loc[rows[column].ne("TIE"), column]
        counts = non_tie.value_counts()
        checks[f"{split}:{column}:all_nontie_levels"] = all(
            int(counts.get(level, 0)) > 0 for level in prereg.SURFACES
        )
        checks[f"{split}:{column}:max_nontie_share"] = bool(
            len(non_tie)
            and float(counts.max() / len(non_tie))
            <= gate["max_nontie_leader_share"]
        )
    for column in ("dispersion", "agreement"):
        vocabulary = prereg.TOKEN_VOCABULARY[column]
        checks[f"{split}:{column}:all_levels_min_share"] = all(
            shares[column].get(level, 0.0)
            >= gate["each_dispersion_agreement_share_min"]
            for level in vocabulary
        )
    transition = "topology_transition"
    checks[f"{split}:{transition}:all_levels_min_share"] = all(
        shares[transition].get(level, 0.0) >= gate["each_transition_share_min"]
        for level in prereg.TOKEN_VOCABULARY[transition]
    )
    breadth = "pressure_breadth"
    checks[f"{split}:{breadth}:directional_min_share"] = all(
        shares[breadth].get(level, 0.0)
        >= gate["breadth_falling_rising_each_share_min"]
        for level in ("FALLING", "RISING")
    )
    checks[f"{split}:{breadth}:max_share"] = bool(
        shares[breadth]
        and max(shares[breadth].values()) <= gate["max_breadth_share"]
    )
    maximum_signature = report["maximum_exact_signature_share"]
    checks[f"{split}:maximum_exact_signature_share"] = bool(
        maximum_signature is not None
        and maximum_signature <= gate["max_exact_signature_share"]
    )
    checks[f"{split}:all_tokens_valid"] = all(
        value in prereg.TOKEN_VOCABULARY[column]
        for column in prereg.TOKEN_COLUMNS
        for value in rows[column].astype(str)
    )
    return report, checks


def support_checks(
    rows: pd.DataFrame,
    *,
    prefix_invariant: bool,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, int]],
    dict[str, Any],
    dict[str, bool],
    dict[str, bool],
]:
    gate = prereg.build_manifest()["source_support_gates"]
    statistics = {name: clock_stats(_window(rows, name)) for name in WINDOWS}
    partitions = {
        str(year): _partition_counts(rows, year)
        for year in range(2020, 2024)
    }
    source_checks: dict[str, bool] = {
        "global_opportunities_min": (
            statistics["global"]["events"] >= gate["global_opportunities_min"]
        ),
        "train_2020_2021_min": (
            statistics["train"]["events"] >= gate["train_2020_2021_min"]
        ),
        "year_2020_min": (
            statistics["2020"]["events"] >= gate["year_2020_min"]
        ),
        "each_year_2021_2023_min": all(
            statistics[str(year)]["events"] >= gate["each_year_2021_2023_min"]
            for year in range(2021, 2024)
        ),
        "year_2020_active_months_min": (
            statistics["2020"]["active_months"]
            >= gate["year_2020_active_months_min"]
        ),
        "each_year_2021_2023_active_months_min": all(
            statistics[str(year)]["active_months"]
            >= gate["each_year_2021_2023_active_months_min"]
            for year in range(2021, 2024)
        ),
        "each_half_2021_2023_min": all(
            partitions[str(year)][f"{year}_h{half}"]
            >= gate["each_half_2021_2023_min"]
            for year in range(2021, 2024)
            for half in (1, 2)
        ),
        "each_quarter_2021_2023_min": all(
            partitions[str(year)][f"{year}_q{quarter}"]
            >= gate["each_quarter_2021_2023_min"]
            for year in range(2021, 2024)
            for quarter in range(1, 5)
        ),
        "year_2020_max_month_share": bool(
            statistics["2020"]["maximum_month_share"] is not None
            and statistics["2020"]["maximum_month_share"]
            <= gate["year_2020_max_month_share"]
        ),
        "each_year_2021_2023_max_month_share": all(
            statistics[str(year)]["maximum_month_share"] is not None
            and statistics[str(year)]["maximum_month_share"]
            <= gate["each_year_2021_2023_max_month_share"]
            for year in range(2021, 2024)
        ),
        "max_entry_gap_days": all(
            statistics[str(year)]["maximum_gap_days"] is not None
            and statistics[str(year)]["maximum_gap_days"]
            <= gate["max_entry_gap_days"]
            for year in range(2020, 2024)
        ),
        "future_append_invariance": prefix_invariant,
        "timing_integrity": _timing_integrity(rows),
        "global_nonoverlap": _reservation_integrity(rows),
        "clock_has_no_outcome_raw_or_action_columns": not any(
            token in column.lower()
            for column in CLOCK_COLUMNS
            for token in FORBIDDEN_CLOCK_TOKENS
        ),
    }
    token_report: dict[str, Any] = {}
    token_checks: dict[str, bool] = {}
    split_rows = {
        "train": _window(rows, "train"),
        "2022": _window(rows, "2022"),
        "2023": _window(rows, "2023"),
    }
    for split, subset in split_rows.items():
        report, checks = _token_checks(subset, split, gate)
        token_report[split] = report
        token_checks.update(checks)
    train_levels = {
        column: set(split_rows["train"][column].astype(str))
        for column in prereg.TOKEN_COLUMNS
    }
    for split in ("2022", "2023"):
        for column in prereg.TOKEN_COLUMNS:
            downstream = set(split_rows[split][column].astype(str))
            token_checks[f"{split}:{column}:seen_in_train"] = (
                downstream <= train_levels[column]
            )
    return (
        statistics,
        partitions,
        token_report,
        source_checks,
        token_checks,
    )


def first_failure(
    source_checks: Mapping[str, bool],
    token_checks: Mapping[str, bool],
    *,
    artifact_eligible: bool,
) -> tuple[str, str | None]:
    for name, passed in source_checks.items():
        if not passed:
            return "source_support", name
    for name, passed in token_checks.items():
        if not passed:
            return "token_support", name
    if not artifact_eligible:
        return "artifact_eligibility", "synthetic_or_injected_build"
    return "none", None


def deterministic_clock_bytes(rows: pd.DataFrame) -> bytes:
    if list(rows.columns) != list(CLOCK_COLUMNS):
        raise RuntimeError("CSPG clock schema drift")
    serialized = rows.sort_values(
        ["entry_time", "signal_id"],
        kind="mergesort",
    ).copy()
    serialized["source_date"] = serialized["source_date"].map(_format_date)
    for column in ("signal_available_time", "entry_time", "exit_time"):
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


def _core_payload(
    rows: pd.DataFrame,
    feature_funnel: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    clock_bytes: bytes,
    *,
    prefix_invariant: bool,
    raw_candidates_count: int,
    artifact_eligible: bool,
) -> dict[str, Any]:
    (
        statistics,
        partitions,
        token_report,
        source_checks,
        token_checks,
    ) = support_checks(rows, prefix_invariant=prefix_invariant)
    source_passed = all(source_checks.values())
    token_passed = bool(source_passed and all(token_checks.values()))
    first_stage, first_check = first_failure(
        source_checks,
        token_checks,
        artifact_eligible=artifact_eligible,
    )
    if not source_passed or not token_passed:
        decision = "retire_CSPG_288_unchanged_before_market_outcomes"
    elif not artifact_eligible:
        decision = "synthetic_build_cannot_authorize_market_outcomes"
    else:
        decision = "advance_to_frozen_cheap_baseline_evaluator"
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": prereg.POLICY_ID,
        "artifact_eligible": artifact_eligible,
        "source_incidence_opened": True,
        "outcomes_opened": False,
        "market_loaded": False,
        "funding_loaded": False,
        "comparators_opened": False,
        "post_2023_loaded": False,
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
        "reservation_funnel": {
            "raw_candidates": raw_candidates_count,
            "globally_reserved": len(rows),
            "suppressed_overlap": raw_candidates_count - len(rows),
        },
        "clock_statistics": statistics,
        "calendar_partition_counts": partitions,
        "token_report": token_report,
        "source_support_checks": source_checks,
        "source_support_passed": source_passed,
        "token_support_checks": token_checks,
        "token_support_passed": token_passed,
        "first_failing_stage": first_stage,
        "first_failing_check": first_check,
        "clock": {
            "path": str(DEFAULT_CLOCK_OUTPUT),
            "sha256": hashlib.sha256(clock_bytes).hexdigest(),
            "rows": len(rows),
            "columns": list(CLOCK_COLUMNS),
        },
        "decision": decision,
        "authorized_next_stage": (
            "freeze_cheap_baseline_and_economic_evaluator"
            if token_passed and artifact_eligible
            else None
        ),
        "outcome_boundary": {
            "term_source_rows_decoded": int(
                source_audit.get("term_rows_decoded", 0)
            ),
            "tail_source_rows_decoded": int(
                source_audit.get("tail_rows_decoded", 0)
            ),
            "option_source_rows_decoded": int(
                source_audit.get("option_rows_decoded", 0)
            ),
            "cspg_pressure_rows_derived": int(
                feature_funnel.get("rank_complete_common_states", 0)
            ),
            "cspg_token_rows_derived": int(
                feature_funnel.get("token_ready_common_states", 0)
            ),
            "cspg_reserved_rows_derived": len(rows),
            "BTC_market_rows_decoded": 0,
            "funding_rows_decoded": 0,
            "comparator_rows_decoded": 0,
            "future_return_rows_decoded": 0,
            "return_or_PnL_fields_decoded": 0,
            "PnL_CAGR_MDD_values_decoded": 0,
            "post_2023_rows_decoded": 0,
            "model_labels_created": 0,
            "model_training_runs": 0,
            "network_calls": 0,
        },
        "binding_manifest_hash": preregistration["manifest_hash"],
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def build_support_from_states(
    states: pd.DataFrame,
    *,
    feature_funnel: Mapping[str, Any] | None = None,
    prefix_invariant: bool = True,
) -> tuple[dict[str, Any], bytes]:
    raw = raw_candidates(states)
    reserved = reserve_nonoverlap(raw)
    clock_bytes = deterministic_clock_bytes(reserved)
    payload = validate_preregistration()
    report = _core_payload(
        reserved,
        feature_funnel
        or {
            "synthetic_or_injected": True,
            "rank_complete_common_states": len(states) + int(bool(len(states))),
            "token_ready_common_states": len(states),
        },
        {
            "term_rows_decoded": 0,
            "tail_rows_decoded": 0,
            "option_rows_decoded": 0,
            "synthetic_or_injected": True,
        },
        payload,
        clock_bytes,
        prefix_invariant=prefix_invariant,
        raw_candidates_count=len(raw),
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
    invariant = prefix_invariance(term, tail, option)
    raw = raw_candidates(states)
    reserved = reserve_nonoverlap(raw)
    clock_bytes = deterministic_clock_bytes(reserved)
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
            reserved,
            funnel,
            source_audit,
            payload,
            clock_bytes,
            prefix_invariant=invariant,
            raw_candidates_count=len(raw),
            artifact_eligible=True,
        ),
        clock_bytes,
    )


def _write_once(path: str | Path, payload: bytes) -> str:
    output = _path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != payload:
            raise RuntimeError(f"CSPG noncanonical existing artifact: {path}")
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
                raise RuntimeError(f"CSPG artifact race drift: {path}")
            return "verified_existing"
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def write_support(
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
) -> dict[str, Any]:
    if Path(clock_output) != DEFAULT_CLOCK_OUTPUT:
        raise RuntimeError("CSPG real clock output path is frozen")
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
        "token_support_passed": report["token_support_passed"],
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
