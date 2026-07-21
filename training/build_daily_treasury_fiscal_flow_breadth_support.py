"""Build source-only DFFB-601 features, clocks, and support diagnostics.

This stage opens the frozen pre-2024 DTS value rows and the preregistered
source-only comparator clocks.  It never opens BTC market data, funding,
returns, labels, PnL, equity, CAGR, or MDD.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Iterable, Sequence
import unicodedata
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from training import build_daily_treasury_fiscal_flow_source as source_builder
from training import preregister_daily_treasury_fiscal_flow_breadth as prereg


POLICY_ID = "DFFB-601"
PROTOCOL_VERSION = "daily_treasury_fiscal_flow_breadth_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_BUILDER = Path("training/build_daily_treasury_fiscal_flow_breadth_support.py")
DEFAULT_PREREGISTRATION = prereg.DEFAULT_OUTPUT
DEFAULT_OUTPUT = Path(
    "results/daily_treasury_fiscal_flow_breadth_support_2026-07-21.json"
)
DEFAULT_PRIMARY_CLOCK = Path(
    "results/daily_treasury_fiscal_flow_breadth_primary_clock_2026-07-21.csv.gz"
)
DEFAULT_CONTROL_CLOCKS = Path(
    "results/daily_treasury_fiscal_flow_breadth_control_clocks_2026-07-21.csv.gz"
)
EXPECTED_PREREGISTRATION_FILE_SHA256 = (
    "9370ead97eb0cf4ad4ffd271cf691b2d2de08a5099b942f7d3348352485ce6d6"
)
EXPECTED_PREREGISTRATION_MANIFEST_HASH = (
    "67c98b014efc5c46c8096677eb6f6d77651e79001896759d915f86d35f6bbc4f"
)
EXPECTED_POLICY_HASH = (
    "14ed526851127c1fdc86f2795b4c3007e9f38f00bff4305f07a0b99e1b2dff4e"
)
EXPECTED_PREREGISTRATION_SOURCE_SHA256 = (
    "9d4fc9bc7e8d496d9002738203a94d2d7c2ac04d42b2077971b198ed46a50993"
)

SOURCE_COLUMNS = list(prereg.SOURCE_ROWS_HEADER)
SOURCE_MATERIALIZED_COLUMNS = [
    "record_date",
    "source_available_not_before_utc",
    "earliest_execution_time_utc",
    "research_stage",
    "table_id",
    "side",
    "parent_section",
    "raw_category_label",
    "normalized_category_label",
    "today_amount_usd_millions",
    "row_kind",
]
ALLOWED_TABLE_SIDES = {
    ("II", "deposit"),
    ("II", "withdrawal"),
    ("IIIA", "issue"),
    ("IIIA", "redemption"),
}
SIDES = ("deposit", "withdrawal", "issue", "redemption")
EXCLUDED_PREFIXES = tuple(prereg.policy()["source_universe"]["excluded_prefixes"])
EXCLUDED_EXACT = frozenset(prereg.policy()["source_universe"]["excluded_exact_labels"])
EXCLUDED_TABLE_II_BRIDGE_PREFIXES = tuple(
    prereg.policy()["source_universe"]["excluded_table_ii_bridge_prefixes"]
)
ACCOUNT_TOTAL_LABELS = {
    "deposit": {
        "Total Federal Reserve Account",
        "Total TGA Deposits",
        "Treasury General Account Total Deposits",
    },
    "withdrawal": {
        "Total Federal Reserve Account",
        "Total TGA Withdrawals",
        "Treasury General Account Total Withdrawals",
    },
}
FEATURE_COLUMNS = [
    "deposit_breadth",
    "withdrawal_breadth",
    "issue_breadth",
    "redemption_breadth",
    "deposit_eligible_categories",
    "withdrawal_eligible_categories",
    "issue_eligible_categories",
    "redemption_eligible_categories",
    "cash_impulse",
    "debt_impulse",
    "cash_rank126",
    "debt_rank126",
    "total_net_cash",
    "total_net_cash_rank126",
]
REPORT_COLUMNS = [
    "record_date",
    "research_stage",
    "decision_time_utc",
    "entry_time_utc",
    "exit_time_utc",
    *FEATURE_COLUMNS,
]
CLOCK_COLUMNS = [
    "policy_id",
    "clock",
    "window",
    "signal_record_date",
    "execution_record_date",
    "decision_time_utc",
    "entry_time_utc",
    "exit_time_utc",
    "side",
    *FEATURE_COLUMNS,
]
CONTROL_NAMES = [
    "cash_only",
    "debt_only",
    "total_net_cash",
    "direction_flip",
    "one_report_delay",
    "deterministic_random_side",
]
COMPONENT_CONTROL_NAMES = ["cash_only", "debt_only", "total_net_cash"]
WINDOWS = {
    "train": (
        pd.Timestamp("2021-01-01T00:00:00Z"),
        pd.Timestamp("2023-01-01T00:00:00Z"),
    ),
    "selection": (
        pd.Timestamp("2023-01-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T00:00:00Z"),
    ),
}
NEW_YORK = ZoneInfo("America/New_York")
FORBIDDEN_COLUMN_TOKENS = tuple(prereg.PROHIBITED_COMPARATOR_COLUMN_TOKENS)
OUTCOME_BOUNDARY_ZERO_FIELDS = {
    "schema_transition_rows_read": 0,
    "market_rows_loaded": 0,
    "market_values_read": 0,
    "funding_rows_loaded": 0,
    "funding_values_read": 0,
    "return_rows_loaded": 0,
    "return_or_pnl_fields_read": 0,
    "network_calls": 0,
    "database_calls": 0,
    "subprocess_calls": 0,
}


@dataclass(frozen=True)
class Config:
    preregistration: str = str(DEFAULT_PREREGISTRATION)
    output: str = str(DEFAULT_OUTPUT)
    primary_clock: str = str(DEFAULT_PRIMARY_CLOCK)
    control_clocks: str = str(DEFAULT_CONTROL_CLOCKS)
    artifact_root: str = "results"


@dataclass
class CategoryHistory:
    side: str
    values: list[int | None]
    printed_non_null: list[bool]


@dataclass(frozen=True)
class ComparatorClock:
    name: str
    decision_dates: frozenset[date]
    intervals: pd.DataFrame | None = None


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def _regular_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    if candidate.is_symlink():
        raise RuntimeError(f"DFFB support input is a symlink: {path}")
    info = candidate.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"DFFB support input is not a regular file: {path}")
    return candidate.resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _regular_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_timestamp(value: Any, field: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if not isinstance(parsed, pd.Timestamp) or parsed.tzinfo is None:
        raise RuntimeError(f"DFFB {field} must be a timezone-aware timestamp")
    return parsed


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"DFFB {field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"DFFB {field} must be an integer") from exc
    if parsed != value:
        raise RuntimeError(f"DFFB {field} must be an exact integer")
    return parsed


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"DFFB support JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(
        _regular_path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"DFFB support JSON must be an object: {path}")
    return payload


def validate_frozen_preregistration(path: str | Path) -> dict[str, Any]:
    artifact_path = _regular_path(path)
    if artifact_path != _repository_path(DEFAULT_PREREGISTRATION):
        raise RuntimeError("DFFB support preregistration path drift")
    if sha256_file(artifact_path) != EXPECTED_PREREGISTRATION_FILE_SHA256:
        raise RuntimeError("DFFB support preregistration file SHA drift")
    artifact = prereg.load_preregistration(artifact_path)
    if artifact.get("manifest_hash") != EXPECTED_PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("DFFB support preregistration manifest drift")
    if artifact.get("policy_hash") != EXPECTED_POLICY_HASH:
        raise RuntimeError("DFFB support policy hash drift")
    if canonical_hash(artifact.get("policy")) != EXPECTED_POLICY_HASH:
        raise RuntimeError("DFFB support policy content drift")
    if artifact.get("policy") != prereg.policy():
        raise RuntimeError("DFFB support singleton policy drift")
    if artifact.get("outcomes_opened") is not False:
        raise RuntimeError("DFFB support preregistration opened outcomes")
    if artifact.get("incidence_or_support_results") is not None:
        raise RuntimeError("DFFB preregistration already contains support results")
    source_binding = artifact.get("preregistration_source")
    expected_source = {
        "path": str(prereg.PREREGISTRATION_SOURCE),
        "sha256": EXPECTED_PREREGISTRATION_SOURCE_SHA256,
    }
    if source_binding != expected_source:
        raise RuntimeError("DFFB support preregistration source binding drift")
    if (
        sha256_file(prereg.PREREGISTRATION_SOURCE)
        != EXPECTED_PREREGISTRATION_SOURCE_SHA256
    ):
        raise RuntimeError("DFFB support preregistration source file drift")
    source_bindings = artifact.get("source_binding")
    if not isinstance(source_bindings, dict):
        raise RuntimeError("DFFB source binding is missing")
    source_builder_binding = source_bindings.get("source_builder")
    if not isinstance(source_builder_binding, dict):
        raise RuntimeError("DFFB source-builder binding is missing")
    bound_source_builder = _repository_path(str(source_builder_binding.get("path", "")))
    runtime_source_builder = Path(source_builder.__file__).resolve()
    if bound_source_builder != runtime_source_builder:
        raise RuntimeError("DFFB source-builder path drift")
    if sha256_file(bound_source_builder) != source_builder_binding.get("sha256"):
        raise RuntimeError("DFFB source-builder SHA drift")
    return artifact


def _validate_config(cfg: Config) -> None:
    paths = {
        "output": _repository_path(cfg.output),
        "primary_clock": _repository_path(cfg.primary_clock),
        "control_clocks": _repository_path(cfg.control_clocks),
    }
    if not str(cfg.output).endswith(".json"):
        raise ValueError("DFFB support output must be JSON")
    for name in ("primary_clock", "control_clocks"):
        if not str(getattr(cfg, name)).endswith(".csv.gz"):
            raise ValueError(f"DFFB support {name} must be .csv.gz")
    if len(set(paths.values())) != len(paths):
        raise ValueError("DFFB support output paths must be distinct")
    protected = {
        _repository_path(path)
        for path in (
            cfg.preregistration,
            prereg.PREREGISTRATION_SOURCE,
            prereg.PREREGISTRATION_DOCUMENT,
            prereg.SOURCE_ROWS,
            prereg.SOURCE_MANIFEST,
            prereg.SOURCE_BUILD_REPORT,
            prereg.AUDIT_MANIFEST,
            prereg.AUDIT_REPORT,
            prereg.FLCC_CLOCK,
            prereg.TADI_CLOCK,
            prereg.AUCTION_MANIFEST,
            prereg.AUCTION_PANEL,
            prereg.AUCTION_RAW_PAGE_0,
            prereg.AUCTION_RAW_PAGE_1,
            SUPPORT_BUILDER,
        )
    }
    if set(paths.values()) & protected:
        raise ValueError("DFFB support output aliases a frozen input")
    root = _repository_path(cfg.artifact_root)
    outside = [name for name, path in paths.items() if not path.is_relative_to(root)]
    if outside:
        raise ValueError(
            f"DFFB support outputs must stay under artifact root: {outside}"
        )
    existing = [name for name, path in paths.items() if path.exists()]
    if existing:
        raise FileExistsError(f"DFFB support artifacts are immutable: {existing}")


def _normalized_column_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _reject_outcome_columns(columns: Iterable[str], label: str) -> None:
    for column in columns:
        normalized = _normalized_column_key(str(column))
        if any(token in normalized for token in FORBIDDEN_COLUMN_TOKENS):
            raise RuntimeError(f"DFFB {label} contains outcome-like column {column!r}")


def load_source_rows(registration: dict[str, Any]) -> pd.DataFrame:
    binding = registration["source_binding"]["source_rows"]
    path = _regular_path(binding["path"])
    compressed = path.read_bytes()
    if hashlib.sha256(compressed).hexdigest() != binding["sha256"]:
        raise RuntimeError("DFFB normalized source SHA drift")
    observed_header = pd.read_csv(
        io.BytesIO(compressed), compression="gzip", nrows=0
    ).columns.tolist()
    if observed_header != SOURCE_COLUMNS:
        raise RuntimeError("DFFB normalized source schema drift")
    frame = pd.read_csv(
        io.BytesIO(compressed),
        compression="gzip",
        usecols=SOURCE_MATERIALIZED_COLUMNS,
        dtype=str,
        keep_default_na=False,
    )[SOURCE_MATERIALIZED_COLUMNS]
    if frame.columns.tolist() != SOURCE_MATERIALIZED_COLUMNS:
        raise RuntimeError("DFFB normalized source materialized-column drift")
    _reject_outcome_columns(frame.columns, "normalized source")
    if len(frame) != 205_589:
        raise RuntimeError("DFFB normalized source row-count drift")
    if frame["record_date"].nunique() != 1_256:
        raise RuntimeError("DFFB normalized source report-count drift")
    expected_stages = {
        "warmup": 503,
        "train": 502,
        "selection": 250,
        "boundary_quarantine": 1,
    }
    observed_stages = (
        frame.groupby("research_stage", sort=True)["record_date"].nunique().to_dict()
    )
    if observed_stages != expected_stages:
        raise RuntimeError("DFFB normalized source stage drift")
    return frame


def exclusion_key(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    normalized = "".join(
        "-" if unicodedata.category(character) == "Pd" else character
        for character in normalized
    )
    return " ".join(normalized.casefold().split())


def identity_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", exclusion_key(value))


def _retained_detail_row(row: Any) -> bool:
    if (str(row.table_id), str(row.side)) not in ALLOWED_TABLE_SIDES:
        return False
    if str(row.row_kind) != "detail":
        return False
    for value in (str(row.raw_category_label), str(row.normalized_category_label)):
        key = exclusion_key(value)
        if key.startswith(EXCLUDED_PREFIXES) or key in EXCLUDED_EXACT:
            return False
        if str(row.table_id) == "II" and key.startswith(
            EXCLUDED_TABLE_II_BRIDGE_PREFIXES
        ):
            return False
    return True


def _category_key(row: Any) -> tuple[str, str, str, str]:
    return (
        str(row.table_id),
        str(row.side),
        identity_key(str(row.parent_section)),
        identity_key(str(row.normalized_category_label)),
    )


def _parse_amount(value: str, *, context: str) -> int | None:
    if value == "":
        return None
    if not re.fullmatch(r"-?(?:0|[1-9]\d*)", value):
        raise RuntimeError(f"DFFB invalid integer amount in {context}: {value!r}")
    return int(value)


def strict_prior_midrank(current: int | float, prior: Sequence[int | float]) -> float:
    if not prior:
        raise ValueError("DFFB strict-prior midrank requires prior values")
    less = sum(value < current for value in prior)
    equal = sum(value == current for value in prior)
    return (less + 0.5 * equal) / len(prior)


def _report_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "record_date",
        "source_available_not_before_utc",
        "earliest_execution_time_utc",
        "research_stage",
    ]
    unique = frame[columns].drop_duplicates()
    if unique["record_date"].duplicated().any():
        raise RuntimeError("DFFB report metadata is inconsistent within a report")
    for column in ("source_available_not_before_utc", "earliest_execution_time_utc"):
        unique[column] = pd.to_datetime(unique[column], utc=True, errors="raise")
    unique = unique.sort_values(
        ["earliest_execution_time_utc", "record_date"], kind="stable"
    ).reset_index(drop=True)
    if unique["record_date"].duplicated().any():
        raise RuntimeError("DFFB report dates must be unique")
    if not unique["earliest_execution_time_utc"].is_monotonic_increasing:
        raise RuntimeError("DFFB execution clocks are not chronological")
    latency = (
        unique["earliest_execution_time_utc"]
        - unique["source_available_not_before_utc"]
    )
    if not latency.eq(pd.Timedelta(minutes=5)).all():
        raise RuntimeError("DFFB report latency drift")
    return unique


def _account_total(group: pd.DataFrame, side: str) -> int | None:
    candidates = group[
        (group["table_id"] == "II")
        & (group["side"] == side)
        & group["normalized_category_label"].isin(ACCOUNT_TOTAL_LABELS[side])
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"DFFB expected one Table-II account total for {side}, found {len(candidates)}"
        )
    row = candidates.iloc[0]
    return _parse_amount(
        str(row["today_amount_usd_millions"]),
        context=f"{row['record_date']} {side} account total",
    )


def build_report_features(
    source: pd.DataFrame, *, enforce_frozen_inventory: bool = True
) -> pd.DataFrame:
    metadata = _report_metadata(source)
    boundary = metadata[metadata["research_stage"] == "boundary_quarantine"]
    if enforce_frozen_inventory and len(boundary) != 1:
        raise RuntimeError("DFFB boundary-quarantine inventory drift")
    usable = metadata[metadata["research_stage"] != "boundary_quarantine"].copy()
    grouped = {key: value for key, value in source.groupby("record_date", sort=False)}
    histories: dict[tuple[str, str, str, str], CategoryHistory] = {}
    impulse_histories: dict[str, list[float]] = {"cash": [], "debt": []}
    total_history: list[int] = []
    records: list[dict[str, Any]] = []

    for meta in usable.itertuples(index=False):
        group = grouped[str(meta.record_date)]
        current: dict[tuple[str, str, str, str], int | None] = {}
        for row in group.itertuples(index=False):
            if not _retained_detail_row(row):
                continue
            key = _category_key(row)
            if key in current:
                raise RuntimeError(
                    f"DFFB canonical category collision on {meta.record_date}: {key}"
                )
            current[key] = _parse_amount(
                str(row.today_amount_usd_millions),
                context=f"{meta.record_date} {key}",
            )

        tail_counts = Counter({side: 0 for side in SIDES})
        eligible_counts = Counter({side: 0 for side in SIDES})
        side_noncomputable = {side: False for side in SIDES}
        for key, history in histories.items():
            if len(history.values) < 60:
                continue
            prior_values = history.values[-60:]
            prior_printed = history.printed_non_null[-60:]
            if sum(prior_printed) < 12:
                continue
            side = history.side
            eligible_counts[side] += 1
            current_value = current.get(key, 0)
            if current_value is None or any(value is None for value in prior_values):
                side_noncomputable[side] = True
                continue
            numeric_prior = [int(value) for value in prior_values if value is not None]
            rank = strict_prior_midrank(current_value, numeric_prior)
            tail_counts[side] += rank >= 0.75

        breadth: dict[str, float] = {}
        for side in SIDES:
            eligible = int(eligible_counts[side])
            if eligible == 0 or side_noncomputable[side]:
                breadth[side] = math.nan
            else:
                breadth[side] = float(tail_counts[side] / eligible)

        cash_impulse = (
            breadth["withdrawal"] - breadth["deposit"]
            if math.isfinite(breadth["withdrawal"])
            and math.isfinite(breadth["deposit"])
            else math.nan
        )
        debt_impulse = (
            breadth["redemption"] - breadth["issue"]
            if math.isfinite(breadth["redemption"]) and math.isfinite(breadth["issue"])
            else math.nan
        )
        cash_rank = (
            strict_prior_midrank(cash_impulse, impulse_histories["cash"][-126:])
            if math.isfinite(cash_impulse) and len(impulse_histories["cash"]) >= 126
            else math.nan
        )
        debt_rank = (
            strict_prior_midrank(debt_impulse, impulse_histories["debt"][-126:])
            if math.isfinite(debt_impulse) and len(impulse_histories["debt"]) >= 126
            else math.nan
        )

        deposit_total = _account_total(group, "deposit")
        withdrawal_total = _account_total(group, "withdrawal")
        total_net_cash = (
            withdrawal_total - deposit_total
            if withdrawal_total is not None and deposit_total is not None
            else None
        )
        total_rank = (
            strict_prior_midrank(total_net_cash, total_history[-126:])
            if total_net_cash is not None and len(total_history) >= 126
            else math.nan
        )

        entry = _require_timestamp(
            meta.earliest_execution_time_utc, "earliest_execution_time_utc"
        )
        records.append(
            {
                "record_date": str(meta.record_date),
                "research_stage": str(meta.research_stage),
                "decision_time_utc": _require_timestamp(
                    meta.source_available_not_before_utc,
                    "source_available_not_before_utc",
                ),
                "entry_time_utc": entry,
                "exit_time_utc": entry + pd.Timedelta(hours=24),
                "deposit_breadth": breadth["deposit"],
                "withdrawal_breadth": breadth["withdrawal"],
                "issue_breadth": breadth["issue"],
                "redemption_breadth": breadth["redemption"],
                "deposit_eligible_categories": int(eligible_counts["deposit"]),
                "withdrawal_eligible_categories": int(eligible_counts["withdrawal"]),
                "issue_eligible_categories": int(eligible_counts["issue"]),
                "redemption_eligible_categories": int(eligible_counts["redemption"]),
                "cash_impulse": cash_impulse,
                "debt_impulse": debt_impulse,
                "cash_rank126": cash_rank,
                "debt_rank126": debt_rank,
                "total_net_cash": total_net_cash,
                "total_net_cash_rank126": total_rank,
            }
        )

        if math.isfinite(cash_impulse):
            impulse_histories["cash"].append(cash_impulse)
        if math.isfinite(debt_impulse):
            impulse_histories["debt"].append(debt_impulse)
        if total_net_cash is not None:
            total_history.append(total_net_cash)

        existing_keys = set(histories)
        for key in existing_keys:
            value = current.get(key, 0)
            histories[key].values.append(value)
            histories[key].printed_non_null.append(key in current and value is not None)
        for key, value in current.items():
            if key in histories:
                continue
            histories[key] = CategoryHistory(
                side=key[1],
                values=[value],
                printed_non_null=[value is not None],
            )

    features = pd.DataFrame.from_records(records, columns=REPORT_COLUMNS)
    if enforce_frozen_inventory and len(features) != 1_255:
        raise RuntimeError("DFFB usable report count drift")
    return features


def _window_name(entry: pd.Timestamp, exit_time: pd.Timestamp) -> str | None:
    for name, (start, end) in WINDOWS.items():
        if start <= entry and exit_time <= end:
            return name
    return None


def _candidate_side(row: Any, mode: str) -> int:
    cash_rank = float(row.cash_rank126)
    debt_rank = float(row.debt_rank126)
    total_rank = float(row.total_net_cash_rank126)
    if mode == "primary":
        if not math.isfinite(cash_rank) or not math.isfinite(debt_rank):
            return 0
        if cash_rank >= 0.75 and debt_rank >= 0.75:
            return 1
        if cash_rank <= 0.25 and debt_rank <= 0.25:
            return -1
    elif mode == "cash_only":
        if not math.isfinite(cash_rank):
            return 0
        if cash_rank >= 0.75:
            return 1
        if cash_rank <= 0.25:
            return -1
    elif mode == "debt_only":
        if not math.isfinite(debt_rank):
            return 0
        if debt_rank >= 0.75:
            return 1
        if debt_rank <= 0.25:
            return -1
    elif mode == "total_net_cash":
        if not math.isfinite(total_rank):
            return 0
        if total_rank >= 0.75:
            return 1
        if total_rank <= 0.25:
            return -1
    else:
        raise ValueError(f"unknown DFFB clock mode {mode!r}")
    return 0


def _clock_record(
    row: Any,
    *,
    clock: str,
    window: str,
    side: int,
    execution_record_date: str | None = None,
    decision_time: pd.Timestamp | None = None,
    entry_time: pd.Timestamp | None = None,
) -> dict[str, Any]:
    entry = pd.Timestamp(row.entry_time_utc if entry_time is None else entry_time)
    decision = pd.Timestamp(
        row.decision_time_utc if decision_time is None else decision_time
    )
    signal_record_date = str(
        getattr(row, "record_date", getattr(row, "signal_record_date", ""))
    )
    if not signal_record_date:
        raise RuntimeError("DFFB clock row is missing its signal record date")
    return {
        "policy_id": POLICY_ID,
        "clock": clock,
        "window": window,
        "signal_record_date": signal_record_date,
        "execution_record_date": (
            signal_record_date
            if execution_record_date is None
            else execution_record_date
        ),
        "decision_time_utc": decision,
        "entry_time_utc": entry,
        "exit_time_utc": entry + pd.Timedelta(hours=24),
        "side": int(side),
        **{column: getattr(row, column) for column in FEATURE_COLUMNS},
    }


def build_clock(features: pd.DataFrame, *, mode: str, clock: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    last_exit: pd.Timestamp | None = None
    for row in features.sort_values(
        ["entry_time_utc", "record_date"], kind="stable"
    ).itertuples(index=False):
        if row.research_stage not in {"train", "selection"}:
            continue
        side = _candidate_side(row, mode)
        if side == 0:
            continue
        entry = _require_timestamp(row.entry_time_utc, "entry_time_utc")
        exit_time = _require_timestamp(row.exit_time_utc, "exit_time_utc")
        window = _window_name(entry, exit_time)
        if window is None or window != row.research_stage:
            continue
        if last_exit is not None and entry < last_exit:
            continue
        records.append(_clock_record(row, clock=clock, window=window, side=side))
        last_exit = exit_time
    return pd.DataFrame.from_records(records, columns=CLOCK_COLUMNS)


def _same_clock(primary: pd.DataFrame, name: str, sides: Iterable[int]) -> pd.DataFrame:
    control = primary.copy()
    control["clock"] = name
    control["side"] = list(sides)
    return control[CLOCK_COLUMNS]


def _random_side(entry: pd.Timestamp) -> int:
    stamp = pd.Timestamp(entry).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = hashlib.sha256(f"DFFB-601|20260721|{stamp}".encode("ascii")).digest()
    return 1 if digest[0] < 128 else -1


def _one_report_delay(features: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    if primary.empty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    ordered = features.sort_values(
        ["entry_time_utc", "record_date"], kind="stable"
    ).reset_index(drop=True)
    index_by_date = {value: index for index, value in enumerate(ordered["record_date"])}
    candidates: list[dict[str, Any]] = []
    for event in primary.itertuples(index=False):
        source_index = index_by_date[str(event.signal_record_date)]
        if source_index + 1 >= len(ordered):
            continue
        next_report = ordered.iloc[source_index + 1]
        if str(next_report["research_stage"]) != str(event.window):
            continue
        entry = pd.Timestamp(next_report["entry_time_utc"])
        exit_time = entry + pd.Timedelta(hours=24)
        if _window_name(entry, exit_time) != event.window:
            continue
        candidates.append(
            _clock_record(
                event,
                clock="one_report_delay",
                window=str(event.window),
                side=_require_int(event.side, "side"),
                execution_record_date=str(next_report["record_date"]),
                decision_time=pd.Timestamp(next_report["decision_time_utc"]),
                entry_time=entry,
            )
        )
    accepted: list[dict[str, Any]] = []
    last_exit: pd.Timestamp | None = None
    for candidate in sorted(
        candidates,
        key=lambda value: (value["entry_time_utc"], value["signal_record_date"]),
    ):
        entry = pd.Timestamp(candidate["entry_time_utc"])
        if last_exit is not None and entry < last_exit:
            continue
        accepted.append(candidate)
        last_exit = pd.Timestamp(candidate["exit_time_utc"])
    return pd.DataFrame.from_records(accepted, columns=CLOCK_COLUMNS)


def build_control_clocks(features: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    controls = [
        build_clock(features, mode="cash_only", clock="cash_only"),
        build_clock(features, mode="debt_only", clock="debt_only"),
        build_clock(features, mode="total_net_cash", clock="total_net_cash"),
        _same_clock(primary, "direction_flip", -primary["side"].astype(int)),
        _one_report_delay(features, primary),
        _same_clock(
            primary,
            "deterministic_random_side",
            [_random_side(value) for value in primary["entry_time_utc"]],
        ),
    ]
    nonempty = [control for control in controls if not control.empty]
    combined = (
        pd.concat(nonempty, ignore_index=True)
        if nonempty
        else pd.DataFrame(columns=CLOCK_COLUMNS)
    )
    if set(combined["clock"].unique()) - set(CONTROL_NAMES):
        raise RuntimeError("DFFB control clock family drift")
    return (
        combined[CLOCK_COLUMNS]
        .sort_values(["clock", "entry_time_utc", "signal_record_date"], kind="stable")
        .reset_index(drop=True)
    )


def _subset(clock: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    entry = clock["entry_time_utc"]
    return clock[(entry >= pd.Timestamp(start)) & (entry < pd.Timestamp(end))]


def _max_month_share(clock: pd.DataFrame) -> float:
    if clock.empty:
        return 1.0
    return float(
        clock["entry_time_utc"].dt.strftime("%Y-%m").value_counts(normalize=True).max()
    )


def support_gate_summary(
    clock: pd.DataFrame, *, control_name: str | None = None
) -> dict[str, Any]:
    train = clock[clock["window"] == "train"]
    selection = clock[clock["window"] == "selection"]
    periods = {
        "train_2021": _subset(clock, "2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z"),
        "train_2022": _subset(clock, "2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z"),
        "selection_2023_h1": _subset(
            clock, "2023-01-01T00:00:00Z", "2023-07-01T00:00:00Z"
        ),
        "selection_2023_h2": _subset(
            clock, "2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"
        ),
    }
    side_counts = {
        "train": {
            "long": int((train["side"] == 1).sum()),
            "short": int((train["side"] == -1).sum()),
        },
        "selection": {
            "long": int((selection["side"] == 1).sum()),
            "short": int((selection["side"] == -1).sum()),
        },
    }
    checks = {
        "train_total_minimum": len(train) >= 24,
        "train_2021_minimum": len(periods["train_2021"]) >= 8,
        "train_2022_minimum": len(periods["train_2022"]) >= 8,
        "train_long_minimum": side_counts["train"]["long"] >= 6,
        "train_short_minimum": side_counts["train"]["short"] >= 6,
        "train_maximum_month_share": _max_month_share(train) <= 0.25,
        "selection_total_minimum": len(selection) >= 12,
        "selection_each_half_minimum": all(
            len(periods[name]) >= 4
            for name in ("selection_2023_h1", "selection_2023_h2")
        ),
        "selection_long_minimum": side_counts["selection"]["long"] >= 3,
        "selection_short_minimum": side_counts["selection"]["short"] >= 3,
        "selection_maximum_month_share": _max_month_share(selection) <= 0.33,
    }
    return {
        "passed": all(checks.values()),
        "control_name": control_name,
        "checks": checks,
        "counts": {
            "clock_total": int(len(clock)),
            "train": int(len(train)),
            "selection": int(len(selection)),
            **{name: int(len(value)) for name, value in periods.items()},
        },
        "side_counts": side_counts,
        "maximum_month_share": {
            "train": _max_month_share(train),
            "selection": _max_month_share(selection),
        },
        "calendar_month_basis": "entry_time UTC",
        "waived_checks": [],
    }


def _validate_materialized_columns(
    frame: pd.DataFrame, allowed: Sequence[str], label: str
) -> pd.DataFrame:
    if set(frame.columns) != set(allowed) or len(frame.columns) != len(allowed):
        raise RuntimeError(f"DFFB {label} materialized-column drift")
    _reject_outcome_columns(frame.columns, label)
    return frame[list(allowed)]


def _read_bound_csv(
    *,
    path: Path,
    expected_sha: str,
    header: Sequence[str],
    allowed: Sequence[str],
    label: str,
) -> pd.DataFrame:
    physical = _regular_path(path)
    compressed = physical.read_bytes()
    if hashlib.sha256(compressed).hexdigest() != expected_sha:
        raise RuntimeError(f"DFFB {label} SHA drift")
    observed_header = pd.read_csv(
        io.BytesIO(compressed), compression="gzip", nrows=0
    ).columns.tolist()
    if observed_header != list(header):
        raise RuntimeError(f"DFFB {label} header drift")
    frame = pd.read_csv(
        io.BytesIO(compressed),
        compression="gzip",
        usecols=list(allowed),
        dtype=str,
        keep_default_na=False,
    )
    return _validate_materialized_columns(frame, allowed, label)


def _parse_clock_intervals(
    frame: pd.DataFrame,
    *,
    entry_column: str,
    exit_column: str,
    side_column: str,
    label: str,
) -> pd.DataFrame:
    side_values = frame[side_column].astype(str).str.strip().str.upper()
    side_map = {"1": 1, "+1": 1, "LONG": 1, "-1": -1, "SHORT": -1}
    if not side_values.isin(side_map).all():
        raise RuntimeError(f"DFFB {label} side drift")
    output = pd.DataFrame(
        {
            "entry_time_utc": pd.to_datetime(
                frame[entry_column], utc=True, errors="raise"
            ),
            "exit_time_utc": pd.to_datetime(
                frame[exit_column], utc=True, errors="raise"
            ),
            "side": side_values.map(side_map).astype(int),
        }
    )
    if output.empty:
        raise RuntimeError(f"DFFB {label} comparator is empty")
    if not (output["entry_time_utc"] < output["exit_time_utc"]).all():
        raise RuntimeError(f"DFFB {label} interval drift")
    return output.sort_values(
        ["entry_time_utc", "exit_time_utc"], kind="stable"
    ).reset_index(drop=True)


def _new_york_dates(values: Iterable[Any]) -> frozenset[date]:
    return frozenset(
        pd.Timestamp(value).tz_convert(NEW_YORK).date() for value in values
    )


def load_strategy_comparators(
    registration: dict[str, Any],
) -> tuple[dict[str, ComparatorClock], dict[str, Any]]:
    binding = registration["comparator_binding"]
    flcc_binding = binding["flcc"]["clock"]
    flcc = _read_bound_csv(
        path=prereg.FLCC_CLOCK,
        expected_sha=flcc_binding["sha256"],
        header=flcc_binding["header"],
        allowed=flcc_binding["allowed_columns"],
        label="FLCC clock",
    )
    flcc_rows_materialized = len(flcc)
    flcc = flcc[flcc["clock_name"] == "primary"].copy()
    if flcc.empty:
        raise RuntimeError("DFFB FLCC primary comparator is empty")
    comparators: dict[str, ComparatorClock] = {}
    for candidate_id, rows in flcc.groupby("candidate_id", sort=True):
        signal = pd.to_datetime(rows["signal_time"], utc=True, errors="raise")
        intervals = _parse_clock_intervals(
            rows,
            entry_column="entry_time",
            exit_column="exit_time",
            side_column="side",
            label=f"FLCC {candidate_id}",
        )
        comparators[f"flcc:{candidate_id}"] = ComparatorClock(
            name=f"flcc:{candidate_id}",
            decision_dates=_new_york_dates(signal),
            intervals=intervals,
        )
    flcc_union = frozenset(
        value
        for comparator in comparators.values()
        for value in comparator.decision_dates
    )
    comparators["flcc:union"] = ComparatorClock(
        name="flcc:union", decision_dates=flcc_union
    )

    tadi_binding = binding["tadi"]["clock"]
    tadi = _read_bound_csv(
        path=prereg.TADI_CLOCK,
        expected_sha=tadi_binding["sha256"],
        header=tadi_binding["header"],
        allowed=tadi_binding["allowed_columns"],
        label="TADI clock",
    )
    tadi_rows_materialized = len(tadi)
    tadi = tadi[tadi["clock_mode"] == "primary"].copy()
    decision = pd.to_datetime(tadi["decision_time"], utc=True, errors="raise")
    tadi_intervals = _parse_clock_intervals(
        tadi,
        entry_column="entry_time",
        exit_column="scheduled_exit_time",
        side_column="side",
        label="TADI primary",
    )
    comparators["tadi:primary"] = ComparatorClock(
        name="tadi:primary",
        decision_dates=_new_york_dates(decision),
        intervals=tadi_intervals,
    )
    audit = {
        "flcc_rows_materialized": int(flcc_rows_materialized),
        "flcc_primary_rows": int(len(flcc)),
        "flcc_columns_materialized": list(flcc_binding["allowed_columns"]),
        "flcc_candidate_counts": {
            str(name): int(len(rows))
            for name, rows in flcc.groupby("candidate_id", sort=True)
        },
        "tadi_rows_materialized": int(tadi_rows_materialized),
        "tadi_primary_rows": int(len(tadi)),
        "tadi_columns_materialized": list(tadi_binding["allowed_columns"]),
    }
    return comparators, audit


def load_auction_settlement_calendar(
    registration: dict[str, Any],
) -> tuple[frozenset[date], dict[str, Any]]:
    binding = registration["comparator_binding"]["official_auction_settlement_calendar"]
    panel_binding = binding["normalized_panel"]
    panel = _read_bound_csv(
        path=prereg.AUCTION_PANEL,
        expected_sha=panel_binding["sha256"],
        header=panel_binding["header"],
        allowed=panel_binding["allowed_columns"],
        label="auction panel",
    )
    if panel.duplicated(["auction_date", "cusip"]).any():
        raise RuntimeError("DFFB auction panel key duplicate")
    eligible = set(zip(panel["auction_date"], panel["cusip"], strict=True))
    allowed_fields = set(binding["raw_allowed_fields"])
    materialized: list[dict[str, str]] = []
    for raw_binding in binding["raw_pages"]:
        path = Path(raw_binding["path"])
        if sha256_file(path) != raw_binding["sha256"]:
            raise RuntimeError("DFFB auction raw page SHA drift")
        with gzip.open(_regular_path(path), "rt", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_unique_object)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("securityList"), list
        ):
            raise RuntimeError("DFFB auction raw page schema drift")
        for row in payload["securityList"]:
            if not isinstance(row, dict):
                raise RuntimeError("DFFB auction raw row schema drift")
            missing_fields = sorted(allowed_fields - row.keys())
            if missing_fields:
                raise RuntimeError(
                    f"DFFB auction raw row missing allowlisted fields: {missing_fields}"
                )
            materialized.append({field: str(row[field]) for field in allowed_fields})
    matched: dict[tuple[str, str], dict[str, str]] = {}
    for row in materialized:
        auction_date = row["auctionDate"][:10]
        key = (auction_date, row["cusip"])
        if key not in eligible:
            continue
        if key in matched:
            raise RuntimeError(f"DFFB duplicate auction raw key: {key}")
        matched[key] = row
    if set(matched) != eligible:
        missing = sorted(eligible - set(matched))[:5]
        raise RuntimeError(f"DFFB auction raw join is incomplete: {missing}")
    calendar_dates: set[date] = set()
    for (auction_date, _), row in matched.items():
        issue_date = row["issueDate"][:10]
        if not issue_date:
            raise RuntimeError("DFFB auction issueDate is missing")
        calendar_dates.add(date.fromisoformat(auction_date))
        calendar_dates.add(date.fromisoformat(issue_date))
    if not calendar_dates:
        raise RuntimeError("DFFB auction/settlement comparator is empty")
    return frozenset(calendar_dates), {
        "panel_rows_materialized": int(len(panel)),
        "panel_columns_materialized": list(panel_binding["allowed_columns"]),
        "raw_rows_parsed": int(len(materialized)),
        "raw_fields_materialized": sorted(allowed_fields),
        "eligible_keys_joined": int(len(matched)),
        "calendar_dates": int(len(calendar_dates)),
    }


def _previous_business_day(value: date) -> date:
    current = value - timedelta(days=1)
    while not source_builder.is_federal_business_day(current):
        current -= timedelta(days=1)
    return current


def _next_business_day(value: date) -> date:
    current = value + timedelta(days=1)
    while not source_builder.is_federal_business_day(current):
        current += timedelta(days=1)
    return current


def novelty_metrics(
    dffb_dates: frozenset[date], comparator_dates: frozenset[date]
) -> dict[str, Any]:
    if not dffb_dates or not comparator_dates:
        return {
            "passed": False,
            "empty_comparator_or_candidate": True,
            "dffb_dates": len(dffb_dates),
            "comparator_dates": len(comparator_dates),
            "intersection_dates": 0,
            "decision_date_jaccard": 1.0,
            "within_one_us_business_day_fraction": 1.0,
        }
    intersection = dffb_dates & comparator_dates
    union = dffb_dates | comparator_dates
    near = sum(
        bool(
            {
                _previous_business_day(candidate),
                candidate,
                _next_business_day(candidate),
            }
            & comparator_dates
        )
        for candidate in dffb_dates
    )
    jaccard = len(intersection) / len(union)
    near_fraction = near / len(dffb_dates)
    return {
        "passed": jaccard <= 0.30 and near_fraction <= 0.50,
        "empty_comparator_or_candidate": False,
        "dffb_dates": len(dffb_dates),
        "comparator_dates": len(comparator_dates),
        "intersection_dates": len(intersection),
        "decision_date_jaccard": float(jaccard),
        "decision_date_jaccard_maximum": 0.30,
        "within_one_us_business_day_count": int(near),
        "within_one_us_business_day_fraction": float(near_fraction),
        "within_one_us_business_day_fraction_maximum": 0.50,
    }


def _validate_nonoverlap(intervals: pd.DataFrame, label: str) -> None:
    ordered = intervals.sort_values(["entry_time_utc", "exit_time_utc"], kind="stable")
    prior_exit: pd.Timestamp | None = None
    for row in ordered.itertuples(index=False):
        entry = _require_timestamp(row.entry_time_utc, "entry_time_utc")
        exit_time = _require_timestamp(row.exit_time_utc, "exit_time_utc")
        if prior_exit is not None and entry < prior_exit:
            raise RuntimeError(f"DFFB {label} comparator intervals overlap")
        prior_exit = exit_time


def _exposure_vector(
    intervals: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, label: str
) -> np.ndarray:
    total_seconds = int((end - start).total_seconds())
    if total_seconds <= 0 or total_seconds % 300:
        raise RuntimeError(f"DFFB {label} common exposure span is not a 5m grid")
    vector = np.zeros(total_seconds // 300, dtype=np.int8)
    _validate_nonoverlap(intervals, label)
    for row in intervals.itertuples(index=False):
        entry = max(_require_timestamp(row.entry_time_utc, "entry_time_utc"), start)
        exit_time = min(_require_timestamp(row.exit_time_utc, "exit_time_utc"), end)
        if entry >= exit_time:
            continue
        left_seconds = int((entry - start).total_seconds())
        right_seconds = int((exit_time - start).total_seconds())
        if left_seconds % 300 or right_seconds % 300:
            raise RuntimeError(f"DFFB {label} interval is not aligned to 5m")
        left = left_seconds // 300
        right = right_seconds // 300
        if np.any(vector[left:right] != 0):
            raise RuntimeError(f"DFFB {label} exposure assignment overlaps")
        vector[left:right] = _require_int(row.side, "side")
    return vector


def occupied_exposure_correlation(
    primary: pd.DataFrame, comparator: pd.DataFrame, label: str
) -> dict[str, Any]:
    if primary.empty or comparator.empty:
        return {"passed": False, "reason": "empty clock"}
    candidate_intervals = primary[["entry_time_utc", "exit_time_utc", "side"]].copy()
    start = max(
        pd.Timestamp(candidate_intervals["entry_time_utc"].min()),
        pd.Timestamp(comparator["entry_time_utc"].min()),
    )
    end = min(
        pd.Timestamp(candidate_intervals["exit_time_utc"].max()),
        pd.Timestamp(comparator["exit_time_utc"].max()),
    )
    if start >= end:
        return {"passed": False, "reason": "empty common span"}
    candidate = _exposure_vector(candidate_intervals, start, end, "DFFB primary")
    reference = _exposure_vector(comparator, start, end, label)
    if candidate.size == 0 or np.var(candidate) == 0.0 or np.var(reference) == 0.0:
        return {"passed": False, "reason": "empty grid or zero variance"}
    correlation = float(np.corrcoef(candidate, reference)[0, 1])
    if not math.isfinite(correlation):
        return {"passed": False, "reason": "non-finite correlation"}
    return {
        "passed": abs(correlation) <= 0.40,
        "reason": None,
        "common_start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "common_end_utc": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "grid_rows_5m": int(candidate.size),
        "candidate_nonzero_rows": int(np.count_nonzero(candidate)),
        "comparator_nonzero_rows": int(np.count_nonzero(reference)),
        "pearson": correlation,
        "absolute_pearson": abs(correlation),
        "absolute_pearson_maximum": 0.40,
    }


def build_novelty_and_exposure(
    primary: pd.DataFrame,
    controls: pd.DataFrame,
    registration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    strategy_comparators, strategy_audit = load_strategy_comparators(registration)
    auction_dates, auction_audit = load_auction_settlement_calendar(registration)
    dffb_dates = _new_york_dates(primary["decision_time_utc"])
    total_control = controls[controls["clock"] == "total_net_cash"]
    all_date_comparators = {
        name: comparator.decision_dates
        for name, comparator in strategy_comparators.items()
    }
    all_date_comparators["official_auction_settlement_calendar"] = auction_dates
    all_date_comparators["dts_total_net_cash"] = _new_york_dates(
        total_control["decision_time_utc"]
    )
    novelty = {
        name: novelty_metrics(dffb_dates, dates)
        for name, dates in sorted(all_date_comparators.items())
    }
    exposure = {
        name: occupied_exposure_correlation(primary, comparator.intervals, name)
        for name, comparator in sorted(strategy_comparators.items())
        if comparator.intervals is not None
    }
    return (
        {
            "passed": all(value["passed"] for value in novelty.values()),
            "dffb_decision_dates": int(len(dffb_dates)),
            "comparators": novelty,
        },
        {
            "passed": all(value["passed"] for value in exposure.values()),
            "comparators": exposure,
            "input_audit": {
                **strategy_audit,
                "auction_calendar": auction_audit,
            },
        },
    )


def _format_clock(clock: pd.DataFrame) -> pd.DataFrame:
    output = clock.copy()
    for column in ("decision_time_utc", "entry_time_utc", "exit_time_utc"):
        output[column] = pd.to_datetime(output[column], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    for column in FEATURE_COLUMNS:
        if column.endswith("eligible_categories"):
            output[column] = pd.to_numeric(output[column], errors="raise").astype(int)
    return output[CLOCK_COLUMNS]


def _frame_hash(clock: pd.DataFrame) -> str:
    records = []
    for raw in _format_clock(clock).to_dict(orient="records"):
        records.append(
            {
                key: None if isinstance(value, float) and math.isnan(value) else value
                for key, value in raw.items()
            }
        )
    return canonical_hash(records)


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    handle.close()
    return Path(handle.name)


def _write_clock(path: Path, clock: pd.DataFrame) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                _format_clock(clock).to_csv(
                    text,
                    index=False,
                    lineterminator="\n",
                    float_format="%.17g",
                )


def _publish_new(temporary: Path, final: Path) -> None:
    os.link(temporary, final)


def _same_inode(left: Path, right: Path) -> bool:
    try:
        left_stat = left.stat()
        right_stat = right.stat()
    except FileNotFoundError:
        return False
    return (left_stat.st_dev, left_stat.st_ino) == (
        right_stat.st_dev,
        right_stat.st_ino,
    )


def build_support_artifacts(cfg: Config) -> dict[str, Any]:
    registration = validate_frozen_preregistration(cfg.preregistration)
    _validate_config(cfg)
    source = load_source_rows(registration)
    features = build_report_features(source)
    primary = build_clock(features, mode="primary", clock="primary")
    controls = build_control_clocks(features, primary)
    support = support_gate_summary(primary)
    control_support = {
        name: support_gate_summary(
            controls[controls["clock"] == name], control_name=name
        )
        for name in CONTROL_NAMES
    }
    novelty, exposure = build_novelty_and_exposure(primary, controls, registration)
    overall_pass = bool(support["passed"] and novelty["passed"] and exposure["passed"])

    output = _repository_path(cfg.output)
    primary_path = _repository_path(cfg.primary_clock)
    control_path = _repository_path(cfg.control_clocks)
    output_tmp = _temporary_path(output)
    primary_tmp = _temporary_path(primary_path)
    control_tmp = _temporary_path(control_path)
    try:
        _write_clock(primary_tmp, primary)
        _write_clock(control_tmp, controls)
        outcome_boundary = {
            "source_values_read": True,
            "source_value_rows_read": int(len(source)),
            "source_feature_rows_derived": int(len(features)),
            "signal_incidence_rows_derived": int(len(primary)),
            "comparator_clock_rows_read": int(
                exposure["input_audit"]["flcc_rows_materialized"]
                + exposure["input_audit"]["tadi_rows_materialized"]
            ),
            "auction_panel_rows_read": int(
                exposure["input_audit"]["auction_calendar"]["panel_rows_materialized"]
            ),
            "auction_raw_rows_parsed": int(
                exposure["input_audit"]["auction_calendar"]["raw_rows_parsed"]
            ),
            **OUTCOME_BOUNDARY_ZERO_FIELDS,
        }
        core = {
            "protocol_version": PROTOCOL_VERSION,
            "policy_id": POLICY_ID,
            "config": asdict(cfg),
            "support_builder": {
                "path": str(SUPPORT_BUILDER),
                "sha256": sha256_file(SUPPORT_BUILDER),
            },
            "preregistration": {
                "path": str(DEFAULT_PREREGISTRATION),
                "sha256": EXPECTED_PREREGISTRATION_FILE_SHA256,
                "manifest_hash": EXPECTED_PREREGISTRATION_MANIFEST_HASH,
                "policy_hash": EXPECTED_POLICY_HASH,
            },
            "source": {
                "path": str(prereg.SOURCE_ROWS),
                "sha256": prereg.SOURCE_ROWS_SHA256,
                "rows_read": int(len(source)),
                "reports_read": int(source["record_date"].nunique()),
                "columns_materialized": SOURCE_MATERIALIZED_COLUMNS,
                "prohibited_signal_value_columns_materialized": [],
                "boundary_quarantine_feature_use": 0,
            },
            "feature_audit": {
                "report_features": int(len(features)),
                "category_rank_prior_reports": 60,
                "minimum_prior_non_null_prints": 12,
                "impulse_rank_prior_computable_reports": 126,
                "cash_rank_ready_reports": int(features["cash_rank126"].notna().sum()),
                "debt_rank_ready_reports": int(features["debt_rank126"].notna().sum()),
                "total_net_cash_rank_ready_reports": int(
                    features["total_net_cash_rank126"].notna().sum()
                ),
                "source_values_summarized": False,
            },
            "artifacts": {
                "primary_clock": {
                    "path": cfg.primary_clock,
                    "sha256": sha256_file(primary_tmp),
                    "frame_hash": _frame_hash(primary),
                    "rows": int(len(primary)),
                    "columns": CLOCK_COLUMNS,
                },
                "control_clocks": {
                    "path": cfg.control_clocks,
                    "sha256": sha256_file(control_tmp),
                    "frame_hash": _frame_hash(controls),
                    "rows": int(len(controls)),
                    "columns": CLOCK_COLUMNS,
                    "clock_counts": {
                        name: int((controls["clock"] == name).sum())
                        for name in CONTROL_NAMES
                    },
                },
            },
            "support_gates": support,
            "control_support_gates": control_support,
            "novelty_gates": novelty,
            "signed_occupied_exposure_gates": exposure,
            "all_source_only_gates_pass": overall_pass,
            "outcome_boundary": outcome_boundary,
            "performance_values_opened": False,
            "next_action": (
                "commit and hash-freeze the strict evaluator before outcomes"
                if overall_pass
                else "reject DFFB-601 without opening outcomes"
            ),
            "stopping_rule": "reject permanently without outcomes on any frozen support, novelty, or exposure failure; no repair",
        }
        artifact = {**core, "manifest_hash": canonical_hash(core)}
        output_tmp.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        published: list[tuple[Path, Path]] = []
        try:
            _publish_new(primary_tmp, primary_path)
            published.append((primary_tmp, primary_path))
            _publish_new(control_tmp, control_path)
            published.append((control_tmp, control_path))
            _publish_new(output_tmp, output)
            published.append((output_tmp, output))
        except BaseException:
            for temporary, final in reversed(published):
                if _same_inode(temporary, final):
                    final.unlink(missing_ok=True)
            raise
        return artifact
    finally:
        for path in (output_tmp, primary_tmp, control_tmp):
            path.unlink(missing_ok=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default=Config.preregistration)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--primary-clock", default=Config.primary_clock)
    parser.add_argument("--control-clocks", default=Config.control_clocks)
    parser.add_argument("--artifact-root", default=Config.artifact_root)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    print(
        json.dumps(
            build_support_artifacts(parse_args()),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
