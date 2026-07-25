"""Build the frozen outcome-blind LAMB-21 source/token support report."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, deque
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from training import preregister_liquidity_aware_microstructure_braid as prereg


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = prereg.POLICY_ID
PROTOCOL_VERSION = "lamb21_source_support_v1"
CONTRACT = "docs/lamb21-source-support-implementation-contract-2026-07-25.md"
CONTRACT_SHA256 = (
    "1500970e46db168a28d97ccd526d6ada441cbda9846f9539215997476ba3f63e"
)
RUNNER = "training/build_lamb21_source_support.py"
RUNNER_TEST = "tests/test_build_lamb21_source_support.py"
TOKEN_OUTPUT = Path("data/lamb21_source_support/token_support.csv.gz")
REPORT_OUTPUT = Path("results/lamb21_source_support_2026-07-25.json")

SOURCE_START = pd.Timestamp("2020-01-01T00:00:00Z")
SOURCE_END = pd.Timestamp("2024-01-01T00:00:00Z")
NEW_YORK = ZoneInfo("America/New_York")

PROTOCOL_BINDINGS = {
    prereg.AUDIT_DOCUMENT: prereg.AUDIT_DOCUMENT_SHA256,
    prereg.BOUNDARY_DOCUMENT: prereg.BOUNDARY_DOCUMENT_SHA256,
    prereg.PRODUCER_SCRIPT: (
        "5e65e0806628c08691591fe9179022c9b840fb6d399b25f168c7bfb4fdf69c36"
    ),
    "tests/test_preregister_liquidity_aware_microstructure_braid.py": (
        "557ed94e7863e279f23b2bc36445c173622b99a121f57f6301b888c98ba540bd"
    ),
    prereg.DEFAULT_OUTPUT: (
        "4ac8bf8f2d54120130c49a90f3d40a5cfaf141673525cb54df4b5333c01290e6"
    ),
    CONTRACT: CONTRACT_SHA256,
}
PROTOCOL_FILES = (*PROTOCOL_BINDINGS, RUNNER, RUNNER_TEST)

TOKEN_FIELDS = prereg.TOKEN_COLUMNS
TOKEN_VOCABULARY = {
    name: tuple(vocabulary) for name, vocabulary in prereg.TOKEN_SCHEMA
}
SAFETY_TOKENS = prereg.SAFETY_TOKENS
SAFETY_BY_COLUMN = dict(zip(TOKEN_FIELDS, SAFETY_TOKENS, strict=True))
CONTROL_IDS = prereg.CONTROL_IDS
MIXED_MACRO_TRANSITION = "MACRO_TRANSITION_MIXED"
MIXED_MICRO_TRANSITION = "MICRO_TRANSITION_MIXED"

FLAG_COLUMNS = (
    "source_observed",
    "source_complete",
    "source_gap_day",
    "verified_zero_volume_empty",
    "post_gap_quarantine",
)
RANK_PRIMITIVES = (
    "coarse_share",
    "coarse_coherence",
    "fine_conviction",
    "collision_share",
    "cascade_share",
    "cascade_coherence",
)
PRIMITIVE_COLUMNS = (
    "h41_delta",
    "rrp_amount_delta",
    "rrp_breadth_delta",
    "h41_age_days",
    "rrp_age_days",
    "coarse_flow",
    "fine_flow",
    *RANK_PRIMITIVES,
    "cascade_flow",
    "source_price_response",
)
BAND_COLUMNS = tuple(f"{name}_band" for name in RANK_PRIMITIVES)
TOKEN_COLUMNS = (
    "boundary_time",
    "core_source_valid",
    "rank_ready",
    "sequence_ready",
    *TOKEN_FIELDS,
)
REPLAY_COLUMNS = (
    "boundary_time",
    "core_source_valid",
    "rank_ready",
    "sequence_ready",
    "_h41_source_position",
    "_rrp_source_position",
    *PRIMITIVE_COLUMNS,
    *BAND_COLUMNS,
    *TOKEN_FIELDS,
)
APPEND_REPLAY_CUTOFFS = (
    "2020-06-30T23:59:59Z",
    "2020-12-31T23:59:59Z",
    "2021-06-30T23:59:59Z",
    "2021-12-31T23:59:59Z",
    "2022-06-30T23:59:59Z",
    "2022-12-31T23:59:59Z",
    "2023-06-30T23:59:59Z",
    "2023-12-31T23:59:59Z",
)
APPEND_CUTOFFS = tuple(pd.Timestamp(value) for value in APPEND_REPLAY_CUTOFFS)
GATE_ORDER = tuple(f"gate_{number:02d}" for number in range(1, 15))

COUNTERS = (
    "source_value_rows_decoded",
    "joint_state_rows_built",
    *(
        "execution_market_rows_opened",
        "funding_rows_opened",
        "future_return_rows_opened",
        "reward_rows_built",
        "model_rows_built",
        "trades_built",
        "pnl_values_computed",
        "cagr_values_computed",
        "mdd_values_computed",
        "post_2023_source_rows_opened",
    ),
)
FORBIDDEN_COUNTERS = (
    "execution_market_rows_opened",
    "funding_rows_opened",
    "future_return_rows_opened",
    "reward_rows_built",
    "model_rows_built",
    "trades_built",
    "pnl_values_computed",
    "cagr_values_computed",
    "mdd_values_computed",
    "post_2023_source_rows_opened",
)


@dataclass(frozen=True)
class SourceBundle:
    h41: pd.DataFrame
    rrp: pd.DataFrame
    lattice: pd.DataFrame
    cascade: pd.DataFrame


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if (
        str(path).startswith("~")
        or candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError("LAMB support path must be repository-relative")
    root = REPOSITORY_ROOT.resolve(strict=True)
    target = REPOSITORY_ROOT / candidate
    current = REPOSITORY_ROOT
    for part in candidate.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError("LAMB support path contains a symlink")
    try:
        target.resolve(strict=True).relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise RuntimeError("LAMB support path is missing or escapes repository") from error
    if not target.is_file():
        raise RuntimeError("LAMB support dependency is not a regular file")
    return target


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
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


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _git_check(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def protocol_hashes() -> dict[str, str]:
    return {path: sha256_file(path) for path in PROTOCOL_FILES}


def assert_clean_protocol_commit() -> dict[str, Any]:
    tracked = _git_check("ls-files", "--error-unmatch", "--", *PROTOCOL_FILES)
    if tracked.returncode:
        raise RuntimeError("LAMB support protocol is not fully committed")
    clean = _git_check("diff", "--quiet", "HEAD", "--", *PROTOCOL_FILES)
    if clean.returncode:
        raise RuntimeError("LAMB support protocol differs from HEAD")
    staged = _git_check("diff", "--cached", "--quiet")
    if staged.returncode:
        raise RuntimeError("LAMB support index differs from HEAD")
    head_result = _git_check("rev-parse", "HEAD")
    head = head_result.stdout.strip()
    if head_result.returncode or len(head) != 40:
        raise RuntimeError("LAMB support HEAD is unavailable")
    for path, expected in PROTOCOL_BINDINGS.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"LAMB support frozen protocol hash drift: {path}")
    for path in (RUNNER, RUNNER_TEST):
        latest = _git_output("log", "-1", "--format=%H", "--", path)
        if latest != head:
            raise RuntimeError(f"LAMB support producer was not committed at HEAD: {path}")
    return {"head": head, "hashes": protocol_hashes()}


def _exact_bool(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    text = series.astype("string")
    if not text.isin(("true", "false")).all():
        raise ValueError(f"{label} is not exact lowercase boolean")
    return text.eq("true")


def _exact_dates(series: pd.Series, label: str) -> pd.Series:
    text = series.astype("string")
    if not text.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError(f"{label} is not exact ISO date")
    parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="raise")
    return cast(pd.Series, parsed)


def _exact_timestamps(series: pd.Series, label: str) -> pd.Series:
    text = series.astype("string")
    suffix = text.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True)
    if not suffix.all():
        raise ValueError(f"{label} lacks an explicit timezone suffix")
    parsed = pd.to_datetime(text, utc=True, errors="raise")
    if parsed.isna().any():
        raise ValueError(f"{label} contains a missing timestamp")
    return cast(pd.Series, parsed)


def _exact_micro_timestamps(series: pd.Series, label: str) -> pd.Series:
    text = series.astype("string")
    if not text.str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}").all():
        raise ValueError(f"{label} is not exact canonical UTC grid text")
    parsed = pd.to_datetime(
        text,
        format="%Y-%m-%d %H:%M:%S",
        utc=True,
        errors="raise",
    )
    if parsed.isna().any():
        raise ValueError(f"{label} contains a missing timestamp")
    return cast(pd.Series, parsed)


def _strictly_increasing(series: pd.Series) -> bool:
    return bool(series.is_monotonic_increasing and not series.duplicated().any())


def _numeric(
    series: pd.Series,
    label: str,
    *,
    finite: bool = True,
) -> pd.Series:
    values = pd.to_numeric(series, errors="raise").astype(float)
    if finite and not np.isfinite(values.to_numpy(float)).all():
        raise ValueError(f"{label} contains non-finite values")
    return cast(pd.Series, values)


def expected_micro_index(
    start: pd.Timestamp = SOURCE_START,
    end: pd.Timestamp = SOURCE_END,
) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="5min", inclusive="left")


def validate_h41_source(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != prereg.H41_ALLOWLIST:
        raise ValueError("LAMB H.4.1 projection order drift")
    result = frame.copy()
    release = _exact_dates(_series(result, "release_date"), "H.4.1 release")
    observation = _exact_dates(
        _series(result, "observation_date"), "H.4.1 observation"
    )
    available = _exact_timestamps(
        _series(result, "available_at_utc"), "H.4.1 availability"
    )
    if not all(
        _strictly_increasing(values)
        for values in (release, observation, available)
    ):
        raise ValueError("LAMB H.4.1 clocks are duplicate or unordered")
    if not observation.lt(release).all():
        raise ValueError("LAMB H.4.1 observation is not strictly prior")
    if not available.dt.tz_convert(NEW_YORK).dt.strftime("%Y-%m-%d").eq(
        result["release_date"]
    ).all():
        raise ValueError("LAMB H.4.1 availability date mismatch")
    level = _numeric(
        _series(result, "net_liquidity_usd_millions"), "H.4.1 net liquidity"
    )
    if not level.gt(0.0).all():
        raise ValueError("LAMB H.4.1 net liquidity is not strictly positive")
    if release.ge(pd.Timestamp("2024-01-01")).any() or available.ge(
        SOURCE_END
    ).any():
        raise ValueError("LAMB H.4.1 contains 2024-or-later source")
    result["release_date"] = release
    result["observation_date"] = observation
    result["available_at_utc"] = available
    result["net_liquidity_usd_millions"] = level
    result["_source_position"] = np.arange(len(result), dtype=np.int64)
    result["h41_delta"] = level.diff()
    return result


def validate_rrp_source(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != prereg.RRP_ALLOWLIST:
        raise ValueError("LAMB ON RRP projection order drift")
    result = frame.copy()
    operation = _exact_dates(_series(result, "operation_date"), "RRP operation")
    available = _exact_timestamps(
        _series(result, "result_available_at_utc"), "RRP availability"
    )
    if not _strictly_increasing(operation) or not _strictly_increasing(available):
        raise ValueError("LAMB ON RRP clocks are duplicate or unordered")
    if operation.ge(pd.Timestamp("2024-01-01")).any() or available.ge(
        SOURCE_END
    ).any():
        raise ValueError("LAMB ON RRP contains 2024-or-later source")
    complete = _exact_bool(_series(result, "source_complete"), "RRP complete")
    quarantine = _series(result, "quarantine_reason").astype("string")
    amount_text = _series(result, "total_amount_accepted_usd").astype("string")
    participating_text = _series(result, "participating_counterparties").astype(
        "string"
    )
    accepted_text = _series(result, "accepted_counterparties").astype("string")
    if not quarantine.loc[complete].eq("").all():
        raise ValueError("LAMB complete RRP row has quarantine reason")
    if not quarantine.loc[~complete].ne("").all():
        raise ValueError("LAMB incomplete RRP row lacks quarantine reason")
    for values in (amount_text, participating_text, accepted_text):
        if not values.loc[~complete].eq("").all():
            raise ValueError("LAMB incomplete RRP row exposes a numeric value")

    amount = pd.Series(np.nan, index=result.index, dtype=float)
    participating = pd.Series(np.nan, index=result.index, dtype=float)
    accepted = pd.Series(np.nan, index=result.index, dtype=float)
    if complete.any():
        amount.loc[complete] = _numeric(
            amount_text.loc[complete], "RRP accepted amount"
        )
        participating.loc[complete] = _numeric(
            participating_text.loc[complete], "RRP participating counterparties"
        )
        accepted.loc[complete] = _numeric(
            accepted_text.loc[complete], "RRP accepted counterparties"
        )
    if (
        amount.loc[complete].lt(0.0).any()
        or participating.loc[complete].lt(0.0).any()
        or accepted.loc[complete].lt(0.0).any()
    ):
        raise ValueError("LAMB complete RRP numeric value is negative")
    integer_values = pd.concat(
        (participating.loc[complete], accepted.loc[complete])
    ).to_numpy(float)
    if not np.array_equal(integer_values, np.rint(integer_values)):
        raise ValueError("LAMB RRP counterparty count is fractional")
    if accepted.loc[complete].gt(participating.loc[complete]).any():
        raise ValueError("LAMB accepted RRP counterparties exceed participating")

    result["operation_date"] = operation
    result["result_available_at_utc"] = available
    result["total_amount_accepted_usd"] = amount
    result["participating_counterparties"] = participating
    result["accepted_counterparties"] = accepted
    result["source_complete"] = complete
    result["_source_position"] = np.arange(len(result), dtype=np.int64)
    previous_complete = complete.shift(1, fill_value=False)
    result["rrp_delta_valid"] = complete & previous_complete
    result["rrp_amount_delta"] = amount.diff().where(result["rrp_delta_valid"])
    result["rrp_breadth_delta"] = accepted.diff().where(
        result["rrp_delta_valid"]
    )
    return result


def _validate_micro_common(
    frame: pd.DataFrame,
    *,
    allowlist: Sequence[str],
    expected_index: pd.DatetimeIndex,
    label: str,
) -> tuple[pd.DataFrame, list[str]]:
    if tuple(frame.columns) != tuple(allowlist):
        raise ValueError(f"LAMB {label} projection order drift")
    result = frame.copy()
    date = _exact_micro_timestamps(_series(result, "date"), f"{label} date")
    if date.isna().any() or not _strictly_increasing(date):
        raise ValueError(f"LAMB {label} timestamps are duplicate or unordered")
    observed_index = pd.DatetimeIndex(date)
    if not observed_index.equals(expected_index):
        raise ValueError(f"LAMB {label} does not reproduce expected 5m grid")
    result["date"] = date
    for column in FLAG_COLUMNS:
        result[column] = _exact_bool(_series(result, column), f"{label} {column}")
    numeric_columns = [
        column for column in allowlist if column not in {"date", *FLAG_COLUMNS}
    ]
    numeric = result.loc[:, numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError(f"LAMB {label} contains non-finite projected values")
    result.loc[:, numeric_columns] = numeric

    observed = _series(result, "source_observed").astype(bool)
    empty = _series(result, "verified_zero_volume_empty").astype(bool)
    gap = _series(result, "source_gap_day").astype(bool)
    if (observed & empty).any():
        raise ValueError(f"LAMB {label} observed and empty flags overlap")
    base_complete = (observed | empty) & ~gap
    result["base_complete"] = base_complete
    expected_post = (
        (~base_complete)
        .shift(1, fill_value=False)
        .rolling(window=24, min_periods=1)
        .max()
        .astype(bool)
    )
    if not _series(result, "post_gap_quarantine").astype(bool).equals(
        expected_post
    ):
        raise ValueError(f"LAMB {label} post-gap quarantine does not replay")
    expected_complete = base_complete & ~expected_post
    if not _series(result, "source_complete").astype(bool).equals(
        expected_complete
    ):
        raise ValueError(f"LAMB {label} source-complete flag does not replay")
    if np.any(result.loc[~observed, numeric_columns].to_numpy(float) != 0.0):
        raise ValueError(f"LAMB {label} nonobserved row has nonzero numeric data")
    return result, numeric_columns


def validate_lattice_source(
    frame: pd.DataFrame,
    expected_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    expected = expected_micro_index() if expected_index is None else expected_index
    result, _ = _validate_micro_common(
        frame,
        allowlist=prereg.LATTICE_ALLOWLIST,
        expected_index=expected,
        label="quantity lattice",
    )
    integer_columns = (
        "agg_trade_count",
        "total_quantity_mbtc",
        "coarse_quantity_mbtc",
        "coarse_signed_quantity_mbtc",
        "fine_quantity_mbtc",
        "fine_signed_quantity_mbtc",
    )
    values = result.loc[:, list(integer_columns)].to_numpy(float)
    if not np.array_equal(values, np.rint(values)):
        raise ValueError("LAMB lattice integer primitive is fractional")
    unsigned = (
        "agg_trade_count",
        "total_quantity_mbtc",
        "coarse_quantity_mbtc",
        "fine_quantity_mbtc",
    )
    if (result.loc[:, list(unsigned)].to_numpy(float) < 0.0).any():
        raise ValueError("LAMB lattice unsigned primitive is negative")
    observed = _series(result, "source_observed").astype(bool)
    if not (
        _series(result.loc[observed], "agg_trade_count").astype(float).gt(0).all()
        and _series(result.loc[observed], "total_quantity_mbtc")
        .astype(float)
        .gt(0)
        .all()
    ):
        raise ValueError("LAMB observed lattice row lacks positive activity")
    total = _series(result, "total_quantity_mbtc").astype(float)
    coarse = _series(result, "coarse_quantity_mbtc").astype(float)
    coarse_signed = _series(result, "coarse_signed_quantity_mbtc").astype(float)
    fine = _series(result, "fine_quantity_mbtc").astype(float)
    fine_signed = _series(result, "fine_signed_quantity_mbtc").astype(float)
    if (
        coarse_signed.abs().gt(coarse).any()
        or fine_signed.abs().gt(fine).any()
        or (coarse + fine).gt(total).any()
    ):
        raise ValueError("LAMB lattice quantity identity is invalid")
    return result


def validate_cascade_source(
    frame: pd.DataFrame,
    expected_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    expected = expected_micro_index() if expected_index is None else expected_index
    result, _ = _validate_micro_common(
        frame,
        allowlist=prereg.CASCADE_ALLOWLIST,
        expected_index=expected,
        label="same-ms cascade",
    )
    integer_columns = (
        "first_transact_time_ms",
        "last_transact_time_ms",
        "agg_trade_count",
    )
    integers = result.loc[:, list(integer_columns)].to_numpy(float)
    if not np.array_equal(integers, np.rint(integers)):
        raise ValueError("LAMB cascade integer primitive is fractional")
    unsigned = (
        *integer_columns,
        "first_price",
        "last_price",
        "quote_notional",
        "collision_quote_notional",
        "max_ms_quote_notional",
    )
    if (result.loc[:, list(unsigned)].to_numpy(float) < 0.0).any():
        raise ValueError("LAMB cascade unsigned primitive is negative")
    observed = _series(result, "source_observed").astype(bool)
    required_positive = (
        "agg_trade_count",
        "first_price",
        "last_price",
        "quote_notional",
        "max_ms_quote_notional",
    )
    if (
        result.loc[observed, list(required_positive)]
        .astype(float)
        .le(0.0)
        .any()
        .any()
    ):
        raise ValueError("LAMB observed cascade row lacks positive activity")
    if observed.any():
        date_ms = (
            _series(result.loc[observed], "date").astype("int64") // 1_000_000
        )
        first_ms = _series(
            result.loc[observed], "first_transact_time_ms"
        ).astype(np.int64)
        last_ms = _series(result.loc[observed], "last_transact_time_ms").astype(
            np.int64
        )
        if not (
            first_ms.ge(date_ms).all()
            and last_ms.ge(first_ms).all()
            and last_ms.lt(date_ms + 300_000).all()
        ):
            raise ValueError("LAMB cascade transaction clock is outside its bar")
    quote = _series(result, "quote_notional").astype(float)
    collision = _series(result, "collision_quote_notional").astype(float)
    maximum = _series(result, "max_ms_quote_notional").astype(float)
    signed = _series(result, "max_ms_signed_quote_notional").astype(float)
    if (
        collision.gt(quote).any()
        or maximum.gt(quote).any()
        or signed.abs().gt(maximum).any()
    ):
        raise ValueError("LAMB cascade notional identity is invalid")
    return result


def validate_micro_pair(
    lattice: pd.DataFrame,
    cascade: pd.DataFrame,
) -> None:
    if not _series(lattice, "date").equals(_series(cascade, "date")):
        raise ValueError("LAMB micro source timestamps differ")
    for column in FLAG_COLUMNS:
        if not _series(lattice, column).astype(bool).equals(
            _series(cascade, column).astype(bool)
        ):
            raise ValueError(f"LAMB micro source flag mismatch: {column}")


def _gzip_mtime(path: str | Path) -> int:
    raw = _repository_path(path).read_bytes()[:10]
    if len(raw) != 10 or raw[:2] != b"\x1f\x8b":
        raise RuntimeError(f"LAMB support source is not gzip: {path}")
    return int.from_bytes(raw[4:8], "little")


def _physical_header_bytes(path: str | Path) -> bytes:
    with gzip.open(_repository_path(path), "rb") as handle:
        header = handle.readline()
    if not header.endswith(b"\n") or b"\r" in header or b"\x00" in header:
        raise RuntimeError(f"LAMB support physical header is malformed: {path}")
    return header


def verify_physical_source(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_header: Sequence[str],
    expected_header_sha256: str,
    allowlist: Sequence[str],
    manifest: str | None = None,
    manifest_sha256: str | None = None,
    implementation: str | None = None,
    implementation_sha256: str | None = None,
) -> Path:
    target = _repository_path(path)
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"LAMB support physical source hash drift: {path}")
    if _gzip_mtime(path) != 0:
        raise RuntimeError(f"LAMB support source gzip mtime is not zero: {path}")
    header_bytes = _physical_header_bytes(path)
    if hashlib.sha256(header_bytes).hexdigest() != expected_header_sha256:
        raise RuntimeError(f"LAMB support physical header hash drift: {path}")
    try:
        decoded = header_bytes.decode("utf-8").removesuffix("\n")
    except UnicodeDecodeError as error:
        raise RuntimeError("LAMB support physical header is not UTF-8") from error
    header = tuple(next(csv.reader([decoded])))
    if header != tuple(expected_header):
        raise RuntimeError(f"LAMB support physical header drift: {path}")
    if len(set(allowlist)) != len(allowlist):
        raise RuntimeError(f"LAMB support allowlist contains duplicates: {path}")
    if not set(allowlist).issubset(header):
        raise RuntimeError(f"LAMB support allowlist escapes physical header: {path}")
    for dependency, expected, label in (
        (manifest, manifest_sha256, "manifest"),
        (implementation, implementation_sha256, "implementation"),
    ):
        if (dependency is None) != (expected is None):
            raise RuntimeError(f"LAMB support incomplete {label} binding")
        if dependency is not None and sha256_file(dependency) != expected:
            raise RuntimeError(f"LAMB support {label} hash drift: {dependency}")
    return target


def load_exact_projection(
    path: str | Path,
    allowlist: Sequence[str],
    *,
    expected_header: Sequence[str],
    expected_sha256: str = "",
    expected_header_sha256: str = "",
    manifest: str | None = None,
    manifest_sha256: str | None = None,
    implementation: str | None = None,
    implementation_sha256: str | None = None,
) -> pd.DataFrame:
    verified = verify_physical_source(
        path,
        expected_sha256=expected_sha256,
        expected_header=expected_header,
        expected_header_sha256=expected_header_sha256,
        allowlist=allowlist,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        implementation=implementation,
        implementation_sha256=implementation_sha256,
    )
    source = verified if isinstance(verified, Path) else Path(path)
    frame = pd.read_csv(
        source,
        usecols=tuple(allowlist),
        dtype="string",
        keep_default_na=False,
        na_filter=False,
    )
    return cast(pd.DataFrame, frame.loc[:, list(allowlist)])


def validate_h41_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return validate_h41_source(frame)


def validate_rrp_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return validate_rrp_source(frame)


def validate_micro_source_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    require_full_grid: bool = True,
) -> pd.DataFrame:
    if require_full_grid:
        expected = expected_micro_index()
    else:
        if frame.empty:
            raise ValueError("LAMB partial micro source is empty")
        timestamps = _exact_micro_timestamps(
            _series(frame, "date"), f"{source} date"
        )
        expected = pd.date_range(
            timestamps.iloc[0],
            timestamps.iloc[-1] + pd.Timedelta(minutes=5),
            freq="5min",
            inclusive="left",
        )
    if source == "lattice":
        return validate_lattice_source(frame, expected)
    if source == "cascade":
        return validate_cascade_source(frame, expected)
    raise ValueError(f"unknown LAMB micro source: {source}")


def load_source_frames() -> SourceBundle:
    prereg.validate_frozen_dependencies()
    artifact = json.loads(
        _repository_path(prereg.DEFAULT_OUTPUT).read_text(encoding="utf-8")
    )
    prereg.validate_manifest(artifact)
    h41 = validate_h41_frame(
        load_exact_projection(
            prereg.H41_SOURCE,
            prereg.H41_ALLOWLIST,
            expected_header=prereg.H41_PHYSICAL_HEADER,
            expected_sha256=prereg.H41_SOURCE_SHA256,
            expected_header_sha256=prereg.H41_HEADER_SHA256,
            manifest=prereg.H41_MANIFEST,
            manifest_sha256=prereg.H41_MANIFEST_SHA256,
            implementation=prereg.H41_BUILDER,
            implementation_sha256=prereg.H41_BUILDER_SHA256,
        )
    )
    rrp = validate_rrp_frame(
        load_exact_projection(
            prereg.RRP_SOURCE,
            prereg.RRP_ALLOWLIST,
            expected_header=prereg.RRP_PHYSICAL_HEADER,
            expected_sha256=prereg.RRP_SOURCE_SHA256,
            expected_header_sha256=prereg.RRP_HEADER_SHA256,
            manifest=prereg.RRP_MANIFEST,
            manifest_sha256=prereg.RRP_MANIFEST_SHA256,
            implementation=prereg.RRP_BUILDER,
            implementation_sha256=prereg.RRP_BUILDER_SHA256,
        )
    )
    lattice = validate_micro_source_frame(
        load_exact_projection(
            prereg.LATTICE_SOURCE,
            prereg.LATTICE_ALLOWLIST,
            expected_header=prereg.LATTICE_PHYSICAL_HEADER,
            expected_sha256=prereg.LATTICE_SOURCE_SHA256,
            expected_header_sha256=prereg.LATTICE_HEADER_SHA256,
            manifest=prereg.LATTICE_MANIFEST,
            manifest_sha256=prereg.LATTICE_MANIFEST_SHA256,
            implementation=prereg.LATTICE_TRANSFORM,
            implementation_sha256=prereg.LATTICE_TRANSFORM_SHA256,
        ),
        source="lattice",
    )
    cascade = validate_micro_source_frame(
        load_exact_projection(
            prereg.CASCADE_SOURCE,
            prereg.CASCADE_ALLOWLIST,
            expected_header=prereg.CASCADE_PHYSICAL_HEADER,
            expected_sha256=prereg.CASCADE_SOURCE_SHA256,
            expected_header_sha256=prereg.CASCADE_HEADER_SHA256,
            manifest=prereg.CASCADE_MANIFEST,
            manifest_sha256=prereg.CASCADE_MANIFEST_SHA256,
            implementation=prereg.CASCADE_TRANSFORM,
            implementation_sha256=prereg.CASCADE_TRANSFORM_SHA256,
        ),
        source="cascade",
    )
    validate_micro_pair(lattice, cascade)
    return SourceBundle(h41=h41, rrp=rrp, lattice=lattice, cascade=cascade)


load_sources = load_source_frames


def _utc_timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def canonical_timestamp(value: Any) -> str:
    return _utc_timestamp(value).isoformat().replace("+00:00", "Z")


def canonical_boundaries(
    *,
    end_exclusive: pd.Timestamp = SOURCE_END,
) -> pd.DatetimeIndex:
    end = min(_utc_timestamp(end_exclusive), SOURCE_END)
    return pd.date_range(SOURCE_START, end, freq="8h", inclusive="left")


def _frame_with_time_index(frame: pd.DataFrame) -> pd.DataFrame:
    if not _series(frame, "date").is_monotonic_increasing:
        raise RuntimeError("LAMB micro frame is not time ordered")
    if _series(frame, "date").duplicated().any():
        raise RuntimeError("LAMB micro frame has duplicate timestamps")
    if isinstance(frame.index, pd.DatetimeIndex) and frame.index.equals(
        pd.DatetimeIndex(_series(frame, "date"))
    ):
        return frame
    return frame.set_index("date", drop=False, verify_integrity=True)


def _macro_h41_selection(
    frame: pd.DataFrame,
    boundary: pd.Timestamp,
    *,
    stale_offset: int,
) -> tuple[pd.Series | None, str | None]:
    available = _series(frame, "available_at_utc")
    asof = int(available.searchsorted(boundary, side="right")) - 1
    selected = asof - stale_offset
    if selected < 1:
        return None, "h41_predecessor_missing"
    row = cast(pd.Series, frame.iloc[selected])
    if not bool(np.isfinite(float(row["h41_delta"]))):
        return None, "h41_delta_invalid"
    age = (boundary - _utc_timestamp(row["available_at_utc"])).total_seconds()
    age_days = age / 86_400.0
    if age < 0:
        return None, "h41_future_selection"
    if stale_offset == 0 and age_days > prereg.Policy().h41_max_age_days:
        return None, "h41_stale"
    return row, None


def _macro_rrp_selection(
    frame: pd.DataFrame,
    boundary: pd.Timestamp,
    *,
    stale_offset: int,
) -> tuple[pd.Series | None, str | None]:
    available = _series(frame, "result_available_at_utc")
    primary = int(available.searchsorted(boundary, side="right")) - 1
    if primary < 0:
        return None, "rrp_missing"
    primary_row = cast(pd.Series, frame.iloc[primary])
    if not bool(primary_row["source_complete"]):
        return None, "rrp_latest_operation_quarantined"
    selected = primary - stale_offset
    if selected < 1:
        return None, "rrp_predecessor_missing"
    row = cast(pd.Series, frame.iloc[selected])
    if not bool(row["source_complete"]):
        return None, "rrp_stale_control_crossed_quarantine"
    if not bool(row["rrp_delta_valid"]):
        return None, "rrp_delta_segment_invalid"
    age = (
        boundary - _utc_timestamp(row["result_available_at_utc"])
    ).total_seconds()
    age_days = age / 86_400.0
    if age < 0:
        return None, "rrp_future_selection"
    if stale_offset == 0 and age_days > prereg.Policy().rrp_max_age_days:
        return None, "rrp_stale"
    return row, None


def _empty_boundary_state(
    boundary: pd.Timestamp,
    reasons: Sequence[str],
    *,
    lattice_rows: int = 0,
    cascade_rows: int = 0,
    h41_row: pd.Series | None = None,
    rrp_row: pd.Series | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "boundary_time": boundary,
        "core_source_valid": False,
        "invalid_reasons": "|".join(sorted(set(reasons))),
        "micro_rows_lattice": int(lattice_rows),
        "micro_rows_cascade": int(cascade_rows),
        "_h41_source_position": (
            int(h41_row["_source_position"]) if h41_row is not None else -1
        ),
        "_rrp_source_position": (
            int(rrp_row["_source_position"]) if rrp_row is not None else -1
        ),
        "h41_available_at_utc": (
            h41_row["available_at_utc"] if h41_row is not None else pd.NaT
        ),
        "rrp_available_at_utc": (
            rrp_row["result_available_at_utc"] if rrp_row is not None else pd.NaT
        ),
        "cascade_source_time": pd.NaT,
    }
    state.update({column: math.nan for column in PRIMITIVE_COLUMNS})
    return state


def build_boundary_state(
    boundary: pd.Timestamp,
    h41: pd.DataFrame,
    rrp: pd.DataFrame,
    lattice: pd.DataFrame,
    cascade: pd.DataFrame,
    *,
    h41_stale_offset: int = 0,
    rrp_stale_offset: int = 0,
) -> dict[str, Any]:
    boundary = _utc_timestamp(boundary)
    h41_row, h41_error = _macro_h41_selection(
        h41, boundary, stale_offset=h41_stale_offset
    )
    rrp_row, rrp_error = _macro_rrp_selection(
        rrp, boundary, stale_offset=rrp_stale_offset
    )
    start = boundary - pd.Timedelta(hours=8)
    lattice_indexed = _frame_with_time_index(lattice)
    cascade_indexed = _frame_with_time_index(cascade)
    end = boundary - pd.Timedelta(nanoseconds=1)
    lattice_window = lattice_indexed.loc[start:end]
    cascade_window = cascade_indexed.loc[start:end]
    reasons = [
        reason for reason in (h41_error, rrp_error) if reason is not None
    ]
    expected = pd.date_range(start, boundary, freq="5min", inclusive="left")
    if len(lattice_window) != 96:
        reasons.append("lattice_window_not_96")
    if len(cascade_window) != 96:
        reasons.append("cascade_window_not_96")
    if not pd.DatetimeIndex(lattice_window.index).equals(expected):
        reasons.append("lattice_window_grid")
    if not pd.DatetimeIndex(cascade_window.index).equals(expected):
        reasons.append("cascade_window_grid")
    for name, window in (
        ("lattice", lattice_window),
        ("cascade", cascade_window),
    ):
        if len(window) != 96:
            continue
        if not _series(window, "source_complete").astype(bool).all():
            reasons.append(f"{name}_source_incomplete")
        if _series(window, "source_gap_day").astype(bool).any():
            reasons.append(f"{name}_source_gap")
        if _series(window, "post_gap_quarantine").astype(bool).any():
            reasons.append(f"{name}_post_gap_quarantine")
        observed = _series(window, "source_observed").astype(bool)
        empty = _series(window, "verified_zero_volume_empty").astype(bool)
        if not (observed ^ empty).all():
            reasons.append(f"{name}_observation_identity")
    if len(lattice_window) == len(cascade_window) == 96:
        if not pd.DatetimeIndex(lattice_window.index).equals(
            pd.DatetimeIndex(cascade_window.index)
        ):
            reasons.append("micro_timestamp_mismatch")
    if reasons:
        return _empty_boundary_state(
            boundary,
            reasons,
            lattice_rows=len(lattice_window),
            cascade_rows=len(cascade_window),
            h41_row=h41_row,
            rrp_row=rrp_row,
        )

    coarse_unsigned = float(lattice_window["coarse_quantity_mbtc"].sum())
    fine_unsigned = float(lattice_window["fine_quantity_mbtc"].sum())
    total_quantity = float(lattice_window["total_quantity_mbtc"].sum())
    quote_notional = float(cascade_window["quote_notional"].sum())
    max_notional = float(cascade_window["max_ms_quote_notional"].sum())
    denominators = (
        coarse_unsigned,
        fine_unsigned,
        total_quantity,
        quote_notional,
        max_notional,
    )
    if not all(np.isfinite(value) and value > 0.0 for value in denominators):
        return _empty_boundary_state(
            boundary,
            ("nonpositive_micro_denominator",),
            lattice_rows=96,
            cascade_rows=96,
            h41_row=h41_row,
            rrp_row=rrp_row,
        )

    coarse_flow = float(lattice_window["coarse_signed_quantity_mbtc"].sum())
    fine_flow = float(lattice_window["fine_signed_quantity_mbtc"].sum())
    cascade_flow = float(
        cascade_window["max_ms_signed_quote_notional"].sum()
    )
    first_price = float(cascade_window["first_price"].iloc[0])
    last_price = float(cascade_window["last_price"].iloc[-1])
    if first_price <= 0.0 or last_price <= 0.0:
        return _empty_boundary_state(
            boundary,
            ("nonpositive_endpoint_price",),
            lattice_rows=96,
            cascade_rows=96,
            h41_row=h41_row,
            rrp_row=rrp_row,
        )
    primitives = {
        "h41_delta": float(cast(pd.Series, h41_row)["h41_delta"]),
        "rrp_amount_delta": float(
            cast(pd.Series, rrp_row)["rrp_amount_delta"]
        ),
        "rrp_breadth_delta": float(
            cast(pd.Series, rrp_row)["rrp_breadth_delta"]
        ),
        "h41_age_days": (
            boundary - _utc_timestamp(cast(pd.Series, h41_row)["available_at_utc"])
        ).total_seconds()
        / 86_400.0,
        "rrp_age_days": (
            boundary
            - _utc_timestamp(
                cast(pd.Series, rrp_row)["result_available_at_utc"]
            )
        ).total_seconds()
        / 86_400.0,
        "coarse_flow": coarse_flow,
        "fine_flow": fine_flow,
        "coarse_share": coarse_unsigned / total_quantity,
        "coarse_coherence": abs(coarse_flow) / coarse_unsigned,
        "fine_conviction": abs(fine_flow / fine_unsigned),
        "cascade_flow": cascade_flow,
        "collision_share": (
            float(cascade_window["collision_quote_notional"].sum())
            / quote_notional
        ),
        "cascade_share": max_notional / quote_notional,
        "cascade_coherence": (
            float(
                cascade_window["max_ms_signed_quote_notional"].abs().sum()
            )
            / max_notional
        ),
        "source_price_response": math.log(last_price / first_price),
    }
    if not all(np.isfinite(value) for value in primitives.values()):
        return _empty_boundary_state(
            boundary,
            ("nonfinite_primitive",),
            lattice_rows=96,
            cascade_rows=96,
            h41_row=h41_row,
            rrp_row=rrp_row,
        )
    if any(
        not 0.0 <= primitives[name] <= 1.0
        for name in RANK_PRIMITIVES
    ):
        return _empty_boundary_state(
            boundary,
            ("ranked_share_or_coherence_out_of_range",),
            lattice_rows=96,
            cascade_rows=96,
            h41_row=h41_row,
            rrp_row=rrp_row,
        )
    source_time = (
        cascade_window["_source_time"].iloc[-1]
        if "_source_time" in cascade_window
        else cascade_window["date"].iloc[-1]
    )
    return {
        "boundary_time": boundary,
        "core_source_valid": True,
        "invalid_reasons": "",
        "micro_rows_lattice": 96,
        "micro_rows_cascade": 96,
        "_h41_source_position": int(
            cast(pd.Series, h41_row)["_source_position"]
        ),
        "_rrp_source_position": int(
            cast(pd.Series, rrp_row)["_source_position"]
        ),
        "h41_available_at_utc": cast(pd.Series, h41_row)["available_at_utc"],
        "rrp_available_at_utc": cast(pd.Series, rrp_row)[
            "result_available_at_utc"
        ],
        "cascade_source_time": source_time,
        **primitives,
    }


def build_states(
    bundle: SourceBundle,
    *,
    boundaries: Sequence[pd.Timestamp] | None = None,
    end_exclusive: pd.Timestamp = SOURCE_END,
    h41_stale_offset: int = 0,
    rrp_stale_offset: int = 0,
) -> pd.DataFrame:
    state_boundaries = (
        canonical_boundaries(end_exclusive=end_exclusive)
        if boundaries is None
        else pd.DatetimeIndex([_utc_timestamp(value) for value in boundaries])
    )
    lattice = _frame_with_time_index(bundle.lattice)
    cascade = _frame_with_time_index(bundle.cascade)
    rows = [
        build_boundary_state(
            boundary,
            bundle.h41,
            bundle.rrp,
            lattice,
            cascade,
            h41_stale_offset=h41_stale_offset,
            rrp_stale_offset=rrp_stale_offset,
        )
        for boundary in state_boundaries
    ]
    return pd.DataFrame(rows)


def _strict_prior_band(
    current: float,
    history: Sequence[float],
) -> str:
    values = np.asarray(history, dtype=float)
    q33, q67 = np.quantile(values, (0.33, 0.67), method="linear")
    if current <= q33:
        return "LOW"
    if current <= q67:
        return "MID"
    return "HIGH"


def attach_strict_prior_ranks(
    states: pd.DataFrame,
    *,
    lookback: int = prereg.Policy().rank_history_max,
    minimum: int = prereg.Policy().rank_history_min,
) -> pd.DataFrame:
    if lookback < minimum or minimum <= 0:
        raise ValueError("LAMB rank history bounds are invalid")
    result = states.reset_index(drop=True).copy()
    histories = {
        name: deque(maxlen=lookback) for name in RANK_PRIMITIVES
    }
    bands: dict[str, list[str | None]] = {
        name: [] for name in RANK_PRIMITIVES
    }
    ready: list[bool] = []
    for _, row in result.iterrows():
        valid = bool(row["core_source_valid"])
        enough = valid and all(
            len(histories[name]) >= minimum for name in RANK_PRIMITIVES
        )
        ready.append(bool(enough))
        for name in RANK_PRIMITIVES:
            if enough:
                bands[name].append(
                    _strict_prior_band(float(row[name]), tuple(histories[name]))
                )
            else:
                bands[name].append(None)
        if valid:
            for name in RANK_PRIMITIVES:
                value = float(row[name])
                if not np.isfinite(value):
                    raise RuntimeError(
                        f"LAMB valid boundary has invalid primitive: {name}"
                    )
                histories[name].append(value)
    result["rank_ready"] = pd.Series(ready, dtype=bool)
    for name in RANK_PRIMITIVES:
        result[f"{name}_band"] = pd.Series(bands[name], dtype="string")
    return result


def _sign(value: float) -> int:
    return 1 if value > 0.0 else -1 if value < 0.0 else 0


def _primitive_tokens(row: pd.Series) -> dict[str, str]:
    h41 = (
        "H41_EXPANDS"
        if float(row["h41_delta"]) > 0.0
        else "H41_CONTRACTS"
        if float(row["h41_delta"]) < 0.0
        else "H41_FLAT"
    )
    amount = float(row["rrp_amount_delta"])
    breadth = float(row["rrp_breadth_delta"])
    rrp = (
        "RRP_RELEASES"
        if amount < 0.0 and breadth <= 0.0
        else "RRP_DRAINS"
        if amount > 0.0 and breadth >= 0.0
        else "RRP_FLAT"
    )
    if h41 == "H41_EXPANDS" and rrp == "RRP_RELEASES":
        sponsorship = "LIQUIDITY_SUPPORTS"
    elif h41 == "H41_CONTRACTS" and rrp == "RRP_DRAINS":
        sponsorship = "LIQUIDITY_RESTRICTS"
    elif h41 == "H41_FLAT" or rrp == "RRP_FLAT":
        sponsorship = "MACRO_NEUTRAL"
    else:
        sponsorship = "MACRO_SPLIT"
    h41_fresh = float(row["h41_age_days"]) <= prereg.Policy().h41_fresh_days
    rrp_fresh = float(row["rrp_age_days"]) <= prereg.Policy().rrp_fresh_days
    age = (
        "BOTH_FRESH"
        if h41_fresh and rrp_fresh
        else "H41_AGING"
        if not h41_fresh and rrp_fresh
        else "RRP_AGING"
        if h41_fresh and not rrp_fresh
        else "BOTH_AGING"
    )
    coarse_sign = _sign(float(row["coarse_flow"]))
    fine_sign = _sign(float(row["fine_flow"]))
    lattice_relation = {
        (1, 1): "COHORTS_BUY",
        (-1, -1): "COHORTS_SELL",
        (1, -1): "COARSE_BUY_FINE_SELL",
        (-1, 1): "COARSE_SELL_FINE_BUY",
    }.get((coarse_sign, fine_sign), "LATTICE_NEUTRAL")
    if (
        row["coarse_share_band"] == "HIGH"
        and row["coarse_coherence_band"] == "HIGH"
    ):
        concentration = "COARSE_DOMINANT"
    elif (
        row["coarse_share_band"] == "LOW"
        or row["fine_conviction_band"] == "HIGH"
    ):
        concentration = "FINE_DOMINANT"
    else:
        concentration = "LATTICE_MIXED"
    cascade_sign = _sign(float(row["cascade_flow"]))
    response = float(row["source_price_response"])
    impact = (
        "CASCADE_BUY_FOLLOWTHROUGH"
        if cascade_sign > 0 and response > 0.0
        else "CASCADE_BUY_ABSORBED"
        if cascade_sign > 0
        else "CASCADE_SELL_FOLLOWTHROUGH"
        if cascade_sign < 0 and response < 0.0
        else "CASCADE_SELL_ABSORBED"
        if cascade_sign < 0
        else "CASCADE_NEUTRAL"
    )
    if (
        row["collision_share_band"] == "HIGH"
        and row["cascade_share_band"] == "HIGH"
        and row["cascade_coherence_band"] == "HIGH"
    ):
        intensity = "CASCADE_LOCAL"
    elif (
        row["collision_share_band"] == "LOW"
        and row["cascade_share_band"] == "LOW"
    ):
        intensity = "CASCADE_BROAD"
    else:
        intensity = "CASCADE_MIXED"
    braid = {
        (1, 1): "MICRO_CONFIRMS_BUY",
        (-1, -1): "MICRO_CONFIRMS_SELL",
        (1, -1): "LATTICE_BUY_CASCADE_SELL",
        (-1, 1): "LATTICE_SELL_CASCADE_BUY",
    }.get((coarse_sign, cascade_sign), "MICRO_NEUTRAL")
    return {
        "h41_impulse": h41,
        "rrp_impulse": rrp,
        "macro_sponsorship": sponsorship,
        "macro_age": age,
        "lattice_relation": lattice_relation,
        "lattice_concentration": concentration,
        "cascade_impact": impact,
        "cascade_intensity": intensity,
        "micro_braid": braid,
    }


def _macro_transition(prior: str, current: str) -> str:
    if prior == current == "LIQUIDITY_SUPPORTS":
        return "SUPPORT_PERSISTS"
    if prior == current == "LIQUIDITY_RESTRICTS":
        return "RESTRICTION_PERSISTS"
    if current == "LIQUIDITY_SUPPORTS":
        return "ROTATES_TO_SUPPORT"
    if current == "LIQUIDITY_RESTRICTS":
        return "ROTATES_TO_RESTRICTION"
    return MIXED_MACRO_TRANSITION


def _micro_state(token: str) -> str:
    if token == "MICRO_CONFIRMS_BUY":
        return "BUY"
    if token == "MICRO_CONFIRMS_SELL":
        return "SELL"
    if token in {"LATTICE_BUY_CASCADE_SELL", "LATTICE_SELL_CASCADE_BUY"}:
        return "CONFLICT"
    return "NEUTRAL"


def _micro_transition(prior: str, current: str) -> str:
    prior_state = _micro_state(prior)
    current_state = _micro_state(current)
    if prior_state == current_state == "BUY":
        return "BUY_PRESSURE_PERSISTS"
    if prior_state == current_state == "SELL":
        return "SELL_PRESSURE_PERSISTS"
    if {prior_state, current_state} == {"BUY", "SELL"}:
        return "PRESSURE_FLIPS"
    if prior_state in {"BUY", "SELL"} and current_state in {
        "CONFLICT",
        "NEUTRAL",
    }:
        return "PRESSURE_DISSIPATES"
    return MIXED_MICRO_TRANSITION


def tokenize_states(
    ranked_states: pd.DataFrame,
    *,
    macro_mask: bool = False,
) -> pd.DataFrame:
    result = ranked_states.reset_index(drop=True).copy()
    token_rows: list[dict[str, str]] = []
    for position, row in result.iterrows():
        if not bool(row["rank_ready"]):
            tokens = dict(SAFETY_BY_COLUMN)
        else:
            tokens = _primitive_tokens(row)
            prior_ready = position > 0 and bool(
                result.loc[position - 1, "rank_ready"]
            )
            if prior_ready:
                prior = token_rows[position - 1]
                tokens["macro_transition"] = _macro_transition(
                    prior["macro_sponsorship"], tokens["macro_sponsorship"]
                )
                tokens["micro_transition"] = _micro_transition(
                    prior["micro_braid"], tokens["micro_braid"]
                )
            else:
                tokens["macro_transition"] = MIXED_MACRO_TRANSITION
                tokens["micro_transition"] = MIXED_MICRO_TRANSITION
        if macro_mask:
            tokens.update(
                {
                    "h41_impulse": "H41_FLAT",
                    "rrp_impulse": "RRP_FLAT",
                    "macro_sponsorship": "MACRO_NEUTRAL",
                    "macro_age": "BOTH_AGING",
                    "macro_transition": MIXED_MACRO_TRANSITION,
                }
            )
        token_rows.append(tokens)
    token_frame = pd.DataFrame(token_rows, columns=TOKEN_FIELDS)
    for name, vocabulary in TOKEN_VOCABULARY.items():
        valid_values = set(vocabulary)
        if name == "h41_impulse":
            valid_values.add("SOURCE_INVALID")
        if not token_frame[name].isin(valid_values).all():
            raise RuntimeError(f"LAMB emitted an invalid token: {name}")
    result.loc[:, list(TOKEN_FIELDS)] = token_frame
    result["sequence_ready"] = result["rank_ready"].astype(bool) & (
        np.arange(len(result)) >= prereg.Policy().sequence_lines - 1
    )
    return result


def serialize_token_line(row: Mapping[str, Any] | pd.Series) -> str:
    return "|".join(str(row[name]) for name in TOKEN_FIELDS)


def build_primary_token_frame(
    bundle: SourceBundle,
    *,
    boundaries: Sequence[pd.Timestamp] | None = None,
    end_exclusive: pd.Timestamp = SOURCE_END,
) -> pd.DataFrame:
    return tokenize_states(
        attach_strict_prior_ranks(
            build_states(
                bundle,
                boundaries=boundaries,
                end_exclusive=end_exclusive,
            )
        )
    )


def _swap_lattice_cohorts(frame: pd.DataFrame) -> pd.DataFrame:
    swapped = frame.copy()
    for left, right in (
        ("coarse_quantity_mbtc", "fine_quantity_mbtc"),
        ("coarse_signed_quantity_mbtc", "fine_signed_quantity_mbtc"),
    ):
        swapped[left] = frame[right].to_numpy(copy=True)
        swapped[right] = frame[left].to_numpy(copy=True)
    return swapped


def _delay_cascade_inside_month(
    frame: pd.DataFrame,
    *,
    rows: int = prereg.Policy().cascade_control_delay_rows,
) -> pd.DataFrame:
    if rows <= 0:
        raise ValueError("LAMB cascade control delay must be positive")
    delayed = frame.copy()
    delayed["_source_time"] = pd.Series(
        pd.NaT,
        index=delayed.index,
        dtype="datetime64[ns, UTC]",
    )
    value_columns = [
        column
        for column in frame.columns
        if column
        not in {
            "date",
            "base_complete",
            "_source_position",
            "_source_time",
        }
    ]
    numeric_columns = [
        column
        for column in value_columns
        if column not in FLAG_COLUMNS
    ]
    months = _series(frame, "date").dt.strftime("%Y-%m")
    for _, indexes in frame.groupby(months, sort=True).groups.items():
        positions = np.asarray(list(indexes), dtype=int)
        destination = positions[rows:]
        source = positions[:-rows]
        if len(destination):
            delayed.loc[destination, value_columns] = frame.loc[
                source, value_columns
            ].to_numpy()
            delayed.loc[destination, "_source_time"] = frame.loc[
                source, "date"
            ].to_numpy()
        invalid = positions[:rows]
        delayed.loc[invalid, FLAG_COLUMNS] = False
        delayed.loc[invalid, "source_gap_day"] = True
        delayed.loc[invalid, numeric_columns] = 0
    delayed["_source_time"] = pd.to_datetime(
        delayed["_source_time"], utc=True, errors="coerce"
    )
    return delayed


def build_control_token_frames(
    bundle: SourceBundle,
    primary: pd.DataFrame | None = None,
    *,
    boundaries: Sequence[pd.Timestamp] | None = None,
    end_exclusive: pd.Timestamp = SOURCE_END,
) -> OrderedDict[str, pd.DataFrame]:
    primary_frame = (
        build_primary_token_frame(
            bundle, boundaries=boundaries, end_exclusive=end_exclusive
        )
        if primary is None
        else primary
    )
    controls: OrderedDict[str, pd.DataFrame] = OrderedDict()
    controls["h41_stale_one_release"] = tokenize_states(
        attach_strict_prior_ranks(
            build_states(
                bundle,
                boundaries=boundaries,
                end_exclusive=end_exclusive,
                h41_stale_offset=1,
            )
        )
    )
    controls["rrp_stale_one_operation"] = tokenize_states(
        attach_strict_prior_ranks(
            build_states(
                bundle,
                boundaries=boundaries,
                end_exclusive=end_exclusive,
                rrp_stale_offset=1,
            )
        )
    )
    controls["lattice_cohort_swap"] = build_primary_token_frame(
        SourceBundle(
            h41=bundle.h41,
            rrp=bundle.rrp,
            lattice=_swap_lattice_cohorts(bundle.lattice),
            cascade=bundle.cascade,
        ),
        boundaries=boundaries,
        end_exclusive=end_exclusive,
    )
    controls["cascade_delay_37"] = build_primary_token_frame(
        SourceBundle(
            h41=bundle.h41,
            rrp=bundle.rrp,
            lattice=bundle.lattice,
            cascade=_delay_cascade_inside_month(bundle.cascade),
        ),
        boundaries=boundaries,
        end_exclusive=end_exclusive,
    )
    controls["macro_relation_mask"] = tokenize_states(
        primary_frame.loc[
            :,
            [
                column
                for column in primary_frame.columns
                if column not in TOKEN_FIELDS and column != "sequence_ready"
            ],
        ],
        macro_mask=True,
    )
    if tuple(controls) != CONTROL_IDS:
        raise RuntimeError("LAMB control order drift")
    return controls


def _prefix_bundle(bundle: SourceBundle, cutoff: pd.Timestamp) -> SourceBundle:
    cutoff = _utc_timestamp(cutoff)
    return SourceBundle(
        h41=bundle.h41.loc[
            _series(bundle.h41, "available_at_utc") <= cutoff
        ].reset_index(drop=True),
        rrp=bundle.rrp.loc[
            _series(bundle.rrp, "result_available_at_utc") <= cutoff
        ].reset_index(drop=True),
        lattice=bundle.lattice.loc[
            _series(bundle.lattice, "date") + pd.Timedelta(minutes=5) <= cutoff
        ].reset_index(drop=True),
        cascade=bundle.cascade.loc[
            _series(bundle.cascade, "date") + pd.Timedelta(minutes=5) <= cutoff
        ].reset_index(drop=True),
    )


def _prefix_boundaries(cutoff: pd.Timestamp) -> pd.DatetimeIndex:
    cutoff = min(_utc_timestamp(cutoff), SOURCE_END - pd.Timedelta(seconds=1))
    final_boundary = cutoff.floor("8h")
    return canonical_boundaries(
        end_exclusive=min(final_boundary + pd.Timedelta(hours=8), SOURCE_END)
    )


def rebuild_from_physical_prefixes(
    cutoff: pd.Timestamp,
    bundle: SourceBundle,
) -> pd.DataFrame:
    prefix = _prefix_bundle(bundle, cutoff)
    return build_primary_token_frame(
        prefix,
        boundaries=_prefix_boundaries(cutoff),
    )


def _canonical_cell(value: Any) -> Any:
    if value is None or value is pd.NA or (
        not isinstance(value, str) and bool(pd.isna(value))
    ):
        return None
    if isinstance(value, pd.Timestamp):
        return canonical_timestamp(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value).hex()
    return str(value)


def serialize_replay_frame(frame: pd.DataFrame) -> bytes:
    missing = [column for column in REPLAY_COLUMNS if column not in frame]
    if missing:
        raise RuntimeError(f"LAMB replay frame lacks columns: {missing}")
    records = [
        [_canonical_cell(row[column]) for column in REPLAY_COLUMNS]
        for _, row in frame.loc[:, list(REPLAY_COLUMNS)].iterrows()
    ]
    payload = {"columns": list(REPLAY_COLUMNS), "rows": records}
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def true_append_replay_audit(
    full: pd.DataFrame,
    bundle: SourceBundle,
    *,
    cutoffs: Sequence[pd.Timestamp] = APPEND_CUTOFFS,
) -> dict[str, Any]:
    details: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for cutoff_value in cutoffs:
        cutoff = _utc_timestamp(cutoff_value)
        rebuilt = rebuild_from_physical_prefixes(cutoff, bundle)
        expected = full.loc[
            _series(full, "boundary_time") <= cutoff
        ].reset_index(drop=True)
        expected_bytes = serialize_replay_frame(expected)
        rebuilt_bytes = serialize_replay_frame(rebuilt.reset_index(drop=True))
        key = canonical_timestamp(cutoff)
        details[key] = {
            "expected_rows": len(expected),
            "rebuilt_rows": len(rebuilt),
            "expected_sha256": hashlib.sha256(expected_bytes).hexdigest(),
            "rebuilt_sha256": hashlib.sha256(rebuilt_bytes).hexdigest(),
            "byte_identical": expected_bytes == rebuilt_bytes,
        }
    return {
        "cutoffs": details,
        "byte_identical": all(
            result["byte_identical"] for result in details.values()
        ),
        "compared_full_build_to_itself": False,
        "filtered_full_result_only": False,
    }


def _plain_token_csv_bytes(frame: pd.DataFrame) -> bytes:
    missing = [column for column in TOKEN_COLUMNS if column not in frame]
    if missing:
        raise RuntimeError(f"LAMB token frame lacks columns: {missing}")
    ordered = frame.sort_values("boundary_time", kind="stable")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(TOKEN_COLUMNS)
    for _, row in ordered.iterrows():
        values: list[Any] = []
        for column in TOKEN_COLUMNS:
            value = row[column]
            if column == "boundary_time":
                values.append(canonical_timestamp(value))
            elif column in {
                "core_source_valid",
                "rank_ready",
                "sequence_ready",
            }:
                values.append("true" if bool(value) else "false")
            else:
                values.append(str(value))
        writer.writerow(values)
    return output.getvalue().encode("utf-8")


def _control_support(
    primary: pd.DataFrame,
    controls: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    primary_bytes = _plain_token_csv_bytes(primary)
    primary_hash = hashlib.sha256(primary_bytes).hexdigest()
    streams: OrderedDict[str, dict[str, Any]] = OrderedDict()
    control_bytes: dict[str, bytes] = {}
    for control_id in CONTROL_IDS:
        frame = controls[control_id]
        payload = _plain_token_csv_bytes(frame)
        control_bytes[control_id] = payload
        streams[control_id] = {
            "rows": len(frame),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "differs_from_primary": payload != primary_bytes,
        }
    pairwise_distinct = len(set(control_bytes.values())) == len(control_bytes)
    return {
        "primary_sha256": primary_hash,
        "controls": streams,
        "all_differ_from_primary": all(
            detail["differs_from_primary"] for detail in streams.values()
        ),
        "controls_pairwise_distinct": pairwise_distinct,
    }


def _jensen_shannon(
    left: Mapping[str, int],
    right: Mapping[str, int],
    vocabulary: Sequence[str],
) -> float | None:
    left_total = sum(left.get(token, 0) for token in vocabulary)
    right_total = sum(right.get(token, 0) for token in vocabulary)
    if left_total <= 0 or right_total <= 0:
        return None
    p = np.asarray(
        [left.get(token, 0) / left_total for token in vocabulary], dtype=float
    )
    q = np.asarray(
        [right.get(token, 0) / right_total for token in vocabulary], dtype=float
    )
    midpoint = 0.5 * (p + q)

    def divergence(values: np.ndarray) -> float:
        mask = values > 0.0
        return float(np.sum(values[mask] * np.log2(values[mask] / midpoint[mask])))

    return 0.5 * divergence(p) + 0.5 * divergence(q)


def _yearly_token_support(primary: pd.DataFrame) -> dict[str, Any]:
    frame = primary.copy()
    timestamps = pd.to_datetime(frame["boundary_time"], utc=True)
    frame["_year"] = timestamps.dt.year
    eligible = frame.loc[
        frame["sequence_ready"].astype(bool)
        & frame["core_source_valid"].astype(bool)
    ]
    reports: dict[str, Any] = {}
    for year in range(2020, 2024):
        annual = frame.loc[frame["_year"] == year]
        annual_eligible = eligible.loc[eligible["_year"] == year]
        denominator = len(annual_eligible)
        counts = {
            name: Counter(annual_eligible[name].astype(str))
            for name in TOKEN_FIELDS
        }
        signatures = Counter(
            annual_eligible.apply(serialize_token_line, axis=1)
        )
        reports[str(year)] = {
            "nominal_boundaries": len(annual),
            "core_valid_boundaries": int(
                annual["core_source_valid"].astype(bool).sum()
            ),
            "core_valid_share": (
                float(annual["core_source_valid"].astype(bool).mean())
                if len(annual)
                else 0.0
            ),
            "sequence_ready_current_core_valid": denominator,
            "category_counts": {
                name: dict(sorted(value.items())) for name, value in counts.items()
            },
            "category_shares": {
                name: {
                    token: value.get(token, 0) / denominator
                    for token in TOKEN_VOCABULARY[name]
                }
                if denominator
                else {}
                for name, value in counts.items()
            },
            "liquidity_support_share": (
                counts["macro_sponsorship"].get("LIQUIDITY_SUPPORTS", 0)
                / denominator
                if denominator
                else 0.0
            ),
            "liquidity_restrict_share": (
                counts["macro_sponsorship"].get("LIQUIDITY_RESTRICTS", 0)
                / denominator
                if denominator
                else 0.0
            ),
            "micro_buy_share": (
                counts["micro_braid"].get("MICRO_CONFIRMS_BUY", 0)
                / denominator
                if denominator
                else 0.0
            ),
            "micro_sell_share": (
                counts["micro_braid"].get("MICRO_CONFIRMS_SELL", 0)
                / denominator
                if denominator
                else 0.0
            ),
            "cascade_followthrough_share": (
                sum(
                    count
                    for token, count in counts["cascade_impact"].items()
                    if token.endswith("FOLLOWTHROUGH")
                )
                / denominator
                if denominator
                else 0.0
            ),
            "cascade_absorption_share": (
                sum(
                    count
                    for token, count in counts["cascade_impact"].items()
                    if token.endswith("ABSORBED")
                )
                / denominator
                if denominator
                else 0.0
            ),
            "distinct_signatures": len(signatures),
            "max_signature_share": (
                max(signatures.values()) / denominator
                if denominator and signatures
                else 1.0
            ),
        }
    reports["adjacent_year_jsd"] = {}
    for left, right in (("2020", "2021"), ("2021", "2022"), ("2022", "2023")):
        reports["adjacent_year_jsd"][f"{left}_{right}"] = {
            name: _jensen_shannon(
                reports[left]["category_counts"][name],
                reports[right]["category_counts"][name],
                TOKEN_VOCABULARY[name],
            )
            for name in TOKEN_FIELDS
        }
    return reports


def _quarter_support(primary: pd.DataFrame) -> dict[str, Any]:
    timestamps = pd.to_datetime(primary["boundary_time"], utc=True)
    quarter = (
        timestamps.dt.year.astype(str)
        + "Q"
        + (((timestamps.dt.month - 1) // 3) + 1).astype(str)
    )
    result: dict[str, Any] = {}
    for label, group in primary.groupby(quarter, sort=True):
        key = str(label)
        if key == "2020Q1":
            continue
        result[key] = {
            "boundaries": len(group),
            "sequence_ready_current_core_valid": int(
                (
                    group["sequence_ready"].astype(bool)
                    & group["core_source_valid"].astype(bool)
                ).sum()
            ),
            "forced_flat_share": float(
                (~group["rank_ready"].astype(bool)).mean()
            ),
        }
    return result


def _annual_micro_join(bundle: SourceBundle) -> dict[str, float]:
    lattice_dates = pd.DatetimeIndex(bundle.lattice["date"])
    cascade_dates = pd.DatetimeIndex(bundle.cascade["date"])
    result: dict[str, float] = {}
    for year in range(2020, 2024):
        grid = pd.date_range(
            f"{year}-01-01T00:00:00Z",
            f"{year + 1}-01-01T00:00:00Z",
            freq="5min",
            inclusive="left",
        )
        joined = grid.intersection(lattice_dates).intersection(cascade_dates)
        result[str(year)] = len(joined) / len(grid)
    return result


def build_support_diagnostics(
    primary: pd.DataFrame,
    bundle: SourceBundle,
    controls: Mapping[str, pd.DataFrame],
    replay: Mapping[str, Any],
    counters: Mapping[str, int],
) -> OrderedDict[str, dict[str, Any]]:
    policy = prereg.Policy()
    annual = _yearly_token_support(primary)
    quarters = _quarter_support(primary)
    joins = _annual_micro_join(bundle)
    control = _control_support(primary, controls)

    gate_02 = all(value >= policy.source_join_min for value in joins.values())
    gate_03 = all(
        annual[str(year)]["core_valid_share"] >= policy.core_valid_min
        for year in range(2020, 2024)
    )
    annual_minimums = {"2020": 750, "2021": 1000, "2022": 1000, "2023": 1000}
    expected_quarters = {
        f"{year}Q{quarter}"
        for year in range(2020, 2024)
        for quarter in range(1, 5)
        if (year, quarter) != (2020, 1)
    }
    gate_04 = all(
        annual[year]["sequence_ready_current_core_valid"] >= minimum
        for year, minimum in annual_minimums.items()
    ) and set(quarters) == expected_quarters and all(
        detail["sequence_ready_current_core_valid"] >= 225
        for detail in quarters.values()
    )
    gate_05 = set(quarters) == expected_quarters and all(
        detail["forced_flat_share"] <= policy.forced_flat_max
        for detail in quarters.values()
    )
    gate_06_checks: dict[str, bool] = {}
    for year in map(str, range(2020, 2024)):
        for name in TOKEN_FIELDS:
            shares = annual[year]["category_shares"][name]
            supported = sum(
                share >= policy.category_support_min for share in shares.values()
            )
            gate_06_checks[f"{year}:{name}"] = (
                supported >= 2
                and bool(shares)
                and max(shares.values()) <= policy.category_share_max
            )
    gate_07_checks = {
        year: (
            annual[year]["liquidity_support_share"] >= 0.05
            and annual[year]["liquidity_restrict_share"] >= 0.05
        )
        for year in map(str, range(2020, 2024))
    }
    gate_08_checks = {
        year: (
            annual[year]["micro_buy_share"] >= 0.10
            and annual[year]["micro_sell_share"] >= 0.10
        )
        for year in map(str, range(2020, 2024))
    }
    gate_09_checks = {
        year: (
            annual[year]["cascade_followthrough_share"] >= 0.075
            and annual[year]["cascade_absorption_share"] >= 0.075
        )
        for year in map(str, range(2020, 2024))
    }
    gate_10_checks = {
        year: (
            annual[year]["distinct_signatures"]
            >= policy.distinct_signatures_min
            and annual[year]["max_signature_share"]
            <= policy.signature_share_max
        )
        for year in map(str, range(2020, 2024))
    }
    gate_11_checks = {
        f"{pair}:{name}": value is not None and value <= policy.jsd_max
        for pair, fields in annual["adjacent_year_jsd"].items()
        for name, value in fields.items()
    }
    gate_12 = (
        control["all_differ_from_primary"]
        and control["controls_pairwise_distinct"]
    )
    gate_14 = (
        set(counters) == set(COUNTERS)
        and all(counters[name] == 0 for name in FORBIDDEN_COUNTERS)
    )
    return OrderedDict(
        (
            (
                "gate_01",
                {
                    "name": "protocol_source_validation",
                    "passed": True,
                },
            ),
            (
                "gate_02",
                {
                    "name": "annual_micro_grid_join",
                    "passed": gate_02,
                    "annual_join_share": joins,
                },
            ),
            (
                "gate_03",
                {
                    "name": "annual_core_valid_share",
                    "passed": gate_03,
                    "annual": {
                        year: annual[year]["core_valid_share"]
                        for year in map(str, range(2020, 2024))
                    },
                },
            ),
            (
                "gate_04",
                {
                    "name": "sequence_ready_counts",
                    "passed": gate_04,
                    "annual": {
                        year: annual[year][
                            "sequence_ready_current_core_valid"
                        ]
                        for year in map(str, range(2020, 2024))
                    },
                    "quarters": {
                        name: value["sequence_ready_current_core_valid"]
                        for name, value in quarters.items()
                    },
                },
            ),
            (
                "gate_05",
                {
                    "name": "quarter_forced_flat_share",
                    "passed": gate_05,
                    "quarters": {
                        name: value["forced_flat_share"]
                        for name, value in quarters.items()
                    },
                },
            ),
            (
                "gate_06",
                {
                    "name": "annual_field_category_support",
                    "passed": all(gate_06_checks.values()),
                    "checks": gate_06_checks,
                },
            ),
            (
                "gate_07",
                {
                    "name": "annual_macro_direction_support",
                    "passed": all(gate_07_checks.values()),
                    "checks": gate_07_checks,
                },
            ),
            (
                "gate_08",
                {
                    "name": "annual_micro_direction_support",
                    "passed": all(gate_08_checks.values()),
                    "checks": gate_08_checks,
                },
            ),
            (
                "gate_09",
                {
                    "name": "annual_cascade_relation_support",
                    "passed": all(gate_09_checks.values()),
                    "checks": gate_09_checks,
                },
            ),
            (
                "gate_10",
                {
                    "name": "annual_signature_support",
                    "passed": all(gate_10_checks.values()),
                    "checks": gate_10_checks,
                },
            ),
            (
                "gate_11",
                {
                    "name": "adjacent_year_jsd",
                    "passed": all(gate_11_checks.values()),
                    "checks": gate_11_checks,
                },
            ),
            (
                "gate_12",
                {
                    "name": "control_distinctness",
                    "passed": gate_12,
                    "control": control,
                },
            ),
            (
                "gate_13",
                {
                    "name": "true_append_replay",
                    "passed": replay.get("byte_identical") is True,
                    "replay": replay,
                },
            ),
            (
                "gate_14",
                {
                    "name": "forbidden_counters_zero",
                    "passed": gate_14,
                    "counters": dict(counters),
                },
            ),
        )
    )


def evaluate_gates(
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    first_failed_gate: str | None = None
    first_failed_name: str | None = None
    evaluated: OrderedDict[str, bool | None] = OrderedDict()
    for gate_id in GATE_ORDER:
        if first_failed_gate is not None:
            evaluated[gate_id] = None
            continue
        evidence = diagnostics.get(gate_id)
        if isinstance(evidence, Mapping):
            passed = evidence.get("passed") is True
            name = str(evidence.get("name", gate_id))
        else:
            passed = evidence is True
            name = gate_id
        evaluated[gate_id] = passed
        if not passed:
            first_failed_gate = gate_id
            first_failed_name = name
    passed_all = first_failed_gate is None
    return {
        "gate_order": list(GATE_ORDER),
        "evaluated": evaluated,
        "first_failed_gate": first_failed_gate,
        "first_failed_name": first_failed_name,
        "decision": "pass" if passed_all else "fail",
        "authorized_next_stage": (
            "authorize_stage_0_5_reward_evaluator_freeze"
            if passed_all
            else None
        ),
        "failure_action": (
            None
            if passed_all
            else "retire_lamb21_unchanged_before_rewards"
        ),
        "gate_denominator": (
            "sequence_ready_current_core_valid_by_utc_year"
        ),
    }


def deterministic_token_gzip_bytes(frame: pd.DataFrame) -> bytes:
    csv_bytes = _plain_token_csv_bytes(frame)
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=output,
        mtime=0,
    ) as handle:
        handle.write(csv_bytes)
    return output.getvalue()


def deterministic_report_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _output_target(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if (
        str(path).startswith("~")
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError("LAMB support output path is unsafe")
    return REPOSITORY_ROOT / candidate


def _existing_payload_status(target: Path, payload: bytes) -> str | None:
    if target.is_symlink():
        raise RuntimeError(f"LAMB support write-once target is unsafe: {target}")
    if not target.exists():
        return None
    if not target.is_file():
        raise RuntimeError(f"LAMB support write-once target is unsafe: {target}")
    if target.read_bytes() != payload:
        raise RuntimeError(f"LAMB support write-once artifact drift: {target}")
    return "verified_existing"


def write_once_bytes(path: str | Path, payload: bytes) -> str:
    target = _output_target(path)
    existing = _existing_payload_status(target, payload)
    if existing is not None:
        return existing
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
            status = "created"
        except FileExistsError:
            status = cast(str, _existing_payload_status(target, payload))
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return status
    finally:
        temporary.unlink(missing_ok=True)


def _assert_outputs_compatible(
    token_bytes: bytes,
    report_bytes: bytes,
) -> None:
    _existing_payload_status(_output_target(TOKEN_OUTPUT), token_bytes)
    _existing_payload_status(_output_target(REPORT_OUTPUT), report_bytes)


def write_outputs(token_bytes: bytes, report_bytes: bytes) -> dict[str, str]:
    _assert_outputs_compatible(token_bytes, report_bytes)
    return {
        str(TOKEN_OUTPUT): write_once_bytes(TOKEN_OUTPUT, token_bytes),
        str(REPORT_OUTPUT): write_once_bytes(REPORT_OUTPUT, report_bytes),
    }


def _assert_forbidden_counters_zero(counters: Mapping[str, int]) -> None:
    if set(counters) != set(COUNTERS):
        raise RuntimeError("LAMB support counter schema drift")
    nonzero = {
        name: counters[name]
        for name in FORBIDDEN_COUNTERS
        if counters[name] != 0
    }
    if nonzero:
        raise RuntimeError(f"LAMB forbidden counter is nonzero: {nonzero}")


def finalize_outputs(
    primary: pd.DataFrame,
    report: Mapping[str, Any],
    counters: Mapping[str, int],
) -> dict[str, Any]:
    _assert_forbidden_counters_zero(counters)
    token_bytes = deterministic_token_gzip_bytes(primary)
    finalized = dict(report)
    finalized["evidence_counters"] = dict(counters)
    finalized["token_output"] = str(TOKEN_OUTPUT)
    finalized["token_output_sha256"] = hashlib.sha256(token_bytes).hexdigest()
    finalized["token_columns"] = list(TOKEN_COLUMNS)
    finalized["report_output"] = str(REPORT_OUTPUT)
    core = {key: value for key, value in finalized.items() if key != "manifest_hash"}
    finalized["manifest_hash"] = canonical_hash(core)
    report_bytes = deterministic_report_bytes(finalized)
    statuses = write_outputs(token_bytes, report_bytes)
    return {
        "report": finalized,
        "write_status": statuses,
        "token_sha256": finalized["token_output_sha256"],
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
    }


def _source_identity_report() -> dict[str, Any]:
    return {
        "h41": {
            "path": prereg.H41_SOURCE,
            "sha256": prereg.H41_SOURCE_SHA256,
            "header_sha256": prereg.H41_HEADER_SHA256,
            "manifest": prereg.H41_MANIFEST,
            "manifest_sha256": prereg.H41_MANIFEST_SHA256,
            "implementation": prereg.H41_BUILDER,
            "implementation_sha256": prereg.H41_BUILDER_SHA256,
            "projection": list(prereg.H41_ALLOWLIST),
        },
        "rrp": {
            "path": prereg.RRP_SOURCE,
            "sha256": prereg.RRP_SOURCE_SHA256,
            "header_sha256": prereg.RRP_HEADER_SHA256,
            "manifest": prereg.RRP_MANIFEST,
            "manifest_sha256": prereg.RRP_MANIFEST_SHA256,
            "implementation": prereg.RRP_BUILDER,
            "implementation_sha256": prereg.RRP_BUILDER_SHA256,
            "projection": list(prereg.RRP_ALLOWLIST),
        },
        "quantity_lattice": {
            "path": prereg.LATTICE_SOURCE,
            "sha256": prereg.LATTICE_SOURCE_SHA256,
            "header_sha256": prereg.LATTICE_HEADER_SHA256,
            "manifest": prereg.LATTICE_MANIFEST,
            "manifest_sha256": prereg.LATTICE_MANIFEST_SHA256,
            "implementation": prereg.LATTICE_TRANSFORM,
            "implementation_sha256": prereg.LATTICE_TRANSFORM_SHA256,
            "projection": list(prereg.LATTICE_ALLOWLIST),
        },
        "same_millisecond_cascade": {
            "path": prereg.CASCADE_SOURCE,
            "sha256": prereg.CASCADE_SOURCE_SHA256,
            "header_sha256": prereg.CASCADE_HEADER_SHA256,
            "manifest": prereg.CASCADE_MANIFEST,
            "manifest_sha256": prereg.CASCADE_MANIFEST_SHA256,
            "implementation": prereg.CASCADE_TRANSFORM,
            "implementation_sha256": prereg.CASCADE_TRANSFORM_SHA256,
            "projection": list(prereg.CASCADE_ALLOWLIST),
        },
    }


def build_real_support_payload() -> tuple[pd.DataFrame, dict[str, Any], dict[str, int]]:
    counters = {name: 0 for name in COUNTERS}
    protocol = assert_clean_protocol_commit()
    bundle = load_source_frames()
    counters["source_value_rows_decoded"] = sum(
        len(frame)
        for frame in (bundle.h41, bundle.rrp, bundle.lattice, bundle.cascade)
    )
    primary = build_primary_token_frame(bundle)
    controls = build_control_token_frames(bundle, primary)
    replay = true_append_replay_audit(primary, bundle)
    counters["joint_state_rows_built"] = (
        len(primary)
        + sum(len(frame) for frame in controls.values())
        + sum(
            int(detail["rebuilt_rows"])
            for detail in replay["cutoffs"].values()
        )
    )
    diagnostics = build_support_diagnostics(
        primary,
        bundle,
        controls,
        replay,
        counters,
    )
    evaluation = evaluate_gates(diagnostics)
    report: dict[str, Any] = {
        "policy_id": POLICY_ID,
        "protocol_version": PROTOCOL_VERSION,
        "head_commit": protocol["head"],
        "protocol_file_sha256": protocol["hashes"],
        "preregistration_manifest_hash": (
            "be035126c30f35c425563ee5b8d8d81c57b64c50dd072e8e7ae9b6acc1fd939e"
        ),
        "source_interval": {
            "start_inclusive": canonical_timestamp(SOURCE_START),
            "end_exclusive": canonical_timestamp(SOURCE_END),
        },
        "source_identities": _source_identity_report(),
        "state_rows": len(primary),
        "controls": list(CONTROL_IDS),
        "append_replay_cutoffs": list(APPEND_REPLAY_CUTOFFS),
        "diagnostics": diagnostics,
        "gate_evaluation": evaluation,
        "decision": evaluation["decision"],
        "authorized_next_stage": evaluation["authorized_next_stage"],
        "failure_action": evaluation["failure_action"],
        "outcome_boundary": (
            "source_support_only_no_reward_model_trade_or_market_outcome"
        ),
    }
    return primary, report, counters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token-output",
        default=str(TOKEN_OUTPUT),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--report-output",
        default=str(REPORT_OUTPUT),
        help=argparse.SUPPRESS,
    )
    arguments = parser.parse_args()
    if Path(arguments.token_output) != TOKEN_OUTPUT:
        raise RuntimeError(f"LAMB token output is frozen at {TOKEN_OUTPUT}")
    if Path(arguments.report_output) != REPORT_OUTPUT:
        raise RuntimeError(f"LAMB report output is frozen at {REPORT_OUTPUT}")
    primary, report, counters = build_real_support_payload()
    result = finalize_outputs(primary, report, counters)
    print(
        json.dumps(
            {
                "decision": result["report"]["decision"],
                "authorized_next_stage": result["report"][
                    "authorized_next_stage"
                ],
                "failure_action": result["report"]["failure_action"],
                "write_status": result["write_status"],
                "token_sha256": result["token_sha256"],
                "report_sha256": result["report_sha256"],
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
