"""Build outcome-blind SABLE-8 pre-2024 source and language support."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_sable8_symbolic_alpha_braid as prereg


PROTOCOL_VERSION = "sable8_symbolic_alpha_braid_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/build_sable8_symbolic_alpha_braid_support.py"
)
TEST_PATH = Path(
    "tests/test_build_sable8_symbolic_alpha_braid_support.py"
)
IMPLEMENTATION_CONTRACT = Path(
    "docs/sable8-source-support-implementation-contract-2026-07-25.md"
)
PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = (
    "e9a45b73a6c8140beef11fdc29668bcdbc719a5687f8c20e5da11c96519a4fd0"
)
PREREGISTRATION_MANIFEST_HASH = (
    "b74ac9a0053290519e3e9155f88cf5911592f474931b447ce5383f1416a3b762"
)

SOURCE_END = pd.Timestamp(prereg.SOURCE_END_EXCLUSIVE)
SOURCE_START = pd.Timestamp("2020-01-01T00:00:00Z")
PREFIX_END = pd.Timestamp("2023-01-01T00:00:00Z")
DEVELOPMENT_END = PREFIX_END
DEFAULT_TOKEN_OUTPUT = Path(
    "data/sable8_source_cuts/pre2024/token_support.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(prereg.SUPPORT_OUTPUT)
DEFAULT_CUT_MANIFEST_OUTPUT = Path(prereg.SOURCE_CUT_MANIFEST)

SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "market": {
        "source": prereg.MARKET_SOURCE,
        "source_sha256": prereg.MARKET_SOURCE_SHA256,
        "physical_header": prereg.MARKET_PHYSICAL_HEADER,
        "physical_header_sha256": prereg.MARKET_HEADER_SHA256,
        "allowlist": prereg.MARKET_ALLOWLIST,
        "timestamp_field": "date",
        "output": prereg.PRE2024_CUTS["market"],
    },
    "funding": {
        "source": prereg.FUNDING_SOURCE,
        "source_sha256": prereg.FUNDING_SOURCE_SHA256,
        "physical_header": prereg.FUNDING_PHYSICAL_HEADER,
        "physical_header_sha256": prereg.FUNDING_HEADER_SHA256,
        "allowlist": prereg.FUNDING_ALLOWLIST,
        "timestamp_field": "funding_time",
        "output": prereg.PRE2024_CUTS["funding"],
    },
    "premium": {
        "source": prereg.PREMIUM_SOURCE,
        "source_sha256": prereg.PREMIUM_SOURCE_SHA256,
        "physical_header": prereg.PREMIUM_PHYSICAL_HEADER,
        "physical_header_sha256": prereg.PREMIUM_HEADER_SHA256,
        "allowlist": prereg.PREMIUM_ALLOWLIST,
        "timestamp_field": "close_time",
        "output": prereg.PRE2024_CUTS["premium"],
    },
}

TOKEN_COLUMNS = (
    "boundary",
    "state_cutoff",
    "decision_time",
    "execution_time",
    "core_source_ready",
    "line_ready",
    "sequence_ready",
    "oi_fresh",
    "kimchi_fresh",
    "usdkrw_fresh",
    "dxy_fresh",
    *tuple(f"{name}_band" for name in prereg.PRIMITIVES),
    "canonical_line",
    "sequence_signature",
)
FORBIDDEN_SUPPORT_COLUMNS = {
    "position",
    "position_age",
    "strict_drawdown",
    "action",
    "target",
    "future_return",
    "return",
    "reward",
    "pnl",
    "cagr",
    "mdd",
    "label",
    "oracle_action",
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


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def assert_protocol_committed() -> None:
    paths = (
        str(SCRIPT_PATH),
        str(TEST_PATH),
        str(IMPLEMENTATION_CONTRACT),
        prereg.BOUNDARY_DOCUMENT,
        str(PREREGISTRATION),
        "training/preregister_sable8_symbolic_alpha_braid.py",
        "tests/test_preregister_sable8_symbolic_alpha_braid.py",
    )
    tracked = _git("ls-files", "--error-unmatch", "--", *paths)
    if tracked.returncode:
        raise RuntimeError("SABLE source-support protocol is not committed")
    clean = _git("diff", "--quiet", "HEAD", "--", *paths)
    if clean.returncode:
        raise RuntimeError("SABLE source-support protocol differs from HEAD")


def validate_preregistration() -> Mapping[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("SABLE preregistration artifact hash drift")
    payload = json.loads(_path(PREREGISTRATION).read_text(encoding="utf-8"))
    prereg.validate_manifest(payload)
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("SABLE preregistration manifest hash drift")
    if sha256_file(prereg.BOUNDARY_DOCUMENT) != (
        prereg.BOUNDARY_DOCUMENT_SHA256
    ):
        raise RuntimeError("SABLE boundary document hash drift")
    return payload


def _header_bytes(path: str | Path) -> bytes:
    source = _path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rb") as handle:
        header = handle.readline()
    if not header.endswith(b"\n") or b"\n" in header[:-1]:
        raise RuntimeError(f"SABLE source header is not one LF line: {path}")
    return header


def _parse_timestamp(raw: str, *, field: str) -> pd.Timestamp:
    value = str(raw).strip()
    if not value:
        raise RuntimeError(f"SABLE {field} timestamp is empty")
    if value.lstrip("-").isdigit():
        integer = int(value)
        magnitude = abs(integer)
        if magnitude >= 10**17:
            unit = "ns"
        elif magnitude >= 10**14:
            unit = "us"
        elif magnitude >= 10**11:
            unit = "ms"
        else:
            unit = "s"
        timestamp = pd.Timestamp(integer, unit=unit, tz="UTC")
    else:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
    if pd.isna(timestamp):
        raise RuntimeError(f"SABLE {field} timestamp is NaT")
    return timestamp


def _finite(raw: str, *, field: str) -> float:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"SABLE {field} is not numeric") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"SABLE {field} is not finite")
    return value


def _availability(raw: str, *, field: str) -> int:
    value = _finite(raw, field=field)
    if value not in (0.0, 1.0):
        raise RuntimeError(f"SABLE {field} availability is not binary")
    return int(value)


def _validate_projected_row(
    source_name: str,
    projected: Mapping[str, str],
) -> None:
    if source_name == "market":
        for field in ("open", "high", "low", "close"):
            if _finite(projected[field], field=field) <= 0.0:
                raise RuntimeError(f"SABLE market {field} must be positive")
        quote = _finite(
            projected["quote_asset_volume"],
            field="quote_asset_volume",
        )
        taker = _finite(
            projected["taker_buy_quote"],
            field="taker_buy_quote",
        )
        if quote < 0.0 or taker < 0.0 or taker > quote:
            raise RuntimeError("SABLE market quote/taker identity failed")
        if quote == 0.0 and taker != 0.0:
            raise RuntimeError(
                "SABLE zero-quote market bar has nonzero taker quote"
            )
        flags = {
            field: _availability(projected[field], field=field)
            for field in (
                "dxy_available",
                "kimchi_available",
                "usdkrw_available",
                "open_interest_available",
            )
        }
        guarded_values = (
            ("dxy", "dxy_available", True),
            ("kimchi_premium", "kimchi_available", False),
            ("usdkrw", "usdkrw_available", True),
            ("open_interest", "open_interest_available", True),
        )
        for value_field, flag_field, positive in guarded_values:
            if not flags[flag_field]:
                continue
            value = _finite(projected[value_field], field=value_field)
            if positive and value <= 0.0:
                raise RuntimeError(
                    f"SABLE fresh {value_field} must be positive"
                )
    elif source_name == "funding":
        _finite(projected["funding_rate"], field="funding_rate")
        _parse_timestamp(projected["date"], field="date")
    elif source_name == "premium":
        _finite(projected["close"], field="premium_close")
        _parse_timestamp(projected["date"], field="date")
    else:
        raise ValueError(f"unknown SABLE source: {source_name}")


def _deterministic_gzip_bytes(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=raw,
        mtime=0,
    ) as compressed:
        with io.TextIOWrapper(
            compressed,
            encoding="utf-8",
            newline="",
        ) as text:
            writer = csv.writer(text, lineterminator="\n")
            writer.writerow(list(header))
            writer.writerows(rows)
    return raw.getvalue()


def _write_bytes_once(path: str | Path, payload: bytes) -> str:
    target = _path(path)
    if target.exists():
        if target.read_bytes() != payload:
            raise RuntimeError(f"SABLE write-once artifact drift: {path}")
        return hashlib.sha256(payload).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=f".{target.name}.",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(target)
    return hashlib.sha256(payload).hexdigest()


def _write_json_once(path: str | Path, payload: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return _write_bytes_once(path, encoded)


def stream_project_source(
    source_name: str,
    spec: Mapping[str, Any],
    *,
    start: pd.Timestamp = SOURCE_START,
    cutoff: pd.Timestamp = SOURCE_END,
    output: str | Path | None = None,
    verify_full_hash: bool = True,
) -> dict[str, Any]:
    """Physically cut one source without converting post-cut or hidden cells."""

    source_path = _path(spec["source"])
    if verify_full_hash and sha256_file(source_path) != spec["source_sha256"]:
        raise RuntimeError(f"SABLE {source_name} full source hash drift")
    header_bytes = _header_bytes(source_path)
    if hashlib.sha256(header_bytes).hexdigest() != (
        spec["physical_header_sha256"]
    ):
        raise RuntimeError(f"SABLE {source_name} physical header hash drift")
    physical_header = tuple(
        header_bytes.decode("utf-8").rstrip("\n").split(",")
    )
    if physical_header != tuple(spec["physical_header"]):
        raise RuntimeError(f"SABLE {source_name} physical header/order drift")

    allowlist = tuple(spec["allowlist"])
    indices = [physical_header.index(field) for field in allowlist]
    timestamp_index = physical_header.index(spec["timestamp_field"])
    row_count = 0
    first_time: pd.Timestamp | None = None
    last_time: pd.Timestamp | None = None
    skipped_before_start = 0
    first_skipped_time: pd.Timestamp | None = None
    last_skipped_time: pd.Timestamp | None = None
    stopped_at: pd.Timestamp | None = None
    previous_time: pd.Timestamp | None = None

    raw_output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=raw_output,
        mtime=0,
    ) as compressed:
        with io.TextIOWrapper(
            compressed,
            encoding="utf-8",
            newline="",
        ) as text:
            writer = csv.writer(text, lineterminator="\n")
            writer.writerow(allowlist)
            opener = gzip.open if source_path.suffix == ".gz" else open
            with opener(
                source_path,
                "rt",
                encoding="utf-8",
                newline="",
            ) as handle:
                reader = csv.reader(handle)
                observed_header = tuple(next(reader))
                if observed_header != physical_header:
                    raise RuntimeError(
                        f"SABLE {source_name} header replay drift"
                    )
                for row_number, row in enumerate(reader, start=2):
                    if len(row) <= timestamp_index:
                        raise RuntimeError(
                            f"SABLE {source_name} row lacks timestamp at "
                            f"{row_number}"
                        )
                    timestamp = _parse_timestamp(
                        row[timestamp_index],
                        field=str(spec["timestamp_field"]),
                    )
                    if timestamp >= cutoff:
                        stopped_at = timestamp
                        break
                    if (
                        previous_time is not None
                        and timestamp <= previous_time
                    ):
                        raise RuntimeError(
                            f"SABLE {source_name} timestamps are not "
                            "unique/increasing"
                        )
                    previous_time = timestamp
                    if timestamp < start:
                        skipped_before_start += 1
                        first_skipped_time = (
                            timestamp
                            if first_skipped_time is None
                            else first_skipped_time
                        )
                        last_skipped_time = timestamp
                        continue
                    if len(row) != len(physical_header):
                        raise RuntimeError(
                            f"SABLE {source_name} row width drift at "
                            f"{row_number}"
                        )
                    projected_values = [row[index] for index in indices]
                    projected = dict(
                        zip(allowlist, projected_values, strict=True)
                    )
                    _validate_projected_row(source_name, projected)
                    writer.writerow(projected_values)
                    row_count += 1
                    first_time = (
                        timestamp if first_time is None else first_time
                    )
                    last_time = timestamp

    if not row_count:
        raise RuntimeError(f"SABLE {source_name} cut is empty")
    if stopped_at is None:
        raise RuntimeError(
            f"SABLE {source_name} source never reached frozen cutoff"
        )
    encoded = raw_output.getvalue()
    output_path = output if output is not None else spec["output"]
    output_sha256 = _write_bytes_once(output_path, encoded)
    return {
        "source_name": source_name,
        "source_path": str(spec["source"]),
        "source_sha256": str(spec["source_sha256"]),
        "physical_header_sha256": str(spec["physical_header_sha256"]),
        "cut_path": str(output_path),
        "cut_sha256": output_sha256,
        "cut_columns": list(allowlist),
        "rows": row_count,
        "first_timestamp": first_time.isoformat(),
        "last_timestamp": last_time.isoformat(),
        "source_start_inclusive": start.isoformat(),
        "skipped_before_start": skipped_before_start,
        "first_skipped_timestamp": (
            first_skipped_time.isoformat()
            if first_skipped_time is not None
            else None
        ),
        "last_skipped_timestamp": (
            last_skipped_time.isoformat()
            if last_skipped_time is not None
            else None
        ),
        "stopped_before_timestamp": stopped_at.isoformat(),
        "post_cut_non_timestamp_values_converted": 0,
        "unprojected_values_converted": 0,
    }


def _load_market(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(
        _path(path),
        compression="gzip",
        usecols=list(prereg.MARKET_ALLOWLIST),
    )
    if tuple(frame.columns) != prereg.MARKET_ALLOWLIST:
        raise RuntimeError("SABLE market cut column order drift")
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    for field in prereg.MARKET_ALLOWLIST[1:]:
        frame[field] = pd.to_numeric(frame[field], errors="raise")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise RuntimeError("SABLE market cut timestamps are not ordered unique")
    delta = frame["date"].diff().iloc[1:]
    if not delta.eq(pd.Timedelta(minutes=5)).all():
        raise RuntimeError("SABLE market cut is not a complete five-minute grid")
    return frame


def _load_funding(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(
        _path(path),
        compression="gzip",
        usecols=list(prereg.FUNDING_ALLOWLIST),
    )
    if tuple(frame.columns) != prereg.FUNDING_ALLOWLIST:
        raise RuntimeError("SABLE funding cut column order drift")
    frame["funding_time"] = pd.to_datetime(
        pd.to_numeric(frame["funding_time"], errors="raise"),
        unit="ms",
        utc=True,
    )
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    frame["funding_rate"] = pd.to_numeric(
        frame["funding_rate"],
        errors="raise",
    )
    if (
        frame["funding_time"].duplicated().any()
        or not frame["funding_time"].is_monotonic_increasing
    ):
        raise RuntimeError("SABLE funding timestamps are not ordered unique")
    return frame


def _load_premium(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(
        _path(path),
        compression="gzip",
        usecols=list(prereg.PREMIUM_ALLOWLIST),
    )
    if tuple(frame.columns) != prereg.PREMIUM_ALLOWLIST:
        raise RuntimeError("SABLE premium cut column order drift")
    frame["close_time"] = pd.to_datetime(
        pd.to_numeric(frame["close_time"], errors="raise"),
        unit="ms",
        utc=True,
    )
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    if (
        frame["close_time"].duplicated().any()
        or not frame["close_time"].is_monotonic_increasing
    ):
        raise RuntimeError("SABLE premium timestamps are not ordered unique")
    return frame


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or numerator <= 0.0
        or denominator <= 0.0
    ):
        return math.nan
    return math.log(numerator / denominator)


def _window_values(
    times_ns: np.ndarray,
    values: np.ndarray,
    *,
    lower_exclusive: pd.Timestamp,
    upper_inclusive: pd.Timestamp,
) -> np.ndarray:
    lower = np.datetime64(lower_exclusive.tz_convert(None), "ns")
    upper = np.datetime64(upper_inclusive.tz_convert(None), "ns")
    left = int(np.searchsorted(times_ns, lower, side="right"))
    right = int(np.searchsorted(times_ns, upper, side="right"))
    return values[left:right]


def build_primitive_frame(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    premium: pd.DataFrame,
    *,
    end: pd.Timestamp = SOURCE_END,
) -> pd.DataFrame:
    """Build only causal current primitives at canonical eight-hour clocks."""

    if set(FORBIDDEN_SUPPORT_COLUMNS).intersection(market.columns):
        raise RuntimeError("SABLE market source contains forbidden outcome fields")
    date = market["date"]
    close = market["close"].to_numpy(float)
    high = market["high"].to_numpy(float)
    low = market["low"].to_numpy(float)
    quote = market["quote_asset_volume"].to_numpy(float)
    taker = market["taker_buy_quote"].to_numpy(float)
    aggressor = 2.0 * taker - quote
    returns = np.full(len(market), np.nan, dtype=float)
    returns[1:] = np.log(close[1:] / close[:-1])

    series_r = pd.Series(returns)
    rv_1d = series_r.pow(2).rolling(288, min_periods=288).sum().to_numpy()
    rv_30d = (
        series_r.pow(2).rolling(8_640, min_periods=8_640).sum().to_numpy()
    )
    bv_term = series_r.abs() * series_r.shift(1).abs()
    bv_1d = (
        (math.pi / 2.0)
        * bv_term.rolling(288, min_periods=288).sum().to_numpy()
    )
    cubic_1d = (
        series_r.pow(3).rolling(288, min_periods=288).sum().to_numpy()
    )
    rolling_low = (
        pd.Series(close).rolling(2_016, min_periods=2_016).min().to_numpy()
    )
    rolling_high = (
        pd.Series(close).rolling(2_016, min_periods=2_016).max().to_numpy()
    )
    path_6h = (
        series_r.abs().rolling(72, min_periods=72).sum().to_numpy()
    )
    imbalance = np.divide(
        aggressor,
        quote,
        out=np.full(len(quote), np.nan, dtype=float),
        where=quote > 0.0,
    )
    recovery = (
        pd.Series(imbalance).rolling(12, min_periods=12).mean()
        - pd.Series(imbalance).rolling(72, min_periods=72).mean()
    ).to_numpy()
    quote_prefix = np.concatenate(([0.0], np.cumsum(quote)))
    aggressor_prefix = np.concatenate(([0.0], np.cumsum(aggressor)))

    market_positions = {
        int(timestamp.value): index
        for index, timestamp in enumerate(date)
    }
    funding_times = funding["funding_time"].to_numpy(dtype="datetime64[ns]")
    funding_values = funding["funding_rate"].to_numpy(float)
    premium_times = premium["close_time"].to_numpy(dtype="datetime64[ns]")
    premium_values = premium["close"].to_numpy(float)

    first_boundary = max(
        pd.Timestamp("2020-01-01T00:00:00Z"),
        date.iloc[0].ceil("8h"),
    )
    last_boundary = min(
        end - pd.Timedelta(hours=8),
        date.iloc[-1].floor("8h"),
    )
    boundaries = pd.date_range(
        first_boundary,
        last_boundary,
        freq="8h",
        tz="UTC",
    )
    rows: list[dict[str, Any]] = []

    for boundary in boundaries:
        cutoff = boundary + pd.Timedelta(minutes=5)
        position = market_positions.get(int(boundary.value))
        record: dict[str, Any] = {
            "boundary": boundary,
            "state_cutoff": cutoff,
            "decision_time": boundary + pd.Timedelta(minutes=10),
            "execution_time": boundary + pd.Timedelta(minutes=15),
        }
        for primitive in prereg.PRIMITIVES:
            record[primitive] = math.nan
        for context in ("oi", "kimchi", "usdkrw", "dxy"):
            record[f"{context}_fresh"] = False
        record["otherwise_eligible"] = False

        if position is None or position < 8_640:
            rows.append(record)
            continue
        t = position
        record["otherwise_eligible"] = True
        record["price_return_1d"] = _safe_log_ratio(
            close[t],
            close[t - 288],
        )
        denominator = rolling_high[t] - rolling_low[t]
        record["range_location_7d"] = (
            0.5
            if denominator == 0.0
            else (close[t] - rolling_low[t]) / denominator
        )
        if rv_1d[t] > 0.0 and math.isfinite(rv_30d[t]):
            record["volatility_ratio_1d_30d"] = math.log(
                (rv_1d[t] + 1e-18) / (rv_30d[t] / 30.0 + 1e-18)
            )
            record["jump_share_1d"] = (
                max(rv_1d[t] - bv_1d[t], 0.0) / rv_1d[t]
            )
            record["signed_jump_1d"] = (
                cubic_1d[t] / rv_1d[t] ** 1.5
            )

        prior_quote = quote_prefix[t] - quote_prefix[t - 288]
        target = 0.25 * prior_quote
        threshold = quote_prefix[t + 1] - target
        j = int(np.searchsorted(quote_prefix[: t + 1], threshold, side="right") - 1)
        if j >= 0:
            interval_quote = quote_prefix[t + 1] - quote_prefix[j]
            interval_aggressor = aggressor_prefix[t + 1] - aggressor_prefix[j]
            duration = t - j + 1
            if interval_quote > 0.0 and duration > 0:
                record["volume_clock_flow_speed_25"] = (
                    interval_aggressor / interval_quote
                ) / duration

        if path_6h[t] > 0.0:
            record["liquidity_signed_efficiency_6h"] = (
                _safe_log_ratio(close[t], close[t - 72]) / path_6h[t]
            )
        record["taker_flow_recovery_1h_6h"] = recovery[t]

        funding_window = _window_values(
            funding_times,
            funding_values,
            lower_exclusive=cutoff - pd.Timedelta(hours=24),
            upper_inclusive=cutoff,
        )
        if len(funding_window) == 3 and np.isfinite(funding_window).all():
            record["funding_sum_24h"] = float(funding_window.sum())

        premium_window = _window_values(
            premium_times,
            premium_values,
            lower_exclusive=cutoff - pd.Timedelta(hours=8),
            upper_inclusive=cutoff,
        )
        if len(premium_window) == 8 and np.isfinite(premium_window).all():
            record["premium_mean_8h"] = float(premium_window.mean())

        availability_specs = (
            (
                "oi",
                "open_interest",
                "open_interest_available",
                288,
            ),
            ("kimchi", "kimchi_premium", "kimchi_available", 144),
            ("usdkrw", "usdkrw", "usdkrw_available", 144),
            ("dxy", "dxy", "dxy_available", 288),
        )
        for context, value_field, flag_field, lag in availability_specs:
            current_flag = float(market.iloc[t][flag_field]) > 0.5
            lag_flag = float(market.iloc[t - lag][flag_field]) > 0.5
            current_value = float(market.iloc[t][value_field])
            lag_value = float(market.iloc[t - lag][value_field])
            fresh = (
                current_flag
                and lag_flag
                and math.isfinite(current_value)
                and math.isfinite(lag_value)
            )
            record[f"{context}_fresh"] = fresh
            if not fresh:
                continue
            if context == "oi":
                record["oi_price_divergence_1d"] = (
                    _safe_log_ratio(current_value, lag_value)
                    - record["price_return_1d"]
                )
            elif context == "kimchi":
                record["kimchi_change_12h"] = current_value - lag_value
            elif context == "usdkrw":
                record["usdkrw_change_12h"] = _safe_log_ratio(
                    current_value,
                    lag_value,
                )
            elif context == "dxy":
                record["dxy_change_1d"] = _safe_log_ratio(
                    current_value,
                    lag_value,
                )
        rows.append(record)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("SABLE primitive frame is empty")
    return frame


def build_token_table(primitives: pd.DataFrame) -> pd.DataFrame:
    histories: dict[str, list[float]] = {
        primitive: [] for primitive in prereg.PRIMITIVES
    }
    previous_lines: list[tuple[pd.Timestamp, str]] = []
    rows: list[dict[str, Any]] = []

    for record in primitives.to_dict(orient="records"):
        values = {
            primitive: float(record[primitive])
            for primitive in prereg.PRIMITIVES
        }
        core_source_ready = all(
            math.isfinite(values[primitive])
            for primitive in prereg.CORE_PRIMITIVES
        )
        bands: dict[str, str] = {
            primitive: "" for primitive in prereg.PRIMITIVES
        }
        line_ready = False
        line = ""
        if core_source_ready:
            core_rank_ready = all(
                len(histories[primitive]) >= prereg.Policy().rank_history_min
                for primitive in prereg.CORE_PRIMITIVES
            )
            if core_rank_ready:
                for primitive in prereg.CORE_PRIMITIVES:
                    rank = prereg.strict_prior_midrank(
                        values[primitive],
                        histories[primitive],
                    )
                    bands[primitive] = prereg.rank_band(rank)
                context_fresh = {
                    "oi_price_divergence_1d": bool(record["oi_fresh"]),
                    "kimchi_change_12h": bool(record["kimchi_fresh"]),
                    "usdkrw_change_12h": bool(record["usdkrw_fresh"]),
                    "dxy_change_1d": bool(record["dxy_fresh"]),
                }
                for primitive in prereg.CONTEXT_PRIMITIVES:
                    ready = (
                        context_fresh[primitive]
                        and math.isfinite(values[primitive])
                        and len(histories[primitive])
                        >= prereg.Policy().rank_history_min
                    )
                    if ready:
                        rank = prereg.strict_prior_midrank(
                            values[primitive],
                            histories[primitive],
                        )
                        bands[primitive] = prereg.rank_band(rank)
                    else:
                        bands[primitive] = "STALE"
                line = prereg.canonical_line(bands)
                line_ready = True

            for primitive in prereg.CORE_PRIMITIVES:
                histories[primitive].append(values[primitive])
                if len(histories[primitive]) > prereg.Policy().rank_history_max:
                    histories[primitive].pop(0)
            context_fresh = {
                "oi_price_divergence_1d": bool(record["oi_fresh"]),
                "kimchi_change_12h": bool(record["kimchi_fresh"]),
                "usdkrw_change_12h": bool(record["usdkrw_fresh"]),
                "dxy_change_1d": bool(record["dxy_fresh"]),
            }
            for primitive in prereg.CONTEXT_PRIMITIVES:
                if context_fresh[primitive] and math.isfinite(values[primitive]):
                    histories[primitive].append(values[primitive])
                    if len(histories[primitive]) > (
                        prereg.Policy().rank_history_max
                    ):
                        histories[primitive].pop(0)

        boundary = pd.Timestamp(record["boundary"])
        if line_ready:
            if (
                previous_lines
                and boundary - previous_lines[-1][0]
                != pd.Timedelta(hours=8)
            ):
                previous_lines = []
            previous_lines.append((boundary, line))
            previous_lines = previous_lines[-prereg.Policy().sequence_lines :]
        else:
            previous_lines = []
        sequence_ready = False
        signature = ""
        if len(previous_lines) == prereg.Policy().sequence_lines:
            times = [int(timestamp.timestamp()) for timestamp, _ in previous_lines]
            lines = [value for _, value in previous_lines]
            signature = prereg.sequence_signature(times, lines)
            sequence_ready = True

        output: dict[str, Any] = {
            "boundary": boundary,
            "state_cutoff": pd.Timestamp(record["state_cutoff"]),
            "decision_time": pd.Timestamp(record["decision_time"]),
            "execution_time": pd.Timestamp(record["execution_time"]),
            "core_source_ready": core_source_ready,
            "line_ready": line_ready,
            "sequence_ready": sequence_ready,
            "oi_fresh": bool(record["oi_fresh"]),
            "kimchi_fresh": bool(record["kimchi_fresh"]),
            "usdkrw_fresh": bool(record["usdkrw_fresh"]),
            "dxy_fresh": bool(record["dxy_fresh"]),
            **{
                f"{primitive}_band": bands[primitive]
                for primitive in prereg.PRIMITIVES
            },
            "canonical_line": line,
            "sequence_signature": signature,
        }
        rows.append(output)

    frame = pd.DataFrame(rows, columns=TOKEN_COLUMNS)
    if set(frame.columns).intersection(FORBIDDEN_SUPPORT_COLUMNS):
        raise RuntimeError("SABLE token support contains forbidden columns")
    return frame


def _iso(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def token_table_bytes(frame: pd.DataFrame) -> bytes:
    if tuple(frame.columns) != TOKEN_COLUMNS:
        raise RuntimeError("SABLE token output schema/order drift")
    rows: list[list[str]] = []
    for record in frame.to_dict(orient="records"):
        row: list[str] = []
        for column in TOKEN_COLUMNS:
            value = record[column]
            if column in (
                "boundary",
                "state_cutoff",
                "decision_time",
                "execution_time",
            ):
                row.append(_iso(value))
            elif column in (
                "core_source_ready",
                "line_ready",
                "sequence_ready",
                "oi_fresh",
                "kimchi_fresh",
                "usdkrw_fresh",
                "dxy_fresh",
            ):
                row.append("1" if bool(value) else "0")
            else:
                row.append(str(value))
        rows.append(row)
    return _deterministic_gzip_bytes(TOKEN_COLUMNS, rows)


def prefix_replay_audit(
    full_tokens: pd.DataFrame,
    *,
    source_specs: Mapping[str, Mapping[str, Any]] = SOURCE_SPECS,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sable8-prefix-replay-") as raw:
        root = Path(raw)
        prefix_cuts = {
            name: stream_project_source(
                name,
                spec,
                start=SOURCE_START,
                cutoff=PREFIX_END,
                output=root / f"{name}.csv.gz",
            )
            for name, spec in source_specs.items()
        }
        prefix_market = _load_market(prefix_cuts["market"]["cut_path"])
        prefix_funding = _load_funding(prefix_cuts["funding"]["cut_path"])
        prefix_premium = _load_premium(prefix_cuts["premium"]["cut_path"])
        prefix_primitives = build_primitive_frame(
            prefix_market,
            prefix_funding,
            prefix_premium,
            end=PREFIX_END,
        )
        prefix_tokens = build_token_table(prefix_primitives)
    full_prefix = full_tokens.loc[
        full_tokens["boundary"] < PREFIX_END
    ].reset_index(drop=True)
    compare_columns = list(TOKEN_COLUMNS)
    left = token_table_bytes(prefix_tokens.loc[:, compare_columns])
    right = token_table_bytes(full_prefix.loc[:, compare_columns])
    return {
        "cutoff": _iso(PREFIX_END),
        "rows": len(prefix_tokens),
        "prefix_sha256": hashlib.sha256(left).hexdigest(),
        "full_prefix_sha256": hashlib.sha256(right).hexdigest(),
        "passed": left == right,
        "physical_prefix_cuts": {
            name: {
                "sha256": audit["cut_sha256"],
                "rows": audit["rows"],
                "first_timestamp": audit["first_timestamp"],
                "last_timestamp": audit["last_timestamp"],
                "stopped_before_timestamp": audit[
                    "stopped_before_timestamp"
                ],
            }
            for name, audit in prefix_cuts.items()
        },
    }


def _period_mask(
    frame: pd.DataFrame,
    start: str,
    end: str,
) -> pd.Series:
    return (frame["boundary"] >= pd.Timestamp(start)) & (
        frame["boundary"] < pd.Timestamp(end)
    )


def evaluate_support(
    primitives: pd.DataFrame,
    tokens: pd.DataFrame,
    *,
    prefix_audit: Mapping[str, Any],
) -> dict[str, Any]:
    sequence = tokens.loc[tokens["sequence_ready"]].copy()
    development = _period_mask(
        sequence,
        "2020-01-01T00:00:00Z",
        "2023-01-01T00:00:00Z",
    )
    year_2021 = _period_mask(
        sequence,
        "2021-01-01T00:00:00Z",
        "2022-01-01T00:00:00Z",
    )
    year_2022 = _period_mask(
        sequence,
        "2022-01-01T00:00:00Z",
        "2023-01-01T00:00:00Z",
    )
    report_2023 = _period_mask(
        sequence,
        "2023-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
    )
    eligible = primitives.loc[primitives["otherwise_eligible"]].copy()
    eligible_period_masks = {
        "development_2020_2022": _period_mask(
            eligible,
            "2020-01-01T00:00:00Z",
            "2023-01-01T00:00:00Z",
        ),
        "report_only_2023": _period_mask(
            eligible,
            "2023-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
        ),
    }

    metrics: dict[str, Any] = {
        "token_ready": {
            "development_2020_2022": int(development.sum()),
            "2021": int(year_2021.sum()),
            "2022": int(year_2022.sum()),
            "report_only_2023": int(report_2023.sum()),
        },
        "active_months": {},
        "core_missing_share": {},
        "band_support": {},
        "freshness": {},
        "sequence": {},
        "prefix_replay": dict(prefix_audit),
    }
    month_counts = (
        sequence.loc[development, "boundary"]
        .dt.strftime("%Y-%m")
        .value_counts()
        .sort_index()
    )
    required_months = pd.period_range(
        "2020-05",
        "2022-12",
        freq="M",
    ).astype(str)
    metrics["active_months"] = {
        "required": list(required_months),
        "missing": [
            month for month in required_months if month_counts.get(month, 0) == 0
        ],
        "counts": {
            str(month): int(month_counts.get(month, 0))
            for month in required_months
        },
    }
    for primitive in prereg.CORE_PRIMITIVES:
        metrics["core_missing_share"][primitive] = {}
        for period, mask in eligible_period_masks.items():
            values = eligible.loc[mask, primitive].to_numpy(float)
            metrics["core_missing_share"][primitive][period] = (
                float(1.0 - np.isfinite(values).mean())
                if len(values)
                else 1.0
            )

    period_masks = {
        "development_2020_2022": development,
        "report_only_2023": report_2023,
    }
    for primitive in prereg.CORE_PRIMITIVES:
        column = f"{primitive}_band"
        metrics["band_support"][primitive] = {}
        for period, mask in period_masks.items():
            counts = sequence.loc[mask, column].value_counts()
            total = int(counts.sum())
            metrics["band_support"][primitive][period] = {
                "occupied": int((counts > 0).sum()),
                "largest_share": (
                    float(counts.max() / total) if total else 1.0
                ),
                "counts": {
                    band: int(counts.get(band, 0)) for band in prereg.BANDS
                },
            }
    oi_column = "oi_price_divergence_1d_band"
    metrics["band_support"]["oi_price_divergence_1d"] = {}
    for period, mask in period_masks.items():
        values = sequence.loc[mask, oi_column]
        counts = values.loc[values != "STALE"].value_counts()
        metrics["band_support"]["oi_price_divergence_1d"][period] = {
            "occupied": int((counts > 0).sum()),
            "counts": {
                band: int(counts.get(band, 0)) for band in prereg.BANDS
            },
        }

    for context in ("oi", "kimchi", "usdkrw", "dxy"):
        metrics["freshness"][context] = {}
        for period, mask in eligible_period_masks.items():
            period_frame = eligible.loc[mask]
            denominator = max(1, len(period_frame))
            fresh = int(period_frame[f"{context}_fresh"].sum())
            metrics["freshness"][context][period] = {
                "eligible": len(period_frame),
                "fresh": fresh,
                "stale": int(len(period_frame) - fresh),
                "fresh_share": fresh / denominator,
                "stale_share": (
                    len(period_frame) - fresh
                ) / denominator,
            }

    current_lines = sequence["canonical_line"].tolist()
    changes = sum(
        current != previous
        for previous, current in zip(current_lines, current_lines[1:])
    )
    signatures = Counter(sequence["sequence_signature"].tolist())
    metrics["sequence"] = {
        "rows": len(sequence),
        "adjacent_comparisons": max(0, len(sequence) - 1),
        "adjacent_change_share": (
            changes / (len(sequence) - 1) if len(sequence) > 1 else 0.0
        ),
        "unique_signatures": len(signatures),
        "max_exact_signature_share": (
            max(signatures.values()) / len(sequence) if len(sequence) else 1.0
        ),
    }

    gates: dict[str, bool] = {
        "development_count": (
            metrics["token_ready"]["development_2020_2022"] >= 3_000
        ),
        "year_2021_count": metrics["token_ready"]["2021"] >= 900,
        "year_2022_count": metrics["token_ready"]["2022"] >= 900,
        "report_2023_count": (
            metrics["token_ready"]["report_only_2023"] >= 900
        ),
        "active_months": not metrics["active_months"]["missing"],
        "core_missing": all(
            value <= 0.01
            for periods in metrics["core_missing_share"].values()
            for value in periods.values()
        ),
        "core_band_occupancy": all(
            metrics["band_support"][primitive][period]["occupied"] >= 4
            for primitive in prereg.CORE_PRIMITIVES
            for period in period_masks
        ),
        "core_band_concentration": all(
            metrics["band_support"][primitive][period]["largest_share"]
            <= 0.45
            for primitive in prereg.CORE_PRIMITIVES
            for period in period_masks
        ),
        "oi_band_occupancy": all(
            metrics["band_support"]["oi_price_divergence_1d"][period][
                "occupied"
            ]
            >= 4
            for period in period_masks
        ),
        "oi_freshness": all(
            period["fresh_share"] >= 0.50
            for period in metrics["freshness"]["oi"].values()
        ),
        "kimchi_freshness": all(
            period["fresh_share"] >= 0.80
            for period in metrics["freshness"]["kimchi"].values()
        ),
        "usdkrw_freshness": all(
            period["fresh_share"] >= 0.55
            and period["stale_share"] >= 0.05
            for period in metrics["freshness"]["usdkrw"].values()
        ),
        "dxy_freshness": all(
            period["fresh_share"] >= 0.55
            and period["stale_share"] >= 0.05
            for period in metrics["freshness"]["dxy"].values()
        ),
        "adjacent_change": (
            metrics["sequence"]["adjacent_change_share"] >= 0.95
        ),
        "signature_concentration": (
            metrics["sequence"]["max_exact_signature_share"] < 0.01
        ),
        "prefix_replay": bool(prefix_audit.get("passed")),
    }
    return {
        "decision": "PASS" if all(gates.values()) else "RETIRE",
        "gates": gates,
        "failed_gates": sorted(
            name for name, passed in gates.items() if not passed
        ),
        "metrics": metrics,
    }


def _protocol_hashes() -> dict[str, str]:
    paths = (
        SCRIPT_PATH,
        TEST_PATH,
        IMPLEMENTATION_CONTRACT,
        Path(prereg.BOUNDARY_DOCUMENT),
        PREREGISTRATION,
        Path("training/preregister_sable8_symbolic_alpha_braid.py"),
        Path("tests/test_preregister_sable8_symbolic_alpha_braid.py"),
    )
    return {str(path): sha256_file(path) for path in paths}


def build_real_support(
    *,
    token_output: str | Path = DEFAULT_TOKEN_OUTPUT,
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
    cut_manifest_output: str | Path = DEFAULT_CUT_MANIFEST_OUTPUT,
    require_committed: bool = True,
) -> dict[str, Any]:
    if require_committed:
        assert_protocol_committed()
    preregistration = validate_preregistration()
    cuts = {
        name: stream_project_source(name, spec)
        for name, spec in SOURCE_SPECS.items()
    }
    cut_manifest_core = {
        "protocol_version": f"{PROTOCOL_VERSION}_cuts",
        "preregistration_manifest_hash": preregistration["manifest_hash"],
        "source_end_exclusive": prereg.SOURCE_END_EXCLUSIVE,
        "cuts": cuts,
        "outcome_boundary": {
            "post_2023_numeric_source_rows_parsed": 0,
            "unprojected_values_converted": 0,
            "future_return_labels_built": 0,
            "rewards_built": 0,
            "models_trained": 0,
        },
    }
    cut_manifest = {
        **cut_manifest_core,
        "manifest_hash": canonical_hash(cut_manifest_core),
    }
    _write_json_once(cut_manifest_output, cut_manifest)

    market = _load_market(cuts["market"]["cut_path"])
    funding = _load_funding(cuts["funding"]["cut_path"])
    premium = _load_premium(cuts["premium"]["cut_path"])
    primitives = build_primitive_frame(market, funding, premium)
    tokens = build_token_table(primitives)
    prefix_audit = prefix_replay_audit(tokens)
    support = evaluate_support(
        primitives,
        tokens,
        prefix_audit=prefix_audit,
    )
    token_payload = token_table_bytes(tokens)
    token_sha256 = _write_bytes_once(token_output, token_payload)
    report_core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": prereg.POLICY_ID,
        "decision": support["decision"],
        "failed_gates": support["failed_gates"],
        "gates": support["gates"],
        "metrics": support["metrics"],
        "source_cut_manifest": {
            "path": str(cut_manifest_output),
            "manifest_hash": cut_manifest["manifest_hash"],
            "sha256": sha256_file(cut_manifest_output),
        },
        "token_support": {
            "path": str(token_output),
            "sha256": token_sha256,
            "rows": len(tokens),
            "columns": list(TOKEN_COLUMNS),
        },
        "protocol_hashes": _protocol_hashes(),
        "outcome_boundary": {
            "candidate_source_values_decoded": int(
                sum(item["rows"] for item in cuts.values())
            ),
            "candidate_token_incidence_calculated": int(
                tokens["sequence_ready"].sum()
            ),
            "future_return_labels_built": 0,
            "rewards_built": 0,
            "market_outcomes_evaluated": 0,
            "funding_cash_flows_evaluated": 0,
            "models_trained": 0,
            "post_2023_numeric_source_rows_parsed": 0,
            "candidate_2023_outcomes_opened": False,
            "candidate_2024_outcomes_opened": False,
            "candidate_2025_outcomes_opened": False,
            "candidate_2026_outcomes_opened": False,
        },
        "next_authority": (
            "commit Stage 0.5 evaluator/model freeze"
            if support["decision"] == "PASS"
            else "retire exact SABLE-8 candidate"
        ),
    }
    report = {
        **report_core,
        "report_hash": canonical_hash(report_core),
    }
    _write_json_once(report_output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-output", default=str(DEFAULT_TOKEN_OUTPUT))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT_OUTPUT))
    parser.add_argument(
        "--cut-manifest-output",
        default=str(DEFAULT_CUT_MANIFEST_OUTPUT),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_real_support(
        token_output=args.token_output,
        report_output=args.report_output,
        cut_manifest_output=args.cut_manifest_output,
        require_committed=True,
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "failed_gates": report["failed_gates"],
                "report_hash": report["report_hash"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
