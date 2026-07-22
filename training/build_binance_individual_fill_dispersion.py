"""Build outcome-blind IFDA raw-fill and aggregate-control five-minute features.

The builder reads official Binance USD-M daily ``trades`` and ``aggTrades``
archives one ZIP at a time, verifies every published checksum, and immediately
reduces raw rows to a complete UTC five-minute grid.  Raw archives are never
written to disk.  The process fails closed before a download when filesystem
usage reaches the frozen 300-GiB limit.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import date, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Final, cast

import numpy as np
import pandas as pd

from preprocessing import individual_fill_dispersion as ifda_transform
from preprocessing.individual_fill_dispersion import (
    OUTPUT_COLUMNS,
    RAW_AGGTRADE_COLUMNS,
    RAW_TRADE_COLUMNS,
    aggregate_day,
)


BASE_URL = "https://data.binance.vision/data/futures/um/daily"
SCHEMA_VERSION = 1
GIB = 1024**3
GZIP_COMPRESSION = {"method": "gzip", "compresslevel": 6, "mtime": 0}
USER_AGENT = "rllm-ifda-source-builder/1.0"
FROZEN_SYMBOL: Final[str] = "BTCUSDT"
FROZEN_START: Final[str] = "2020-01-01"
FROZEN_END: Final[str] = "2024-01-01"
PREREGISTRATION_MANIFEST_SHA256: Final[str] = (
    "757401202eacd3dcf0540ae54f4c121ba595d2495d0f4ae69fe248bfa360e02c"
)
PREREGISTRATION_FILE_SHA256: Final[str] = (
    "abecfddfaad7a7640d8d16fc04ff3938dad79cfa94d4b97095d01baf9f7c9b70"
)

SOURCE_PROTOCOL: Final[dict[str, Any]] = {
    "protocol_version": "ifda_source_v1",
    "preregistration_manifest_sha256": PREREGISTRATION_MANIFEST_SHA256,
    "preregistration_file_sha256": PREREGISTRATION_FILE_SHA256,
    "symbol": FROZEN_SYMBOL,
    "range": [FROZEN_START, FROZEN_END],
    "end_is_exclusive": True,
    "individual_source": {
        "kind": "trades",
        "url_template": (
            f"{BASE_URL}/trades/{FROZEN_SYMBOL}/"
            f"{FROZEN_SYMBOL}-trades-{{YYYY-MM-DD}}.zip"
        ),
        "columns": list(RAW_TRADE_COLUMNS),
        "notional": "quote_qty",
    },
    "aggregate_control_source": {
        "kind": "aggTrades",
        "url_template": (
            f"{BASE_URL}/aggTrades/{FROZEN_SYMBOL}/"
            f"{FROZEN_SYMBOL}-aggTrades-{{YYYY-MM-DD}}.zip"
        ),
        "columns": list(RAW_AGGTRADE_COLUMNS),
        "notional": "price*quantity",
    },
    "checksum_suffix": ".CHECKSUM",
    "checksum_revision_policy": "reject; never rebuild in place",
    "bucket": "completed UTC five-minute bar",
    "aggressive_side": "is_buyer_maker=false => +1; true => -1",
    "quote_qty_reconciliation_tolerance": (
        ifda_transform.QUOTE_QTY_RECONCILIATION_TOLERANCE
    ),
    "formula": {
        "side_hhi": "sum(q_i^2)/sum(q_i)^2",
        "side_equalization": "(1/side_hhi)/side_event_count",
        "dominant_side": "sign(buy_notional-sell_notional)",
        "flow_coherence": "abs(buy_notional-sell_notional)/(buy_notional+sell_notional)",
        "equalization_gap": "max(dominant_equalization-opposing_equalization,0)",
        "score": "flow_coherence*dominant_equalization*equalization_gap",
    },
    "output_columns": list(OUTPUT_COLUMNS),
}
SOURCE_PROTOCOL_SHA256: Final[str] = hashlib.sha256(
    json.dumps(SOURCE_PROTOCOL, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class BuildConfig:
    symbol: str = FROZEN_SYMBOL
    start: str = FROZEN_START
    end: str = FROZEN_END
    output_dir: str = "data/binance_um_individual_fill_dispersion_btc_2020_2023"
    retries: int = 5
    timeout_seconds: int = 90
    disk_used_abort_gib: int = 300
    overwrite: bool = False


def archive_url(symbol: str, day: date, kind: str) -> str:
    if kind not in {"trades", "aggTrades"}:
        raise ValueError(f"unsupported Binance archive kind: {kind}")
    stamp = day.isoformat()
    return f"{BASE_URL}/{kind}/{symbol}/{symbol}-{kind}-{stamp}.zip"


def checksum_url(symbol: str, day: date, kind: str) -> str:
    return archive_url(symbol, day, kind) + ".CHECKSUM"


def _fetch_bytes(url: str, *, retries: int, timeout: int) -> bytes:
    error: BaseException | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(url) from exc
            error = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = exc
        if attempt + 1 < retries:
            time.sleep(min(16.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"failed to fetch {url} after {retries} attempts") from error


def ensure_disk_budget(
    *,
    path: str | Path = "/",
    used_bytes: int | None = None,
    limit_gib: int = BuildConfig.disk_used_abort_gib,
) -> int:
    if limit_gib <= 0:
        raise ValueError("disk_used_abort_gib must be positive")
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    current = shutil.disk_usage(target).used if used_bytes is None else int(used_bytes)
    if current >= limit_gib * GIB:
        raise RuntimeError(
            f"IFDA disk guard: used={current / GIB:.2f} GiB >= {limit_gib} GiB"
        )
    return current


def implementation_sha256() -> dict[str, str]:
    transform_file = ifda_transform.__file__
    if transform_file is None:
        raise RuntimeError("cannot locate IFDA transform implementation")
    paths = {
        "builder": Path(__file__).resolve(),
        "transform": Path(transform_file).resolve(),
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


def expected_sha256(checksum_payload: bytes) -> str:
    fields = checksum_payload.decode("utf-8").strip().split()
    if not fields or len(fields[0]) != 64:
        raise ValueError("invalid Binance checksum payload")
    int(fields[0], 16)
    return fields[0].lower()


def verify_sha256(payload: bytes, expected: str) -> str:
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected.lower():
        raise ValueError(f"archive checksum mismatch: expected {expected}, got {actual}")
    return actual


def _normalized_header(value: object) -> str:
    return "".join(character for character in str(value).strip().lower() if character.isalnum())


TRADE_HEADER_MAP = {
    "id": "trade_id",
    "tradeid": "trade_id",
    "price": "price",
    "qty": "quantity",
    "quantity": "quantity",
    "quoteqty": "quote_qty",
    "time": "transact_time",
    "transacttime": "transact_time",
    "isbuyermaker": "is_buyer_maker",
}
AGGTRADE_HEADER_MAP = {
    "aggtradeid": "agg_trade_id",
    "price": "price",
    "quantity": "quantity",
    "qty": "quantity",
    "firsttradeid": "first_trade_id",
    "lasttradeid": "last_trade_id",
    "transacttime": "transact_time",
    "time": "transact_time",
    "isbuyermaker": "is_buyer_maker",
}


def _read_csv_member(payload: bytes) -> tuple[zipfile.ZipFile, str, bytes]:
    archive = zipfile.ZipFile(io.BytesIO(payload))
    members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    if len(members) != 1:
        archive.close()
        raise ValueError(f"expected exactly one CSV in archive, found {members}")
    member = members[0]
    with archive.open(member) as handle:
        first_line = handle.readline()
    return archive, member, first_line


def _parse_maker_flag(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    if not text.isin(["true", "false"]).all():
        raise ValueError("is_buyer_maker contains an unknown value")
    return text.eq("true")


def _read_archive(
    payload: bytes,
    *,
    expected_columns: tuple[str, ...],
    header_map: dict[str, str],
    id_columns: tuple[str, ...],
) -> pd.DataFrame:
    archive, member, first_line = _read_csv_member(payload)
    try:
        first_token = first_line.split(b",", 1)[0].strip()
        has_header = not first_token.lstrip(b"-").isdigit()
        with archive.open(member) as handle:
            frame = pd.read_csv(
                handle,
                header=0 if has_header else None,
                names=None if has_header else list(expected_columns),
                dtype="string",
                low_memory=False,
            )
    finally:
        archive.close()
    if has_header:
        normalized = [_normalized_header(column) for column in frame.columns]
        if any(column not in header_map for column in normalized):
            raise ValueError(f"unexpected archive columns: {list(frame.columns)}")
        frame.columns = [header_map[column] for column in normalized]
    if tuple(frame.columns) != expected_columns:
        raise ValueError(f"unexpected canonical columns: {list(frame.columns)}")
    if frame.empty:
        raise ValueError("Binance archive is empty")

    numeric_columns = [column for column in expected_columns if column != "is_buyer_maker"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in id_columns:
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.array_equal(values, np.rint(values)):
            raise ValueError(f"{column} contains non-integral values")
        frame[column] = frame[column].astype("int64")
    maker_flags = cast(pd.Series, frame["is_buyer_maker"])
    frame["is_buyer_maker"] = _parse_maker_flag(maker_flags)
    return frame


def read_trades_archive(payload: bytes) -> pd.DataFrame:
    return _read_archive(
        payload,
        expected_columns=RAW_TRADE_COLUMNS,
        header_map=TRADE_HEADER_MAP,
        id_columns=("trade_id", "transact_time"),
    )


def read_aggtrades_archive(payload: bytes) -> pd.DataFrame:
    return _read_archive(
        payload,
        expected_columns=RAW_AGGTRADE_COLUMNS,
        header_map=AGGTRADE_HEADER_MAP,
        id_columns=(
            "agg_trade_id",
            "first_trade_id",
            "last_trade_id",
            "transact_time",
        ),
    )


def validate_daily_cross_source(
    trades: pd.DataFrame,
    aggtrades: pd.DataFrame,
) -> None:
    first_trade = int(trades["trade_id"].iloc[0])
    last_trade = int(trades["trade_id"].iloc[-1])
    first_underlying = int(aggtrades["first_trade_id"].iloc[0])
    last_underlying = int(aggtrades["last_trade_id"].iloc[-1])
    if (first_trade, last_trade) != (first_underlying, last_underlying):
        raise ValueError("trades and aggTrades underlying ID boundaries disagree")
    first_ids = aggtrades["first_trade_id"].to_numpy(np.int64)
    last_ids = aggtrades["last_trade_id"].to_numpy(np.int64)
    if (last_ids < first_ids).any():
        raise ValueError("aggTrades contains an inverted underlying ID span")
    if len(first_ids) > 1 and not np.array_equal(first_ids[1:], last_ids[:-1] + 1):
        raise ValueError("aggTrades underlying ID spans are not exactly contiguous")
    span_count = int(np.sum(last_ids - first_ids + 1))
    if span_count != len(trades):
        raise ValueError("aggTrades underlying spans do not cover raw trades exactly once")


def _fetch_verified_archive(
    cfg: BuildConfig,
    day: date,
    kind: str,
    *,
    fetcher: Callable[..., bytes],
    disk_guard: Callable[..., int],
) -> tuple[bytes, str]:
    disk_guard(path=cfg.output_dir, limit_gib=cfg.disk_used_abort_gib)
    checksum_payload = fetcher(
        checksum_url(cfg.symbol, day, kind),
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    expected = expected_sha256(checksum_payload)
    disk_guard(path=cfg.output_dir, limit_gib=cfg.disk_used_abort_gib)
    payload = fetcher(
        archive_url(cfg.symbol, day, kind),
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    return payload, verify_sha256(payload, expected)


def _month_starts(start: date, end: date) -> list[date]:
    current = date(start.year, start.month, 1)
    months: list[date] = []
    while current < end:
        months.append(current)
        current = date(
            current.year + (current.month == 12),
            1 if current.month == 12 else current.month + 1,
            1,
        )
    return months


def _month_days(month: date, start: date, end: date) -> list[date]:
    next_month = date(
        month.year + (month.month == 12),
        1 if month.month == 12 else month.month + 1,
        1,
    )
    current = max(start, month)
    limit = min(end, next_month)
    days: list[date] = []
    while current < limit:
        days.append(current)
        current += timedelta(days=1)
    return days


def _write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        compression=GZIP_COMPRESSION,
        float_format="%.12g",
    )


def _resume_metadata_is_current(
    metadata: dict[str, Any],
    *,
    cfg: BuildConfig,
    month: date,
    expected_days: list[date],
    output_path: Path,
    fetcher: Callable[..., bytes],
    disk_guard: Callable[..., int],
) -> bool:
    if metadata.get("source_protocol_sha256") != SOURCE_PROTOCOL_SHA256:
        raise ValueError("resume artifact source protocol differs from frozen IFDA semantics")
    if metadata.get("implementation_sha256") != implementation_sha256():
        raise ValueError("resume artifact implementation differs from current frozen IFDA code")
    expected_dates = [day.isoformat() for day in expected_days]
    archives = metadata.get("archives")
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("month") != f"{month:%Y-%m}"
        or metadata.get("symbol") != cfg.symbol
        or metadata.get("requested_dates") != expected_dates
        or metadata.get("columns") != list(OUTPUT_COLUMNS)
        or not isinstance(archives, list)
        or [item.get("date") for item in archives] != expected_dates
    ):
        return False
    if hashlib.sha256(output_path.read_bytes()).hexdigest() != metadata.get(
        "output_sha256"
    ):
        raise ValueError(f"resume artifact hash mismatch: {output_path}")
    for day, archive in zip(expected_days, archives, strict=True):
        for kind, key in (
            ("trades", "trades_archive_sha256"),
            ("aggTrades", "aggtrades_archive_sha256"),
        ):
            disk_guard(path=cfg.output_dir, limit_gib=cfg.disk_used_abort_gib)
            current = expected_sha256(
                fetcher(
                    checksum_url(cfg.symbol, day, kind),
                    retries=cfg.retries,
                    timeout=cfg.timeout_seconds,
                )
            )
            if current != archive.get(key):
                raise ValueError(
                    "upstream checksum revision rejects frozen IFDA source: "
                    f"{day} {kind}"
                )
    return True


def _process_month(
    month: date,
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
    disk_guard: Callable[..., int] = ensure_disk_budget,
) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    output_dir = Path(cfg.output_dir)
    monthly_dir = output_dir / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.symbol}_individual_fill_dispersion_5m_{month:%Y-%m}"
    output_path = monthly_dir / f"{stem}.csv.gz"
    metadata_path = monthly_dir / f"{stem}.json"
    expected_days = _month_days(month, start, end)
    if output_path.exists() and metadata_path.exists() and not cfg.overwrite:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if _resume_metadata_is_current(
            metadata,
            cfg=cfg,
            month=month,
            expected_days=expected_days,
            output_path=output_path,
            fetcher=fetcher,
            disk_guard=disk_guard,
        ):
            return metadata

    frames: list[pd.DataFrame] = []
    archives: list[dict[str, Any]] = []
    for day in expected_days:
        trades_payload, trades_hash = _fetch_verified_archive(
            cfg, day, "trades", fetcher=fetcher, disk_guard=disk_guard
        )
        trades = read_trades_archive(trades_payload)
        del trades_payload
        agg_payload, agg_hash = _fetch_verified_archive(
            cfg, day, "aggTrades", fetcher=fetcher, disk_guard=disk_guard
        )
        aggtrades = read_aggtrades_archive(agg_payload)
        del agg_payload
        validate_daily_cross_source(trades, aggtrades)
        aggregated = aggregate_day(trades, aggtrades, day)
        if tuple(aggregated.columns) != OUTPUT_COLUMNS:
            raise ValueError("IFDA daily aggregate schema differs from frozen output")
        frames.append(aggregated)
        archives.append(
            {
                "date": day.isoformat(),
                "trades_archive_sha256": trades_hash,
                "aggtrades_archive_sha256": agg_hash,
                "trade_rows": int(len(trades)),
                "aggtrade_rows": int(len(aggtrades)),
                "first_trade_id": int(trades["trade_id"].iloc[0]),
                "last_trade_id": int(trades["trade_id"].iloc[-1]),
                "first_agg_trade_id": int(aggtrades["agg_trade_id"].iloc[0]),
                "last_agg_trade_id": int(aggtrades["agg_trade_id"].iloc[-1]),
            }
        )
        del trades, aggtrades, aggregated

    combined = pd.concat(frames, ignore_index=True)
    if combined["date"].duplicated().any() or not combined["date"].is_monotonic_increasing:
        raise ValueError(f"month {month:%Y-%m} has duplicate or unordered bins")
    _write_gzip_csv(combined, output_path)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "month": f"{month:%Y-%m}",
        "symbol": cfg.symbol,
        "requested_dates": [day.isoformat() for day in expected_days],
        "output": str(output_path),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "rows": int(len(combined)),
        "first_date": str(combined["date"].min()),
        "last_date": str(combined["date"].max()),
        "columns": list(combined.columns),
        "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
        "implementation_sha256": implementation_sha256(),
        "archives": archives,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def _validate_archive_continuity(metadata: list[dict[str, Any]]) -> None:
    archives = [archive for month in metadata for archive in month["archives"]]
    for previous, current in zip(archives, archives[1:]):
        if int(current["first_trade_id"]) != int(previous["last_trade_id"]) + 1:
            raise ValueError(
                "raw trade IDs are not exactly continuous across daily archives: "
                f"{previous['date']} -> {current['date']}"
            )
        if int(current["first_agg_trade_id"]) != int(previous["last_agg_trade_id"]) + 1:
            raise ValueError(
                "aggregate trade IDs are not exactly continuous across daily archives: "
                f"{previous['date']} -> {current['date']}"
            )


def build(
    cfg: BuildConfig, *, allow_partial_source_for_tests: bool = False
) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if cfg.retries < 1 or cfg.timeout_seconds < 1:
        raise ValueError("retries and timeout_seconds must be positive")
    if not allow_partial_source_for_tests and (
        cfg.symbol != FROZEN_SYMBOL
        or cfg.start != FROZEN_START
        or cfg.end != FROZEN_END
    ):
        raise ValueError(
            "production IFDA source is frozen to BTCUSDT [2020-01-01, 2024-01-01)"
        )
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_disk_budget(path=output_dir, limit_gib=cfg.disk_used_abort_gib)

    metadata: list[dict[str, Any]] = []
    for month in _month_starts(start, end):
        result = _process_month(month, cfg)
        metadata.append(result)
        print(f"completed {month:%Y-%m}: rows={result['rows']}", flush=True)
    _validate_archive_continuity(metadata)

    monthly_frames = [
        pd.read_csv(item["output"], compression="gzip", parse_dates=["date"])
        for item in metadata
    ]
    combined = pd.concat(monthly_frames, ignore_index=True)
    if combined["date"].duplicated().any() or not combined["date"].is_monotonic_increasing:
        raise ValueError("combined IFDA output has duplicate or unordered timestamps")
    expected = pd.date_range(
        start, end, freq="5min", inclusive="left", tz=timezone.utc
    )
    actual = pd.DatetimeIndex(combined["date"])
    if not actual.equals(expected):
        raise ValueError("combined IFDA output is not the complete frozen five-minute grid")
    if tuple(combined.columns) != OUTPUT_COLUMNS:
        raise ValueError("combined IFDA output schema differs from frozen schema")

    last_day = end - timedelta(days=1)
    combined_path = output_dir / (
        f"{cfg.symbol}_individual_fill_dispersion_5m_{cfg.start}_{last_day}.csv.gz"
    )
    _write_gzip_csv(combined, combined_path)
    manifest = {
        "config": asdict(cfg),
        "source_protocol": SOURCE_PROTOCOL,
        "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
        "implementation_sha256": implementation_sha256(),
        "protocol": {
            "source": (
                "official Binance USD-M Futures daily trades and aggTrades archives"
            ),
            "archive_checksums_verified": True,
            "end_is_exclusive": True,
            "five_minute_bin": "UTC floor of official transaction timestamp",
            "buyer_maker_semantics": (
                "true = buyer passive / seller aggressive, therefore signed side -1"
            ),
            "raw_archives_persisted": False,
            "one_archive_in_memory_at_a_time": True,
            "disk_used_abort_gib": cfg.disk_used_abort_gib,
            "outcomes_opened": False,
            "source_incidence_opened": True,
            "future_ohlc_read": False,
            "funding_or_pnl_read": False,
        },
        "combined_output": str(combined_path),
        "combined_sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
        "rows": int(len(combined)),
        "first_date": str(combined["date"].min()),
        "last_date": str(combined["date"].max()),
        "columns": list(combined.columns),
        "months": metadata,
    }
    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--retries", type=int, default=BuildConfig.retries)
    parser.add_argument("--timeout-seconds", type=int, default=BuildConfig.timeout_seconds)
    parser.add_argument(
        "--disk-used-abort-gib", type=int, default=BuildConfig.disk_used_abort_gib
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build(BuildConfig(**vars(args)))
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in ("combined_output", "combined_sha256", "rows", "first_date", "last_date")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
