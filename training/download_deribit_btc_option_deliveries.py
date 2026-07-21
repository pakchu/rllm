"""Download and aggregate pre-2023 Deribit BTC option delivery rows.

Official endpoint:
https://docs.deribit.com/api-reference/market-data/public-get_last_settlements_by_currency

The raw response is not persisted.  The committed surface is a hash-bound,
expiry-level source aggregate containing no market outcome after delivery.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pandas as pd


ENDPOINT = (
    "https://www.deribit.com/api/v2/"
    "public/get_last_settlements_by_currency"
)
OFFICIAL_DOCS = (
    "https://docs.deribit.com/api-reference/market-data/"
    "public-get_last_settlements_by_currency"
)
OPTION_RE = re.compile(
    r"^BTC-(?P<expiry>\d{1,2}[A-Z]{3}\d{2})-"
    r"(?P<strike>\d+(?:\.\d+)?)-(?P<option_type>[CP])$"
)
FUTURE_RE = re.compile(r"^BTC-\d{1,2}[A-Z]{3}\d{2}$")
EXPIRY_RE = re.compile(
    r"^(?P<day>\d{1,2})(?P<month>[A-Z]{3})(?P<year>\d{2})$"
)
MONTH_BY_CODE = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


@dataclass(frozen=True)
class Config:
    output_csv: str = (
        "data/deribit_btc_option_delivery_release_2019_2022.csv.gz"
    )
    manifest_output: str = (
        "results/deribit_btc_option_delivery_source_manifest_2026-07-20.json"
    )
    start: str = "2019-01-01"
    end_exclusive: str = "2023-01-01"
    currency: str = "BTC"
    settlement_type: str = "delivery"
    page_size: int = 1000
    timeout_sec: float = 30.0
    request_pause_sec: float = 0.20
    maximum_retries: int = 8
    maximum_event_row_span_seconds: float = 5.0


Fetch = Callable[[dict[str, Any]], dict[str, Any]]
Aggregate = Callable[
    [list[dict[str, Any]], Config],
    tuple[pd.DataFrame, dict[str, Any]],
]


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = cast(pd.Timestamp, pd.Timestamp(value))
    if timestamp.tzinfo is None:
        return cast(pd.Timestamp, timestamp.tz_localize("UTC"))
    return cast(pd.Timestamp, timestamp.tz_convert("UTC"))


def _http_page(cfg: Config, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{ENDPOINT}?{urllib.parse.urlencode(params)}"
    for attempt in range(cfg.maximum_retries + 1):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "rllm-private-research/1.0"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=cfg.timeout_sec
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("Deribit delivery response is not an object")
            return payload
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code <= 599
            if not retryable or attempt >= cfg.maximum_retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 2.0**attempt
            except ValueError:
                delay = 2.0**attempt
            time.sleep(max(cfg.request_pause_sec, min(60.0, delay)))
        except (TimeoutError, urllib.error.URLError):
            if attempt >= cfg.maximum_retries:
                raise
            time.sleep(
                max(cfg.request_pause_sec, min(60.0, 2.0**attempt))
            )
    raise AssertionError("unreachable retry loop")


def _parse_payload(
    payload: dict[str, Any], *, page_size: int
) -> tuple[list[dict[str, Any]], str | None]:
    if payload.get("error") is not None:
        raise RuntimeError(f"Deribit delivery API error: {payload['error']!r}")
    if payload.get("jsonrpc") not in {None, "2.0"}:
        raise RuntimeError("Deribit delivery JSON-RPC version mismatch")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("Deribit delivery result is not an object")
    rows = result.get("settlements")
    if not isinstance(rows, list):
        raise RuntimeError("Deribit delivery settlements is not a list")
    if len(rows) > page_size:
        raise RuntimeError("Deribit delivery page exceeds frozen page size")
    if any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("Deribit delivery row is not an object")
    continuation = result.get("continuation")
    if continuation in {None, ""}:
        return rows, None
    if not isinstance(continuation, str):
        raise RuntimeError("Deribit delivery continuation is not a string")
    return rows, continuation


def _number(row: dict[str, Any], key: str, *, positive: bool) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Deribit delivery {key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Deribit delivery {key} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"Deribit delivery {key} must be positive")
    if not positive and number < 0.0:
        raise ValueError(f"Deribit delivery {key} must be non-negative")
    return number


def _timestamp_ms(row: dict[str, Any]) -> int:
    value = row.get("timestamp")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("Deribit delivery timestamp must be a positive integer")
    return value


def _normalise_option(
    row: dict[str, Any], cfg: Config
) -> dict[str, Any] | None:
    if row.get("type") != cfg.settlement_type:
        raise ValueError("Deribit delivery row has another settlement type")
    instrument = row.get("instrument_name")
    if not isinstance(instrument, str) or not instrument:
        raise ValueError("Deribit delivery instrument_name must be non-empty")
    if FUTURE_RE.fullmatch(instrument):
        return None
    match = OPTION_RE.fullmatch(instrument)
    if match is None:
        raise ValueError(f"unsupported Deribit BTC delivery instrument: {instrument}")

    raw_timestamp = cast(
        pd.Timestamp,
        pd.Timestamp(_timestamp_ms(row), unit="ms", tz="UTC"),
    )
    expiry_match = EXPIRY_RE.fullmatch(match.group("expiry"))
    if expiry_match is None:
        raise ValueError(f"invalid Deribit option expiry: {instrument}")
    try:
        expiry_date = cast(
            pd.Timestamp,
            pd.Timestamp(
                datetime(
                    2000 + int(expiry_match.group("year")),
                    MONTH_BY_CODE[expiry_match.group("month")],
                    int(expiry_match.group("day")),
                    tzinfo=timezone.utc,
                )
            )
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid Deribit option expiry: {instrument}") from exc
    scheduled_expiry_time = cast(
        pd.Timestamp,
        expiry_date + pd.Timedelta(hours=8),
    )
    if (
        raw_timestamp < scheduled_expiry_time
        or raw_timestamp.normalize() != scheduled_expiry_time.normalize()
    ):
        raise ValueError(
            "Deribit delivery timestamp is outside the frozen expiry-date window"
        )

    strike = float(match.group("strike"))
    if not math.isfinite(strike) or strike <= 0.0:
        raise ValueError("Deribit option strike must be finite and positive")
    position = _number(row, "position", positive=True)
    index_price = _number(row, "index_price", positive=True)
    mark_price = _number(row, "mark_price", positive=False)
    option_type = match.group("option_type")
    if option_type == "C":
        terminal_state = "itm" if index_price > strike else (
            "otm" if index_price < strike else "atm"
        )
    else:
        terminal_state = "itm" if index_price < strike else (
            "otm" if index_price > strike else "atm"
        )
    return {
        "expiry_time": scheduled_expiry_time,
        "raw_timestamp": raw_timestamp,
        "instrument_name": instrument,
        "strike": strike,
        "option_type": option_type,
        "position": position,
        "index_price": index_price,
        "mark_price": mark_price,
        "terminal_state": terminal_state,
    }


def aggregate_deliveries(
    rows: list[dict[str, Any]], cfg: Config
) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    futures = 0
    for row in rows:
        parsed = _normalise_option(row, cfg)
        if parsed is None:
            futures += 1
        else:
            normalised.append(parsed)
    if not normalised:
        raise RuntimeError("Deribit delivery source has no BTC options")
    options = pd.DataFrame.from_records(normalised)
    duplicate = options.duplicated(["expiry_time", "instrument_name"])
    if duplicate.any():
        raise RuntimeError("Deribit delivery source contains duplicate instruments")
    options = options.sort_values(
        ["expiry_time", "strike", "option_type"], ignore_index=True
    )

    records: list[dict[str, Any]] = []
    for expiry_time, group in options.groupby("expiry_time", sort=True):
        raw_span = (
            group["raw_timestamp"].max() - group["raw_timestamp"].min()
        ).total_seconds()
        if raw_span > cfg.maximum_event_row_span_seconds:
            raise RuntimeError("Deribit option rows disagree on delivery clock")
        delivery_event_time = group["raw_timestamp"].max()
        index_prices = group["index_price"].to_numpy(float)
        if not np.allclose(index_prices, index_prices[0], rtol=1e-12, atol=1e-8):
            raise RuntimeError("Deribit option rows disagree on delivery index")
        position = group["position"].to_numpy(float)
        total_position = float(position.sum())
        call = group["option_type"].eq("C")
        put = ~call
        itm = group["terminal_state"].eq("itm")
        otm = group["terminal_state"].eq("otm")
        atm = group["terminal_state"].eq("atm")
        itm_call = float(group.loc[call & itm, "position"].sum())
        itm_put = float(group.loc[put & itm, "position"].sum())
        net_release = itm_put - itm_call
        records.append(
            {
                "expiry_time": expiry_time,
                "delivery_event_time": delivery_event_time,
                "source_observation_earliest": delivery_event_time
                + pd.Timedelta(minutes=65),
                "index_price": float(index_prices[0]),
                "option_count": int(len(group)),
                "call_count": int(call.sum()),
                "put_count": int(put.sum()),
                "itm_call_count": int((call & itm).sum()),
                "itm_put_count": int((put & itm).sum()),
                "total_position": total_position,
                "call_position": float(group.loc[call, "position"].sum()),
                "put_position": float(group.loc[put, "position"].sum()),
                "itm_call_position": itm_call,
                "itm_put_position": itm_put,
                "otm_position": float(group.loc[otm, "position"].sum()),
                "atm_position": float(group.loc[atm, "position"].sum()),
                "net_release_position": net_release,
                "absolute_release_position": abs(net_release),
                "release_side": int(np.sign(net_release)),
                "largest_instrument_share": float(position.max() / total_position),
                "delivery_delay_seconds": float(
                    (delivery_event_time - expiry_time).total_seconds()
                ),
                "maximum_event_row_span_seconds": float(raw_span),
            }
        )
    aggregate = pd.DataFrame.from_records(records)
    if aggregate["expiry_time"].duplicated().any():
        raise AssertionError("Deribit aggregate expiry clock is not unique")
    audit = {
        "option_rows_selected": int(len(options)),
        "futures_rows_excluded": int(futures),
        "expiry_events": int(len(aggregate)),
        "first_expiry": aggregate["expiry_time"].iloc[0].isoformat(),
        "last_expiry": aggregate["expiry_time"].iloc[-1].isoformat(),
        "all_positions_positive": bool(options["position"].gt(0.0).all()),
        "unique_instrument_per_expiry": True,
        "all_expiries_at_08_utc": bool(
            aggregate["expiry_time"].dt.hour.eq(8).all()
            and aggregate["expiry_time"].dt.minute.eq(0).all()
        ),
        "maximum_delivery_delay_seconds": float(
            aggregate["delivery_delay_seconds"].max()
        ),
        "delayed_expiry_events": int(
            aggregate["delivery_delay_seconds"].gt(5.0).sum()
        ),
        "maximum_event_row_span_seconds": float(
            aggregate["maximum_event_row_span_seconds"].max()
        ),
        "rows_by_year": {
            str(year): int(count)
            for year, count in options["expiry_time"]
            .dt.year.value_counts()
            .sort_index()
            .items()
        },
        "expiries_by_year": {
            str(year): int(count)
            for year, count in aggregate["expiry_time"]
            .dt.year.value_counts()
            .sort_index()
            .items()
        },
    }
    return aggregate, audit


def download(
    cfg: Config,
    *,
    fetch: Fetch | None = None,
    sleep: Callable[[float], None] = time.sleep,
    aggregate: Aggregate | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not 1 <= cfg.page_size <= 1000:
        raise ValueError("Deribit delivery page_size must be in [1, 1000]")
    if cfg.currency != "BTC" or cfg.settlement_type != "delivery":
        raise ValueError("Deribit delivery currency/type contract is frozen")
    start = _utc(cfg.start)
    end = _utc(cfg.end_exclusive)
    if start >= end or any(
        [
            start.hour,
            start.minute,
            start.second,
            end.hour,
            end.minute,
            end.second,
        ]
    ):
        raise ValueError("Deribit delivery interval must use UTC day boundaries")
    fetch = fetch or (lambda params: _http_page(cfg, params))

    continuation: str | None = None
    seen_continuations: set[str] = set()
    rows: list[dict[str, Any]] = []
    page_lengths: list[int] = []
    page_hashes: list[str] = []
    previous_oldest: int | None = None
    crossed_start_boundary = False
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while True:
        params: dict[str, Any] = {
            "currency": cfg.currency,
            "type": cfg.settlement_type,
            "count": cfg.page_size,
            "search_start_timestamp": int(end.timestamp() * 1000) - 1,
        }
        if continuation is not None:
            params["continuation"] = continuation
        payload = fetch(params)
        page_hashes.append(canonical_hash(payload))
        batch, next_continuation = _parse_payload(
            payload, page_size=cfg.page_size
        )
        page_lengths.append(len(batch))
        if not batch:
            if not crossed_start_boundary:
                raise RuntimeError(
                    "Deribit delivery history ended before crossing the frozen "
                    "source start boundary"
                )
            break
        timestamps = [_timestamp_ms(row) for row in batch]
        if timestamps != sorted(timestamps, reverse=True):
            raise RuntimeError("Deribit delivery page is not newest-first")
        if previous_oldest is not None and max(timestamps) > previous_oldest:
            raise RuntimeError("Deribit delivery pagination moved forward")
        previous_oldest = min(timestamps)
        if any(timestamp >= end_ms for timestamp in timestamps):
            raise RuntimeError("Deribit ignored the frozen source end boundary")
        rows.extend(
            row
            for row, timestamp in zip(batch, timestamps)
            if start_ms <= timestamp < end_ms
        )
        if min(timestamps) < start_ms:
            crossed_start_boundary = True
            break
        if next_continuation is None:
            raise RuntimeError(
                "Deribit delivery history ended before crossing the frozen "
                "source start boundary"
            )
        if next_continuation in seen_continuations:
            raise RuntimeError("Deribit delivery continuation loop detected")
        seen_continuations.add(next_continuation)
        continuation = next_continuation
        if cfg.request_pause_sec:
            sleep(cfg.request_pause_sec)

    if not rows:
        raise RuntimeError("Deribit delivery source returned no frozen rows")
    aggregate_frame, aggregate_audit = (aggregate or aggregate_deliveries)(rows, cfg)
    if (
        aggregate_frame["expiry_time"].min() < start
        or aggregate_frame["expiry_time"].max() >= end
    ):
        raise RuntimeError("Deribit aggregate escaped frozen interval")
    audit = {
        "endpoint": ENDPOINT,
        "official_docs": OFFICIAL_DOCS,
        "request_pages": len(page_lengths),
        "page_lengths": page_lengths,
        "page_canonical_sha256": page_hashes,
        "rows_received_in_interval": len(rows),
        "pagination_newest_first": True,
        "continuation_tokens_unique": True,
        "crossed_start_boundary": crossed_start_boundary,
        "start": cfg.start,
        "end_exclusive": cfg.end_exclusive,
        **aggregate_audit,
    }
    return aggregate_frame, audit


def _write_deterministic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                frame.to_csv(
                    text,
                    index=False,
                    lineterminator="\n",
                    date_format="%Y-%m-%d %H:%M:%S.%f",
                    float_format="%.12g",
                )
    os.replace(temporary, path)


def run(
    cfg: Config,
    *,
    fetch: Fetch | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    frame, audit = download(cfg, fetch=fetch, sleep=sleep)
    output = Path(cfg.output_csv)
    _write_deterministic_csv(output, frame)
    core = {
        "protocol_version": "deribit_btc_option_delivery_source_v3",
        "config": asdict(cfg),
        "source_audit": audit,
        "aggregate": {
            "path": str(output),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "rows": len(frame),
            "columns": list(frame.columns),
        },
        "outcome_boundary": {
            "binance_market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "post_delivery_return_or_pnl_loaded": False,
            "raw_deribit_rows_persisted": False,
        },
        "causal_availability": {
            "deribit_publication_sla_known": False,
            "source_observation_rule": (
                "delivery_event_time + 65 minutes after two identical "
                "canonical delivery sets observed five minutes apart"
            ),
            "source_observation_latency_seconds": 3900,
            "earliest_next_5m_entry_latency_seconds": 4200,
        },
        "data_use": (
            "private internal research; raw source responses are not persisted "
            "or redistributed by the repository"
        ),
    }
    manifest = {
        **core,
        "manifest_hash": canonical_hash(core),
    }
    manifest_path = Path(cfg.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", default=Config.output_csv)
    parser.add_argument("--manifest-output", default=Config.manifest_output)
    parser.add_argument("--start", default=Config.start)
    parser.add_argument("--end-exclusive", default=Config.end_exclusive)
    parser.add_argument("--currency", default=Config.currency)
    parser.add_argument("--settlement-type", default=Config.settlement_type)
    parser.add_argument("--page-size", type=int, default=Config.page_size)
    parser.add_argument("--timeout-sec", type=float, default=Config.timeout_sec)
    parser.add_argument(
        "--request-pause-sec", type=float, default=Config.request_pause_sec
    )
    parser.add_argument(
        "--maximum-retries", type=int, default=Config.maximum_retries
    )
    parser.add_argument(
        "--maximum-event-row-span-seconds",
        type=float,
        default=Config.maximum_event_row_span_seconds,
    )
    return Config(**vars(parser.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
