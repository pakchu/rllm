"""Frozen five-minute source loader for QQQ, KODEX 200, and GLD.

The Investing.com TVC endpoint is an unofficial research source.  Raw provider
JSON is cached locally and cross-checked against recent Yahoo five-minute
closes.  Corporate-action factors come from the frozen Yahoo daily payloads.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training import preregister_cross_asset_5m_transfer as prereg
from training.evaluate_cross_asset_alpha_transfer import download_payload as download_daily_payload


CACHE_DIR = "data/cache_cross_asset_5m_transfer"
YAHOO_DAILY_CACHE = "data/cache_cross_asset_alpha_transfer"
START_UTC = pd.Timestamp("2024-03-05T00:00:00Z")
END_UTC = pd.Timestamp("2026-07-19T00:00:00Z")
YAHOO_OVERLAP_DAYS = 59
OUTPUT = "results/cross_asset_5m_source_audit_2026-07-19.json"
DOCS_OUTPUT = "docs/cross-asset-5m-source-audit-2026-07-19.md"
_DIRECT_BLOCKED = False
US_EARLY_CLOSE_DATES = {
    "2024-07-03",
    "2024-11-29",
    "2024-12-24",
    "2025-07-03",
    "2025-11-28",
    "2025-12-24",
    "2026-07-02",
}
KRX_DELAYED_OPEN_DATES = {
    "2024-11-14",
    "2025-01-02",
    "2025-11-13",
    "2026-01-02",
}
UNUSABLE_PREFIX_THROUGH = {
    "QQQ": "2024-08-05",
    "GLD": "2024-08-05",
}
US_CLOSED_PRINT_DATES = {"2025-12-25"}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def chunk_ranges(
    start: pd.Timestamp = START_UTC,
    end: pd.Timestamp = END_UTC,
    days: int = 45,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("chunk boundaries must be timezone-aware")
    if start >= end or days <= 0:
        raise ValueError("invalid chunk range")
    output: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + pd.Timedelta(days=days), end)
        output.append((cursor, stop))
        cursor = stop
    return output


def provider_url(investing_id: int, start: pd.Timestamp, end: pd.Timestamp) -> str:
    seed = f"CAT-XA-5M-1|{investing_id}|{start.isoformat()}|{end.isoformat()}"
    nonce = hashlib.sha256(seed.encode()).hexdigest()[:32]
    query = urllib.parse.urlencode(
        {
            "symbol": int(investing_id),
            "from": int(start.timestamp()),
            "to": int(end.timestamp()),
            "resolution": 5,
        }
    )
    return f"https://tvc6.investing.com/{nonce}/0/0/0/0/history?{query}"


def _request(url: str, *, timeout: int = 120) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/104.0.5112.102 Safari/537.36"
            ),
            "Referer": "https://tvc-invdn-com.investing.com/",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - frozen public source
        return response.read()


def extract_jina_payload(raw: bytes) -> bytes:
    text = raw.decode("utf-8")
    marker = "Markdown Content:\n"
    if marker not in text:
        raise RuntimeError("Jina response does not contain provider payload")
    body = text.split(marker, 1)[1].strip()
    if body.startswith("```json"):
        body = body[len("```json") :].strip()
    if body.endswith("```"):
        body = body[:-3].strip()
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise RuntimeError("Jina response payload is not a provider object")
    return canonical_json(payload)


def _download_provider_json(url: str) -> tuple[bytes, str]:
    global _DIRECT_BLOCKED
    if not _DIRECT_BLOCKED:
        try:
            raw = _request(url)
            return canonical_json(json.loads(raw)), "direct"
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            _DIRECT_BLOCKED = True
    proxy_url = "https://r.jina.ai/http://" + url.removeprefix("https://")
    waits = (0, 3, 10, 30)
    last_error: Exception | None = None
    for wait in waits:
        if wait:
            time.sleep(wait)
        try:
            return extract_jina_payload(_request(proxy_url)), "jina_read_through"
        except Exception as exc:  # pragma: no cover - network retry surface
            last_error = exc
    raise RuntimeError(f"provider download failed after retries: {url}") from last_error


def download_provider_chunks(
    symbol: str,
    investing_id: int,
    *,
    cache_dir: str = CACHE_DIR,
    refresh: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    directory = Path(cache_dir) / "provider" / symbol
    directory.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[pd.Timestamp, pd.Timestamp, Path, str]] = []
    for start, end in chunk_ranges(days=int(prereg.manifest()["source_contract"]["chunk_calendar_days"])):
        name = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.json"
        cache = directory / name
        url = provider_url(investing_id, start, end)
        entries.append((start, end, cache, url))

    def load_entry(entry: tuple[pd.Timestamp, pd.Timestamp, Path, str]) -> tuple[dict[str, Any], dict[str, Any]]:
        start, end, cache, url = entry
        if cache.is_file() and not refresh:
            raw = cache.read_bytes()
            transport = "cache"
            try:
                cached_payload = json.loads(raw)
            except json.JSONDecodeError:
                cached_payload = None
            if not isinstance(cached_payload, dict) or cached_payload.get("s") != "ok":
                cache.unlink()
                raw, transport = _download_provider_json(url)
        else:
            raw, transport = _download_provider_json(url)
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise RuntimeError(f"provider payload is not an object for {symbol} {cache.name}")
        status = payload.get("s")
        if status != "ok":
            raise RuntimeError(f"provider status for {symbol} {name}: {status}")
        rows = len(payload.get("t") or [])
        if rows <= 0 or rows >= 5000:
            raise RuntimeError(
                f"provider chunk row count is empty or capped for {symbol} {cache.name}: {rows}"
            )
        if transport != "cache":
            cache.write_bytes(raw)
        return payload, {
            "start_utc": start.isoformat(),
            "end_utc_exclusive": end.isoformat(),
            "rows_returned": rows,
            "canonical_sha256": _sha256(raw),
            "payload_format": "canonical_extracted_provider_json",
            "provider_url": url,
        }

    # Make one synchronous request so a direct-source rejection flips the
    # process-wide fallback flag before parallel read-through requests begin.
    output: list[tuple[dict[str, Any], dict[str, Any]] | None] = [None] * len(entries)
    missing = [index for index, (_, _, cache, _) in enumerate(entries) if refresh or not cache.is_file()]
    if missing:
        first = missing.pop(0)
        output[first] = load_entry(entries[first])
    with ThreadPoolExecutor(max_workers=4) as executor:
        loaded = executor.map(load_entry, (entries[index] for index in missing))
        for index, row in zip(missing, loaded):
            output[index] = row
    for index, row in enumerate(output):
        if row is None:
            output[index] = load_entry(entries[index])
    payloads = [row[0] for row in output if row is not None]
    metadata = [row[1] for row in output if row is not None]
    return payloads, metadata


def payload_rows(payload: dict[str, Any]) -> pd.DataFrame:
    keys = ("t", "o", "h", "l", "c", "v")
    vectors = {key: payload.get(key) for key in keys}
    lengths = {key: len(value) if isinstance(value, list) else -1 for key, value in vectors.items()}
    if len(set(lengths.values())) != 1 or next(iter(lengths.values())) <= 0:
        raise RuntimeError(f"provider vector mismatch: {lengths}")
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(vectors["t"], unit="s", utc=True),
            "raw_open": vectors["o"],
            "raw_high": vectors["h"],
            "raw_low": vectors["l"],
            "raw_close": vectors["c"],
            "volume": vectors["v"],
        }
    )
    for column in ("raw_open", "raw_high", "raw_low", "raw_close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _minute(value: str) -> int:
    hour, minute = (int(part) for part in value.split(":"))
    return hour * 60 + minute


def normalize_provider_rows(
    payloads: Iterable[dict[str, Any]],
    *,
    symbol: str,
    timezone: str,
    regular_session: tuple[str, str] | list[str],
    start_utc: pd.Timestamp = START_UTC,
    end_utc: pd.Timestamp = END_UTC,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = [payload_rows(payload) for payload in payloads]
    frame = pd.concat(frames, ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)
    frame = frame.loc[(frame["timestamp_utc"] >= start_utc) & (frame["timestamp_utc"] < end_utc)].copy()
    boundary_duplicates_removed = 0
    if frame["timestamp_utc"].duplicated().any():
        duplicate_rows = frame.loc[frame["timestamp_utc"].duplicated(keep=False)]
        allowed = {row[0] for row in chunk_ranges(start_utc, end_utc)[1:]}
        for timestamp_value, group in duplicate_rows.groupby("timestamp_utc"):
            if timestamp_value not in allowed or len(group.drop(columns="timestamp_utc").drop_duplicates()) != 1:
                raise RuntimeError(f"non-boundary or conflicting duplicate provider timestamp for {symbol}")
        before = len(frame)
        frame = frame.drop_duplicates("timestamp_utc", keep="last")
        boundary_duplicates_removed = before - len(frame)
    if frame.empty:
        raise RuntimeError(f"empty provider timestamps for {symbol}")
    timestamp = frame["timestamp_utc"]
    if not ((timestamp.dt.second == 0) & (timestamp.dt.microsecond == 0) & (timestamp.dt.minute % 5 == 0)).all():
        raise RuntimeError(f"non-five-minute provider timestamp for {symbol}")
    local = timestamp.dt.tz_convert(timezone)
    frame["date"] = local.dt.tz_localize(None)
    frame["session_date"] = frame["date"].dt.normalize()
    minute = frame["date"].dt.hour * 60 + frame["date"].dt.minute
    session_start, session_end = (_minute(value) for value in regular_session)
    frame = frame.loc[(minute >= session_start) & (minute < session_end)].copy()

    invalid = ~np.isfinite(frame[["raw_open", "raw_high", "raw_low", "raw_close", "volume"]]).all(axis=1)
    invalid |= ~(frame[["raw_open", "raw_high", "raw_low", "raw_close"]] > 0.0).all(axis=1)
    invalid |= frame["volume"] < 0.0
    discarded_prefix_rows = 0
    discarded_prefix_through: str | None = None
    invalid_prefix_rows = int(invalid.sum())
    configured_prefix = pd.Timestamp(UNUSABLE_PREFIX_THROUGH[symbol]) if symbol in UNUSABLE_PREFIX_THROUGH else None
    if invalid.any() or configured_prefix is not None:
        last_invalid_session = frame.loc[invalid, "session_date"].max() if invalid.any() else configured_prefix
        if configured_prefix is not None:
            last_invalid_session = max(last_invalid_session, configured_prefix)
        train_start = pd.Timestamp(prereg.manifest()["calendar_contract"]["train"][0])
        if last_invalid_session >= train_start:
            raise RuntimeError(f"invalid provider OHLCV at or after train start for {symbol}")
        keep = frame["session_date"] > last_invalid_session
        discarded_prefix_rows = int((~keep).sum())
        discarded_prefix_through = last_invalid_session.strftime("%Y-%m-%d")
        frame = frame.loc[keep].copy()

    values = frame[["raw_open", "raw_high", "raw_low", "raw_close", "volume"]]
    if not np.isfinite(values).all(axis=None):
        raise RuntimeError(f"non-finite provider OHLCV after prefix recovery for {symbol}")
    if not (values[["raw_open", "raw_high", "raw_low", "raw_close"]] > 0.0).all(axis=None):
        raise RuntimeError(f"nonpositive provider OHLC after prefix recovery for {symbol}")
    if not (values["volume"] >= 0.0).all():
        raise RuntimeError(f"negative provider volume after prefix recovery for {symbol}")
    high_floor = values[["raw_open", "raw_close"]].max(axis=1)
    low_ceiling = values[["raw_open", "raw_close"]].min(axis=1)
    high_shortfall_bps = ((high_floor / values["raw_high"] - 1.0).clip(lower=0.0) * 10_000.0)
    low_excess_bps = ((values["raw_low"] / low_ceiling - 1.0).clip(lower=0.0) * 10_000.0)
    geometry_repair = (high_shortfall_bps > 0.0) | (low_excess_bps > 0.0)
    if max(float(high_shortfall_bps.max()), float(low_excess_bps.max())) > 0.50:
        raise RuntimeError(f"provider OHLC geometry failure beyond rounding tolerance for {symbol}")
    frame["raw_high"] = np.maximum(frame["raw_high"], high_floor)
    frame["raw_low"] = np.minimum(frame["raw_low"], low_ceiling)

    # Provider half-day payloads can append one or two non-tradable closing
    # prints.  Keep only the 42 continuous 09:30..12:55 bars on the exchange's
    # frozen early-close dates.
    drop_index: list[int] = []
    if timezone == "America/New_York":
        for session, group in frame.groupby("session_date", sort=True):
            if session.strftime("%Y-%m-%d") not in US_EARLY_CLOSE_DATES:
                continue
            expected = pd.date_range(session + pd.Timedelta(hours=9, minutes=30), periods=42, freq="5min")
            observed = pd.DatetimeIndex(group.sort_values("date")["date"].iloc[:42])
            if len(observed) != 42 or not observed.equals(expected):
                raise RuntimeError(f"incomplete U.S. early-close session for {symbol} on {session.date()}")
            drop_index.extend(int(index) for index in group.index[42:])
        closed_print = frame["session_date"].dt.strftime("%Y-%m-%d").isin(US_CLOSED_PRINT_DATES)
        drop_index.extend(int(index) for index in frame.index[closed_print])
    if drop_index:
        frame = frame.drop(index=sorted(set(drop_index)))

    # KRX delayed-open days contain two provider pre-open indication rows at
    # 09:30/09:35.  They are not continuous-session tradable bars.
    if timezone == "Asia/Seoul":
        preopen = frame["session_date"].dt.strftime("%Y-%m-%d").isin(KRX_DELAYED_OPEN_DATES)
        preopen &= (frame["date"].dt.hour * 60 + frame["date"].dt.minute) < 10 * 60
        drop_index.extend(int(index) for index in frame.index[preopen])
        frame = frame.loc[~preopen].copy()

    short_sessions: list[str] = []
    counts: dict[str, int] = {}
    no_bar_gaps: list[dict[str, Any]] = []
    for session, group in frame.groupby("session_date", sort=True):
        group = group.sort_values("date")
        gaps = group["date"].diff().dropna()
        first = group["date"].iloc[0].strftime("%H:%M")
        last = group["date"].iloc[-1].strftime("%H:%M")
        count = len(group)
        counts[str(count)] = counts.get(str(count), 0) + 1
        if timezone == "America/New_York":
            if len(gaps) and not gaps.eq(pd.Timedelta(minutes=5)).all():
                raise RuntimeError(f"missing interior regular-session bar for {symbol} on {session.date()}")
            valid = (count, first, last) in {(78, "09:30", "15:55"), (42, "09:30", "12:55")}
        else:
            large = gaps[gaps > pd.Timedelta(minutes=5)]
            if len(large) and (
                not (large.dt.total_seconds() % 300 == 0).all()
                or large.max() > pd.Timedelta(minutes=30)
            ):
                raise RuntimeError(f"unexplained KRX intraday gap for {symbol} on {session.date()}")
            for index, gap in large.items():
                no_bar_gaps.append(
                    {
                        "session_date": session.strftime("%Y-%m-%d"),
                        "previous_bar": group.loc[group.index[group.index.get_loc(index) - 1], "date"].strftime(
                            "%Y-%m-%dT%H:%M:%S"
                        ),
                        "next_bar": group.loc[index, "date"].strftime("%Y-%m-%dT%H:%M:%S"),
                        "gap_minutes": int(gap.total_seconds() // 60),
                    }
                )
            valid = first == "09:00" and last == "15:15" and count <= 76 or (
                session.strftime("%Y-%m-%d") in KRX_DELAYED_OPEN_DATES
                and first == "10:00"
                and last == "15:15"
                and count <= 64
            )
        if not valid:
            raise RuntimeError(
                f"unexpected regular session for {symbol} on {session.date()}: {count} {first}-{last}"
            )
        full_count = 78 if timezone == "America/New_York" else 76
        if count != full_count:
            short_sessions.append(session.strftime("%Y-%m-%d"))
    frame = frame.sort_values("timestamp_utc").reset_index(drop=True)
    return frame, {
        "rows_regular_session": len(frame),
        "sessions": int(frame["session_date"].nunique()),
        "session_row_count_histogram": counts,
        "short_session_dates": short_sessions,
        "provider_stable_no_bar_gap_count": len(no_bar_gaps),
        "provider_stable_no_bar_gaps": no_bar_gaps,
        "provider_stable_no_bar_gaps_sha256": _sha256(canonical_json(no_bar_gaps)),
        "invalid_prefix_rows": invalid_prefix_rows,
        "discarded_unusable_prefix_rows": discarded_prefix_rows,
        "discarded_unusable_prefix_through": discarded_prefix_through,
        "nontradable_or_closed_prints_dropped": len(set(drop_index)),
        "identical_chunk_boundary_duplicates_removed": boundary_duplicates_removed,
        "geometry_rounding_rows_repaired": int(geometry_repair.sum()),
        "geometry_max_repair_bps": max(
            float(high_shortfall_bps.max()), float(low_excess_bps.max())
        ),
        "start_utc": frame["timestamp_utc"].iloc[0].isoformat(),
        "end_utc": frame["timestamp_utc"].iloc[-1].isoformat(),
    }


def _daily_adjustment_factors(symbol: str, *, cache_dir: str = YAHOO_DAILY_CACHE) -> tuple[pd.Series, dict[str, Any]]:
    yahoo_symbol = "069500.KS" if symbol == "069500" else symbol
    raw, mode = download_daily_payload(yahoo_symbol, cache_dir, refresh=False)
    payload = json.loads(raw)
    result = (payload.get("chart", {}).get("result") or [None])[0]
    if result is None:
        raise RuntimeError(f"missing Yahoo daily result for {symbol}")
    quote = ((result.get("indicators") or {}).get("quote") or [None])[0]
    adjusted = ((result.get("indicators") or {}).get("adjclose") or [None])[0]
    if quote is None or adjusted is None:
        raise RuntimeError(f"missing Yahoo daily adjustment vectors for {symbol}")
    timestamps = result.get("timestamp") or []
    closes = pd.to_numeric(pd.Series(quote.get("close")), errors="coerce")
    adjclose = pd.to_numeric(pd.Series(adjusted.get("adjclose")), errors="coerce")
    timezone = str((result.get("meta") or {}).get("exchangeTimezoneName"))
    dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(timezone).tz_localize(None).normalize()
    factor = pd.Series((adjclose / closes.replace(0.0, np.nan)).to_numpy(), index=dates)
    factor = factor.loc[~factor.index.duplicated(keep="last")].dropna()
    if factor.empty or not np.isfinite(factor).all() or not (factor > 0.0).all():
        raise RuntimeError(f"invalid Yahoo daily adjustment factor for {symbol}")
    events = result.get("events") or {}
    event_dates: set[pd.Timestamp] = set()
    for family in ("dividends", "splits"):
        for event in (events.get(family) or {}).values():
            if "date" in event:
                event_dates.add(pd.to_datetime(int(event["date"]), unit="s", utc=True).tz_localize(None).normalize())
    factor.attrs["corporate_action_dates"] = sorted(event_dates)
    return factor, {
        "yahoo_symbol": yahoo_symbol,
        "raw_sha256": _sha256(raw),
        "factor_min": float(factor.min()),
        "factor_max": float(factor.max()),
        "corporate_action_dates_sha256": _sha256(
            canonical_json([value.strftime("%Y-%m-%d") for value in sorted(event_dates)])
        ),
    }


def apply_daily_adjustment(
    frame: pd.DataFrame,
    factors: pd.Series,
) -> pd.DataFrame:
    output = frame.copy()
    mapped = output["session_date"].map(factors)
    if mapped.isna().any():
        session_dates = pd.DatetimeIndex(output["session_date"].unique()).sort_values()
        aligned = factors.reindex(factors.index.union(session_dates)).sort_index()
        previous = aligned.ffill().reindex(session_dates)
        following = aligned.bfill().reindex(session_dates)
        missing_sessions = pd.DatetimeIndex(output.loc[mapped.isna(), "session_date"].unique())
        approximately_equal = np.isclose(
            previous.reindex(missing_sessions).to_numpy(float),
            following.reindex(missing_sessions).to_numpy(float),
            rtol=0.0,
            atol=2e-6,
            equal_nan=False,
        )
        action_dates = set(factors.attrs.get("corporate_action_dates", []))
        action_bridge = np.asarray([value in action_dates for value in missing_sessions], dtype=bool)
        safe = approximately_equal | action_bridge
        if not safe.all():
            dates = missing_sessions[~safe].strftime("%Y-%m-%d").tolist()
            raise RuntimeError(f"daily adjustment gap crosses a factor change: {dates[:5]}")
        bridge_values = previous.reindex(missing_sessions).to_numpy(float)
        bridge_values[action_bridge] = following.reindex(missing_sessions).to_numpy(float)[action_bridge]
        bridge = pd.Series(bridge_values, index=missing_sessions)
        mapped = mapped.fillna(output["session_date"].map(bridge))
    output["adjustment_factor_bridged"] = ~output["session_date"].isin(factors.index)
    output["adjustment_factor"] = mapped.to_numpy(float)
    for source, target in (
        ("raw_open", "open"),
        ("raw_high", "high"),
        ("raw_low", "low"),
        ("raw_close", "close"),
    ):
        output[target] = output[source] * output["adjustment_factor"]
    return output


def yahoo_5m_url(symbol: str) -> str:
    yahoo_symbol = "069500.KS" if symbol == "069500" else symbol
    start = END_UTC - pd.Timedelta(days=YAHOO_OVERLAP_DAYS)
    query = urllib.parse.urlencode(
        {
            "period1": int(start.timestamp()),
            "period2": int(END_UTC.timestamp()),
            "interval": "5m",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
    )
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(yahoo_symbol, safe='')}?{query}"


def load_yahoo_5m_close(
    symbol: str,
    *,
    cache_dir: str = CACHE_DIR,
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache = Path(cache_dir) / "yahoo_5m" / f"{symbol}.json"
    if cache.is_file() and not refresh:
        raw = cache.read_bytes()
        mode = "cache"
    else:
        raw = _request(yahoo_5m_url(symbol), timeout=60)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)
        mode = "download"
    payload = json.loads(raw)
    chart = payload.get("chart", {})
    if chart.get("error") is not None:
        raise RuntimeError(f"Yahoo five-minute error for {symbol}: {chart['error']}")
    result = (chart.get("result") or [None])[0]
    if result is None:
        raise RuntimeError(f"Yahoo five-minute result missing for {symbol}")
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [None])[0]
    if quote is None or len(quote.get("close") or []) != len(timestamps):
        raise RuntimeError(f"Yahoo five-minute vector mismatch for {symbol}")
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(timestamps, unit="s", utc=True),
            "yahoo_close": pd.to_numeric(quote["close"], errors="coerce"),
        }
    ).dropna()
    if frame.empty or frame["timestamp_utc"].duplicated().any():
        raise RuntimeError(f"Yahoo five-minute timestamps invalid for {symbol}")
    return frame, {"source_url": yahoo_5m_url(symbol), "raw_sha256": _sha256(raw)}


def cross_source_parity(
    provider: pd.DataFrame,
    yahoo: pd.DataFrame,
) -> dict[str, Any]:
    joined = provider[["timestamp_utc", "raw_close"]].merge(yahoo, on="timestamp_utc", how="inner")
    if len(joined) < 1000:
        raise RuntimeError(f"insufficient Yahoo/provider matching bars: {len(joined)}")
    difference = (joined["raw_close"] / joined["yahoo_close"] - 1.0).abs() * 10_000.0
    median = float(difference.median())
    p95 = float(difference.quantile(0.95))
    control = prereg.manifest()["source_contract"]["cross_source_control"]
    if median > float(control["median_absolute_difference_bps_at_most"]):
        raise RuntimeError(f"provider/Yahoo median close difference failed: {median:.4f} bp")
    if p95 > float(control["p95_absolute_difference_bps_at_most"]):
        raise RuntimeError(f"provider/Yahoo p95 close difference failed: {p95:.4f} bp")
    return {
        "matching_bars": len(joined),
        "median_absolute_difference_bps": median,
        "p95_absolute_difference_bps": p95,
        "max_absolute_difference_bps": float(difference.max()),
        "start_utc": joined["timestamp_utc"].iloc[0].isoformat(),
        "end_utc": joined["timestamp_utc"].iloc[-1].isoformat(),
    }


def load_instrument(
    symbol: str,
    *,
    cache_dir: str = CACHE_DIR,
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = prereg.manifest()["instruments"][symbol]
    payloads, chunks = download_provider_chunks(
        symbol, int(spec["investing_id"]), cache_dir=cache_dir, refresh=refresh
    )
    frame, session_meta = normalize_provider_rows(
        payloads,
        symbol=symbol,
        timezone=str(spec["exchange_timezone"]),
        regular_session=spec["regular_session"],
    )
    yahoo, yahoo_meta = load_yahoo_5m_close(symbol, cache_dir=cache_dir, refresh=refresh)
    parity = cross_source_parity(frame, yahoo)
    factors, adjustment_meta = _daily_adjustment_factors(symbol)
    frame = apply_daily_adjustment(frame, factors)
    bridged_dates = (
        frame.loc[frame["adjustment_factor_bridged"], "session_date"]
        .dt.strftime("%Y-%m-%d")
        .drop_duplicates()
        .tolist()
    )
    adjustment_meta["bridged_session_dates"] = bridged_dates
    adjustment_meta["bridged_session_dates_sha256"] = _sha256(canonical_json(bridged_dates))
    combined_hash = _sha256(canonical_json([row["canonical_sha256"] for row in chunks]))
    meta = {
        "provider": "Investing.com TVC chart service",
        "provider_status": "unofficial research source",
        "investing_id": int(spec["investing_id"]),
        "provider_chunk_count": len(chunks),
        "provider_chunk_hashes_sha256": combined_hash,
        "provider_chunks": chunks,
        "session_integrity": session_meta,
        "yahoo_5m_control_source": yahoo_meta,
        "cross_source_parity": parity,
        "daily_adjustment": adjustment_meta,
    }
    return frame, meta


def render_docs(report: dict[str, Any]) -> str:
    lines = [
        "# Cross-asset five-minute source audit — 2026-07-19",
        "",
        f"Preregistration: `{report['preregistration_manifest_hash']}`",
        "",
        "The strategy universe is QQQ, KODEX 200, and GLD only. No KOSPI price or signal is used.",
        "",
        "| Asset | 5m rows | Sessions | Range (UTC) | Yahoo matches | Median / p95 close diff |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for symbol, meta in report["instruments"].items():
        session = meta["session_integrity"]
        parity = meta["cross_source_parity"]
        lines.append(
            f"| {symbol} | {session['rows_regular_session']:,} | {session['sessions']} | "
            f"{session['start_utc']} → {session['end_utc']} | {parity['matching_bars']:,} | "
            f"{parity['median_absolute_difference_bps']:.3f} / "
            f"{parity['p95_absolute_difference_bps']:.3f} bp |"
        )
    lines += [
        "",
        "## Limitation",
        "",
        "Investing.com TVC is an unofficial source and is not suitable as a production market-data contract.",
        "The committed artifact records canonical chunk hashes; extracted raw payloads remain local.",
        "Production replication should use an entitled broker feed such as IBKR or KIS and re-run parity checks.",
    ]
    return "\n".join(lines) + "\n"


def run_source_audit(
    *,
    output: str = OUTPUT,
    docs_output: str = DOCS_OUTPUT,
    cache_dir: str = CACHE_DIR,
    refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frozen = prereg.manifest()
    if not Path(prereg.OUTPUT).is_file() or json.loads(Path(prereg.OUTPUT).read_text()) != frozen:
        raise RuntimeError("five-minute preregistration artifact mismatch")
    frames: dict[str, pd.DataFrame] = {}
    instruments: dict[str, Any] = {}
    for symbol in frozen["instruments"]:
        frame, meta = load_instrument(symbol, cache_dir=cache_dir, refresh=refresh)
        frames[symbol] = frame
        instruments[symbol] = meta
    report: dict[str, Any] = {
        "schema_version": 1,
        "research_id": "CAT-XA-5M-SOURCE-1",
        "preregistration_manifest_hash": frozen["manifest_hash"],
        "period_start_utc": START_UTC.isoformat(),
        "period_end_exclusive_utc": END_UTC.isoformat(),
        "instruments": instruments,
    }
    report["result_hash"] = _sha256(canonical_json(report))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(docs_output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    Path(docs_output).write_text(render_docs(report))
    return frames, report


def main() -> None:
    _, report = run_source_audit()
    compact = {
        symbol: {
            "rows": row["session_integrity"]["rows_regular_session"],
            "sessions": row["session_integrity"]["sessions"],
            "parity": row["cross_source_parity"],
        }
        for symbol, row in report["instruments"].items()
    }
    print(json.dumps({"output": OUTPUT, "summary": compact}, indent=2))


if __name__ == "__main__":
    main()
