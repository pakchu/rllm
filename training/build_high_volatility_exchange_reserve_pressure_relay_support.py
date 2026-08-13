"""Build outcome-blind source support for frozen HVEXRP-24.

This module deliberately stops at source incidence.  It downloads the current
Coin Metrics vintage, reads only completed BTC minute ``ts/open/close`` bars,
and never opens an entry/exit price, return, PnL, funding, or comparison set.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from training import preregister_high_volatility_exchange_reserve_pressure_relay as prereg


POLICY_ID = "HVEXRP-24"
PREREG_SHA256 = "bb1ed431c3d1fe28235f16d7e53318382999ecb74dbf41ad5cac44561b2fb0d0"
ENDPOINT = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
METRICS = ("SplyExNtv", "AssetEODCompletionTime")
STATUS_KEYS = frozenset({"SplyExNtv-status", "SplyExNtv-status-time"})
RAW_ROW_KEYS = frozenset({"asset", "time", *METRICS, *STATUS_KEYS})
SOURCE_START = pd.Timestamp("2022-01-01T00:00:00Z")
SOURCE_END = pd.Timestamp("2026-07-30T00:00:00Z")  # exclusive; July 29 included
PAGE_SIZE = 10_000
ENV_FILE = "/home/pakchu/rllm/.env"
BTC_QUERY = """SELECT ts,open,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""
BTC_QUERY_START = SOURCE_START
BTC_QUERY_END = SOURCE_END + pd.Timedelta(hours=12)

SPLITS = {
    name: tuple(pd.Timestamp(value) for value in bounds)
    for name, bounds in prereg.build()["stages"].items()
}
GATES = prereg.build()["source_support_gates"]
CONTROLS = tuple(prereg.build()["diagnostic_controls"]["names"])

SOURCE_DIR = Path("data/high_volatility_exchange_reserve_pressure_relay_sources_2022_2026")
DAILY_SOURCE = SOURCE_DIR / "coinmetrics_btc_daily.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "daily_feature_panel.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_exchange_reserve_pressure_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_exchange_reserve_pressure_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_exchange_reserve_pressure_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_exchange_reserve_pressure_relay_support_2026-08-13.json")

SOURCE_COLUMNS = (
    "observation_time", "feature_available_time", "SplyExNtv",
)
PANEL_COLUMNS = (
    "observation_time", "feature_available_time", "decision_time", "source_valid",
    "minute_count", "SplyExNtv", "reserve_log_change", "reserve_level_rank",
    "btc_variation", "btc_variation_rank",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "observation_time", "feature_available_time",
    "decision_time", "entry_time", "exit_time", "side", "SplyExNtv",
    "reserve_log_change", "reserve_level_rank", "btc_variation", "btc_variation_rank",
)
UTC_TIMESTAMP_RE = re.compile(
    r"^(?P<head>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d+))?(?P<offset>Z|\+00:00)$"
)
Fetch = Callable[[str], dict[str, Any]]


def canonical_hash(payload: Any) -> str:
    def normalize(value: Any) -> Any:
        if isinstance(value, Decimal):
            return {"__exact_json_decimal__": str(value)}
        if isinstance(value, (pd.Timestamp, datetime)):
            return pd.Timestamp(value).isoformat()
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    encoded = json.dumps(
        normalize(payload), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_url() -> str:
    params = {
        "assets": "btc",
        "metrics": ",".join(METRICS),
        "frequency": "1d",
        "start_time": "2022-01-01",
        "end_time": "2026-07-29",
        "page_size": str(PAGE_SIZE),
    }
    return f"{ENDPOINT}?{urllib.parse.urlencode(params)}"


def _validate_next_page_url(first_url: str, candidate: str) -> str:
    resolved = urllib.parse.urljoin(first_url, candidate)
    first = urllib.parse.urlparse(first_url)
    next_page = urllib.parse.urlparse(resolved)
    if (
        next_page.scheme != first.scheme
        or next_page.netloc != first.netloc
        or next_page.path != first.path
        or next_page.params
        or next_page.fragment
    ):
        raise ValueError("Coin Metrics next_page_url left the frozen endpoint")
    first_query = urllib.parse.parse_qs(first.query, keep_blank_values=True, strict_parsing=True)
    next_query = urllib.parse.parse_qs(next_page.query, keep_blank_values=True, strict_parsing=True)
    frozen = {"assets", "metrics", "frequency", "start_time", "end_time", "page_size"}
    if any(next_query.get(key) != first_query.get(key) for key in frozen):
        raise ValueError("Coin Metrics next_page_url changed the frozen query")
    if set(next_query) != frozen | {"next_page_token"}:
        raise ValueError("Coin Metrics next_page_url has unexpected query fields")
    token = next_query.get("next_page_token")
    if token is None or len(token) != 1 or not token[0]:
        raise ValueError("Coin Metrics next_page_url has an invalid page token")
    return resolved


def _parse_utc(value: Any, name: str) -> pd.Timestamp:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an exact UTC timestamp")
    match = UTC_TIMESTAMP_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"{name} must be an exact ISO-8601 UTC timestamp")
    fraction = match.group("fraction") or ""
    if len(fraction) > 6 and any(digit != "0" for digit in fraction[6:]):
        raise ValueError(f"{name} has unsupported non-zero sub-microseconds")
    text = f"{match.group('head')}.{fraction[:6].ljust(6, '0')}+00:00"
    try:
        return pd.Timestamp(datetime.fromisoformat(text).astimezone(timezone.utc))
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid UTC timestamp") from exc


def _positive_decimal(value: Any, name: str, *, integer: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be positive and finite")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError(f"{name} must be positive and finite")
    if integer and number != number.to_integral_value():
        raise ValueError(f"{name} must be a positive integer")
    return number


def _completion_timestamp(value: Any) -> pd.Timestamp:
    seconds = _positive_decimal(value, "AssetEODCompletionTime")
    whole = seconds.to_integral_value(rounding=ROUND_FLOOR)
    micros = (seconds - whole) * Decimal(1_000_000)
    if micros != micros.to_integral_value():
        raise ValueError("AssetEODCompletionTime has unsupported sub-microsecond precision")
    try:
        value = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
            seconds=int(whole), microseconds=int(micros)
        )
    except (OverflowError, ValueError) as exc:
        raise ValueError("AssetEODCompletionTime is outside the UTC range") from exc
    return pd.Timestamp(value)


def parse_source_row(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or frozenset(raw) != RAW_ROW_KEYS:
        keys = frozenset(raw) if isinstance(raw, dict) else frozenset()
        raise ValueError(
            "Coin Metrics row schema drift: "
            f"missing={sorted(RAW_ROW_KEYS - keys)!r} unexpected={sorted(keys - RAW_ROW_KEYS)!r}"
        )
    if raw["asset"] != "btc":
        raise ValueError("Coin Metrics row asset must be exactly 'btc'")
    if not isinstance(raw["SplyExNtv-status"], str) or not raw["SplyExNtv-status"].strip():
        raise ValueError("SplyExNtv-status must be a non-empty string")
    _parse_utc(raw["SplyExNtv-status-time"], "SplyExNtv-status-time")
    observation = _parse_utc(raw["time"], "time")
    if observation != observation.floor("1d") or not SOURCE_START <= observation < SOURCE_END:
        raise ValueError("Coin Metrics observation is not a frozen-interval UTC day")
    available = _completion_timestamp(raw["AssetEODCompletionTime"])
    reserve = _positive_decimal(raw["SplyExNtv"], "SplyExNtv")
    return {
        "observation_time": observation,
        "feature_available_time": available,
        "SplyExNtv": str(reserve),
    }


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Coin Metrics response used non-standard JSON value {value}")


def _http_page(url: str, *, timeout: float = 30.0, retries: int = 8) -> dict[str, Any]:
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": "rllm-private-research/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(
                    response.read().decode("utf-8"), parse_float=Decimal,
                    parse_constant=_reject_json_constant,
                )
            if not isinstance(payload, dict):
                raise RuntimeError("Coin Metrics response is not an object")
            return payload
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code <= 599
            if not retryable or attempt >= retries:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt >= retries:
                raise
        time.sleep(min(60.0, 2.0**attempt))
    raise AssertionError("unreachable retry loop")


def download_coinmetrics(
    *, fetch: Fetch | None = None, sleep: Callable[[float], None] = time.sleep
) -> tuple[pd.DataFrame, dict[str, Any]]:
    first_url = source_url()
    fetch = fetch or _http_page
    url: str | None = first_url
    seen_urls: set[str] = set()
    by_day: dict[pd.Timestamp, dict[str, Any]] = {}
    page_hashes: list[str] = []
    page_lengths: list[int] = []
    expected = pd.date_range(SOURCE_START, SOURCE_END, freq="1d", inclusive="left")
    maximum_pages = math.ceil(len(expected) / PAGE_SIZE) + 4
    while url is not None:
        if url in seen_urls:
            raise RuntimeError("Coin Metrics pagination loop detected")
        if len(page_hashes) >= maximum_pages:
            raise RuntimeError("Coin Metrics pagination exceeded the frozen maximum")
        seen_urls.add(url)
        payload = fetch(url)
        if not isinstance(payload, dict):
            raise RuntimeError("Coin Metrics response is not an object")
        if payload.get("error") is not None:
            raise RuntimeError(f"Coin Metrics API error: {payload['error']!r}")
        data = payload.get("data")
        if not isinstance(data, list) or len(data) > PAGE_SIZE:
            raise ValueError("Coin Metrics data field or page size is invalid")
        for raw in data:
            row = parse_source_row(raw)
            day = row["observation_time"]
            if day in by_day:
                raise RuntimeError(f"Coin Metrics duplicate day {day.isoformat()}")
            by_day[day] = row
        next_url = payload.get("next_page_url")
        if next_url is None:
            if frozenset(payload) != {"data"}:
                raise ValueError("Coin Metrics terminal response schema drift")
            url = None
        elif isinstance(next_url, str) and next_url:
            if frozenset(payload) != {"data", "next_page_token", "next_page_url"}:
                raise ValueError("Coin Metrics paginated response schema drift")
            if not data:
                raise RuntimeError("Coin Metrics returned an empty non-terminal page")
            url = _validate_next_page_url(first_url, next_url)
            token = urllib.parse.parse_qs(
                urllib.parse.urlparse(url).query, keep_blank_values=True, strict_parsing=True
            ).get("next_page_token")
            if not isinstance(payload["next_page_token"], str) or token != [payload["next_page_token"]]:
                raise ValueError("Coin Metrics response and URL page tokens disagree")
        else:
            raise ValueError("Coin Metrics next_page_url must be a non-empty string or null")
        page_hashes.append(canonical_hash(payload))
        page_lengths.append(len(data))
        if url is not None:
            sleep(0.2)
    observed = pd.DatetimeIndex(sorted(by_day))
    if not observed.equals(expected):
        missing = expected.difference(observed)
        extra = observed.difference(expected)
        raise RuntimeError(
            f"Coin Metrics daily source incomplete: missing={list(missing[:5])!r} extra={list(extra[:5])!r}"
        )
    frame = pd.DataFrame([by_day[day] for day in expected], columns=SOURCE_COLUMNS)
    audit = {
        "endpoint": ENDPOINT,
        "source_url": first_url,
        "metrics": list(METRICS),
        "start_time": "2022-01-01",
        "end_time_inclusive": "2026-07-29",
        "response_pages": len(page_hashes),
        "response_page_lengths": page_lengths,
        "response_page_sha256": page_hashes,
        "response_chain_sha256": canonical_hash(page_hashes),
        "expected_rows": len(expected),
        "observed_rows": len(frame),
        "current_vintage_not_historical_revision_archive": True,
        "duplicates": 0,
        "missing_days": 0,
    }
    return frame, audit


def strict_prior_midrank(
    values: pd.Series, maximum: int = 180, minimum: int = 120
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(numeric), np.nan)
    history: list[float] = []
    for index, current in enumerate(numeric):
        prior = np.asarray(history[-maximum:], dtype=float)
        if math.isfinite(current) and len(prior) >= minimum:
            output[index] = ((prior < current).sum() + 0.5 * (prior == current).sum()) / len(prior)
        if math.isfinite(current):
            history.append(float(current))
    return pd.Series(output, index=values.index, dtype=float)


def prepare_btc_bars(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.columns.tolist() != ["ts", "open", "close"]:
        raise RuntimeError("BTC source schema drift; only ts/open/close are permitted")
    frame = raw.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if frame.ts.isna().any() or frame.ts.duplicated(keep=False).any():
        raise RuntimeError("BTC source has an invalid or duplicate timestamp")
    return frame.set_index("ts").sort_index()


def build_panel(source: pd.DataFrame, btc_raw: pd.DataFrame) -> pd.DataFrame:
    if source.columns.tolist() != list(SOURCE_COLUMNS):
        raise RuntimeError("normalized Coin Metrics source schema drift")
    daily = source.copy()
    daily["observation_time"] = pd.to_datetime(daily.observation_time, utc=True, errors="coerce")
    daily["feature_available_time"] = pd.to_datetime(daily.feature_available_time, utc=True, errors="coerce")
    daily["SplyExNtv"] = pd.to_numeric(daily.SplyExNtv, errors="coerce")
    expected = pd.date_range(SOURCE_START, SOURCE_END, freq="1d", inclusive="left")
    if (
        daily.observation_time.isna().any()
        or daily.observation_time.duplicated(keep=False).any()
        or not pd.DatetimeIndex(daily.observation_time).equals(expected)
    ):
        raise RuntimeError("normalized Coin Metrics source is not the exact frozen daily grid")
    decision = daily.feature_available_time.dt.ceil("5min")
    completion_valid = (
        daily.feature_available_time.gt(daily.observation_time + pd.Timedelta(days=1))
        & daily.feature_available_time.le(
            daily.observation_time + pd.Timedelta(days=1, hours=12)
        )
    )
    value_valid = np.isfinite(daily.SplyExNtv) & daily.SplyExNtv.gt(0)
    daily_row_valid = completion_valid & value_valid
    consecutive_prior = daily.observation_time.diff().eq(pd.Timedelta(days=1))
    pair_valid = daily_row_valid & daily_row_valid.shift(1, fill_value=False) & consecutive_prior
    reserve_change = np.log(daily.SplyExNtv / daily.SplyExNtv.shift(1)).where(pair_valid)

    btc = prepare_btc_bars(btc_raw)
    variation: list[float] = []
    minute_count: list[int] = []
    variation_valid: list[bool] = []
    for at in decision:
        if pd.isna(at):
            minute_count.append(0)
            variation.append(math.nan)
            variation_valid.append(False)
            continue
        minutes = pd.date_range(at - pd.Timedelta(hours=24), at, freq="1min", inclusive="left")
        window = btc.reindex(minutes)
        good_rows = (
            np.isfinite(window[["open", "close"]]).all(axis=1)
            & window.open.gt(0) & window.close.gt(0)
        )
        count = int(good_rows.sum())
        good = len(window) == 1440 and count == 1440
        value = math.nan
        if good:
            log_returns = np.log(window.close.to_numpy(float) / window.open.to_numpy(float))
            value = float(np.sqrt(np.square(log_returns).sum()))
            good = math.isfinite(value) and value > 0
        minute_count.append(count)
        variation.append(value)
        variation_valid.append(good)

    base_valid = pair_valid & pd.Series(variation_valid, index=daily.index)
    panel = pd.DataFrame({
        "observation_time": daily.observation_time,
        "feature_available_time": daily.feature_available_time,
        "decision_time": decision,
        "source_valid": base_valid.astype(bool),
        "minute_count": minute_count,
        "SplyExNtv": daily.SplyExNtv.astype(float),
        "reserve_log_change": reserve_change,
        "btc_variation": variation,
    })
    panel["reserve_level_rank"] = strict_prior_midrank(
        panel.SplyExNtv.where(panel.source_valid)
    )
    panel["btc_variation_rank"] = strict_prior_midrank(panel.btc_variation.where(panel.source_valid))
    return panel.loc[:, PANEL_COLUMNS]


def active(panel: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(f"unknown control {control!r}")
    used = panel.copy()
    signal = used.reserve_log_change
    available = used.feature_available_time
    valid = used.source_valid.eq(True)
    if control == "one_day_stale_reserve_change":
        signal = used.reserve_log_change.shift(1)
        available = used.feature_available_time.shift(1)
        valid = valid & signal.notna()
    elif control == "exchange_reserve_level_rank":
        signal = used.reserve_level_rank - 0.5
        valid = valid & signal.notna()
    volatility = (
        pd.Series(True, index=panel.index)
        if control == "no_btc_variation_gate"
        else used.btc_variation_rank.ge(0.65)
    )
    eligible = valid & signal.notna() & signal.ne(0) & volatility
    side = pd.Series(np.where(signal.lt(0), 1, -1), index=panel.index, dtype=int)
    if control == "exchange_reserve_direction_flip":
        side = -side
    elif control == "same_clock_forced_long":
        side = pd.Series(1, index=panel.index, dtype=int)
    adjusted = used.copy()
    adjusted["reserve_log_change"] = signal
    adjusted["feature_available_time"] = available
    return eligible, side, adjusted


def build_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    eligible, side, used = active(panel, control)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in panel.index[eligible]:
        decision = pd.Timestamp(panel.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=24)
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        reserved_until = exit_time
        rows.append({
            "candidate": POLICY_ID,
            "control": control,
            "split": split,
            "observation_time": panel.at[index, "observation_time"],
            "feature_available_time": used.at[index, "feature_available_time"],
            "decision_time": decision,
            "entry_time": entry,
            "exit_time": exit_time,
            "side": int(side.at[index]),
            "SplyExNtv": float(panel.at[index, "SplyExNtv"]),
            "reserve_log_change": float(used.at[index, "reserve_log_change"]),
            "reserve_level_rank": float(panel.at[index, "reserve_level_rank"]),
            "btc_variation": float(panel.at[index, "btc_variation"]),
            "btc_variation_rank": float(panel.at[index, "btc_variation_rank"]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = clock[clock.split.eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(subset.side.eq(1).sum())
    shorts = int(subset.side.eq(-1).sum())
    months = pd.to_datetime(subset.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(subset),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": float(months.max() / len(subset)),
    }


def deterministic_csv_gzip(frame: pd.DataFrame) -> bytes:
    raw = frame.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=buffer, mode="wb", compresslevel=6, mtime=0) as handle:
        handle.write(raw)
    return buffer.getvalue()


def deterministic_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"refusing to overwrite non-identical artifact {path}")
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_btc_source() -> pd.DataFrame:
    from sqlalchemy import text

    engine = postgres_engine()
    try:
        with engine.connect() as connection:
            return pd.read_sql_query(
                text(BTC_QUERY), connection,
                params={"start": BTC_QUERY_START.to_pydatetime(), "end": BTC_QUERY_END.to_pydatetime()},
            )
    finally:
        engine.dispose()


def run(*, fetch: Fetch | None = None) -> dict[str, Any]:
    if sha256_file(prereg.DEFAULT_OUTPUT) != PREREG_SHA256:
        raise RuntimeError("HVEXRP frozen preregistration artifact drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    if registration != prereg.build():
        raise RuntimeError("HVEXRP frozen preregistration payload drift")
    source, transport = download_coinmetrics(fetch=fetch)
    btc = load_btc_source()
    panel = build_panel(source, btc)
    primary = build_clock(panel)
    controls = {name: build_clock(panel, name) for name in CONTROLS}
    splits = {name: primary[primary.split.eq(name)].copy() for name in SPLITS}

    write_immutable(DAILY_SOURCE, deterministic_csv_gzip(source))
    write_immutable(FEATURE_PANEL, deterministic_csv_gzip(panel))
    write_immutable(CLOCK, deterministic_csv_gzip(primary))
    for name, frame in splits.items():
        write_immutable(SPLIT_DIR / f"{name}.csv.gz", deterministic_csv_gzip(frame))
    for name, frame in controls.items():
        write_immutable(CONTROL_DIR / f"{name}.csv.gz", deterministic_csv_gzip(frame))

    source_core = {
        "protocol_version": "hvexrp_24_source_materialization_v1",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA256, "manifest_hash": registration["manifest_hash"]},
        "coin_metrics_transport": transport,
        "revision_boundary": "current download vintage; not a historical Coin Metrics revision archive",
        "daily_source": {"path": str(DAILY_SOURCE), "sha256": sha256_file(DAILY_SOURCE), "rows": len(source)},
        "btc_query": BTC_QUERY,
        "btc_query_sha256": hashlib.sha256(BTC_QUERY.encode("utf-8")).hexdigest(),
        "btc_query_columns": ["ts", "open", "close"],
        "btc_rows": len(btc),
        "feature_panel": {"path": str(FEATURE_PANEL), "sha256": sha256_file(FEATURE_PANEL), "rows": len(panel), "valid_rows": int(panel.source_valid.sum())},
        "no_imputation": True,
        "entry_exit_price_fields_opened": False,
        "postentry_return_or_pnl_opened": False,
        "gross9_rows_opened": False,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    write_immutable(SOURCE_MANIFEST, deterministic_json(source_manifest))

    support = {name: support_stats(primary, name) for name in SPLITS}
    checks = {
        key: passed
        for name, values in support.items()
        for key, passed in (
            (f"{name}_minimum_events", values["events"] >= GATES["minimum_events"][name]),
            (f"{name}_side_balance", values["minority_side_share"] >= GATES["minority_side_share_min"]),
            (f"{name}_month_concentration", values["max_month_share"] <= GATES["max_month_share"]),
        )
    }
    passed = all(checks.values())
    core = {
        "protocol_version": "hvexrp_24_source_support_v1",
        "policy_id": POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA256, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha256_file(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True,
        "candidate_incidence_opened": True,
        "entry_exit_price_fields_opened": False,
        "postentry_return_pnl_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256_file(CLOCK), "rows": len(primary)},
        "split_artifacts": {name: {"path": str(SPLIT_DIR / f"{name}.csv.gz"), "sha256": sha256_file(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(frame)} for name, frame in splits.items()},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha256_file(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    write_immutable(RESULT, deterministic_json(result))
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2, sort_keys=True))
