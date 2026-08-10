"""Build outcome-blind source support for frozen HVMRSVP-24."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_mstr_relative_short_volume_pressure_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA256 = "4e51be73a940bffa9a4cf9534c2802490322d12e85c8f1feb3a833434db877eb"
TEMPLATE = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
REGISTRATION = prereg.build()
POLICY = REGISTRATION["policy"]
SUPPORT_GATES = REGISTRATION["source_support_gates"]
SPLITS = {
    name: (pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))
    for name, bounds in REGISTRATION["stages"].items()
}
MINIMUM = SUPPORT_GATES["minimum_events"]
CONTROLS = tuple(REGISTRATION["diagnostic_controls"]["names"])

SOURCE_DIR = Path("data/high_volatility_mstr_relative_short_volume_pressure_relay_sources_2023_2026")
PAIR_PANEL = SOURCE_DIR / "mstr_qqq_short_volume_panel.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "mstr_qqq_feature_panel.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_mstr_relative_short_volume_pressure_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_mstr_relative_short_volume_pressure_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_mstr_relative_short_volume_pressure_relay_support_2026-08-10.json")

QUERY = """SELECT ts,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""

FINRA_COLUMNS = [
    "Date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume", "Market",
]
PAIR_COLUMNS = (
    "source_date", "feature_available_time", "mstr_short_volume", "mstr_total_volume",
    "qqq_short_volume", "qqq_total_volume",
)
FEATURE_COLUMNS = (
    *PAIR_COLUMNS, "btc_source_valid", "source_valid", "mstr_short_share", "qqq_short_share",
    "relative_pressure", "pressure_change", "mstr_share_change", "realized_variation",
    "absolute_pressure_change_rank", "realized_variation_rank",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "source_date", "feature_available_time", "decision_time",
    "entry_time", "exit_time", "side", "mstr_short_volume", "mstr_total_volume",
    "qqq_short_volume", "qqq_total_volume", "mstr_short_share", "qqq_short_share",
    "relative_pressure", "pressure_change", "mstr_share_change", "absolute_pressure_change_rank",
    "realized_variation", "realized_variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def strict_prior_midrank(
    values: pd.Series,
    lookback: int = 252,
    minimum: int = 126,
) -> pd.Series:
    """Rank finite values against at most ``lookback`` finite strict priors."""
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior, dtype=float)
            output.at[index] = float(
                (np.count_nonzero(array < current) + 0.5 * np.count_nonzero(array == current))
                / len(array)
            )
        if math.isfinite(current):
            history.append(float(current))
    return output


def strict_prior_source_day_midrank(
    values: pd.Series,
    lookback: int = 252,
    minimum: int = 126,
) -> pd.Series:
    """Rank within the fixed preceding source-day positions, excluding nonfinite values."""
    numeric = pd.to_numeric(values, errors="coerce").astype(float).reset_index(drop=True)
    output = pd.Series(np.nan, index=values.index, dtype=float)
    for position, current in enumerate(numeric):
        prior = numeric.iloc[max(0, position - lookback):position]
        array = prior[np.isfinite(prior)].to_numpy(dtype=float)
        if math.isfinite(current) and len(array) >= minimum:
            output.iloc[position] = float(
                (np.count_nonzero(array < current) + 0.5 * np.count_nonzero(array == current))
                / len(array)
            )
    return output


def postgres_engine():
    """Create the repository-standard PostgreSQL engine from the existing env helper."""
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(
        postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10}
    )


def load_bars() -> pd.DataFrame:
    """Read only causal BTCUSDT one-minute OHLC needed by the frozen feature."""
    engine = postgres_engine()
    try:
        with engine.connect() as connection:
            bars = pd.read_sql_query(
                QUERY,
                connection,
                params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        engine.dispose()
    if bars.columns.tolist() != ["ts", "close"]:
        raise RuntimeError("HVMRSVP bars_binance schema drift")
    return bars


def _strict_volume(value: str, field: str) -> int:
    if not value or not value.isascii() or not value.isdecimal():
        raise ValueError(f"FINRA {field} is not an unsigned integer")
    return int(value)


def parse_target_rows(raw: bytes, source_date: pd.Timestamp) -> list[dict[str, Any]]:
    """Parse a full FINRA daily file and return its exact MSTR/QQQ pair."""
    try:
        text = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("FINRA daily file is not strict UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter="|")
    if reader.fieldnames != FINRA_COLUMNS:
        raise ValueError("FINRA daily file schema drift")

    date_text = source_date.strftime("%Y%m%d")
    targets: list[dict[str, Any]] = []
    seen_symbols: set[str] = set()
    for row in reader:
        if list(row) != FINRA_COLUMNS or any(row[column] is None for column in FINRA_COLUMNS):
            raise ValueError("FINRA daily row schema drift")
        if row["Date"] != date_text or not row["Symbol"] or not row["Market"]:
            raise ValueError("FINRA daily row identity invalid")
        if row["Symbol"] in seen_symbols:
            raise ValueError("FINRA daily file has duplicate symbol/date")
        seen_symbols.add(row["Symbol"])
        short = _strict_volume(row["ShortVolume"], "ShortVolume")
        exempt = _strict_volume(row["ShortExemptVolume"], "ShortExemptVolume")
        total = _strict_volume(row["TotalVolume"], "TotalVolume")
        if total <= 0 or short > total:
            raise ValueError("FINRA daily row volume invalid")
        if row["Symbol"] in {"MSTR", "QQQ"}:
            targets.append(
                {
                    "source_date": source_date,
                    "symbol": row["Symbol"],
                    "short_volume": short,
                    "short_exempt_volume": exempt,
                    "total_volume": total,
                    "market": row["Market"],
                }
            )

    counts = {symbol: sum(row["symbol"] == symbol for row in targets) for symbol in ("MSTR", "QQQ")}
    if counts != {"MSTR": 1, "QQQ": 1}:
        raise RuntimeError("FINRA source day lacks exactly one MSTR and one QQQ row")
    return sorted(targets, key=lambda row: row["symbol"])


def _response_binding(url: str, status: int, raw: bytes, headers: Any) -> dict[str, Any]:
    return {
        "url": url,
        "status": status,
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "etag": headers.get("ETag") if headers is not None else None,
        "last_modified": headers.get("Last-Modified") if headers is not None else None,
    }


def download_date(
    source_date: pd.Timestamp,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """Fetch one official calendar date; only HTTP 200 and 404 are admissible."""
    url = TEMPLATE.format(date=source_date.strftime("%Y%m%d"))
    request = urllib.request.Request(
        url,
        headers={"Accept-Encoding": "identity", "User-Agent": "rllm-source-audit/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = response.status
            raw = response.read()
            headers = response.headers
            final_url = response.geturl()
            if final_url != url:
                raise RuntimeError(f"FINRA redirect forbidden: requested={url} final={final_url}")
    except urllib.error.HTTPError as error:
        status = error.code
        raw = error.read()
        headers = error.headers
        if status != 404:
            digest = hashlib.sha256(raw).hexdigest()
            raise RuntimeError(
                f"FINRA fail-closed HTTP status {status} for {url}; response_sha256={digest}"
            ) from error

    binding = _response_binding(url, status, raw, headers)
    if status == 404:
        return None, binding
    if status != 200:
        raise RuntimeError(f"FINRA fail-closed HTTP status {status} for {url}")
    return parse_target_rows(raw, source_date), binding


def _normalized_panel_hash(pair: pd.DataFrame) -> str:
    records = []
    for row in pair.loc[:, PAIR_COLUMNS].itertuples(index=False):
        records.append(
            {
                "source_date": pd.Timestamp(row.source_date).isoformat(),
                "feature_available_time": pd.Timestamp(row.feature_available_time).isoformat(),
                "mstr_short_volume": int(row.mstr_short_volume),
                "mstr_total_volume": int(row.mstr_total_volume),
                "qqq_short_volume": int(row.qqq_short_volume),
                "qqq_total_volume": int(row.qqq_total_volume),
            }
        )
    return canonical_hash(records)


def download_pair_panel(workers: int = 8) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Probe every preregistered calendar date and normalize exact target pairs."""
    dates = list(pd.date_range(START, END, freq="1d", inclusive="left"))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        downloaded = list(executor.map(download_date, dates))

    rows: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    for date, (target_rows, binding) in zip(dates, downloaded, strict=True):
        responses.append({"date": date.strftime("%Y-%m-%d"), **binding})
        if target_rows is not None:
            rows.extend(target_rows)

    raw = pd.DataFrame(
        rows,
        columns=[
            "source_date", "symbol", "short_volume", "short_exempt_volume", "total_volume", "market",
        ],
    )
    if raw.duplicated(["source_date", "symbol"], keep=False).any():
        raise RuntimeError("duplicate FINRA target date/symbol")
    by_symbol = {
        symbol: raw[raw.symbol.eq(symbol)].set_index("source_date").sort_index()
        for symbol in ("MSTR", "QQQ")
    }
    if not by_symbol["MSTR"].index.equals(by_symbol["QQQ"].index):
        raise RuntimeError("FINRA MSTR/QQQ source-day mismatch")
    index = by_symbol["MSTR"].index
    pair = pd.DataFrame(
        {
            "source_date": index,
            "feature_available_time": index + pd.Timedelta(days=1),
            "mstr_short_volume": by_symbol["MSTR"].short_volume.to_numpy(dtype="int64"),
            "mstr_total_volume": by_symbol["MSTR"].total_volume.to_numpy(dtype="int64"),
            "qqq_short_volume": by_symbol["QQQ"].short_volume.to_numpy(dtype="int64"),
            "qqq_total_volume": by_symbol["QQQ"].total_volume.to_numpy(dtype="int64"),
        },
        columns=PAIR_COLUMNS,
    ).reset_index(drop=True)
    statuses = pd.Series([item["status"] for item in responses]).value_counts()
    transport = {
        "official_template": TEMPLATE,
        "window_start_inclusive": START.strftime("%Y-%m-%d"),
        "window_end_exclusive": END.strftime("%Y-%m-%d"),
        "calendar_dates_requested": len(dates),
        "responses": responses,
        "http_200_days": int(statuses.get(200, 0)),
        "http_404_days": int(statuses.get(404, 0)),
        "source_days": len(pair),
        "all_responses_sha256_bound": all("response_sha256" in item for item in responses),
        "normalized_panel_sha256": _normalized_panel_hash(pair),
    }
    return pair, transport


def _prepare_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.columns.tolist() != ["ts", "close"]:
        raise RuntimeError("HVMRSVP bars_binance schema drift")
    market = bars.copy()
    market["ts"] = pd.to_datetime(market.ts, utc=True, errors="coerce")
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    if market.ts.isna().any() or market.duplicated("ts", keep=False).any():
        raise RuntimeError("HVMRSVP invalid or duplicate bars_binance timestamp")
    return market.set_index("ts").sort_index(kind="mergesort")


def feature_panel(pair: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    if pair.columns.tolist() != list(PAIR_COLUMNS):
        raise RuntimeError("HVMRSVP normalized FINRA panel schema drift")
    frame = pair.sort_values("source_date", kind="mergesort").reset_index(drop=True).copy()
    if frame.source_date.duplicated().any() or not frame.source_date.is_monotonic_increasing:
        raise RuntimeError("HVMRSVP normalized FINRA source-day order invalid")
    market = _prepare_bars(bars)

    variations: list[float] = []
    btc_valid: list[bool] = []
    for decision in frame.feature_available_time:
        expected = pd.date_range(
            decision - pd.Timedelta(days=1), decision, freq="1min", inclusive="left"
        )
        window = market.reindex(expected)
        valid = bool(
            len(window) == 1440
            and np.isfinite(window.close).all()
            and window.close.gt(0).all()
        )
        variation = float("nan")
        if valid:
            variation = float(np.square(np.diff(np.log(window.close.to_numpy(float)))).sum())
            valid = math.isfinite(variation) and variation > 0
        variations.append(variation)
        btc_valid.append(valid)

    frame["btc_source_valid"] = btc_valid
    frame["source_valid"] = frame.btc_source_valid
    frame["mstr_short_share"] = frame.mstr_short_volume / frame.mstr_total_volume
    frame["qqq_short_share"] = frame.qqq_short_volume / frame.qqq_total_volume
    mstr = frame.mstr_short_share.clip(1e-9, 1 - 1e-9)
    qqq = frame.qqq_short_share.clip(1e-9, 1 - 1e-9)
    frame["relative_pressure"] = np.log(mstr / (1 - mstr)) - np.log(qqq / (1 - qqq))
    frame["pressure_change"] = frame.relative_pressure.diff()
    frame["mstr_share_change"] = frame.mstr_short_share.diff()
    frame["realized_variation"] = variations
    # Pressure history is defined by official source days, independent of BTC-bar completeness.
    frame["absolute_pressure_change_rank"] = strict_prior_midrank(
        frame.pressure_change.abs(),
        POLICY["history_source_days"],
        POLICY["minimum_history_source_days"],
    )
    frame["realized_variation_rank"] = strict_prior_source_day_midrank(
        frame.realized_variation.where(frame.btc_source_valid),
        POLICY["history_source_days"],
        POLICY["minimum_history_source_days"],
    )
    return frame.loc[:, FEATURE_COLUMNS]


def _used_features(panel: pd.DataFrame, control: str) -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(f"unknown HVMRSVP control: {control}")
    used = panel.copy()
    if control == "one_source_day_stale_features":
        for column in FEATURE_COLUMNS:
            if column not in {"source_valid", "btc_source_valid"}:
                used[column] = panel[column].shift(1)
        used["source_valid"] = panel.source_valid.shift(1, fill_value=False).astype(bool)
        used["btc_source_valid"] = panel.btc_source_valid.shift(1, fill_value=False).astype(bool)
    return used


def eligible_and_side(
    panel: pd.DataFrame, control: str = "primary"
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    used = _used_features(panel, control)
    side_source = (
        used.mstr_share_change if control == "mstr_share_change_only" else used.pressure_change
    )
    pressure_gate = (
        pd.Series(True, index=used.index)
        if control == "no_pressure_magnitude_gate"
        else used.absolute_pressure_change_rank.ge(POLICY["absolute_pressure_change_rank_min"])
    )
    variation_gate = (
        pd.Series(True, index=used.index)
        if control == "no_volatility_gate"
        else used.realized_variation_rank.ge(POLICY["btc_variation_rank_min"])
    )
    eligible = (
        used.source_valid.fillna(False).astype(bool)
        & side_source.notna()
        & side_source.ne(0)
        & pressure_gate
        & variation_gate
    )
    side = -np.sign(pd.to_numeric(side_source, errors="coerce")).fillna(0).astype(int)
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = pd.Series(1, index=used.index, dtype=int)
    return eligible, side, used


def candidate_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    ordered = panel.sort_values("source_date", kind="mergesort").reset_index(drop=True)
    eligible, side, used = eligible_and_side(ordered, control)
    onset = eligible & ~eligible.shift(1, fill_value=False)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in ordered.index[onset]:
        decision = pd.Timestamp(ordered.at[index, "feature_available_time"])
        entry = decision + pd.Timedelta(minutes=POLICY["entry_delay_minutes"])
        exit_time = entry + pd.Timedelta(hours=POLICY["hold_hours"])
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next(
            (
                name
                for name, (start, end) in SPLITS.items()
                if entry >= start and exit_time <= end
            ),
            None,
        )
        if split is None:
            continue
        reserved_until = exit_time
        rows.append(
            {
                "candidate": "HVMRSVP-24",
                "control": control,
                "split": split,
                "source_date": pd.Timestamp(used.at[index, "source_date"]),
                "feature_available_time": pd.Timestamp(used.at[index, "feature_available_time"]),
                "decision_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(side.at[index]),
                **{
                    column: float(used.at[index, column])
                    for column in CLOCK_COLUMNS[9:]
                },
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stage_stats(candidate: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = candidate[candidate.split.eq(split)]
    if subset.empty:
        return {
            "events": 0, "longs": 0, "shorts": 0,
            "minority_side_share": 0.0, "max_month_share": 0.0,
        }
    longs = int(subset.side.eq(1).sum())
    shorts = int(subset.side.eq(-1).sum())
    months = pd.to_datetime(subset.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(subset),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(months.max()) / len(subset),
    }


def _install_immutable(temp: Path, destination: Path) -> None:
    if destination.exists():
        if sha(destination) != sha(temp):
            raise RuntimeError(f"immutable HVMRSVP artifact already differs: {destination}")
        temp.unlink()
        return
    os.replace(temp, destination)


def write_immutable_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    frame.to_csv(
        buffer,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        float_format="%.12g",
    )
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(buffer.getvalue())
        _install_immutable(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def write_immutable_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
        _install_immutable(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def run(workers: int = 8) -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA256:
        raise RuntimeError("HVMRSVP preregistration artifact drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    if registration != prereg.build():
        raise RuntimeError("HVMRSVP preregistration payload drift")
    if tuple(registration["diagnostic_controls"]["names"]) != CONTROLS:
        raise RuntimeError("HVMRSVP diagnostic-control drift")

    pair, transport = download_pair_panel(workers=workers)
    bars = load_bars()
    panel = feature_panel(pair, bars)
    primary = candidate_clock(panel)
    controls = {name: candidate_clock(panel, name) for name in CONTROLS}

    write_immutable_csv(pair, PAIR_PANEL)
    write_immutable_csv(panel, FEATURE_PANEL)
    write_immutable_csv(primary, CLOCK)
    control_bindings: dict[str, dict[str, Any]] = {}
    for name, clock in controls.items():
        path = CONTROL_DIR / f"{name}.csv.gz"
        write_immutable_csv(clock, path)
        control_bindings[name] = {
            "path": str(path), "sha256": sha(path), "rows": len(clock),
            "promotion_authorized": False,
        }

    source_core = {
        "protocol_version": "hvmrsvp_24_source_materialization_v1",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA256,
            "manifest_hash": registration["manifest_hash"],
        },
        "transport": transport,
        "pair_panel": {
            "path": str(PAIR_PANEL), "sha256": sha(PAIR_PANEL), "rows": len(pair),
            "normalized_sha256": transport["normalized_panel_sha256"],
        },
        "bars_query": QUERY,
        "bars_query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "bars_rows": len(bars),
        "feature_panel": {
            "path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(panel),
            "valid_rows": int(panel.source_valid.sum()),
        },
        "no_imputation": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
    }
    source = {**source_core, "manifest_hash": canonical_hash(source_core)}
    write_immutable_json(source, SOURCE_MANIFEST)

    support = {name: stage_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {
        "complete_target_pair_per_source_day": bool(
            transport["source_days"] == len(pair)
            and not pair.source_date.duplicated().any()
        )
    }
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = (
            values["minority_side_share"] >= SUPPORT_GATES["minority_side_share_min"]
        )
        checks[f"{name}_month_concentration"] = (
            values["max_month_share"] <= SUPPORT_GATES["max_month_share"]
        )
    passed = all(checks.values())
    core = {
        "protocol_version": "hvmrsvp_24_oos_source_support_v1",
        "policy_id": "HVMRSVP-24",
        "as_of_date": "2026-08-10",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA256,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST),
            "sha256": sha(SOURCE_MANIFEST),
            "manifest_hash": source["manifest_hash"],
        },
        "ranking": {
            "lookback_source_days": POLICY["history_source_days"],
            "minimum_prior_source_days": POLICY["minimum_history_source_days"],
            "current_excluded": True,
        },
        "reservation": {
            "scope": "global",
            "interval": "half_open",
            "equal_time_entry_after_exit_allowed": True,
            "split_crossing_action": "skip",
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": control_bindings,
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    write_immutable_json(result, RESULT)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
