#!/usr/bin/env python3
"""Build the outcome-blind QQQ/GLD cash-open feature prefix for COGR-12.

The emitted table is safe to hand to an outcome evaluator: the only current
session prices are the raw QQQ and GLD opening prints. The frozen provider's
OHLC history is already split-normalized, so split events are audited but never
multiplied into prices a second time. Effective-date cash dividends convert
opens to a prior-close-comparable economic value without reading same-session
closes or adjusted closes. Current-session high, low, close, and volume are
never emitted. Every other numeric feature is shifted through the preceding
completed US cash session.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
BUILDER_PATH = Path(
    "training/build_cash_open_cross_asset_gap_features.py"
)
SYMBOLS = ("QQQ", "GLD")
SELECTION_CUTOFF = "2025-01-01"
NEW_YORK_TIMEZONE = "America/New_York"
FEATURE_AVAILABLE_LOCAL_TIME = "09:35"
ENTRY_LOCAL_TIME = "09:40"
FROZEN_INPUT_SHA256 = {
    "QQQ": "e9d0cbb6bbe41345f8897071198322f14f82f065c2f8ba0b9896be1ad434f162",
    "GLD": "f564a4f7f4fb582dafc40a06a02b12bedd599f0300bf1874ce20bf9507ccd928",
}
SYMBOL_SUFFIXES = (
    "gap_open",
    "prior_close_return_1d",
    "prior_close_return_5d",
    "prior_close_return_20d",
    "prior_intraday_return",
    "prior_range",
    "prior_realized_vol_5d",
    "prior_realized_vol_20d",
    "prior_volume_z60",
)
FEATURE_COLUMNS = tuple(
    f"{suffix}_{symbol.lower()}"
    for symbol in SYMBOLS
    for suffix in SYMBOL_SUFFIXES
) + (
    "gap_risk_rotation",
    "gap_joint_liquidity",
    "gap_abs_total",
    "gap_direction_agreement",
    "prior_return_spread_1d",
    "prior_return_spread_5d",
    "prior_return_spread_20d",
    "prior_return_corr_20d",
    "prior_return_beta_20d",
    "prior_vol_ratio_20d",
    "prior_volume_z_spread",
    "weekday_sin",
    "weekday_cos",
)
OUTPUT_COLUMNS = (
    "session_date",
    "feature_available_time_utc",
    "entry_time_utc",
    *FEATURE_COLUMNS,
    "feature_valid",
    "feature_invalid_reason",
)
FORBIDDEN_OUTPUT_COLUMNS = (
    "current_high",
    "current_low",
    "current_close",
    "current_volume",
    "btc_open",
    "btc_high",
    "btc_low",
    "btc_close",
    "future_return",
    "pnl",
)


@dataclass(frozen=True)
class BuildConfig:
    qqq_cache: str = "data/cache_cross_asset_alpha_transfer/QQQ.json"
    gld_cache: str = "data/cache_cross_asset_alpha_transfer/GLD.json"
    output: str = (
        "data/cash_open_cross_asset_gap_relay_pre2025/"
        "qqq_gld_cash_open_safe_features_pre2025.csv.gz"
    )
    manifest: str = (
        "data/cash_open_cross_asset_gap_relay_pre2025/build_manifest.json"
    )
    cutoff: str = SELECTION_CUTOFF
    enforce_frozen_inputs: bool = True


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def deterministic_gzip_csv(frame: pd.DataFrame, path: str | Path) -> None:
    text = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        date_format="%Y-%m-%dT%H:%M:%S",
    )
    buffer = io.BytesIO()
    with gzip.GzipFile(
        fileobj=buffer,
        mode="wb",
        filename="",
        mtime=0,
    ) as handle:
        handle.write(text.encode())
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(buffer.getvalue())


def _frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(",".join(frame.columns).encode())
    digest.update(frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        date_format="%Y-%m-%dT%H:%M:%S",
    ).encode())
    return digest.hexdigest()


def _safe_session_time(
    dates: pd.Series | pd.DatetimeIndex,
    local_time: str,
) -> pd.DatetimeIndex:
    hour, minute = (int(value) for value in local_time.split(":"))
    local = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(
        NEW_YORK_TIMEZONE
    )
    local = local + pd.Timedelta(hours=hour, minutes=minute)
    return local.tz_convert("UTC").tz_localize(None)


def _robust_previous_observation_z(
    values: pd.Series,
    *,
    window: int = 60,
) -> pd.Series:
    observed = values.shift(1)
    reference = values.shift(2).rolling(
        window=window,
        min_periods=window,
    )
    center = reference.quantile(0.50, interpolation="linear")
    lower = reference.quantile(0.25, interpolation="linear")
    upper = reference.quantile(0.75, interpolation="linear")
    scale = (upper - lower) / 1.349
    return (observed - center) / scale.where(scale > 0.0)


def _prepare_market(
    frame: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    required = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "cash_dividend",
        "split_ratio",
        "open_valid",
        "history_valid",
    }
    if set(frame.columns) != required:
        raise ValueError(
            f"unexpected normalized cross-asset columns: {frame.columns}"
        )
    output = frame.loc[
        pd.to_datetime(frame["date"]) < cutoff,
        list(required),
    ].copy()
    output["date"] = pd.to_datetime(output["date"])
    output = output.sort_values("date").reset_index(drop=True)
    if (
        output.empty
        or output["date"].duplicated().any()
        or not output["date"].is_monotonic_increasing
    ):
        raise RuntimeError("invalid cross-asset prefix calendar")
    numeric = output[["open", "high", "low", "close", "volume"]]
    output["open_valid"] = (
        output["open_valid"].astype(bool)
        & np.isfinite(output["open"])
        & output["open"].gt(0.0)
        & np.isfinite(output["cash_dividend"])
        & output["cash_dividend"].ge(0.0)
        & np.isfinite(output["split_ratio"])
        & output["split_ratio"].gt(0.0)
    )
    output["history_valid"] = (
        output["history_valid"].astype(bool)
        & np.isfinite(numeric.to_numpy(float)).all(axis=1)
        & (numeric[["open", "high", "low", "close"]] > 0.0).all(
            axis=1
        )
        & numeric["volume"].ge(0.0)
        & np.isfinite(output["cash_dividend"])
        & output["cash_dividend"].ge(0.0)
        & np.isfinite(output["split_ratio"])
        & output["split_ratio"].gt(0.0)
    )
    return output.set_index("date")


def build_safe_feature_frame(
    markets: dict[str, pd.DataFrame],
    *,
    cutoff: str | pd.Timestamp = SELECTION_CUTOFF,
) -> pd.DataFrame:
    """Build features after truncating both sources at the frozen cutoff."""
    if tuple(markets) != SYMBOLS:
        raise ValueError("COGR source universe/order drifted")
    boundary = pd.Timestamp(cutoff)
    prepared = {
        symbol: _prepare_market(markets[symbol], cutoff=boundary)
        for symbol in SYMBOLS
    }
    sessions = prepared["QQQ"].index.intersection(
        prepared["GLD"].index
    ).sort_values()
    if len(sessions) < 100:
        raise RuntimeError("insufficient common QQQ/GLD sessions")
    aligned = {
        symbol: prepared[symbol].reindex(sessions)
        for symbol in SYMBOLS
    }
    common_open_valid = np.logical_and.reduce(
        [
            aligned[symbol]["open_valid"].fillna(False).to_numpy(bool)
            for symbol in SYMBOLS
        ]
    )

    columns: dict[str, pd.Series] = {}
    returns: dict[str, pd.Series] = {}
    volume_z: dict[str, pd.Series] = {}
    vol20: dict[str, pd.Series] = {}
    for symbol in SYMBOLS:
        suffix = symbol.lower()
        frame = aligned[symbol]
        history_valid = frame["history_valid"].fillna(False).astype(bool)
        raw_open = frame["open"].where(frame["open_valid"])
        raw_high = frame["high"].where(history_valid)
        raw_low = frame["low"].where(history_valid)
        raw_close = frame["close"].where(history_valid)
        raw_volume = frame["volume"].where(history_valid)
        cash_dividend = frame["cash_dividend"].where(
            frame["open_valid"]
        )
        prior_close = raw_close.shift(1)
        economic_open = raw_open + cash_dividend
        economic_close = raw_close + frame[
            "cash_dividend"
        ].where(history_valid)
        log_open = np.log(raw_open.where(raw_open > 0.0))
        log_high = np.log(raw_high.where(raw_high > 0.0))
        log_low = np.log(raw_low.where(raw_low > 0.0))
        log_close = np.log(raw_close.where(raw_close > 0.0))
        close_return = np.log(
            economic_close.where(economic_close > 0.0)
            / prior_close.where(prior_close > 0.0)
        )
        returns[symbol] = close_return
        current_gap = np.log(
            economic_open.where(economic_open > 0.0)
            / prior_close.where(prior_close > 0.0)
        )
        prior_vol5 = close_return.shift(1).rolling(
            window=5, min_periods=5
        ).std(ddof=0)
        prior_vol20 = close_return.shift(1).rolling(
            window=20, min_periods=20
        ).std(ddof=0)
        vol20[symbol] = prior_vol20
        log_volume = np.log1p(raw_volume.where(raw_volume >= 0))
        volume_z[symbol] = _robust_previous_observation_z(log_volume)
        columns[f"gap_open_{suffix}"] = current_gap
        columns[f"prior_close_return_1d_{suffix}"] = close_return.shift(
            1
        )
        columns[f"prior_close_return_5d_{suffix}"] = close_return.shift(
            1
        ).rolling(window=5, min_periods=5).sum()
        columns[f"prior_close_return_20d_{suffix}"] = close_return.shift(
            1
        ).rolling(window=20, min_periods=20).sum()
        columns[f"prior_intraday_return_{suffix}"] = (
            log_close.shift(1) - log_open.shift(1)
        )
        columns[f"prior_range_{suffix}"] = (
            log_high.shift(1) - log_low.shift(1)
        )
        columns[f"prior_realized_vol_5d_{suffix}"] = prior_vol5
        columns[f"prior_realized_vol_20d_{suffix}"] = prior_vol20
        columns[f"prior_volume_z60_{suffix}"] = volume_z[symbol]

    q_gap = columns["gap_open_qqq"]
    g_gap = columns["gap_open_gld"]
    columns["gap_risk_rotation"] = q_gap - g_gap
    columns["gap_joint_liquidity"] = q_gap + g_gap
    columns["gap_abs_total"] = q_gap.abs() + g_gap.abs()
    columns["gap_direction_agreement"] = np.sign(q_gap) * np.sign(g_gap)
    for horizon in (1, 5, 20):
        columns[f"prior_return_spread_{horizon}d"] = (
            columns[f"prior_close_return_{horizon}d_qqq"]
            - columns[f"prior_close_return_{horizon}d_gld"]
        )
    q_prior = returns["QQQ"].shift(1)
    g_prior = returns["GLD"].shift(1)
    columns["prior_return_corr_20d"] = q_prior.rolling(
        window=20, min_periods=20
    ).corr(g_prior)
    covariance = q_prior.rolling(
        window=20, min_periods=20
    ).cov(g_prior, ddof=0)
    g_variance = g_prior.rolling(
        window=20, min_periods=20
    ).var(ddof=0)
    columns["prior_return_beta_20d"] = covariance / g_variance.where(
        g_variance > 0.0
    )
    columns["prior_vol_ratio_20d"] = np.log(
        vol20["QQQ"] / vol20["GLD"].where(vol20["GLD"] > 0.0)
    )
    columns["prior_volume_z_spread"] = (
        volume_z["QQQ"] - volume_z["GLD"]
    )
    weekday = pd.Series(
        sessions.dayofweek.to_numpy(float),
        index=sessions,
    )
    columns["weekday_sin"] = np.sin(2.0 * np.pi * weekday / 5.0)
    columns["weekday_cos"] = np.cos(2.0 * np.pi * weekday / 5.0)

    features = pd.DataFrame(columns, index=sessions).loc[
        :, list(FEATURE_COLUMNS)
    ]
    features = features.replace([np.inf, -np.inf], np.nan)
    finite = np.isfinite(features.to_numpy(float)).all(axis=1)
    valid = common_open_valid & finite
    output = features.reset_index(names="session_date")
    output.insert(
        1,
        "feature_available_time_utc",
        _safe_session_time(
            output["session_date"],
            FEATURE_AVAILABLE_LOCAL_TIME,
        ),
    )
    output.insert(
        2,
        "entry_time_utc",
        _safe_session_time(output["session_date"], ENTRY_LOCAL_TIME),
    )
    output["feature_valid"] = valid
    output["feature_invalid_reason"] = np.where(
        valid,
        "ok",
        np.where(
            common_open_valid,
            "insufficient_strictly_prior_history",
            "invalid_current_open_or_action_metadata",
        ),
    )
    output = output.loc[:, list(OUTPUT_COLUMNS)]
    if pd.to_datetime(output["session_date"]).max() >= boundary:
        raise RuntimeError("COGR source output breached cutoff")
    if set(FORBIDDEN_OUTPUT_COLUMNS).intersection(output.columns):
        raise RuntimeError("COGR safe output exposed forbidden values")
    return output


def parse_research_payload(
    raw: bytes,
    symbol: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse raw daily OHLCV plus pre-open corporate-action metadata.

    The current opening print is never adjusted with same-day close or
    adjusted-close data. Dividends and split ratios are explicit event fields
    whose effective date is known by the cash open. Yahoo's chart OHLC is
    already split-normalized, so split metadata is retained for audit and
    validation rather than multiplied into prices again.
    """
    payload = json.loads(raw)
    chart = payload.get("chart", {})
    if chart.get("error") is not None:
        raise RuntimeError(f"COGR Yahoo chart error for {symbol}")
    results = chart.get("result") or []
    if len(results) != 1:
        raise RuntimeError(f"COGR Yahoo result count drifted for {symbol}")
    result = results[0]
    timestamps = result.get("timestamp") or []
    quote_rows = ((result.get("indicators") or {}).get("quote") or [])
    if len(quote_rows) != 1 or not timestamps:
        raise RuntimeError(f"COGR Yahoo payload is incomplete for {symbol}")
    quote = quote_rows[0]
    raw_columns = {
        name: quote.get(name)
        for name in ("open", "high", "low", "close", "volume")
    }
    if any(
        values is None or len(values) != len(timestamps)
        for values in raw_columns.values()
    ):
        raise RuntimeError(f"COGR Yahoo vector length drifted for {symbol}")
    timezone_name = str(
        (result.get("meta") or {}).get("exchangeTimezoneName")
        or NEW_YORK_TIMEZONE
    )
    if timezone_name != NEW_YORK_TIMEZONE:
        raise RuntimeError(f"COGR exchange timezone drifted for {symbol}")
    dates = (
        pd.to_datetime(timestamps, unit="s", utc=True)
        .tz_convert(timezone_name)
        .tz_localize(None)
        .normalize()
    )
    frame = pd.DataFrame({"date": dates, **raw_columns})
    for name in ("open", "high", "low", "close", "volume"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if frame["date"].duplicated().any():
        raise RuntimeError(f"COGR Yahoo dates duplicated for {symbol}")

    dividend_by_date: dict[pd.Timestamp, float] = {}
    split_by_date: dict[pd.Timestamp, float] = {}
    events = result.get("events") or {}
    for row in (events.get("dividends") or {}).values():
        date = (
            pd.Timestamp(int(row["date"]), unit="s", tz="UTC")
            .tz_convert(timezone_name)
            .tz_localize(None)
            .normalize()
        )
        amount = float(row["amount"])
        if not np.isfinite(amount) or amount < 0.0:
            raise RuntimeError(f"COGR invalid dividend for {symbol}")
        dividend_by_date[date] = dividend_by_date.get(date, 0.0) + amount
    for row in (events.get("splits") or {}).values():
        date = (
            pd.Timestamp(int(row["date"]), unit="s", tz="UTC")
            .tz_convert(timezone_name)
            .tz_localize(None)
            .normalize()
        )
        numerator = float(row["numerator"])
        denominator = float(row["denominator"])
        ratio = numerator / denominator
        if not np.isfinite(ratio) or ratio <= 0.0:
            raise RuntimeError(f"COGR invalid split for {symbol}")
        split_by_date[date] = split_by_date.get(date, 1.0) * ratio
    frame["cash_dividend"] = frame["date"].map(
        dividend_by_date
    ).fillna(0.0)
    frame["split_ratio"] = frame["date"].map(split_by_date).fillna(1.0)
    frame["open_valid"] = (
        np.isfinite(frame["open"]) & frame["open"].gt(0.0)
    )
    history = frame[["open", "high", "low", "close", "volume"]]
    frame["history_valid"] = (
        np.isfinite(history.to_numpy(float)).all(axis=1)
        & (history[["open", "high", "low", "close"]] > 0.0).all(
            axis=1
        )
        & history["volume"].ge(0.0)
    )
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame, {
        "symbol": symbol,
        "exchange_timezone": timezone_name,
        "rows": int(len(frame)),
        "open_valid_rows": int(frame["open_valid"].sum()),
        "history_valid_rows": int(frame["history_valid"].sum()),
        "first_session": str(frame["date"].iloc[0].date()),
        "last_session": str(frame["date"].iloc[-1].date()),
        "dividend_events": int(len(dividend_by_date)),
        "split_events": int(len(split_by_date)),
        "current_open_depends_on_same_day_close": False,
        "adjusted_close_read": False,
    }


def _load_cache(
    path: str | Path,
    *,
    symbol: str,
    enforce_hash: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = Path(path)
    observed = sha256_file(source)
    if enforce_hash and observed != FROZEN_INPUT_SHA256[symbol]:
        raise RuntimeError(f"COGR frozen {symbol} cache hash drifted")
    raw = source.read_bytes()
    frame, metadata = parse_research_payload(raw, symbol)
    return frame, {
        "path": str(source),
        "sha256": observed,
        "bytes": source.stat().st_size,
        "provider": "Yahoo Finance chart cache (research only)",
        "provider_url": (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{symbol}"
        ),
        "exchange_timezone": metadata["exchange_timezone"],
        "first_session": metadata["first_session"],
        "last_session": metadata["last_session"],
        "open_valid_rows": metadata["open_valid_rows"],
        "history_valid_rows": metadata["history_valid_rows"],
        "dividend_events": metadata["dividend_events"],
        "split_events": metadata["split_events"],
        "current_open_depends_on_same_day_close": False,
        "adjusted_close_read": False,
    }


def build(cfg: BuildConfig = BuildConfig()) -> dict[str, Any]:
    if cfg.enforce_frozen_inputs and cfg.cutoff != SELECTION_CUTOFF:
        raise ValueError("COGR source cutoff is frozen")
    paths = {"QQQ": cfg.qqq_cache, "GLD": cfg.gld_cache}
    frames: dict[str, pd.DataFrame] = {}
    inputs: dict[str, dict[str, Any]] = {}
    for symbol in SYMBOLS:
        frames[symbol], inputs[symbol] = _load_cache(
            paths[symbol],
            symbol=symbol,
            enforce_hash=cfg.enforce_frozen_inputs,
        )
    features = build_safe_feature_frame(frames, cutoff=cfg.cutoff)
    deterministic_gzip_csv(features, cfg.output)
    output_path = Path(cfg.output)
    valid = features["feature_valid"].to_numpy(bool)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": "cash_open_cross_asset_gap_safe_feature_prefix",
        "config": asdict(cfg),
        "symbols": list(SYMBOLS),
        "source_contract": {
            "research_provider": "Frozen Yahoo Finance daily chart caches",
            "production_provider_requirement": (
                "Entitled QQQ/GLD US cash-open feed with a frozen parity audit "
                "is mandatory before live promotion."
            ),
            "price_normalization": (
                "Provider OHLC is used on its split-normalized price basis. "
                "Split events are audited but never multiplied into prices "
                "again. Effective-date cash dividends come only from explicit "
                "event metadata and convert the current open/daily close "
                "return into a causal economic return without reading "
                "adjusted close or same-day close for the current gap."
            ),
            "current_session_values_allowed": [
                "raw QQQ open",
                "raw GLD open",
                "effective-date dividend metadata",
                "effective-date split metadata",
            ],
            "current_session_values_forbidden": [
                "high",
                "low",
                "close",
                "volume",
            ],
            "feature_available_local_time": FEATURE_AVAILABLE_LOCAL_TIME,
            "entry_local_time": ENTRY_LOCAL_TIME,
            "timezone": NEW_YORK_TIMEZONE,
            "selection_cutoff_exclusive": SELECTION_CUTOFF,
            "future_source_rows_can_influence_prefix": False,
            "btc_or_portfolio_values_read": False,
            "post_entry_outcomes_opened": False,
        },
        "inputs": inputs,
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
            "rows": int(len(features)),
            "valid_rows": int(valid.sum()),
            "invalid_rows": int((~valid).sum()),
            "first_session": str(features["session_date"].iloc[0]),
            "last_session": str(features["session_date"].iloc[-1]),
            "first_valid_session": str(
                features.loc[valid, "session_date"].iloc[0]
            ),
            "columns": list(OUTPUT_COLUMNS),
            "feature_columns": list(FEATURE_COLUMNS),
            "feature_count": len(FEATURE_COLUMNS),
            "frame_hash": _frame_hash(features),
        },
        "builder": {
            "path": str(BUILDER_PATH),
            "sha256": sha256_file(BUILDER_PATH),
        },
    }
    manifest_path = Path(cfg.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qqq-cache", default=BuildConfig.qqq_cache)
    parser.add_argument("--gld-cache", default=BuildConfig.gld_cache)
    parser.add_argument("--output", default=BuildConfig.output)
    parser.add_argument("--manifest", default=BuildConfig.manifest)
    parser.add_argument("--cutoff", default=BuildConfig.cutoff)
    parser.add_argument(
        "--no-enforce-frozen-inputs",
        action="store_true",
    )
    args = parser.parse_args()
    payload = build(
        BuildConfig(
            qqq_cache=args.qqq_cache,
            gld_cache=args.gld_cache,
            output=args.output,
            manifest=args.manifest,
            cutoff=args.cutoff,
            enforce_frozen_inputs=not args.no_enforce_frozen_inputs,
        )
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
