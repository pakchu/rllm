"""Pure causal and structural primitives for the Gross9 clock bundle."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.ensemble import ExtraTreesRegressor


RANK7_FIT_START = pd.Timestamp("2020-07-01")
RANK7_SEEDS = (7, 71, 715, 2026, 71515)
RANK7_TREES = 300
_RANK7_BARRIER_HORIZONS = (144, 576, 2016)
_RANK7_BRAID_SCALE_WINDOW = 2016
_RANK7_BRAID_SCALE_MIN_PERIODS = 1008
_RANK7_BRAID_OI_PASSAGE_Z = 1.0
_RANK7_BRAID_PREMIUM_PASSAGE_Z = 1.0

RANK7_PA_COLUMNS = (
    "rex_144_range_pos",
    "rex_576_range_pos",
    "rex_2016_range_pos",
    "rex_8640_range_pos",
    "rex_2016_range_width_pct",
    "htf_4h_return_4",
    "htf_1d_return_4",
    "htf_1w_return_1",
    "htf_1d_range_pos",
    "htf_1w_range_pos",
)
RANK7_OTHER_COLUMNS = (
    "dxy_momentum",
    "usdkrw_zscore",
    "kimchi_premium_change",
    "taker_imbalance",
    "volume_zscore",
    "funding_zscore",
    "premium_index_zscore",
    "funding_rate",
    "premium_index_change",
)
RANK7_STATE_COLUMNS = (
    "k_slope",
    "k_innov",
    "b_segment",
    "b_reset",
    "b_flow",
    "s_trend",
    "s_vol",
    "s_flow",
    "s_age",
    "funding_leg",
    "premium_leg",
)
RANK7_WEAK_COLUMNS = (
    "nested_high_work_ratio",
    "nested_low_work_ratio",
    "nested_high_coalescence",
    "nested_low_coalescence",
    "nested_recent_24h_side",
    "nested_recent_48h_side",
    "nested_recent_age_capped",
    "braid_recent_24h_side",
    "braid_recent_48h_side",
    "braid_recent_age_capped",
)
RANK7_FEATURE_COLUMNS = (
    RANK7_STATE_COLUMNS
    + RANK7_PA_COLUMNS
    + RANK7_OTHER_COLUMNS
    + RANK7_WEAK_COLUMNS
)


@dataclass(frozen=True)
class StructuralTrade:
    signal_position: int
    entry_position: int
    exit_position: int
    side: int
    exit_kind: str


class Rank7FeatureError(RuntimeError):
    """Raised when a prefix cannot reproduce the frozen Rank7 feature graph."""


@dataclass(frozen=True)
class LearnerSpec:
    max_depth: int
    min_samples_leaf: int
    max_features: float


@dataclass(frozen=True)
class SelectionSpec:
    risk_lambda: float
    funding_quantile: float
    premium_quantile: float
    risk_quantile: float


def _coerce_naive_utc(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert(None)
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="raise", utc=True).dt.tz_convert(None)


def _normalise_aux_market_dates(market: pd.DataFrame) -> pd.DataFrame:
    if "date" not in market.columns:
        raise ValueError("market frame must contain a date column")
    out = market.copy()
    out["date"] = _coerce_naive_utc(out["date"])
    out["_row"] = np.arange(len(out))
    return out.sort_values("date").reset_index(drop=True)


def normalise_funding_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "funding_time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"funding_time": "date"})
    missing = {"date", "funding_rate"}.difference(df.columns)
    if missing:
        raise ValueError(f"funding frame lacks columns: {sorted(missing)}")
    out = df.loc[:, ["date", "funding_rate"]].copy()
    out["date"] = _coerce_naive_utc(out["date"])
    out["funding_rate"] = pd.to_numeric(out["funding_rate"], errors="coerce")
    return (
        out.dropna(subset=["date", "funding_rate"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def normalise_premium_index_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "premium_index" in df.columns and "date" in df.columns:
        out = df.loc[:, ["date", "premium_index"]].copy()
        out["date"] = _coerce_naive_utc(out["date"])
        out["premium_index"] = pd.to_numeric(
            out["premium_index"], errors="coerce"
        )
        return (
            out.dropna(subset=["date", "premium_index"])
            .sort_values("date")
            .drop_duplicates("date", keep="last")
            .reset_index(drop=True)
        )
    if "close" not in df.columns:
        raise ValueError("premium frame lacks close column")
    if "close_time" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["close_time"]):
            dates = _coerce_naive_utc(df["close_time"])
        else:
            numeric = pd.to_numeric(df["close_time"], errors="coerce")
            if numeric.notna().any():
                dates = pd.to_datetime(
                    numeric, unit="ms", utc=True
                ).dt.tz_convert(None)
            else:
                dates = pd.to_datetime(
                    df["close_time"], errors="raise", utc=True
                ).dt.tz_convert(None)
    elif "date" in df.columns:
        dates = pd.to_datetime(
            df["date"], errors="raise", utc=True
        ).dt.tz_convert(None)
    else:
        raise ValueError("premium frame lacks close_time/date column")
    out = pd.DataFrame(
        {
            "date": dates,
            "premium_index": pd.to_numeric(df["close"], errors="coerce"),
        }
    )
    return (
        out.dropna(subset=["date", "premium_index"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def _merge_aux(
    market: pd.DataFrame,
    aux: pd.DataFrame,
    *,
    value_cols: list[str],
    tolerance: str | pd.Timedelta | None,
) -> tuple[pd.DataFrame, pd.Series]:
    if aux.empty:
        return market.copy(), pd.Series(0.0, index=market.index)
    market_sorted = _normalise_aux_market_dates(market)
    aux_sorted = aux.loc[:, ["date", *value_cols]].copy().sort_values("date")
    rename = {
        column: f"__aux_{column}"
        for column in value_cols
        if column in market_sorted.columns
    }
    joined = pd.merge_asof(
        market_sorted,
        aux_sorted.rename(columns=rename),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(tolerance) if tolerance else None,
    )
    availability_source = rename.get(value_cols[0], value_cols[0])
    joined["__aux_available"] = joined[availability_source].notna().astype(float)
    for column in value_cols:
        source = rename.get(column, column)
        if source not in joined.columns:
            continue
        source_values = pd.to_numeric(joined[source], errors="coerce")
        if column in market.columns:
            base = pd.to_numeric(joined[column], errors="coerce")
            joined[column] = source_values.combine_first(base)
            if source != column:
                joined = joined.drop(columns=[source])
        else:
            joined[column] = source_values
            if source != column:
                joined = joined.drop(columns=[source])
    joined = joined.sort_values("_row").reset_index(drop=True)
    available = joined.pop("__aux_available").astype(float)
    return (
        joined.drop(columns=["_row"]),
        available.reset_index(drop=True),
    )


def _aux_rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    window = max(1, int(window))
    min_periods = min(window, max(2, window // 3))
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return (
        ((series - mean) / std.replace(0.0, np.nan))
        .clip(-5.0, 5.0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def attach_binance_um_aux_frames(
    market: pd.DataFrame,
    *,
    funding_frame: pd.DataFrame | None = None,
    premium_frame: pd.DataFrame | None = None,
    funding_tolerance: str | pd.Timedelta | None = "12h",
    premium_tolerance: str | pd.Timedelta | None = "2h",
    zscore_window: int = 96,
) -> pd.DataFrame:
    out = market.copy()
    availability_cols: list[str] = []
    if funding_frame is not None and not funding_frame.empty:
        out, available = _merge_aux(
            out,
            normalise_funding_history_frame(funding_frame),
            value_cols=["funding_rate"],
            tolerance=funding_tolerance,
        )
        out["funding_available"] = available.to_numpy(dtype=float)
        availability_cols.append("funding_available")
    if premium_frame is not None and not premium_frame.empty:
        out, available = _merge_aux(
            out,
            normalise_premium_index_frame(premium_frame),
            value_cols=["premium_index"],
            tolerance=premium_tolerance,
        )
        premium = pd.to_numeric(out["premium_index"], errors="coerce")
        out["premium_index_zscore"] = _aux_rolling_zscore(
            premium, zscore_window
        )
        out["premium_index_change"] = (
            premium.diff(max(1, int(zscore_window)))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        out["premium_available"] = available.to_numpy(dtype=float)
        availability_cols.append("premium_available")
    if availability_cols:
        out["binance_aux_any_available"] = out[availability_cols].max(axis=1)
    return out.replace([np.inf, -np.inf], np.nan)


def normalise_market(
    market: pd.DataFrame,
    *,
    exclude_from: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Apply the authenticated generic market ordering and duplicate policy."""

    if "date" not in market:
        raise ValueError("market frame must contain a date column")
    out = market.copy()
    out["date"] = pd.to_datetime(
        out["date"], utc=True, errors="raise"
    ).dt.tz_convert(None)
    out = (
        out.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    if exclude_from is not None:
        out = out[out["date"] < pd.Timestamp(exclude_from)].reset_index(drop=True)
    return out


def load_market(
    path: str | Path,
    *,
    funding_path: str | Path | None = None,
    premium_path: str | Path | None = None,
    exclude_from: str | pd.Timestamp | None = None,
    funding_tolerance: str | pd.Timedelta | None = "12h",
    premium_tolerance: str | pd.Timedelta | None = "2h",
) -> pd.DataFrame:
    """Load and causally attach the two Binance auxiliary sources."""

    market = pd.read_csv(path, parse_dates=["date"], compression="infer")
    market = normalise_market(market, exclude_from=exclude_from)
    if funding_path is None and premium_path is None:
        return market
    funding = (
        pd.read_csv(funding_path, compression="infer")
        if funding_path is not None
        else None
    )
    premium = (
        pd.read_csv(premium_path, compression="infer")
        if premium_path is not None
        else None
    )
    return attach_binance_um_aux_frames(
        market,
        funding_frame=funding,
        premium_frame=premium,
        funding_tolerance=funding_tolerance,
        premium_tolerance=premium_tolerance,
    )


def attach_open_interest(
    market: pd.DataFrame,
    open_interest: pd.DataFrame,
) -> pd.DataFrame:
    """Attach exact-timestamp OI and its positive finite availability flag."""

    required = {"date", "open_interest"}
    missing = sorted(required - set(open_interest.columns))
    if missing:
        raise ValueError(f"open-interest frame lacks columns: {missing}")
    aux = open_interest.loc[:, ["date", "open_interest"]].copy()
    aux["date"] = pd.to_datetime(
        aux["date"], utc=True, errors="raise"
    ).dt.tz_convert(None)
    aux["open_interest"] = pd.to_numeric(aux["open_interest"], errors="coerce")
    if aux["date"].duplicated().any() or not aux["date"].is_monotonic_increasing:
        raise ValueError("open-interest frame is duplicate or unsorted")
    out = normalise_market(market)
    out = out.drop(
        columns=[
            name
            for name in ("open_interest", "open_interest_available")
            if name in out
        ]
    ).merge(aux, on="date", how="left", validate="one_to_one")
    values = out["open_interest"].to_numpy(float)
    out["open_interest_available"] = (
        np.isfinite(values) & (values > 0.0)
    ).astype(float)
    return out


def _market_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean()
    relative_strength = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def _market_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int = 14,
) -> pd.Series:
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume
    change = typical_price.diff()
    positive = raw_money_flow.where(change > 0.0, 0.0)
    negative = raw_money_flow.where(change < 0.0, 0.0).abs()
    positive_sum = positive.rolling(length, min_periods=length).sum()
    negative_sum = negative.rolling(length, min_periods=length).sum()
    money_ratio = positive_sum / negative_sum
    return 100.0 - (100.0 / (1.0 + money_ratio))


def _market_clean(
    series: pd.Series, *, clip: float | None = None
) -> pd.Series:
    out = series.replace([np.inf, -np.inf], np.nan)
    if clip is not None:
        out = out.clip(-float(clip), float(clip))
    return out.fillna(0.0)


def _market_ratio_to_mean(series: pd.Series, window: int) -> pd.Series:
    average = series.rolling(window, min_periods=1).mean()
    return _market_clean(
        (series - average) / average.replace(0.0, np.nan)
    )


def _market_return_over(close: pd.Series, periods: int) -> pd.Series:
    reference = close.shift(max(1, int(periods)))
    return _market_clean(
        (close - reference) / reference.replace(0.0, np.nan)
    )


def _market_rolling_zscore(
    series: pd.Series, window: int, *, clip: float = 5.0
) -> pd.Series:
    window = max(1, int(window))
    min_periods = min(window, max(2, window // 3))
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return _market_clean(
        (series - mean) / std.replace(0.0, np.nan), clip=clip
    )


def _market_optional_column(
    frame: pd.DataFrame, name: str
) -> pd.Series | None:
    return frame[name].astype(float) if name in frame.columns else None


def _market_datetime(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert(None)
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def _completed_timeframe_features(
    market: pd.DataFrame,
    *,
    prefix: str,
    resample_rule: str,
    min_source_rows: int,
) -> dict[str, pd.Series]:
    defaults = {
        f"{prefix}_return_1": pd.Series(0.0, index=market.index),
        f"{prefix}_return_4": pd.Series(0.0, index=market.index),
        f"{prefix}_range_1": pd.Series(0.0, index=market.index),
        f"{prefix}_range_pos": pd.Series(0.0, index=market.index),
        f"{prefix}_drawdown_4": pd.Series(0.0, index=market.index),
    }
    if "date" not in market.columns or len(market) < int(min_source_rows):
        return defaults
    source = market[["date", "open", "high", "low", "close"]].copy()
    source["date"] = _market_datetime(source["date"])
    source = source.dropna(subset=["date"]).sort_values("date")
    if source.empty:
        return defaults
    source = source.set_index("date")
    higher = pd.DataFrame(
        {
            "open": source["open"]
            .resample(resample_rule, label="right", closed="right")
            .first(),
            "high": source["high"]
            .resample(resample_rule, label="right", closed="right")
            .max(),
            "low": source["low"]
            .resample(resample_rule, label="right", closed="right")
            .min(),
            "close": source["close"]
            .resample(resample_rule, label="right", closed="right")
            .last(),
        }
    ).dropna()
    if len(higher) < 2:
        return defaults
    previous = higher.shift(1)
    higher_range = (previous["high"] - previous["low"]).replace(0.0, np.nan)
    features = pd.DataFrame(index=higher.index)
    features[f"{prefix}_return_1"] = _market_clean(
        previous["close"] / previous["open"].replace(0.0, np.nan) - 1.0
    )
    features[f"{prefix}_return_4"] = _market_clean(
        previous["close"]
        / previous["close"].shift(4).replace(0.0, np.nan)
        - 1.0
    )
    features[f"{prefix}_range_1"] = _market_clean(
        higher_range / previous["close"].replace(0.0, np.nan)
    )
    features[f"{prefix}_range_pos"] = _market_clean(
        ((previous["close"] - previous["low"]) / higher_range) * 2.0 - 1.0
    )
    peak = previous["close"].rolling(4, min_periods=1).max()
    features[f"{prefix}_drawdown_4"] = _market_clean(
        1.0 - previous["close"] / peak.replace(0.0, np.nan)
    )
    features = (
        features.replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .reset_index(names="date")
    )
    target = pd.DataFrame({"date": _market_datetime(market["date"])})
    target["_row"] = np.arange(len(target))
    aligned = pd.merge_asof(
        target.sort_values("date"),
        features.sort_values("date"),
        on="date",
        direction="backward",
    ).sort_values("_row")
    return {
        column: pd.Series(
            aligned[column].fillna(0.0).to_numpy(), index=market.index
        )
        for column in defaults
    }


def _completed_multitimeframe_features(
    market: pd.DataFrame,
) -> dict[str, pd.Series]:
    specs = (
        ("htf_4h", "4h", 4 * 60 * 4),
        ("htf_1d", "1D", 24 * 60 * 4),
        ("htf_3d", "3D", 3 * 24 * 60 * 4),
        ("htf_1w", "W-SUN", 7 * 24 * 60),
    )
    out: dict[str, pd.Series] = {}
    for prefix, rule, min_rows in specs:
        out.update(
            _completed_timeframe_features(
                market,
                prefix=prefix,
                resample_rule=rule,
                min_source_rows=min_rows,
            )
        )
    for alias, source in {
        "weekly_return_1w": "htf_1w_return_1",
        "weekly_return_4w": "htf_1w_return_4",
        "weekly_range_1w": "htf_1w_range_1",
        "weekly_range_pos": "htf_1w_range_pos",
        "weekly_drawdown_4w": "htf_1w_drawdown_4",
    }.items():
        out[alias] = out.get(source, pd.Series(0.0, index=market.index))
    return out


def build_market_feature_frame(
    market: pd.DataFrame,
    *,
    window_size: int = 96,
    zscore_window: int = 48,
    volume_window: int = 48,
) -> pd.DataFrame:
    if market.empty:
        return pd.DataFrame(index=market.index)
    close = market["close"].astype(float)
    open_ = market["open"].astype(float)
    high = market["high"].astype(float)
    low = market["low"].astype(float)
    volume = market["volume"].astype(float)
    rolling_high = high.rolling(
        window_size, min_periods=window_size
    ).max()
    rolling_low = low.rolling(window_size, min_periods=window_size).min()
    range_mid = (rolling_high + rolling_low) / 2.0
    range_span = (rolling_high - rolling_low).replace(0.0, np.nan)
    features: dict[str, pd.Series] = {
        "range_vol": _market_clean(
            (rolling_high - rolling_low) / range_mid.replace(0.0, np.nan)
        ),
        "trend_12": _market_return_over(close, 11),
        "trend_24": _market_return_over(close, 23),
        "trend_96": _market_return_over(close, max(window_size - 1, 1)),
        "sma12_ratio": _market_ratio_to_mean(close, 12),
        "sma24_ratio": _market_ratio_to_mean(close, 24),
        "sma48_ratio": _market_ratio_to_mean(close, 48),
        "rsi_norm": _market_clean((_market_rsi(close) - 50.0) / 50.0),
        "mfi_norm": _market_clean(
            (_market_mfi(high, low, close, volume) - 50.0) / 50.0
        ),
    }
    bb_mean = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=1).std(ddof=0)
    features["bb_z"] = _market_clean(
        (close - bb_mean) / bb_std.replace(0.0, np.nan)
    )
    features["range_pos"] = _market_clean(
        ((close - rolling_low) / range_span) * 2.0 - 1.0
    )
    positive_close = close.where(close > 0.0)
    log_return = np.log(positive_close / positive_close.shift(1))
    features["close_zscore_48"] = _market_rolling_zscore(
        close, zscore_window
    )
    features["return_zscore_48"] = _market_rolling_zscore(
        log_return, zscore_window
    )
    upper_body = np.maximum(open_, close)
    lower_body = np.minimum(open_, close)
    candle_range = (high - low).replace(0.0, np.nan)
    features["body_ratio"] = _market_clean(
        (close - open_) / close.replace(0.0, np.nan)
    )
    features["upper_shadow"] = _market_clean(
        (high - upper_body) / close.replace(0.0, np.nan)
    )
    features["lower_shadow"] = _market_clean(
        (lower_body - low) / close.replace(0.0, np.nan)
    )
    features["candle_range"] = _market_clean(
        (high - low) / close.replace(0.0, np.nan)
    )
    features["body_to_range"] = _market_clean(
        (close - open_).abs() / candle_range
    )
    features["shadow_imbalance"] = _market_clean(
        features["lower_shadow"] - features["upper_shadow"]
    )
    volume_mean = volume.rolling(
        volume_window, min_periods=max(5, volume_window // 3)
    ).mean()
    volume_std = volume.rolling(
        volume_window, min_periods=max(5, volume_window // 3)
    ).std(ddof=0)
    features["volume_ratio"] = _market_clean(
        np.log1p(volume / volume_mean.replace(0.0, np.nan))
    )
    features["volume_zscore"] = _market_clean(
        (volume - volume_mean) / volume_std.replace(0.0, np.nan), clip=5.0
    )
    peak = close.rolling(window_size, min_periods=1).max()
    features["window_drawdown"] = _market_clean(
        1.0 - close / peak.replace(0.0, np.nan)
    )
    trades = _market_optional_column(market, "number_of_trades")
    if trades is None:
        features["trades_ratio"] = pd.Series(0.0, index=market.index)
    else:
        trades_mean = trades.rolling(
            volume_window, min_periods=max(5, volume_window // 3)
        ).mean()
        features["trades_ratio"] = _market_clean(
            np.log1p(trades / trades_mean.replace(0.0, np.nan))
        )
    taker_buy = _market_optional_column(market, "taker_buy_base")
    if taker_buy is None:
        features["taker_buy_ratio"] = pd.Series(0.5, index=market.index)
        features["taker_imbalance"] = pd.Series(0.0, index=market.index)
    else:
        ratio = (taker_buy / volume.replace(0.0, np.nan)).fillna(0.5)
        features["taker_buy_ratio"] = _market_clean(ratio)
        features["taker_imbalance"] = _market_clean(ratio * 2.0 - 1.0)
    funding = _market_optional_column(market, "funding_rate")
    if funding is None:
        features["funding_rate"] = pd.Series(0.0, index=market.index)
        features["funding_zscore"] = pd.Series(0.0, index=market.index)
    else:
        features["funding_rate"] = _market_clean(funding, clip=1.0)
        features["funding_zscore"] = _market_rolling_zscore(
            funding, volume_window
        )
    for column, default in {
        "funding_available": 0.0,
        "premium_index": 0.0,
        "premium_index_zscore": 0.0,
        "premium_index_change": 0.0,
        "premium_available": 0.0,
        "binance_aux_any_available": 0.0,
    }.items():
        series = _market_optional_column(market, column)
        features[column] = (
            _market_clean(series, clip=5.0)
            if series is not None
            else pd.Series(default, index=market.index)
        )
    open_interest = _market_optional_column(market, "open_interest")
    if open_interest is None:
        features["oi_change"] = pd.Series(0.0, index=market.index)
        features["oi_zscore"] = pd.Series(0.0, index=market.index)
    else:
        positive = open_interest.where(open_interest > 0.0)
        features["oi_change"] = _market_clean(
            np.log(positive / positive.shift(1))
        )
        features["oi_zscore"] = _market_rolling_zscore(
            open_interest, volume_window
        )
    for column, default in {
        "dxy": 0.0,
        "dxy_zscore": 0.0,
        "dxy_momentum": 0.0,
        "kimchi_premium": 0.0,
        "kimchi_premium_zscore": 0.0,
        "kimchi_premium_change": 0.0,
        "usdkrw_zscore": 0.0,
        "usdkrw_momentum": 0.0,
        "btckrw_zscore": 0.0,
        "btckrw_momentum": 0.0,
        "dxy_available": 0.0,
        "kimchi_available": 0.0,
        "usdkrw_available": 0.0,
        "external_any_available": 0.0,
    }.items():
        series = _market_optional_column(market, column)
        features[column] = (
            _market_clean(series, clip=5.0)
            if series is not None
            else pd.Series(default, index=market.index)
        )
    for raw_column in market.columns:
        column = str(raw_column)
        if not column.startswith("fx_") or not column.endswith(
            ("_zscore", "_momentum")
        ):
            continue
        series = _market_optional_column(market, column)
        if series is not None:
            features[column] = _market_clean(series, clip=5.0)
    for window in (36, 144, 576, 2016, 8640):
        min_periods = min(window, max(12, window // 4))
        rex_high = high.rolling(window, min_periods=min_periods).max()
        rex_low = low.rolling(window, min_periods=min_periods).min()
        rex_span = (rex_high - rex_low).replace(0.0, np.nan)
        prefix = f"rex_{window}"
        features[f"{prefix}_range_width_pct"] = _market_clean(
            rex_span / close.replace(0.0, np.nan)
        )
        features[f"{prefix}_range_pos"] = _market_clean(
            ((close - rex_low) / rex_span) * 2.0 - 1.0
        )
        features[f"{prefix}_cur_to_max_pct"] = _market_clean(
            close / rex_high.replace(0.0, np.nan) - 1.0
        )
        features[f"{prefix}_cur_to_min_pct"] = _market_clean(
            close / rex_low.replace(0.0, np.nan) - 1.0
        )
        features[f"{prefix}_max_to_cur_pct"] = _market_clean(
            rex_high / close.replace(0.0, np.nan) - 1.0
        )
    features.update(_completed_multitimeframe_features(market))
    return (
        pd.DataFrame(features, index=market.index)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(10, window // 5)).mean()
    std = series.rolling(window, min_periods=max(10, window // 5)).std(ddof=0)
    return (
        (series - mean)
        / std.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _relative_activity(series: pd.Series, fast: int, slow: int) -> pd.Series:
    fast_mean = series.rolling(
        fast, min_periods=max(5, fast // 5)
    ).mean()
    slow_mean = series.rolling(
        slow, min_periods=max(10, slow // 5)
    ).mean()
    return np.log(
        (fast_mean + 1.0) / (slow_mean + 1.0)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_interest_features(
    market: pd.DataFrame,
    base_features: pd.DataFrame,
) -> pd.DataFrame:
    """Source-equivalent causal interest/activity feature frame."""

    quote = market["quote_asset_volume"].astype(float)
    trades = market["number_of_trades"].astype(float)
    volume = market["volume"].astype(float)
    out = pd.DataFrame(index=market.index)
    out["quote_vol_z_1d"] = _rolling_z(np.log1p(quote), 288)
    out["quote_vol_rel_1d_30d"] = _relative_activity(quote, 288, 8640)
    out["trades_rel_1d_30d"] = _relative_activity(trades, 288, 8640)
    out["volume_rel_1d_30d"] = _relative_activity(volume, 288, 8640)
    out["dollar_flow_rel_4h_30d"] = _relative_activity(quote, 48, 8640)
    premium = pd.Series(
        np.abs(
            base_features.get(
                "premium_index", pd.Series(0.0, index=market.index)
            ).to_numpy(float)
        ),
        index=market.index,
    )
    funding = pd.Series(
        np.abs(
            base_features.get(
                "funding_rate", pd.Series(0.0, index=market.index)
            ).to_numpy(float)
        ),
        index=market.index,
    )
    out["premium_abs_z"] = _rolling_z(premium, 288)
    out["funding_abs_z"] = _rolling_z(funding, 288)
    out["interest_score"] = out[
        [
            "quote_vol_rel_1d_30d",
            "trades_rel_1d_30d",
            "volume_rel_1d_30d",
            "dollar_flow_rel_4h_30d",
            "premium_abs_z",
            "funding_abs_z",
        ]
    ].mean(axis=1)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_bidirectional_features(
    market: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Add the source-derived bidirectional flow features in place."""

    out = features.copy()
    close = market["close"].astype(float)
    quote = market["quote_asset_volume"].astype(float)
    buy = market["taker_buy_quote"].astype(float)
    imbalance = (2.0 * buy / quote.replace(0.0, np.nan) - 1.0).clip(-1, 1)
    for bars in (12, 24, 48, 96, 144):
        out[f"bd_ret_{bars}"] = np.log(close / close.shift(bars))
        out[f"bd_imb_{bars}"] = imbalance.rolling(
            bars, min_periods=bars
        ).mean()
    out["bd_flow_accel"] = out["bd_imb_12"] - out["bd_imb_48"]
    return out


def build_kimchi_features(
    market: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Add the source-derived causal Kimchi/FX lead-lag features."""

    def zscore(series: pd.Series, bars: int) -> pd.Series:
        mean = series.rolling(bars, min_periods=bars).mean()
        std = series.rolling(bars, min_periods=bars).std()
        return (series - mean) / std.replace(0.0, np.nan)

    out = features.copy()
    btc = np.log(market["close"].astype(float))
    kimchi = pd.to_numeric(
        market["kimchi_premium"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).ffill()
    krw = np.log(
        pd.to_numeric(market["usdkrw"], errors="coerce")
        .replace(0.0, np.nan)
        .ffill()
    )
    for bars in (12, 48, 144, 288):
        kimchi_delta = kimchi - kimchi.shift(bars)
        btc_return = btc - btc.shift(bars)
        fx_delta = krw - krw.shift(bars)
        out[f"kl_kimchi_delta_{bars}"] = kimchi_delta
        out[f"kl_btc_ret_{bars}"] = btc_return
        out[f"kl_fx_delta_{bars}"] = fx_delta
        out[f"kl_kimchi_btc_gap_{bars}"] = (
            zscore(kimchi_delta, 576) - zscore(btc_return, 576)
        )
        out[f"kl_local_impulse_{bars}"] = (
            zscore(kimchi_delta, 576) - zscore(fx_delta, 576)
        )
    out["kl_accel_48_144"] = (
        out["kl_kimchi_delta_48"] - out["kl_kimchi_delta_144"] / 3
    )
    return out.replace([np.inf, -np.inf], np.nan)


def gate_mask(
    frame: pd.DataFrame,
    gates: Sequence[Mapping[str, Any]],
    *,
    fallback: pd.DataFrame | None = None,
) -> np.ndarray:
    """Evaluate numeric ``ge/le`` and ``>=/<=`` gate dialects."""

    active = np.ones(len(frame), dtype=bool)
    for gate in gates:
        feature = str(gate["feature"])
        source = frame if feature in frame else fallback
        if source is None or feature not in source:
            raise ValueError(f"gate feature is absent: {feature}")
        values = source[feature].to_numpy(float)
        threshold = float(gate["threshold"])
        operator = str(gate["op"])
        if operator in ("ge", ">="):
            active &= np.isfinite(values) & (values >= threshold)
        elif operator in ("le", "<="):
            active &= np.isfinite(values) & (values <= threshold)
        else:
            raise ValueError(f"unsupported gate operator: {operator}")
    return active


def availability_mask(
    market: pd.DataFrame,
    columns: Iterable[str],
) -> np.ndarray:
    mask = np.ones(len(market), dtype=bool)
    for column in columns:
        if column not in market:
            raise ValueError(f"market source missing availability flag: {column}")
        values = pd.to_numeric(
            market[column], errors="coerce"
        ).to_numpy(float)
        mask &= np.isfinite(values) & (values > 0.5)
    return mask


def fresh_masks(
    market: pd.DataFrame,
    features: pd.DataFrame,
    *,
    long_conditions: Sequence[Mapping[str, Any]],
    short_conditions: Sequence[Mapping[str, Any]],
    long_availability: Sequence[str],
    short_availability: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Apply source gates and freshness flags without resolving side overlap."""

    raw_long = gate_mask(features, long_conditions)
    raw_short = gate_mask(features, short_conditions)
    long_fresh = availability_mask(market, long_availability)
    short_fresh = availability_mask(market, short_availability)
    long_active = raw_long & long_fresh
    short_active = raw_short & short_fresh
    return long_active, short_active, {
        "raw_long_rows": int(raw_long.sum()),
        "raw_short_rows": int(raw_short.sum()),
        "fresh_long_rows": int(long_active.sum()),
        "fresh_short_rows": int(short_active.sum()),
        "blocked_stale_long_rows": int((raw_long & ~long_fresh).sum()),
        "blocked_stale_short_rows": int((raw_short & ~short_fresh).sum()),
        "fresh_long_availability_violations": int(
            (long_active & ~long_fresh).sum()
        ),
        "fresh_short_availability_violations": int(
            (short_active & ~short_fresh).sum()
        ),
    }


def structural_trade_at(
    market: pd.DataFrame,
    signal_position: int,
    side: int,
    hold_bars: int,
    take_bps: int | float,
    stop_bps: int | float,
    *,
    entry_delay_bars: int = 1,
) -> StructuralTrade | None:
    """Replay only OHLC barrier geometry; stop wins a same-bar tie."""

    signal = int(signal_position)
    direction = int(side)
    if direction not in (-1, 1):
        raise ValueError("structural side must be -1 or 1")
    entry = signal + int(entry_delay_bars)
    fixed_exit = entry + int(hold_bars)
    if entry < 0 or entry >= len(market) or fixed_exit >= len(market):
        return None
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    entry_price = float(opens[entry])
    if not np.isfinite(entry_price) or entry_price <= 0.0:
        return None
    take = float(take_bps) / 10_000.0
    stop = float(stop_bps) / 10_000.0
    for position in range(entry, fixed_exit):
        stop_hit = (
            float(lows[position]) <= entry_price * (1.0 - stop)
            if direction > 0
            else float(highs[position]) >= entry_price * (1.0 + stop)
        )
        take_hit = (
            float(highs[position]) >= entry_price * (1.0 + take)
            if direction > 0
            else float(lows[position]) <= entry_price * (1.0 - take)
        )
        if stop_hit:
            return StructuralTrade(signal, entry, position, direction, "stop")
        if take_hit:
            return StructuralTrade(signal, entry, position, direction, "take")
    return StructuralTrade(signal, entry, fixed_exit, direction, "fixed")


def walk_structural_schedule(
    market: pd.DataFrame,
    signal_positions: Sequence[int] | np.ndarray,
    sides: Sequence[int] | np.ndarray,
    *,
    hold_bars: int,
    take_bps: int | float,
    stop_bps: int | float,
    period_mask: Sequence[bool] | np.ndarray | None = None,
    entry_delay_bars: int = 1,
) -> list[StructuralTrade]:
    """Greedily walk accepted signals with source-equivalent no-overlap."""

    positions = np.asarray(signal_positions, dtype=np.int64)
    side_values = np.asarray(sides, dtype=np.int8)
    if positions.shape != side_values.shape:
        raise ValueError("signal positions and sides must be aligned")
    period = (
        np.ones(len(market), dtype=bool)
        if period_mask is None
        else np.asarray(period_mask, dtype=bool)
    )
    if period.shape != (len(market),):
        raise ValueError("period mask has invalid shape")
    trades: list[StructuralTrade] = []
    next_allowed = 0
    for raw_signal, raw_side in zip(positions, side_values, strict=True):
        signal = int(raw_signal)
        if (
            signal < next_allowed
            or signal < 0
            or signal >= len(period)
            or not period[signal]
        ):
            continue
        trade = structural_trade_at(
            market,
            signal,
            int(raw_side),
            hold_bars,
            take_bps,
            stop_bps,
            entry_delay_bars=entry_delay_bars,
        )
        if trade is None or not period[trade.exit_position]:
            continue
        trades.append(trade)
        next_allowed = trade.exit_position + 1
    return trades


def completed_hourly_markov_features(
    market: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the exact completed-hour frame used by the Markov clock."""

    source = market.set_index(pd.to_datetime(market["date"])).sort_index()
    quote = source["quote_asset_volume"].astype(float)
    buy = source["taker_buy_quote"].astype(float)
    hourly = pd.DataFrame(
        {
            "open": source["open"].resample(
                "1h", closed="right", label="right"
            ).first(),
            "high": source["high"].resample(
                "1h", closed="right", label="right"
            ).max(),
            "low": source["low"].resample(
                "1h", closed="right", label="right"
            ).min(),
            "close": source["close"].resample(
                "1h", closed="right", label="right"
            ).last(),
            "quote": quote.resample(
                "1h", closed="right", label="right"
            ).sum(),
            "buy": buy.resample(
                "1h", closed="right", label="right"
            ).sum(),
        }
    ).dropna()
    returns = np.log(hourly["close"]).diff()
    flow = 2 * hourly["buy"] / hourly["quote"].replace(0, np.nan) - 1
    features = pd.DataFrame(index=hourly.index)
    features["ret1"] = returns
    features["trend24"] = np.log(
        hourly["close"] / hourly["close"].shift(24)
    )
    features["trend72"] = np.log(
        hourly["close"] / hourly["close"].shift(72)
    )
    features["vol24"] = returns.rolling(24).std()
    features["vol168"] = returns.rolling(168).std()
    features["volterm"] = (
        features["vol24"] / features["vol168"].replace(0, np.nan)
    )
    features["range24"] = (
        hourly["high"].rolling(24).max()
        - hourly["low"].rolling(24).min()
    ) / hourly["close"]
    features["flow24"] = flow.rolling(24).mean()
    log_quote = np.log1p(hourly["quote"])
    features["volume_z"] = (
        log_quote - log_quote.rolling(168).mean()
    ) / log_quote.rolling(168).std().replace(0, np.nan)
    return hourly, features.replace([np.inf, -np.inf], np.nan)


def markov_transition_keys(
    market: pd.DataFrame,
    *,
    trend_low: float,
    trend_high: float,
    vol_median: float,
    flow_median: float,
) -> np.ndarray:
    """Map completed hourly states backward onto the five-minute market."""

    _, hourly = completed_hourly_markov_features(market)
    trend = np.where(
        hourly["trend24"] <= float(trend_low),
        0,
        np.where(hourly["trend24"] >= float(trend_high), 2, 1),
    )
    volatility = (hourly["vol24"] >= float(vol_median)).astype(int)
    flow = (hourly["flow24"] >= float(flow_median)).astype(int)
    current = trend * 4 + volatility * 2 + flow
    previous = (
        pd.Series(current, index=hourly.index)
        .shift(1)
        .fillna(-1)
        .astype(int)
    )
    transitions = previous * 12 + current
    mapped = pd.merge_asof(
        pd.DataFrame(
            {
                "date": pd.to_datetime(market["date"]),
                "position": np.arange(len(market)),
            }
        ),
        pd.DataFrame(
            {
                "date": hourly.index.to_numpy(),
                "transition": transitions.to_numpy(),
            }
        ),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta("2h"),
    ).sort_values("position")
    return mapped["transition"].fillna(-1).to_numpy(int)


def markov_active(
    market: pd.DataFrame,
    setup: Sequence[bool] | np.ndarray,
    state_model: Mapping[str, Any],
) -> np.ndarray:
    transitions = markov_transition_keys(
        market,
        trend_low=float(state_model["trend_low"]),
        trend_high=float(state_model["trend_high"]),
        vol_median=float(state_model["vol_median"]),
        flow_median=float(state_model["flow_median"]),
    )
    allowed = np.asarray(state_model["allowed_transition_keys"], dtype=int)
    return np.asarray(setup, dtype=bool) & np.isin(transitions, allowed)


def _clean_rex(series: pd.Series, clip: float | None = None) -> pd.Series:
    out = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return out.clip(-float(clip), float(clip)) if clip is not None else out


def _rex_rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 3)).mean()
    std = series.rolling(
        window, min_periods=max(5, window // 3)
    ).std(ddof=0)
    return _clean_rex(
        (series - mean) / std.replace(0.0, np.nan), clip=5.0
    )


def build_light_rex_features(market: pd.DataFrame) -> pd.DataFrame:
    """Build the exact lightweight numeric frame used by REX veto gates."""

    close = market["close"].astype(float)
    high = market["high"].astype(float)
    low = market["low"].astype(float)
    volume = market["volume"].astype(float)
    out: dict[str, pd.Series] = {}
    roll_high = high.rolling(144, min_periods=144).max()
    roll_low = low.rolling(144, min_periods=144).min()
    span = (roll_high - roll_low).replace(0.0, np.nan)
    mid = (roll_high + roll_low) / 2.0
    out["range_vol"] = _clean_rex(
        (roll_high - roll_low) / mid.replace(0.0, np.nan)
    )
    out["range_pos"] = _clean_rex(
        ((close - roll_low) / span) * 2.0 - 1.0
    )
    out["trend_24"] = _clean_rex(
        close / close.shift(23).replace(0.0, np.nan) - 1.0
    )
    out["trend_96"] = _clean_rex(
        close / close.shift(143).replace(0.0, np.nan) - 1.0
    )
    log_return = np.log(close / close.shift(1).replace(0.0, np.nan))
    out["return_zscore_48"] = _rex_rolling_z(log_return, 48)
    difference = close.diff()
    gain = difference.clip(lower=0.0).ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    loss = (-difference.clip(upper=0.0)).ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    rsi = 100.0 - (100.0 / (1.0 + gain / loss.replace(0.0, np.nan)))
    out["rsi_norm"] = _clean_rex((rsi - 50.0) / 50.0)
    bb_mean = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=1).std(ddof=0)
    out["bb_z"] = _clean_rex(
        (close - bb_mean) / bb_std.replace(0.0, np.nan), clip=5.0
    )
    vol_mean = volume.rolling(48, min_periods=16).mean()
    vol_std = volume.rolling(48, min_periods=16).std(ddof=0)
    out["volume_zscore"] = _clean_rex(
        (volume - vol_mean) / vol_std.replace(0.0, np.nan), clip=5.0
    )
    if "taker_buy_base" in market:
        out["taker_imbalance"] = _clean_rex(
            (
                market["taker_buy_base"].astype(float)
                / volume.replace(0.0, np.nan)
            ).fillna(0.5)
            * 2.0
            - 1.0
        )
    else:
        out["taker_imbalance"] = pd.Series(0.0, index=market.index)
    peak = close.rolling(144, min_periods=1).max()
    out["window_drawdown"] = _clean_rex(
        1.0 - close / peak.replace(0.0, np.nan)
    )
    for name, bars in (
        ("htf_4h_return_1", 48),
        ("htf_4h_return_4", 192),
        ("htf_1d_return_1", 288),
        ("htf_1d_return_4", 1152),
        ("htf_3d_return_4", 3456),
        ("htf_1w_return_4", 8064),
    ):
        out[name] = _clean_rex(
            close / close.shift(bars).replace(0.0, np.nan) - 1.0
        )
    for column in (
        "dxy_zscore",
        "dxy_momentum",
        "usdkrw_zscore",
        "usdkrw_momentum",
        "kimchi_premium_zscore",
        "kimchi_premium_change",
    ):
        out[column] = (
            _clean_rex(market[column], clip=5.0)
            if column in market
            else pd.Series(0.0, index=market.index)
        )
    out["oi_zscore"] = (
        _rex_rolling_z(market["open_interest"].astype(float), 48)
        if "open_interest" in market
        else pd.Series(0.0, index=market.index)
    )
    for bars in (144, 576, 2016, 8640):
        min_periods = min(bars, max(12, bars // 4))
        rolling_high = high.rolling(bars, min_periods=min_periods).max()
        rolling_low = low.rolling(bars, min_periods=min_periods).min()
        rolling_span = (rolling_high - rolling_low).replace(0.0, np.nan)
        prefix = f"rex_{bars}"
        out[f"{prefix}_range_width_pct"] = _clean_rex(
            rolling_span / close.replace(0.0, np.nan)
        )
        out[f"{prefix}_range_pos"] = _clean_rex(
            ((close - rolling_low) / rolling_span) * 2.0 - 1.0
        )
        out[f"{prefix}_max_to_cur_pct"] = _clean_rex(
            rolling_high / close.replace(0.0, np.nan) - 1.0
        )
        out[f"{prefix}_cur_to_min_pct"] = _clean_rex(
            close / rolling_low.replace(0.0, np.nan) - 1.0
        )
    return (
        pd.DataFrame(out, index=market.index)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )


def rex_veto_gate_match(
    gates: Sequence[Mapping[str, Any]],
    features: pd.DataFrame,
    source_row: Mapping[str, Any],
) -> bool:
    """Evaluate the REX-veto dialect (numeric frame plus state tokens)."""

    position = int(source_row.get("signal_pos", -1))
    if position < 0 or position >= len(features):
        return False
    tokens = {
        f"tok:{key}": str(value)
        for key, value in (source_row.get("state_tokens") or {}).items()
    }
    for gate in gates:
        name = str(gate["feature"])
        if name.startswith("tok:"):
            if tokens.get(name, "") != str(gate["threshold"]):
                return False
            continue
        if name not in features:
            return False
        value = float(features.iloc[position][name])
        if not np.isfinite(value):
            return False
        threshold = float(gate["threshold"])
        if str(gate["op"]) == ">=" and not value >= threshold:
            return False
        if str(gate["op"]) == "<=" and not value <= threshold:
            return False
    return True


def rex_taker_gate_match(
    source_row: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
) -> bool:
    """Evaluate the REX-taker feature-snapshot dialect."""

    snapshot = source_row.get("feature_snapshot") or {}
    for gate in gates:
        try:
            value = float(snapshot[str(gate["feature"])])
        except (KeyError, TypeError, ValueError):
            return False
        operator = str(gate["op"])
        threshold = float(gate["threshold"])
        if operator == ">=" and value < threshold:
            return False
        if operator == "<=" and value > threshold:
            return False
        if operator not in {">=", "<="}:
            raise ValueError(f"unsupported gate op: {operator}")
    return True


def annual_rank7_masks(
    base: Mapping[str, Any],
    start: str,
    end: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build annual expanding-fit masks and purge labels reaching cutoff."""

    cutoff = pd.Timestamp(start)
    fit = np.asarray(
        (base["signal_dates"] >= RANK7_FIT_START)
        & (base["signal_dates"] < cutoff)
        & np.isfinite(base["targets"]).all(axis=1)
        & (base["exit_dates"] < cutoff.to_datetime64()),
        dtype=bool,
    )
    predict = np.asarray(
        (base["signal_dates"] >= cutoff)
        & (base["signal_dates"] < pd.Timestamp(end)),
        dtype=bool,
    )
    if not fit.any():
        raise RuntimeError(f"empty cadence fit mask for {start}..{end}")
    return fit, predict


def balanced_rank7_weights(
    base: Mapping[str, Any],
    fit: np.ndarray,
) -> np.ndarray:
    """Balance every observed ``(year, source)`` fit group equally."""

    years = pd.to_datetime(
        base["context"]["dates"].iloc[base["signals"][fit]]
    ).dt.year.to_numpy()
    sources = base["funding_source"][fit]
    groups = list(zip(years.tolist(), sources.tolist(), strict=True))
    counts = {group: groups.count(group) for group in set(groups)}
    weights = np.asarray([1.0 / counts[group] for group in groups], dtype=float)
    return weights * (len(weights) / weights.sum())


def deterministic_extra_trees_predict(
    model: Any,
    matrix: np.ndarray,
) -> np.ndarray:
    """Force fixed prediction order after deterministic parallel fitting."""

    model.n_jobs = 1
    return np.asarray(model.predict(matrix), dtype=float)


def annual_rank7_windows(
    start: str,
    end: str,
) -> tuple[tuple[str, str, str], ...]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    rows: list[tuple[str, str, str]] = []
    cursor = start_ts
    while cursor < end_ts:
        next_year = pd.Timestamp(year=cursor.year + 1, month=1, day=1)
        fold_end = min(next_year, end_ts)
        rows.append(
            (
                f"year_{cursor:%Y}",
                cursor.strftime("%Y-%m-%d"),
                fold_end.strftime("%Y-%m-%d"),
            )
        )
        cursor = fold_end
    return tuple(rows)


def fit_annual_rank7_folds(
    base: Mapping[str, Any],
    learner: LearnerSpec,
    *,
    start: str = "2023-01-01",
    end: str = "2026-06-02",
    seeds: Sequence[int] = RANK7_SEEDS,
    trees: int = RANK7_TREES,
) -> list[dict[str, Any]]:
    """Fit deterministic annual ExtraTrees folds on purged examples."""

    matrix = np.asarray(base["context"]["matrix"], dtype=float)
    folds: list[dict[str, Any]] = []
    for name, fold_start, fold_end in annual_rank7_windows(start, end):
        fit, predict = annual_rank7_masks(base, fold_start, fold_end)
        x_fit = matrix[base["signals"][fit]]
        y_fit = base["targets"][fit]
        weights = balanced_rank7_weights(base, fit)
        train_predictions: list[np.ndarray] = []
        period_predictions: list[np.ndarray] = []
        for seed in seeds:
            model = ExtraTreesRegressor(
                n_estimators=int(trees),
                max_depth=learner.max_depth,
                min_samples_leaf=learner.min_samples_leaf,
                max_features=learner.max_features,
                bootstrap=False,
                random_state=int(seed),
                n_jobs=-1,
            ).fit(x_fit, y_fit, sample_weight=weights)
            train_predictions.append(
                deterministic_extra_trees_predict(model, x_fit)
            )
            if predict.any():
                period_predictions.append(
                    deterministic_extra_trees_predict(
                        model, matrix[base["signals"][predict]]
                    )
                )
            else:
                period_predictions.append(
                    np.empty((0, y_fit.shape[1]), dtype=float)
                )
        folds.append(
            {
                "name": name,
                "start": fold_start,
                "end": fold_end,
                "fit": fit,
                "predict": predict,
                "fit_examples": int(fit.sum()),
                "predict_events": int(predict.sum()),
                "latest_fit_exit": str(
                    pd.Timestamp(base["exit_dates"][fit].max())
                ),
                "train_prediction": np.mean(
                    np.stack(train_predictions), axis=0
                ),
                "period_prediction": np.mean(
                    np.stack(period_predictions), axis=0
                ),
            }
        )
    return folds


def rank7_source_thresholds(
    train_predictions: np.ndarray,
    train_is_funding: np.ndarray,
    *,
    funding_q: float,
    premium_q: float,
) -> tuple[float, float]:
    predictions = np.asarray(train_predictions, dtype=float)
    funding = np.asarray(train_is_funding, dtype=bool)
    if predictions.shape != funding.shape or not funding.any() or funding.all():
        raise ValueError(
            "both aligned funding and premium train examples are required"
        )
    return (
        float(np.quantile(predictions[funding], funding_q)),
        float(np.quantile(predictions[~funding], premium_q)),
    )


def rank7_activation(
    base: Mapping[str, Any],
    folds: Iterable[Mapping[str, Any]],
    spec: SelectionSpec,
) -> np.ndarray:
    """Apply frozen source score/risk gates and funding interaction."""

    active = np.zeros(len(base["context"]["market"]), dtype=bool)
    for fold in folds:
        fit = np.asarray(fold["fit"], dtype=bool)
        predict = np.asarray(fold["predict"], dtype=bool)
        fit_source = base["funding_source"][fit]
        predict_source = base["funding_source"][predict]
        train_prediction = np.asarray(fold["train_prediction"], dtype=float)
        period_prediction = np.asarray(
            fold["period_prediction"], dtype=float
        )
        train_score = (
            train_prediction[:, 0]
            - spec.risk_lambda * train_prediction[:, 1]
        )
        period_score = (
            period_prediction[:, 0]
            - spec.risk_lambda * period_prediction[:, 1]
        )
        funding_threshold, premium_threshold = rank7_source_thresholds(
            train_score,
            fit_source,
            funding_q=spec.funding_quantile,
            premium_q=spec.premium_quantile,
        )
        funding_risk_cap = float(
            np.quantile(
                train_prediction[fit_source, 1], spec.risk_quantile
            )
        )
        premium_risk_cap = float(
            np.quantile(
                train_prediction[~fit_source, 1], spec.risk_quantile
            )
        )
        positions = base["signals"][predict]
        fit_positions = base["signals"][fit]
        funding_interaction = (
            base["width"][positions]
            > float(
                np.quantile(
                    base["width"][fit_positions][fit_source], 0.20
                )
            )
        ) | (
            base["pullback"][positions]
            <= float(
                np.quantile(
                    base["pullback"][fit_positions][fit_source], 0.40
                )
            )
        )
        selected = (
            predict_source
            & (period_score >= funding_threshold)
            & (period_prediction[:, 1] <= funding_risk_cap)
            & funding_interaction
        ) | (
            (~predict_source)
            & (period_score >= premium_threshold)
            & (period_prediction[:, 1] <= premium_risk_cap)
        )
        active[positions] = selected
    return active


def apply_rank7_delay(
    matrix: np.ndarray,
    *,
    bars: int,
    initial_fill: float = 0.0,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    delayed = np.full(values.shape, float(initial_fill), dtype=float)
    lag = int(bars)
    if lag <= 0:
        delayed[:] = values
    elif lag < len(values):
        delayed[lag:] = values[:-lag]
    return delayed


def immutable_anchors(active: np.ndarray, cooldown: int) -> np.ndarray:
    source = np.asarray(active, dtype=bool)
    anchors = np.zeros(len(source), dtype=bool)
    next_allowed = 0
    for position in np.flatnonzero(source):
        if int(position) < next_allowed:
            continue
        anchors[int(position)] = True
        next_allowed = int(position) + int(cooldown)
    return anchors


def causal_shift(
    values: np.ndarray,
    fill: float | int = 0,
) -> np.ndarray:
    array = np.asarray(values)
    out = np.full(
        array.shape,
        fill,
        dtype=np.result_type(array.dtype, np.asarray(fill).dtype),
    )
    out[1:] = array[:-1]
    return out


def recent_side(side: np.ndarray, bars: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(side, dtype=np.int8)
    index = np.arange(len(values))
    last = np.maximum.accumulate(np.where(values != 0, index, -1))
    age = index - last
    out = np.zeros(len(values), dtype=np.int8)
    valid = (last >= 0) & (age <= int(bars))
    out[valid] = values[last[valid]]
    return out, np.where(last >= 0, age, 9999)


def _rank7_prior_extreme_index(
    values: np.ndarray, window: int, kind: str
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.full(len(values), -1, dtype=np.int64)
    candidates: deque[int] = deque()
    for position in range(len(values)):
        previous = position - 1
        if previous >= 0 and np.isfinite(values[previous]):
            if kind == "max":
                while (
                    candidates
                    and values[candidates[-1]] <= values[previous]
                ):
                    candidates.pop()
            elif kind == "min":
                while (
                    candidates
                    and values[candidates[-1]] >= values[previous]
                ):
                    candidates.pop()
            else:
                raise KeyError(kind)
            candidates.append(previous)
        cutoff = position - window
        while candidates and candidates[0] < cutoff:
            candidates.popleft()
        if position >= window and candidates:
            output[position] = candidates[0]
    return output


def _rank7_directional_work(
    market: pd.DataFrame, bars: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    imbalance = (2.0 * taker_buy - quote) / quote.replace(0.0, np.nan)
    buy_work = (
        imbalance.clip(lower=0.0)
        .rolling(bars, min_periods=bars)
        .sum()
        .to_numpy(float)
    )
    sell_work = (
        (-imbalance.clip(upper=0.0))
        .rolling(bars, min_periods=bars)
        .sum()
        .to_numpy(float)
    )
    return buy_work, sell_work


def _rank7_build_barrier_bank(
    market: pd.DataFrame,
) -> dict[Any, Any]:
    high = market["high"].to_numpy(float)
    low = market["low"].to_numpy(float)
    bank: dict[Any, Any] = {}
    for horizon in _RANK7_BARRIER_HORIZONS:
        high_index = _rank7_prior_extreme_index(high, horizon, "max")
        low_index = _rank7_prior_extreme_index(low, horizon, "min")
        high_price = np.full(len(high), np.nan)
        low_price = np.full(len(low), np.nan)
        high_valid = high_index >= 0
        low_valid = low_index >= 0
        high_price[high_valid] = high[high_index[high_valid]]
        low_price[low_valid] = low[low_index[low_valid]]
        bank[horizon] = {
            "high_index": high_index,
            "low_index": low_index,
            "high_price": high_price,
            "low_price": low_price,
        }
    bank["buy_work"], bank["sell_work"] = _rank7_directional_work(market)
    return bank


def _rank7_coalesced_barrier_signals(
    market: pd.DataFrame,
    bank: dict[Any, Any],
    *,
    min_coalescence: int,
    touch_width: float,
    branch: str,
    work_low: float = 0.75,
    work_high: float = 1.25,
    max_origin_separation: int = 3,
    flip: bool = False,
    ignore_witness: bool = False,
    ignore_coalescence: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    close = market["close"].to_numpy(float)
    buy_work = bank["buy_work"]
    sell_work = bank["sell_work"]
    high_active = np.zeros(len(market), dtype=bool)
    low_active = np.zeros(len(market), dtype=bool)
    high_ratio = np.full(len(market), np.nan)
    low_ratio = np.full(len(market), np.nan)
    high_count = np.zeros(len(market), dtype=np.int8)
    low_count = np.zeros(len(market), dtype=np.int8)
    for position in range(len(market)):
        for barrier_side in ("high", "low"):
            touched: list[tuple[int, int, float]] = []
            for horizon in _RANK7_BARRIER_HORIZONS:
                witness_index = int(
                    bank[horizon][f"{barrier_side}_index"][position]
                )
                level = float(
                    bank[horizon][f"{barrier_side}_price"][position]
                )
                if (
                    witness_index >= 0
                    and np.isfinite(level)
                    and abs(close[position] / level - 1.0) <= touch_width
                ):
                    touched.append((horizon, witness_index, level))
            required = 1 if ignore_coalescence else min_coalescence
            if len(touched) < required:
                continue
            indices = [item[1] for item in touched]
            if (
                not ignore_coalescence
                and max(indices) - min(indices) > max_origin_separation
            ):
                continue
            _, witness_index, _ = max(touched, key=lambda item: item[0])
            levels = [item[2] for item in touched]
            if barrier_side == "high":
                high_count[position] = len(touched)
                origin_work = buy_work[witness_index]
                current_work = buy_work[position]
                price_not_closed_through = close[position] <= max(levels)
                ratio = (
                    current_work / origin_work
                    if (
                        np.isfinite(origin_work)
                        and origin_work > 1e-4
                        and np.isfinite(current_work)
                    )
                    else np.nan
                )
                high_ratio[position] = ratio
                if branch == "depleted_continuation":
                    selected = price_not_closed_through and (
                        ignore_witness
                        or (np.isfinite(ratio) and ratio <= work_low)
                    )
                elif branch == "reinforced_fade":
                    selected = price_not_closed_through and (
                        ignore_witness
                        or (np.isfinite(ratio) and ratio >= work_high)
                    )
                else:
                    raise KeyError(branch)
                high_active[position] = selected
            else:
                low_count[position] = len(touched)
                origin_work = sell_work[witness_index]
                current_work = sell_work[position]
                price_not_closed_through = close[position] >= min(levels)
                ratio = (
                    current_work / origin_work
                    if (
                        np.isfinite(origin_work)
                        and origin_work > 1e-4
                        and np.isfinite(current_work)
                    )
                    else np.nan
                )
                low_ratio[position] = ratio
                if branch == "depleted_continuation":
                    selected = price_not_closed_through and (
                        ignore_witness
                        or (np.isfinite(ratio) and ratio <= work_low)
                    )
                elif branch == "reinforced_fade":
                    selected = price_not_closed_through and (
                        ignore_witness
                        or (np.isfinite(ratio) and ratio >= work_high)
                    )
                else:
                    raise KeyError(branch)
                low_active[position] = selected
    high_onset = high_active & ~np.r_[False, high_active[:-1]]
    low_onset = low_active & ~np.r_[False, low_active[:-1]]
    if branch == "depleted_continuation":
        long_active = high_onset & ~low_onset
        short_active = low_onset & ~high_onset
    else:
        long_active = low_onset & ~high_onset
        short_active = high_onset & ~low_onset
    if flip:
        long_active, short_active = short_active, long_active
    return long_active, short_active, {
        "high_work_ratio": high_ratio,
        "low_work_ratio": low_ratio,
        "high_coalescence": high_count,
        "low_coalescence": low_count,
    }


def _rank7_braid_prior_std(values: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(values, errors="coerce")
        .shift(1)
        .rolling(
            _RANK7_BRAID_SCALE_WINDOW,
            min_periods=_RANK7_BRAID_SCALE_MIN_PERIODS,
        )
        .std(ddof=0)
        .replace(0.0, np.nan)
    )


def _rank7_build_braid_state(
    market: pd.DataFrame,
) -> dict[str, np.ndarray]:
    spot = np.log(
        pd.to_numeric(market["spot_close"], errors="coerce").where(
            lambda value: value > 0.0
        )
    )
    perp = np.log(
        pd.to_numeric(market["close"], errors="coerce").where(
            lambda value: value > 0.0
        )
    )
    premium = pd.to_numeric(
        market["premium_index_1m_close"], errors="coerce"
    )
    raw_oi = pd.to_numeric(market["open_interest"], errors="coerce")
    raw_available = pd.to_numeric(
        market["open_interest_available"], errors="coerce"
    ).eq(1.0)
    delayed_oi = raw_oi.shift(1)
    delayed_available = raw_available.shift(1, fill_value=False)
    log_oi = np.log(delayed_oi.where(delayed_oi > 0.0))
    complete = (
        pd.to_numeric(market["spot_rows"], errors="coerce").eq(5.0)
        & pd.to_numeric(market["premium_rows"], errors="coerce").eq(5.0)
        & delayed_available
        & spot.notna()
        & perp.notna()
        & premium.notna()
        & log_oi.notna()
    )
    pair_complete = complete & complete.shift(1, fill_value=False)
    spot_return = spot.diff().where(pair_complete)
    perp_return = perp.diff().where(pair_complete)
    common_return = 0.5 * (spot_return + perp_return)
    twelve_complete = complete & complete.shift(12, fill_value=False)
    return {
        "spot": spot.to_numpy(float),
        "perp": perp.to_numpy(float),
        "log_oi": log_oi.to_numpy(float),
        "premium": premium.to_numpy(float),
        "valid": complete.to_numpy(bool),
        "shock_z": (
            common_return / _rank7_braid_prior_std(common_return)
        ).to_numpy(float),
        "spot_unit": _rank7_braid_prior_std(
            spot.diff(12).where(twelve_complete)
        ).to_numpy(float),
        "perp_unit": _rank7_braid_prior_std(
            perp.diff(12).where(twelve_complete)
        ).to_numpy(float),
        "oi_unit": _rank7_braid_prior_std(
            log_oi.diff(12).where(twelve_complete)
        ).to_numpy(float),
        "premium_unit": _rank7_braid_prior_std(
            premium.diff(12).where(twelve_complete)
        ).to_numpy(float),
    }


def _rank7_market_braid_events(
    state: dict[str, np.ndarray],
    *,
    shock_z: float,
    passage_z: float,
    max_age: int,
    topology_mode: str,
    leverage_mode: str = "joint",
) -> pd.DataFrame:
    if topology_mode not in {"strict_chain", "relative_order"}:
        raise KeyError(topology_mode)
    if leverage_mode not in {"joint", "oi_only", "premium_only", "none"}:
        raise KeyError(leverage_mode)
    size = len(state["spot"])
    signal_side = np.zeros(size, dtype=np.int8)
    impulse_side = np.zeros(size, dtype=np.int8)
    episode_age = np.zeros(size, dtype=np.int16)
    sequence = np.full(size, "", dtype=object)
    departure_event = np.zeros(size, dtype=bool)
    tie_discarded = np.zeros(size, dtype=bool)
    active = False
    for position in range(1, size):
        usable = state["valid"][position] and all(
            np.isfinite(state[name][position])
            for name in (
                "spot",
                "perp",
                "log_oi",
                "premium",
                "spot_unit",
                "perp_unit",
                "oi_unit",
                "premium_unit",
            )
        )
        usable = bool(
            usable
            and state["spot_unit"][position] > 0.0
            and state["perp_unit"][position] > 0.0
            and state["oi_unit"][position] > 0.0
            and state["premium_unit"][position] > 0.0
        )
        if active:
            if not usable:
                active = False
                continue
            age += 1
            hits: list[str] = []
            if (
                "spot" not in order
                and side * (state["spot"][position] - anchor_spot)
                >= passage_z * spot_unit
            ):
                hits.append("spot")
            if (
                "perp" not in order
                and side * (state["perp"][position] - anchor_perp)
                >= passage_z * perp_unit
            ):
                hits.append("perp")
            if leverage_mode != "none" and "leverage" not in order:
                oi_cross = (
                    state["log_oi"][position] - anchor_oi
                    >= _RANK7_BRAID_OI_PASSAGE_Z * oi_unit
                )
                premium_cross = (
                    side * (state["premium"][position] - anchor_premium)
                    >= _RANK7_BRAID_PREMIUM_PASSAGE_Z * premium_unit
                )
                leverage_cross = {
                    "joint": oi_cross and premium_cross,
                    "oi_only": oi_cross,
                    "premium_only": premium_cross,
                }[leverage_mode]
                if leverage_cross:
                    hits.append("leverage")
            if len(hits) > 1:
                tie_discarded[position] = True
                active = False
                continue
            if hits:
                order.append(hits[0])
                required = 2 if leverage_mode == "none" else 3
                if len(order) == required:
                    text = ">".join(order)
                    mapped = 0
                    if leverage_mode == "none":
                        mapped = side if order == ["spot", "perp"] else -side
                    elif topology_mode == "strict_chain":
                        if order == ["spot", "perp", "leverage"]:
                            mapped = side
                        elif order == ["leverage", "perp", "spot"]:
                            mapped = -side
                    else:
                        mapped = (
                            side
                            if order.index("spot") < order.index("leverage")
                            else -side
                        )
                    signal_side[position] = mapped
                    impulse_side[position] = side
                    episode_age[position] = age
                    sequence[position] = text
                    active = False
            if active and age >= max_age:
                active = False
        elif (
            usable
            and np.isfinite(state["shock_z"][position])
            and abs(state["shock_z"][position]) >= shock_z
        ):
            active = True
            departure_event[position] = True
            side = int(np.sign(state["shock_z"][position]))
            age = 0
            order = []
            anchor_spot = state["spot"][position]
            anchor_perp = state["perp"][position]
            anchor_oi = state["log_oi"][position]
            anchor_premium = state["premium"][position]
            spot_unit = state["spot_unit"][position]
            perp_unit = state["perp_unit"][position]
            oi_unit = state["oi_unit"][position]
            premium_unit = state["premium_unit"][position]
    return pd.DataFrame(
        {
            "signal_side": signal_side,
            "impulse_side": impulse_side,
            "episode_age": episode_age,
            "sequence": sequence,
            "departure_event": departure_event,
            "tie_discarded": tie_discarded,
        }
    )


def _rank7_decision_mask(
    dates: pd.Series,
    mode: str,
    *,
    window_size: int = 144,
) -> np.ndarray:
    parsed = pd.to_datetime(dates)
    if mode == "legacy_positional":
        mask = np.zeros(len(parsed), dtype=bool)
        mask[np.arange(max(143, int(window_size) - 1), len(parsed), 12)] = True
        return mask
    if mode == "live_hour_signal_bar":
        return ((parsed.dt.minute == 0) & (parsed.dt.second == 0)).to_numpy(bool)
    raise ValueError(f"unknown decision clock: {mode}")


def _rank7_live_decision_features(features: pd.DataFrame) -> pd.DataFrame:
    shifted = features.shift(1)
    for column in (
        "funding_rate",
        "funding_zscore",
        "funding_available",
        "premium_index",
        "premium_index_zscore",
        "premium_index_change",
        "premium_available",
        "binance_aux_any_available",
    ):
        if column in features:
            shifted[column] = features[column]
    return shifted


_RANK7_COMPONENTS = {
    "funding10_trend70": (
        ("funding_rate", "le", -0.0000167),
        ("trend_96", "ge", 0.007485218212390219),
    ),
    "premium20_mom90": (
        ("premium_index_change", "le", -0.00023471),
        ("htf_1d_return_4", "ge", 0.0940403008961932),
    ),
}


def _rank7_component_mask(features: pd.DataFrame, name: str) -> np.ndarray:
    mask = np.ones(len(features), dtype=bool)
    for feature, op, threshold in _RANK7_COMPONENTS[name]:
        if feature not in features.columns:
            return np.zeros(len(features), dtype=bool)
        values = features[feature].to_numpy(float)
        mask &= (
            (values <= threshold) if op == "le" else (values >= threshold)
        ) & np.isfinite(values)
    return mask


def _rank7_kalman_filter(
    log_price: np.ndarray,
    q_level: float,
    q_slope: float,
    r_obs: float,
    train_var: float,
) -> np.ndarray:
    values = np.asarray(log_price, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("log_price must be a non-empty 1-D array")
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    observation = np.array([1.0, 0.0])
    process_cov = np.diag([q_level * train_var, q_slope * train_var])
    observation_var = max(r_obs * train_var, 1e-12)
    state = np.array([values[0], 0.0])
    covariance = np.eye(2) * train_var * 100.0
    output = np.empty((len(values), 4), dtype=float)
    identity = np.eye(2)
    for idx, observed in enumerate(values):
        predicted_state = transition @ state
        predicted_cov = transition @ covariance @ transition.T + process_cov
        innovation = float(observed - observation @ predicted_state)
        innovation_var = float(
            observation @ predicted_cov @ observation + observation_var
        )
        gain = (predicted_cov @ observation) / innovation_var
        state = predicted_state + gain * innovation
        covariance = (identity - np.outer(gain, observation)) @ predicted_cov
        output[idx] = (
            state[0],
            state[1],
            innovation / np.sqrt(innovation_var),
            state[1] / np.sqrt(max(covariance[1, 1], 1e-12)),
        )
    return output


@dataclass(frozen=True)
class _Rank7BocpdState:
    weights: np.ndarray
    mean: np.ndarray
    kappa: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    previous_expected: float
    hazard_lambda: float
    max_run_length: int
    prior_kappa: float
    prior_alpha: float
    prior_beta: float
    short_run_horizon: int


def _rank7_student_t_log_predictive(
    observation: np.ndarray,
    mean: np.ndarray,
    kappa: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    degrees = 2.0 * alpha
    scale2 = beta * (kappa[:, None] + 1.0) / (alpha * kappa[:, None])
    centered2 = (observation[None, :] - mean) ** 2
    per_dimension = (
        gammaln((degrees + 1.0) / 2.0)
        - gammaln(degrees / 2.0)
        - 0.5 * (np.log(degrees * np.pi) + np.log(scale2))
        - 0.5
        * (degrees + 1.0)
        * np.log1p(centered2 / (degrees * scale2))
    )
    return per_dimension.sum(axis=1)


def _rank7_posterior_update(
    observation: np.ndarray,
    mean: np.ndarray,
    kappa: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    next_kappa = kappa + 1.0
    next_mean = (
        kappa[:, None] * mean + observation[None, :]
    ) / next_kappa[:, None]
    next_alpha = alpha + 0.5
    next_beta = beta + 0.5 * (
        kappa[:, None]
        * (observation[None, :] - mean) ** 2
        / next_kappa[:, None]
    )
    return next_mean, next_kappa, next_alpha, next_beta


def _rank7_bocpd(observations: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(observations, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("observations must be a non-empty 1-D or 2-D array")
    if not np.isfinite(values).all():
        raise ValueError("observations must be finite")
    dimensions = values.shape[1]
    state = _Rank7BocpdState(
        weights=np.array([1.0]),
        mean=np.zeros((1, dimensions), dtype=float),
        kappa=np.array([0.1], dtype=float),
        alpha=np.full((1, dimensions), 2.0, dtype=float),
        beta=np.full((1, dimensions), 1.0, dtype=float),
        previous_expected=0.0,
        hazard_lambda=336.0,
        max_run_length=1000,
        prior_kappa=0.1,
        prior_alpha=2.0,
        prior_beta=1.0,
        short_run_horizon=6,
    )
    hazard = 1.0 / state.hazard_lambda
    weights = state.weights.copy()
    mean = state.mean.copy()
    kappa = state.kappa.copy()
    alpha = state.alpha.copy()
    beta = state.beta.copy()
    expected_run = np.empty(len(values), dtype=float)
    map_run = np.empty(len(values), dtype=float)
    short_mass = np.empty(len(values), dtype=float)
    run_drop = np.empty(len(values), dtype=float)
    surprise = np.empty(len(values), dtype=float)
    posterior_mean = np.empty((len(values), dimensions), dtype=float)
    previous_expected = float(state.previous_expected)
    prior_mean = np.zeros((1, dimensions), dtype=float)
    prior_kappa = np.array([state.prior_kappa], dtype=float)
    prior_alpha = np.full((1, dimensions), state.prior_alpha, dtype=float)
    prior_beta = np.full((1, dimensions), state.prior_beta, dtype=float)
    for idx, observation in enumerate(values):
        log_predictive = _rank7_student_t_log_predictive(
            observation, mean, kappa, alpha, beta
        )
        log_joint = np.log(np.maximum(weights, 1e-300)) + log_predictive
        offset = float(np.max(log_joint))
        joint = np.exp(log_joint - offset)
        surprise[idx] = -(offset + np.log(np.sum(joint)))
        reset_probability = hazard * float(np.sum(joint))
        growth_probability = (1.0 - hazard) * joint
        next_weights = np.r_[reset_probability, growth_probability]
        reset_params = _rank7_posterior_update(
            observation, prior_mean, prior_kappa, prior_alpha, prior_beta
        )
        growth_params = _rank7_posterior_update(
            observation, mean, kappa, alpha, beta
        )
        next_mean = np.vstack([reset_params[0], growth_params[0]])
        next_kappa = np.r_[reset_params[1], growth_params[1]]
        next_alpha = np.vstack([reset_params[2], growth_params[2]])
        next_beta = np.vstack([reset_params[3], growth_params[3]])
        keep = min(len(next_weights), state.max_run_length + 1)
        weights = next_weights[:keep]
        weights /= np.sum(weights)
        mean = next_mean[:keep]
        kappa = next_kappa[:keep]
        alpha = next_alpha[:keep]
        beta = next_beta[:keep]
        run_axis = np.arange(keep, dtype=float)
        current_expected = float(weights @ run_axis)
        expected_run[idx] = current_expected
        map_run[idx] = float(np.argmax(weights))
        short_mass[idx] = float(
            weights[: min(state.short_run_horizon + 1, keep)].sum()
        )
        expected_without_reset = previous_expected + 1.0
        run_drop[idx] = max(
            0.0, expected_without_reset - current_expected
        ) / max(expected_without_reset, 1.0)
        posterior_mean[idx] = weights @ mean
        previous_expected = current_expected
    return {
        "expected_run": expected_run,
        "map_run": map_run,
        "short_mass": short_mass,
        "run_drop": run_drop,
        "surprise": surprise,
        "posterior_mean": posterior_mean,
    }


def _rank7_completed_hourly_features(
    market: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = market.copy()
    source["date"] = pd.to_datetime(
        source["date"], utc=False, errors="raise"
    )
    source = source.sort_values("date").set_index("date")
    quote = pd.to_numeric(source["quote_asset_volume"], errors="coerce")
    buy = pd.to_numeric(source["taker_buy_quote"], errors="coerce")
    grouped = pd.DataFrame(
        {
            "open": pd.to_numeric(source["open"], errors="coerce")
            .resample("1h", closed="left", label="right")
            .first(),
            "high": pd.to_numeric(source["high"], errors="coerce")
            .resample("1h", closed="left", label="right")
            .max(),
            "low": pd.to_numeric(source["low"], errors="coerce")
            .resample("1h", closed="left", label="right")
            .min(),
            "close": pd.to_numeric(source["close"], errors="coerce")
            .resample("1h", closed="left", label="right")
            .last(),
            "quote": quote.resample(
                "1h", closed="left", label="right"
            ).sum(),
            "buy": buy.resample("1h", closed="left", label="right").sum(),
            "bar_count": source["close"]
            .resample("1h", closed="left", label="right")
            .count(),
        }
    )
    hourly = (
        grouped.loc[grouped["bar_count"] >= 12]
        .drop(columns="bar_count")
        .dropna()
    )
    return hourly, _rank7_hourly_state_features(hourly)


def _rank7_hourly_state_features(hourly: pd.DataFrame) -> pd.DataFrame:
    required = {"high", "low", "close", "quote", "buy"}
    missing = sorted(required - set(hourly.columns))
    if missing:
        raise ValueError(f"completed hourly frame missing columns: {missing}")
    returns = np.log(hourly["close"]).diff()
    flow = 2.0 * hourly["buy"] / hourly["quote"].replace(0.0, np.nan) - 1.0
    features = pd.DataFrame(index=hourly.index)
    features["ret1"] = returns
    features["trend24"] = np.log(
        hourly["close"] / hourly["close"].shift(24)
    )
    features["trend72"] = np.log(
        hourly["close"] / hourly["close"].shift(72)
    )
    features["vol24"] = returns.rolling(24).std()
    features["vol168"] = returns.rolling(168).std()
    features["volterm"] = (
        features["vol24"] / features["vol168"].replace(0.0, np.nan)
    )
    features["range24"] = (
        hourly["high"].rolling(24).max()
        - hourly["low"].rolling(24).min()
    ) / hourly["close"]
    features["flow24"] = flow.rolling(24).mean()
    log_quote = np.log1p(hourly["quote"])
    features["volume_z"] = (
        log_quote - log_quote.rolling(168).mean()
    ) / log_quote.rolling(168).std().replace(0.0, np.nan)
    return features.replace([np.inf, -np.inf], np.nan)


def _rank7_map_hourly(
    dates: pd.Series,
    hourly: pd.DataFrame,
    value_column: str,
) -> np.ndarray:
    mapped = pd.merge_asof(
        pd.DataFrame(
            {"date": pd.to_datetime(dates), "pos": np.arange(len(dates))}
        ),
        hourly[["date", value_column]].sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta("2h"),
    ).sort_values("pos")
    return mapped[value_column].fillna(-1).to_numpy(int)


def _rank7_observable_state(
    features: pd.DataFrame, fit: np.ndarray
) -> np.ndarray:
    training = features.loc[np.asarray(fit, dtype=bool)]
    trend = features["trend24"].to_numpy(float)
    volatility = features["vol24"].to_numpy(float)
    flow = features["flow24"].to_numpy(float)
    trend_low = float(training["trend24"].quantile(0.33))
    trend_high = float(training["trend24"].quantile(0.67))
    vol_median = float(training["vol24"].quantile(0.5))
    flow_median = float(training["flow24"].quantile(0.5))
    state = (
        np.where(
            trend <= trend_low,
            0,
            np.where(trend >= trend_high, 2, 1),
        )
        * 4
        + (volatility >= vol_median).astype(int) * 2
        + (flow >= flow_median).astype(int)
    )
    valid = np.isfinite(trend) & np.isfinite(volatility) & np.isfinite(flow)
    return np.where(valid, state, -1)


def _rank7_duration_key(
    state: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    values = np.asarray(state, dtype=int)
    time_ns = timestamps.to_numpy(
        dtype="datetime64[ns]"
    ).astype(np.int64, copy=False)
    continuity_ns = pd.Timedelta("90min").value
    age = np.zeros(len(values), dtype=int)
    for idx, current in enumerate(values):
        if current < 0:
            continue
        continuous = (
            idx == 0
            or time_ns[idx] - time_ns[idx - 1] <= continuity_ns
        )
        if (
            idx > 0
            and continuous
            and current == values[idx - 1]
            and age[idx - 1] > 0
        ):
            age[idx] = age[idx - 1] + 1
        else:
            age[idx] = 1
    return np.where(
        values >= 0,
        values * 5 + np.digitize(age, (1, 6, 24, 72), right=True),
        -1,
    )


def _rank7_state_bank_from_hourly(
    hourly: pd.DataFrame,
    hourly_features: pd.DataFrame,
    dates: pd.Series,
) -> dict[str, np.ndarray]:
    if hourly.empty or hourly_features.empty:
        missing = np.full(len(dates), -1, dtype=int)
        return {
            "kalman": missing.copy(),
            "bocpd": missing.copy(),
            "semimarkov": missing.copy(),
        }
    hourly = hourly.sort_index()
    hourly_features = hourly_features.sort_index().reindex(hourly.index)
    fit = np.asarray(
        (hourly_features.index >= "2020-07-01")
        & (hourly_features.index < "2023-01-01"),
        dtype=bool,
    )
    if not fit.any():
        raise ValueError(
            "completed hourly history does not cover the frozen Rank7 fit window"
        )

    log_price = np.log(hourly["close"].to_numpy(float))
    train_var = float(np.nanvar(np.diff(log_price)[fit[1:]]))
    filtered = _rank7_kalman_filter(
        log_price, 0.1, 0.001, 0.5, train_var
    )
    kalman_frame = pd.DataFrame(
        {
            "date": hourly.index.to_numpy(),
            "slope_z": filtered[:, 3],
            "innovation_z": filtered[:, 2],
        }
    )
    fit_frame = kalman_frame.loc[fit]
    slope_low = float(fit_frame["slope_z"].quantile(0.25))
    slope_high = float(fit_frame["slope_z"].quantile(0.75))
    innovation_low = float(fit_frame["innovation_z"].quantile(0.25))
    innovation_high = float(fit_frame["innovation_z"].quantile(0.75))
    kalman_frame["state"] = (
        np.where(
            kalman_frame["slope_z"] <= slope_low,
            0,
            np.where(kalman_frame["slope_z"] >= slope_high, 2, 1),
        )
        * 3
        + np.where(
            kalman_frame["innovation_z"] <= innovation_low,
            0,
            np.where(
                kalman_frame["innovation_z"] >= innovation_high, 2, 1
            ),
        )
    )
    kalman = _rank7_map_hourly(dates, kalman_frame, "state")

    good = hourly_features[["ret1", "flow24"]].notna().all(axis=1).to_numpy()
    fit_good = good & fit
    raw_train = hourly_features.loc[
        fit_good, ["ret1", "flow24"]
    ].to_numpy(float)
    mean = raw_train.mean(axis=0)
    std = raw_train.std(axis=0)
    std[std < 1e-8] = 1.0
    index = pd.DatetimeIndex(hourly_features.index[good])
    standardized = np.clip(
        (
            hourly_features.loc[good, ["ret1", "flow24"]].to_numpy(float)
            - mean
        )
        / std,
        -12,
        12,
    )
    posterior = _rank7_bocpd(standardized)
    bocpd_output = pd.DataFrame(
        {
            "date": index.to_numpy(),
            "primary": posterior["posterior_mean"][:, 0],
            "short_mass": posterior["short_mass"],
            "secondary": posterior["posterior_mean"][:, 1],
        }
    )
    fit_bocpd = bocpd_output[
        (bocpd_output["date"] >= "2020-07-01")
        & (bocpd_output["date"] < "2023-01-01")
    ]
    mapped = pd.merge_asof(
        pd.DataFrame(
            {"date": pd.to_datetime(dates), "pos": np.arange(len(dates))}
        ),
        bocpd_output.sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta("2h"),
    ).sort_values("pos")
    primary = mapped["primary"].to_numpy(float)
    short_mass = mapped["short_mass"].to_numpy(float)
    secondary = mapped["secondary"].to_numpy(float)
    bocpd = (
        np.where(
            primary <= float(fit_bocpd["primary"].quantile(0.33)),
            0,
            np.where(
                primary >= float(fit_bocpd["primary"].quantile(0.67)),
                2,
                1,
            ),
        )
        * 4
        + (
            short_mass
            >= float(fit_bocpd["short_mass"].quantile(0.50))
        ).astype(int)
        * 2
        + (
            secondary >= float(fit_bocpd["secondary"].quantile(0.50))
        ).astype(int)
    )
    bocpd = np.where(
        np.isfinite(primary)
        & np.isfinite(short_mass)
        & np.isfinite(secondary),
        bocpd,
        -1,
    )

    semi_state = _rank7_observable_state(hourly_features, fit)
    semi_key = _rank7_duration_key(semi_state, hourly_features.index)
    semimarkov = _rank7_map_hourly(
        dates,
        pd.DataFrame(
            {"date": hourly_features.index.to_numpy(), "key": semi_key}
        ),
        "key",
    )
    return {"kalman": kalman, "bocpd": bocpd, "semimarkov": semimarkov}


def _rank7_state_bank(
    market: pd.DataFrame, dates: pd.Series
) -> dict[str, np.ndarray]:
    hourly, features = _rank7_completed_hourly_features(market)
    return _rank7_state_bank_from_hourly(hourly, features, dates)


def _rank7_state_feature_matrix(
    bank: Mapping[str, np.ndarray],
    funding: np.ndarray,
    premium: np.ndarray,
) -> np.ndarray:
    kalman = bank["kalman"]
    bocpd = bank["bocpd"]
    semimarkov = bank["semimarkov"]
    semi_state = semimarkov // 5
    return np.column_stack(
        [
            kalman // 3,
            kalman % 3,
            bocpd // 4,
            (bocpd % 4) // 2,
            bocpd % 2,
            semi_state // 4,
            (semi_state % 4) // 2,
            semi_state % 2,
            semimarkov % 5,
            np.asarray(funding, dtype=np.int8),
            np.asarray(premium, dtype=np.int8),
        ]
    ).astype(float)


def _normalise_rank7_market(market: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "open_interest",
        "open_interest_available",
        "funding_available",
        "premium_available",
        "spot_close",
        "spot_rows",
        "premium_index_1m_close",
        "premium_rows",
    }
    missing = sorted(required - set(market.columns))
    if missing:
        raise Rank7FeatureError(
            f"Rank7 market frame missing columns: {missing}"
        )
    out = market.copy()
    out["date"] = pd.to_datetime(
        out["date"], utc=True, errors="raise"
    ).dt.tz_convert(None)
    out = (
        out.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    if out.empty:
        raise Rank7FeatureError("Rank7 market frame is empty")
    intervals = out["date"].diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise Rank7FeatureError(
            "Rank7 market frame is not a complete 5-minute grid"
        )
    for column in required - {"date"}:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ("spot_rows", "premium_rows"):
        counts = (
            pd.to_numeric(out[column], errors="coerce")
            .tail(3_000)
            .to_numpy(float)
        )
        if not np.isfinite(counts).all() or not np.equal(counts, 5.0).all():
            raise Rank7FeatureError(
                f"recent Rank7 {column} values must equal 5"
            )
    latest = out.iloc[-1]
    for column in (
        "open_interest_available",
        "funding_available",
        "premium_available",
    ):
        value = float(latest[column])
        if not np.isfinite(value) or value <= 0.5:
            raise Rank7FeatureError(f"latest {column} must be available")
    if (
        not np.isfinite(float(latest["open_interest"]))
        or float(latest["open_interest"]) <= 0.0
    ):
        raise Rank7FeatureError("latest open_interest must be positive")
    return out


def rank7_rebuild_feature_context(
    market,
    *,
    medians,
    clip,
    delay_bars,
    hourly_history,
) -> dict:
    """Source-equivalent Rank7 causal feature-context reconstruction."""

    market = _normalise_rank7_market(market)
    dates = pd.to_datetime(market["date"])
    base_features = build_market_feature_frame(market, window_size=144)
    raw_features = pd.concat(
        [base_features, build_interest_features(market, base_features)], axis=1
    )
    raw_features = raw_features.loc[
        :, ~raw_features.columns.duplicated(keep="last")
    ]
    features = _rank7_live_decision_features(raw_features)
    decisions = _rank7_decision_mask(
        dates, "live_hour_signal_bar", window_size=144
    )
    funding_leg = decisions & _rank7_component_mask(
        features, "funding10_trend70"
    )
    premium_leg = decisions & _rank7_component_mask(
        features, "premium20_mom90"
    )
    base = funding_leg | premium_leg

    if hourly_history is None:
        bank = _rank7_state_bank(market, dates)
    else:
        current, _ = _rank7_completed_hourly_features(market)
        history = hourly_history.copy()
        history["date"] = pd.to_datetime(
            history["date"], utc=True, errors="raise"
        ).dt.tz_convert(None)
        history = history.set_index("date").sort_index()
        if current.empty:
            combined = history
        else:
            overlap = history.index.intersection(current.index)
            if len(overlap):
                columns = ["open", "high", "low", "close", "quote", "buy"]
                left = history.loc[overlap, columns].to_numpy(float)
                right = current.loc[overlap, columns].to_numpy(float)
                if not np.allclose(
                    left,
                    right,
                    rtol=1e-9,
                    atol=1e-7,
                    equal_nan=False,
                ):
                    raise Rank7FeatureError(
                        "live/hourly warm-start overlap mismatch"
                    )
            first = current.index.min()
            if (
                len(history)
                and first > history.index.max() + pd.Timedelta("1h")
            ):
                raise Rank7FeatureError(
                    "live market tail does not overlap Rank7 hourly warm start"
                )
            combined = pd.concat(
                [history.loc[history.index < first], current]
            ).sort_index()
        combined = combined.loc[combined.index <= market["date"].max()].copy()
        intervals = combined.index.to_series().diff().dropna()
        if len(intervals) and not intervals.eq(pd.Timedelta("1h")).all():
            raise Rank7FeatureError(
                "combined Rank7 hourly state grid is incomplete"
            )
        bank = _rank7_state_bank_from_hourly(
            combined, _rank7_hourly_state_features(combined), dates
        )
    valid_state = (
        (bank["kalman"] >= 0)
        & (bank["bocpd"] >= 0)
        & (bank["semimarkov"] >= 0)
    )
    base &= valid_state

    barrier_bank = _rank7_build_barrier_bank(market)
    long_signal, short_signal, info = _rank7_coalesced_barrier_signals(
        market,
        barrier_bank,
        min_coalescence=3,
        touch_width=0.001,
        branch="depleted_continuation",
    )
    nested_side = causal_shift(
        long_signal.astype(np.int8) - short_signal.astype(np.int8)
    )
    nested24, nested_age = recent_side(nested_side, 288)
    nested48, _ = recent_side(nested_side, 576)

    braid_state = _rank7_build_braid_state(market)
    braid_events = _rank7_market_braid_events(
        braid_state,
        shock_z=2.0,
        passage_z=0.5,
        max_age=144,
        topology_mode="relative_order",
    )
    braid_side = causal_shift(
        braid_events.signal_side.to_numpy(np.int8)
    )
    braid24, braid_age = recent_side(braid_side, 288)
    braid48, _ = recent_side(braid_side, 576)

    state = _rank7_state_feature_matrix(bank, funding_leg, premium_leg)
    raw = np.column_stack(
        [
            state,
            *[
                pd.to_numeric(
                    features[column], errors="coerce"
                ).to_numpy(float)
                for column in RANK7_PA_COLUMNS + RANK7_OTHER_COLUMNS
            ],
        ]
    )
    weak = np.column_stack(
        [
            causal_shift(info["high_work_ratio"], np.nan),
            causal_shift(info["low_work_ratio"], np.nan),
            causal_shift(info["high_coalescence"]),
            causal_shift(info["low_coalescence"]),
            nested24,
            nested48,
            np.minimum(nested_age, 576),
            braid24,
            braid48,
            np.minimum(braid_age, 576),
        ]
    )
    unfilled = np.column_stack([raw, weak])
    median_values = np.asarray(medians, dtype=float)
    if (
        median_values.shape != (len(RANK7_FEATURE_COLUMNS),)
        or not np.isfinite(median_values).all()
    ):
        raise Rank7FeatureError("Rank7 median vector is invalid")
    lower, upper = map(float, clip)
    if not (
        np.isfinite(lower)
        and np.isfinite(upper)
        and lower < upper
    ):
        raise Rank7FeatureError("Rank7 clip contract is invalid")
    filled = np.clip(
        np.where(np.isfinite(unfilled), unfilled, median_values),
        lower,
        upper,
    )
    matrix = apply_rank7_delay(filled, bars=int(delay_bars))
    anchors = immutable_anchors(base, 144)
    return {
        "market": market,
        "dates": dates,
        "features": features,
        "matrix": matrix,
        "unfilled_matrix": unfilled,
        "base": base,
        "anchors": anchors,
        "funding_leg": funding_leg,
        "premium_leg": premium_leg,
        "nested_side": nested_side,
        "braid_side": braid_side,
        "feature_columns": RANK7_FEATURE_COLUMNS,
    }


__all__ = (
    "LearnerSpec",
    "Rank7FeatureError",
    "RANK7_FEATURE_COLUMNS",
    "RANK7_SEEDS",
    "RANK7_TREES",
    "SelectionSpec",
    "StructuralTrade",
    "annual_rank7_masks",
    "annual_rank7_windows",
    "apply_rank7_delay",
    "attach_open_interest",
    "balanced_rank7_weights",
    "build_bidirectional_features",
    "build_interest_features",
    "build_kimchi_features",
    "build_light_rex_features",
    "build_market_feature_frame",
    "completed_hourly_markov_features",
    "deterministic_extra_trees_predict",
    "fit_annual_rank7_folds",
    "fresh_masks",
    "gate_mask",
    "immutable_anchors",
    "load_market",
    "markov_active",
    "markov_transition_keys",
    "normalise_market",
    "rank7_activation",
    "rank7_rebuild_feature_context",
    "rank7_source_thresholds",
    "rex_taker_gate_match",
    "rex_veto_gate_match",
    "structural_trade_at",
    "walk_structural_schedule",
)
