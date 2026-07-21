"""Build the frozen live-portfolio comparator clocks without outcomes.

This module reproduces only the three executable sleeve entry policies.  It
does not import portfolio optimizers, legacy searches, prediction artifacts,
backtests, order code, or post-entry prices.  Real-source execution remains
disabled until a separate hash-bound export preregistration is written.
"""

# pandas-stubs still widens DataFrame column selection to scalar/Index unions;
# runtime schema checks and synthetic parity tests provide the narrow contract.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportOperatorIssue=false, reportReturnType=false

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from training import cchr_comparator_clock_common as clock_common


FAMILY = "live"
PRE_2024_END = pd.Timestamp("2024-01-01T00:00:00Z")
OI_BASE_WINDOW = 144
OI_RETURN_Z_WINDOW = 48
FUNDING_TREND_PERIODS = 95
DAILY_MIN_SOURCE_ROWS = 24 * 60 * 4

MARKET_PATH = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz")
MARKET_SHA256 = "dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192"
MARKET_COLUMNS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "taker_buy_base",
    "usdkrw",
    "usdkrw_available",
    "open_interest",
    "open_interest_available",
)

FUNDING_PATH = Path(
    "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
)
FUNDING_SHA256 = "4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7"
FUNDING_COLUMNS = ("date", "funding_time", "funding_rate")

PREMIUM_PATH = Path(
    "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
)
PREMIUM_SHA256 = "b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7"
PREMIUM_COLUMNS = ("date", "close", "close_time")

UPBIT_PATH = Path(
    "/home/pakchu/workspace/wave_trading/data/"
    "2020-01-01_2025-12-15_4bd081fc54811fccdee66850692c435e.csv.gz"
)
UPBIT_SHA256 = "7c377c402b4c1c3db3dafb5e15cd06e93f6e9c2c08d154ed88dd47e91f86eb35"
UPBIT_COLUMNS = ("date", "close", "volume")

PORTFOLIO_CONFIG_PATH = Path(
    "configs/live/portfolio_gross385_trainmdd40_2026-07-12.json"
)
PORTFOLIO_CONFIG_SHA256 = (
    "86f255ca3967245b8b0676b00025b955d7f33668ab1ef9d813623191b4ecd1e7"
)
OI_CONFIG_PATH = Path("configs/live/oi_upbit_ratio288_low_candidate.json")
OI_CONFIG_SHA256 = "659239373e1f51fc2df9615f5387686fd9252a56e1c366b45421bf39d3d6223f"
FUNDING_CONFIG_PATH = Path(
    "configs/live/new_long_minimal_funding_premium_candidate.json"
)
FUNDING_CONFIG_SHA256 = (
    "f0848c5fea1fcc7823ed15b6e4b865a8dc2731c2d2bfd2ba21b0f92c534f0f03"
)
REX_CONFIG_PATH = Path("configs/live/rex_veto_7_candidate.json")
REX_CONFIG_SHA256 = "36df47c4737eb99f4ca5e2b257d9bd2fbf130df9d731b9ac02fcfe5192acd4db"

COMPONENT_SPECS: dict[str, dict[str, Any]] = {
    "oi_upbit_ratio288_low": {
        "side": 1,
        "weight": 0.65,
        "hold_bars": 30,
        "stride_bars": 6,
        "stride_offset_bars": 5,
        "gates": (
            ("oi_minus_px_4h_z", ">=", 0.18084217361514066),
            ("return_zscore_48", "<=", -0.3382356464105935),
            ("range_vol", ">=", 0.03684324822682795),
            ("rsi_norm", "<=", 0.006924814138880962),
            ("oi_ret_4h_z", ">=", -0.675926157602224),
            ("sma24_ratio", "<=", -0.0005569113930713103),
            ("vg_upbit_binance_vol_ratio_z_288", "<=", -0.6044603870294258),
        ),
    },
    "new_long_minimal_funding_premium": {
        "side": 1,
        "weight": 1.75,
        "hold_bars": 576,
        "stride_bars": 12,
        "stride_offset_bars": 11,
        "gate_clauses": (
            (
                ("funding_rate", "<=", -0.0000167),
                ("trend_96", ">=", 0.007485218212390219),
            ),
            (
                ("premium_index_change", "<=", -0.00023471),
                ("htf_1d_return_4", ">=", 0.0940403008961932),
            ),
        ),
    },
    "cand_rex_veto_7": {
        "weight": 1.45,
        "hold_bars": 144,
        "stride_bars": 24,
        "stride_offset_bars": 11,
        "strength_threshold": 0.177552250256958,
        "gates": (
            ("htf_1w_return_4", ">=", -0.26588062806734514),
            ("oi_zscore", "<=", 1.5910475818293068),
        ),
    },
}

AVAILABILITY_FLAGS = {
    "funding_rate": ("funding_available",),
    "premium_index_change": ("premium_available",),
    "oi_minus_px_4h_z": ("open_interest_available",),
    "oi_ret_4h_z": ("open_interest_available",),
    "oi_zscore": ("open_interest_available",),
    "vg_upbit_binance_vol_ratio_z_288": (
        "upbit_volume_available",
        "usdkrw_available",
    ),
}


def candidate_map() -> dict[str, dict[str, Any]]:
    members: dict[str, dict[str, Any]] = {}
    for component, spec in COMPONENT_SPECS.items():
        members[f"live:{component}"] = {
            "family": FAMILY,
            "parameters": {"component": component},
            "hold_bars": int(spec["hold_bars"]),
            "component_weight": float(spec["weight"]),
        }
    return dict(sorted(members.items()))


def input_bindings() -> dict[str, dict[str, Any]]:
    return {
        "market": {
            "path": str(MARKET_PATH),
            "sha256": MARKET_SHA256,
            "columns": list(MARKET_COLUMNS),
        },
        "funding": {
            "path": str(FUNDING_PATH),
            "sha256": FUNDING_SHA256,
            "columns": list(FUNDING_COLUMNS),
        },
        "premium": {
            "path": str(PREMIUM_PATH),
            "sha256": PREMIUM_SHA256,
            "columns": list(PREMIUM_COLUMNS),
        },
        "upbit": {
            "path": str(UPBIT_PATH),
            "sha256": UPBIT_SHA256,
            "columns": list(UPBIT_COLUMNS),
        },
    }


def load_causal_inputs() -> dict[str, pd.DataFrame]:
    """Load hash-bound causal columns only; callers must preregister first."""

    return {
        "market": clock_common.read_hash_bound_prefix(
            MARKET_PATH,
            expected_sha256=MARKET_SHA256,
            columns=MARKET_COLUMNS,
            date_column="date",
            end_exclusive=PRE_2024_END,
        ),
        "funding": clock_common.read_hash_bound_prefix(
            FUNDING_PATH,
            expected_sha256=FUNDING_SHA256,
            columns=FUNDING_COLUMNS,
            date_column="date",
            end_exclusive=PRE_2024_END,
        ),
        "premium": clock_common.read_hash_bound_prefix(
            PREMIUM_PATH,
            expected_sha256=PREMIUM_SHA256,
            columns=PREMIUM_COLUMNS,
            date_column="date",
            end_exclusive=PRE_2024_END,
        ),
        "upbit": clock_common.read_hash_bound_prefix(
            UPBIT_PATH,
            expected_sha256=UPBIT_SHA256,
            columns=UPBIT_COLUMNS,
            date_column="date",
            end_exclusive=PRE_2024_END,
        ),
    }


def _utc_series(values: pd.Series, *, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="raise")
    if bool(parsed.isna().to_numpy(dtype=bool).any()):
        raise ValueError(f"{label} contains an invalid timestamp")
    return cast(pd.Series, parsed)


def _causal_panel(frame: pd.DataFrame, *, date_column: str, label: str) -> pd.DataFrame:
    out = frame.copy()
    out[date_column] = _utc_series(out[date_column], label=label)
    out = out.loc[out[date_column] < PRE_2024_END].sort_values(date_column)
    if bool(out[date_column].duplicated().to_numpy(dtype=bool).any()):
        raise ValueError(f"{label} contains duplicate timestamps")
    return out.reset_index(drop=True)


def _assert_complete_market_grid(panel: pd.DataFrame) -> None:
    dates = _utc_series(panel["date"], label="market grid")
    intervals = dates.diff().dropna()
    if len(intervals) and not bool(intervals.eq(pd.Timedelta("5min")).all()):
        raise ValueError("market rows must form a complete five-minute grid")


def _premium_availability(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if bool(numeric.notna().to_numpy(dtype=bool).any()):
        return cast(
            pd.Series, pd.to_datetime(numeric, unit="ms", utc=True, errors="raise")
        )
    return _utc_series(values, label="premium close_time")


def attach_live_auxiliary(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    premium: pd.DataFrame,
) -> pd.DataFrame:
    """Backward-as-of join live funding (12h) and completed premium (10m)."""

    base = _causal_panel(market, date_column="date", label="market")
    _assert_complete_market_grid(base)
    fund = _causal_panel(funding, date_column="date", label="funding")
    fund["funding_rate"] = pd.to_numeric(fund["funding_rate"], errors="coerce")
    fund = fund.dropna(subset=["funding_rate"])
    fund = fund.loc[:, ["date", "funding_rate"]].rename(
        columns={"date": "funding_source_time"}
    )
    joined = pd.merge_asof(
        base.sort_values("date"),
        fund.sort_values("funding_source_time"),
        left_on="date",
        right_on="funding_source_time",
        direction="backward",
        tolerance=pd.Timedelta("12h"),
    )
    joined["funding_available"] = joined["funding_source_time"].notna().astype(float)

    prem = premium.copy()
    prem["premium_source_time"] = _premium_availability(prem["close_time"])
    prem = prem.loc[prem["premium_source_time"] < PRE_2024_END].copy()
    prem["premium_index"] = pd.to_numeric(prem["close"], errors="coerce")
    prem = (
        prem.dropna(subset=["premium_source_time", "premium_index"])
        .sort_values("premium_source_time")
        .drop_duplicates("premium_source_time", keep="last")
    )
    joined = pd.merge_asof(
        joined.sort_values("date"),
        prem.loc[:, ["premium_source_time", "premium_index"]],
        left_on="date",
        right_on="premium_source_time",
        direction="backward",
        tolerance=pd.Timedelta("10min"),
    )
    joined["premium_available"] = joined["premium_source_time"].notna().astype(float)
    premium_index = pd.to_numeric(joined["premium_index"], errors="coerce")
    joined["premium_index_change"] = (
        premium_index.diff(96).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    return joined.drop(columns=["funding_source_time", "premium_source_time"])


def _rolling_z(
    values: pd.Series,
    window: int,
    *,
    minimum: int,
    clip: float,
    fill: bool,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    mean = numeric.rolling(window, min_periods=minimum).mean()
    std = numeric.rolling(window, min_periods=minimum).std(ddof=0)
    out = ((numeric - mean) / std.replace(0.0, np.nan)).clip(-clip, clip)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0) if fill else out


def _clean(values: pd.Series, *, clip: float | None = None) -> pd.Series:
    out = values.replace([np.inf, -np.inf], np.nan)
    if clip is not None:
        out = out.clip(-clip, clip)
    return out.fillna(0.0)


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    relative_strength = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + relative_strength)


def _completed_daily_return_4(market: pd.DataFrame) -> pd.Series:
    if len(market) < DAILY_MIN_SOURCE_ROWS:
        return pd.Series(0.0, index=market.index)
    source = market.loc[:, ["date", "open", "high", "low", "close"]].copy()
    source = source.set_index("date")
    daily = pd.DataFrame(
        {
            "open": source["open"]
            .resample("1D", label="right", closed="right")
            .first(),
            "high": source["high"].resample("1D", label="right", closed="right").max(),
            "low": source["low"].resample("1D", label="right", closed="right").min(),
            "close": source["close"]
            .resample("1D", label="right", closed="right")
            .last(),
        }
    ).dropna()
    previous = daily.shift(1)
    features = pd.DataFrame(
        {
            "date": daily.index,
            "htf_1d_return_4": _clean(
                previous["close"] / previous["close"].shift(4).replace(0.0, np.nan)
                - 1.0
            ).to_numpy(float),
        }
    )
    target = market.loc[:, ["date"]].copy()
    target["_row"] = np.arange(len(target))
    aligned = pd.merge_asof(
        target.sort_values("date"),
        features.sort_values("date"),
        on="date",
        direction="backward",
    ).sort_values("_row")
    return pd.Series(
        aligned["htf_1d_return_4"].fillna(0.0).to_numpy(float), index=market.index
    )


def build_live_features(market: pd.DataFrame) -> pd.DataFrame:
    """Reproduce only the causal columns used by the three frozen configs."""

    panel = _causal_panel(market, date_column="date", label="enriched market")
    close = pd.to_numeric(panel["close"], errors="coerce").astype(float)
    high = pd.to_numeric(panel["high"], errors="coerce").astype(float)
    low = pd.to_numeric(panel["low"], errors="coerce").astype(float)
    volume = pd.to_numeric(panel["volume"], errors="coerce").astype(float)
    out = pd.DataFrame(index=panel.index)

    rolling_high = high.rolling(OI_BASE_WINDOW, min_periods=OI_BASE_WINDOW).max()
    rolling_low = low.rolling(OI_BASE_WINDOW, min_periods=OI_BASE_WINDOW).min()
    midpoint = (rolling_high + rolling_low) / 2.0
    out["range_vol"] = _clean(
        (rolling_high - rolling_low) / midpoint.replace(0.0, np.nan)
    )
    out["trend_24"] = _clean(close / close.shift(23).replace(0.0, np.nan) - 1.0)
    out["trend_96"] = _clean(
        close / close.shift(FUNDING_TREND_PERIODS).replace(0.0, np.nan) - 1.0
    )
    mean24 = close.rolling(24, min_periods=1).mean()
    out["sma24_ratio"] = _clean((close - mean24) / mean24.replace(0.0, np.nan))
    out["rsi_norm"] = _clean((_rsi(close) - 50.0) / 50.0)
    log_return = np.log(close.where(close > 0.0) / close.where(close > 0.0).shift(1))
    out["return_zscore_48"] = _rolling_z(
        log_return,
        OI_RETURN_Z_WINDOW,
        minimum=max(2, OI_RETURN_Z_WINDOW // 3),
        clip=5.0,
        fill=True,
    )
    out["volume_zscore"] = _rolling_z(volume, 48, minimum=16, clip=5.0, fill=True)
    taker_ratio = pd.to_numeric(
        panel["taker_buy_base"], errors="coerce"
    ) / volume.replace(0.0, np.nan)
    out["taker_imbalance"] = _clean(taker_ratio.fillna(0.5) * 2.0 - 1.0)

    open_interest_available = pd.to_numeric(
        panel["open_interest_available"], errors="coerce"
    ).fillna(0.0)
    out["open_interest_available"] = open_interest_available.to_numpy(float)
    open_interest = pd.to_numeric(panel["open_interest"], errors="coerce").where(
        open_interest_available > 0.5
    )
    out["oi_zscore"] = _rolling_z(open_interest, 48, minimum=16, clip=5.0, fill=True)
    oi_series = open_interest.replace(0.0, np.nan)
    price_series = close.where(close > 0.0)
    oi_return = np.log(oi_series / oi_series.shift(48)).replace(
        [np.inf, -np.inf], np.nan
    )
    price_return = np.log(price_series / price_series.shift(48)).replace(
        [np.inf, -np.inf], np.nan
    )
    out["oi_ret_4h_z"] = _rolling_z(oi_return, 288, minimum=50, clip=5.0, fill=False)
    out["oi_minus_px_4h_z"] = _rolling_z(
        oi_return - price_return, 288, minimum=50, clip=5.0, fill=False
    )

    out["funding_rate"] = _clean(
        pd.to_numeric(panel["funding_rate"], errors="coerce"), clip=1.0
    )
    out["funding_available"] = pd.to_numeric(
        panel["funding_available"], errors="coerce"
    ).fillna(0.0)
    out["premium_index_change"] = _clean(
        pd.to_numeric(panel["premium_index_change"], errors="coerce"), clip=5.0
    )
    out["premium_available"] = pd.to_numeric(
        panel["premium_available"], errors="coerce"
    ).fillna(0.0)
    out["htf_1d_return_4"] = _completed_daily_return_4(panel)
    out["usdkrw_available"] = pd.to_numeric(
        panel["usdkrw_available"], errors="coerce"
    ).fillna(0.0)

    for window in (144, 576, 2016, 8640):
        minimum = min(window, max(12, window // 4))
        highest = high.rolling(window, min_periods=minimum).max()
        lowest = low.rolling(window, min_periods=minimum).min()
        span = (highest - lowest).replace(0.0, np.nan)
        out[f"rex_{window}_range_pos"] = _clean(((close - lowest) / span) * 2.0 - 1.0)
        out[f"rex_{window}_max_to_cur_pct"] = _clean(
            highest / close.replace(0.0, np.nan) - 1.0
        )
        out[f"rex_{window}_cur_to_min_pct"] = _clean(
            close / lowest.replace(0.0, np.nan) - 1.0
        )
    for name, periods in {
        "htf_4h_return_1": 48,
        "htf_1d_return_4_rex": 1152,
        "htf_3d_return_4": 3456,
        "htf_1w_return_4": 8064,
    }.items():
        out[name] = _clean(close / close.shift(periods).replace(0.0, np.nan) - 1.0)
    return out.replace([np.inf, -np.inf], np.nan)


def attach_upbit_volume_feature(
    market: pd.DataFrame,
    features: pd.DataFrame,
    upbit: pd.DataFrame,
) -> pd.DataFrame:
    panel = _causal_panel(market, date_column="date", label="market")
    local = _causal_panel(upbit, date_column="date", label="upbit")
    local["upbit_close"] = pd.to_numeric(local["close"], errors="coerce")
    local["upbit_volume"] = pd.to_numeric(local["volume"], errors="coerce")
    joined = pd.merge_asof(
        panel.loc[:, ["date", "quote_asset_volume", "usdkrw"]].sort_values("date"),
        local.loc[:, ["date", "upbit_close", "upbit_volume"]].sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta("7min"),
    ).sort_index()
    upbit_value = joined["upbit_volume"] * joined["upbit_close"]
    binance_value = pd.to_numeric(
        joined["quote_asset_volume"], errors="coerce"
    ) * pd.to_numeric(joined["usdkrw"], errors="coerce")
    available = upbit_value.notna() & binance_value.notna() & binance_value.ne(0.0)
    ratio = (upbit_value / binance_value.replace(0.0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )
    output = features.copy()
    output["vg_upbit_binance_vol_ratio_z_288"] = _rolling_z(
        ratio.fillna(0.0), 288, minimum=72, clip=8.0, fill=True
    ).to_numpy(float)
    output["upbit_volume_available"] = available.to_numpy(float)
    return output


def rex_strength_direction(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    def values(name: str) -> np.ndarray:
        if name not in features:
            return np.zeros(len(features), dtype=float)
        return (
            pd.to_numeric(features[name], errors="coerce").fillna(0.0).to_numpy(float)
        )

    windows = (144, 576, 2016, 8640)
    location = np.nanmean(
        np.vstack([values(f"rex_{window}_range_pos") for window in windows]), axis=0
    )
    maximum_gap = np.nanmean(
        np.vstack([values(f"rex_{window}_max_to_cur_pct") for window in windows]),
        axis=0,
    )
    minimum_gap = np.nanmean(
        np.vstack([values(f"rex_{window}_cur_to_min_pct") for window in windows]),
        axis=0,
    )
    local_trend = values("trend_24") + 0.5 * values("htf_4h_return_1")
    higher_trend = (
        values("htf_1d_return_4_rex")
        + values("htf_3d_return_4")
        + values("htf_1w_return_4")
    )
    volume_confirmation = np.maximum(0.0, values("volume_zscore")) + 0.5 * np.abs(
        values("taker_imbalance")
    )
    direction = np.sign(higher_trend)
    pullback_alignment = -np.sign(location) * direction
    pullback = np.maximum(0.0, pullback_alignment) * (
        np.abs(higher_trend) + 0.25 * np.abs(maximum_gap - minimum_gap)
    )
    reclaim = np.maximum(0.0, np.sign(local_trend) * direction)
    strength = pullback * (0.5 + reclaim) * (1.0 + 0.25 * volume_confirmation)
    return np.nan_to_num(strength), np.nan_to_num(direction)


def interval_slots(dates: pd.Series, *, stride: int, offset: int) -> np.ndarray:
    if stride < 1:
        raise ValueError("stride must be positive")
    timestamps = _utc_series(dates, label="stride date")
    minute_number = (
        timestamps.astype("int64").to_numpy(dtype=np.int64) // 60_000_000_000
    )
    return ((minute_number // 5) % stride) == (offset % stride)


def _gate_mask(
    features: pd.DataFrame,
    gates: Sequence[tuple[str, str, float]],
) -> np.ndarray:
    active = np.ones(len(features), dtype=bool)
    for feature, operation, threshold in gates:
        for flag in AVAILABILITY_FLAGS.get(feature, ()):
            values = pd.to_numeric(features.get(flag, np.nan), errors="coerce")
            if not isinstance(values, pd.Series):
                values = pd.Series(values, index=features.index)
            array = values.to_numpy(float)
            active &= np.isfinite(array) & (array > 0.5)
        if feature not in features:
            active &= False
            continue
        values = pd.to_numeric(features[feature], errors="coerce").to_numpy(float)
        if operation == ">=":
            passed = values >= threshold
        elif operation == "<=":
            passed = values <= threshold
        else:
            raise ValueError(f"unsupported gate operation: {operation}")
        active &= np.isfinite(values) & passed
    return active


def component_signals(
    dates: pd.Series,
    features: pd.DataFrame,
) -> dict[str, np.ndarray]:
    if len(dates) != len(features):
        raise ValueError("dates and features must align")
    output: dict[str, np.ndarray] = {}
    for component in ("oi_upbit_ratio288_low", "new_long_minimal_funding_premium"):
        spec = COMPONENT_SPECS[component]
        if "gate_clauses" in spec:
            clauses = [_gate_mask(features, clause) for clause in spec["gate_clauses"]]
            active = np.logical_or.reduce(clauses)
        else:
            active = _gate_mask(features, spec["gates"])
        active &= interval_slots(
            dates,
            stride=int(spec["stride_bars"]),
            offset=int(spec["stride_offset_bars"]),
        )
        output[component] = np.where(active, int(spec["side"]), 0).astype(np.int8)

    rex_spec = COMPONENT_SPECS["cand_rex_veto_7"]
    strength, direction = rex_strength_direction(features)
    rex_active = (
        (strength > float(rex_spec["strength_threshold"]))
        & (direction != 0.0)
        & _gate_mask(features, rex_spec["gates"])
        & interval_slots(
            dates,
            stride=int(rex_spec["stride_bars"]),
            offset=int(rex_spec["stride_offset_bars"]),
        )
    )
    output["cand_rex_veto_7"] = np.where(rex_active, np.sign(direction), 0).astype(
        np.int8
    )
    return output


def build_clock(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    premium: pd.DataFrame,
    upbit: pd.DataFrame,
) -> pd.DataFrame:
    panel = attach_live_auxiliary(market, funding, premium)
    _assert_complete_market_grid(panel)
    features = attach_upbit_volume_feature(panel, build_live_features(panel), upbit)
    signals = component_signals(panel["date"], features)
    candidates: list[clock_common.ClockCandidate] = []
    for component, signal in signals.items():
        spec = COMPONENT_SPECS[component]
        for position in np.flatnonzero(signal):
            exit_position = int(position) + 1 + int(spec["hold_bars"])
            if exit_position >= len(panel):
                continue
            signal_time = pd.Timestamp(
                panel.iloc[int(position)]["date"]
            ).to_pydatetime()
            entry_time = signal_time + timedelta(minutes=5)
            candidates.append(
                clock_common.ClockCandidate(
                    candidate_id=f"live:{component}",
                    causal_origins=(signal_time,),
                    signal_time=signal_time,
                    decision_time=entry_time,
                    entry_time=entry_time,
                    exit_time=entry_time
                    + timedelta(minutes=5 * int(spec["hold_bars"])),
                    side=int(signal[int(position)]),
                )
            )
    return clock_common.schedule_candidates(candidates)


def validate_config_hashes() -> dict[str, str]:
    bindings = {
        str(PORTFOLIO_CONFIG_PATH): PORTFOLIO_CONFIG_SHA256,
        str(OI_CONFIG_PATH): OI_CONFIG_SHA256,
        str(FUNDING_CONFIG_PATH): FUNDING_CONFIG_SHA256,
        str(REX_CONFIG_PATH): REX_CONFIG_SHA256,
    }
    for path, expected in bindings.items():
        if clock_common.sha256_file(path) != expected:
            raise ValueError(f"live comparator config hash drifted: {path}")
    return bindings


def frozen_contract() -> Mapping[str, Any]:
    return {
        "family": FAMILY,
        "input_bindings": input_bindings(),
        "config_hashes": {
            str(PORTFOLIO_CONFIG_PATH): PORTFOLIO_CONFIG_SHA256,
            str(OI_CONFIG_PATH): OI_CONFIG_SHA256,
            str(FUNDING_CONFIG_PATH): FUNDING_CONFIG_SHA256,
            str(REX_CONFIG_PATH): REX_CONFIG_SHA256,
        },
        "candidate_map": candidate_map(),
        "candidate_map_sha256": clock_common.candidate_map_hash(candidate_map()),
        "timing": {
            "signal_bar": "completed five-minute bar identified by open timestamp",
            "decision_delay_bars": 1,
            "entry_delay_bars": 1,
            "interval": "[entry_time,exit_time)",
            "scheduling": "split containment, then nonoverlap independently per component",
        },
        "outcomes_opened": False,
        "real_source_execution_authorized": False,
    }
