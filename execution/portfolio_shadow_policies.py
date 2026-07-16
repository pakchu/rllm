"""Causal feature adapters used only by forward-shadow portfolio sleeves.

These helpers intentionally stop at signal-time feature construction.  They do
not place orders, size positions, or simulate exits.  Keeping the adapters
separate from the live executor makes research-only policies explicit while
allowing exact parity tests against their frozen research implementations.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame


def build_fresh_kimchi_feature_frame(
    market: pd.DataFrame,
    base_features: pd.DataFrame,
) -> pd.DataFrame:
    """Reproduce the frozen Fresh-Kimchi signal-time custom features.

    The equations mirror ``search_bidirectional_state_alpha.extra`` and
    ``search_kimchi_leadlag_bidirectional_alpha.features``.  Availability is
    deliberately *not* inferred here; the policy config supplies explicit
    fail-closed source flags for each directional clause.
    """

    out = base_features.copy()
    close = pd.to_numeric(market["close"], errors="coerce").astype(float)
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce").astype(float)
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce").astype(float)
    imbalance = (2.0 * taker_buy / quote.replace(0.0, np.nan) - 1.0).clip(-1.0, 1.0)
    for window in (12, 24, 48, 96, 144):
        out[f"bd_ret_{window}"] = np.log(close / close.shift(window))
        out[f"bd_imb_{window}"] = imbalance.rolling(window, min_periods=window).mean()
    out["bd_flow_accel"] = out["bd_imb_12"] - out["bd_imb_48"]

    btc_log = np.log(close)
    kimchi = (
        pd.to_numeric(market["kimchi_premium"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )
    usdkrw_log = np.log(
        pd.to_numeric(market["usdkrw"], errors="coerce")
        .replace(0.0, np.nan)
        .ffill()
    )

    def rolling_z(values: pd.Series, window: int = 576) -> pd.Series:
        mean = values.rolling(window, min_periods=window).mean()
        std = values.rolling(window, min_periods=window).std().replace(0.0, np.nan)
        return (values - mean) / std

    for window in (12, 48, 144, 288):
        kimchi_delta = kimchi - kimchi.shift(window)
        btc_return = btc_log - btc_log.shift(window)
        fx_delta = usdkrw_log - usdkrw_log.shift(window)
        out[f"kl_kimchi_delta_{window}"] = kimchi_delta
        out[f"kl_btc_ret_{window}"] = btc_return
        out[f"kl_fx_delta_{window}"] = fx_delta
        out[f"kl_kimchi_btc_gap_{window}"] = rolling_z(kimchi_delta) - rolling_z(btc_return)
        out[f"kl_local_impulse_{window}"] = rolling_z(kimchi_delta) - rolling_z(fx_delta)
    out["kl_accel_48_144"] = out["kl_kimchi_delta_48"] - out["kl_kimchi_delta_144"] / 3.0
    return out.replace([np.inf, -np.inf], np.nan)


def build_markov_feature_frame(
    market: pd.DataFrame,
    base_features: pd.DataFrame,
    *,
    window_size: int = 144,
    zscore_window: int = 48,
    volume_window: int = 48,
) -> pd.DataFrame:
    """Rebuild the frozen Markov base-setup feature contract.

    The research policy called ``build_market_feature_frame(...,
    window_size=144)``.  In that builder ``trend_96`` intentionally uses
    ``window_size - 1`` bars, so the generic live 288-window frame is not
    interchangeable.  Source availability remains runtime metadata and is
    copied from the live frame for fail-closed gates.
    """

    out = build_market_feature_frame(
        market,
        window_size=int(window_size),
        zscore_window=int(zscore_window),
        volume_window=int(volume_window),
    )
    for column in base_features.columns:
        if str(column).endswith("_available"):
            out[column] = base_features[column].to_numpy(copy=False)
    return out.replace([np.inf, -np.inf], np.nan)


def observable_markov_transition_keys(
    market: pd.DataFrame,
    state_model: dict[str, Any],
) -> np.ndarray:
    """Map the frozen completed-hour Markov transition key to each 5m row.

    Resampling and backward-as-of semantics match the research implementation:
    only a completed hourly label at or before the 5m signal timestamp can be
    observed.  No smoothed state or future hour is used.
    """

    indexed = market.set_index(pd.to_datetime(market["date"])).sort_index()
    quote = pd.to_numeric(indexed["quote_asset_volume"], errors="coerce").astype(float)
    taker_buy = pd.to_numeric(indexed["taker_buy_quote"], errors="coerce").astype(float)
    hourly = pd.DataFrame(
        {
            "open": indexed["open"].resample("1h", closed="right", label="right").first(),
            "high": indexed["high"].resample("1h", closed="right", label="right").max(),
            "low": indexed["low"].resample("1h", closed="right", label="right").min(),
            "close": indexed["close"].resample("1h", closed="right", label="right").last(),
            "quote": quote.resample("1h", closed="right", label="right").sum(),
            "buy": taker_buy.resample("1h", closed="right", label="right").sum(),
        }
    ).dropna()
    returns = np.log(hourly["close"]).diff()
    flow = 2.0 * hourly["buy"] / hourly["quote"].replace(0.0, np.nan) - 1.0
    trend24 = np.log(hourly["close"] / hourly["close"].shift(24))
    vol24 = returns.rolling(24).std()
    flow24 = flow.rolling(24).mean()
    trend = np.where(
        trend24 <= float(state_model["trend_low"]),
        0,
        np.where(trend24 >= float(state_model["trend_high"]), 2, 1),
    )
    volatility = (vol24 >= float(state_model["vol_median"])).astype(int)
    flow_bucket = (flow24 >= float(state_model["flow_median"])).astype(int)
    state = trend * 4 + volatility * 2 + flow_bucket
    previous = pd.Series(state, index=hourly.index).shift(1).fillna(-1).astype(int)
    transitions = previous * 12 + state
    mapped = pd.merge_asof(
        pd.DataFrame(
            {
                "date": pd.to_datetime(market["date"]),
                "position": np.arange(len(market)),
            }
        ).sort_values("date"),
        pd.DataFrame(
            {"date": hourly.index.to_numpy(), "transition": transitions.to_numpy()}
        ).sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta("2h"),
    ).sort_values("position")
    return mapped["transition"].fillna(-1).to_numpy(dtype=int)
