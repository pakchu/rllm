"""Live executor for fixed-weight portfolio sleeve candidates.

This runner is intentionally narrower than a general portfolio engine.  It
supports the current gross<=6 BTCUSDT candidate family by:

* building one latest 5m feature row from the live DB, including open interest,
* evaluating fixed sleeve gates from ``configs/live/portfolio_gross6_*.json``,
* allocating margin by fixed sleeve weights against the research leverage
  budget (weight/leverage by default),
* placing hedge-mode LONG/SHORT maker orders through ``wave_trading``, and
* tracking per-sleeve exit timestamps in a local ledger.

It does not select new weights, change sleeve rules, or net active sleeves.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import select
import time
from dataclasses import dataclass, asdict, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from execution.rex_llm_live import RexLivePolicyConfig, RexLlmSelectorConfig, build_rex_live_policy_record
from execution.wave_execution import (
    WaveExecutionConfig,
    _StaticSignalGenerator,
    _load_api_credentials,
    load_wave_execution_classes,
)
from preprocessing.binance_aux_features import attach_binance_um_aux_frames
from preprocessing.external_features import attach_external_features, build_external_feature_frame
from preprocessing.live_db_features import (
    LiveDbFeatureConfig,
    _normalise_bar_frame,
    query_live_source_frames,
    resample_market_bars,
    sqlalchemy_engine_from_env,
)
from preprocessing.market_features import build_market_feature_frame
from training.evaluate_oi_llm_selector import _context_id, _tokens
from training.evaluate_portfolio_llm_selector import _base_context_tokens
from training.long_regime_interest_gate_validation import build_interest_features
from training.long_regime_score_gate_validation import _build_score_frame, _score_variant
from training.wave_feature_ridge_policy import build_wave_feature_frame


Side = Literal["LONG", "SHORT"]


SOURCE_FRAME_OVERLAP_MINUTES = 30
FEATURE_TAIL_CONTEXT_BARS = 8_640
FEATURE_TAIL_OUTPUT_BARS = 96
EXTERNAL_TAIL_CONTEXT_BARS = 288
ALT_POOL_SYMBOLS = ("ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "ADAUSDT")

LOG = logging.getLogger("portfolio_live")


def _frame_time_col(key: str) -> str:
    return "funding_time" if key == "funding" else "date"


def _frame_dedupe_cols(key: str, frame: pd.DataFrame) -> list[str]:
    time_col = _frame_time_col(key)
    if "tic" in frame.columns:
        return [time_col, "tic"]
    if "symbol" in frame.columns:
        return [time_col, "symbol"]
    return [time_col]


def _as_utc_ts(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert("UTC")
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.tz_localize("UTC")
    return pd.to_datetime(series, utc=True, errors="coerce")


def _to_naive_utc(series: pd.Series) -> pd.Series:
    ts = _as_utc_ts(series)
    return ts.dt.tz_convert(None)


@dataclass
class LiveSourceFrameCache:
    """In-process DB source-frame cache for the live portfolio loop.

    The first cycle still loads the full research lookback. Later cycles query a
    small overlap window, merge committed updates, and trim back to the required
    lookback. This keeps feature semantics unchanged while avoiding repeated
    45k-minute source reads every five minutes.
    """

    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    overlap_minutes: int = SOURCE_FRAME_OVERLAP_MINUTES
    last_query_mode: str = "cold"

    def _latest_source_ts(self) -> pd.Timestamp | None:
        latest: list[pd.Timestamp] = []
        for key, frame in self.frames.items():
            if key == "funding" or frame.empty:
                continue
            col = _frame_time_col(key)
            if col not in frame.columns:
                continue
            value = _as_utc_ts(frame[col]).max()
            if pd.notna(value):
                latest.append(pd.Timestamp(value))
        return min(latest) if latest else None

    def refresh(self, engine: Any, *, asof: pd.Timestamp, cfg: LiveDbFeatureConfig) -> dict[str, pd.DataFrame]:
        asof_ts = pd.Timestamp(asof)
        asof_ts = asof_ts.tz_localize("UTC") if asof_ts.tzinfo is None else asof_ts.tz_convert("UTC")
        lookback_start = asof_ts - pd.Timedelta(minutes=int(cfg.lookback_minutes))

        latest = self._latest_source_ts()
        if not self.frames or latest is None:
            query_cfg = cfg
            self.last_query_mode = "cold_full"
        else:
            query_start = max(lookback_start, latest - pd.Timedelta(minutes=max(1, int(self.overlap_minutes))))
            query_minutes = max(1, int(np.ceil((asof_ts - query_start).total_seconds() / 60.0)))
            query_cfg = LiveDbFeatureConfig(**{**asdict(cfg), "lookback_minutes": query_minutes})
            self.last_query_mode = f"incremental_{query_minutes}m"

        fresh = query_live_source_frames(engine, asof=asof_ts, cfg=query_cfg)
        self.frames = self._merge_and_trim(fresh, lookback_start=lookback_start, asof=asof_ts)
        return {key: value.copy() for key, value in self.frames.items()}

    def _merge_and_trim(
        self,
        fresh: dict[str, pd.DataFrame],
        *,
        lookback_start: pd.Timestamp,
        asof: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for key, new_frame in fresh.items():
            old = self.frames.get(key)
            if (old is None or old.empty) and (new_frame is None or new_frame.empty):
                out[key] = pd.DataFrame()
                continue
            time_col = _frame_time_col(key)

            old_keep = old
            if old_keep is not None and not old_keep.empty and time_col in old_keep.columns:
                old_ts = _as_utc_ts(old_keep[time_col])
                if key != "funding":
                    old_keep = old_keep.loc[(old_ts >= lookback_start) & (old_ts <= asof)]

            new_keep = new_frame
            if new_keep is not None and not new_keep.empty and time_col in new_keep.columns:
                new_keep = new_keep.copy()
                new_keep[time_col] = _to_naive_utc(new_keep[time_col])
                new_ts = _as_utc_ts(new_keep[time_col])
                if key != "funding":
                    new_keep = new_keep.loc[(new_ts >= lookback_start) & (new_ts <= asof)]
                    new_ts = new_ts.loc[new_keep.index]
                # Incremental refreshes intentionally overlap.  Drop the old
                # overlap before concat so we don't re-deduplicate the full
                # history every cycle.
                if old_keep is not None and not old_keep.empty and not new_keep.empty and time_col in old_keep.columns:
                    cutoff = new_ts.min()
                    if pd.notna(cutoff):
                        old_keep = old_keep.loc[_as_utc_ts(old_keep[time_col]) < cutoff]

            pieces = [
                frame
                for frame in (old_keep, new_keep)
                if frame is not None and not frame.empty
            ]
            if not pieces:
                out[key] = pd.DataFrame()
                continue
            merged = pd.concat(pieces, ignore_index=True, copy=False)
            dedupe_cols = [c for c in _frame_dedupe_cols(key, merged) if c in merged.columns]
            if dedupe_cols:
                merged = merged.drop_duplicates(dedupe_cols, keep="last")
            if time_col in merged.columns:
                merged[time_col] = _to_naive_utc(merged[time_col])
                sort_cols = [time_col] + (["tic"] if "tic" in merged.columns else [])
                merged = merged.sort_values(sort_cols)
            out[key] = merged.reset_index(drop=True)
        return out


@dataclass
class LiveOiFrameCache:
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    overlap_minutes: int = SOURCE_FRAME_OVERLAP_MINUTES
    last_query_mode: str = "cold"

    async def refresh(self, engine: Any, *, asof: pd.Timestamp, start: pd.Timestamp, symbol: str) -> pd.DataFrame:
        latest: pd.Timestamp | None = None
        if not self.frame.empty and "date" in self.frame.columns:
            value = _as_utc_ts(self.frame["date"]).max()
            if pd.notna(value):
                latest = pd.Timestamp(value)

        if latest is None:
            query_start = start
            self.last_query_mode = "cold_full_oi"
        else:
            query_start = max(start, latest - pd.Timedelta(minutes=max(1, int(self.overlap_minutes))))
            minutes = max(1, int(np.ceil((pd.Timestamp(asof) - query_start).total_seconds() / 60.0)))
            self.last_query_mode = f"incremental_oi_{minutes}m"

        fresh = await _query_oi(engine, asof=asof, start=query_start, symbol=symbol)
        self.frame = self._merge(fresh, start=start, asof=asof)
        return self.frame

    def _merge(self, fresh: pd.DataFrame, *, start: pd.Timestamp, asof: pd.Timestamp) -> pd.DataFrame:
        old = self.frame
        pieces = []
        if old is not None and not old.empty:
            old_ts = _as_utc_ts(old["date"])
            if fresh is not None and not fresh.empty:
                cutoff = _as_utc_ts(fresh["date"]).min()
                old = old.loc[(old_ts >= start) & (old_ts <= asof) & (old_ts < cutoff)]
            else:
                old = old.loc[(old_ts >= start) & (old_ts <= asof)]
            if not old.empty:
                pieces.append(old)
        if fresh is not None and not fresh.empty:
            new = fresh.copy()
            new["date"] = _to_naive_utc(new["date"])
            ts = _as_utc_ts(new["date"])
            new = new.loc[(ts >= start) & (ts <= asof)]
            if not new.empty:
                pieces.append(new)
        if not pieces:
            return pd.DataFrame(columns=["date", "open_interest"])
        out = pd.concat(pieces, ignore_index=True, copy=False)
        out["date"] = _to_naive_utc(out["date"])
        return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)



@dataclass
class LiveAltPoolFrameCache:
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    overlap_minutes: int = SOURCE_FRAME_OVERLAP_MINUTES
    last_query_mode: str = "cold"

    async def refresh(self, engine: Any, *, asof: pd.Timestamp, start: pd.Timestamp, symbols: tuple[str, ...] = ALT_POOL_SYMBOLS) -> pd.DataFrame:
        latest: pd.Timestamp | None = None
        if not self.frame.empty and "date" in self.frame.columns:
            value = _as_utc_ts(self.frame["date"]).max()
            if pd.notna(value):
                latest = pd.Timestamp(value)
        if latest is None:
            query_start = start
            self.last_query_mode = "cold_full_alt_pool"
        else:
            query_start = max(start, latest - pd.Timedelta(minutes=max(1, int(self.overlap_minutes))))
            minutes = max(1, int(np.ceil((pd.Timestamp(asof) - query_start).total_seconds() / 60.0)))
            self.last_query_mode = f"incremental_alt_pool_{minutes}m"
        fresh = await _query_alt_pool(engine, asof=asof, start=query_start, symbols=symbols)
        self.frame = self._merge(fresh, start=start, asof=asof)
        return self.frame.copy()

    def _merge(self, fresh: pd.DataFrame, *, start: pd.Timestamp, asof: pd.Timestamp) -> pd.DataFrame:
        old = self.frame
        pieces = []
        if old is not None and not old.empty:
            old_ts = _as_utc_ts(old["date"])
            if fresh is not None and not fresh.empty:
                cutoff = _as_utc_ts(fresh["date"]).min()
                old = old.loc[(old_ts >= start) & (old_ts <= asof) & (old_ts < cutoff)]
            else:
                old = old.loc[(old_ts >= start) & (old_ts <= asof)]
            if not old.empty:
                pieces.append(old)
        if fresh is not None and not fresh.empty:
            new = fresh.copy()
            new["date"] = _to_naive_utc(new["date"])
            ts = _as_utc_ts(new["date"])
            new = new.loc[(ts >= start) & (ts <= asof)]
            if not new.empty:
                pieces.append(new)
        if not pieces:
            return pd.DataFrame(columns=["date", "symbol", "quote_asset_volume"])
        out = pd.concat(pieces, ignore_index=True, copy=False)
        out["date"] = _to_naive_utc(out["date"])
        return out.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last").reset_index(drop=True)

def _filter_frame_since(frame: pd.DataFrame, *, col: str, start: pd.Timestamp) -> pd.DataFrame:
    if frame.empty or col not in frame.columns:
        return frame
    ts = _as_utc_ts(frame[col])
    return frame.loc[ts >= start].copy()


def _build_external_from_frames(
    *,
    market: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    cfg: LiveDbFeatureConfig,
) -> pd.DataFrame:
    btckrw = _normalise_bar_frame(frames["btckrw_1m"], tic="KRW-BTC")
    usdkrw = _normalise_bar_frame(frames["usdkrw_1m"], tic="USDKRW")
    forex = _normalise_bar_frame(frames["forex_1m"])
    return build_external_feature_frame(
        market,
        forex_bars=forex,
        btckrw_bars=btckrw,
        usdkrw_bars=usdkrw,
        interval=cfg.decision_interval,
        include_forex_components=cfg.include_forex_components,
    )


@dataclass
class LiveExternalFrameCache:
    """Tail cache for external/DXY/kimchi attachment.

    DXY is a cumulative index, so tail recomputation rescales its level to the
    previous cached overlap before derived zscore/momentum features are built.
    """

    enriched: pd.DataFrame | None = None
    external: pd.DataFrame | None = None
    context_bars: int = EXTERNAL_TAIL_CONTEXT_BARS
    output_bars: int = FEATURE_TAIL_OUTPUT_BARS
    last_mode: str = "cold"

    def refresh(self, *, market: pd.DataFrame, frames: dict[str, pd.DataFrame], cfg: LiveDbFeatureConfig) -> pd.DataFrame:
        if self.enriched is None or self.external is None or self.enriched.empty:
            external = _build_external_from_frames(market=market, frames=frames, cfg=cfg)
            enriched = attach_external_features(
                market,
                external,
                tolerance=cfg.external_tolerance,
                zscore_window=cfg.zscore_window,
                momentum_period=cfg.zscore_window,
            )
            self.external = external.copy()
            self.enriched = enriched.copy()
            self.last_mode = "cold_full_external"
            return enriched

        output_start = max(0, len(market) - max(1, int(self.output_bars)))
        context_start = max(0, output_start - max(1, int(self.context_bars)))
        market_tail = market.iloc[context_start:].reset_index(drop=True)
        tail_start = pd.Timestamp(market_tail["date"].iloc[0])
        if tail_start.tzinfo is None:
            tail_start = tail_start.tz_localize("UTC")
        else:
            tail_start = tail_start.tz_convert("UTC")
        source_start = tail_start - pd.Timedelta(str(cfg.external_tolerance))
        tail_frames = {
            **frames,
            "btckrw_1m": _filter_frame_since(frames["btckrw_1m"], col="date", start=source_start),
            "usdkrw_1m": _filter_frame_since(frames["usdkrw_1m"], col="date", start=source_start),
            "forex_1m": _filter_frame_since(frames["forex_1m"], col="date", start=source_start),
        }
        external_tail = _build_external_from_frames(market=market_tail, frames=tail_frames, cfg=cfg)
        external_tail = self._rescale_tail_dxy(external_tail)
        enriched_tail = attach_external_features(
            market_tail,
            external_tail,
            tolerance=cfg.external_tolerance,
            zscore_window=cfg.zscore_window,
            momentum_period=cfg.zscore_window,
        ).reset_index(drop=True)
        replace_from_tail = output_start - context_start
        replacement = enriched_tail.iloc[replace_from_tail:].copy()
        replacement.index = range(output_start, output_start + len(replacement))
        prefix = self.enriched.iloc[:output_start].copy()
        enriched = pd.concat([prefix, replacement], axis=0).reindex(range(len(market)))
        self.enriched = enriched

        external_prefix = self.external
        if external_prefix is not None and not external_prefix.empty and "date" in external_prefix.columns and not external_tail.empty:
            cutoff = pd.Timestamp(external_tail["date"].min())
            external_prefix = external_prefix.loc[pd.to_datetime(external_prefix["date"]) < cutoff]
            self.external = (
                pd.concat([external_prefix, external_tail], ignore_index=True, copy=False)
                .sort_values("date")
                .drop_duplicates("date", keep="last")
                .reset_index(drop=True)
            )
        self.last_mode = f"tail_external_context={len(market_tail)} output={len(replacement)}"
        return enriched

    def _rescale_tail_dxy(self, external_tail: pd.DataFrame) -> pd.DataFrame:
        if (
            self.external is None
            or self.external.empty
            or external_tail.empty
            or "dxy" not in external_tail.columns
            or "dxy" not in self.external.columns
        ):
            return external_tail
        old = self.external.loc[:, ["date", "dxy"]].dropna().copy()
        new = external_tail.loc[:, ["date", "dxy"]].dropna().copy()
        if old.empty or new.empty:
            return external_tail
        joined = new.merge(old, on="date", how="inner", suffixes=("_new", "_old"))
        joined = joined[(joined["dxy_new"].abs() > 1e-12) & np.isfinite(joined["dxy_new"]) & np.isfinite(joined["dxy_old"])]
        if joined.empty:
            return external_tail
        ratio = float(joined.iloc[0]["dxy_old"] / joined.iloc[0]["dxy_new"])
        if not np.isfinite(ratio) or ratio <= 0.0:
            return external_tail
        out = external_tail.copy()
        out["dxy"] = pd.to_numeric(out["dxy"], errors="coerce") * ratio
        return out


def _add_portfolio_oi_features(enriched: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    available = pd.to_numeric(
        enriched.get("open_interest_available", enriched["open_interest"].notna().astype(float)),
        errors="coerce",
    ).fillna(0.0)
    out["open_interest_available"] = available.to_numpy(dtype=float)
    oi_raw = pd.to_numeric(enriched["open_interest"], errors="coerce").where(available > 0.5)
    oi_s = pd.Series(oi_raw.astype(float).replace(0, np.nan), index=enriched.index)
    px = pd.Series(enriched["close"].astype(float), index=enriched.index)
    for w, name in [(6, "30m"), (12, "1h"), (24, "2h"), (48, "4h"), (96, "8h")]:
        oi_ret = np.log(oi_s / oi_s.shift(w)).replace([np.inf, -np.inf], np.nan)
        px_ret = np.log(px / px.shift(w)).replace([np.inf, -np.inf], np.nan)
        div = oi_ret - px_ret
        for nm, series in [
            (f"oi_ret_{name}", oi_ret),
            (f"px_ret_{name}", px_ret),
            (f"oi_minus_px_{name}", div),
            (f"px_minus_oi_{name}", px_ret - oi_ret),
        ]:
            mu = series.rolling(288, min_periods=50).mean()
            sd = series.rolling(288, min_periods=50).std(ddof=0)
            out[nm] = series
            out[nm + "_z"] = ((series - mu) / sd.replace(0, np.nan)).clip(-5, 5)
    return out


def _add_activity_flow_feature(enriched: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    try:
        interest = build_interest_features(enriched, out)
        raw = _build_score_frame(enriched, out, interest)
        train_mask = np.ones(len(enriched), dtype=bool)
        score, _ = _score_variant(raw, train_mask, "activity_flow_htf")
        out["activity_flow_htf"] = score
    except Exception:
        pass
    return out



def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    mean = values.rolling(window, min_periods=max(12, window // 4)).mean()
    std = values.rolling(window, min_periods=max(12, window // 4)).std(ddof=0)
    return ((values - mean) / std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(-8, 8).fillna(0.0)


def _momentum(series: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return (values / values.shift(window).replace(0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan).clip(-10, 10).fillna(0.0)


def _add_live_volume_wave_features(
    enriched: pd.DataFrame,
    features: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    alt_pool: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add live-only wave/volume columns required by current portfolio scans."""

    out = features.copy()
    wave = build_wave_feature_frame(enriched, window=144).add_prefix("wr_")
    for col in ["wr_vwap_dev_z"]:
        if col in wave.columns:
            out[col] = wave[col].to_numpy(float)

    volume = pd.to_numeric(enriched.get("volume"), errors="coerce").astype(float)
    close = pd.to_numeric(enriched.get("close"), errors="coerce").astype(float)
    quote_volume = pd.to_numeric(enriched.get("quote_asset_volume", volume * close), errors="coerce").astype(float)
    out["vg_vol_mom_288"] = _momentum(volume + 1.0, 288).to_numpy(float)
    out["vg_qvol_mom_288"] = _momentum(quote_volume + 1.0, 288).to_numpy(float)

    try:
        if alt_pool is None or alt_pool.empty:
            raise ValueError("alt_pool_empty")
        pool = alt_pool[["date", "symbol", "quote_asset_volume"]].copy()
        pool["date"] = _to_naive_utc(pool["date"])
        pool["quote_asset_volume"] = pd.to_numeric(pool["quote_asset_volume"], errors="coerce").fillna(0.0)
        alt_5m = (
            pool.set_index("date")
            .groupby("symbol")["quote_asset_volume"]
            .resample("5min", label="left", closed="left")
            .sum()
            .reset_index()
        )
        alt_total = alt_5m.groupby("date", as_index=False)["quote_asset_volume"].sum().rename(columns={"quote_asset_volume": "alt_quote_volume"})
        joined_alt = pd.merge_asof(
            enriched[["date", "quote_asset_volume"]].sort_values("date"),
            alt_total.sort_values("date"),
            on="date",
            direction="backward",
            tolerance=pd.Timedelta("5min"),
        ).sort_index()
        alt_value = pd.to_numeric(joined_alt["alt_quote_volume"], errors="coerce")
        btc_value = pd.to_numeric(joined_alt["quote_asset_volume"], errors="coerce")
        alt_available = alt_value.notna() & btc_value.notna() & (btc_value != 0)
        alt_ratio = (alt_value / btc_value.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["vg_alt_btc_qv_ratio_z_72"] = _rolling_z(alt_ratio, 72).to_numpy(float)
        out["vg_alt_btc_qv_ratio_z_288"] = _rolling_z(alt_ratio, 288).to_numpy(float)
        out["alt_pool_available"] = alt_available.to_numpy(dtype=float)
    except Exception:
        out["vg_alt_btc_qv_ratio_z_72"] = 0.0
        out["vg_alt_btc_qv_ratio_z_288"] = 0.0
        out["alt_pool_available"] = 0.0

    try:
        upbit = resample_market_bars(frames["btckrw_1m"], "5min")
        upbit = upbit.rename(columns={"close": "upbit_close", "volume": "upbit_volume"})
        joined = pd.merge_asof(
            enriched[["date", "quote_asset_volume", "usdkrw"]].sort_values("date"),
            upbit[["date", "upbit_close", "upbit_volume"]].sort_values("date"),
            on="date",
            direction="backward",
            tolerance=pd.Timedelta("7min"),
        ).sort_index()
        upbit_value_krw = pd.to_numeric(joined["upbit_volume"], errors="coerce") * pd.to_numeric(joined["upbit_close"], errors="coerce")
        binance_value_krw = pd.to_numeric(joined["quote_asset_volume"], errors="coerce") * pd.to_numeric(joined["usdkrw"], errors="coerce")
        upbit_available = upbit_value_krw.notna() & binance_value_krw.notna() & (binance_value_krw != 0)
        upbit_ratio = (upbit_value_krw / binance_value_krw.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["vg_upbit_binance_vol_ratio_z_288"] = _rolling_z(upbit_ratio, 288).to_numpy(float)
        out["upbit_volume_available"] = upbit_available.to_numpy(dtype=float)
    except Exception:
        out["vg_upbit_binance_vol_ratio_z_288"] = 0.0
        out["upbit_volume_available"] = 0.0
    return out.replace([np.inf, -np.inf], np.nan)

def _build_portfolio_feature_frame(enriched: pd.DataFrame, cfg: LiveDbFeatureConfig) -> pd.DataFrame:
    features = build_market_feature_frame(
        enriched,
        window_size=cfg.feature_window_size,
        zscore_window=cfg.zscore_window,
        volume_window=cfg.volume_window,
    ).copy()
    features = _add_portfolio_oi_features(enriched, features)
    features = _add_activity_flow_feature(enriched, features)
    return features.replace([np.inf, -np.inf], np.nan)


@dataclass
class LiveFeatureFrameCache:
    """Tail-only feature-frame cache with full-compute equivalence guardrails."""

    enriched: pd.DataFrame | None = None
    features: pd.DataFrame | None = None
    context_bars: int = FEATURE_TAIL_CONTEXT_BARS
    output_bars: int = FEATURE_TAIL_OUTPUT_BARS
    last_mode: str = "cold"

    def refresh(self, enriched: pd.DataFrame, cfg: LiveDbFeatureConfig) -> pd.DataFrame:
        if self.enriched is None or self.features is None or self.features.empty:
            self.enriched = enriched.copy()
            self.features = _build_portfolio_feature_frame(enriched, cfg)
            self.last_mode = "cold_full_features"
            return self.features.copy()

        if len(enriched) < max(10, int(self.context_bars) // 2):
            self.enriched = enriched.copy()
            self.features = _build_portfolio_feature_frame(enriched, cfg)
            self.last_mode = "fallback_short_full_features"
            return self.features.copy()

        output_start = max(0, len(enriched) - max(1, int(self.output_bars)))
        context_start = max(0, output_start - max(1, int(self.context_bars)))
        tail_enriched = enriched.iloc[context_start:].reset_index(drop=True)
        tail_features = _build_portfolio_feature_frame(tail_enriched, cfg).reset_index(drop=True)
        replace_from_tail = output_start - context_start
        replacement = tail_features.iloc[replace_from_tail:].copy()
        replacement.index = range(output_start, output_start + len(replacement))

        prefix = self.features.iloc[:output_start].copy()
        spliced = pd.concat([prefix, replacement], axis=0)
        spliced = spliced.reindex(range(len(enriched)))

        # activity_flow_htf standardizes over the full available live window.
        # Recompute it over the spliced cache so tail rows match full compute.
        spliced = _add_activity_flow_feature(enriched, spliced)
        self.enriched = enriched.copy()
        self.features = spliced.replace([np.inf, -np.inf], np.nan)
        self.last_mode = f"tail_features_context={len(tail_enriched)} output={len(replacement)}"
        return self.features.copy()


@dataclass(frozen=True)
class PortfolioLiveConfig:
    portfolio_config: Path
    execution_config: Path
    env_path: Path = Path(".env")
    state_file: Path = Path(".omx/state/portfolio_live_state.json")
    strategy_name: str = "rllm"
    exchange: str = "binance"
    lookback_minutes: int = 45_000
    close_delay_sec: float = 0.25
    max_freshness_wait_sec: float = 8.0
    freshness_poll_sec: float = 0.5
    freshness_notify_channel: str = "market_data_bar"
    run_immediately: bool = False
    live: bool = False
    allow_live_orders: bool = False
    leverage: int = 6
    allocation_mode: Literal["research_gross", "normalize_weights"] = "research_gross"
    max_iterations: int | None = None
    entry_timeout_fraction: float = 0.25
    max_entry_wait_sec: int = 300
    max_exit_wait_sec: int = 600
    maker_refresh_interval_sec: int = 60
    entry_maker_max_deviation_pct: float = 0.003
    exit_maker_max_deviation_pct: float = 0.002
    cancel_stale_open_orders: bool = True
    portfolio_selector_overlay: Path | None = None
    rex_selector_adapter_dir: Path | None = None
    rex_selector_model_name: str = RexLlmSelectorConfig.model_name
    rex_selector_score_normalization: str = "sum"
    rex_selector_fail_closed: bool = True
    rex_selector_require_cuda: bool = True


@dataclass(frozen=True)
class FreshnessRequirement:
    table: str
    symbol: str
    interval: str | None
    required_ts: pd.Timestamp
    source: str
    period: str | None = None

    @property
    def key(self) -> str:
        parts = [self.table, self.symbol]
        if self.interval:
            parts.append(self.interval)
        if self.period:
            parts.append(self.period)
        return ":".join(parts)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text())


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")


def _ensure_trade_executions_table(engine: Any) -> None:
    """Create the shared execution ledger table if it does not exist."""

    from sqlalchemy import text

    ddl = """
        CREATE TABLE IF NOT EXISTS trade_executions (
            id BIGSERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            strategy_name TEXT NOT NULL,
            sub_strategy_name TEXT,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            quote_asset TEXT,
            action TEXT NOT NULL,
            side TEXT,
            position_side TEXT,
            order_type TEXT,
            signal_id TEXT,
            status TEXT,
            execution_started_at TIMESTAMPTZ,
            expires_at TIMESTAMPTZ,
            execution_finished_at TIMESTAMPTZ,
            computing_wall_time_sec DOUBLE PRECISION,
            order_id TEXT,
            client_order_id TEXT,
            quantity_requested NUMERIC,
            quantity_filled NUMERIC,
            reference_price NUMERIC,
            avg_price NUMERIC,
            maker_max_deviation_pct DOUBLE PRECISION,
            refresh_interval_sec INTEGER,
            error TEXT,
            realized_pnl NUMERIC,
            commission NUMERIC,
            commission_asset TEXT,
            net_realized_pnl NUMERIC,
            payload JSONB NOT NULL DEFAULT '{}'::jsonb
        )
        """
    migrations = [
        "ALTER TABLE trade_executions ADD COLUMN IF NOT EXISTS realized_pnl NUMERIC",
        "ALTER TABLE trade_executions ADD COLUMN IF NOT EXISTS commission NUMERIC",
        "ALTER TABLE trade_executions ADD COLUMN IF NOT EXISTS commission_asset TEXT",
        "ALTER TABLE trade_executions ADD COLUMN IF NOT EXISTS net_realized_pnl NUMERIC",
    ]
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_trade_executions_strategy_created ON trade_executions(strategy_name, sub_strategy_name, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_trade_executions_signal ON trade_executions(strategy_name, signal_id)",
        "CREATE INDEX IF NOT EXISTS idx_trade_executions_order ON trade_executions(exchange, symbol, order_id, client_order_id)",
    ]
    with engine.begin() as conn:
        conn.execute(text(ddl))
        for stmt in migrations:
            conn.execute(text(stmt))
        for stmt in indexes:
            conn.execute(text(stmt))


def _safe_decimal_value(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        return str(Decimal(str(value)))
    except Exception:
        return None


def _log_trade_execution(
    engine: Any,
    *,
    strategy_name: str,
    sub_strategy_name: str | None,
    exchange: str,
    symbol: str,
    action: str,
    side: str | None = None,
    position_side: str | None = None,
    order_type: str | None = None,
    signal_id: str | None = None,
    status: str | None = None,
    order_info: dict[str, Any] | None = None,
    computing_wall_time_sec: float | None = None,
    error: str | None = None,
) -> None:
    """Append one execution event to the shared DB ledger."""

    from sqlalchemy import text

    info = order_info or {}
    payload = json.dumps(info, ensure_ascii=False, default=str)
    sql = text(
        """
        INSERT INTO trade_executions (
            strategy_name, sub_strategy_name, exchange, symbol, quote_asset,
            action, side, position_side, order_type, signal_id, status,
            execution_started_at, expires_at, execution_finished_at,
            computing_wall_time_sec, order_id, client_order_id,
            quantity_requested, quantity_filled, reference_price, avg_price,
            maker_max_deviation_pct, refresh_interval_sec, error,
            realized_pnl, commission, commission_asset, net_realized_pnl, payload
        ) VALUES (
            :strategy_name, :sub_strategy_name, :exchange, :symbol, :quote_asset,
            :action, :side, :position_side, :order_type, :signal_id, :status,
            :execution_started_at, :expires_at, :execution_finished_at,
            :computing_wall_time_sec, :order_id, :client_order_id,
            :quantity_requested, :quantity_filled, :reference_price, :avg_price,
            :maker_max_deviation_pct, :refresh_interval_sec, :error,
            :realized_pnl, :commission, :commission_asset, :net_realized_pnl, CAST(:payload AS jsonb)
        )
        """
    )
    raw_order = info.get("raw_order") if isinstance(info.get("raw_order"), dict) else {}
    trade_report = info.get("trade_report") if isinstance(info.get("trade_report"), dict) else {}
    order_id = info.get("order_id") or raw_order.get("orderId")
    client_order_id = info.get("client_order_id") or raw_order.get("clientOrderId")
    with engine.begin() as conn:
        conn.execute(
            sql,
            {
                "strategy_name": strategy_name,
                "sub_strategy_name": sub_strategy_name,
                "exchange": exchange,
                "symbol": symbol,
                "quote_asset": "USDT" if symbol.endswith("USDT") else None,
                "action": action,
                "side": side,
                "position_side": position_side,
                "order_type": order_type,
                "signal_id": signal_id,
                "status": status or info.get("status"),
                "execution_started_at": info.get("started_at"),
                "expires_at": info.get("deadline_at"),
                "execution_finished_at": info.get("finished_at"),
                "computing_wall_time_sec": computing_wall_time_sec if computing_wall_time_sec is not None else info.get("wall_time_sec"),
                "order_id": None if order_id is None else str(order_id),
                "client_order_id": None if client_order_id is None else str(client_order_id),
                "quantity_requested": _safe_decimal_value(info.get("requested_quantity")),
                "quantity_filled": _safe_decimal_value(info.get("filled_quantity")),
                "reference_price": _safe_decimal_value(info.get("reference_price")),
                "avg_price": _safe_decimal_value(info.get("avg_price")),
                "maker_max_deviation_pct": info.get("max_deviation_pct"),
                "refresh_interval_sec": info.get("refresh_interval_sec"),
                "error": error or info.get("error"),
                "realized_pnl": _safe_decimal_value(trade_report.get("realized_pnl")),
                "commission": _safe_decimal_value(trade_report.get("commission")),
                "commission_asset": trade_report.get("commission_asset"),
                "net_realized_pnl": _safe_decimal_value(trade_report.get("net_realized_pnl")),
                "payload": payload,
            },
        )


def _load_state(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"open_sleeves": {}, "processed_signals": {}}
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {"open_sleeves": {}, "processed_signals": {}}
    if not isinstance(data, dict):
        return {"open_sleeves": {}, "processed_signals": {}}
    data.setdefault("open_sleeves", {})
    data.setdefault("processed_signals", {})
    return data


def _required_availability_flags(feature: str) -> tuple[str, ...]:
    """Return live source flags required before a gate may use ``feature``.

    Live feature construction intentionally uses backward as-of joins.  Numeric
    values alone are therefore not enough proof that a gate is executable at the
    decision timestamp; source-family availability flags must also be fresh.
    """

    name = str(feature)
    if name == "open_interest" or name.startswith(("oi_", "oi_minus_px", "px_minus_oi")):
        return ("open_interest_available",)
    if name == "premium_index" or name.startswith("premium_"):
        return ("premium_available",)
    if name == "funding_rate" or name.startswith("funding_"):
        return ("funding_available",)
    if name.startswith("kimchi_"):
        return ("kimchi_available",)
    if name.startswith("usdkrw"):
        return ("usdkrw_available",)
    if name.startswith("dxy"):
        return ("dxy_available",)
    if name.startswith("fx_"):
        return ("external_any_available",)
    if name.startswith("vg_alt_"):
        return ("alt_pool_available",)
    if name.startswith("vg_upbit_"):
        return ("upbit_volume_available", "usdkrw_available")
    return ()


def _gate_pass(row: pd.Series, gates: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    ok = True
    for gate in gates:
        feature = str(gate["feature"])
        for flag in _required_availability_flags(feature):
            flag_value = float(row.get(flag, np.nan))
            flag_passed = np.isfinite(flag_value) and flag_value > 0.5
            reasons.append(f"{feature}_source={flag}:{'pass' if flag_passed else 'fail'}")
            ok &= bool(flag_passed)
        op = str(gate["op"])
        threshold = float(gate["threshold"])
        value = float(row.get(feature, np.nan))
        passed = np.isfinite(value) and ((value >= threshold) if op in {">=", "ge"} else (value <= threshold))
        reasons.append(f"{feature}={value:.6g}{op}{threshold:.6g}:{'pass' if passed else 'fail'}")
        ok &= bool(passed)
    return ok, reasons


def _interval_slot(ts: pd.Timestamp, stride_bars: int, interval_minutes: int = 5) -> bool:
    """Return True when a timestamp is on the sleeve's stride grid."""

    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    minutes = int(t.timestamp() // 60)
    return (minutes // int(interval_minutes)) % int(stride_bars) == 0


async def _query_oi(engine: Any, *, asof: pd.Timestamp, start: pd.Timestamp, symbol: str) -> pd.DataFrame:
    from sqlalchemy import text

    sql = text(
        """
        SELECT ts AS date, sum_open_interest AS open_interest
        FROM open_interest_binance
        WHERE symbol = :symbol AND period = '5m' AND ts >= :start AND ts <= :asof
        ORDER BY ts
        """
    )
    with engine.connect() as conn:
        return pd.read_sql_query(sql, conn, params={"symbol": symbol, "start": start.to_pydatetime(), "asof": asof.to_pydatetime()})



async def _query_alt_pool(engine: Any, *, asof: pd.Timestamp, start: pd.Timestamp, symbols: tuple[str, ...]) -> pd.DataFrame:
    from sqlalchemy import bindparam, text

    if not symbols:
        return pd.DataFrame(columns=["date", "symbol", "quote_asset_volume"])
    sql = text(
        """
        SELECT ts AS date, symbol, quote_asset_volume
        FROM bars_binance
        WHERE symbol IN :symbols AND interval = '1m' AND ts >= :start AND ts <= :asof
        ORDER BY ts, symbol
        """
    ).bindparams(bindparam("symbols", expanding=True))
    with engine.connect() as conn:
        return pd.read_sql_query(
            sql,
            conn,
            params={"symbols": tuple(symbols), "start": start.to_pydatetime(), "asof": asof.to_pydatetime()},
        )

async def build_live_portfolio_frames(
    *,
    engine: Any,
    asof: pd.Timestamp,
    cfg: LiveDbFeatureConfig,
    source_cache: LiveSourceFrameCache | None = None,
    feature_cache: LiveFeatureFrameCache | None = None,
    oi_cache: LiveOiFrameCache | None = None,
    external_cache: LiveExternalFrameCache | None = None,
    alt_pool_cache: LiveAltPoolFrameCache | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build enriched market and feature frame with live OI included."""

    start = asof - pd.Timedelta(minutes=int(cfg.lookback_minutes))
    frames = source_cache.refresh(engine, asof=asof, cfg=cfg) if source_cache is not None else query_live_source_frames(engine, asof=asof, cfg=cfg)
    oi = await oi_cache.refresh(engine, asof=asof, start=start, symbol=cfg.symbol) if oi_cache is not None else await _query_oi(engine, asof=asof, start=start, symbol=cfg.symbol)
    alt_pool = await alt_pool_cache.refresh(engine, asof=asof, start=start) if alt_pool_cache is not None else await _query_alt_pool(engine, asof=asof, start=start, symbols=ALT_POOL_SYMBOLS)

    market = resample_market_bars(frames["btcusdt_1m"], cfg.decision_interval)
    if oi.empty:
        raise RuntimeError("open_interest_binance returned no rows; portfolio OI sleeves cannot run")
    oi = oi.copy()
    oi["date"] = pd.to_datetime(oi["date"], utc=True).dt.tz_convert(None)
    oi["open_interest"] = pd.to_numeric(oi["open_interest"], errors="coerce")
    market = pd.merge_asof(
        market.sort_values("date"),
        oi[["date", "open_interest"]].sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta("10min"),
    )
    market["open_interest_available"] = market["open_interest"].notna().astype(float)

    enriched = (
        external_cache.refresh(market=market, frames=frames, cfg=cfg)
        if external_cache is not None
        else attach_external_features(
            market,
            _build_external_from_frames(market=market, frames=frames, cfg=cfg),
            tolerance=cfg.external_tolerance,
            zscore_window=cfg.zscore_window,
            momentum_period=cfg.zscore_window,
        )
    )
    enriched = attach_binance_um_aux_frames(
        enriched,
        funding_frame=frames["funding"],
        premium_frame=frames["premium_1m"],
        funding_tolerance=cfg.funding_tolerance,
        premium_tolerance=cfg.premium_tolerance,
        zscore_window=cfg.zscore_window,
    )
    features = feature_cache.refresh(enriched, cfg) if feature_cache is not None else _build_portfolio_feature_frame(enriched, cfg)
    features = _add_live_volume_wave_features(enriched, features, frames, alt_pool)
    return enriched, features.replace([np.inf, -np.inf], np.nan)


def _current_atr(enriched: pd.DataFrame, period: int = 15) -> float:
    high = pd.to_numeric(enriched["high"], errors="coerce")
    low = pd.to_numeric(enriched["low"], errors="coerce")
    close = pd.to_numeric(enriched["close"], errors="coerce")
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    out = float(tr.rolling(max(1, int(period)), min_periods=1).mean().iloc[-1])
    return out if np.isfinite(out) else 0.0


def _dynamic_exit_due(
    *,
    open_state: dict[str, Any],
    latest_feature_row: pd.Series,
    latest_bar: pd.Timestamp,
    interval_minutes: int,
) -> tuple[bool, list[str]]:
    dynamic = open_state.get("dynamic_exit")
    if not isinstance(dynamic, dict):
        return False, []
    signal_ts = pd.Timestamp(open_state.get("signal_date"))
    if signal_ts.tzinfo is None:
        signal_ts = signal_ts.tz_localize("UTC")
    latest = pd.Timestamp(latest_bar)
    if latest.tzinfo is None:
        latest = latest.tz_localize("UTC")
    else:
        latest = latest.tz_convert("UTC")
    bars_elapsed = int(max(0, (latest - signal_ts).total_seconds() // (60 * max(1, int(interval_minutes)))))
    min_bars = int(dynamic.get("min_bars", 0) or 0)
    if bars_elapsed < min_bars:
        return False, [f"dynamic_exit_min_bars={bars_elapsed}<{min_bars}:wait"]
    gates = dynamic.get("gates", [])
    if not isinstance(gates, list) or not gates:
        return False, ["dynamic_exit_gates=missing:wait"]
    ok, reasons = _gate_pass(latest_feature_row, gates)
    return bool(ok), [f"dynamic_exit_bars={bars_elapsed}>=min{min_bars}:pass", *reasons]



def _load_portfolio_selector_overlay(portfolio: dict[str, Any], explicit_path: Path | None) -> dict[str, Any] | None:
    """Load a bounded portfolio-level selector overlay if configured.

    The selector is intentionally constrained to ALLOW/BLOCK_RISK for already
    triggered sleeves.  It cannot create signals, alter exits, resize sleeves,
    or change leverage.
    """

    raw_path = explicit_path or portfolio.get("portfolio_selector_overlay") or portfolio.get("llm_selector_overlay")
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"portfolio selector overlay not found: {path}")
    overlay = _load_json(path)
    sel = overlay.get("symbolic_proxy")
    if not isinstance(sel, dict):
        raise ValueError(f"portfolio selector overlay missing symbolic_proxy: {path}")
    keys = sel.get("context_keys")
    if not isinstance(keys, list) or not keys:
        raise ValueError(f"portfolio selector overlay missing context_keys: {path}")
    return {**overlay, "_path": str(path)}


def _apply_portfolio_selector_overlay(
    sleeve_scores: list[dict[str, Any]],
    *,
    overlay: dict[str, Any] | None,
    enriched: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, Any] | None:
    """Apply the bounded LLM-selector proxy to pending live entries.

    This mirrors ``training.evaluate_portfolio_llm_selector``: build the same
    compact state tokens from signal-time features, compute a context id, and
    block only if that train-only bad context is in the frozen overlay. Existing
    open sleeves are not force-closed; only new entries are suppressed.
    """

    active = [s for s in sleeve_scores if s.get("active")]
    if not overlay or not active:
        return None
    sel = overlay.get("symbolic_proxy", {})
    keys = tuple(str(k) for k in sel.get("context_keys", []))
    blocked = {str(x.get("context_id")) for x in sel.get("blocked_contexts", []) if isinstance(x, dict)}
    pos = len(features) - 1
    tokens = _base_context_tokens(pos, market=enriched, feat=features)
    cid = _context_id(tokens, keys) if keys else ""
    allowed = cid not in blocked
    action = "ALLOW" if allowed else "BLOCK_RISK"
    pending = [str(s.get("name")) for s in active]
    record = {
        "selector": str(overlay.get("name") or overlay.get("_path") or "portfolio_selector_overlay"),
        "overlay_path": overlay.get("_path"),
        "output_space": overlay.get("output_space", ["ALLOW", "BLOCK_RISK"]),
        "action": action,
        "allowed": allowed,
        "context_id": cid,
        "context_keys": list(keys),
        "pending_sleeves": pending,
        "state_tokens": tokens,
        "bounded_contract": {
            "can_create_signals": False,
            "can_change_side": False,
            "can_change_size": False,
            "can_change_exit": False,
            "applies_to": "new_entries_only",
        },
    }
    reason = f"portfolio_selector_context={cid}:{action}"
    for sleeve in active:
        sleeve.setdefault("reasons", []).append(reason)
        sleeve["portfolio_selector"] = {k: v for k, v in record.items() if k != "state_tokens"}
        if not allowed:
            sleeve["active"] = False
    return record

def _score_sleeves(
    *,
    portfolio: dict[str, Any],
    enriched: pd.DataFrame,
    features: pd.DataFrame,
    exec_cfg: WaveExecutionConfig,
    asof: pd.Timestamp,
    rex_selector_cfg: RexLlmSelectorConfig | None = None,
) -> list[dict[str, Any]]:
    row = features.iloc[-1]
    ts = pd.Timestamp(enriched.iloc[-1]["date"])
    close = float(enriched.iloc[-1]["close"])
    atr = _current_atr(enriched, exec_cfg.atr_period)
    out: list[dict[str, Any]] = []
    for sleeve in portfolio["base_sleeves"]:
        name = str(sleeve["name"])
        source = str(sleeve["source"])
        weight = float(sleeve["weight"])
        configured_side = str(sleeve["side"]).upper()
        side = configured_side
        active = False
        reasons: list[str] = []
        hold = 0
        stride = 1
        dynamic_exit: dict[str, Any] | None = None

        if source.endswith(".json") and Path(source).exists():
            cfg = _load_json(source)
            overlay_cfg = cfg if "selector_overlay" in source else None
            if "base_candidate" in cfg and "gates" not in cfg and "signal" not in cfg:
                cfg = _load_json(str(cfg["base_candidate"]))
            if "signal" in cfg:
                gates = cfg["signal"]["gates"]
                hold = int(cfg["signal"].get("hold_bars_5m", cfg["signal"].get("hold_bars", 0)))
                stride = int(cfg["signal"].get("stride_bars_5m", cfg["signal"].get("stride_bars", 1)))
            else:
                gates = cfg["gates"]
                hold = int(cfg.get("hold_bars", cfg.get("hold_bars_5m", 0)))
                stride = int(cfg.get("stride_bars", cfg.get("stride_bars_5m", 1)))
            gate_ok, reasons = _gate_pass(row, gates)
            stride_ok = _interval_slot(ts, stride, exec_cfg.interval_minutes)
            active = bool(gate_ok and stride_ok)
            reasons.append(f"stride={stride}:{'pass' if stride_ok else 'fail'}")
            dynamic_exit = cfg.get("dynamic_exit") if isinstance(cfg.get("dynamic_exit"), dict) else None

            if overlay_cfg is not None:
                sel = overlay_cfg.get("symbolic_proxy", {})
                blocked = {x["context_id"] for x in sel.get("blocked_contexts", [])}
                keys = tuple(sel.get("context_keys", []))
                cid = _context_id(_tokens(len(features) - 1, market=enriched, feat=features), keys) if keys else ""
                selector_ok = cid not in blocked
                active &= selector_ok
                reasons.append(f"selector_context={cid}:{'ALLOW' if selector_ok else 'BLOCK'}")
        else:
            # REX sleeve uses the frozen REX live policy path.  Research treats
            # this sleeve as a directional candidate selected by the REX rule
            # itself, so live configs may set side=AUTO/BOTH to execute either
            # LONG or SHORT.  Explicit LONG/SHORT configs remain directional
            # filters for older short-only deployments.
            record = build_rex_live_policy_record(
                enriched,
                features,
                policy_cfg=RexLivePolicyConfig(),
                execution_cfg=exec_cfg,
                scorer_asof=asof,
                selector_cfg=rex_selector_cfg,
            )
            candidate_side = str(record.get("candidate_side", "")).upper()
            side_ok = candidate_side in {"LONG", "SHORT"}
            side_filter_ok = configured_side in {"AUTO", "BOTH"} or candidate_side == configured_side
            active = bool(record.get("prediction") == "TRADE" and side_ok and side_filter_ok)
            if side_ok and configured_side in {"AUTO", "BOTH"}:
                side = candidate_side
            hold = int(record.get("action", {}).get("hold_bars", 144))
            stride = 1
            reasons = [
                str(record.get("reason", "")),
                f"rex_candidate_side={candidate_side or 'NONE'}",
                f"configured_side={configured_side}:{'pass' if side_filter_ok else 'fail'}",
            ]

        out.append(
            {
                "name": name,
                "source": source,
                "side": side,
                "weight": weight,
                "active": active,
                "hold_bars": hold,
                "stride_bars": stride,
                "date": str(ts),
                "current_close": close,
                "current_atr": atr,
                "signal_id": f"{name}:{side}:{ts.isoformat()}",
                "reasons": reasons,
                "dynamic_exit": dynamic_exit,
            }
        )
    return out


def _seconds_until_next_interval(now: pd.Timestamp, *, interval_minutes: int, close_delay_sec: float) -> float:
    ts = pd.Timestamp(now)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    interval_sec = int(interval_minutes) * 60
    current_sec = ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1_000_000
    shifted = current_sec - float(close_delay_sec)
    next_sec = (int(np.floor(shifted / interval_sec)) + 1) * interval_sec + float(close_delay_sec)
    wait = next_sec - current_sec
    return float(wait if wait > 0 else wait + interval_sec)


def _status(msg: str) -> None:
    LOG.info(msg)


def _configure_logging(*, log_file: str | Path, log_level: str = "INFO") -> None:
    path = Path(log_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, str(log_level).upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(file_handler)
    for noisy in ("httpx", "httpcore", "urllib3", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)



PORTFOLIO_ORDER_PREFIX = "rpf"


def _entry_ttl_seconds(
    sleeve: dict[str, Any],
    *,
    interval_minutes: int,
    timeout_fraction: float,
    max_entry_wait_sec: int,
) -> int:
    """Bound post-only entry waiting by the sleeve's signal cycle.

    Entry freshness is tied to the sleeve stride because a missed maker entry
    should not keep chasing after the next decision opportunity is near.  Hold
    bars are only used as a fallback for sleeves without a positive stride.
    """

    stride = int(sleeve.get("stride_bars") or 0)
    hold = int(sleeve.get("hold_bars") or 0)
    cycle_bars = stride if stride > 0 else max(1, hold)
    cycle_sec = max(1, cycle_bars) * int(interval_minutes) * 60
    ttl = int(cycle_sec * max(0.0, float(timeout_fraction)))
    ttl = max(30, ttl)
    if max_entry_wait_sec > 0:
        ttl = min(ttl, int(max_entry_wait_sec))
    return ttl


def _portfolio_sleeve_key(sleeve_name: str) -> str:
    return hashlib.sha1(str(sleeve_name).encode("utf-8")).hexdigest()[:6]


def _portfolio_client_order_id(signal_id: str, *, sleeve_name: str, now_sec: int | None = None) -> str:
    """Create a Binance-safe, bot-owned client order id.

    Binance's limit is short; this prefix lets stale-order cleanup avoid
    touching unrelated bots that may share the same symbol/account.
    """

    epoch = int(now_sec if now_sec is not None else time.time())
    sleeve_key = _portfolio_sleeve_key(sleeve_name)
    digest = hashlib.sha1(signal_id.encode("utf-8")).hexdigest()[:8]
    return f"{PORTFOLIO_ORDER_PREFIX}_{epoch}_{sleeve_key}_{digest}"


def _portfolio_order_parts(client_order_id: str) -> dict[str, str] | None:
    """Parse this runner's compact client order id.

    Format: rpf_<epoch>_<sleeve_sha6>_<signal_sha8>.  The digest is enough to
    recover the originating sleeve/signal from local config history after a
    process restart, without storing secrets in the order id.
    """

    parts = str(client_order_id or "").split("_")
    if len(parts) < 4 or parts[0] != PORTFOLIO_ORDER_PREFIX:
        return None
    return {"epoch": parts[1], "sleeve_key": parts[2], "signal_digest": parts[3]}


def _load_sleeve_runtime_spec(sleeve: dict[str, Any]) -> dict[str, Any]:
    """Best-effort runtime metadata for a configured sleeve."""

    source = str(sleeve.get("source") or sleeve.get("source_predictions") or "")
    hold = int(sleeve.get("hold_bars", sleeve.get("hold_bars_5m", 0)) or 0)
    stride = int(sleeve.get("stride_bars", sleeve.get("stride_bars_5m", 1)) or 1)
    dynamic_exit = sleeve.get("dynamic_exit") if isinstance(sleeve.get("dynamic_exit"), dict) else None
    if source.endswith(".json") and Path(source).exists():
        try:
            cfg = _load_json(source)
            if isinstance(cfg.get("dynamic_exit"), dict):
                dynamic_exit = cfg["dynamic_exit"]
            if "base_candidate" in cfg and "gates" not in cfg and "signal" not in cfg:
                cfg = _load_json(str(cfg["base_candidate"]))
                if dynamic_exit is None and isinstance(cfg.get("dynamic_exit"), dict):
                    dynamic_exit = cfg["dynamic_exit"]
            if "signal" in cfg:
                hold = int(cfg["signal"].get("hold_bars_5m", cfg["signal"].get("hold_bars", hold)) or hold)
                stride = int(cfg["signal"].get("stride_bars_5m", cfg["signal"].get("stride_bars", stride)) or stride)
            else:
                hold = int(cfg.get("hold_bars", cfg.get("hold_bars_5m", hold)) or hold)
                stride = int(cfg.get("stride_bars", cfg.get("stride_bars_5m", stride)) or stride)
        except Exception:
            pass
    return {"source": source, "hold_bars": hold, "stride_bars": stride, "dynamic_exit": dynamic_exit}


def _known_portfolio_sleeves(portfolio: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index current and historical live portfolio sleeves by name.

    This lets a newly-started process recover positions opened by the previous
    gross/leverage mix, because Binance only retains the compact sleeve hash in
    clientOrderId.
    """

    out: dict[str, dict[str, Any]] = {}

    def add(sleeve: dict[str, Any]) -> None:
        name = str(sleeve.get("name") or "")
        if not name:
            return
        out.setdefault(
            name,
            {
                "name": name,
                "source": str(sleeve.get("source") or sleeve.get("source_predictions") or ""),
                "side": str(sleeve.get("side", "")).upper(),
                "weight": float(sleeve.get("weight", 0.0) or 0.0),
                **_load_sleeve_runtime_spec(sleeve),
            },
        )

    for sleeve in portfolio.get("base_sleeves", []):
        add(sleeve)
    for path in Path("configs/live").glob("portfolio*.json"):
        try:
            cfg = _load_json(path)
        except Exception:
            continue
        for sleeve in cfg.get("base_sleeves", []):
            add(sleeve)
    return out


def _summarize_exchange_trade_fills(fills: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a queryable close report from Binance user-trade fills."""

    quantity = Decimal("0")
    quote = Decimal("0")
    realized = Decimal("0")
    commission = Decimal("0")
    commission_assets: set[str] = set()
    order_ids: list[str] = []
    times: list[int] = []
    for fill in fills:
        qty = Decimal(str(fill.get("qty", "0") or "0"))
        price = Decimal(str(fill.get("price", "0") or "0"))
        quantity += qty
        quote += Decimal(str(fill.get("quoteQty", "0") or "0")) or qty * price
        realized += Decimal(str(fill.get("realizedPnl", "0") or "0"))
        fee = Decimal(str(fill.get("commission", "0") or "0"))
        commission += fee
        asset = str(fill.get("commissionAsset") or "")
        if asset:
            commission_assets.add(asset)
        order_id = fill.get("orderId")
        if order_id is not None and str(order_id) not in order_ids:
            order_ids.append(str(order_id))
        if fill.get("time") is not None:
            times.append(int(fill["time"]))
    avg_price = quote / quantity if quantity > 0 else Decimal("0")
    commission_asset = ",".join(sorted(commission_assets)) or None
    quote_fee = commission if not commission_assets or commission_assets <= {"USDT", "USDC", "BUSD"} else Decimal("0")
    return {
        "fill_count": len(fills),
        "quantity": str(quantity),
        "quote_quantity": str(quote),
        "avg_price": str(avg_price),
        "realized_pnl": str(realized),
        "commission": str(commission),
        "commission_asset": commission_asset,
        "net_realized_pnl": str(realized - quote_fee),
        "order_ids": order_ids,
        "first_fill_at": None if not times else str(pd.Timestamp(min(times), unit="ms", tz="UTC")),
        "last_fill_at": None if not times else str(pd.Timestamp(max(times), unit="ms", tz="UTC")),
        "fills": fills,
    }


async def _attach_exchange_trade_report(
    *, client: Any, symbol: str, order_info: dict[str, Any]
) -> dict[str, Any]:
    """Attach actual fill PnL/fees to a submitted close order result."""

    order_ids = {
        str(order_id)
        for order_id in [order_info.get("order_id"), *(o.get("orderId") for o in (order_info.get("raw_orders") or []) if isinstance(o, dict))]
        if order_id is not None
    }
    if not order_ids:
        return order_info
    fills: list[dict[str, Any]] = []
    last_error: str | None = None
    for attempt in range(5):
        try:
            trades = await client.get_trades(symbol, limit=1000)
            fills = [t for t in trades if str(t.get("orderId")) in order_ids]
        except Exception as exc:
            last_error = str(exc)
        if fills:
            break
        if attempt < 4:
            await asyncio.sleep(0.2)
    if not fills:
        return {**order_info, "trade_report_error": last_error or "no_user_trade_fills_for_close_order"}
    return {**order_info, "trade_report": _summarize_exchange_trade_fills(fills)}


async def _reconcile_exchange_flat_sleeves(
    *, state: dict[str, Any], client: Any, exec_cfg: WaveExecutionConfig
) -> list[dict[str, Any]]:
    """Report state-owned sleeves that became flat while the runner was down.

    Reconciliation is deliberately limited to a completely flat Binance hedge
    side.  A smaller aggregate position is ambiguous when multiple sleeves
    share LONG or SHORT, so it must not silently close one sleeve in state.
    """

    try:
        positions = await client.get_positions(exec_cfg.symbol)
    except TypeError:
        positions = await client.get_positions()
    except Exception as exc:
        state["last_position_reconcile_error"] = str(exc)
        return []
    active_sides: set[str] = set()
    for pos in positions:
        try:
            qty = abs(Decimal(str(pos.get("positionAmt", "0") or "0")))
        except Exception:
            qty = Decimal("0")
        if qty > 0:
            side = str(pos.get("positionSide") or "").upper()
            if side in {"LONG", "SHORT"}:
                active_sides.add(side)
    candidates = [
        (name, sleeve)
        for name, sleeve in state.setdefault("open_sleeves", {}).items()
        if str(sleeve.get("side", "")).upper() in {"LONG", "SHORT"}
        and str(sleeve.get("side", "")).upper() not in active_sides
    ]
    if not candidates:
        return []
    try:
        trades = await client.get_trades(exec_cfg.symbol, limit=1000)
    except Exception:
        trades = []
    reconciled: list[dict[str, Any]] = []
    for name, sleeve in candidates:
        side = str(sleeve["side"]).upper()
        close_side = "SELL" if side == "LONG" else "BUY"
        signal_id = str(sleeve.get("signal_id") or "")
        signal_ts = pd.Timestamp(sleeve.get("signal_date") or 0)
        if signal_ts.tzinfo is None:
            signal_ts = signal_ts.tz_localize("UTC")
        start_ms = int(signal_ts.timestamp() * 1000)
        possible = [
            t
            for t in trades
            if str(t.get("positionSide", "")).upper() == side
            and str(t.get("side", "")).upper() == close_side
            and int(t.get("time", 0) or 0) >= start_ms
        ]
        expected_key = _portfolio_sleeve_key(name)
        expected_digest = hashlib.sha1(signal_id.encode("utf-8")).hexdigest()[:8]
        attributed: list[dict[str, Any]] = []
        client_order_ids: list[str] = []
        for order_id in dict.fromkeys(t.get("orderId") for t in possible if t.get("orderId") is not None):
            try:
                order = await client.get_order(exec_cfg.symbol, order_id=order_id)
            except Exception:
                continue
            cid = str(order.get("clientOrderId") or "")
            parts = _portfolio_order_parts(cid)
            if parts and parts["sleeve_key"] == expected_key and parts["signal_digest"] == expected_digest:
                attributed.extend(t for t in possible if t.get("orderId") == order_id)
                client_order_ids.append(cid)
        report = _summarize_exchange_trade_fills(attributed) if attributed else None
        requested = str(sleeve.get("quantity", "0") or "0")
        order_info = {
            "status": "FILLED_RECONCILED" if report else "EXCHANGE_FLAT_UNATTRIBUTED",
            "requested_quantity": requested,
            "filled_quantity": report["quantity"] if report else requested,
            "avg_price": report["avg_price"] if report else None,
            "order_id": ",".join(report["order_ids"]) if report else None,
            "client_order_id": ",".join(client_order_ids) or None,
            "started_at": report["first_fill_at"] if report else None,
            "finished_at": report["last_fill_at"] if report else str(pd.Timestamp.utcnow()),
            "trade_report": report,
            "reconciliation": {
                "reason": "state_open_but_exchange_side_flat",
                "state_exit_at": sleeve.get("exit_at"),
                "dynamic_exit": sleeve.get("dynamic_exit"),
                "attributed_by_client_order_id": bool(report),
            },
        }
        reconciled.append({"name": name, "side": side, "signal_id": signal_id, "order_info": order_info})
    return reconciled


def _infer_signal_id_from_digest(
    *,
    sleeve_name: str,
    digest: str,
    order_time_ms: int | None,
    interval_minutes: int,
    search_bars: int = 48,
) -> tuple[str | None, pd.Timestamp | None]:
    if not order_time_ms:
        return None, None
    order_ts = pd.Timestamp(int(order_time_ms), unit="ms", tz="UTC").floor(f"{int(interval_minutes)}min")
    for offset in range(-search_bars, search_bars + 1):
        ts = order_ts + pd.Timedelta(minutes=int(interval_minutes) * offset)
        naive = ts.tz_convert(None)
        for stamp in (naive.isoformat(), str(naive)):
            signal_id = f"{sleeve_name}:{stamp}"
            if hashlib.sha1(signal_id.encode("utf-8")).hexdigest()[:8] == digest:
                return signal_id, ts
            # New auto-direction signal ids include the effective side between
            # name and timestamp.  Historical orders did not, but support both.
            for side in ("LONG", "SHORT"):
                signal_id = f"{sleeve_name}:{side}:{stamp}"
                if hashlib.sha1(signal_id.encode("utf-8")).hexdigest()[:8] == digest:
                    return signal_id, ts
    return None, None


async def _recover_exchange_positions_into_state(
    *,
    state: dict[str, Any],
    client: Any,
    exec_cfg: WaveExecutionConfig,
    portfolio: dict[str, Any],
    leverage_budget: float,
    allocation_mode: str,
) -> list[dict[str, Any]]:
    """Import live Binance positions opened by this runner but missing in state.

    State can be lost across reboots or strategy switches.  If an exchange
    position remains open, reconstruct the sleeve from the rpf_* client order id
    so normal exit-at handling continues instead of orphaning the position.
    """

    recovered: list[dict[str, Any]] = []
    known_by_name = _known_portfolio_sleeves(portfolio)
    known_by_key = {_portfolio_sleeve_key(name): spec for name, spec in known_by_name.items()}
    open_sleeves = state.setdefault("open_sleeves", {})
    try:
        positions = await client.get_positions(exec_cfg.symbol)
    except TypeError:
        positions = await client.get_positions()
    except Exception as exc:
        state["last_position_recovery_error"] = str(exc)
        return recovered
    active_positions = []
    for pos in positions:
        try:
            qty = abs(Decimal(str(pos.get("positionAmt", "0") or "0")))
        except Exception:
            qty = Decimal("0")
        if qty <= Decimal("0"):
            continue
        side = str(pos.get("positionSide") or "").upper()
        if side not in {"LONG", "SHORT"}:
            try:
                side = "LONG" if Decimal(str(pos.get("positionAmt", "0"))) > 0 else "SHORT"
            except Exception:
                side = "LONG"
        active_positions.append((pos, side, qty))
    if not active_positions:
        return recovered
    existing_sides = {str(v.get("side", "")).upper() for v in open_sleeves.values() if Decimal(str(v.get("quantity", "0") or "0")) > 0}
    try:
        trades = await client._private_request("GET", "/fapi/v1/userTrades", {"symbol": exec_cfg.symbol, "limit": 100})
    except Exception:
        trades = []
    for pos, side, qty in active_positions:
        if side in existing_sides:
            continue
        open_side = "BUY" if side == "LONG" else "SELL"
        close_side = "SELL" if side == "LONG" else "BUY"
        side_trades = [t for t in trades if str(t.get("positionSide", "")).upper() == side]
        last_close_ms = max((int(t.get("time", 0) or 0) for t in side_trades if str(t.get("side", "")).upper() == close_side), default=0)
        entries = [t for t in side_trades if str(t.get("side", "")).upper() == open_side and int(t.get("time", 0) or 0) >= last_close_ms]
        if not entries:
            continue
        entry = max(entries, key=lambda t: int(t.get("time", 0) or 0))
        order_id = entry.get("orderId")
        order: dict[str, Any] = {}
        if order_id is not None:
            try:
                order = await client.get_order(exec_cfg.symbol, order_id=order_id)
            except Exception:
                try:
                    order = await client._private_request("GET", "/fapi/v1/order", {"symbol": exec_cfg.symbol, "orderId": order_id})
                except Exception:
                    order = {}
        cid = str(order.get("clientOrderId") or "")
        parts = _portfolio_order_parts(cid)
        if parts is None:
            continue
        spec = known_by_key.get(parts["sleeve_key"])
        if spec is None:
            continue
        name = str(spec["name"])
        if name in open_sleeves:
            continue
        signal_id, signal_ts = _infer_signal_id_from_digest(
            sleeve_name=name,
            digest=parts["signal_digest"],
            order_time_ms=int(order.get("time", entry.get("time", 0)) or 0),
            interval_minutes=exec_cfg.interval_minutes,
        )
        if signal_ts is None:
            signal_ts = pd.Timestamp(int(entry.get("time", 0) or 0), unit="ms", tz="UTC").floor(f"{int(exec_cfg.interval_minutes)}min")
        if signal_id is None:
            signal_id = f"{name}:{side}:recovered:{order_id}"
        hold_bars = int(spec.get("hold_bars") or exec_cfg.max_holding_bars)
        exit_at = signal_ts + pd.Timedelta(minutes=exec_cfg.interval_minutes * (1 + hold_bars))
        total_weight = float(sum(float(s.get("weight", 0.0) or 0.0) for s in portfolio.get("base_sleeves", [])))
        weight = float(spec.get("weight") or 0.0)
        margin_fraction = _margin_fraction_for_weight(
            weight=weight,
            total_weight=total_weight,
            leverage_budget=float(leverage_budget),
            allocation_mode=allocation_mode,
        ) if weight > 0 else 0.0
        open_sleeves[name] = {
            "name": name,
            "side": side,
            "signal_id": signal_id,
            "signal_date": str(signal_ts.tz_convert(None)),
            "exit_at": str(exit_at),
            "weight": weight,
            "margin_fraction": margin_fraction,
            "allocation_mode": allocation_mode,
            "quantity": str(qty),
            "order_info": {
                "status": "RECOVERED_FROM_EXCHANGE",
                "order_id": order_id,
                "client_order_id": cid,
                "entry_trade_time": entry.get("time"),
                "entry_price": entry.get("price", pos.get("entryPrice")),
                "position": {k: pos.get(k) for k in ["symbol", "positionSide", "positionAmt", "entryPrice", "updateTime"]},
            },
            "dynamic_exit": spec.get("dynamic_exit"),
            "recovered_from_exchange": True,
        }
        state.setdefault("processed_signals", {})[name] = signal_id
        rec = {"name": name, "side": side, "quantity": str(qty), "signal_id": signal_id, "exit_at": str(exit_at), "order_id": order_id}
        recovered.append(rec)
    if recovered:
        history = list(state.get("exchange_position_recovery_history", []))
        history.extend({**r, "recovered_at": str(pd.Timestamp.utcnow())} for r in recovered)
        state["exchange_position_recovery_history"] = history[-100:]
        state["last_exchange_position_recovery"] = recovered
    return recovered


def _order_timestamp_ms(order: dict[str, Any]) -> int | None:
    for key in ("time", "updateTime", "workingTime"):
        value = order.get(key)
        if value in (None, ""):
            continue
        try:
            value_i = int(value)
        except (TypeError, ValueError):
            continue
        if value_i > 0:
            return value_i
    cid = str(order.get("clientOrderId") or order.get("origClientOrderId") or "")
    parts = cid.split("_")
    if len(parts) >= 2 and parts[0] == PORTFOLIO_ORDER_PREFIX:
        try:
            return int(parts[1]) * 1000
        except ValueError:
            return None
    return None


async def _cancel_stale_portfolio_orders(
    *,
    client: Any,
    symbol: str,
    now: pd.Timestamp,
    max_age_sec: int,
) -> list[dict[str, Any]]:
    """Cancel only this runner's stale post-only orders.

    Other live bots can share BTCUSDT, so cleanup is restricted to the
    client-order-id prefix generated by this module.
    """

    if max_age_sec <= 0:
        return []
    now_ms = int(pd.Timestamp(now).timestamp() * 1000)
    cancelled: list[dict[str, Any]] = []
    try:
        open_orders = await client.get_open_orders(symbol)
    except Exception as exc:
        return [{"status": "scan_failed", "error": str(exc)}]
    for order in open_orders:
        cid = str(order.get("clientOrderId") or "")
        if not cid.startswith(PORTFOLIO_ORDER_PREFIX + "_"):
            continue
        ts_ms = _order_timestamp_ms(order)
        if ts_ms is None or (now_ms - ts_ms) < int(max_age_sec) * 1000:
            continue
        try:
            result = await client.cancel_order(symbol, client_order_id=cid)
            cancelled.append(
                {
                    "status": "cancelled",
                    "order_id": order.get("orderId"),
                    "client_order_id": cid,
                    "age_sec": round((now_ms - ts_ms) / 1000, 3),
                    "result": result,
                }
            )
        except Exception as exc:
            if "-2011" not in str(exc):
                cancelled.append(
                    {
                        "status": "cancel_failed",
                        "order_id": order.get("orderId"),
                        "client_order_id": cid,
                        "age_sec": round((now_ms - ts_ms) / 1000, 3),
                        "error": str(exc),
                    }
                )
    return cancelled


async def _cancel_portfolio_orders_for_sleeve(
    *,
    client: Any,
    symbol: str,
    sleeve_name: str,
    reason: str,
) -> list[dict[str, Any]]:
    """Cancel this runner's open entry orders for one sleeve before replacing them."""

    sleeve_key = _portfolio_sleeve_key(sleeve_name)
    target_prefix = f"{PORTFOLIO_ORDER_PREFIX}_"
    cancelled: list[dict[str, Any]] = []
    try:
        open_orders = await client.get_open_orders(symbol)
    except Exception as exc:
        return [{"status": "scan_failed", "sleeve": sleeve_name, "reason": reason, "error": str(exc)}]
    for order in open_orders:
        cid = str(order.get("clientOrderId") or "")
        parts = cid.split("_")
        if not (cid.startswith(target_prefix) and len(parts) >= 4 and parts[2] == sleeve_key):
            continue
        try:
            result = await client.cancel_order(symbol, client_order_id=cid)
            cancelled.append(
                {
                    "status": "cancelled",
                    "reason": reason,
                    "sleeve": sleeve_name,
                    "order_id": order.get("orderId"),
                    "client_order_id": cid,
                    "result": result,
                }
            )
        except Exception as exc:
            if "-2011" not in str(exc):
                cancelled.append(
                    {
                        "status": "cancel_failed",
                        "reason": reason,
                        "sleeve": sleeve_name,
                        "order_id": order.get("orderId"),
                        "client_order_id": cid,
                        "error": str(exc),
                    }
                )
    return cancelled


async def _place_portfolio_maker_order_with_deadline(
    *,
    client: Any,
    executor: Any,
    exec_cfg: WaveExecutionConfig,
    order_side: Literal["BUY", "SELL"],
    quantity: Decimal,
    position_side: Side,
    signal_id: str,
    sleeve_name: str,
    ttl_sec: int,
    reduce_only: bool = False,
    reference_price: float | None = None,
    max_deviation_pct: float | None = None,
    refresh_interval_sec: int = 60,
    poll_interval_sec: float = 1.0,
) -> dict[str, Any]:
    """Place a post-only order and refresh it on a bounded one-minute cadence."""

    started = pd.Timestamp.utcnow()
    deadline = asyncio.get_event_loop().time() + max(0, int(ttl_sec))
    remaining_qty = Decimal(str(quantity))
    total_filled = Decimal("0")
    avg_price = Decimal("0")
    active_order_id: Any = None
    active_client_order_id = ""
    active_price = Decimal("0")
    raw_orders: list[dict[str, Any]] = []
    refresh_history: list[dict[str, Any]] = []
    last_status: dict[str, Any] = {}
    active_counted_executed = Decimal("0")
    terminal_statuses = {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}
    min_qty = Decimal("0.001")

    def _order_executed_qty(order: dict[str, Any] | None) -> Decimal:
        if not isinstance(order, dict):
            return Decimal("0")
        for key in ("executedQty", "cumQty", "filled_quantity", "filledQuantity"):
            value = order.get(key)
            if value in (None, ""):
                continue
            try:
                return max(Decimal("0"), Decimal(str(value)))
            except Exception:
                continue
        return Decimal("0")

    def _order_avg_price(order: dict[str, Any] | None) -> Decimal:
        if not isinstance(order, dict):
            return Decimal("0")
        for key in ("avgPrice", "avg_price", "price"):
            value = order.get(key)
            if value in (None, ""):
                continue
            try:
                price = Decimal(str(value))
            except Exception:
                continue
            if price > 0:
                return price
        return Decimal("0")

    def reconcile_active_fill(order: dict[str, Any] | None, reason: str) -> Decimal:
        """Apply newly filled quantity for the currently active order once.

        Binance cancel responses can report a larger executedQty than the last
        order-status poll when a maker order fills while it is being cancelled.
        Refreshing with the pre-cancel remaining quantity would over-order, so
        every terminal/cancel path reconciles the cumulative executed quantity
        before placing the next chasing order.
        """

        nonlocal active_counted_executed, total_filled, remaining_qty, avg_price
        observed = _order_executed_qty(order)
        if observed <= active_counted_executed:
            return Decimal("0")
        delta = observed - active_counted_executed
        active_counted_executed = observed
        total_filled += delta
        remaining_qty = max(Decimal("0"), remaining_qty - delta)
        reported_avg = _order_avg_price(order)
        if reported_avg > 0:
            avg_price = reported_avg
        refresh_history.append(
            {
                "action": "fill_reconciled",
                "reason": reason,
                "order_id": active_order_id,
                "client_order_id": active_client_order_id,
                "newly_filled_qty": str(delta),
                "observed_executed_qty": str(observed),
                "remaining_qty": str(remaining_qty),
                "at": str(pd.Timestamp.utcnow()),
            }
        )
        return delta

    def deviation_ok(price: float) -> tuple[bool, float | None]:
        if reference_price is None or max_deviation_pct is None or max_deviation_pct <= 0:
            return True, None
        ref = float(reference_price)
        if ref <= 0:
            return True, None
        deviation = abs(float(price) / ref - 1.0)
        return deviation <= float(max_deviation_pct), deviation

    async def place_new(reason: str) -> bool:
        nonlocal active_order_id, active_client_order_id, active_price, last_status, active_counted_executed
        if remaining_qty < min_qty:
            refresh_history.append({"action": "skip_place_min_remaining", "reason": reason, "remaining_qty": str(remaining_qty), "at": str(pd.Timestamp.utcnow())})
            return False
        maker_price = float(await executor.get_maker_price(order_side, None))
        ok, deviation = deviation_ok(maker_price)
        if not ok:
            refresh_history.append({"action": "skip_place_deviation", "reason": reason, "price": maker_price, "reference_price": reference_price, "deviation_pct": deviation, "max_deviation_pct": max_deviation_pct, "at": str(pd.Timestamp.utcnow())})
            return False
        cid = _portfolio_client_order_id(signal_id, sleeve_name=sleeve_name)
        try:
            order = await client.place_order(
                symbol=exec_cfg.symbol,
                side=order_side,
                order_type="LIMIT",
                quantity=float(remaining_qty),
                price=float(maker_price),
                time_in_force="GTX",
                reduce_only=bool(reduce_only),
                client_order_id=cid,
                position_side=position_side,
            )
        except Exception as exc:
            refresh_history.append({"action": "place_rejected", "reason": reason, "client_order_id": cid, "price": maker_price, "reference_price": reference_price, "deviation_pct": deviation, "error": str(exc), "at": str(pd.Timestamp.utcnow())})
            return False
        active_order_id = order.get("orderId")
        active_client_order_id = cid
        active_price = Decimal(str(maker_price))
        active_counted_executed = Decimal("0")
        last_status = dict(order)
        raw_orders.append(order)
        refresh_history.append({"action": "placed" if reason == "initial" else "replaced", "reason": reason, "order_id": active_order_id, "client_order_id": cid, "price": str(active_price), "quantity": str(remaining_qty), "reference_price": reference_price, "deviation_pct": deviation, "at": str(pd.Timestamp.utcnow())})
        return True

    if not await place_new("initial"):
        finished = pd.Timestamp.utcnow()
        return {"status": "REJECTED_DEVIATION" if refresh_history and refresh_history[-1]["action"] == "skip_place_deviation" else "REJECTED", "client_order_id": active_client_order_id, "requested_quantity": str(quantity), "filled_quantity": "0", "avg_price": "0", "price": "0", "ttl_sec": int(ttl_sec), "started_at": str(started), "deadline_at": str(started + pd.Timedelta(seconds=int(ttl_sec))), "finished_at": str(finished), "wall_time_sec": float((finished - started).total_seconds()), "refresh_history": refresh_history}

    last_refresh = asyncio.get_event_loop().time()
    final_status = "UNKNOWN"
    cancel_result: dict[str, Any] | None = None

    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(max(0.05, float(poll_interval_sec)))
        if active_order_id is None:
            if asyncio.get_event_loop().time() - last_refresh >= max(1, int(refresh_interval_sec)):
                await place_new("refresh_after_deviation_skip")
                last_refresh = asyncio.get_event_loop().time()
            continue
        try:
            last_status = await client.get_order(exec_cfg.symbol, order_id=active_order_id)
        except Exception as exc:
            if "-2013" in str(exc):
                last_status = {**last_status, "status": "UNKNOWN", "query_error": str(exc)}
                break
            last_status = {**last_status, "query_error": str(exc)}
        status = str(last_status.get("status") or "UNKNOWN")
        observed_executed = _order_executed_qty(last_status)
        reported_avg = _order_avg_price(last_status)
        if reported_avg > 0:
            avg_price = reported_avg
        if status == "FILLED":
            reconcile_active_fill(last_status, "status_filled")
            final_status = "FILLED"
            break
        if status in terminal_statuses:
            reconcile_active_fill(last_status, f"terminal_{status.lower()}")
            final_status = status
            if remaining_qty < min_qty:
                break
            if asyncio.get_event_loop().time() < deadline:
                active_order_id = None
                active_client_order_id = ""
                await place_new(f"terminal_{status.lower()}")
                last_refresh = asyncio.get_event_loop().time()
                continue
            break

        if asyncio.get_event_loop().time() - last_refresh >= max(1, int(refresh_interval_sec)):
            reconcile_active_fill(last_status, "before_refresh_cancel")
            if remaining_qty < min_qty:
                final_status = "FILLED" if remaining_qty <= 0 else "PARTIAL_FILLED_MIN_REMAINING" if total_filled > 0 else status
                break
            try:
                cancel_result = await client.cancel_order(exec_cfg.symbol, client_order_id=active_client_order_id)
                reconcile_active_fill(cancel_result, "after_refresh_cancel")
                refresh_history.append({"action": "cancel_for_refresh", "order_id": active_order_id, "client_order_id": active_client_order_id, "executed_qty_before_cancel": str(observed_executed), "executed_qty_after_cancel": str(_order_executed_qty(cancel_result)), "remaining_qty": str(remaining_qty), "result": cancel_result, "at": str(pd.Timestamp.utcnow())})
            except Exception as exc:
                if "-2011" not in str(exc):
                    refresh_history.append({"action": "cancel_for_refresh_failed", "order_id": active_order_id, "client_order_id": active_client_order_id, "error": str(exc), "remaining_qty": str(remaining_qty), "at": str(pd.Timestamp.utcnow())})
            active_order_id = None
            active_client_order_id = ""
            if remaining_qty < min_qty:
                final_status = "FILLED" if remaining_qty <= 0 else "PARTIAL_FILLED_MIN_REMAINING" if total_filled > 0 else status
                break
            await place_new("refresh_60s")
            last_refresh = asyncio.get_event_loop().time()

    if final_status == "UNKNOWN":
        final_status = str(last_status.get("status") or "UNKNOWN")
    if active_client_order_id and final_status not in {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}:
        reconcile_active_fill(last_status, "before_timeout_cancel")
        try:
            cancel_result = await client.cancel_order(exec_cfg.symbol, client_order_id=active_client_order_id)
            reconcile_active_fill(cancel_result, "after_timeout_cancel")
            if remaining_qty < min_qty:
                final_status = "FILLED"
            else:
                final_status = "TIMEOUT_CANCELLED" if total_filled <= 0 else "PARTIAL_CANCELLED"
        except Exception as exc:
            if "-2011" not in str(exc):
                cancel_result = {"status": "cancel_failed", "error": str(exc)}
    if avg_price <= 0 and active_price > 0:
        avg_price = active_price
    finished = pd.Timestamp.utcnow()

    return {"status": final_status, "order_id": active_order_id, "client_order_id": active_client_order_id, "requested_quantity": str(quantity), "filled_quantity": str(total_filled), "avg_price": str(avg_price), "price": str(active_price), "ttl_sec": int(ttl_sec), "refresh_interval_sec": int(refresh_interval_sec), "reference_price": reference_price, "max_deviation_pct": max_deviation_pct, "started_at": str(started), "deadline_at": str(started + pd.Timedelta(seconds=int(ttl_sec))), "finished_at": str(finished), "wall_time_sec": float((finished - started).total_seconds()), "cancel_result": cancel_result, "raw_order": raw_orders[0] if raw_orders else {}, "raw_orders": raw_orders, "last_status": last_status, "refresh_history": refresh_history}


def _margin_fraction_for_weight(
    *,
    weight: float,
    total_weight: float,
    leverage_budget: float,
    allocation_mode: str,
) -> float:
    """Return account-equity margin fraction for a sleeve weight.

    Research portfolio metrics apply sleeve weights directly as notional
    exposure per 1.0 account equity.  On an exchange account with leverage L,
    matching that assumption requires margin_fraction = weight / L.

    ``normalize_weights`` intentionally scales all weights to consume the full
    leverage budget even when gross_weight < leverage_budget; this is more
    aggressive than the saved research candidate.
    """

    if weight < 0:
        raise ValueError("sleeve weight must be non-negative")
    if leverage_budget <= 0:
        raise ValueError("leverage_budget must be positive")
    if allocation_mode == "research_gross":
        return float(weight) / float(leverage_budget)
    if allocation_mode == "normalize_weights":
        if total_weight <= 0:
            raise ValueError("total_weight must be positive for normalize_weights")
        return float(weight) / float(total_weight)
    raise ValueError(f"unsupported allocation_mode: {allocation_mode}")


def _allocation_audit(portfolio: dict[str, Any], *, leverage_budget: float, allocation_mode: str) -> dict[str, Any]:
    sleeves = portfolio.get("base_sleeves", [])
    total_weight = float(sum(float(s["weight"]) for s in sleeves))
    rows = []
    for sleeve in sleeves:
        w = float(sleeve["weight"])
        mf = _margin_fraction_for_weight(
            weight=w,
            total_weight=total_weight,
            leverage_budget=leverage_budget,
            allocation_mode=allocation_mode,
        )
        rows.append(
            {
                "name": sleeve["name"],
                "side": sleeve["side"],
                "research_weight_notional_per_equity": w,
                "margin_fraction": mf,
                "live_notional_per_equity_at_budget": mf * float(leverage_budget),
                "research_match": abs((mf * float(leverage_budget)) - w) < 1e-9,
            }
        )
    return {
        "allocation_mode": allocation_mode,
        "leverage_budget": float(leverage_budget),
        "research_gross_weight": total_weight,
        "margin_fraction_sum_if_all_active": float(sum(r["margin_fraction"] for r in rows)),
        "live_gross_if_all_active": float(sum(r["live_notional_per_equity_at_budget"] for r in rows)),
        "unused_margin_fraction_vs_full_budget": max(0.0, 1.0 - float(sum(r["margin_fraction"] for r in rows))),
        "sleeves": rows,
        "notes": [
            "research_gross mode matches saved research weights exactly and leaves unused leverage capacity when gross_weight < leverage_budget",
            "normalize_weights consumes 100% margin but scales risk/return versus research when gross_weight != leverage_budget",
        ],
    }


async def _make_executor(exec_cfg: WaveExecutionConfig):
    key, secret = _load_api_credentials(exec_cfg.testnet, exec_cfg.wave_trading_path, dry_run=exec_cfg.dry_run)
    client_cls, executor_cls = load_wave_execution_classes(exec_cfg.wave_trading_path)
    client = client_cls(api_key=key, api_secret=secret, testnet=exec_cfg.testnet)
    signal_generator = _StaticSignalGenerator(
        atr_period=exec_cfg.atr_period,
        pt_mult=exec_cfg.pt_mult,
        max_holding_bars=exec_cfg.max_holding_bars,
    )
    executor = executor_cls(
        client=client,
        signal_generator=signal_generator,
        symbol=exec_cfg.symbol,
        leverage=exec_cfg.leverage,
        position_size_pct=1.0,
        maker_offset_pct=exec_cfg.maker_offset_pct,
        max_retries=exec_cfg.max_retries,
        order_timeout_sec=exec_cfg.order_timeout_sec,
    )
    await client.sync_time()
    if not await client.is_hedge_mode(force_refresh=True):
        raise RuntimeError("Binance account must be in hedge mode for distributed LONG/SHORT portfolio execution")
    await client.set_leverage(exec_cfg.symbol, exec_cfg.leverage)
    return client, executor


async def _open_sleeve(
    *,
    client: Any,
    executor: Any,
    exec_cfg: WaveExecutionConfig,
    sleeve: dict[str, Any],
    margin_fraction: float,
    entry_ttl_sec: int,
) -> dict[str, Any]:
    balance = await client.get_usdt_balance()
    total_equity = float(balance["total"])
    price = await client.get_ticker_price(exec_cfg.symbol)
    notional = total_equity * float(margin_fraction) * float(exec_cfg.leverage)
    quantity = Decimal(str(notional / float(price)))
    side: Side = sleeve["side"]
    order_side = "BUY" if side == "LONG" else "SELL"
    order = await _place_portfolio_maker_order_with_deadline(
        client=client,
        executor=executor,
        exec_cfg=exec_cfg,
        order_side=order_side,
        quantity=quantity,
        position_side=side,
        signal_id=sleeve["signal_id"],
        sleeve_name=str(sleeve["name"]),
        ttl_sec=entry_ttl_sec,
        reference_price=float(sleeve.get("current_close", 0.0) or 0.0),
        max_deviation_pct=float(sleeve.get("entry_maker_max_deviation_pct", 0.0) or 0.0),
        refresh_interval_sec=int(sleeve.get("maker_refresh_interval_sec", 60) or 60),
    )
    return {
        "order": order,
        "requested_quantity": str(quantity),
        "filled_quantity": order.get("filled_quantity", "0"),
        "entry_ttl_sec": int(entry_ttl_sec),
        "notional": notional,
        "margin_fraction": margin_fraction,
        "equity_basis": total_equity,
    }

async def _close_sleeve(
    *,
    client: Any,
    executor: Any,
    sleeve_state: dict[str, Any],
    exec_cfg: WaveExecutionConfig,
    ttl_sec: int,
    reference_price: float,
    max_deviation_pct: float,
    refresh_interval_sec: int,
) -> dict[str, Any]:
    side: Side = sleeve_state["side"]
    close_side = "SELL" if side == "LONG" else "BUY"
    quantity = Decimal(str(sleeve_state["quantity"]))
    order = await _place_portfolio_maker_order_with_deadline(
        client=client,
        executor=executor,
        exec_cfg=exec_cfg,
        order_side=close_side,
        quantity=quantity,
        position_side=side,
        signal_id=str(sleeve_state.get("signal_id", "portfolio-exit")),
        sleeve_name=str(sleeve_state.get("name", "portfolio-exit")),
        ttl_sec=int(ttl_sec),
        reduce_only=True,
        reference_price=float(reference_price),
        max_deviation_pct=float(max_deviation_pct),
        refresh_interval_sec=int(refresh_interval_sec),
    )
    try:
        maker_filled = Decimal(str(order.get("filled_quantity", "0") or "0"))
    except Exception:
        maker_filled = Decimal("0")
    remaining = max(Decimal("0"), quantity - maker_filled)
    if remaining > Decimal("0"):
        taker_started = pd.Timestamp.utcnow()
        taker_order = await client.place_market(
            symbol=exec_cfg.symbol,
            side=close_side,
            quantity=float(remaining),
            reduce_only=True,
            position_side=side,
        )
        taker_finished = pd.Timestamp.utcnow()
        order["taker_fallback_order"] = taker_order
        order["taker_fallback_started_at"] = str(taker_started)
        order["taker_fallback_finished_at"] = str(taker_finished)
        order["taker_fallback_wall_time_sec"] = float((taker_finished - taker_started).total_seconds())
        order["taker_fallback_quantity"] = str(remaining)
        order["filled_quantity"] = str(quantity)
        order["status"] = "TAKER_FALLBACK_FILLED"
    return order



def _expected_decision_bar(now: pd.Timestamp, *, interval_minutes: int) -> pd.Timestamp:
    ts = pd.Timestamp(now)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    interval = f"{int(interval_minutes)}min"
    return ts.floor(interval) - pd.Timedelta(minutes=int(interval_minutes))


def _completed_decision_data_asof(expected_bar: pd.Timestamp, *, interval_minutes: int) -> pd.Timestamp:
    """Return the final 1m timestamp belonging to a completed decision bar.

    At 12:05, for example, the 12:00 five-minute candle is complete only after
    the 12:04 one-minute row commits.  Querying with wall-clock 12:05 may also
    include an already-started 12:05 row, which resampling would turn into an
    incomplete candle and accidentally score it.
    """

    bar = pd.Timestamp(expected_bar)
    if bar.tzinfo is None:
        bar = bar.tz_localize("UTC")
    else:
        bar = bar.tz_convert("UTC")
    return bar + pd.Timedelta(minutes=max(0, int(interval_minutes) - 1))


def _validate_pg_notify_channel(channel: str) -> str:
    name = str(channel or "").strip()
    if not name or not name.replace("_", "a").isalnum() or not name[0].isalpha():
        raise ValueError(f"invalid Postgres notify channel: {channel!r}")
    return name


def _latest_requirement_ts(engine_or_conn: Any, req: FreshnessRequirement) -> pd.Timestamp | None:
    from sqlalchemy import text

    allowed_tables = {"bars_binance", "bars_upbit", "bars_polygon", "bars_binance_premium", "open_interest_binance"}
    if req.table not in allowed_tables:
        raise ValueError(f"unsupported freshness table: {req.table}")
    if req.table == "open_interest_binance":
        sql = text(
            f"""
            SELECT MAX(ts) AS max_ts
            FROM {req.table}
            WHERE symbol = :symbol AND period = :period
            """
        )
        params = {"symbol": req.symbol, "period": req.period or "5m"}
    else:
        sql = text(
            f"""
            SELECT MAX(ts) AS max_ts
            FROM {req.table}
            WHERE symbol = :symbol AND interval = :interval
            """
        )
        params = {"symbol": req.symbol, "interval": req.interval or "1m"}
    if hasattr(engine_or_conn, "connect"):
        with engine_or_conn.connect() as conn:
            row = conn.execute(sql, params).mappings().one()
    else:
        row = engine_or_conn.execute(sql, params).mappings().one()
    value = row.get("max_ts")
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _latest_requirement_map(engine: Any, requirements: list[FreshnessRequirement]) -> dict[str, pd.Timestamp | None]:
    with engine.connect() as conn:
        return {req.key: _latest_requirement_ts(conn, req) for req in requirements}


def _freshness_missing(
    latest: dict[str, pd.Timestamp | None],
    requirements: list[FreshnessRequirement],
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for req in requirements:
        value = latest.get(req.key)
        if value is None or value < req.required_ts:
            missing.append(
                {
                    "key": req.key,
                    "table": req.table,
                    "source": req.source,
                    "symbol": req.symbol,
                    "interval": req.interval,
                    "period": req.period,
                    "required_ts": str(req.required_ts),
                    "latest_ts": None if value is None else str(value),
                }
            )
    return missing


def _freshness_requirements_for_decision(
    *,
    symbol: str,
    expected_bar: pd.Timestamp,
    required_1m: pd.Timestamp,
) -> list[FreshnessRequirement]:
    """Return source rows that must be current before opening a new live trade."""

    requirements = [
        FreshnessRequirement("bars_binance", symbol, "1m", required_1m, "binance_perp"),
        FreshnessRequirement("bars_binance_premium", symbol, "1m", required_1m, "binance_premium"),
        FreshnessRequirement("open_interest_binance", symbol, None, expected_bar, "binance_open_interest", period="5m"),
        FreshnessRequirement("bars_upbit", "KRW-BTC", "1m", required_1m, "upbit"),
        FreshnessRequirement("bars_polygon", "USDKRW", "1m", required_1m, "kimchi_usdkrw"),
    ]
    requirements.extend(
        FreshnessRequirement("bars_binance", alt_symbol, "1m", required_1m, "binance_alt_pool")
        for alt_symbol in ALT_POOL_SYMBOLS
    )
    return requirements


def _wait_for_source_updates_notify(
    engine: Any,
    *,
    symbol: str,
    expected_bar: pd.Timestamp,
    required_1m: pd.Timestamp,
    max_wait_sec: float,
    channel: str,
) -> tuple[float, dict[str, pd.Timestamp | None], list[dict[str, Any]]]:
    """Block until all feature-source requirements for a decision bar are committed."""

    deadline = time.monotonic() + max(0.0, float(max_wait_sec))
    started = deadline - max(0.0, float(max_wait_sec))
    requirements = _freshness_requirements_for_decision(symbol=symbol, expected_bar=expected_bar, required_1m=required_1m)
    latest: dict[str, pd.Timestamp | None] = {}
    raw = engine.raw_connection()
    try:
        dbapi_conn = getattr(raw, "driver_connection", None) or getattr(raw, "connection", raw)
        if hasattr(dbapi_conn, "autocommit"):
            dbapi_conn.autocommit = True
        cur = raw.cursor()
        channel = _validate_pg_notify_channel(channel)
        cur.execute(f"LISTEN {channel}")

        # Check after LISTEN to avoid missing a notification between the initial
        # schedule wake-up and listener registration.
        latest = _latest_requirement_map(engine, requirements)
        missing = _freshness_missing(latest, requirements)
        if not missing:
            return 0.0, latest, []

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            readable, _, _ = select.select([dbapi_conn], [], [], min(0.5, remaining))
            if readable:
                dbapi_conn.poll()
                while getattr(dbapi_conn, "notifies", []):
                    notify = dbapi_conn.notifies.pop(0)
                    try:
                        json.loads(notify.payload)
                    except Exception:
                        continue
            latest = _latest_requirement_map(engine, requirements)
            missing = _freshness_missing(latest, requirements)
            if not missing:
                return max(0.0, time.monotonic() - started), latest, []

        latest = _latest_requirement_map(engine, requirements)
        return max(0.0, time.monotonic() - started), latest, _freshness_missing(latest, requirements)
    finally:
        try:
            raw.close()
        except Exception:
            pass


async def _wait_for_expected_1m_tail(
    *,
    engine: Any,
    symbol: str,
    asof: pd.Timestamp,
    interval_minutes: int,
    max_wait_sec: float,
    notify_channel: str,
) -> tuple[float, pd.Timestamp | None, pd.Timestamp, str, dict[str, pd.Timestamp | None], list[dict[str, Any]]]:
    """Wait for ingest's Postgres notification for the expected decision bar."""

    expected_bar = _expected_decision_bar(asof, interval_minutes=interval_minutes)
    required_1m = expected_bar + pd.Timedelta(minutes=max(0, int(interval_minutes) - 1))
    waited, latest_map, missing = await asyncio.to_thread(
        _wait_for_source_updates_notify,
        engine,
        symbol=symbol,
        expected_bar=expected_bar,
        required_1m=required_1m,
        max_wait_sec=max_wait_sec,
        channel=notify_channel,
    )
    binance_key = f"bars_binance:{symbol}:1m"
    return waited, latest_map.get(binance_key), expected_bar, "notify" if not missing else "notify_timeout", latest_map, missing


async def run_portfolio_loop(cfg: PortfolioLiveConfig) -> None:
    portfolio = _load_json(cfg.portfolio_config)
    selector_overlay = _load_portfolio_selector_overlay(portfolio, cfg.portfolio_selector_overlay)
    rex_selector_cfg = RexLlmSelectorConfig(
        enabled=bool(cfg.rex_selector_adapter_dir),
        adapter_dir=str(cfg.rex_selector_adapter_dir or RexLlmSelectorConfig.adapter_dir),
        model_name=str(cfg.rex_selector_model_name),
        score_normalization=str(cfg.rex_selector_score_normalization),
        fail_closed=bool(cfg.rex_selector_fail_closed),
        require_cuda=bool(cfg.rex_selector_require_cuda),
    )
    exec_cfg_raw = WaveExecutionConfig.from_json(cfg.execution_config)
    exec_cfg = WaveExecutionConfig(
        **{
            **asdict(exec_cfg_raw),
            "dry_run": not cfg.live,
            "allow_live_orders": bool(cfg.allow_live_orders),
            "leverage": int(cfg.leverage),
            "position_size_pct": 1.0,
            "allowed_signals": ("LONG", "SHORT"),
            "require_flat_position": False,
            "require_no_open_orders": True,
        }
    )
    if cfg.live and not cfg.allow_live_orders:
        raise SystemExit("--live requires --allow-live-orders")
    weights = {str(s["name"]): float(s["weight"]) for s in portfolio["base_sleeves"]}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise RuntimeError("portfolio weights sum to zero")
    audit = _allocation_audit(portfolio, leverage_budget=float(cfg.leverage), allocation_mode=cfg.allocation_mode)

    engine = sqlalchemy_engine_from_env(cfg.env_path)
    _ensure_trade_executions_table(engine)
    source_cache = LiveSourceFrameCache()
    feature_cache = LiveFeatureFrameCache()
    oi_cache = LiveOiFrameCache()
    external_cache = LiveExternalFrameCache()
    alt_pool_cache = LiveAltPoolFrameCache()
    client = executor = None
    if not exec_cfg.dry_run:
        client, executor = await _make_executor(exec_cfg)
    first = True
    try:
        iterations = 0
        while True:
            if first and cfg.run_immediately:
                first = False
            else:
                first = False
                wait = _seconds_until_next_interval(pd.Timestamp.utcnow(), interval_minutes=exec_cfg.interval_minutes, close_delay_sec=cfg.close_delay_sec)
                _status(f"[portfolio-live] waiting {wait:.1f}s for next {exec_cfg.interval_minutes}m close")
                await asyncio.sleep(wait)

            asof = pd.Timestamp.utcnow()
            if asof.tzinfo is None:
                asof = asof.tz_localize("UTC")
            live_cfg = LiveDbFeatureConfig(lookback_minutes=int(cfg.lookback_minutes))
            freshness_waited, latest_1m_ts, expected_bar, freshness_mode, latest_source_ts, freshness_missing = await _wait_for_expected_1m_tail(
                engine=engine,
                symbol=live_cfg.symbol,
                asof=asof,
                interval_minutes=exec_cfg.interval_minutes,
                max_wait_sec=cfg.max_freshness_wait_sec,
                notify_channel=cfg.freshness_notify_channel,
            )
            decision_data_asof = _completed_decision_data_asof(
                expected_bar,
                interval_minutes=exec_cfg.interval_minutes,
            )
            frame_t0 = time.perf_counter()
            enriched, features = await build_live_portfolio_frames(
                engine=engine,
                asof=decision_data_asof,
                cfg=live_cfg,
                source_cache=source_cache,
                feature_cache=feature_cache,
                oi_cache=oi_cache,
                external_cache=external_cache,
                alt_pool_cache=alt_pool_cache,
            )
            frame_build_sec = time.perf_counter() - frame_t0
            sleeve_scores = _score_sleeves(
                portfolio=portfolio,
                enriched=enriched,
                features=features,
                exec_cfg=exec_cfg,
                asof=decision_data_asof,
                rex_selector_cfg=rex_selector_cfg,
            )
            latest_decision_bar = pd.Timestamp(enriched.iloc[-1]["date"])
            if latest_decision_bar.tzinfo is None:
                latest_decision_bar = latest_decision_bar.tz_localize("UTC")
            else:
                latest_decision_bar = latest_decision_bar.tz_convert("UTC")
            decision_bar_complete = latest_decision_bar == expected_bar
            data_fresh = not freshness_missing and decision_bar_complete
            if not data_fresh:
                for sleeve in sleeve_scores:
                    sleeve["active"] = False
                    if freshness_missing:
                        sleeve.setdefault("reasons", []).append(
                            "source_freshness=fail:" + ",".join(str(item["key"]) for item in freshness_missing[:5])
                        )
                    if not decision_bar_complete:
                        sleeve.setdefault("reasons", []).append(
                            f"completed_decision_bar=fail:expected={expected_bar.isoformat()},actual={latest_decision_bar.isoformat()}"
                        )
            portfolio_selector_record = _apply_portfolio_selector_overlay(
                sleeve_scores,
                overlay=selector_overlay,
                enriched=enriched,
                features=features,
            )
            state = _load_state(cfg.state_file)
            if portfolio_selector_record is not None:
                state["last_portfolio_selector"] = portfolio_selector_record
            now = pd.Timestamp.utcnow()
            stale_cancelled: list[dict[str, Any]] = []
            if cfg.cancel_stale_open_orders and not exec_cfg.dry_run:
                assert client is not None
                stale_cancelled = await _cancel_stale_portfolio_orders(
                    client=client,
                    symbol=exec_cfg.symbol,
                    now=now,
                    max_age_sec=int(cfg.max_entry_wait_sec),
                )
                if stale_cancelled:
                    state["last_stale_order_cancels"] = stale_cancelled
                    history = list(state.get("stale_order_cancel_history", []))
                    history.extend(stale_cancelled)
                    state["stale_order_cancel_history"] = history[-200:]

            recovered_positions: list[dict[str, Any]] = []
            reconciled_positions: list[dict[str, Any]] = []
            if not exec_cfg.dry_run:
                assert client is not None
                recovered_positions = await _recover_exchange_positions_into_state(
                    state=state,
                    client=client,
                    exec_cfg=exec_cfg,
                    portfolio=portfolio,
                    leverage_budget=float(cfg.leverage),
                    allocation_mode=cfg.allocation_mode,
                )
                for rec in recovered_positions:
                    _log_trade_execution(
                        engine,
                        strategy_name=cfg.strategy_name,
                        sub_strategy_name=str(rec.get("name")),
                        exchange=cfg.exchange,
                        symbol=exec_cfg.symbol,
                        action="RECOVER_POSITION",
                        side=str(rec.get("side")),
                        position_side=str(rec.get("side")),
                        order_type="RECOVERY",
                        signal_id=str(rec.get("signal_id")),
                        status="RECOVERED",
                        order_info=rec,
                    )
                reconciled_positions = await _reconcile_exchange_flat_sleeves(
                    state=state,
                    client=client,
                    exec_cfg=exec_cfg,
                )
                for rec in reconciled_positions:
                    info = rec["order_info"]
                    _log_trade_execution(
                        engine,
                        strategy_name=cfg.strategy_name,
                        sub_strategy_name=str(rec["name"]),
                        exchange=cfg.exchange,
                        symbol=exec_cfg.symbol,
                        action="CLOSE",
                        side="SELL" if str(rec["side"]).upper() == "LONG" else "BUY",
                        position_side=str(rec["side"]),
                        order_type="EXCHANGE_FLAT_RECONCILE",
                        signal_id=str(rec["signal_id"]),
                        status=str(info.get("status")),
                        order_info=info,
                    )
                    state["open_sleeves"].pop(str(rec["name"]), None)
                if reconciled_positions:
                    history = list(state.get("exchange_flat_reconcile_history", []))
                    history.extend(
                        {**rec, "reconciled_at": str(pd.Timestamp.utcnow())}
                        for rec in reconciled_positions
                    )
                    state["exchange_flat_reconcile_history"] = history[-100:]
                    state["last_exchange_flat_reconcile"] = reconciled_positions

            closed: list[str] = []
            for key, open_state in list(state["open_sleeves"].items()):
                time_exit_due = pd.Timestamp(open_state["exit_at"]) <= now
                dynamic_exit_due, dynamic_exit_reasons = _dynamic_exit_due(
                    open_state=open_state,
                    latest_feature_row=features.iloc[-1],
                    latest_bar=pd.Timestamp(enriched.iloc[-1]["date"]),
                    interval_minutes=exec_cfg.interval_minutes,
                )
                if dynamic_exit_reasons:
                    open_state["last_dynamic_exit_reasons"] = dynamic_exit_reasons
                    state["open_sleeves"][key] = open_state
                if time_exit_due or dynamic_exit_due:
                    if not exec_cfg.dry_run:
                        assert client is not None and executor is not None
                        close_reference_price = float(await client.get_ticker_price(exec_cfg.symbol))
                        close_info = await _close_sleeve(
                            client=client,
                            executor=executor,
                            sleeve_state=open_state,
                            exec_cfg=exec_cfg,
                            ttl_sec=int(cfg.max_exit_wait_sec),
                            reference_price=close_reference_price,
                            max_deviation_pct=float(cfg.exit_maker_max_deviation_pct),
                            refresh_interval_sec=int(cfg.maker_refresh_interval_sec),
                        )
                        close_info = await _attach_exchange_trade_report(
                            client=client,
                            symbol=exec_cfg.symbol,
                            order_info=close_info,
                        )
                        open_state["last_close_order_info"] = close_info
                        try:
                            close_filled = Decimal(str(close_info.get("filled_quantity", "0") or "0"))
                        except Exception:
                            close_filled = Decimal("0")
                        if close_filled < Decimal(str(open_state.get("quantity", "0") or "0")):
                            _log_trade_execution(
                                engine,
                                strategy_name=cfg.strategy_name,
                                sub_strategy_name=str(open_state.get("name", key)),
                                exchange=cfg.exchange,
                                symbol=exec_cfg.symbol,
                                action="CLOSE",
                                side="SELL" if str(open_state.get("side")).upper() == "LONG" else "BUY",
                                position_side=str(open_state.get("side")),
                                order_type="POST_ONLY_DYNAMIC_EXIT" if dynamic_exit_due and not time_exit_due else "POST_ONLY_EXIT",
                                signal_id=str(open_state.get("signal_id")),
                                status=str(close_info.get("status", "PARTIAL_OR_TIMEOUT")),
                                order_info=close_info,
                            )
                            open_state["close_pending"] = True
                            state["open_sleeves"][key] = open_state
                            continue
                        _log_trade_execution(
                            engine,
                            strategy_name=cfg.strategy_name,
                            sub_strategy_name=str(open_state.get("name", key)),
                            exchange=cfg.exchange,
                            symbol=exec_cfg.symbol,
                            action="CLOSE",
                            side="SELL" if str(open_state.get("side")).upper() == "LONG" else "BUY",
                            position_side=str(open_state.get("side")),
                            order_type=(
                                "POST_ONLY_DYNAMIC_EXIT_WITH_TAKER_FALLBACK"
                                if dynamic_exit_due and close_info.get("taker_fallback_order")
                                else "POST_ONLY_DYNAMIC_EXIT"
                                if dynamic_exit_due
                                else "POST_ONLY_EXIT_WITH_TAKER_FALLBACK"
                                if close_info.get("taker_fallback_order")
                                else "POST_ONLY_EXIT"
                            ),
                            signal_id=str(open_state.get("signal_id")),
                            status=str(close_info.get("status", "FILLED")),
                            order_info=close_info,
                        )
                    closed.append(key)
                    state["open_sleeves"].pop(key, None)

            opened: list[str] = []
            replaced_entry_orders = 0
            for sleeve in sleeve_scores:
                name = sleeve["name"]
                signal_id = sleeve["signal_id"]
                if not sleeve["active"]:
                    continue
                if name in state["open_sleeves"]:
                    continue
                if state["processed_signals"].get(name) == signal_id:
                    continue
                margin_fraction = _margin_fraction_for_weight(
                    weight=float(sleeve["weight"]),
                    total_weight=total_weight,
                    leverage_budget=float(cfg.leverage),
                    allocation_mode=cfg.allocation_mode,
                )
                quantity = "0"
                order_info: dict[str, Any] = {}
                entry_ttl_sec = _entry_ttl_seconds(
                    sleeve,
                    interval_minutes=exec_cfg.interval_minutes,
                    timeout_fraction=cfg.entry_timeout_fraction,
                    max_entry_wait_sec=cfg.max_entry_wait_sec,
                )
                sleeve["entry_maker_max_deviation_pct"] = float(cfg.entry_maker_max_deviation_pct)
                sleeve["maker_refresh_interval_sec"] = int(cfg.maker_refresh_interval_sec)
                if not exec_cfg.dry_run:
                    assert client is not None and executor is not None
                    replaced = await _cancel_portfolio_orders_for_sleeve(
                        client=client,
                        symbol=exec_cfg.symbol,
                        sleeve_name=name,
                        reason="new_signal_replaces_stale_entry",
                    )
                    if replaced:
                        replaced_entry_orders += len(replaced)
                        state["last_replaced_entry_orders"] = replaced
                        history = list(state.get("replaced_entry_order_history", []))
                        history.extend(replaced)
                        state["replaced_entry_order_history"] = history[-200:]
                    order_info = await _open_sleeve(
                        client=client,
                        executor=executor,
                        exec_cfg=exec_cfg,
                        sleeve=sleeve,
                        margin_fraction=margin_fraction,
                        entry_ttl_sec=entry_ttl_sec,
                    )
                    _log_trade_execution(
                        engine,
                        strategy_name=cfg.strategy_name,
                        sub_strategy_name=name,
                        exchange=cfg.exchange,
                        symbol=exec_cfg.symbol,
                        action="OPEN",
                        side="BUY" if sleeve["side"] == "LONG" else "SELL",
                        position_side=str(sleeve["side"]),
                        order_type="POST_ONLY_ENTRY",
                        signal_id=signal_id,
                        status=str(order_info.get("order", {}).get("status", "")),
                        order_info=order_info.get("order", order_info),
                        computing_wall_time_sec=frame_build_sec,
                    )
                    quantity = str(order_info.get("filled_quantity") or "0")
                    if Decimal(str(quantity)) <= Decimal("0"):
                        state["processed_signals"][name] = signal_id
                        missed = list(state.get("missed_entries", []))
                        missed.append(
                            {
                                "name": name,
                                "signal_id": signal_id,
                                "reason": "post_only_entry_not_filled",
                                "entry_ttl_sec": entry_ttl_sec,
                                "order_info": order_info,
                                "recorded_at": str(pd.Timestamp.utcnow()),
                            }
                        )
                        state["missed_entries"] = missed[-200:]
                        continue
                else:
                    quantity = str((100.0 * margin_fraction * exec_cfg.leverage) / float(sleeve["current_close"]))
                    order_info = {"status": "DRY_RUN", "entry_ttl_sec": entry_ttl_sec, "filled_quantity": quantity}
                signal_ts = pd.Timestamp(sleeve["date"])
                if signal_ts.tzinfo is None:
                    signal_ts = signal_ts.tz_localize("UTC")
                exit_at = signal_ts + pd.Timedelta(minutes=exec_cfg.interval_minutes * (1 + int(sleeve["hold_bars"])))
                state["open_sleeves"][name] = {
                    "name": name,
                    "side": sleeve["side"],
                    "signal_id": signal_id,
                    "signal_date": sleeve["date"],
                    "exit_at": str(exit_at),
                    "weight": sleeve["weight"],
                    "margin_fraction": margin_fraction,
                    "allocation_mode": cfg.allocation_mode,
                    "quantity": quantity,
                    "entry_reference_price": float(sleeve.get("current_close", 0.0) or 0.0),
                    "maker_refresh_interval_sec": int(cfg.maker_refresh_interval_sec),
                    "entry_maker_max_deviation_pct": float(cfg.entry_maker_max_deviation_pct),
                    "exit_maker_max_deviation_pct": float(cfg.exit_maker_max_deviation_pct),
                    "order_info": order_info,
                    "dynamic_exit": sleeve.get("dynamic_exit"),
                }
                state["processed_signals"][name] = signal_id
                opened.append(name)

            state["updated_at"] = str(pd.Timestamp.utcnow())
            state["last_scores"] = sleeve_scores
            state["allocation_audit"] = audit
            state["last_timing"] = {
                "frame_build_sec": round(frame_build_sec, 3),
                "freshness_waited_sec": round(freshness_waited, 3),
                "latest_bar": str(enriched.iloc[-1]["date"]),
                "latest_1m_ts": str(latest_1m_ts),
                "expected_bar": str(expected_bar),
                "decision_data_asof": str(decision_data_asof),
                "decision_bar_complete": decision_bar_complete,
                "asof": str(asof),
                "freshness_mode": freshness_mode,
                "source_latest_ts": {k: None if v is None else str(v) for k, v in latest_source_ts.items()},
                "source_freshness_missing": freshness_missing,
                "source_cache_mode": source_cache.last_query_mode,
                "feature_cache_mode": feature_cache.last_mode,
                "oi_cache_mode": oi_cache.last_query_mode,
                "external_cache_mode": external_cache.last_mode,
            }
            _write_json(cfg.state_file, state)
            active = [s["name"] for s in sleeve_scores if s["active"]]
            selector_status = "none"
            if portfolio_selector_record is not None:
                selector_status = f"{portfolio_selector_record.get('action')}:{portfolio_selector_record.get('context_id')}"
            _status(
                f"[portfolio-live] {pd.Timestamp.utcnow().isoformat()} active={active} opened={opened} closed={closed} "
                f"open={list(state['open_sleeves'])} recovered={len(recovered_positions)} reconciled={len(reconciled_positions)} stale_cancel={len(stale_cancelled)} repl={replaced_entry_orders} gross={total_weight:.3f} lev={exec_cfg.leverage} "
                f"alloc={cfg.allocation_mode} live_gross={audit['live_gross_if_all_active']:.3f} selector={selector_status} "
                f"fb={frame_build_sec:.2f}s fw={freshness_waited:.1f}s fm={freshness_mode} miss={len(freshness_missing)} src={source_cache.last_query_mode} oi={oi_cache.last_query_mode} ext={external_cache.last_mode} alt={alt_pool_cache.last_query_mode} feat={feature_cache.last_mode} dry_run={exec_cfg.dry_run}"
            )
            iterations += 1
            if cfg.max_iterations is not None and iterations >= cfg.max_iterations:
                LOG.info("portfolio_live.max_iterations_reached", extra={"iterations": iterations})
                return
    finally:
        if client is not None:
            await client.aclose()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run fixed-weight gross portfolio live executor")
    p.add_argument("--portfolio-config", default="configs/live/portfolio_gross6_mdd20_ratio5_return_best_candidate.json")
    p.add_argument("--execution-config", default="configs/live/rex_llm_binance_mainnet_bear_pilot_lev6.local.json")
    p.add_argument("--env", default=".env")
    p.add_argument("--state-file", default=".omx/state/portfolio_live_state.json")
    p.add_argument("--log-file", default="logs/portfolio_live/rllm.log")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--strategy-name", default="rllm")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--lookback-minutes", type=int, default=45_000)
    p.add_argument("--close-delay-sec", type=float, default=0.25)
    p.add_argument("--max-freshness-wait-sec", type=float, default=8.0)
    p.add_argument("--freshness-poll-sec", type=float, default=0.5, help="Deprecated fallback knob; live freshness uses Postgres NOTIFY")
    p.add_argument("--freshness-notify-channel", default="market_data_bar")
    p.add_argument("--run-immediately", action="store_true", default=False)
    p.add_argument("--live", action="store_true", default=False)
    p.add_argument("--allow-live-orders", action="store_true", default=False)
    p.add_argument("--leverage", type=int, default=6)
    p.add_argument(
        "--allocation-mode",
        choices=["research_gross", "normalize_weights"],
        default="research_gross",
        help="research_gross matches saved weights as notional exposure; normalize_weights uses 100% margin budget",
    )
    p.add_argument("--entry-timeout-fraction", type=float, default=0.25, help="Fraction of a sleeve stride cycle to wait for a post-only entry fill")
    p.add_argument("--max-entry-wait-sec", type=int, default=300, help="Hard cap for post-only entry wait/stale cancel age")
    p.add_argument("--max-exit-wait-sec", type=int, default=600, help="Hard cap for one post-only exit refresh cycle; still retried next loop while exit is due")
    p.add_argument("--maker-refresh-interval-sec", type=int, default=60, help="Refresh live post-only maker orders on this cadence")
    p.add_argument("--entry-maker-max-deviation-pct", type=float, default=0.003, help="Entry maker refresh band as fraction of signal reference price; calibrated to 0.3%")
    p.add_argument("--exit-maker-max-deviation-pct", type=float, default=0.002, help="Exit maker refresh band as fraction of exit reference price; calibrated to 0.2%")
    p.add_argument(
        "--portfolio-selector-overlay",
        default="",
        help="Optional bounded portfolio LLM selector overlay; ALLOW/BLOCK_RISK only for new entries",
    )
    p.add_argument("--rex-selector-adapter-dir", default="", help="Optional bounded REX TRADE/ABSTAIN LoRA adapter directory")
    p.add_argument("--rex-selector-model-name", default="gemma4-e4b-it")
    p.add_argument("--rex-selector-score-normalization", choices=["sum", "mean", "first_token"], default="sum")
    p.add_argument("--rex-selector-fail-open", action="store_true", default=False, help="If set, adapter errors do not block an otherwise valid REX candidate")
    p.add_argument("--rex-selector-allow-cpu", action="store_true", default=False, help="Allow selector inference without CUDA; default is fail-closed when CUDA is unavailable")
    stale = p.add_mutually_exclusive_group()
    stale.add_argument("--cancel-stale-open-orders", dest="cancel_stale_open_orders", action="store_true", default=True)
    stale.add_argument("--no-cancel-stale-open-orders", dest="cancel_stale_open_orders", action="store_false")
    p.add_argument("--max-iterations", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    _configure_logging(log_file=a.log_file, log_level=a.log_level)
    LOG.info("portfolio_live.start", extra={"log_file": str(a.log_file), "live": bool(a.live), "portfolio_config": str(a.portfolio_config)})
    asyncio.run(
        run_portfolio_loop(
            PortfolioLiveConfig(
                portfolio_config=Path(a.portfolio_config),
                execution_config=Path(a.execution_config),
                env_path=Path(a.env),
                state_file=Path(a.state_file),
                strategy_name=str(a.strategy_name),
                exchange=str(a.exchange),
                lookback_minutes=int(a.lookback_minutes),
                close_delay_sec=float(a.close_delay_sec),
                max_freshness_wait_sec=float(a.max_freshness_wait_sec),
                freshness_poll_sec=float(a.freshness_poll_sec),
                freshness_notify_channel=str(a.freshness_notify_channel),
                run_immediately=bool(a.run_immediately),
                live=bool(a.live),
                allow_live_orders=bool(a.allow_live_orders),
                leverage=int(a.leverage),
                allocation_mode=a.allocation_mode,
                max_iterations=a.max_iterations,
                entry_timeout_fraction=float(a.entry_timeout_fraction),
                max_entry_wait_sec=int(a.max_entry_wait_sec),
                max_exit_wait_sec=int(a.max_exit_wait_sec),
                maker_refresh_interval_sec=int(a.maker_refresh_interval_sec),
                entry_maker_max_deviation_pct=float(a.entry_maker_max_deviation_pct),
                exit_maker_max_deviation_pct=float(a.exit_maker_max_deviation_pct),
                cancel_stale_open_orders=bool(a.cancel_stale_open_orders),
                portfolio_selector_overlay=Path(a.portfolio_selector_overlay) if a.portfolio_selector_overlay else None,
                rex_selector_adapter_dir=Path(a.rex_selector_adapter_dir) if a.rex_selector_adapter_dir else None,
                rex_selector_model_name=str(a.rex_selector_model_name),
                rex_selector_score_normalization=str(a.rex_selector_score_normalization),
                rex_selector_fail_closed=not bool(a.rex_selector_fail_open),
                rex_selector_require_cuda=not bool(a.rex_selector_allow_cpu),
            )
        )
    )


if __name__ == "__main__":
    main()
