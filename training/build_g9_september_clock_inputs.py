#!/usr/bin/env python3
"""Materialize current G9 constituent clock inputs through 2026-09-05 00:00 UTC.

This builder is intentionally an input materializer for portfolio optimization.  It
reuses the live/monthly replay signal adapters and accounting contracts without
turning on any live configuration.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.portfolio_live import build_live_portfolio_frames, _build_portfolio_feature_frame
from preprocessing.binance_aux_features import normalise_funding_history_frame
from preprocessing.live_db_features import LiveDbFeatureConfig, sqlalchemy_engine_from_env
from training import backtest_added_alpha_month as month
from training.build_pposm_fresh_forward_signal_inventory_v2 import canonicalize_funding_aliases

OUT_DIR = Path("research/g9_september_inputs")
DEFAULT_PORTFOLIO = month.DEFAULT_PORTFOLIO
DEFAULT_OI_ARCHIVE = Path(
    "/home/pakchu/rllm/data/binance_um_metrics_BTCUSDT_2026-05-01_2026-09-04_extension.csv.gz"
)
CURRENT_SLEEVES = month.CURRENT_SLEEVES
BASE_LEVERAGE = month.BASE_LEVERAGE
COST_RATE = month.COST_RATE
INTERVAL_MINUTES = month.INTERVAL_MINUTES


@dataclass(frozen=True)
class Config:
    portfolio_config: Path = DEFAULT_PORTFOLIO
    env_path: Path = Path("/home/pakchu/rllm/.env")
    oi_archive: Path = DEFAULT_OI_ARCHIVE
    output_dir: Path = OUT_DIR
    warmup_start: str = "2026-03-01T00:00:00Z"
    eval_start: str = "2026-06-01T00:00:00Z"
    end: str = "2026-09-05T00:00:00Z"
    asof: str = "2026-09-05T00:05:00Z"
    lookback_minutes: int = 280_000
    enriched_cache: Path | None = None
    features_cache: Path | None = None
    funding_cache: Path | None = None


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _naive(value: Any) -> pd.Timestamp:
    return _utc(value).tz_localize(None)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.loc[:, columns].copy()
    for col in selected:
        if "date" in col:
            selected[col] = pd.to_datetime(selected[col], utc=True).astype("int64")
    payload = pd.util.hash_pandas_object(selected, index=False).to_numpy(dtype="<u8", copy=False)
    h = hashlib.sha256()
    h.update("\x1f".join(columns).encode())
    h.update(payload.tobytes())
    return h.hexdigest()


def archive_oi(raw: pd.DataFrame) -> pd.DataFrame:
    """Return publication-delayed official OI observations (+ one completed 5m bar)."""
    required = {"create_time", "sum_open_interest"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"OI archive missing columns: {sorted(missing)}")
    dates = pd.to_datetime(raw["create_time"], utc=True).dt.tz_convert(None)
    values = pd.to_numeric(raw["sum_open_interest"], errors="raise")
    if dates.duplicated().any():
        raise ValueError("duplicate OI archive create_time")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("invalid OI archive values")
    if not ((dates.astype("int64") % pd.Timedelta("5min").value) == 0).all():
        raise ValueError("off-grid OI archive timestamps")
    return pd.DataFrame(
        {"date": dates + pd.Timedelta(minutes=INTERVAL_MINUTES), "open_interest": values}
    ).sort_values("date")


def overlay_official_oi(enriched: pd.DataFrame, archive: pd.DataFrame, *, asof: pd.Timestamp) -> pd.DataFrame:
    """Overlay official OI without cash/zero substitution and preserve pre-archive DB OI."""
    out = enriched.sort_values("date").copy()
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_convert(None)
    archive = archive[archive["date"] <= asof.tz_localize(None)].copy()
    if archive.empty:
        raise ValueError("no OI archive rows available before asof")
    first_archive = pd.Timestamp(archive["date"].min())
    official = pd.merge_asof(
        out[["date"]].sort_values("date"),
        archive.sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(minutes=10),
    )
    use_official = out["date"] >= first_archive
    open_interest = pd.to_numeric(out.get("open_interest"), errors="coerce").to_numpy(float)
    official_values = pd.to_numeric(official["open_interest"], errors="coerce").to_numpy(float)
    open_interest[use_official.to_numpy()] = official_values[use_official.to_numpy()]
    out["open_interest"] = open_interest
    out["open_interest_available"] = np.where(np.isfinite(open_interest) & (open_interest > 0), 1.0, 0.0)
    # Hard gate: no silent zero/cash substitution in the requested evaluation interval.
    if (pd.to_numeric(out.loc[out["date"] < first_archive, "open_interest_available"], errors="coerce") <= 0.5).any():
        # Only pre-May warmup can rely on DB OI; if DB lacks it, Rank7/REX source is not genuine.
        raise RuntimeError("pre-archive DB open-interest source is incomplete")
    return out


class HistoricalOiSource:
    """Read archived DB observations, not the giant live-snapshot sort."""
    async def refresh(self, engine, *, asof, start, symbol, live_snapshot_cutoff=None):
        from sqlalchemy import text
        with engine.connect() as conn:
            return pd.read_sql_query(text("SELECT ts AS date, sum_open_interest AS open_interest FROM open_interest_binance WHERE symbol=:symbol AND period='5m' AND ts>=:start AND ts<=:end ORDER BY ts"), conn,
                                     params={"symbol":symbol,"start":start.to_pydatetime(),"end":asof.to_pydatetime()})


async def _query_frames(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    engine = sqlalchemy_engine_from_env(cfg.env_path)
    live_cfg = LiveDbFeatureConfig(lookback_minutes=int(cfg.lookback_minutes), include_spot_source=True)
    asof = _utc(cfg.asof)
    enriched, features = await build_live_portfolio_frames(
        engine=engine,
        asof=asof,
        cfg=live_cfg,
        live_oi_snapshot_cutoff=asof + pd.Timedelta(minutes=2),
        include_activity_flow=False,
        include_alt_pool=False,
        oi_cache=HistoricalOiSource(),
    )
    from sqlalchemy import text

    with engine.connect() as conn:
        funding = pd.read_sql_query(
            text(
                """
                SELECT funding_time AS date, funding_rate, mark_price
                FROM funding_rates_binance
                WHERE symbol = 'BTCUSDT'
                  AND funding_time <= :asof
                ORDER BY funding_time
                """
            ),
            conn,
            params={"asof": asof.to_pydatetime()},
        )
    return enriched, features, funding, engine


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    return pd.read_csv(path, compression="infer")


def _auto_cache_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / "raw_enriched_cache.pkl",
        output_dir / "raw_features_cache.pkl",
        output_dir / "raw_funding_cache.csv.gz",
    )


def _load_frames(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any | None]:
    paths = (cfg.enriched_cache, cfg.features_cache, cfg.funding_cache)
    if any(paths):
        if not all(paths):
            raise ValueError("all frame caches must be supplied together")
        assert cfg.enriched_cache and cfg.features_cache and cfg.funding_cache
        return _read_frame(cfg.enriched_cache), _read_frame(cfg.features_cache), _read_frame(cfg.funding_cache), None
    auto = _auto_cache_paths(cfg.output_dir)
    if all(path.exists() for path in auto):
        return _read_frame(auto[0]), _read_frame(auto[1]), _read_frame(auto[2]), None
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    enriched, features, funding, engine = asyncio.run(_query_frames(cfg))
    enriched.to_pickle(auto[0])
    features.to_pickle(auto[1])
    funding.to_csv(auto[2], index=False, compression="gzip")
    return enriched, features, funding, engine


def _assert_grid(enriched: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp) -> None:
    dates = pd.to_datetime(enriched["date"])
    if dates.duplicated().any():
        raise RuntimeError("market frame has duplicate dates")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta(minutes=INTERVAL_MINUTES)).all():
        raise RuntimeError("market frame is not a complete 5-minute grid")
    mask = (dates >= start) & (dates < end)
    expected = int((end - start) / pd.Timedelta(minutes=INTERVAL_MINUTES))
    if int(mask.sum()) != expected:
        raise RuntimeError(f"evaluation window completeness mismatch: {int(mask.sum())} != {expected}")


def _required_runtime_columns(source_cfg: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    del source_cfg
    common = ["date", "open", "high", "low", "close"]
    external = [
        "spot_rows",
        "premium_rows",
        "open_interest",
        "open_interest_available",
        "funding_available",
        "premium_available",
        "usdkrw_available",
        "dxy_available",
        "kimchi_available",
    ]
    return {name: sorted(set(common + external)) for name in CURRENT_SLEEVES}

def _gate_no_missing(frame: pd.DataFrame, columns_by_sleeve: dict[str, list[str]], *, eval_mask: np.ndarray) -> dict[str, Any]:
    summary = {}
    for sleeve, columns in columns_by_sleeve.items():
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise RuntimeError(f"{sleeve} missing required source columns: {missing}")
        missing_counts = {
            c: int(pd.to_numeric(frame.loc[eval_mask, c], errors="coerce").isna().sum())
            for c in columns
            if c not in {"date"}
        }
        bad = {c: n for c, n in missing_counts.items() if n > 0 and c in {"open", "high", "low", "close"}}
        if bad:
            raise RuntimeError(f"{sleeve} has missing market OHLC in eval window: {bad}")
        summary[sleeve] = {"required_columns": columns, "missing_value_counts_eval": missing_counts}
    return summary


def _empty_arrays(n: int) -> dict[str, Any]:
    return {"R": np.zeros(n), "L": np.zeros(n), "H": np.zeros(n), "signal": np.zeros(n, dtype=np.int8), "trades": [], "skipped_overlap": 0, "skipped_boundary": 0}


def _fixed_hold_arrays_with_trades(market: pd.DataFrame, signal: np.ndarray, *, name: str, hold_bars: int, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    dates = pd.to_datetime(market["date"])
    out = _empty_arrays(len(market)); out["signal"] = signal.astype(np.int8)
    next_allowed = 0
    for raw_position in np.flatnonzero(signal):
        position = int(raw_position)
        if not (start <= dates.iloc[position] < end):
            continue
        if position < next_allowed:
            out["skipped_overlap"] += 1; continue
        side = "long" if int(signal[position]) > 0 else "short"
        path = month._event_path(market, position, side=side, hold=int(hold_bars), cost_rate=COST_RATE, entry_delay=1, leverage=BASE_LEVERAGE)
        if path is None:
            out["skipped_boundary"] += 1; continue
        event_return, event_adverse, realized = path
        nonzero = np.flatnonzero(np.abs(event_return) > 1e-15)
        if not len(nonzero):
            out["skipped_boundary"] += 1; continue
        exit_position = int(nonzero[-1])
        if not (dates.iloc[exit_position] < end):
            out["skipped_boundary"] += 1; continue
        event_favorable = month.favorable_path(market, signal_position=position, exit_position=exit_position, side=side, leverage=BASE_LEVERAGE)
        out["R"] += event_return
        if side == "long":
            out["L"] += event_adverse; out["H"] += event_favorable
        else:
            out["L"] += event_favorable; out["H"] += event_adverse
        entry_position = position + 1
        out["trades"].append({
            "sleeve": name,
            "signal_position": position,
            "entry_position": entry_position,
            "exit_position": exit_position,
            "signal_date": str(dates.iloc[position]),
            "entry_date": str(dates.iloc[entry_position]),
            "exit_date": str(dates.iloc[exit_position]),
            "side": side.upper(),
            "entry_price": float(market["open"].iloc[entry_position]),
            "exit_price": float(market["open"].iloc[exit_position]),
            "exit_kind": "open",
            "barrier": None,
            "hold_bars": int(hold_bars),
            "unit_leverage": BASE_LEVERAGE,
            "net_return": float(realized),
        })
        next_allowed = exit_position + 1
    return out


def _barrier_arrays_with_trades(market: pd.DataFrame, funding: pd.DataFrame, signal: np.ndarray, *, name: str, lifecycle: Callable[[int], dict[str, Any]], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    dates = pd.to_datetime(market["date"])
    out = _empty_arrays(len(market)); out["signal"] = signal.astype(np.int8)
    execution_cfg = month._research_execution_config(end)
    engine = month.ExecutionEngine(market, funding, execution_cfg)
    trades = [] ; hold_by_signal: dict[int, int] = {}; spec_by_signal: dict[int, dict[str, Any]] = {}
    next_allowed = 0
    for raw_position in np.flatnonzero(signal):
        position = int(raw_position)
        if not (start <= dates.iloc[position] < end):
            continue
        if position < next_allowed:
            out["skipped_overlap"] += 1; continue
        spec = lifecycle(position)
        hold = int(spec["hold_bars"])
        take_bps = month.NO_BARRIER_BPS if spec.get("take_bps") is None else float(spec["take_bps"])
        stop_bps = month.NO_BARRIER_BPS if spec.get("stop_bps") is None else float(spec["stop_bps"])
        trade = engine.trade_at(position, int(signal[position]), hold, round(take_bps), round(stop_bps))
        if trade is None or not (dates.iloc[trade.exit_position] < end):
            out["skipped_boundary"] += 1; continue
        trades.append(trade); hold_by_signal[position] = hold; spec_by_signal[position] = spec
        next_allowed = int(trade.exit_position) + 1
    if trades:
        path = month.subaccount_bar_path(market, funding, trades, execution_cfg, start=str(start), end=str(end), hold_bars=lambda t: hold_by_signal[int(t.signal_position)])
        event = month.path_event(market, path, split="window", sleeve=name, trades=trades)
        out["R"] = event["ret"]; out["L"] = event["low"]; out["H"] = event["high"]
        cost_factor = 1.0 - BASE_LEVERAGE * COST_RATE
        for trade in trades:
            spec = spec_by_signal[int(trade.signal_position)]
            entry_price = float(market["open"].iloc[trade.entry_position])
            cap = int(trade.entry_position) + int(hold_by_signal[int(trade.signal_position)])
            side = 1 if int(trade.side) > 0 else -1
            if int(trade.exit_position) >= cap:
                exit_kind = "open"; exit_price = float(market["open"].iloc[trade.exit_position])
            elif float(trade.gross_return) >= 0:
                exit_kind = "barrier"; exit_price = entry_price * (1.0 + side * abs(float(trade.gross_return)))
            else:
                exit_kind = "barrier"; exit_price = entry_price * (1.0 + side * float(trade.gross_return))
            out["trades"].append({
                "sleeve": name,
                "signal_position": int(trade.signal_position),
                "entry_position": int(trade.entry_position),
                "exit_position": int(trade.exit_position),
                "signal_date": str(dates.iloc[trade.signal_position]),
                "entry_date": str(dates.iloc[trade.entry_position]),
                "exit_date": str(dates.iloc[trade.exit_position]),
                "side": "LONG" if side > 0 else "SHORT",
                "entry_price": entry_price,
                "exit_price": float(exit_price),
                "exit_kind": exit_kind,
                "barrier": {"take_bps": spec.get("take_bps"), "stop_bps": spec.get("stop_bps")},
                "hold_bars": int(hold_by_signal[int(trade.signal_position)]),
                "unit_leverage": BASE_LEVERAGE,
                "gross_return": float(trade.gross_return),
                "funding_factor": float(trade.funding_factor),
                "net_return": float(cost_factor * float(trade.price_factor) * float(trade.funding_factor) * cost_factor - 1.0),
                "source": spec.get("source"),
            })
    return out


def _metric(arrays: dict[str, dict[str, Any]], weights: dict[str, float], dates: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    return month._strict_metric(arrays, weights, dates=dates, start=start, end=end)


def run(cfg: Config) -> dict[str, Any]:
    portfolio = month._load_json(cfg.portfolio_config)
    weights = {str(k): float(v) for k, v in portfolio["weights"].items()}
    if tuple(weights) != CURRENT_SLEEVES:
        raise RuntimeError(f"unexpected G9 sleeve order: {tuple(weights)}")
    if not np.isclose(sum(weights.values()), 8.0):
        raise RuntimeError("unexpected repository G8 source-config gross weight")

    enriched, features, raw_funding, engine = _load_frames(cfg)
    enriched = enriched.copy()
    enriched["date"] = pd.to_datetime(enriched["date"], utc=True).dt.tz_convert(None)
    features = features.reset_index(drop=True)
    if len(enriched) != len(features):
        raise RuntimeError("enriched/features length mismatch")
    archive = archive_oi(pd.read_csv(cfg.oi_archive, compression="infer"))
    enriched = overlay_official_oi(enriched, archive, asof=_utc(cfg.asof))
    # OI overlay must refresh both predictors and availability metadata.
    features = _build_portfolio_feature_frame(enriched, LiveDbFeatureConfig(include_spot_source=True), include_activity_flow=False)
    raw_funding = raw_funding.copy()
    raw_funding["date"] = pd.to_datetime(raw_funding["date"], utc=True, format="mixed").dt.tz_convert(None)
    raw_funding = raw_funding[raw_funding["date"] >= enriched["date"].min()]
    canonical, funding_diagnostics = canonicalize_funding_aliases(raw_funding)
    if not funding_diagnostics["passed"]:
        raise RuntimeError(f"Ambiguous funding aliases: {funding_diagnostics['reason']}")
    funding = normalise_funding_history_frame(canonical)

    warmup_start = _naive(cfg.warmup_start); start = _naive(cfg.eval_start); end = _naive(cfg.end)
    dates = pd.to_datetime(enriched["date"])
    if dates.iloc[0] > warmup_start:
        raise RuntimeError(f"warmup starts too late: {dates.iloc[0]} > {warmup_start}")
    _assert_grid(enriched, start=start, end=end)
    eval_mask = ((dates >= start) & (dates < end)).to_numpy(bool)

    source_cfg = {row["name"]: month._load_json(row["source"]) for row in portfolio["base_sleeves"]}
    coverage = _gate_no_missing(enriched, _required_runtime_columns(source_cfg), eval_mask=eval_mask)

    fresh_signal = month._fresh_signal(enriched, features, source_cfg["fresh_kimchi_fx"])
    rank7_signal, rank7_lifecycles, rank7_diagnostics = month._rank7_signal(enriched, source_cfg["frozen_annual_rank7"])
    rex_taker_signal = month._rex_signal(enriched, features, source_cfg["rex_taker_low_range_position"])
    rex_veto_signal = month._rex_signal(enriched, features, source_cfg["cand_rex_veto_7"])
    markov_signal = month._markov_signal(enriched, features, source_cfg["markov_transition_long"])

    fresh_cfg = source_cfg["fresh_kimchi_fx"]
    arrays = {
        "fresh_kimchi_fx": _barrier_arrays_with_trades(enriched, funding, fresh_signal, name="fresh_kimchi_fx", lifecycle=lambda _p: {"hold_bars": int(fresh_cfg["hold_bars"]), "take_bps": float(fresh_cfg["take_bps"]), "stop_bps": float(fresh_cfg["stop_bps"]), "source": None}, start=start, end=end),
        "frozen_annual_rank7": _barrier_arrays_with_trades(enriched, funding, rank7_signal, name="frozen_annual_rank7", lifecycle=lambda p: rank7_lifecycles[int(p)], start=start, end=end),
        "rex_taker_low_range_position": _fixed_hold_arrays_with_trades(enriched, rex_taker_signal, name="rex_taker_low_range_position", hold_bars=int(source_cfg["rex_taker_low_range_position"]["hold_bars"]), start=start, end=end),
        "cand_rex_veto_7": _fixed_hold_arrays_with_trades(enriched, rex_veto_signal, name="cand_rex_veto_7", hold_bars=int(source_cfg["cand_rex_veto_7"]["hold_bars"]), start=start, end=end),
        "markov_transition_long": _fixed_hold_arrays_with_trades(enriched, markov_signal, name="markov_transition_long", hold_bars=int(source_cfg["markov_transition_long"]["hold_bars"]), start=start, end=end),
    }
    signals = {"fresh_kimchi_fx": fresh_signal, "frozen_annual_rank7": rank7_signal, "rex_taker_low_range_position": rex_taker_signal, "cand_rex_veto_7": rex_veto_signal, "markov_transition_long": markov_signal}

    out = cfg.output_dir; out.mkdir(parents=True, exist_ok=True)
    trades = sorted([t for name in CURRENT_SLEEVES for t in arrays[name]["trades"]], key=lambda t: (t["entry_date"], t["sleeve"]))
    trades_csv = out / "g9_constituent_trades_2026-06-01_2026-09-05.csv"
    pd.DataFrame(trades).to_csv(trades_csv, index=False)
    market_csv = out / "g9_market_5m_2026-06-01_2026-09-05.csv.gz"
    funding_csv = out / "g9_funding_2026-06-01_2026-09-05.csv.gz"
    enriched.loc[eval_mask, ["date", "open", "high", "low", "close"]].to_csv(market_csv, index=False, compression="gzip")
    funding_out = funding.copy()
    funding_out["date"] = pd.to_datetime(funding_out["date"], utc=True).dt.tz_convert(None)
    funding_out = funding_out[(funding_out["date"] >= start) & (funding_out["date"] < end)]
    funding_out.to_csv(funding_csv, index=False, compression="gzip")
    cache_npz = out / "g9_market_funding_arrays_2026-06-01_2026-09-05.npz"
    np.savez_compressed(
        cache_npz,
        date_ns=pd.to_datetime(enriched.loc[eval_mask, "date"]).astype("int64").to_numpy(),
        open=enriched.loc[eval_mask, "open"].to_numpy(float),
        high=enriched.loc[eval_mask, "high"].to_numpy(float),
        low=enriched.loc[eval_mask, "low"].to_numpy(float),
        close=enriched.loc[eval_mask, "close"].to_numpy(float),
        open_interest=enriched.loc[eval_mask, "open_interest"].to_numpy(float),
        funding_rate=enriched.loc[eval_mask, "funding_rate"].to_numpy(float),
    )
    sleeve_npz = out / "g9_sleeve_return_risk_arrays_2026-06-01_2026-09-05.npz"
    np.savez_compressed(
        sleeve_npz,
        **{f"{name}_R": arrays[name]["R"][eval_mask] for name in CURRENT_SLEEVES},
        **{f"{name}_L": arrays[name]["L"][eval_mask] for name in CURRENT_SLEEVES},
        **{f"{name}_H": arrays[name]["H"][eval_mask] for name in CURRENT_SLEEVES},
        **{f"{name}_signal": signals[name][eval_mask] for name in CURRENT_SLEEVES},
    )

    source_files = {__file__: _sha256(__file__), month.__file__: _sha256(month.__file__), str(cfg.portfolio_config): _sha256(cfg.portfolio_config), str(cfg.oi_archive): _sha256(cfg.oi_archive)}
    for row in portfolio["base_sleeves"]:
        source_files[str(row["source"])] = _sha256(row["source"])
    manifest = Path(source_cfg["frozen_annual_rank7"]["bundle_path"]) / "manifest.json"
    source_files[str(manifest)] = _sha256(manifest)

    report = {
        "schema_version": 1,
        "funding_aliases": funding_diagnostics,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "mode": "g9_constituent_clock_input_materialization",
        "live_enabled": False,
        "retrospective_not_pristine_oos": True,
        "config": {**asdict(cfg), "portfolio_config": str(cfg.portfolio_config), "env_path": "<redacted>", "oi_archive": str(cfg.oi_archive), "output_dir": str(cfg.output_dir), "enriched_cache": None if cfg.enriched_cache is None else str(cfg.enriched_cache), "features_cache": None if cfg.features_cache is None else str(cfg.features_cache), "funding_cache": None if cfg.funding_cache is None else str(cfg.funding_cache)},
        "window": {"warmup_start_requested": str(warmup_start), "market_start": str(dates.iloc[0]), "eval_start": str(start), "end_exclusive": str(end), "bars": int(eval_mask.sum()), "last_eval_bar": str(dates[eval_mask].iloc[-1])},
        "portfolio": {"name": portfolio["name"], "weights": weights, "gross_weight": float(sum(weights.values())), "source_config_status": portfolio.get("status")},
        "outputs": {"trades_csv": str(trades_csv.resolve()), "market_csv": str(market_csv.resolve()), "funding_csv": str(funding_csv.resolve()), "market_funding_npz": str(cache_npz.resolve()), "sleeve_arrays_npz": str(sleeve_npz.resolve())},
        "market_csv": str(market_csv.resolve()),
        "funding_csv": str(funding_csv.resolve()),
        "sleeves": {name: {"trades": arrays[name]["trades"]} for name in CURRENT_SLEEVES},
        "data_quality": {"source_coverage_gate": coverage, "availability_eval": month._availability_summary(enriched, eval_mask), "rank7_minimum_feature_history_bars": int(source_cfg["frozen_annual_rank7"].get("minimum_feature_history_bars", 0)), "oi_archive_rows": int(len(archive)), "oi_archive_first_delayed": str(archive["date"].min()), "oi_archive_last_delayed_before_asof": str(archive.loc[archive["date"] <= _utc(cfg.asof).tz_localize(None), "date"].max()), "market_hash_eval": _frame_hash(enriched.loc[eval_mask].reset_index(drop=True), ["date", "open", "high", "low", "close", "open_interest", "funding_rate", "premium_index", "usdkrw", "kimchi_premium"])},
        "signal_diagnostics": {name: {"scheduled_raw": int(np.count_nonzero(signals[name][eval_mask])), "raw_longs": int(np.count_nonzero(signals[name][eval_mask] > 0)), "raw_shorts": int(np.count_nonzero(signals[name][eval_mask] < 0)), "accepted_trades": len(arrays[name]["trades"]), "skipped_overlap": int(arrays[name]["skipped_overlap"]), "skipped_boundary": int(arrays[name]["skipped_boundary"])} for name in CURRENT_SLEEVES},
        "rank7_diagnostics": rank7_diagnostics,
        "metrics": {"repository_g8_source_control": _metric(arrays, weights, dates, start, end), "standalone": {name: _metric(arrays, {name: 1.0}, dates, start, end) for name in CURRENT_SLEEVES}, "weighted": {name: _metric(arrays, {name: weights[name]}, dates, start, end) for name in CURRENT_SLEEVES}},
        "trade_count": len(trades),
        "trades_head": trades[:20],
        "source_sha256": source_files,
        "passed": True,
        "market_sha256": _sha256(market_csv),
        "receipts": {"source_sha256": source_files, "rank7_diagnostics": rank7_diagnostics, "data_quality": {"availability_eval": month._availability_summary(enriched, eval_mask), "market_hash_eval": _frame_hash(enriched.loc[eval_mask].reset_index(drop=True), ["date", "open", "high", "low", "close", "open_interest", "funding_rate", "premium_index", "usdkrw", "kimchi_premium"])}},
        "notes": ["G9 constituent inputs only; no live config changes.", "Rank7 and REX sources are hard gated; missing sources are not replaced with zeros.", "Official OI archive observations are shifted +5m as a publication proxy.", "Fixed-hold REX/Markov accounting preserves frozen monthly replay contract; Fresh/Rank7 include realized funding."],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n")
    if engine is not None:
        engine.dispose()
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--portfolio-config", default=str(DEFAULT_PORTFOLIO))
    p.add_argument("--env", default="/home/pakchu/rllm/.env")
    p.add_argument("--oi-archive", default=str(DEFAULT_OI_ARCHIVE))
    p.add_argument("--output-dir", default=str(OUT_DIR))
    p.add_argument("--warmup-start", default=Config.warmup_start)
    p.add_argument("--eval-start", default=Config.eval_start)
    p.add_argument("--end", default=Config.end)
    p.add_argument("--asof", default=Config.asof)
    p.add_argument("--lookback-minutes", type=int, default=Config.lookback_minutes)
    p.add_argument("--enriched-cache", default="")
    p.add_argument("--features-cache", default="")
    p.add_argument("--funding-cache", default="")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    report = run(Config(portfolio_config=Path(a.portfolio_config), env_path=Path(a.env), oi_archive=Path(a.oi_archive), output_dir=Path(a.output_dir), warmup_start=str(a.warmup_start), eval_start=str(a.eval_start), end=str(a.end), asof=str(a.asof), lookback_minutes=int(a.lookback_minutes), enriched_cache=Path(a.enriched_cache) if a.enriched_cache else None, features_cache=Path(a.features_cache) if a.features_cache else None, funding_cache=Path(a.funding_cache) if a.funding_cache else None))
    print(json.dumps({"report": str(Path(report["outputs"]["trades_csv"]).parent / "report.json"), "outputs": report["outputs"], "window": report["window"], "metrics": report["metrics"]["repository_g8_source_control"], "signal_diagnostics": report["signal_diagnostics"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
