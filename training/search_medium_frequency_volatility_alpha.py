"""Select and evaluate a frozen medium-frequency BTC volatility alpha."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_funding_premium_external_state_gate_alpha import _frame_hash
from training.search_inventory_purge_reclaim_alpha import Config as EngineConfig
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine, Trade, equity_stats
from training.search_positioning_hgb_path_alpha import _read_before

TRAIN_START, TRAIN_END = "2020-01-01", "2023-01-01"
SELECTION_END = "2024-01-01"
EVAL_END = "2026-06-02"
NO_STOP_BPS = 1_000_000
SELECTION_WINDOWS = {
    "2020H2": ("2020-07-01", "2021-01-01"),
    "2021": ("2021-01-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
}
EVAL_WINDOWS = {
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
    "2026H1": ("2026-01-01", EVAL_END),
}
COMBINED_SELECTION = ("2020-07-01", SELECTION_END)
COMBINED_EVAL = ("2024-01-01", EVAL_END)


@dataclass(frozen=True)
class Config:
    market_csv: str = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
    funding_csv: str = "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
    output: str = "results/medium_frequency_volatility_alpha.json"
    artifact: str = "results/medium_frequency_volatility_alpha_top1.json"
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    stress_cost_rate: float = 0.0010


def _engine_config(cfg: Config) -> EngineConfig:
    return EngineConfig(cfg.market_csv, "", cfg.funding_csv, "", "", leverage=cfg.leverage,
                        fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate,
                        stress_cost_rate=cfg.stress_cost_rate)


def _load(cfg: Config, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    market = _read_before(cfg.market_csv, "date", cutoff)
    funding = _read_before(cfg.funding_csv, "date", cutoff)
    market = market[["date", "open", "high", "low", "close"]].copy()
    market["date"] = pd.to_datetime(market["date"], utc=True, format="mixed").dt.tz_localize(None)
    funding = funding[["date", "funding_rate"]].copy()
    funding["date"] = pd.to_datetime(funding["date"], utc=True, format="mixed").dt.tz_localize(None)
    funding["funding_rate"] = pd.to_numeric(funding["funding_rate"], errors="raise")
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    funding = funding.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if len(market) > 1 and not (market["date"].diff().iloc[1:] == pd.Timedelta(minutes=5)).all():
        raise RuntimeError("market is not a complete frozen 5m grid")
    return market, funding, {"market": _frame_hash(market), "funding": _frame_hash(funding)}


def decision_mask(dates: pd.Series) -> np.ndarray:
    """Exact completed 00:00/08:00/16:00 UTC boundaries."""
    d = pd.to_datetime(dates)
    return ((d.dt.minute == 0) & (d.dt.second == 0) & d.dt.hour.isin((0, 8, 16))).to_numpy(bool)


def build_features(market: pd.DataFrame) -> pd.DataFrame:
    """Features at D use only bars in intervals ending strictly before D."""
    o = pd.to_numeric(market["open"], errors="raise")
    h = pd.to_numeric(market["high"], errors="raise")
    l = pd.to_numeric(market["low"], errors="raise")
    c = pd.to_numeric(market["close"], errors="raise")
    previous_close = c.shift(1)
    log_return = np.log(c / c.shift(1))
    low24, high24 = l.shift(1).rolling(288, min_periods=288).min(), h.shift(1).rolling(288, min_periods=288).max()
    width = high24 - low24
    return pd.DataFrame({
        "return_24h": previous_close / o.shift(288) - 1.0,
        "return_72h": previous_close / o.shift(864) - 1.0,
        "realized_vol_24h": np.sqrt(log_return.pow(2).shift(1).rolling(288, min_periods=288).sum()),
        "range_position_24h": (previous_close - low24) / width.where(width > 0),
    })


def fit_thresholds(features: pd.DataFrame, dates: pd.Series) -> dict[str, float]:
    d = pd.to_datetime(dates)
    fit = decision_mask(d) & (d >= pd.Timestamp(TRAIN_START)).to_numpy() & (d < pd.Timestamp(TRAIN_END)).to_numpy()
    values = features.loc[fit, "realized_vol_24h"].dropna().to_numpy(float)
    if not len(values):
        raise ValueError("no train-only realized-vol observations")
    return {"rv_q50": float(np.quantile(values, .50)), "rv_q70": float(np.quantile(values, .70))}


def candidate_grid() -> list[dict[str, Any]]:
    keys = itertools.product(("trend", "contrarian"), (24, 72), ("none", "q50", "q70"),
                             ("none", "middle", "extreme"), (4, 8), (None, .02, .04))
    return [dict(zip(("side_mode", "return_hours", "vol_gate", "range_gate", "hold_hours", "tp"), row)) for row in keys]


def signal_sides(features: pd.DataFrame, dates: pd.Series, thresholds: dict[str, float], spec: dict[str, Any]) -> np.ndarray:
    ret = features[f"return_{spec['return_hours']}h"].to_numpy(float)
    rv = features["realized_vol_24h"].to_numpy(float)
    rp = features["range_position_24h"].to_numpy(float)
    active = decision_mask(dates) & np.isfinite(ret) & (ret != 0)
    if spec["vol_gate"] != "none": active &= rv >= thresholds[f"rv_{spec['vol_gate']}"]
    if spec["range_gate"] == "middle": active &= (rp >= .25) & (rp <= .75)
    elif spec["range_gate"] == "extreme": active &= (rp <= .25) | (rp >= .75)
    side = np.where(np.isfinite(ret), np.sign(ret), 0).astype(np.int8)
    if spec["side_mode"] == "contrarian": side *= -1
    return np.where(active, side, 0)


def _schedule(engine: ExecutionEngine, sides: np.ndarray, start: str, end: str, spec: dict[str, Any]) -> list[Trade]:
    dates = pd.to_datetime(engine.market["date"])
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    out: list[Trade] = []
    next_allowed = 0
    tp_bps = NO_STOP_BPS if spec["tp"] is None else int(round(10_000 * spec["tp"]))
    for signal in np.flatnonzero(period & (sides != 0)):
        if int(signal) < next_allowed:
            continue
        trade = engine.trade_at(int(signal), int(sides[signal]), int(spec["hold_hours"] * 12), tp_bps, NO_STOP_BPS)
        if trade is not None and period[trade.exit_position]:
            out.append(trade)
            next_allowed = int(trade.exit_position) + 1
    return out


def _stats(engine: ExecutionEngine, sides: np.ndarray, spec: dict[str, Any], windows: dict[str, tuple[str, str]], cfg: Config) -> tuple[dict[str, Any], dict[str, list[Trade]]]:
    result, schedules = {}, {}
    ecfg = _engine_config(cfg)
    for name, (start, end) in windows.items():
        trades = _schedule(engine, sides, start, end, spec); schedules[name] = trades
        weeks = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (7 * 86400)
        base = equity_stats(trades, start=start, end=end, cfg=ecfg)
        stress = equity_stats(trades, start=start, end=end, cfg=ecfg, cost_rate=cfg.stress_cost_rate)
        result[name] = {"base": base, "stress": stress, "avg_trades_per_week": len(trades) / weeks}
    return result, schedules


def frequency_passes(stats: dict[str, Any]) -> bool:
    return all(row["avg_trades_per_week"] >= 3.0 for row in stats.values())


def _window_gate(stats: dict[str, Any]) -> bool:
    return frequency_passes(stats) and all(
        row[mode]["absolute_return_pct"] > 0 and row[mode]["strict_mdd_pct"] <= 20
        for row in stats.values() for mode in ("base", "stress")
    )


def _net_returns(trades: list[Trade], cfg: Config) -> np.ndarray:
    factor = 1 - cfg.leverage * (cfg.fee_rate + cfg.slippage_rate)
    return np.asarray([factor * t.price_factor * t.funding_factor * factor - 1 for t in trades])


def _one_sided_p(trades: list[Trade], cfg: Config) -> float:
    x = _net_returns(trades, cfg)
    if len(x) < 2 or x.std(ddof=1) == 0: return 0.0 if len(x) and x.mean() > 0 else 1.0
    z = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    return float(.5 * math.erfc(z / np.sqrt(2)))


def _write(path: str, payload: dict[str, Any]) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def select(cfg: Config) -> dict[str, Any]:
    market, funding, source_hashes = _load(cfg, SELECTION_END)
    features, dates = build_features(market), market["date"]
    thresholds = fit_thresholds(features, dates)
    engine = ExecutionEngine(market, funding, _engine_config(cfg))
    rows = []
    for spec in candidate_grid():
        sides = signal_sides(features, dates, thresholds, spec)
        stats, _ = _stats(engine, sides, spec, SELECTION_WINDOWS, cfg)
        combined, _ = _stats(engine, sides, spec, {"combined": COMBINED_SELECTION}, cfg)
        passed = _window_gate(stats) and combined["combined"]["base"]["cagr_to_strict_mdd"] >= 1.5
        ratios = [x["base"]["cagr_to_strict_mdd"] for x in stats.values()]
        score = [min(ratios), combined["combined"]["base"]["cagr_to_strict_mdd"], combined["combined"]["base"]["absolute_return_pct"]]
        rows.append({"spec": spec, "passed": passed, "score": score, "stats": stats, "combined": combined["combined"]})
    rows.sort(key=lambda x: (x["passed"], *x["score"]), reverse=True)
    if not rows[0]["passed"]: raise RuntimeError("no candidate passed selection")
    frozen = {"protocol": "medium_frequency_volatility_alpha_v1", "selection_end": SELECTION_END,
              "source_hashes": source_hashes, "thresholds": thresholds, "top1": rows[0], "config": asdict(cfg)}
    frozen["artifact_hash"] = _hash(frozen)
    _write(cfg.artifact, frozen)
    result = {"phase": "select", "tested": len(rows), "top1": rows[0], "artifact_hash": frozen["artifact_hash"]}
    _write(cfg.output, result); return result


def evaluate(cfg: Config) -> dict[str, Any]:
    frozen = json.loads(Path(cfg.artifact).read_text())
    expected = frozen.pop("artifact_hash")
    if _hash(frozen) != expected or frozen.get("selection_end") != SELECTION_END: raise RuntimeError("invalid frozen top1 artifact")
    if frozen.get("config") != asdict(cfg): raise RuntimeError("runtime config differs from frozen artifact")
    market, funding, _ = _load(cfg, EVAL_END)
    dates, features = market["date"], build_features(market)
    if fit_thresholds(features, dates) != frozen["thresholds"]: raise RuntimeError("train-only thresholds drifted")
    spec = frozen["top1"]["spec"]  # Deliberately no candidate_grid/ranking call in eval.
    sides = signal_sides(features, dates, frozen["thresholds"], spec)
    engine = ExecutionEngine(market, funding, _engine_config(cfg))
    stats, _ = _stats(engine, sides, spec, EVAL_WINDOWS, cfg)
    combined_stats, schedules = _stats(engine, sides, spec, {"combined": COMBINED_EVAL}, cfg)
    p = _one_sided_p(schedules["combined"], cfg)
    passed = _window_gate(stats) and p <= .20
    result = {"phase": "eval", "artifact_hash": expected, "top1": spec, "stats": stats,
              "combined": combined_stats["combined"], "combined_p_value": p, "passed": passed}
    _write(cfg.output, result); return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("command", choices=("select", "eval"))
    for name, field in Config.__dataclass_fields__.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    args = vars(parser.parse_args()); command = args.pop("command")
    print(json.dumps(select(Config(**args)) if command == "select" else evaluate(Config(**args)), indent=2))


if __name__ == "__main__": main()
