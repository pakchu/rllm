"""Build source-only HVVRCR-12 clocks before opening outcomes or Gross9."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import backtest_all_alpha_month as month
from training import preregister_high_volatility_volume_return_covariance_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.long_regime_combo_scan import LongComboScanConfig, _load_market

PREREG_SHA = "f636bed76ff62a636d3a6ec6b14dabd1017847aeab300eab351c05b69397d8bb"
MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
FUNDING = "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
PREMIUM = "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
SNAPSHOT = Path("data/high_volatility_volume_return_covariance_relay_sources_2023_2026/volume_return_covariance_daily.csv.gz")
CLOCK = Path("data/high_volatility_volume_return_covariance_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_volume_return_covariance_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_volume_return_covariance_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_volatility_gate", "no_correlation_tail", "unweighted_return", "one_day_stale_features", "direction_flip", "forced_long")
CLOCK_COLUMNS = ("candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side", "realized_variation", "volume_return_correlation", "variation_rank", "absolute_correlation_rank", "unweighted_return")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def strict_prior_midrank(values: pd.Series, lookback: int = 90, minimum: int = 60) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            output.at[index] = (np.sum(array < current) + 0.5 * np.sum(array == current)) / len(array)
        if math.isfinite(current):
            history.append(current)
    return output


def load_combined_market() -> tuple[pd.DataFrame, dict[str, Any]]:
    historical = _load_market(LongComboScanConfig(input_csv=MARKET, output="", funding_csv=FUNDING, premium_csv=PREMIUM, exclude_from="2026-06-02"))
    historical["date"] = pd.to_datetime(historical.date, utc=True)
    cfg = month.Config(start="2026-05-01T00:00:00Z", end="2026-08-01T00:00:00Z", asof="2026-08-01T00:02:00Z", lookback_minutes=150_000)
    live, _features, _funding, engine = asyncio.run(month._query_frames(cfg))
    if engine is not None:
        engine.dispose()
    live = live.copy()
    live["date"] = pd.to_datetime(live.date, utc=True)
    combined = pd.concat([historical, live], ignore_index=True, sort=False).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    combined = combined[combined.date < pd.Timestamp("2026-08-01T00:00:00Z")].reset_index(drop=True)
    if combined.date.duplicated().any() or not combined.date.diff().dropna().eq(pd.Timedelta(minutes=5)).all():
        raise RuntimeError("HVVRCR combined market continuity drift")
    return combined, {"historical_rows": len(historical), "live_rows": len(live), "combined_rows": len(combined), "first": str(combined.date.iloc[0]), "last": str(combined.date.iloc[-1]), "mode": "hash_bound_historical_cache_plus_postgres_completed_bar_extension"}


def score_snapshot(market: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(market.date, utc=True)
    opens = pd.to_numeric(market.open, errors="coerce")
    closes = pd.to_numeric(market.close, errors="coerce")
    quote_volume = pd.to_numeric(market.quote_asset_volume, errors="coerce")
    returns = np.log(closes / opens)
    log_volume = np.log1p(quote_volume)
    positions = np.flatnonzero((dates.dt.hour.eq(2) & dates.dt.minute.eq(0)).to_numpy())
    rows: list[dict[str, Any]] = []
    for position in positions:
        if position < 288: continue
        r = returns.iloc[position - 288:position].to_numpy(float)
        v = log_volume.iloc[position - 288:position].to_numpy(float)
        valid = len(r) == 288 and np.isfinite(r).all() and np.isfinite(v).all() and (quote_volume.iloc[position - 288:position] >= 0).all() and float(np.std(r, ddof=1)) > 0 and float(np.std(v, ddof=1)) > 0
        correlation = float(np.corrcoef(r, v)[0, 1]) if valid else math.nan
        variation = float(np.sqrt(np.sum(r ** 2))) if valid else math.nan
        rows.append({"decision_time": dates.iloc[position], "source_valid": bool(valid and math.isfinite(correlation) and correlation != 0), "realized_variation": variation, "volume_return_correlation": correlation, "unweighted_return": float(np.sum(r)) if valid else math.nan, "valid_bar_count": 288 if valid else 0})
    frame = pd.DataFrame(rows).sort_values("decision_time").reset_index(drop=True)
    valid = frame.source_valid.astype(bool)
    frame["variation_rank"] = strict_prior_midrank(frame.realized_variation.where(valid))
    frame["absolute_correlation_rank"] = strict_prior_midrank(frame.volume_return_correlation.abs().where(valid))
    return frame[frame.decision_time.ge(pd.Timestamp("2023-01-01T00:00:00Z"))].reset_index(drop=True)


def conditions(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    work = frame.copy()
    if control == "one_day_stale_features":
        columns = ["realized_variation", "volume_return_correlation", "unweighted_return", "variation_rank", "absolute_correlation_rank", "source_valid"]
        work[columns] = work[columns].shift(1)
    factor = work.unweighted_return if control == "unweighted_return" else work.volume_return_correlation
    volatility = pd.Series(True, index=work.index) if control == "no_volatility_gate" else work.variation_rank.ge(0.65)
    tail = pd.Series(True, index=work.index) if control == "no_correlation_tail" else work.absolute_correlation_rank.ge(0.70)
    active = work.source_valid.fillna(False).astype(bool) & np.isfinite(factor) & factor.ne(0) & volatility & tail
    side = np.sign(factor)
    if control == "direction_flip": side = -side
    if control == "forced_long": side = pd.Series(1.0, index=work.index)
    return active, side


def build_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = conditions(frame, control)
    rows: list[dict[str, Any]] = []
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=12)
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        rows.append({"candidate": "HVVRCR-12", "control": control, "split": split, "decision_time": decision, "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]), "realized_variation": float(frame.at[index, "realized_variation"]), "volume_return_correlation": float(frame.at[index, "volume_return_correlation"]), "variation_rank": float(frame.at[index, "variation_rank"]), "absolute_correlation_rank": float(frame.at[index, "absolute_correlation_rank"]), "unweighted_return": float(frame.at[index, "unweighted_return"])})
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    rows = clock[clock.split.eq(split)]
    if rows.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(rows.side.eq(1).sum()), int(rows.side.eq(-1).sum())
    months = pd.to_datetime(rows.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(rows), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(rows), "max_month_share": int(months.max()) / len(rows)}


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVVRCR preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    market, source = load_combined_market()
    scores = score_snapshot(market)
    primary = build_clock(scores)
    controls = {name: build_clock(scores, name) for name in CONTROLS}
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(scores, SNAPSHOT)
    _write_gzip_csv(primary, CLOCK)
    for name, rows in controls.items():
        _write_gzip_csv(rows, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {check: passed for name, item in support.items() for check, passed in ((f"{name}_minimum_events", item["events"] >= MINIMUM[name]), (f"{name}_side_balance", item["minority_side_share"] >= 0.20), (f"{name}_month_concentration", item["max_month_share"] <= 0.45))}
    passed = all(checks.values())
    core = {"protocol_version": "hvvrcr_12_source_support_v1", "policy_id": "HVVRCR-12", "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "source": source, "source_snapshot": {"path": str(SNAPSHOT), "sha256": sha256(SNAPSHOT), "rows": len(scores)}, "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(rows), "promotion_authorized": False} for name, rows in controls.items()}, "support": support, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    report = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
