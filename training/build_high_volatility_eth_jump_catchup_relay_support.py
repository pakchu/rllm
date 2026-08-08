"""Build source-only HVEJCR-6 clocks before opening outcomes or Gross9."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training import backtest_all_alpha_month as month
from training import preregister_high_volatility_eth_jump_catchup_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.long_regime_combo_scan import LongComboScanConfig, _load_market


PREREG_SHA = "df6f4adc36eaf2f64f854a39b989ed3182f84d8442eed133d2fa3239013e74c2"
MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
FUNDING = "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
PREMIUM = "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
CLOCK = Path("data/high_volatility_eth_jump_catchup_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_eth_jump_catchup_relay_controls_2023_2026")
SNAPSHOT = Path("data/high_volatility_eth_jump_catchup_relay_sources_2023_2026/eth_jump_scores.csv.gz")
RESULT = Path("results/high_volatility_eth_jump_catchup_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_volatility_gate", "no_muted_btc_response_gate", "one_bar_stale_jump_inputs", "direction_flip")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env
    env_file = "/home/pakchu/rllm/.env"
    load_env_file(env_file)
    return create_engine(postgres_url_from_env(env_file), connect_args={"connect_timeout": 10})


def load_combined_market() -> tuple[pd.DataFrame, dict[str, Any]]:
    from sqlalchemy import text
    query = text("""
        SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS date,
               (array_agg(open ORDER BY ts))[1] AS open, max(high) AS high, min(low) AS low,
               (array_agg(close ORDER BY ts DESC))[1] AS close, count(*) AS source_rows
        FROM bars_binance
        WHERE interval='1m' AND symbol=:symbol AND ts>=:start AND ts<:end
        GROUP BY 1 ORDER BY 1
    """)
    start = pd.Timestamp("2023-01-01T00:00:00Z")
    end = pd.Timestamp("2026-08-01T00:00:00Z")
    db = postgres_engine()
    with db.connect() as connection:
        btc = pd.read_sql_query(query, connection, params={"symbol":"BTCUSDT","start":start.to_pydatetime(),"end":end.to_pydatetime()})
        eth = pd.read_sql_query(query, connection, params={"symbol":"ETHUSDT","start":start.to_pydatetime(),"end":end.to_pydatetime()})
    db.dispose()
    for frame in (btc, eth):
        frame["date"] = pd.to_datetime(frame.date, utc=True)
        if not frame.source_rows.eq(5).all() or not frame.date.diff().dropna().eq(pd.Timedelta(minutes=5)).all():
            raise RuntimeError("HVEJCR exact aligned 5m source incomplete")
    merged = btc.merge(eth, on="date", suffixes=("_btc","_eth"), validate="one_to_one")
    if len(merged) != len(btc) or len(merged) != len(eth):
        raise RuntimeError("HVEJCR BTC/ETH alignment drift")
    return merged, {
        "mode":"postgres_exact_1m_to_5m_source_snapshot", "table":"bars_binance",
        "symbols":["BTCUSDT","ETHUSDT"], "interval":"1m",
        "rows_per_symbol":len(btc), "first":str(merged.date.iloc[0]), "last":str(merged.date.iloc[-1]),
    }


def score_snapshot(market: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    dates = pd.to_datetime(market.date, utc=True)
    btc_close = pd.to_numeric(market.close_btc, errors="coerce")
    eth_close = pd.to_numeric(market.close_eth, errors="coerce")
    btc_high = pd.to_numeric(market.high_btc, errors="coerce")
    btc_low = pd.to_numeric(market.low_btc, errors="coerce")
    btc_return = np.log(btc_close).diff()
    eth_return = np.log(eth_close).diff()
    eth_abs = eth_return.abs()
    adjacent = eth_abs * eth_abs.shift(1)
    prior_bipower = (np.pi / 2.0) * adjacent.shift(1).rolling(288, min_periods=288).sum()
    eth_jump_score = eth_abs / np.sqrt(prior_bipower / 288.0).replace(0, np.nan)
    btc_range = btc_high.shift(1).rolling(288, min_periods=288).max() / btc_low.shift(1).rolling(288, min_periods=288).min() - 1.0
    calibration_mask = dates.ge(pd.Timestamp("2023-01-01T00:00:00Z")) & dates.lt(pd.Timestamp("2023-07-01T00:00:00Z"))
    calibration = pd.DataFrame({"jump":eth_jump_score[calibration_mask],"range":btc_range[calibration_mask]}).replace([np.inf,-np.inf],np.nan).dropna()
    if len(calibration) < 30_000:
        raise RuntimeError("HVEJCR source-only calibration floor failed")
    thresholds = {
        "btc_range_q60": float(calibration["range"].quantile(0.60)),
        "eth_jump_score_q99": float(calibration.jump.quantile(0.99)),
    }
    positions = np.arange(290, len(market), dtype=np.int64)
    frame = pd.DataFrame({
        "position":positions, "decision_bar_time":dates.iloc[positions].to_numpy(),
        "btc_five_minute_return":btc_return.iloc[positions].to_numpy(float),
        "eth_five_minute_return":eth_return.iloc[positions].to_numpy(float),
        "eth_prior_bipower_variation":prior_bipower.iloc[positions].to_numpy(float),
        "eth_jump_score":eth_jump_score.iloc[positions].to_numpy(float),
        "btc_range_volatility":btc_range.iloc[positions].to_numpy(float),
    })
    return frame[frame.decision_bar_time.ge(pd.Timestamp("2023-07-01T00:00:00Z"))].reset_index(drop=True), thresholds


def build_clock(scores: pd.DataFrame, thresholds: dict[str, float], control: str = "primary") -> pd.DataFrame:
    frame = scores.copy()
    columns = ["btc_five_minute_return","eth_five_minute_return","eth_jump_score","btc_range_volatility"]
    if control == "one_bar_stale_jump_inputs": frame[columns] = frame[columns].shift(1)
    valid = np.isfinite(frame[columns]).all(axis=1) & frame.eth_five_minute_return.ne(0)
    high_vol = frame.btc_range_volatility.ge(thresholds["btc_range_q60"])
    if control == "no_volatility_gate": high_vol[:] = True
    jump_level = frame.eth_jump_score.ge(thresholds["eth_jump_score_q99"])
    muted = frame.btc_five_minute_return.abs().le(0.5 * frame.eth_five_minute_return.abs())
    if control == "no_muted_btc_response_gate": muted[:] = True
    active = valid & high_vol & jump_level & muted
    onset = active & ~active.shift(1, fill_value=False)
    rows: list[dict[str, Any]]=[]; next_allowed: pd.Timestamp|None=None
    for item in frame[onset].itertuples(index=False):
        entry=pd.Timestamp(item.decision_bar_time)+pd.Timedelta(minutes=5);exit_time=entry+pd.Timedelta(hours=6)
        if next_allowed is not None and entry<next_allowed:continue
        split=next((name for name,(start,end) in SPLITS.items() if entry>=start and exit_time<=end),None)
        if split is None:continue
        raw_side=1 if item.eth_five_minute_return>0 else -1;next_allowed=exit_time
        rows.append({"candidate":"HVEJCR-6","control":control,"split":split,"decision_time":entry,"feature_available_time":entry,"entry_time":entry,"exit_time":exit_time,"side":-raw_side if control=="direction_flip" else raw_side})
    return pd.DataFrame(rows,columns=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side"))

def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    rows = clock[clock.split.eq(split)]
    if rows.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(rows.side.eq(1).sum()), int(rows.side.eq(-1).sum())
    months = pd.to_datetime(rows.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(rows),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(rows),
        "max_month_share": int(months.max()) / len(rows),
    }


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVEJCR preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core_registration = {key: value for key, value in registration.items() if key != "manifest_hash"}
    if registration["manifest_hash"] != prereg.canonical_hash(core_registration):
        raise RuntimeError("HVEJCR preregistration manifest drift")
    market, source = load_combined_market()
    scores, thresholds = score_snapshot(market)
    primary = build_clock(scores, thresholds)
    controls = {name: build_clock(scores, thresholds, name) for name in CONTROLS}
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(scores, SNAPSHOT)
    _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {
        check: passed
        for name, item in support.items()
        for check, passed in (
            (f"{name}_minimum_events", item["events"] >= MINIMUM[name]),
            (f"{name}_side_balance", item["minority_side_share"] >= 0.2),
            (f"{name}_month_concentration", item["max_month_share"] <= 0.45),
        )
    }
    passed = all(checks.values())
    core = {
        "protocol_version": "hvejcr_6_source_support_v1",
        "policy_id": "HVEJCR-6",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source": source,
        "calibration": {"window": ["2023-01-01T00:00:00Z", "2023-07-01T00:00:00Z"], **thresholds, "outcomes_opened": False},
        "source_snapshot": {"path": str(SNAPSHOT), "sha256": sha256(SNAPSHOT), "rows": len(scores)},
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    report = {**core, "manifest_hash": prereg.canonical_hash(core)}
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "calibration": report["calibration"], "support": report["support"]}, indent=2))
