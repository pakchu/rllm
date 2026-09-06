"""Build source-only HVCVCOIR-12 clocks before opening outcomes or Gross9."""
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
from training import preregister_high_volatility_variance_concentration_open_interest_release as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.long_regime_combo_scan import LongComboScanConfig, _load_market


PREREG_SHA = "867fb4fd8b702477f9a492f2839e0d7c3d1f810aa70dda4b31ee07119e209cff"
MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
FUNDING = "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
PREMIUM = "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
CLOCK = Path("data/high_volatility_variance_concentration_open_interest_release_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_variance_concentration_open_interest_release_controls_2023_2026")
SNAPSHOT = Path("data/high_volatility_variance_concentration_open_interest_release_sources_2023_2026/variance_concentration.csv.gz")
OI_SNAPSHOT = Path("data/high_volatility_variance_concentration_open_interest_release_sources_2023_2026/open_interest_5m.csv.gz")
RESULT = Path("results/high_volatility_variance_concentration_open_interest_release_support_2026-08-16.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_high_volatility_gate", "no_release_requirement", "one_bar_stale_variance", "direction_flip")
OI_QUERY = """SELECT ts,sum_open_interest FROM open_interest_binance WHERE symbol='BTCUSDT' AND period='5m' AND ts>=:start AND ts<:end ORDER BY ts"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_combined_market() -> tuple[pd.DataFrame, dict[str, Any]]:
    historical = pd.read_csv(MARKET, compression="infer")
    historical["date"] = pd.to_datetime(historical["date"], utc=True, errors="raise")
    historical = historical.loc[historical.date.lt(pd.Timestamp("2026-06-02T00:00:00Z"))].copy()
    cfg = month.Config(
        start="2026-05-01T00:00:00Z",
        end="2026-08-01T00:00:00Z",
        asof="2026-08-01T00:02:00Z",
        lookback_minutes=150_000,
    )
    live, _features, _funding, engine = asyncio.run(month._query_frames(cfg))
    if engine is not None:
        engine.dispose()
    live = live.copy()
    live["date"] = pd.to_datetime(live["date"], utc=True)
    combined = (
        pd.concat([historical, live], ignore_index=True, sort=False)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    combined = combined[combined.date < pd.Timestamp("2026-08-01T00:00:00Z")].reset_index(drop=True)
    if combined.date.duplicated().any() or not combined.date.diff().dropna().eq(pd.Timedelta(minutes=5)).all():
        raise RuntimeError("HVCVCOIR combined market continuity drift")
    return combined, {
        "historical_rows": len(historical),
        "live_rows": len(live),
        "combined_rows": len(combined),
        "first": str(combined.date.iloc[0]),
        "last": str(combined.date.iloc[-1]),
        "mode": "hash_bound_historical_cache_plus_postgres_completed_bar_extension",
    }


def load_open_interest() -> pd.DataFrame:
    from sqlalchemy import create_engine, text
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    env_file = "/home/pakchu/rllm/.env"
    load_env_file(env_file)
    engine = create_engine(postgres_url_from_env(env_file), connect_args={"connect_timeout": 10})
    try:
        with engine.connect() as connection:
            frame = pd.read_sql_query(
                text(OI_QUERY), connection,
                params={"start": pd.Timestamp("2023-06-30T00:00:00Z"), "end": pd.Timestamp("2026-08-01T00:00:00Z")},
            )
    finally:
        engine.dispose()
    if frame.columns.tolist() != ["ts", "sum_open_interest"]:
        raise RuntimeError("HVCVCOIR OI schema drift")
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="raise")
    frame["sum_open_interest"] = pd.to_numeric(frame["sum_open_interest"], errors="raise")
    if frame.ts.duplicated().any() or not np.isfinite(frame.sum_open_interest).all() or not frame.sum_open_interest.ge(0).all():
        raise RuntimeError("HVCVCOIR invalid OI source")
    return frame.sort_values("ts").reset_index(drop=True)


def oi_expansion_at(oi: pd.Series, decision: pd.Timestamp) -> float | None:
    start = decision - pd.Timedelta(hours=4)
    if decision not in oi.index or start not in oi.index:
        return None
    first, last = float(oi.loc[start]), float(oi.loc[decision])
    if not (np.isfinite(first) and np.isfinite(last) and first > 0 and last > 0):
        return None
    change = float(np.log(last / first))
    return change if change > 0 else None


def score_snapshot(market: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    dates = pd.to_datetime(market.date, utc=True)
    closes = pd.to_numeric(market.close, errors="coerce")
    highs = pd.to_numeric(market.high, errors="coerce")
    lows = pd.to_numeric(market.low, errors="coerce")
    log_return = np.log(closes).diff()
    squared = log_return.pow(2)
    day_variance = squared.rolling(288, min_periods=288).sum()
    hour_variance = squared.rolling(12, min_periods=12).sum()
    concentration = hour_variance / day_variance.replace(0, np.nan)
    high_volatility = highs.rolling(288, min_periods=288).max() / lows.rolling(288, min_periods=288).min() - 1.0
    hour_return = closes / closes.shift(12) - 1.0
    positions = np.arange(288, len(market), dtype=np.int64)
    frame = pd.DataFrame({
        "position": positions,
        "decision_bar_time": dates.iloc[positions].to_numpy(),
        "day_variance": day_variance.iloc[positions].to_numpy(float),
        "hour_variance": hour_variance.iloc[positions].to_numpy(float),
        "concentration": concentration.iloc[positions].to_numpy(float),
        "high_volatility": high_volatility.iloc[positions].to_numpy(float),
        "hour_return": hour_return.iloc[positions].to_numpy(float),
    })
    calibration = frame[
        frame.decision_bar_time.ge(pd.Timestamp("2023-01-01T00:00:00Z"))
        & frame.decision_bar_time.lt(pd.Timestamp("2023-07-01T00:00:00Z"))
    ].replace([np.inf, -np.inf], np.nan).dropna()
    if len(calibration) < 10_000:
        raise RuntimeError("HVCVCOIR source-only calibration floor failed")
    thresholds = {
        "high_volatility_q60": float(calibration.high_volatility.quantile(0.60)),
        "concentration_onset_q90": float(calibration.concentration.quantile(0.90)),
        "concentration_release_q60": float(calibration.concentration.quantile(0.60)),
    }
    return frame[frame.decision_bar_time.ge(pd.Timestamp("2023-07-01T00:00:00Z"))].reset_index(drop=True), thresholds


def build_clock(scores: pd.DataFrame, thresholds: dict[str, float], oi_frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    frame = scores.copy()
    features = ["concentration", "high_volatility", "hour_return"]
    if control == "one_bar_stale_variance":
        frame[features] = frame[features].shift(1)
    valid = np.isfinite(frame[features]).all(axis=1) & frame.hour_return.ne(0)
    high_vol = frame.high_volatility.ge(thresholds["high_volatility_q60"])
    if control == "no_high_volatility_gate":
        high_vol[:] = True
    onset_level = frame.concentration.ge(thresholds["concentration_onset_q90"])
    onset = valid & high_vol & onset_level & ~onset_level.shift(1, fill_value=False)
    times = pd.to_datetime(frame.decision_bar_time, utc=True)
    oi = oi_frame.set_index("ts")["sum_open_interest"]
    signals: list[tuple[pd.Timestamp, int, float]] = []
    pending: dict[str, Any] | None = None
    for index in range(len(frame)):
        now = times.iloc[index]
        if pending is not None:
            if now > pending["deadline"]:
                pending = None
            elif frame.concentration.iloc[index] <= thresholds["concentration_release_q60"]:
                change = oi_expansion_at(oi, now)
                if change is not None:
                    signals.append((now + pd.Timedelta(minutes=5), -pending["shock_side"], change))
                pending = None
        if pending is None and bool(onset.iloc[index]):
            shock_side = 1 if frame.hour_return.iloc[index] > 0 else -1
            if control == "no_release_requirement":
                change = oi_expansion_at(oi, now)
                if change is not None:
                    signals.append((now + pd.Timedelta(minutes=5), -shock_side, change))
            else:
                pending = {"shock_side": shock_side, "deadline": now + pd.Timedelta(hours=6)}
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for entry, raw_side, oi_change in signals:
        exit_time = entry + pd.Timedelta(hours=12)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        next_allowed = exit_time
        rows.append({
            "candidate": "HVCVCOIR-12",
            "control": control,
            "split": split,
            "decision_time": entry,
            "feature_available_time": entry,
            "entry_time": entry,
            "exit_time": exit_time,
            "side": -raw_side if control == "direction_flip" else raw_side,
            "oi_change": oi_change,
        })
    return pd.DataFrame(rows, columns=("candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side", "oi_change"))

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
        raise RuntimeError("HVCVCOIR preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core_registration = {key: value for key, value in registration.items() if key != "manifest_hash"}
    if registration["manifest_hash"] != prereg.canonical_hash(core_registration):
        raise RuntimeError("HVCVCOIR preregistration manifest drift")
    market, source = load_combined_market()
    oi = load_open_interest()
    scores, thresholds = score_snapshot(market)
    primary = build_clock(scores, thresholds, oi)
    controls = {name: build_clock(scores, thresholds, oi, name) for name in CONTROLS}
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(scores, SNAPSHOT)
    _write_gzip_csv(oi, OI_SNAPSHOT)
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
        "protocol_version": "hvcvcoir_12_source_support_v1",
        "policy_id": "HVCVCOIR-12",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source": source,
        "calibration": {"window": ["2023-01-01T00:00:00Z", "2023-07-01T00:00:00Z"], **thresholds, "outcomes_opened": False},
        "source_snapshot": {"path": str(SNAPSHOT), "sha256": sha256(SNAPSHOT), "rows": len(scores)},
        "open_interest_snapshot": {"path": str(OI_SNAPSHOT), "sha256": sha256(OI_SNAPSHOT), "rows": len(oi), "query_sha256": hashlib.sha256(OI_QUERY.encode()).hexdigest()},
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
