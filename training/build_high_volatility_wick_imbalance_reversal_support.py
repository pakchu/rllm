"""Build source-only HVWIR-8 clocks before opening outcomes or Gross9."""
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
from training import preregister_high_volatility_wick_imbalance_reversal as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.long_regime_combo_scan import LongComboScanConfig, _load_market


PREREG_SHA = "64c8ce112c20c191f8a71d87393c8f3ed8e61c08def95dd340b599a525ec948b"
MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
FUNDING = "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
PREMIUM = "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
CLOCK = Path("data/high_volatility_wick_imbalance_reversal_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_wick_imbalance_reversal_controls_2023_2026")
SNAPSHOT = Path("data/high_volatility_wick_imbalance_reversal_sources_2023_2026/four_hour_wick_morphology.csv.gz")
RESULT = Path("results/high_volatility_wick_imbalance_reversal_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_high_range_gate", "no_wick_imbalance_gate", "one_anchor_stale_morphology", "direction_flip")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_combined_market() -> tuple[pd.DataFrame, dict[str, Any]]:
    historical = _load_market(
        LongComboScanConfig(
            input_csv=MARKET,
            output="",
            funding_csv=FUNDING,
            premium_csv=PREMIUM,
            exclude_from="2026-06-02",
        )
    )
    historical["date"] = pd.to_datetime(historical["date"], utc=True)
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
        raise RuntimeError("HVWIR combined market continuity drift")
    return combined, {
        "historical_rows": len(historical),
        "live_rows": len(live),
        "combined_rows": len(combined),
        "first": str(combined.date.iloc[0]),
        "last": str(combined.date.iloc[-1]),
        "mode": "hash_bound_historical_cache_plus_postgres_completed_bar_extension",
    }


def score_snapshot(market: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    dates = pd.to_datetime(market.date, utc=True)
    opens = pd.to_numeric(market.open, errors="coerce").to_numpy(float)
    highs = pd.to_numeric(market.high, errors="coerce").to_numpy(float)
    lows = pd.to_numeric(market.low, errors="coerce").to_numpy(float)
    closes = pd.to_numeric(market.close, errors="coerce").to_numpy(float)
    anchor_mask = dates.dt.hour.mod(4).eq(0) & dates.dt.minute.eq(0)
    rows: list[dict[str, Any]] = []
    for position in np.flatnonzero(anchor_mask.to_numpy()):
        if position < 48:
            continue
        left = position - 48
        bar_open = opens[left]
        bar_high = highs[left:position].max()
        bar_low = lows[left:position].min()
        bar_close = closes[position - 1]
        values = np.array([bar_open, bar_high, bar_low, bar_close], dtype=float)
        if not np.isfinite(values).all() or bar_open <= 0 or bar_low <= 0:
            continue
        upper = (bar_high - max(bar_open, bar_close)) / bar_open
        lower = (min(bar_open, bar_close) - bar_low) / bar_open
        total = upper + lower
        rows.append({
            "position": int(position),
            "decision_time": dates.iloc[position],
            "bar_open": float(bar_open),
            "bar_high": float(bar_high),
            "bar_low": float(bar_low),
            "bar_close": float(bar_close),
            "range": float(bar_high / bar_low - 1.0),
            "upper_wick": float(upper),
            "lower_wick": float(lower),
            "wick_imbalance": float(abs(upper - lower) / total) if total > 0 else np.nan,
        })
    frame = pd.DataFrame(rows)
    calibration = frame[
        frame.decision_time.ge(pd.Timestamp("2023-01-01T00:00:00Z"))
        & frame.decision_time.lt(pd.Timestamp("2023-07-01T00:00:00Z"))
    ].dropna(subset=["range", "wick_imbalance"])
    if len(calibration) < 500:
        raise RuntimeError("HVWIR source-only calibration floor failed")
    thresholds = {
        "range_q70": float(calibration["range"].quantile(0.70)),
        "wick_imbalance_q70": float(calibration.wick_imbalance.quantile(0.70)),
    }
    return frame[frame.decision_time.ge(pd.Timestamp("2023-07-01T00:00:00Z"))].reset_index(drop=True), thresholds


def build_clock(scores: pd.DataFrame, thresholds: dict[str, float], control: str = "primary") -> pd.DataFrame:
    frame = scores.copy()
    morphology = ["range", "upper_wick", "lower_wick", "wick_imbalance"]
    if control == "one_anchor_stale_morphology":
        frame[morphology] = frame[morphology].shift(1)
    high_range = frame["range"].ge(thresholds["range_q70"])
    imbalance = frame.wick_imbalance.ge(thresholds["wick_imbalance_q70"])
    if control == "no_high_range_gate":
        high_range[:] = True
    if control == "no_wick_imbalance_gate":
        imbalance[:] = True
    frame = frame[
        high_range & imbalance & np.isfinite(frame[morphology]).all(axis=1)
        & frame.upper_wick.add(frame.lower_wick).gt(0)
        & frame.upper_wick.ne(frame.lower_wick)
    ].copy()
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for item in frame.itertuples(index=False):
        entry = pd.Timestamp(item.decision_time)
        exit_time = entry + pd.Timedelta(hours=8)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        raw_side = -1 if item.upper_wick > item.lower_wick else 1
        next_allowed = exit_time
        rows.append({
            "candidate": "HVWIR-8",
            "control": control,
            "split": split,
            "decision_time": entry,
            "feature_available_time": entry,
            "entry_time": entry,
            "exit_time": exit_time,
            "side": -raw_side if control == "direction_flip" else raw_side,
        })
    return pd.DataFrame(rows, columns=("candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side"))

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
        raise RuntimeError("HVWIR preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core_registration = {key: value for key, value in registration.items() if key != "manifest_hash"}
    if registration["manifest_hash"] != prereg.canonical_hash(core_registration):
        raise RuntimeError("HVWIR preregistration manifest drift")
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
        "protocol_version": "hvwir_8_source_support_v1",
        "policy_id": "HVWIR-8",
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
