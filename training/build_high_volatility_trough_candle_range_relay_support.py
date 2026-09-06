"""Build source-only HVTCR-24 clocks before opening outcomes or Gross9."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import backtest_all_alpha_month as month
from training import preregister_high_volatility_trough_candle_range_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.long_regime_combo_scan import LongComboScanConfig, _load_market


PREREG_SHA = "0da132979e20f8042aca49a19682c22eb111bd6a898c5de5ff0d1dcc3a6162f0"
MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
CLOCK = Path("data/high_volatility_trough_candle_range_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_trough_candle_range_relay_controls_2023_2026")
SNAPSHOT = Path(
    "data/high_volatility_trough_candle_range_relay_sources_2023_2026/trough_candle_scores.csv.gz"
)
RESULT = Path("results/high_volatility_trough_candle_range_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_volatility_gate",
    "first_trough_tie_break",
    "current_candle_range",
    "direction_flip",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_combined_market() -> tuple[pd.DataFrame, dict[str, Any]]:
    historical = _load_market(
        LongComboScanConfig(input_csv=MARKET, output="", exclude_from="2026-06-02")
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
    combined = combined[combined.date < pd.Timestamp("2026-08-01T00:00:00Z")].reset_index(
        drop=True
    )
    if combined.date.duplicated().any() or not combined.date.diff().dropna().eq(
        pd.Timedelta(minutes=5)
    ).all():
        raise RuntimeError("HVTCR combined market continuity drift")
    return combined, {
        "historical_rows": len(historical),
        "live_rows": len(live),
        "combined_rows": len(combined),
        "first": str(combined.date.iloc[0]),
        "last": str(combined.date.iloc[-1]),
        "mode": "hash_bound_historical_cache_plus_postgres_completed_bar_extension",
    }


def score_snapshot(market: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    dates = pd.DatetimeIndex(pd.to_datetime(market.date, utc=True))
    highs = pd.to_numeric(market.high, errors="coerce").to_numpy(float)
    lows = pd.to_numeric(market.low, errors="coerce").to_numpy(float)
    closes = pd.to_numeric(market.close, errors="coerce").to_numpy(float)
    positions = np.flatnonzero(dates.minute.to_numpy() == 55).astype(np.int64)
    positions = positions[positions >= 143]
    complete = dates[positions] - dates[positions - 143] == pd.Timedelta(minutes=715)
    positions = positions[np.asarray(complete)]
    rows: list[dict[str, Any]] = []
    for position in positions:
        trough_lows = lows[position - 71 : position + 1]
        minimum = np.nanmin(trough_lows)
        ties = np.flatnonzero(trough_lows == minimum)
        first_trough = position - 71 + int(ties[0])
        last_trough = position - 71 + int(ties[-1])
        range_high = np.nanmax(highs[position - 143 : position + 1])
        range_low = np.nanmin(lows[position - 143 : position + 1])
        midpoint = 0.5 * (range_high + range_low)
        current_close = closes[position]
        rows.append(
            {
                "position": position,
                "decision_bar_time": dates[position],
                "trough_candle_range_last": (highs[last_trough] - lows[last_trough]) / current_close,
                "trough_candle_range_first": (highs[first_trough] - lows[first_trough]) / current_close,
                "current_candle_range": (highs[position] - lows[position]) / current_close,
                "range_vol": (range_high - range_low) / midpoint,
            }
        )
    frame = pd.DataFrame(rows)
    calibration = frame[
        frame.decision_bar_time.ge(pd.Timestamp("2023-01-01T00:00:00Z"))
        & frame.decision_bar_time.lt(pd.Timestamp("2023-07-01T00:00:00Z"))
    ].replace([np.inf, -np.inf], np.nan).dropna()
    if len(calibration) < 4_000:
        raise RuntimeError("HVTCR source-only calibration floor failed")
    thresholds = {
        "trough_candle_range_q15": float(calibration.trough_candle_range_last.quantile(0.15)),
        "trough_candle_range_q85": float(calibration.trough_candle_range_last.quantile(0.85)),
        "range_vol_q60": float(calibration.range_vol.quantile(0.60)),
    }
    return (
        frame[frame.decision_bar_time.ge(pd.Timestamp("2023-07-01T00:00:00Z"))]
        .reset_index(drop=True),
        thresholds,
    )


def build_clock(
    scores: pd.DataFrame, thresholds: dict[str, float], control: str = "primary"
) -> pd.DataFrame:
    feature = "trough_candle_range_last"
    if control == "first_trough_tie_break":
        feature = "trough_candle_range_first"
    elif control == "current_candle_range":
        feature = "current_candle_range"
    values = pd.to_numeric(scores[feature], errors="coerce").to_numpy(float)
    volatility = pd.to_numeric(scores.range_vol, errors="coerce").to_numpy(float)
    high_tail = values >= thresholds["trough_candle_range_q85"]
    low_tail = values <= thresholds["trough_candle_range_q15"]
    high_onset = high_tail & ~np.r_[False, high_tail[:-1]]
    low_onset = low_tail & ~np.r_[False, low_tail[:-1]]
    high_volatility = volatility >= thresholds["range_vol_q60"]
    if control == "no_volatility_gate":
        high_volatility[:] = True
    finite = np.isfinite(values) & np.isfinite(volatility)
    onset = finite & high_volatility & (high_onset | low_onset)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in scores.index[onset]:
        item = scores.loc[index]
        entry = pd.Timestamp(item.decision_bar_time) + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=24)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next(
            (
                name
                for name, (start, end) in SPLITS.items()
                if entry >= start and exit_time <= end
            ),
            None,
        )
        if split is None:
            continue
        if high_onset[index] and low_onset[index]:
            continue
        side = 1 if high_onset[index] else -1
        if control == "direction_flip":
            side *= -1
        next_allowed = exit_time
        rows.append(
            {
                "candidate": "HVTCR-24",
                "control": control,
                "split": split,
                "decision_time": pd.Timestamp(item.decision_bar_time),
                "feature_available_time": entry,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": side,
            }
        )
    columns = (
        "candidate",
        "control",
        "split",
        "decision_time",
        "feature_available_time",
        "entry_time",
        "exit_time",
        "side",
    )
    return pd.DataFrame(rows, columns=columns)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    rows = clock[clock.split.eq(split)]
    if rows.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "minority_side_share": 0.0,
            "max_month_share": 0.0,
        }
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
        raise RuntimeError("HVTCR preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
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
        "protocol_version": "hvtcr_24_source_support_v1",
        "policy_id": "HVTCR-24",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source": source,
        "feature_contract": {
            "calibration_window": ["2023-01-01T00:00:00Z", "2023-07-01T00:00:00Z"],
            **thresholds,
            "outcomes_opened": False,
        },
        "source_snapshot": {"path": str(SNAPSHOT), "sha256": sha256(SNAPSHOT), "rows": len(scores)},
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(frame),
                "promotion_authorized": False,
            }
            for name, frame in controls.items()
        },
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
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
