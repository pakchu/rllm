"""Build source-only HVSER-12 clocks before opening outcomes or Gross9."""
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
from training import preregister_high_volatility_sign_entropy_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.long_regime_combo_scan import LongComboScanConfig, _load_market


PREREG_SHA = "9ab221716d525997b4da303ee01afebe0336436d92d01e0b18a51db4e97f5854"
MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
FUNDING = "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
PREMIUM = "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
CLOCK = Path("data/high_volatility_sign_entropy_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_sign_entropy_relay_controls_2023_2026")
SNAPSHOT = Path("data/high_volatility_sign_entropy_relay_sources_2023_2026/sign_entropy.csv.gz")
RESULT = Path("results/high_volatility_sign_entropy_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_entropy_gate", "no_volatility_gate", "one_anchor_stale_features", "direction_flip")


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
        raise RuntimeError("HVSER combined market continuity drift")
    return combined, {
        "historical_rows": len(historical),
        "live_rows": len(live),
        "combined_rows": len(combined),
        "first": str(combined.date.iloc[0]),
        "last": str(combined.date.iloc[-1]),
        "mode": "hash_bound_historical_cache_plus_postgres_completed_bar_extension",
    }


def score_snapshot(market: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    opens = pd.to_numeric(market.open, errors="coerce")
    returns = np.log(opens / opens.shift(1))
    valid = returns.notna() & returns.ne(0)
    positive = returns.gt(0).where(valid).astype(float)
    count = valid.astype(int).rolling(288, min_periods=288).sum()
    positive_count = positive.fillna(0).rolling(288, min_periods=288).sum()
    probability = positive_count / count.replace(0, np.nan)
    entropy = -(probability * np.log2(probability) + (1 - probability) * np.log2(1 - probability))
    entropy = entropy.where((probability > 0) & (probability < 1), 0.0).where(count >= 276)
    direction_return = opens / opens.shift(288) - 1.0
    range_vol = pd.to_numeric(build_market_feature_frame(market, window_size=144)["range_vol"], errors="coerce")
    positions = np.arange(143, len(market), 72, dtype=np.int64)
    dates = pd.to_datetime(market.date, utc=True)
    frame = pd.DataFrame(
        {
            "position": positions,
            "decision_bar_time": dates.iloc[positions].to_numpy(),
            "sign_entropy": entropy.iloc[positions].to_numpy(float),
            "range_vol": range_vol.iloc[positions].to_numpy(float),
            "direction_return_24h": direction_return.iloc[positions].to_numpy(float),
            "valid_sign_count": count.iloc[positions].to_numpy(float),
        }
    )
    calibration = frame[
        frame.decision_bar_time.ge(pd.Timestamp("2023-01-01T00:00:00Z"))
        & frame.decision_bar_time.lt(pd.Timestamp("2023-07-01T00:00:00Z"))
    ]
    calibration = calibration[
        calibration.valid_sign_count.ge(276)
        & np.isfinite(calibration.sign_entropy)
        & np.isfinite(calibration.range_vol)
    ]
    if len(calibration) < 500:
        raise RuntimeError("HVSER source-only calibration floor failed")
    thresholds = {
        "sign_entropy_q35": float(calibration.sign_entropy.quantile(0.35)),
        "range_vol_q65": float(calibration.range_vol.quantile(0.65)),
    }
    return frame[frame.decision_bar_time.ge(pd.Timestamp("2023-07-01T00:00:00Z"))].reset_index(drop=True), thresholds


def build_clock(scores: pd.DataFrame, thresholds: dict[str, float], control: str = "primary") -> pd.DataFrame:
    frame = scores.copy()
    if control == "one_anchor_stale_features":
        frame[["sign_entropy", "range_vol", "direction_return_24h", "valid_sign_count"]] = frame[
            ["sign_entropy", "range_vol", "direction_return_24h", "valid_sign_count"]
        ].shift(1)
    valid = (
        frame.valid_sign_count.ge(276)
        & np.isfinite(frame.sign_entropy)
        & np.isfinite(frame.range_vol)
        & np.isfinite(frame.direction_return_24h)
        & frame.direction_return_24h.ne(0)
    )
    entropy_gate = frame.sign_entropy.le(thresholds["sign_entropy_q35"])
    volatility_gate = frame.range_vol.ge(thresholds["range_vol_q65"])
    if control == "no_entropy_gate":
        entropy_gate[:] = True
    if control == "no_volatility_gate":
        volatility_gate[:] = True
    active = valid & entropy_gate & volatility_gate
    decisions = pd.to_datetime(frame.decision_bar_time, utc=True) + pd.Timedelta(minutes=5)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in np.flatnonzero(active.to_numpy(bool)):
        entry = decisions.iloc[index]
        exit_time = entry + pd.Timedelta(hours=12)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        next_allowed = exit_time
        side = 1 if frame.direction_return_24h.iloc[index] > 0 else -1
        rows.append(
            {
                "candidate": "HVSER-12",
                "control": control,
                "split": split,
                "decision_time": entry,
                "feature_available_time": entry,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": -side if control == "direction_flip" else side,
            }
        )
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
        raise RuntimeError("HVSER preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core_registration = {key: value for key, value in registration.items() if key != "manifest_hash"}
    if registration["manifest_hash"] != prereg.canonical_hash(core_registration):
        raise RuntimeError("HVSER preregistration manifest drift")
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
        "protocol_version": "hvser_12_source_support_v1",
        "policy_id": "HVSER-12",
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
