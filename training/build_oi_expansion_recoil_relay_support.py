"""Build source-only OIER-8 clocks; never compute post-entry BTC outcomes."""
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
from training import preregister_oi_expansion_recoil_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


CLOCK = Path("data/oi_expansion_recoil_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/oi_expansion_recoil_relay_controls_2023_2026")
SNAPSHOT = Path("data/oi_expansion_recoil_relay_sources_2023_2026/signal_features.csv.gz")
RESULT = Path("results/oi_expansion_recoil_relay_support_2026-08-09.json")
SPLITS = {"train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")), "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")), "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")), "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z"))}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_range_vol_gate", "no_rsi_gate", "one_stride_stale_features", "direction_flip", "same_clock_forced_long")
FEATURES = ("oi_ret_4h_z", "px_ret_4h_z", "range_vol", "rsi_norm")
ECONOMIC_OUTCOMES_AUTHORIZED = False
PREREG_SHA = "f1261431cded1858134707b05549a349cf6601c0ffdd86978ea75f4d7cd2861d"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def interval_slots(dates: pd.Series, stride: int = 6) -> np.ndarray:
    return month._interval_slots(dates, stride, month._research_offset(stride))


def state_signals(snapshot: pd.DataFrame, control: str = "primary") -> tuple[np.ndarray, np.ndarray]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    oi = pd.to_numeric(snapshot.oi_ret_4h_z, errors="coerce").to_numpy(float)
    price = pd.to_numeric(snapshot.px_ret_4h_z, errors="coerce").to_numpy(float)
    range_vol = pd.to_numeric(snapshot.range_vol, errors="coerce").to_numpy(float)
    rsi = pd.to_numeric(snapshot.rsi_norm, errors="coerce").to_numpy(float)
    finite = np.isfinite(oi) & np.isfinite(price) & np.isfinite(range_vol) & np.isfinite(rsi)
    side = np.where(np.isfinite(price), -np.sign(price), 0).astype(int)
    active = finite & (oi >= 0.8954018630586817) & (np.abs(price) >= 0.7389570664259131)
    if control != "no_range_vol_gate": active &= range_vol >= 0.04008415457867338
    if control != "no_rsi_gate": active &= side * rsi <= -0.04507656773717145
    dates = pd.to_datetime(snapshot["date"], utc=True).dt.tz_convert(None)
    active &= interval_slots(dates, 6)
    if control == "one_stride_stale_features":
        active = np.roll(active, 6); side = np.roll(side, 6); active[:6] = False; side[:6] = 0
    long_signal = active & (side == 1); short_signal = active & (side == -1)
    return long_signal, short_signal

def build_clock(snapshot: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    long_signal, short_signal = state_signals(snapshot, control)
    dates = pd.to_datetime(snapshot["date"], utc=True)
    rows: list[dict[str, Any]] = []; next_allowed: pd.Timestamp | None = None
    for position in np.flatnonzero(long_signal | short_signal):
        decision = dates.iloc[position] + pd.Timedelta(minutes=5); entry = decision
        side = 1 if long_signal[position] else -1; exit_time = entry + pd.Timedelta(hours=8)
        if next_allowed is not None and entry < next_allowed: continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None: continue
        next_allowed = exit_time
        output_side = -side if control == "direction_flip" else 1 if control == "same_clock_forced_long" else side
        rows.append({"candidate": "OIER-8", "control": control, "split": split, "decision_time": decision, "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time, "side": output_side, "state": "long_recoil" if side == 1 else "short_recoil"})
    return pd.DataFrame(rows, columns=("candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side", "state"))

def query_snapshot() -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = month.Config(start="2023-07-01T00:00:00Z", end="2026-08-01T00:00:00Z", asof="2026-08-01T00:02:00Z", lookback_minutes=1_650_000)
    market, feature_frame, _funding, engine = asyncio.run(month._query_frames(cfg))
    if engine is not None: engine.dispose()
    missing = sorted(set(FEATURES) - set(feature_frame.columns))
    if missing: raise RuntimeError(f"OIER feature columns missing: {missing}")
    dates = pd.to_datetime(market["date"], utc=True)
    snapshot = pd.DataFrame({"date": dates, **{name: pd.to_numeric(feature_frame[name], errors="coerce") for name in FEATURES}})
    snapshot = snapshot[(snapshot.date >= pd.Timestamp("2023-06-20T00:00:00Z")) & (snapshot.date < pd.Timestamp("2026-08-01T00:00:00Z"))].reset_index(drop=True)
    if snapshot.date.duplicated().any() or not snapshot.date.is_monotonic_increasing: raise RuntimeError("OIER snapshot time drift")
    return snapshot, {"mode": "postgres_live_feature_builder_completed_bar", "rows": len(snapshot), "first": str(snapshot.date.iloc[0]), "last": str(snapshot.date.iloc[-1]), "signal_dependent_tables": ["bars_binance", "open_interest_binance"]}


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    rows = clock[clock.split.eq(split)]
    if rows.empty: return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(rows.side.eq(1).sum()); shorts = int(rows.side.eq(-1).sum()); months = pd.to_datetime(rows.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(rows), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(rows), "max_month_share": int(months.max()) / len(rows)}


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA: raise RuntimeError("OIER preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); prereg.validate(registration)
    snapshot, source = query_snapshot(); SNAPSHOT.parent.mkdir(parents=True, exist_ok=True); _write_gzip_csv(snapshot, SNAPSHOT)
    primary = build_clock(snapshot); controls = {name: build_clock(snapshot, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True); _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items(): _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}; checks = {}
    for name, item in support.items(): checks[f"{name}_minimum_events"] = item["events"] >= MINIMUM[name]; checks[f"{name}_side_balance"] = item["minority_side_share"] >= 0.2; checks[f"{name}_month_concentration"] = item["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {"protocol_version": "oier_8_source_support_v1", "policy_id": "OIER-8", "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha256(prereg.DEFAULT_OUTPUT), "manifest_hash": registration["manifest_hash"]}, "source": source, "source_snapshot": {"path": str(SNAPSHOT), "sha256": sha256(SNAPSHOT), "rows": len(snapshot)}, "completed_preentry_sources_opened": True, "btc_postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()}, "support": support, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": ECONOMIC_OUTCOMES_AUTHORIZED, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    result = {**core, "manifest_hash": canonical_hash(core)}; RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); report = run(); print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
