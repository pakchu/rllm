"""Build source-only JVFCR-8 clocks without post-entry BTC outcomes."""
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
from training import preregister_jump_variation_flow_continuation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


CLOCK = Path("data/jump_variation_flow_continuation_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/jump_variation_flow_continuation_relay_controls_2023_2026")
SNAPSHOT = Path(
    "data/jump_variation_flow_continuation_relay_sources_2023_2026/signal_features.csv.gz"
)
RESULT = Path("results/jump_variation_flow_continuation_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("jump_only", "no_flow_recovery", "one_bar_stale_features", "direction_flip")
ECONOMIC_OUTCOMES_AUTHORIZED = False


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def feature_snapshot(market: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(market["date"], utc=True)
    close = pd.to_numeric(market["close"], errors="coerce")
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    returns = np.log(close.where(close > 0)).diff()
    realized = returns.pow(2).rolling(72, min_periods=72).sum()
    bipower = (np.pi / 2.0) * (
        returns.abs() * returns.shift(1).abs()
    ).rolling(72, min_periods=72).sum()
    jump = (realized - bipower).clip(lower=0.0)
    imbalance = (2.0 * buy / quote.where(quote > 0) - 1.0).clip(-1.0, 1.0)
    out = pd.DataFrame(
        {
            "date": dates,
            "jv_jump_ratio_72": jump / realized.where(realized > 0),
            "jv_signed_jump_72": returns.pow(3).rolling(72, min_periods=72).sum()
            / realized.where(realized > 0).pow(1.5),
            "jv_flow_recovery": imbalance.rolling(12, min_periods=12).mean()
            - imbalance.rolling(48, min_periods=48).mean(),
        }
    )
    return out.replace([np.inf, -np.inf], np.nan)


def state_signals(snapshot: pd.DataFrame, control: str = "primary") -> tuple[np.ndarray, np.ndarray]:
    reg = prereg.build()["frozen_states"]
    frame = snapshot.shift(1) if control == "one_bar_stale_features" else snapshot
    jump = pd.to_numeric(frame["jv_jump_ratio_72"], errors="coerce").to_numpy(float)
    signed = pd.to_numeric(frame["jv_signed_jump_72"], errors="coerce").to_numpy(float)
    flow = pd.to_numeric(frame["jv_flow_recovery"], errors="coerce").to_numpy(float)
    long_gates = reg["long_jump_continuation"]["gates"]
    short_gates = reg["short_jump_continuation"]["gates"]
    jump_threshold = float(long_gates[0]["threshold"])
    if control == "jump_only":
        long_signal = (jump >= jump_threshold) & (signed > 0)
        short_signal = (jump >= jump_threshold) & (signed < 0)
    else:
        long_signal = (jump >= jump_threshold) & (signed >= float(long_gates[1]["threshold"]))
        short_signal = (jump >= jump_threshold) & (signed <= float(short_gates[1]["threshold"]))
        if control != "no_flow_recovery":
            long_signal &= flow >= float(long_gates[2]["threshold"])
            short_signal &= flow <= float(short_gates[2]["threshold"])
    dates = pd.to_datetime(snapshot["date"], utc=True)
    grid = dates.dt.minute.mod(30).eq(0).to_numpy(bool)
    finite = np.isfinite(jump) & np.isfinite(signed)
    if control not in ("jump_only", "no_flow_recovery"):
        finite &= np.isfinite(flow)
    return long_signal & grid & finite, short_signal & grid & finite


def build_clock(snapshot: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    long_signal, short_signal = state_signals(snapshot, control)
    dates = pd.to_datetime(snapshot["date"], utc=True)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for position in np.flatnonzero(long_signal | short_signal):
        if long_signal[position] and short_signal[position]:
            continue
        decision = dates.iloc[position] + pd.Timedelta(minutes=5)
        entry = decision
        exit_time = entry + pd.Timedelta(hours=8)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        next_allowed = exit_time
        side = 1 if long_signal[position] else -1
        rows.append(
            {
                "candidate": "JVFCR-8",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": -side if control == "direction_flip" else side,
            }
        )
    return pd.DataFrame(
        rows,
        columns=("candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side"),
    )


def query_snapshot() -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = month.Config(
        start="2023-07-01T00:00:00Z",
        end="2026-08-01T00:00:00Z",
        asof="2026-08-01T00:02:00Z",
        lookback_minutes=1_650_000,
    )
    market, _features, _funding, engine = asyncio.run(month._query_frames(cfg))
    if engine is not None:
        engine.dispose()
    required = {"date", "close", "quote_asset_volume", "taker_buy_quote"}
    missing = sorted(required - set(market.columns))
    if missing:
        raise RuntimeError(f"JVFCR market columns missing: {missing}")
    snapshot = feature_snapshot(market)
    snapshot = snapshot[
        (snapshot.date >= pd.Timestamp("2023-06-20T00:00:00Z"))
        & (snapshot.date < pd.Timestamp("2026-08-01T00:00:00Z"))
    ].reset_index(drop=True)
    if snapshot.date.duplicated().any() or not snapshot.date.is_monotonic_increasing:
        raise RuntimeError("JVFCR snapshot time drift")
    return snapshot, {
        "mode": "postgres_live_feature_builder_completed_5m_bar",
        "rows": len(snapshot),
        "first": str(snapshot.date.iloc[0]),
        "last": str(snapshot.date.iloc[-1]),
        "signal_dependent_table": "bars_binance",
        "symbol": "BTCUSDT",
        "post_entry_prices_opened": False,
    }


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    rows = clock[clock.split.eq(split)]
    if rows.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(rows.side.eq(1).sum())
    shorts = int(rows.side.eq(-1).sum())
    months = pd.to_datetime(rows.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(rows),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(rows),
        "max_month_share": int(months.max()) / len(rows),
    }


def run() -> dict[str, Any]:
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    snapshot, source = query_snapshot()
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(snapshot, SNAPSHOT)
    primary = build_clock(snapshot)
    controls = {name: build_clock(snapshot, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, item in support.items():
        checks[f"{name}_minimum_events"] = item["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = item["minority_side_share"] >= 0.2
        checks[f"{name}_month_concentration"] = item["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "jvfcr_8_source_support_v1",
        "policy_id": "JVFCR-8",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha256(prereg.DEFAULT_OUTPUT), "manifest_hash": registration["manifest_hash"]},
        "source": source,
        "source_snapshot": {"path": str(SNAPSHOT), "sha256": sha256(SNAPSHOT), "rows": len(snapshot)},
        "completed_preentry_sources_opened": True,
        "btc_postentry_return_or_pnl_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "controls": {
            name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False}
            for name, frame in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": ECONOMIC_OUTCOMES_AUTHORIZED,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    report = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
