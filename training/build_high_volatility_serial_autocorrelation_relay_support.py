"""Build source-only HVSAR-12 clocks before opening outcomes or Gross9."""
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
from training import preregister_high_volatility_serial_autocorrelation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.long_regime_combo_scan import LongComboScanConfig, _load_market


PREREG_SHA = "82407cce2a8b6c74a7ac9a849be070697431cc80ce3eaae4c054d5c7ad49321d"
MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
FUNDING = "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
PREMIUM = "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
CLOCK = Path("data/high_volatility_serial_autocorrelation_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_serial_autocorrelation_relay_controls_2023_2026")
SNAPSHOT = Path("data/high_volatility_serial_autocorrelation_relay_sources_2023_2026/serial_autocorrelation.csv.gz")
RESULT = Path("results/high_volatility_serial_autocorrelation_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_variation_gate",
    "no_autocorrelation_tail_gate",
    "one_boundary_stale_features",
    "fixed_momentum",
    "fixed_reversal",
    "direction_flip",
)
SCORE_COLUMNS = (
    "decision_time",
    "completed_return_12h",
    "realized_variation",
    "lag_one_autocorrelation",
    "absolute_autocorrelation",
    "variation_rank",
    "absolute_autocorrelation_rank",
    "source_valid",
)
CLOCK_COLUMNS = (
    "candidate",
    "control",
    "split",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "completed_return_12h",
    "realized_variation",
    "lag_one_autocorrelation",
    "variation_rank",
    "absolute_autocorrelation_rank",
)


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
        raise RuntimeError("HVSAR combined market continuity drift")
    return combined, {
        "historical_rows": len(historical),
        "live_rows": len(live),
        "combined_rows": len(combined),
        "first": str(combined.date.iloc[0]),
        "last": str(combined.date.iloc[-1]),
        "mode": "hash_bound_historical_cache_plus_postgres_completed_bar_extension",
    }


def _strict_prior_midrank(values: np.ndarray, lookback: int = 270, minimum: int = 252) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=float)
    history: list[float] = []
    for index, current in enumerate(values):
        prior = np.asarray(history[-lookback:], dtype=float)
        if math.isfinite(float(current)) and len(prior) >= minimum:
            output[index] = (np.sum(prior < current) + 0.5 * np.sum(prior == current)) / len(prior)
        if math.isfinite(float(current)):
            history.append(float(current))
    return output


def score_snapshot(market: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(market.date, utc=True)
    close = pd.to_numeric(market.close, errors="coerce").to_numpy(float)
    index_by_time = {timestamp: index for index, timestamp in enumerate(dates)}
    first_decision = dates.iloc[0].ceil("12h")
    last_decision = pd.Timestamp("2026-08-01T00:00:00Z")
    decisions = pd.date_range(first_decision, last_decision, freq="12h", inclusive="left")
    rows: list[dict[str, Any]] = []
    for decision in decisions:
        end_index = index_by_time.get(decision - pd.Timedelta(minutes=5))
        source_valid = end_index is not None and end_index >= 144
        completed_return = variation = autocorrelation = np.nan
        if source_valid:
            path = close[end_index - 144 : end_index + 1]
            source_valid = bool(len(path) == 145 and np.isfinite(path).all() and np.all(path > 0.0))
            if source_valid:
                returns = np.diff(np.log(path))
                left, right = returns[:-1], returns[1:]
                source_valid = bool(np.std(left, ddof=1) > 0.0 and np.std(right, ddof=1) > 0.0)
                if source_valid:
                    completed_return = float(np.sum(returns))
                    variation = float(np.sqrt(np.sum(np.square(returns))))
                    autocorrelation = float(np.corrcoef(left, right)[0, 1])
                    source_valid = bool(
                        math.isfinite(completed_return)
                        and math.isfinite(variation)
                        and variation > 0.0
                        and math.isfinite(autocorrelation)
                    )
        rows.append(
            {
                "decision_time": decision,
                "completed_return_12h": completed_return,
                "realized_variation": variation,
                "lag_one_autocorrelation": autocorrelation,
                "source_valid": source_valid,
            }
        )
    frame = pd.DataFrame(rows)
    frame["absolute_autocorrelation"] = frame.lag_one_autocorrelation.abs()
    rankable_variation = frame.realized_variation.where(frame.source_valid)
    rankable_autocorrelation = frame.absolute_autocorrelation.where(frame.source_valid)
    frame["variation_rank"] = _strict_prior_midrank(rankable_variation.to_numpy(float))
    frame["absolute_autocorrelation_rank"] = _strict_prior_midrank(
        rankable_autocorrelation.to_numpy(float)
    )
    return frame.loc[
        frame.decision_time >= pd.Timestamp("2023-07-01T00:00:00Z"), SCORE_COLUMNS
    ].reset_index(drop=True)


def build_clock(scores: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    frame = scores.copy()
    if control == "one_boundary_stale_features":
        feature_columns = [
            "completed_return_12h",
            "realized_variation",
            "lag_one_autocorrelation",
            "absolute_autocorrelation",
            "variation_rank",
            "absolute_autocorrelation_rank",
            "source_valid",
        ]
        frame[feature_columns] = frame[feature_columns].shift(1)
    valid = (
        frame.source_valid.fillna(False).astype(bool)
        & np.isfinite(frame.completed_return_12h)
        & frame.completed_return_12h.ne(0)
        & np.isfinite(frame.lag_one_autocorrelation)
        & frame.lag_one_autocorrelation.ne(0)
        & np.isfinite(frame.variation_rank)
        & np.isfinite(frame.absolute_autocorrelation_rank)
    )
    variation_gate = frame.variation_rank.ge(0.70)
    autocorrelation_gate = frame.absolute_autocorrelation_rank.ge(0.75)
    if control == "no_variation_gate":
        variation_gate[:] = True
    if control == "no_autocorrelation_tail_gate":
        autocorrelation_gate[:] = True
    active = valid & variation_gate & autocorrelation_gate
    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(active.to_numpy(bool)):
        decision = pd.Timestamp(frame.decision_time.iloc[index])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=12)
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        return_side = 1 if frame.completed_return_12h.iloc[index] > 0 else -1
        dependence_side = 1 if frame.lag_one_autocorrelation.iloc[index] > 0 else -1
        side = return_side * dependence_side
        if control == "fixed_momentum":
            side = return_side
        elif control == "fixed_reversal":
            side = -return_side
        elif control == "direction_flip":
            side = -side
        rows.append(
            {
                "candidate": "HVSAR-12",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": side,
                "completed_return_12h": float(frame.completed_return_12h.iloc[index]),
                "realized_variation": float(frame.realized_variation.iloc[index]),
                "lag_one_autocorrelation": float(frame.lag_one_autocorrelation.iloc[index]),
                "variation_rank": float(frame.variation_rank.iloc[index]),
                "absolute_autocorrelation_rank": float(frame.absolute_autocorrelation_rank.iloc[index]),
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


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
        raise RuntimeError("HVSAR preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    market, source = load_combined_market()
    scores = score_snapshot(market)
    primary = build_clock(scores)
    controls = {name: build_clock(scores, name) for name in CONTROLS}
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
            (f"{name}_side_balance", item["minority_side_share"] >= 0.20),
            (f"{name}_month_concentration", item["max_month_share"] <= 0.45),
        )
    }
    passed = all(checks.values())
    core = {
        "protocol_version": "hvsar_12_source_support_v1",
        "policy_id": "HVSAR-12",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source": source,
        "ranking": {
            "lookback_boundaries": 270,
            "minimum_prior_boundaries": 252,
            "current_excluded": True,
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
