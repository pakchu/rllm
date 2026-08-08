"""Build source-only HVRBR-12 clocks before opening outcomes or Gross9."""
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
from training import preregister_high_volatility_renewal_breakout_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.long_regime_combo_scan import LongComboScanConfig, _load_market


PREREG_SHA = "8347d92eb4dcbc835dce97aefff6dee14b7f23e71773017856fc3dc9ce434123"
MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
CLOCK = Path("data/high_volatility_renewal_breakout_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_renewal_breakout_relay_controls_2023_2026")
SNAPSHOT = Path(
    "data/high_volatility_renewal_breakout_relay_sources_2023_2026/renewal_scores.csv.gz"
)
RESULT = Path("results/high_volatility_renewal_breakout_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_quiet_spell",
    "no_renewal_crossing",
    "one_hour_stale_direction",
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
        raise RuntimeError("HVRBR combined market continuity drift")
    return combined, {
        "historical_rows": len(historical),
        "live_rows": len(live),
        "combined_rows": len(combined),
        "first": str(combined.date.iloc[0]),
        "last": str(combined.date.iloc[-1]),
        "mode": "hash_bound_historical_cache_plus_postgres_completed_bar_extension",
    }


def score_snapshot(market: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(market.date, utc=True))
    opens = pd.to_numeric(market.open, errors="coerce")
    closes = pd.to_numeric(market.close, errors="coerce")
    five_minute_returns = np.log(closes).diff()
    realized_variation = np.sqrt(
        five_minute_returns.pow(2).rolling(12, min_periods=12).sum()
    )
    positions = np.flatnonzero(dates.minute.to_numpy() == 55).astype(np.int64)
    positions = positions[positions >= 11]
    complete = dates[positions] - dates[positions - 11] == pd.Timedelta(minutes=55)
    positions = positions[np.asarray(complete)]
    hourly = pd.DataFrame(
        {
            "position": positions,
            "decision_bar_time": dates[positions].to_numpy(),
            "hour_return": np.log(
                closes.iloc[positions].to_numpy(float) / opens.iloc[positions - 11].to_numpy(float)
            ),
            "hourly_realized_variation": realized_variation.iloc[positions].to_numpy(float),
        }
    )
    rv = hourly.hourly_realized_variation
    hourly["prior_q50"] = rv.shift(1).rolling(2160, min_periods=1440).quantile(0.50)
    hourly["prior_q90"] = rv.shift(1).rolling(2160, min_periods=1440).quantile(0.90)
    quiet = pd.Series(True, index=hourly.index)
    for lag in range(1, 7):
        quiet &= rv.shift(lag).le(hourly.prior_q50.shift(lag))
    hourly["quiet_six"] = quiet
    hourly["previous_below_q90"] = rv.shift(1).lt(hourly.prior_q90.shift(1))
    return hourly[
        hourly.decision_bar_time >= pd.Timestamp("2023-07-01T00:00:00Z")
    ].reset_index(drop=True)


def build_clock(scores: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    finite = np.isfinite(
        scores[["hour_return", "hourly_realized_variation", "prior_q50", "prior_q90"]]
    ).all(axis=1)
    tail = scores.hourly_realized_variation.ge(scores.prior_q90)
    quiet = scores.quiet_six.astype(bool)
    crossing = scores.previous_below_q90.astype(bool)
    if control == "no_quiet_spell":
        quiet[:] = True
    if control == "no_renewal_crossing":
        crossing[:] = True
    onset = finite & scores.hour_return.ne(0.0) & tail & quiet & crossing
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in scores.index[onset]:
        item = scores.loc[index]
        entry = pd.Timestamp(item.decision_bar_time) + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=12)
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
        direction_value = item.hour_return
        if control == "one_hour_stale_direction":
            if index == 0 or not np.isfinite(scores.loc[index - 1, "hour_return"]):
                continue
            direction_value = scores.loc[index - 1, "hour_return"]
        if direction_value == 0.0:
            continue
        side = 1 if direction_value > 0.0 else -1
        if control == "direction_flip":
            side *= -1
        next_allowed = exit_time
        rows.append(
            {
                "candidate": "HVRBR-12",
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
        raise RuntimeError("HVRBR preregistration drift")
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
            (f"{name}_side_balance", item["minority_side_share"] >= 0.2),
            (f"{name}_month_concentration", item["max_month_share"] <= 0.45),
        )
    }
    passed = all(checks.values())
    core = {
        "protocol_version": "hvrbr_12_source_support_v1",
        "policy_id": "HVRBR-12",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source": source,
        "feature_contract": {
            "reference_hours": 2160,
            "minimum_reference_hours": 1440,
            "quiet_hours": 6,
            "tail_quantile": 0.90,
            "quiet_quantile": 0.50,
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
