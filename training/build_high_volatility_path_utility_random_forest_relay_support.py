"""Build OOS source-only HVPRF-48 clocks after the model freeze."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training import backtest_all_alpha_month as month
from training import preregister_high_volatility_path_utility_random_forest_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.long_regime_combo_scan import LongComboScanConfig, _load_market
from training.long_regime_interest_gate_validation import build_interest_features


PREREG_SHA = "fa99c46a6811985cacd516c45e7f5b4c9d1f3f19a132a2ce7962a89851096b5a"
MODEL_FREEZE = Path(
    "results/high_volatility_path_utility_random_forest_relay_model_freeze_2026-08-09.json"
)
MODEL_FREEZE_SHA = "367861f642fb80f7d178230b15a014a3a918c1e58ed225b725942150ceb96bc8"
MODEL = Path("data/high_volatility_path_utility_random_forest_relay_model_2026-08-09.joblib")
MODEL_SHA = "c2fa4a34cabb83723784d8a1b4bf9781806cc403e9dd128a152c9fb611b4a80a"
CLOCK = Path("data/high_volatility_path_utility_random_forest_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_path_utility_random_forest_relay_controls_2023_2026")
SNAPSHOT = Path(
    "data/high_volatility_path_utility_random_forest_relay_sources_2023_2026/utility_predictions.csv.gz"
)
RESULT = Path(
    "results/high_volatility_path_utility_random_forest_relay_support_2026-08-09.json"
)
MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
FUNDING = "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
PREMIUM = "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_volatility_gate", "no_utility_tail_gate", "one_anchor_stale_features", "direction_flip")
ECONOMIC_OUTCOMES_AUTHORIZED = False


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return prereg.canonical_hash(payload)


def verify_predecessors() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVPRF preregistration drift")
    if sha256(MODEL_FREEZE) != MODEL_FREEZE_SHA or sha256(MODEL) != MODEL_SHA:
        raise RuntimeError("HVPRF model predecessor drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    freeze = json.loads(MODEL_FREEZE.read_text())
    freeze_core = {key: value for key, value in freeze.items() if key != "manifest_hash"}
    if freeze.get("manifest_hash") != canonical_hash(freeze_core):
        raise RuntimeError("HVPRF model freeze manifest drift")
    if (
        freeze.get("advance_to_oos_source_support") is not True
        or freeze.get("oos_source_incidence_opened") is not False
        or freeze.get("oos_post_entry_outcomes_opened") is not False
    ):
        raise RuntimeError("HVPRF model freeze state drift")
    artifact = joblib.load(MODEL)
    if artifact.get("policy_id") != "HVPRF-48":
        raise RuntimeError("HVPRF model policy drift")
    return registration, freeze, artifact


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
    live_market, _live_features, _funding, engine = asyncio.run(month._query_frames(cfg))
    if engine is not None:
        engine.dispose()
    live_market = live_market.copy()
    live_market["date"] = pd.to_datetime(live_market["date"], utc=True)
    combined = pd.concat([historical, live_market], ignore_index=True, sort=False)
    combined = (
        combined.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    combined = combined[combined.date < pd.Timestamp("2026-08-01T00:00:00Z")].reset_index(drop=True)
    if combined.date.duplicated().any() or not combined.date.is_monotonic_increasing:
        raise RuntimeError("HVPRF combined market time drift")
    deltas = combined.date.diff().dropna()
    if not deltas.eq(pd.Timedelta(minutes=5)).all():
        raise RuntimeError("HVPRF combined market is not continuous 5m")
    return combined, {
        "historical_rows": len(historical),
        "live_rows": len(live_market),
        "combined_rows": len(combined),
        "first": str(combined.date.iloc[0]),
        "last": str(combined.date.iloc[-1]),
        "mode": "hash_bound_historical_cache_plus_postgres_completed_bar_extension",
    }


def score_snapshot(market: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    base = build_market_feature_frame(market, window_size=144)
    features = pd.concat([base, build_interest_features(market, base)], axis=1)
    features = features.loc[:, ~features.columns.duplicated(keep="last")]
    ordered = list(artifact["ordered_features"])
    missing = sorted(set(ordered) - set(features.columns))
    if missing:
        raise RuntimeError(f"HVPRF OOS feature drift: {missing}")
    matrix = features.loc[:, ordered].to_numpy(float)
    matrix[~np.isfinite(matrix)] = np.nan
    offset = int(artifact["anchor_offset"])
    stride = int(artifact["anchor_stride"])
    positions = np.arange(offset, len(market), stride, dtype=np.int64)
    dates = pd.to_datetime(market["date"], utc=True)
    oos = (dates.iloc[positions] >= pd.Timestamp("2023-07-01T00:00:00Z")).to_numpy(bool)
    positions = positions[oos]
    long_predictions = artifact["long_pipeline"].predict(matrix[positions])
    short_predictions = artifact["short_pipeline"].predict(matrix[positions])
    range_vol = pd.to_numeric(features["range_vol"], errors="coerce").to_numpy(float)[positions]
    return pd.DataFrame(
        {
            "position": positions,
            "decision_bar_time": dates.iloc[positions].to_numpy(),
            "predicted_long_utility": long_predictions,
            "predicted_short_utility": short_predictions,
            "range_vol": range_vol,
        }
    )


def build_clock(scores: pd.DataFrame, artifact: dict[str, Any], control: str = "primary") -> pd.DataFrame:
    frame = scores.copy()
    if control == "one_anchor_stale_features":
        frame[["predicted_long_utility", "predicted_short_utility", "range_vol"]] = frame[
            ["predicted_long_utility", "predicted_short_utility", "range_vol"]
        ].shift(1)
    long_prediction = pd.to_numeric(frame.predicted_long_utility, errors="coerce").to_numpy(float)
    short_prediction = pd.to_numeric(frame.predicted_short_utility, errors="coerce").to_numpy(float)
    volatility = pd.to_numeric(frame.range_vol, errors="coerce").to_numpy(float)
    long_excess = (long_prediction - float(artifact["long_utility_threshold"])) / float(artifact["long_utility_iqr"])
    short_excess = (short_prediction - float(artifact["short_utility_threshold"])) / float(artifact["short_utility_iqr"])
    if control == "no_utility_tail_gate":
        long_signal = long_excess > short_excess
        short_signal = short_excess > long_excess
    else:
        long_eligible = long_prediction >= float(artifact["long_utility_threshold"])
        short_eligible = short_prediction >= float(artifact["short_utility_threshold"])
        long_signal = long_eligible & (~short_eligible | (long_excess > short_excess))
        short_signal = short_eligible & (~long_eligible | (short_excess > long_excess))
    if control != "no_volatility_gate":
        high_volatility = volatility >= float(artifact["range_vol_threshold"])
        long_signal &= high_volatility
        short_signal &= high_volatility
    finite = np.isfinite(long_prediction) & np.isfinite(short_prediction) & np.isfinite(volatility)
    active = (long_signal | short_signal) & finite
    decisions = pd.to_datetime(frame.decision_bar_time, utc=True) + pd.Timedelta(minutes=5)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in np.flatnonzero(active):
        if long_signal[index] and short_signal[index]:
            continue
        entry = decisions.iloc[index]
        exit_time = entry + pd.Timedelta(hours=48)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        next_allowed = exit_time
        side = 1 if long_signal[index] else -1
        rows.append(
            {
                "candidate": "HVPRF-48",
                "control": control,
                "split": split,
                "decision_time": entry,
                "feature_available_time": entry,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": -side if control == "direction_flip" else side,
            }
        )
    return pd.DataFrame(
        rows,
        columns=("candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side"),
    )


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
    registration, freeze, artifact = verify_predecessors()
    market, source = load_combined_market()
    scores = score_snapshot(market, artifact)
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(scores, SNAPSHOT)
    primary = build_clock(scores, artifact)
    controls = {name: build_clock(scores, artifact, name) for name in CONTROLS}
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
        "protocol_version": "hvprf_48_oos_source_support_v1",
        "policy_id": "HVPRF-48",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "model_freeze": {"path": str(MODEL_FREEZE), "sha256": MODEL_FREEZE_SHA, "manifest_hash": freeze["manifest_hash"], "model_sha256": MODEL_SHA},
        "source": source,
        "source_snapshot": {"path": str(SNAPSHOT), "sha256": sha256(SNAPSHOT), "rows": len(scores)},
        "oos_source_incidence_opened": True,
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
