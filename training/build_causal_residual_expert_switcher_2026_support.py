"""Build the outcome-blind CRES-1 base-event support clock for 2026H1."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.build_leave_one_out_residual_exhaustion_support import (
    build_feature_panels,
    candidate_frame,
)
from training.export_causal_residual_expert_switcher_2026_sources import (
    CONFIRMATION_START,
    END,
    EXPECTED_PROTOCOL_HASH,
    START,
    SYMBOLS,
)
from training.export_leave_one_out_residual_exhaustion_sources import (
    deterministic_csv_gz,
    sha256_file,
)
from training.preregister_causal_residual_expert_switcher_2026 import canonical_hash


SOURCE_MANIFEST = "results/causal_residual_expert_switcher_2026_source_manifest_2026-07-17.json"
EXPECTED_SOURCE_MANIFEST_HASH = "d1fc86b0084727c09451b89b412c685822611a5a567b889a88f9d2e82b0a8bfb"
DEFAULT_SOURCE_DIR = "data/binance_um_cres_2025_2026h1"
DEFAULT_CLOCK = "data/causal_residual_expert_switcher_2026_base_clock.csv.gz"
DEFAULT_MANIFEST = "results/causal_residual_expert_switcher_2026_support_manifest_2026-07-17.json"
DEFAULT_DOCS = "docs/causal-residual-expert-switcher-2026-support-2026-07-17.md"

POLICY_ID = "CRES01_BASE"
HORIZON_HOURS = 12
HOLD_HOURS = 12
RISK_LOOKBACK_BARS = 3 * 24 * 12
FORBIDDEN_OUTCOME_TOKENS = {
    "return",
    "pnl",
    "profit",
    "loss",
    "target",
    "label",
    "edge",
    "choice",
    "prediction",
    "future",
}
CLOCK_COLUMNS = (
    "policy_id",
    "signal_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "residual_horizon_hours",
    "hold_hours",
    "continuation_long_symbol",
    "continuation_short_symbol",
    "continuation_long_weight_gross1",
    "continuation_short_weight_abs_gross1",
    "continuation_long_beta",
    "continuation_short_beta",
    "loser_residual_z",
    "winner_residual_z",
    "loser_flow_z",
    "winner_flow_z",
    "setup_score",
    "range_risk",
)


def _load_source_manifest(path: str = SOURCE_MANIFEST) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("manifest_hash") != EXPECTED_SOURCE_MANIFEST_HASH:
        raise RuntimeError("CRES source manifest hash changed")
    body = {key: value for key, value in payload.items() if key not in {"manifest_hash", "created_at"}}
    if canonical_hash(body) != EXPECTED_SOURCE_MANIFEST_HASH:
        raise RuntimeError("CRES source manifest body changed")
    if payload.get("preregistration_protocol_hash") != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("CRES source/preregistration protocol mismatch")
    if payload.get("post_entry_2026_strategy_returns_calculated") is not False:
        raise RuntimeError("CRES source freeze already contains strategy outcomes")
    return payload


def load_hourly_panel(
    source_dir: str = DEFAULT_SOURCE_DIR,
    source_manifest: str = SOURCE_MANIFEST,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, np.ndarray], pd.DatetimeIndex]:
    manifest = _load_source_manifest(source_manifest)
    records = {str(row["symbol"]): row for row in manifest["records"]}
    if set(records) != set(SYMBOLS):
        raise RuntimeError("CRES source symbol set changed")
    panels: dict[str, pd.DataFrame] = {}
    quality: dict[str, pd.Series] = {}
    ranges: dict[str, np.ndarray] = {}
    five_minute_dates: pd.DatetimeIndex | None = None
    for symbol in sorted(SYMBOLS):
        path = Path(source_dir) / f"{symbol}_5m_2025_2026h1.csv.gz"
        if sha256_file(path) != records[symbol]["output_market_sha256"]:
            raise RuntimeError(f"{symbol} CRES frozen source changed")
        raw = pd.read_csv(
            path,
            usecols=[
                "date",
                "high",
                "low",
                "close",
                "quote_asset_volume",
                "taker_buy_quote",
                "tic",
            ],
            parse_dates=["date"],
        ).sort_values("date")
        if raw["date"].duplicated().any() or not raw["tic"].astype(str).eq(symbol).all():
            raise RuntimeError(f"{symbol} CRES source identity/grid failure")
        current_dates = pd.DatetimeIndex(raw["date"])
        expected_five = pd.date_range(START, END - pd.Timedelta(minutes=5), freq="5min")
        if not current_dates.equals(expected_five):
            raise RuntimeError(f"{symbol} CRES source is not the exact physical prefix")
        if five_minute_dates is None:
            five_minute_dates = current_dates
        elif not five_minute_dates.equals(current_dates):
            raise RuntimeError("CRES symbol 5m grids differ")
        raw = raw.set_index("date")
        high = pd.to_numeric(raw["high"], errors="raise")
        low = pd.to_numeric(raw["low"], errors="raise")
        close = pd.to_numeric(raw["close"], errors="raise")
        quote = pd.to_numeric(raw["quote_asset_volume"], errors="raise")
        buy = pd.to_numeric(raw["taker_buy_quote"], errors="raise")
        ranges[symbol] = np.log(high.to_numpy(dtype=float) / low.to_numpy(dtype=float))
        hourly = pd.DataFrame(
            {
                "close": close.resample("1h", closed="left", label="right").last(),
                "quote": quote.resample("1h", closed="left", label="right").sum(),
                "buy": buy.resample("1h", closed="left", label="right").sum(),
                "bar_count": close.resample("1h", closed="left", label="right").count(),
                "positive_quote_bars": (quote > 0.0)
                .resample("1h", closed="left", label="right")
                .sum(),
            }
        )
        clean = (
            hourly["bar_count"].eq(12)
            & hourly["positive_quote_bars"].eq(12)
            & hourly["close"].gt(0.0)
            & hourly["quote"].gt(0.0)
            & hourly["buy"].ge(0.0)
            & hourly["buy"].le(
                hourly["quote"] + np.maximum(1e-6, hourly["quote"] * 1e-10)
            )
        )
        panels[symbol] = hourly[["close", "quote", "buy"]]
        quality[symbol] = clean.rename(symbol)
    assert five_minute_dates is not None
    expected_hourly = pd.date_range(START + pd.Timedelta(hours=1), END, freq="1h")
    common = panels[sorted(SYMBOLS)[0]].index
    if not common.equals(expected_hourly):
        raise RuntimeError("CRES hourly grid escaped physical prefix")
    for panel in panels.values():
        if not panel.index.equals(common):
            raise RuntimeError("CRES hourly symbol grids differ")
    return panels, pd.DataFrame(quality, index=common).astype(bool), ranges, five_minute_dates


def _range_risk(
    ranges: dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
    signal: pd.Timestamp,
    long_symbol: str,
    short_symbol: str,
) -> float:
    end_index = int(dates.searchsorted(signal, side="left"))
    start_index = end_index - RISK_LOOKBACK_BARS
    if start_index < 0:
        return float("nan")
    long_rms = float(
        np.sqrt(np.mean(ranges[long_symbol][start_index:end_index] ** 2))
    )
    short_rms = float(
        np.sqrt(np.mean(ranges[short_symbol][start_index:end_index] ** 2))
    )
    return max(long_rms, short_rms)


def reserve_base_clock(
    candidates: pd.DataFrame,
    joint_quality: pd.Series,
    ranges: dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    use = candidates.copy()
    clean_13 = (
        joint_quality.rolling(HORIZON_HOURS + 1, min_periods=HORIZON_HOURS + 1)
        .sum()
        .eq(HORIZON_HOURS + 1)
    )
    use["eligible"] &= use["signal_time"].map(clean_13).fillna(False).to_numpy(dtype=bool)
    selected: list[dict[str, Any]] = []
    next_signal_allowed = CONFIRMATION_START
    for row in use.loc[use["eligible"]].itertuples(index=False):
        signal = pd.Timestamp(row.signal_time)
        if not CONFIRMATION_START <= signal < END:
            continue
        entry = signal + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=HOLD_HOURS)
        if signal < next_signal_allowed or exit_time >= END:
            continue
        # candidate_frame is mean reversion; freeze its canonical continuation
        # orientation only as an expert reference, not as the selected action.
        continuation_long = str(row.short_symbol)
        continuation_short = str(row.long_symbol)
        risk = _range_risk(
            ranges, dates, signal, continuation_long, continuation_short
        )
        selected.append(
            {
                "policy_id": POLICY_ID,
                "signal_time": signal,
                "feature_available_time": signal,
                "entry_time": entry,
                "exit_time": exit_time,
                "residual_horizon_hours": HORIZON_HOURS,
                "hold_hours": HOLD_HOURS,
                "continuation_long_symbol": continuation_long,
                "continuation_short_symbol": continuation_short,
                "continuation_long_weight_gross1": float(row.short_weight_abs),
                "continuation_short_weight_abs_gross1": float(row.long_weight),
                "continuation_long_beta": float(row.short_beta),
                "continuation_short_beta": float(row.long_beta),
                "loser_residual_z": float(row.loser_residual_z),
                "winner_residual_z": float(row.winner_residual_z),
                "loser_flow_z": float(row.loser_flow_z),
                "winner_flow_z": float(row.winner_flow_z),
                "setup_score": float(row.exhaustion_score),
                "range_risk": risk,
            }
        )
        next_signal_allowed = exit_time
    return pd.DataFrame(selected, columns=CLOCK_COLUMNS)


def assert_clock_contract(clock: pd.DataFrame) -> None:
    if clock.empty:
        raise RuntimeError("empty CRES 2026 base clock")
    for column in clock.columns:
        if set(column.lower().split("_")) & FORBIDDEN_OUTCOME_TOKENS:
            raise RuntimeError(f"outcome-like CRES support column forbidden: {column}")
    for column in ("signal_time", "feature_available_time", "entry_time", "exit_time"):
        clock[column] = pd.to_datetime(clock[column], errors="raise")
    if not clock["policy_id"].eq(POLICY_ID).all():
        raise RuntimeError("CRES base policy identity drift")
    if not (clock["feature_available_time"] == clock["signal_time"]).all():
        raise RuntimeError("CRES feature availability drift")
    if not (clock["entry_time"] == clock["signal_time"] + pd.Timedelta(minutes=5)).all():
        raise RuntimeError("CRES entry delay drift")
    if not (clock["exit_time"] == clock["entry_time"] + pd.Timedelta(hours=HOLD_HOURS)).all():
        raise RuntimeError("CRES hold drift")
    if not (
        (clock["signal_time"] >= CONFIRMATION_START) & (clock["exit_time"] < END)
    ).all():
        raise RuntimeError("CRES base clock escaped confirmation window")
    if not clock["signal_time"].is_monotonic_increasing or clock["signal_time"].duplicated().any():
        raise RuntimeError("CRES base clock ordering drift")
    if (clock["entry_time"].iloc[1:].to_numpy() < clock["exit_time"].iloc[:-1].to_numpy()).any():
        raise RuntimeError("CRES base clock overlaps")
    long_weight = clock["continuation_long_weight_gross1"].to_numpy(dtype=float)
    short_weight = clock["continuation_short_weight_abs_gross1"].to_numpy(dtype=float)
    if not np.allclose(long_weight + short_weight, 1.0, atol=1e-12):
        raise RuntimeError("CRES canonical expert gross drift")
    if (np.minimum(long_weight, short_weight) < 0.25 - 1e-12).any():
        raise RuntimeError("CRES canonical expert minimum leg drift")
    exposure = (
        long_weight * clock["continuation_long_beta"].to_numpy(dtype=float)
        - short_weight * clock["continuation_short_beta"].to_numpy(dtype=float)
    )
    if not np.allclose(exposure, 0.0, atol=1e-12):
        raise RuntimeError("CRES canonical expert is not factor-beta neutral")
    numeric = clock.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not clock["range_risk"].gt(0.0).all():
        raise RuntimeError("CRES base clock has invalid features")


def support_statistics(clock: pd.DataFrame) -> dict[str, Any]:
    pairs = clock["continuation_long_symbol"].astype(str) + "/" + clock[
        "continuation_short_symbol"
    ].astype(str)
    counts = pairs.value_counts()
    checks = {
        "events_at_least_35": len(clock) >= 35,
        "q1_events_at_least_15": int((clock["signal_time"] < pd.Timestamp("2026-04-01")).sum()) >= 15,
        "q2_events_at_least_15": int((clock["signal_time"] >= pd.Timestamp("2026-04-01")).sum()) >= 15,
        "unique_ordered_pairs_at_least_8": int(pairs.nunique()) >= 8,
        "maximum_ordered_pair_share_at_most_0_20": float(counts.max() / len(clock)) <= 0.20,
        "symbols_seen_each_side_at_least_5": min(
            int(clock["continuation_long_symbol"].nunique()),
            int(clock["continuation_short_symbol"].nunique()),
        )
        >= 5,
    }
    return {
        "events": int(len(clock)),
        "q1_events": int((clock["signal_time"] < pd.Timestamp("2026-04-01")).sum()),
        "q2_events": int((clock["signal_time"] >= pd.Timestamp("2026-04-01")).sum()),
        "unique_ordered_pairs": int(pairs.nunique()),
        "maximum_ordered_pair_share": float(counts.max() / len(clock)),
        "continuation_long_symbols": sorted(clock["continuation_long_symbol"].unique()),
        "continuation_short_symbols": sorted(clock["continuation_short_symbol"].unique()),
        "checks": checks,
        "passes_support": all(checks.values()),
    }


def _markdown(manifest: dict[str, Any]) -> str:
    support = manifest["support"]
    return f"""# CRES-1 2026 outcome-blind support

## Decision

- Support gate: **{'PASS' if support['passes_support'] else 'FAIL'}**.
- 2026 post-entry strategy returns calculated: **no**.
- Historical 2023-2025 training seed is physically separate and ends before 2026.

## Base-event clock

- events: {support['events']} (Q1 {support['q1_events']}, Q2 {support['q2_events']});
- unique ordered continuation-reference pairs: {support['unique_ordered_pairs']};
- maximum ordered-pair share: {support['maximum_ordered_pair_share']:.3f};
- event features use only completed hourly bars;
- range risk uses only the 864 completed 5m bars strictly before each signal;
- the clock contains no return, PnL, target, label, edge, prediction, or choice column.

The long/short columns are only the canonical continuation expert reference.
The frozen evaluator must choose continuation, reversion, or flat online using
only the historical seed and earlier 2026 events whose exits are already
observable.

Clock SHA256: `{manifest['clock_sha256']}`
Source manifest hash: `{manifest['source_manifest_hash']}`
Historical seed SHA256: `{manifest['historical_seed_sha256']}`
"""


def run(
    source_dir: str = DEFAULT_SOURCE_DIR,
    source_manifest: str = SOURCE_MANIFEST,
    clock_path: str = DEFAULT_CLOCK,
    manifest_path: str = DEFAULT_MANIFEST,
    docs_path: str = DEFAULT_DOCS,
) -> dict[str, Any]:
    source = _load_source_manifest(source_manifest)
    panels, quality, ranges, five_minute_dates = load_hourly_panel(source_dir, source_manifest)
    features = build_feature_panels(panels, quality, HORIZON_HOURS)
    candidates = candidate_frame(features, HORIZON_HOURS)
    joint_quality = quality.loc[:, sorted(SYMBOLS)].all(axis=1)
    clock = reserve_base_clock(candidates, joint_quality, ranges, five_minute_dates)
    assert_clock_contract(clock)
    deterministic_csv_gz(clock, Path(clock_path))
    reread = pd.read_csv(clock_path)
    assert_clock_contract(reread)
    support = support_statistics(reread)
    seed_record = source["historical_training_seed"]
    if sha256_file(Path(seed_record["path"])) != seed_record["sha256"]:
        raise RuntimeError("CRES historical seed changed before support freeze")
    manifest: dict[str, Any] = {
        "protocol_version": "cres_v1_2026_outcome_blind_support_2026-07-17",
        "preregistration_protocol_hash": EXPECTED_PROTOCOL_HASH,
        "source_manifest_path": source_manifest,
        "source_manifest_hash": EXPECTED_SOURCE_MANIFEST_HASH,
        "source_manifest_file_sha256": sha256_file(Path(source_manifest)),
        "historical_seed_path": seed_record["path"],
        "historical_seed_sha256": seed_record["sha256"],
        "clock_path": clock_path,
        "clock_sha256": sha256_file(Path(clock_path)),
        "post_entry_2026_strategy_returns_calculated": False,
        "outcome_like_columns_present": False,
        "support": support,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    Path(docs_path).parent.mkdir(parents=True, exist_ok=True)
    Path(docs_path).write_text(_markdown(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--source-manifest", default=SOURCE_MANIFEST)
    parser.add_argument("--clock", default=DEFAULT_CLOCK)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--docs", default=DEFAULT_DOCS)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.source_dir, args.source_manifest, args.clock, args.manifest, args.docs),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
