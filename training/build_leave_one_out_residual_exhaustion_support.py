"""Build outcome-blind LORE v1 features and immutable event clocks."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.export_leave_one_out_residual_exhaustion_sources import (
    END,
    EXPECTED_PROTOCOL_HASH,
    START,
    SYMBOLS,
    deterministic_csv_gz,
    sha256_file,
)
from training.preregister_leave_one_out_residual_exhaustion import canonical_hash, protocol


SOURCE_MANIFEST = "results/leave_one_out_residual_exhaustion_v1_source_manifest_2026-07-17.json"
EXPECTED_SOURCE_MANIFEST_HASH = "1c54fddc45fcc516d8ce42741904e018e8d00e646eff40be514273cf10eee7ed"
DEFAULT_SOURCE_DIR = "data/binance_um_lore_2023_2024"
DEFAULT_CLOCKS = "data/leave_one_out_residual_exhaustion_v1_support_clocks_2023_2024.csv.gz"
DEFAULT_FEATURES = "data/leave_one_out_residual_exhaustion_v1_support_features_2023_2024.csv.gz"
DEFAULT_MANIFEST = "results/leave_one_out_residual_exhaustion_v1_support_manifest_2026-07-17.json"
FORBIDDEN_OUTPUT_TOKENS = (
    "future", "return", "pnl", "profit", "funding", "high", "low", "open", "close",
    "cagr", "mdd", "price", "label", "target",
)
CLOCK_COLUMNS = (
    "policy_id", "signal_time", "feature_available_time", "entry_time", "exit_time",
    "residual_horizon_hours", "hold_hours", "long_symbol", "short_symbol",
    "long_weight", "short_weight_abs", "long_beta", "short_beta",
    "loser_residual_z", "winner_residual_z", "loser_flow_z", "winner_flow_z",
    "exhaustion_score",
)


def _read_source_manifest(path: str = SOURCE_MANIFEST) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("manifest_hash") != EXPECTED_SOURCE_MANIFEST_HASH:
        raise RuntimeError("LORE source manifest hash changed")
    body = {k: v for k, v in payload.items() if k not in {"manifest_hash", "created_at"}}
    if canonical_hash(body) != EXPECTED_SOURCE_MANIFEST_HASH:
        raise RuntimeError("LORE source manifest body mismatch")
    if payload.get("future_2025_plus_rows_written") != 0:
        raise RuntimeError("LORE source manifest contains future rows")
    return payload


def _manifest_record_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {str(r["symbol"]): r for r in manifest["records"]}
    if set(out) != set(SYMBOLS):
        raise RuntimeError("LORE source symbol set changed")
    return out


def load_hourly_panel(
    source_dir: str = DEFAULT_SOURCE_DIR,
    source_manifest: str = SOURCE_MANIFEST,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    manifest = _read_source_manifest(source_manifest)
    records = _manifest_record_map(manifest)
    panels: dict[str, pd.DataFrame] = {}
    quality: dict[str, pd.Series] = {}
    for symbol in sorted(SYMBOLS):
        path = Path(source_dir) / f"{symbol}_5m_2023_2024.csv.gz"
        if sha256_file(path) != records[symbol]["output_market_sha256"]:
            raise RuntimeError(f"{symbol} frozen market hash changed")
        raw = pd.read_csv(
            path,
            usecols=["date", "close", "quote_asset_volume", "taker_buy_quote", "tic"],
            parse_dates=["date"],
        )
        if not raw["tic"].astype(str).eq(symbol).all():
            raise RuntimeError(f"{symbol} source identity mismatch")
        raw = raw.sort_values("date").set_index("date")
        if raw.index.duplicated().any():
            raise RuntimeError(f"{symbol} duplicate source bar")
        close = pd.to_numeric(raw["close"], errors="raise")
        quote = pd.to_numeric(raw["quote_asset_volume"], errors="raise")
        buy = pd.to_numeric(raw["taker_buy_quote"], errors="raise")
        hourly = pd.DataFrame({
            "close": close.resample("1h", closed="left", label="right").last(),
            "quote": quote.resample("1h", closed="left", label="right").sum(),
            "buy": buy.resample("1h", closed="left", label="right").sum(),
            "bar_count": close.resample("1h", closed="left", label="right").count(),
            "positive_quote_bars": (quote > 0).resample("1h", closed="left", label="right").sum(),
        })
        clean = (
            hourly["bar_count"].eq(12)
            & hourly["positive_quote_bars"].eq(12)
            & hourly["close"].gt(0)
            & hourly["quote"].gt(0)
            & hourly["buy"].ge(0)
            & hourly["buy"].le(hourly["quote"] + np.maximum(1e-6, hourly["quote"] * 1e-10))
        )
        panels[symbol] = hourly[["close", "quote", "buy"]]
        quality[symbol] = clean.rename(symbol)
    common = panels[sorted(SYMBOLS)[0]].index
    for panel in panels.values():
        if not panel.index.equals(common):
            raise RuntimeError("LORE hourly grids differ")
    quality_frame = pd.DataFrame(quality, index=common).astype(bool)
    return panels, quality_frame


def _shifted_z(frame: pd.DataFrame, lookback: int = 2160, minimum: int = 1080) -> pd.DataFrame:
    history = frame.shift(1)
    mean = history.rolling(lookback, min_periods=minimum).mean()
    std = history.rolling(lookback, min_periods=minimum).std()
    return (frame - mean) / std.replace(0.0, np.nan)


def build_feature_panels(
    panels: dict[str, pd.DataFrame],
    quality: pd.DataFrame,
    horizon_hours: int,
) -> dict[str, Any]:
    symbols = list(sorted(SYMBOLS))
    index = quality.index
    close = pd.DataFrame({s: panels[s]["close"] for s in symbols}, index=index)
    quote = pd.DataFrame({s: panels[s]["quote"] for s in symbols}, index=index)
    buy = pd.DataFrame({s: panels[s]["buy"] for s in symbols}, index=index)
    ret1 = np.log(close).diff()
    factor = pd.DataFrame(index=index, columns=symbols, dtype=float)
    for symbol in symbols:
        factor[symbol] = ret1.drop(columns=symbol).median(axis=1)
    beta = pd.DataFrame(index=index, columns=symbols, dtype=float)
    for symbol in symbols:
        covariance = ret1[symbol].rolling(720, min_periods=336).cov(factor[symbol])
        variance = factor[symbol].rolling(720, min_periods=336).var()
        beta[symbol] = (covariance / variance.replace(0.0, np.nan)).shift(1).clip(0.25, 2.5)
    asset_move = np.log(close / close.shift(horizon_hours))
    factor_move = factor.rolling(horizon_hours, min_periods=horizon_hours).sum()
    residual = asset_move - beta * factor_move
    residual_z = _shifted_z(residual)
    quote_h = quote.rolling(horizon_hours, min_periods=horizon_hours).sum()
    buy_h = buy.rolling(horizon_hours, min_periods=horizon_hours).sum()
    flow = 2.0 * buy_h / quote_h.replace(0.0, np.nan) - 1.0
    flow_z = _shifted_z(flow)
    source_clean = quality.loc[:, symbols].all(axis=1)
    finite = (
        residual_z.notna().all(axis=1)
        & flow_z.notna().all(axis=1)
        & beta.notna().all(axis=1)
        & np.isfinite(residual_z.to_numpy()).all(axis=1)
        & np.isfinite(flow_z.to_numpy()).all(axis=1)
        & np.isfinite(beta.to_numpy()).all(axis=1)
    )
    return {
        "symbols": symbols,
        "close": close,
        "beta": beta,
        "residual_z": residual_z,
        "flow_z": flow_z,
        "source_clean": source_clean,
        "finite": pd.Series(finite, index=index),
    }


def candidate_frame(features: dict[str, Any], horizon_hours: int) -> pd.DataFrame:
    symbols: list[str] = features["symbols"]
    index: pd.DatetimeIndex = features["residual_z"].index
    rz = features["residual_z"].to_numpy(dtype=float)
    fz = features["flow_z"].to_numpy(dtype=float)
    beta = features["beta"].to_numpy(dtype=float)
    valid = features["source_clean"].to_numpy(dtype=bool) & features["finite"].to_numpy(dtype=bool)
    safe_rz = np.where(np.isfinite(rz), rz, 0.0)
    winner_idx = np.argmax(safe_rz, axis=1)
    loser_idx = np.argmin(safe_rz, axis=1)
    rows = np.arange(len(index))
    winner_rz = rz[rows, winner_idx]
    loser_rz = rz[rows, loser_idx]
    winner_fz = fz[rows, winner_idx]
    loser_fz = fz[rows, loser_idx]
    winner_beta = beta[rows, winner_idx]
    loser_beta = beta[rows, loser_idx]
    denominator = winner_beta + loser_beta
    long_weight = winner_beta / denominator
    short_weight = loser_beta / denominator
    score = np.minimum(winner_rz - winner_fz, loser_fz - loser_rz)
    valid &= (
        (winner_rz >= 1.5)
        & (loser_rz <= -1.5)
        & ((winner_rz - winner_fz) >= 1.0)
        & ((loser_fz - loser_rz) >= 1.0)
        & np.isfinite(long_weight)
        & np.isfinite(short_weight)
        & (np.minimum(long_weight, short_weight) >= 0.25)
    )
    out = pd.DataFrame({
        "signal_time": index,
        "residual_horizon_hours": int(horizon_hours),
        "long_symbol": np.asarray(symbols, dtype=object)[loser_idx],
        "short_symbol": np.asarray(symbols, dtype=object)[winner_idx],
        "long_weight": long_weight,
        "short_weight_abs": short_weight,
        "long_beta": loser_beta,
        "short_beta": winner_beta,
        "loser_residual_z": loser_rz,
        "winner_residual_z": winner_rz,
        "loser_flow_z": loser_fz,
        "winner_flow_z": winner_fz,
        "exhaustion_score": score,
        "eligible": valid,
    })
    return out


def reserve_clock(candidates: pd.DataFrame, policy_id: str, hold_hours: int) -> pd.DataFrame:
    selected: list[dict[str, Any]] = []
    next_signal_allowed = pd.Timestamp.min
    for row in candidates.loc[candidates["eligible"]].itertuples(index=False):
        signal = pd.Timestamp(row.signal_time)
        entry = signal + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=hold_hours)
        if signal < next_signal_allowed or exit_time >= END:
            continue
        selected.append({
            "policy_id": policy_id,
            "signal_time": signal,
            "feature_available_time": signal,
            "entry_time": entry,
            "exit_time": exit_time,
            "residual_horizon_hours": int(row.residual_horizon_hours),
            "hold_hours": int(hold_hours),
            "long_symbol": row.long_symbol,
            "short_symbol": row.short_symbol,
            "long_weight": float(row.long_weight),
            "short_weight_abs": float(row.short_weight_abs),
            "long_beta": float(row.long_beta),
            "short_beta": float(row.short_beta),
            "loser_residual_z": float(row.loser_residual_z),
            "winner_residual_z": float(row.winner_residual_z),
            "loser_flow_z": float(row.loser_flow_z),
            "winner_flow_z": float(row.winner_flow_z),
            "exhaustion_score": float(row.exhaustion_score),
        })
        next_signal_allowed = exit_time
    return pd.DataFrame(selected, columns=CLOCK_COLUMNS)


def assert_clock_contract(clock: pd.DataFrame) -> None:
    for col in clock.columns:
        parts = set(col.lower().split("_"))
        if any(token in parts for token in FORBIDDEN_OUTPUT_TOKENS):
            raise RuntimeError(f"outcome-like clock column forbidden: {col}")
    if clock.empty:
        raise RuntimeError("empty LORE support clock")
    for col in ["signal_time", "feature_available_time", "entry_time", "exit_time"]:
        clock[col] = pd.to_datetime(clock[col], errors="raise")
    if not (clock["feature_available_time"] == clock["signal_time"]).all():
        raise RuntimeError("feature availability drift")
    if not (clock["entry_time"] == clock["signal_time"] + pd.Timedelta(minutes=5)).all():
        raise RuntimeError("entry delay drift")
    expected_exit = clock["entry_time"] + pd.to_timedelta(clock["hold_hours"], unit="h")
    if not (clock["exit_time"] == expected_exit).all():
        raise RuntimeError("exit timing drift")
    if not (clock["exit_time"] < END).all():
        raise RuntimeError("LORE support clock escaped 2023-2024")
    if not np.allclose(clock["long_weight"] + clock["short_weight_abs"], 1.0, atol=1e-12):
        raise RuntimeError("LORE gross weights drifted")
    beta_exposure = clock["long_weight"] * clock["long_beta"] - clock["short_weight_abs"] * clock["short_beta"]
    if not np.allclose(beta_exposure, 0.0, atol=1e-12):
        raise RuntimeError("LORE factor beta is not neutral")
    for _, group in clock.groupby("policy_id"):
        ordered = group.sort_values("signal_time")
        previous_exit = ordered["exit_time"].shift(1)
        if (ordered["signal_time"] < previous_exit).fillna(False).any():
            raise RuntimeError("LORE support clock overlaps reserved position")


def support_stats(clock: pd.DataFrame, max_monthly_quarantine: float) -> dict[str, Any]:
    pair = clock["short_symbol"] + ">" + clock["long_symbol"]
    counts = Counter(pair)
    n = len(clock)
    year_counts = clock.groupby(clock["signal_time"].dt.year).size().to_dict()
    stats = {
        "events": n,
        "year_counts": {str(k): int(v) for k, v in year_counts.items()},
        "half_year_counts": {
            f"{year}H{half}": int(((clock["signal_time"].dt.year == year) & (((clock["signal_time"].dt.month - 1) // 6 + 1) == half)).sum())
            for year in (2023, 2024) for half in (1, 2)
        },
        "unique_ordered_pairs": len(counts),
        "maximum_ordered_pair_share": max(counts.values()) / n,
        "ordered_pair_counts": dict(sorted(counts.items())),
        "long_symbols": sorted(clock["long_symbol"].unique()),
        "short_symbols": sorted(clock["short_symbol"].unique()),
        "maximum_monthly_source_quarantine": max_monthly_quarantine,
    }
    gates = {
        "combined_events_at_least_150": n >= 150,
        "each_year_events_at_least_60": all(year_counts.get(y, 0) >= 60 for y in (2023, 2024)),
        "unique_ordered_pairs_at_least_10": len(counts) >= 10,
        "maximum_ordered_pair_share_at_most_0_15": stats["maximum_ordered_pair_share"] <= 0.15,
        "symbols_seen_as_long_at_least_5": len(stats["long_symbols"]) >= 5,
        "symbols_seen_as_short_at_least_5": len(stats["short_symbols"]) >= 5,
        "monthly_source_quarantine_at_most_0_01": max_monthly_quarantine <= 0.01,
    }
    stats["gates"] = gates
    stats["passes_support"] = all(gates.values())
    return stats


def _feature_export(candidates_by_horizon: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for horizon, candidates in sorted(candidates_by_horizon.items()):
        use = candidates.loc[candidates["eligible"]].copy()
        use.insert(0, "feature_key", f"h{horizon}")
        rows.append(use.drop(columns=["eligible"]))
    out = pd.concat(rows, ignore_index=True)
    for col in out.columns:
        parts = set(col.lower().split("_"))
        if any(token in parts for token in FORBIDDEN_OUTPUT_TOKENS):
            raise RuntimeError(f"outcome-like feature column forbidden: {col}")
    return out


def run(
    source_dir: str = DEFAULT_SOURCE_DIR,
    source_manifest: str = SOURCE_MANIFEST,
    clocks_path: str = DEFAULT_CLOCKS,
    features_path: str = DEFAULT_FEATURES,
    manifest_path: str = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    if canonical_hash(protocol()) != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("LORE preregistration drifted")
    source = _read_source_manifest(source_manifest)
    panels, quality = load_hourly_panel(source_dir, source_manifest)
    joint_clean = quality.all(axis=1)
    source_hours = joint_clean.loc[(joint_clean.index >= START + pd.Timedelta(hours=1)) & (joint_clean.index < END)]
    monthly_quarantine = (1.0 - source_hours.groupby(source_hours.index.to_period("M")).mean()).astype(float)
    max_monthly_quarantine = float(monthly_quarantine.max())
    candidates_by_horizon = {h: candidate_frame(build_feature_panels(panels, quality, h), h) for h in (6, 12)}
    policies = protocol()["policies"]
    clocks = []
    for policy in policies:
        clocks.append(reserve_clock(
            candidates_by_horizon[int(policy["residual_horizon_hours"])],
            str(policy["policy_id"]),
            int(policy["hold_hours"]),
        ))
    clock = pd.concat(clocks, ignore_index=True)
    assert_clock_contract(clock)
    features = _feature_export(candidates_by_horizon)
    deterministic_csv_gz(clock.loc[:, CLOCK_COLUMNS], Path(clocks_path))
    deterministic_csv_gz(features, Path(features_path))
    reread_clock = pd.read_csv(clocks_path)
    assert_clock_contract(reread_clock)
    stats = {
        policy["policy_id"]: support_stats(
            clock.loc[clock["policy_id"] == policy["policy_id"]].copy(),
            max_monthly_quarantine,
        )
        for policy in policies
    }
    manifest: dict[str, Any] = {
        "protocol_version": "lore_v1_support_freeze_2026-07-17",
        "preregistration_protocol_hash": EXPECTED_PROTOCOL_HASH,
        "source_manifest_hash": source["manifest_hash"],
        "outcomes_calculated": False,
        "future_2025_plus_opened": False,
        "loaded_source_columns": ["date", "close", "quote_asset_volume", "taker_buy_quote", "tic"],
        "source_quality": {
            "hourly_rows": int(len(source_hours)),
            "joint_clean_hours": int(source_hours.sum()),
            "joint_quarantined_hours": int((~source_hours).sum()),
            "global_quarantine_fraction": float((~source_hours).mean()),
            "maximum_monthly_quarantine_fraction": max_monthly_quarantine,
        },
        "clock_path": clocks_path,
        "clock_sha256": sha256_file(Path(clocks_path)),
        "clock_rows": len(clock),
        "feature_path": features_path,
        "feature_sha256": sha256_file(Path(features_path)),
        "feature_rows": len(features),
        "policies": stats,
        "all_policies_pass_support": all(s["passes_support"] for s in stats.values()),
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--source-manifest", default=SOURCE_MANIFEST)
    parser.add_argument("--clocks", default=DEFAULT_CLOCKS)
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(run(args.source_dir, args.source_manifest, args.clocks, args.features, args.manifest), indent=2))


if __name__ == "__main__":
    main()
