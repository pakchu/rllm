"""Build the outcome-blind calendar-2025 LORC event clock."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.build_leave_one_out_residual_exhaustion_support import (
    build_feature_panels,
    candidate_frame,
)
from training.export_leave_one_out_residual_continuation_2025_sources import (
    DEFAULT_OUTPUT_DIR as DEFAULT_SOURCE_DIR,
    END,
    EXPECTED_PROTOCOL_HASH,
    HOLDOUT_START,
    START,
    SYMBOLS,
)
from training.export_leave_one_out_residual_exhaustion_sources import deterministic_csv_gz, sha256_file
from training.preregister_leave_one_out_residual_continuation import canonical_hash, protocol


SOURCE_MANIFEST = "results/leave_one_out_residual_continuation_v1_source_manifest_2026-07-17.json"
EXPECTED_SOURCE_MANIFEST_HASH = "3ef36c5b77c6c2c48e77ab17af3b285152216b92ff031d6e496dc5255cd34a13"
DEFAULT_CLOCKS = "data/leave_one_out_residual_continuation_v1_support_clock_2025.csv.gz"
DEFAULT_MANIFEST = "results/leave_one_out_residual_continuation_v1_support_manifest_2026-07-17.json"
DEFAULT_DOCS = "docs/leave-one-out-residual-continuation-v1-support-2026-07-17.md"
POLICY_ID = "LORC01"
HORIZON_HOURS = 12
HOLD_HOURS = 12
FORBIDDEN_OUTPUT_TOKENS = {
    "future", "return", "pnl", "profit", "funding", "high", "low", "open", "close",
    "cagr", "mdd", "price", "label", "target",
}
CLOCK_COLUMNS = (
    "policy_id", "signal_time", "feature_available_time", "entry_time", "exit_time",
    "residual_horizon_hours", "hold_hours", "long_symbol", "short_symbol",
    "long_weight", "short_weight_abs", "long_beta", "short_beta",
    "loser_residual_z", "winner_residual_z", "loser_flow_z", "winner_flow_z",
    "continuation_score",
)


def _read_source_manifest(path: str = SOURCE_MANIFEST) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("manifest_hash") != EXPECTED_SOURCE_MANIFEST_HASH:
        raise RuntimeError("LORC source manifest hash changed")
    body = {k: v for k, v in payload.items() if k not in {"manifest_hash", "created_at"}}
    if canonical_hash(body) != EXPECTED_SOURCE_MANIFEST_HASH:
        raise RuntimeError("LORC source manifest body mismatch")
    if payload.get("future_2026_plus_rows_written") != 0:
        raise RuntimeError("LORC source manifest contains future rows")
    return payload


def load_hourly_panel(
    source_dir: str = DEFAULT_SOURCE_DIR,
    source_manifest: str = SOURCE_MANIFEST,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    manifest = _read_source_manifest(source_manifest)
    records = {str(row["symbol"]): row for row in manifest["records"]}
    if set(records) != set(SYMBOLS):
        raise RuntimeError("LORC source symbol set changed")
    panels: dict[str, pd.DataFrame] = {}
    quality: dict[str, pd.Series] = {}
    for symbol in sorted(SYMBOLS):
        path = Path(source_dir) / f"{symbol}_5m_2024_2025.csv.gz"
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
            raise RuntimeError("LORC hourly grids differ")
    expected = pd.date_range(START + pd.Timedelta(hours=1), END, freq="1h")
    if not common.equals(expected):
        raise RuntimeError("LORC hourly source grid escaped physical prefix")
    return panels, pd.DataFrame(quality, index=common).astype(bool)


def reserve_continuation_clock(candidates: pd.DataFrame, joint_quality: pd.Series) -> pd.DataFrame:
    use = candidates.copy()
    # The 12-hour factor/flow window plus the residual's t-12 close must all be
    # sourced from clean completed hours. No repair or fill is allowed.
    clean_13 = joint_quality.rolling(HORIZON_HOURS + 1, min_periods=HORIZON_HOURS + 1).sum().eq(HORIZON_HOURS + 1)
    use["eligible"] &= use["signal_time"].map(clean_13).fillna(False).to_numpy(dtype=bool)
    selected: list[dict[str, Any]] = []
    next_signal_allowed = HOLDOUT_START
    for row in use.loc[use["eligible"]].itertuples(index=False):
        signal = pd.Timestamp(row.signal_time)
        if not HOLDOUT_START <= signal < END:
            continue
        entry = signal + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=HOLD_HOURS)
        if signal < next_signal_allowed or exit_time >= END:
            continue
        # candidate_frame names the mean-reversion legs. LORC freezes their
        # exact direction flip, including weights and betas.
        selected.append({
            "policy_id": POLICY_ID,
            "signal_time": signal,
            "feature_available_time": signal,
            "entry_time": entry,
            "exit_time": exit_time,
            "residual_horizon_hours": HORIZON_HOURS,
            "hold_hours": HOLD_HOURS,
            "long_symbol": row.short_symbol,
            "short_symbol": row.long_symbol,
            "long_weight": float(row.short_weight_abs),
            "short_weight_abs": float(row.long_weight),
            "long_beta": float(row.short_beta),
            "short_beta": float(row.long_beta),
            "loser_residual_z": float(row.loser_residual_z),
            "winner_residual_z": float(row.winner_residual_z),
            "loser_flow_z": float(row.loser_flow_z),
            "winner_flow_z": float(row.winner_flow_z),
            "continuation_score": float(row.exhaustion_score),
        })
        next_signal_allowed = exit_time
    return pd.DataFrame(selected, columns=CLOCK_COLUMNS)


def assert_clock_contract(clock: pd.DataFrame) -> None:
    if clock.empty:
        raise RuntimeError("empty LORC 2025 support clock")
    for col in clock.columns:
        if set(col.lower().split("_")) & FORBIDDEN_OUTPUT_TOKENS:
            raise RuntimeError(f"outcome-like clock column forbidden: {col}")
    for col in ("signal_time", "feature_available_time", "entry_time", "exit_time"):
        clock[col] = pd.to_datetime(clock[col], errors="raise")
    if not clock["policy_id"].eq(POLICY_ID).all():
        raise RuntimeError("LORC policy identity drift")
    if not (clock["feature_available_time"] == clock["signal_time"]).all():
        raise RuntimeError("LORC feature availability drift")
    if not (clock["entry_time"] == clock["signal_time"] + pd.Timedelta(minutes=5)).all():
        raise RuntimeError("LORC entry delay drift")
    if not (clock["exit_time"] == clock["entry_time"] + pd.Timedelta(hours=HOLD_HOURS)).all():
        raise RuntimeError("LORC exit timing drift")
    if not ((clock["signal_time"] >= HOLDOUT_START) & (clock["exit_time"] < END)).all():
        raise RuntimeError("LORC clock escaped calendar 2025")
    if not np.allclose(clock["long_weight"] + clock["short_weight_abs"], 1.0, atol=1e-12):
        raise RuntimeError("LORC gross weights drifted")
    exposure = clock["long_weight"] * clock["long_beta"] - clock["short_weight_abs"] * clock["short_beta"]
    if not np.allclose(exposure, 0.0, atol=1e-12):
        raise RuntimeError("LORC factor beta is not neutral")
    ordered = clock.sort_values("signal_time")
    if (ordered["signal_time"] < ordered["exit_time"].shift(1)).fillna(False).any():
        raise RuntimeError("LORC clock overlaps reserved position")


def support_stats(clock: pd.DataFrame, max_monthly_quarantine: float) -> dict[str, Any]:
    pairs = clock["long_symbol"] + ">" + clock["short_symbol"]
    counts = Counter(pairs)
    n = len(clock)
    stats: dict[str, Any] = {
        "events": n,
        "h1_events": int((clock["signal_time"] < pd.Timestamp("2025-07-01")).sum()),
        "h2_events": int((clock["signal_time"] >= pd.Timestamp("2025-07-01")).sum()),
        "unique_ordered_pairs": len(counts),
        "maximum_ordered_pair_share": max(counts.values()) / n,
        "ordered_pair_counts": dict(sorted(counts.items())),
        "long_symbols": sorted(clock["long_symbol"].unique()),
        "short_symbols": sorted(clock["short_symbol"].unique()),
        "maximum_monthly_source_quarantine": max_monthly_quarantine,
    }
    gates = {
        "events_at_least_60": n >= 60,
        "h1_events_at_least_25": stats["h1_events"] >= 25,
        "h2_events_at_least_25": stats["h2_events"] >= 25,
        "unique_ordered_pairs_at_least_10": len(counts) >= 10,
        "maximum_ordered_pair_share_at_most_0_15": stats["maximum_ordered_pair_share"] <= 0.15,
        "symbols_seen_long_at_least_5": len(stats["long_symbols"]) >= 5,
        "symbols_seen_short_at_least_5": len(stats["short_symbols"]) >= 5,
        "monthly_source_quarantine_at_most_0_01": max_monthly_quarantine <= 0.01,
    }
    stats["gates"] = gates
    stats["passes_support"] = all(gates.values())
    return stats


def _markdown(result: dict[str, Any]) -> str:
    s = result["support"]
    rows = "\n".join(f"| {pair} | {count} |" for pair, count in s["ordered_pair_counts"].items())
    return f"""# LORC v1 calendar-2025 support freeze — 2026-07-17

> Outcome-blind event support only. No post-entry 2025 return, PnL, CAGR, or MDD was calculated.

## Support

- events: `{s['events']}` (H1 `{s['h1_events']}`, H2 `{s['h2_events']}`)
- unique ordered pairs: `{s['unique_ordered_pairs']}`
- maximum pair share: `{s['maximum_ordered_pair_share']:.4%}`
- maximum monthly source quarantine: `{s['maximum_monthly_source_quarantine']:.4%}`
- support decision: **{'PASS' if s['passes_support'] else 'REJECT'}**
- clock SHA-256: `{result['clock_sha256']}`

| Long > short | Events |
|---|---:|
{rows}

Every clock row uses completed data through `signal_time`, enters at +5m, exits
after exactly 12h, is factor-beta neutral, and does not overlap another LORC
position. A 13-completed-hour source-integrity gate covers the 12h factor/flow
window and the residual's `t-12` close.
"""


def run(
    source_dir: str = DEFAULT_SOURCE_DIR,
    source_manifest: str = SOURCE_MANIFEST,
    clocks_path: str = DEFAULT_CLOCKS,
    manifest_path: str = DEFAULT_MANIFEST,
    docs_path: str = DEFAULT_DOCS,
) -> dict[str, Any]:
    if canonical_hash(protocol()) != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("LORC preregistration drifted")
    source = _read_source_manifest(source_manifest)
    panels, quality = load_hourly_panel(source_dir, source_manifest)
    features = build_feature_panels(panels, quality, HORIZON_HOURS)
    candidates = candidate_frame(features, HORIZON_HOURS)
    clock = reserve_continuation_clock(candidates, quality.all(axis=1))
    assert_clock_contract(clock)
    holdout_quality = quality.all(axis=1).loc[
        (quality.index >= HOLDOUT_START + pd.Timedelta(hours=1)) & (quality.index < END)
    ]
    monthly_quarantine = (~holdout_quality).groupby(holdout_quality.index.to_period("M")).mean()
    max_monthly_quarantine = float(monthly_quarantine.max())
    stats = support_stats(clock, max_monthly_quarantine)
    if not stats["passes_support"]:
        raise RuntimeError(f"LORC 2025 support failed: {stats['gates']}")
    deterministic_csv_gz(clock, Path(clocks_path))
    reread = pd.read_csv(clocks_path)
    assert_clock_contract(reread)
    result: dict[str, Any] = {
        "protocol_version": "lorc_v1_2025_support_2026-07-17",
        "preregistration_protocol_hash": EXPECTED_PROTOCOL_HASH,
        "source_manifest_hash": EXPECTED_SOURCE_MANIFEST_HASH,
        "post_entry_returns_calculated": False,
        "holdout_2025_opened": False,
        "final_2026_opened": False,
        "source_hour_rows": int(len(quality)),
        "holdout_source_hours": int(len(holdout_quality)),
        "holdout_quarantined_hours": int((~holdout_quality).sum()),
        "clock_path": clocks_path,
        "clock_sha256": sha256_file(Path(clocks_path)),
        "support": stats,
        "source_records": [
            {
                "symbol": row["symbol"],
                "market_sha256": row["output_market_sha256"],
                "funding_sha256": row["output_funding_sha256"],
            }
            for row in source["records"]
        ],
    }
    result["manifest_hash"] = canonical_hash(result)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    Path(docs_path).parent.mkdir(parents=True, exist_ok=True)
    Path(docs_path).write_text(_markdown(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--source-manifest", default=SOURCE_MANIFEST)
    parser.add_argument("--clocks", default=DEFAULT_CLOCKS)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--docs", default=DEFAULT_DOCS)
    args = parser.parse_args()
    print(json.dumps(run(args.source_dir, args.source_manifest, args.clocks, args.manifest, args.docs), indent=2))


if __name__ == "__main__":
    main()
