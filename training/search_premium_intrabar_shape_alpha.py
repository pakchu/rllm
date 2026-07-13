"""Search causal intrabar premium-index shape alphas.

Five completed 1-minute Binance premium-index bars are summarized into a 5m
high/low/close candle.  Extreme-range onsets with wick and close-location
shapes are selected before 2024, frozen, and replayed on 2024-2026.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training.search_causal_online_expert_alpha import (
    ALPHAS,
    _build_expert_events,
    _global_nonoverlap,
    _load_bundle as _load_expert_bundle,
    _metric,
)
from training.search_funding_premium_external_state_gate_alpha import (
    _file_sha256,
    _frame_hash,
    _manifest_core_hash,
    _validate_manifest,
)
from training.search_positioning_hgb_path_alpha import _feature_hash, _read_before
from training.search_spot_perp_absorption_alpha import (
    SELECTION_END,
    SpotPerpConfig,
    _expert_config,
    _jaccard,
    _make_event,
    _merge_with_priority,
    _prior_z,
)


WINDOWS = {
    "fit": ("2020-06-01", "2023-01-01"),
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023_h1": ("2023-01-01", "2023-07-01"),
    "select_2023_h2": ("2023-07-01", "2024-01-01"),
    "test_2024": ("2024-01-01", "2025-01-01"),
    "eval_2025": ("2025-01-01", "2026-01-01"),
    "holdout_2026": ("2026-01-01", "2026-06-02"),
    "oos_2024_2026": ("2024-01-01", "2026-06-02"),
}
QUARTER_WINDOWS = {
    "2024Q1": ("2024-01-01", "2024-04-01"),
    "2024Q2": ("2024-04-01", "2024-07-01"),
    "2024Q3": ("2024-07-01", "2024-10-01"),
    "2024Q4": ("2024-10-01", "2025-01-01"),
    "2025Q1": ("2025-01-01", "2025-04-01"),
    "2025Q2": ("2025-04-01", "2025-07-01"),
    "2025Q3": ("2025-07-01", "2025-10-01"),
    "2025Q4": ("2025-10-01", "2026-01-01"),
    "2026Q1": ("2026-01-01", "2026-04-01"),
    "2026Q2_to_Jun02": ("2026-04-01", "2026-06-02"),
}
ROLLING_WINDOWS = (2016, 8640)
RANGE_Z = (2.0, 3.0)
SHAPE_THRESHOLDS = (0.5, 0.75)
MODES = ("wick", "close_location", "agreement", "disagreement")
DIRECTIONS = ("follow", "fade")
HOLDS = (24, 48, 96)


@dataclass(frozen=True)
class PremiumShapeConfig:
    input_csv: str
    spot_csv: str
    funding_csv: str
    premium_csv: str
    output: str
    manifest_output: str
    docs_output: str
    exclude_from: str = "2026-06-02"
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    stress_fee_rate: float = 0.0009
    top_n: int = 10
    top_per_mode: int = 3
    refresh_manifest: bool = False


def _window_mask(dates: pd.Series, name: str) -> np.ndarray:
    start, end = WINDOWS[name]
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)


def _source_hashes(cfg: PremiumShapeConfig) -> dict[str, str]:
    return {
        str(Path(path)): _file_sha256(path)
        for path in (cfg.input_csv, cfg.spot_csv, cfg.funding_csv, cfg.premium_csv)
    }


def _load_bundle(cfg: PremiumShapeConfig, *, cutoff: str) -> tuple[pd.DataFrame, dict[str, str]]:
    market_raw = _read_before(cfg.input_csv, "date", cutoff)
    spot_raw = _read_before(cfg.spot_csv, "date", cutoff)
    prefix_hashes = {"market": _frame_hash(market_raw), "spot_premium_1m": _frame_hash(spot_raw)}
    market, spot = market_raw.copy(), spot_raw.copy()
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    spot["date"] = pd.to_datetime(spot["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last")
    spot = spot.sort_values("date").drop_duplicates("date", keep="last")
    columns = [
        "date",
        "premium_index_1m_close",
        "premium_index_1m_low",
        "premium_index_1m_high",
        "premium_rows",
    ]
    market = market.merge(spot[columns], on="date", how="left", validate="one_to_one").reset_index(drop=True)
    market["premium_shape_available"] = (
        pd.to_numeric(market["premium_rows"], errors="coerce").eq(5)
        & pd.to_numeric(market["premium_index_1m_close"], errors="coerce").notna()
        & pd.to_numeric(market["premium_index_1m_low"], errors="coerce").notna()
        & pd.to_numeric(market["premium_index_1m_high"], errors="coerce").notna()
    )
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(cutoff):
        raise RuntimeError("premium shape bundle was not physically truncated")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise ValueError("premium shape search requires a complete futures grid")
    return market, prefix_hashes


def build_features(market: pd.DataFrame) -> pd.DataFrame:
    available = market["premium_shape_available"].to_numpy(bool)
    close = pd.to_numeric(market["premium_index_1m_close"], errors="coerce")
    low = pd.to_numeric(market["premium_index_1m_low"], errors="coerce")
    high = pd.to_numeric(market["premium_index_1m_high"], errors="coerce")
    open_proxy = close.shift(1)
    candle_range = (high - low).where(available)
    safe_range = candle_range.replace(0.0, np.nan)
    upper_wick = (high - pd.concat([open_proxy, close], axis=1).max(axis=1)).clip(lower=0.0)
    lower_wick = (pd.concat([open_proxy, close], axis=1).min(axis=1) - low).clip(lower=0.0)
    wick_imbalance = ((lower_wick - upper_wick) / safe_range).clip(-1.0, 1.0)
    close_location = (2.0 * (close - low) / safe_range - 1.0).clip(-1.0, 1.0)
    body_fraction = ((close - open_proxy) / safe_range).clip(-2.0, 2.0)
    out: dict[str, pd.Series] = {
        "premium_shape_available": pd.Series(available, index=market.index, dtype=float),
        "psi_premium_close": close.where(available),
        "psi_range": candle_range,
        "psi_wick_imbalance": wick_imbalance.where(available),
        "psi_close_location": close_location.where(available),
        "psi_body_fraction": body_fraction.where(available),
    }
    for window in ROLLING_WINDOWS:
        out[f"psi_range_z_{window}"] = _prior_z(candle_range, window)
        out[f"psi_body_z_{window}"] = _prior_z(close - open_proxy, window)
    frame = pd.DataFrame(out, index=market.index).replace([np.inf, -np.inf], np.nan)
    frame.loc[~available, :] = np.nan
    frame.loc[:, "premium_shape_available"] = available.astype(float)
    return frame.astype(np.float32)


def _policy_specs() -> list[dict[str, Any]]:
    return [
        {"window": window, "range_z": z, "shape_threshold": threshold, "mode": mode, "direction": direction, "hold": hold}
        for window in ROLLING_WINDOWS
        for z in RANGE_Z
        for threshold in SHAPE_THRESHOLDS
        for mode in MODES
        for direction in DIRECTIONS
        for hold in HOLDS
    ]


def _signals(features: pd.DataFrame, spec: dict[str, Any], *, flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    window = int(spec["window"])
    range_z = pd.to_numeric(features[f"psi_range_z_{window}"], errors="coerce").to_numpy(float)
    wick = pd.to_numeric(features["psi_wick_imbalance"], errors="coerce").to_numpy(float)
    close_location = pd.to_numeric(features["psi_close_location"], errors="coerce").to_numpy(float)
    previous_range_z = np.r_[np.nan, range_z[:-1]]
    active = (
        np.isfinite(range_z)
        & np.isfinite(previous_range_z)
        & (previous_range_z < float(spec["range_z"]))
        & (range_z >= float(spec["range_z"]))
    )
    threshold = float(spec["shape_threshold"])
    mode = str(spec["mode"])
    if mode == "wick":
        active &= np.isfinite(wick) & (np.abs(wick) >= threshold)
        shape = np.sign(wick)
    elif mode == "close_location":
        active &= np.isfinite(close_location) & (np.abs(close_location) >= threshold)
        shape = np.sign(close_location)
    elif mode == "agreement":
        active &= (
            np.isfinite(wick)
            & np.isfinite(close_location)
            & (np.abs(wick) >= threshold)
            & (np.abs(close_location) >= threshold)
            & (np.sign(wick) == np.sign(close_location))
        )
        shape = np.sign(wick)
    elif mode == "disagreement":
        active &= (
            np.isfinite(wick)
            & np.isfinite(close_location)
            & (np.abs(wick) >= threshold)
            & (np.abs(close_location) >= threshold)
            & (np.sign(wick) == -np.sign(close_location))
        )
        shape = np.sign(wick)
    else:
        raise KeyError(mode)
    # Shape values are allowed to be missing off-signal.  Replace them before
    # the integer cast; ``active`` remains the sole signal mask below.
    side = np.nan_to_num(shape, nan=0.0).astype(np.int8)
    if str(spec["direction"]) == "fade":
        side = -side
    if flip:
        side = -side
    side[~active] = 0
    return active, side


def _build_events(
    market: pd.DataFrame,
    features: pd.DataFrame,
    spec: dict[str, Any],
    cfg: PremiumShapeConfig,
    *,
    cost_rate: float,
    flip: bool = False,
) -> list[dict[str, Any]]:
    active, sides = _signals(features, spec, flip=flip)
    dummy_z = np.zeros(len(market), dtype=float)
    events: list[dict[str, Any]] = []
    next_allowed = 0
    for pos in np.flatnonzero(active):
        if int(pos) < next_allowed:
            continue
        event = _make_event(
            market,
            dummy_z,
            int(pos),
            int(sides[pos]),
            0,
            max_hold=int(spec["hold"]),
            dynamic_exit=False,
            exit_abs_z=0.0,
            cost_rate=cost_rate,
            leverage=float(cfg.leverage),
            name="premium_intrabar_shape_flip" if flip else "premium_intrabar_shape",
        )
        if event is None:
            continue
        events.append(event)
        next_allowed = int(event["exit_pos"])
    return events


def _stats(events: list[dict[str, Any]], dates: pd.Series, names: Iterable[str]) -> dict[str, Any]:
    return {name: _metric(events, dates, *WINDOWS[name]) for name in names}


def _path_hash(events: list[dict[str, Any]], dates: pd.Series, name: str) -> str:
    mask = _window_mask(dates, name)
    positions = np.flatnonzero(mask)
    first, last = int(positions[0]), int(positions[-1]) + 1
    rows = [(event["side"], int(event["signal_pos"]), int(event["exit_pos"])) for event in events if first <= int(event["signal_pos"]) and int(event["exit_pos"]) < last]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _selection_score(stats: dict[str, Any]) -> float:
    fit, full = stats["fit"], stats["select_2023"]
    h1, h2 = stats["select_2023_h1"], stats["select_2023_h2"]
    if (
        fit["return_pct"] <= 0.0 or fit["trades"] < 50
        or full["return_pct"] <= 0.0 or full["ratio"] < 1.0 or full["trades"] < 20
        or min(h1["return_pct"], h2["return_pct"]) <= 0.0
        or min(h1["trades"], h2["trades"]) < 8
    ):
        return -1e12
    return float(min(full["ratio"], h1["ratio"], h2["ratio"]) + 0.1 * fit["ratio"] + 0.01 * min(full["trades"], 100))


def _select_top(rows: list[dict[str, Any]], cfg: PremiumShapeConfig) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: (-row["selection_score"], json.dumps(row["spec"], sort_keys=True)))
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in rows:
        mode = str(row["spec"]["mode"])
        if counts.get(mode, 0) >= int(cfg.top_per_mode):
            continue
        selected.append(row)
        counts[mode] = counts.get(mode, 0) + 1
        if len(selected) >= int(cfg.top_n):
            break
    return selected


def _select_manifest(cfg: PremiumShapeConfig) -> dict[str, Any]:
    market, prefix_hashes = _load_bundle(cfg, cutoff=SELECTION_END)
    dates = pd.to_datetime(market["date"])
    features = build_features(market)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in _policy_specs():
        events = _build_events(market, features, spec, cfg, cost_rate=float(cfg.fee_rate + cfg.slippage_rate))
        stats = _stats(events, dates, ("fit", "select_2023", "select_2023_h1", "select_2023_h2"))
        score = _selection_score(stats)
        if score <= -1e11:
            continue
        path_hash = _path_hash(events, dates, "select_2023")
        if path_hash in seen:
            continue
        seen.add(path_hash)
        rows.append({"spec": spec, "selection_score": score, "selection_stats": stats, "selection_path_hash": path_hash})
    selected = _select_top(rows, cfg)
    core = {
        "protocol": {
            "hypothesis": "extreme completed 5m premium-index intrabar range shape contains short-horizon BTC price-discovery information",
            "feature": "five completed 1m premium-index bars; previous completed close is candle-open proxy",
            "normalization": "range z-score uses rolling statistics ending at t-1",
            "selection": {name: WINDOWS[name] for name in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")},
            "all_market_and_premium_rows_physically_excluded_before_manifest": True,
            "later_metrics_included": False,
            "search_cap": f"{len(_policy_specs())} fixed onset/shape policies",
            "entry_exit": "next 5m open; fixed hold; global non-overlap",
            "cost": "6bp/side base, 10bp/side stress, 0.5x",
            "mdd": "strict entry cost plus intrabar adverse OHLC and realized high-water",
            "marginal_rule": "must improve deterministic six-sleeve union on combined return and CAGR/MDD",
            "status_ceiling": "shadow research",
        },
        "source_prefix_hashes": prefix_hashes,
        "feature_hash": _feature_hash(features, dates),
        "search_space": {"raw_specs": len(_policy_specs()), "eligible_unique_paths": len(rows), "top_n": int(cfg.top_n)},
        "selected": selected,
    }
    manifest = {"as_of": datetime.now(timezone.utc).isoformat(), "sha256": _manifest_core_hash(core), **core}
    path = Path(cfg.manifest_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    return manifest


def _spot_config(cfg: PremiumShapeConfig) -> SpotPerpConfig:
    return SpotPerpConfig(
        input_csv=cfg.input_csv,
        spot_csv=cfg.spot_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        output=cfg.output,
        manifest_output=cfg.manifest_output,
        docs_output=cfg.docs_output,
        exclude_from=cfg.exclude_from,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        stress_fee_rate=cfg.stress_fee_rate,
    )


def _replay(cfg: PremiumShapeConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    _validate_manifest(manifest)
    prefix_market, prefix_hashes = _load_bundle(cfg, cutoff=SELECTION_END)
    prefix_dates = pd.to_datetime(prefix_market["date"])
    prefix_features = build_features(prefix_market)
    if prefix_hashes != manifest["source_prefix_hashes"] or _feature_hash(prefix_features, prefix_dates) != manifest["feature_hash"]:
        raise RuntimeError("pre-2024 premium shape reconstruction drift")
    market, _ = _load_bundle(cfg, cutoff=cfg.exclude_from)
    dates = pd.to_datetime(market["date"])
    features = build_features(market)
    prefix = dates < pd.Timestamp(SELECTION_END)
    if _feature_hash(features.loc[prefix].reset_index(drop=True), dates.loc[prefix].reset_index(drop=True)) != manifest["feature_hash"]:
        raise RuntimeError("full replay premium shape prefix drift")

    expert_cfg = _expert_config(_spot_config(cfg))
    expert_market, expert_features, _ = _load_expert_bundle(expert_cfg, cutoff=cfg.exclude_from)
    if not pd.to_datetime(expert_market["date"]).equals(dates):
        raise RuntimeError("premium shape and expert baseline grids differ")
    base_events = _build_expert_events(expert_market, expert_features, expert_cfg, cost_rate=float(cfg.fee_rate + cfg.slippage_rate))
    base_by_expert = {name: [event for event in base_events if event["expert"] == name] for name in ALPHAS}
    base_union = _global_nonoverlap(base_events)
    eval_windows = ("test_2024", "eval_2025", "holdout_2026", "oos_2024_2026")
    baseline_stats = _stats(base_union, dates, eval_windows)
    stress_cfg = replace(cfg, fee_rate=cfg.stress_fee_rate, slippage_rate=0.0001)
    stress_expert_cfg = _expert_config(_spot_config(stress_cfg))
    stress_base_events = _build_expert_events(expert_market, expert_features, stress_expert_cfg, cost_rate=float(stress_cfg.fee_rate + stress_cfg.slippage_rate))
    stress_base_union = _global_nonoverlap(stress_base_events)

    rows = []
    for rank, frozen in enumerate(manifest["selected"], start=1):
        events = _build_events(market, features, frozen["spec"], cfg, cost_rate=float(cfg.fee_rate + cfg.slippage_rate))
        selection_stats = _stats(events, dates, ("fit", "select_2023", "select_2023_h1", "select_2023_h2"))
        if selection_stats != frozen["selection_stats"] or _path_hash(events, dates, "select_2023") != frozen["selection_path_hash"]:
            raise RuntimeError(f"pre-2024 policy replay drift at rank {rank}")
        stats = _stats(events, dates, WINDOWS)
        flipped = _build_events(market, features, frozen["spec"], cfg, cost_rate=float(cfg.fee_rate + cfg.slippage_rate), flip=True)
        stress_events = _build_events(market, features, frozen["spec"], stress_cfg, cost_rate=float(stress_cfg.fee_rate + stress_cfg.slippage_rate))
        combined = _merge_with_priority(base_union, events)
        stress_combined = _merge_with_priority(stress_base_union, stress_events)
        combined_stats = _stats(combined, dates, eval_windows)
        stress_stats = _stats(stress_events, dates, eval_windows)
        quarterly = {name: _metric(events, dates, start, end) for name, (start, end) in QUARTER_WINDOWS.items()}
        quarter_summary = {
            "positive_return_quarters": sum(row["return_pct"] > 0.0 for row in quarterly.values()),
            "negative_return_quarters": sum(row["return_pct"] < 0.0 for row in quarterly.values()),
            "flat_quarters": sum(row["trades"] == 0 for row in quarterly.values()),
            "total_quarters": len(quarterly),
        }
        jaccards = {name: _jaccard(events, source, dates) for name, source in base_by_expert.items()}
        test, evaluation = stats["test_2024"], stats["eval_2025"]
        holdout, all_oos = stats["holdout_2026"], stats["oos_2024_2026"]
        standalone = (
            test["return_pct"] > 0.0 and test["ratio"] >= 3.0 and test["trades"] >= 20
            and evaluation["return_pct"] > 0.0 and evaluation["ratio"] >= 3.0 and evaluation["trades"] >= 20
            and holdout["return_pct"] > 0.0 and holdout["trades"] >= 12
            and all_oos["return_pct"] > 0.0
        )
        base_all, merged_all = baseline_stats["oos_2024_2026"], combined_stats["oos_2024_2026"]
        marginal = merged_all["return_pct"] > base_all["return_pct"] and merged_all["ratio"] > base_all["ratio"]
        stress_ok = min(stress_stats[name]["ratio"] for name in ("test_2024", "eval_2025", "holdout_2026")) >= 2.5
        bonferroni = min(1.0, all_oos["p_value_mean_return_approx"] * max(1, len(manifest["selected"])))
        qualifies = standalone and marginal and stress_ok and max(jaccards.values(), default=0.0) <= 0.25 and bonferroni <= 0.05
        rows.append(
            {
                "manifest_rank": rank,
                **frozen,
                "stats": stats,
                "direction_flipped": _stats(flipped, dates, eval_windows),
                "stress_10bp_each_side": stress_stats,
                "combined_with_six_sleeve_union": combined_stats,
                "stress_combined_with_six_sleeve_union": _stats(stress_combined, dates, eval_windows),
                "quarterly_stats": quarterly,
                "quarterly_summary": quarter_summary,
                "signal_jaccard_vs_fixed_experts": jaccards,
                "top_n_bonferroni_p_value": float(bonferroni),
                "passes_standalone_gate": bool(standalone),
                "adds_value_vs_six_sleeve_union": bool(marginal),
                "passes_cost_stress": bool(stress_ok),
                "passes_alpha_pool": bool(qualifies),
                "passes_live_grade": False,
            }
        )
    fit_mask = _window_mask(dates, "fit")
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "manifest": cfg.manifest_output,
        "manifest_sha256": manifest["sha256"],
        "protocol": manifest["protocol"],
        "source_file_hashes_after_manifest_freeze": _source_hashes(cfg),
        "feature_correlation_audit": {
            "wick_vs_premium_close_fit_spearman": float(features.loc[fit_mask, "psi_wick_imbalance"].corr(features.loc[fit_mask, "psi_premium_close"], method="spearman")),
            "close_location_vs_premium_close_fit_spearman": float(features.loc[fit_mask, "psi_close_location"].corr(features.loc[fit_mask, "psi_premium_close"], method="spearman")),
        },
        "six_sleeve_union_baseline": baseline_stats,
        "selected": rows,
        "alpha_pool_qualifiers": [row for row in rows if row["passes_alpha_pool"]],
        "live_grade": [],
    }


def _fmt(row: dict[str, Any]) -> str:
    return f"{row['return_pct']:.2f}/{row['cagr_pct']:.2f}/{row['strict_mdd_pct']:.2f}/{row['ratio']:.2f}/{row['trades']}"


def _write_doc(cfg: PremiumShapeConfig, report: dict[str, Any]) -> None:
    manifest = json.loads(Path(cfg.manifest_output).read_text())
    search_space = manifest.get("search_space", {})
    lines = [
        "# Premium-index intrabar shape alpha search (2026-07-13)",
        "",
        "Metric format: `absolute return / CAGR / strict MDD / CAGR-MDD / trades`.",
        "",
        "| rank | policy | 2024 | 2025 | 2026 | combined | +union combined | alpha |",
        "|---:|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["selected"]:
        stats = row["stats"]
        merged = row["combined_with_six_sleeve_union"]["oos_2024_2026"]
        lines.append(
            f"| {row['manifest_rank']} | `{row['spec']}` | {_fmt(stats['test_2024'])} | {_fmt(stats['eval_2025'])} | {_fmt(stats['holdout_2026'])} | {_fmt(stats['oos_2024_2026'])} | {_fmt(merged)} | {'yes' if row['passes_alpha_pool'] else 'no'} |"
        )
    baseline = report["six_sleeve_union_baseline"]["oos_2024_2026"]
    rank_one = report["selected"][0] if report["selected"] else None
    lines += [
        "",
        "## Interpretation",
        "",
        f"- Alpha-pool qualifiers: {len(report['alpha_pool_qualifiers'])}; baseline union `{_fmt(baseline)}`.",
        f"- Pre-2024 selection admitted {search_space.get('eligible_unique_paths', 0)} unique paths from {search_space.get('raw_specs', len(_policy_specs()))} fixed policies.",
        "- Only complete five-row premium-index intervals are used; range normalization ends at t-1 and execution begins at t+1 open.",
        "- Standalone strength is insufficient without positive marginal contribution to the fixed six-sleeve union.",
        "- 2024-2026 are replay evidence, not fresh future data for live promotion.",
    ]
    if rank_one is not None:
        rank_stats = rank_one["stats"]["oos_2024_2026"]
        flipped = rank_one["direction_flipped"]["oos_2024_2026"]
        stress = rank_one["stress_10bp_each_side"]["oos_2024_2026"]
        merged = rank_one["combined_with_six_sleeve_union"]["oos_2024_2026"]
        lines += [
            f"- Rank 1 failed with `{_fmt(rank_stats)}`; direction flip also failed at `{_fmt(flipped)}`, and 10bp/side stress fell to `{_fmt(stress)}`.",
            f"- Adding rank 1 reduced the union from `{_fmt(baseline)}` to `{_fmt(merged)}`. Signal disjointness therefore did not translate into marginal alpha.",
            "- Preserve continuous wick/close-location/range fields only as beta research context; reject this exact extreme-range-onset plus fixed-direction/fixed-hold mapping as gamma.",
        ]
    lines += [
        "",
        "## Reproduction",
        "",
        "```bash",
        f"python -m training.search_premium_intrabar_shape_alpha --input-csv {cfg.input_csv} --spot-csv {cfg.spot_csv} --funding-csv {cfg.funding_csv} --premium-csv {cfg.premium_csv} --manifest-output {cfg.manifest_output} --output {cfg.output} --docs-output {cfg.docs_output}",
        "```",
    ]
    path = Path(cfg.docs_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run(cfg: PremiumShapeConfig) -> dict[str, Any]:
    path = Path(cfg.manifest_output)
    if path.exists() and not cfg.refresh_manifest:
        manifest = json.loads(path.read_text())
        _validate_manifest(manifest)
    else:
        manifest = _select_manifest(cfg)
    report = _replay(cfg, manifest)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    _write_doc(cfg, report)
    return report


def parse_args() -> PremiumShapeConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--spot-csv", required=True)
    parser.add_argument("--funding-csv", required=True)
    parser.add_argument("--premium-csv", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--docs-output", required=True)
    parser.add_argument("--exclude-from", default=PremiumShapeConfig.exclude_from)
    parser.add_argument("--refresh-manifest", action="store_true")
    return PremiumShapeConfig(**vars(parser.parse_args()))


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"manifest": report["manifest"], "qualifiers": len(report["alpha_pool_qualifiers"]), "top": report["selected"][:3]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
