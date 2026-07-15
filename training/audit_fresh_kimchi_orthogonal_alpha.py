#!/usr/bin/env python3
"""Audit a freshness-safe Kimchi/FX sleeve against the frozen rank-7 alpha.

The candidate is not re-optimized here.  It pins the historical
``funding_relief_vs_fx_stress`` policy and its 2024-selected Kimchi/FX gate,
then repairs the live-parity defect that allowed forward-filled weekend FX and
Kimchi values to trigger entries.  The audit includes realized funding,
strict intratrade drawdown, and trade-level orthogonality diagnostics against
the frozen annual-refit rank-7 schedule.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.market_features import build_market_feature_frame
from training.evaluate_expanding_extratrees_rank7_refit_cadence_oos import (
    ALL_WINDOWS as RANK7_WINDOWS,
    assert_annual_reference,
    assert_frozen_prefix,
    schedule_stats as rank7_schedule_stats,
    validate_cadence_manifest,
)
from training.evaluate_expanding_extratrees_top10_oos import FULL_CUTOFF, build_replay_base
from training.evaluate_stable_ensemble_conditional_pullback_oos import Config as Rank7Config
from training.long_regime_interest_gate_validation import build_interest_features
from training.search_bidirectional_state_alpha import Config as LegacyConfig
from training.search_bidirectional_state_alpha import extra, mk
from training.search_inventory_purge_reclaim_alpha import (
    Config as ExecutionConfig,
    ExecutionEngine,
    Trade,
    _schedule_hash,
    equity_stats,
)
from training.search_kimchi_leadlag_bidirectional_alpha import features as kimchi_features
from training.search_stable_ensemble_conditional_pullback_alpha import routed_schedule
from training.compare_expanding_extratrees_rank7_refit_cadence_pre2025 import (
    LearnerSpec,
    SelectionSpec,
    activation_for,
    fit_cadence_folds,
)


DEFAULT_OUTPUT = "results/fresh_kimchi_orthogonal_alpha_audit_2026-07-16.json"
DEFAULT_DOCS = "docs/fresh-kimchi-orthogonal-alpha-audit-2026-07-16.md"
DEFAULT_INPUT = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
DEFAULT_FUNDING = (
    "data/binance_um_aux_btc_2020_2026/"
    "BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
)
DEFAULT_PREMIUM = (
    "data/binance_um_aux_btc_2020_2026/"
    "BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
)

WINDOWS: dict[str, tuple[str, str]] = {
    "fit_2020_2023": ("2020-01-01", "2024-01-01"),
    "fit_2023": ("2023-01-01", "2024-01-01"),
    "selection_2024": ("2024-01-01", "2025-01-01"),
    "eval_2025": ("2025-01-01", "2026-01-01"),
    "holdout_2026h1": ("2026-01-01", FULL_CUTOFF),
    "selection_2023_2024": ("2023-01-01", "2025-01-01"),
    "future_2025_2026h1": ("2025-01-01", FULL_CUTOFF),
    "oos_2024_2026h1": ("2024-01-01", FULL_CUTOFF),
    "all_2023_2026h1": ("2023-01-01", FULL_CUTOFF),
}

CANDIDATE_SPEC: dict[str, Any] = {
    "name": "fresh_kimchi_fx_bidirectional",
    "long_conditions": [
        {"feature": "funding_rate", "op": "le", "threshold": -1.67e-05},
        {"feature": "bd_flow_accel", "op": "ge", "threshold": 0.04082480376734276},
        {"feature": "kl_local_impulse_144", "op": "le", "threshold": -2.2728671107514256},
    ],
    "short_conditions": [
        {"feature": "usdkrw_momentum", "op": "ge", "threshold": 0.0026584830588613},
        {"feature": "htf_1d_return_1", "op": "le", "threshold": -0.03403745751424336},
    ],
    "availability": {
        "long": ["funding_available", "usdkrw_available", "kimchi_available"],
        "short": ["usdkrw_available"],
    },
    "hold_bars": 288,
    "stride_bars": 6,
    "take_bps": 400,
    "stop_bps": 250,
    "entry_delay_bars": 1,
    "leverage": 0.5,
    "fee_rate": 0.0005,
    "slippage_rate": 0.0001,
    "selection_history": {
        "base_threshold_fit": "2020-01-01 through 2023-12-31",
        "kimchi_gate_fit": "2020-01-01 through 2023-12-31 q0.10",
        "gate_rank_window": "2024",
        "future_research_already_viewed": True,
    },
}

ORTHOGONALITY_LIMITS = {
    "exact_entry_jaccard": 0.02,
    "candidate_entries_near_6h_fraction": 0.25,
    "position_jaccard": 0.15,
    "absolute_daily_pnl_pearson": 0.30,
}


@dataclass(frozen=True)
class Config:
    input_csv: str = DEFAULT_INPUT
    funding_csv: str = DEFAULT_FUNDING
    premium_csv: str = DEFAULT_PREMIUM
    output: str = DEFAULT_OUTPUT
    docs_output: str = DEFAULT_DOCS
    exclude_from: str = FULL_CUTOFF


def resolve_existing(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()
    fallback = Path("/home/pakchu/rllm") / path
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(path)


def load_funding(path: str, cutoff: str) -> pd.DataFrame:
    frame = pd.read_csv(resolve_existing(path), compression="infer")
    required = {"date", "funding_rate"}
    if not required.issubset(frame.columns):
        raise ValueError(f"funding source missing columns: {sorted(required - set(frame.columns))}")
    frame = frame[["date", "funding_rate"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise", format="mixed").dt.tz_convert(None)
    frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="raise")
    frame = frame[frame["date"] < pd.Timestamp(cutoff)]
    return frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def availability_mask(market: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    mask = np.ones(len(market), dtype=bool)
    for column in columns:
        if column not in market:
            raise ValueError(f"market source missing availability flag: {column}")
        values = pd.to_numeric(market[column], errors="coerce").to_numpy(float)
        mask &= np.isfinite(values) & (values > 0.5)
    return mask


def fresh_candidate_masks(market: pd.DataFrame, feature_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    long_conditions = [
        (row["feature"], row["op"], float(row["threshold"]))
        for row in CANDIDATE_SPEC["long_conditions"]
    ]
    short_conditions = [
        (row["feature"], row["op"], float(row["threshold"]))
        for row in CANDIDATE_SPEC["short_conditions"]
    ]
    raw_long = mk(feature_frame, long_conditions)
    raw_short = mk(feature_frame, short_conditions)
    long_fresh = availability_mask(market, CANDIDATE_SPEC["availability"]["long"])
    short_fresh = availability_mask(market, CANDIDATE_SPEC["availability"]["short"])
    long_active = raw_long & long_fresh
    short_active = raw_short & short_fresh
    diagnostics = {
        "raw_long_rows": int(raw_long.sum()),
        "raw_short_rows": int(raw_short.sum()),
        "fresh_long_rows": int(long_active.sum()),
        "fresh_short_rows": int(short_active.sum()),
        "blocked_stale_long_rows": int((raw_long & ~long_fresh).sum()),
        "blocked_stale_short_rows": int((raw_short & ~short_fresh).sum()),
        "fresh_long_availability_violations": int((long_active & ~long_fresh).sum()),
        "fresh_short_availability_violations": int((short_active & ~short_fresh).sum()),
    }
    return long_active, short_active, diagnostics


def build_candidate_context(cfg: Config) -> dict[str, Any]:
    legacy_cfg = LegacyConfig(
        input_csv=str(resolve_existing(cfg.input_csv)),
        output="/tmp/no_write_fresh_kimchi.json",
        funding_csv=str(resolve_existing(cfg.funding_csv)),
        premium_csv=str(resolve_existing(cfg.premium_csv)),
        exclude_from=cfg.exclude_from,
        leverage=float(CANDIDATE_SPEC["leverage"]),
        fee_rate=float(CANDIDATE_SPEC["fee_rate"]),
        slippage_rate=float(CANDIDATE_SPEC["slippage_rate"]),
    )
    # Import locally so this audit remains explicit about the legacy loader it
    # reproduces without changing the old search artifacts.
    from training.long_regime_combo_scan import _load_market

    market = _load_market(legacy_cfg)
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(cfg.exclude_from):
        raise RuntimeError("candidate market crossed the frozen cutoff")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise RuntimeError("candidate market is not a complete 5-minute grid")
    base = build_market_feature_frame(market, window_size=legacy_cfg.window_size)
    feature_frame = kimchi_features(
        market,
        extra(market, pd.concat([base, build_interest_features(market, base)], axis=1)),
    )
    long_active, short_active, freshness = fresh_candidate_masks(market, feature_frame)
    funding = load_funding(cfg.funding_csv, cfg.exclude_from)
    execution_cfg = ExecutionConfig(
        input_csv=str(resolve_existing(cfg.input_csv)),
        metrics_csv="",
        funding_csv=str(resolve_existing(cfg.funding_csv)),
        output="/tmp/no_write_fresh_kimchi.json",
        manifest_output="/tmp/no_write_fresh_kimchi_manifest.json",
        exclude_from=cfg.exclude_from,
        leverage=float(CANDIDATE_SPEC["leverage"]),
        fee_rate=float(CANDIDATE_SPEC["fee_rate"]),
        slippage_rate=float(CANDIDATE_SPEC["slippage_rate"]),
    )
    return {
        "market": market,
        "dates": dates,
        "features": feature_frame,
        "long_active": long_active,
        "short_active": short_active,
        "freshness": freshness,
        "funding": funding,
        "execution_cfg": execution_cfg,
        "engine": ExecutionEngine(market, funding, execution_cfg),
    }


def candidate_schedule(context: dict[str, Any], *, start: str, end: str) -> list[Trade]:
    dates: pd.Series = context["dates"]
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    hold = int(CANDIDATE_SPEC["hold_bars"])
    stride = int(CANDIDATE_SPEC["stride_bars"])
    positions = np.arange(143, len(dates) - hold - 2, stride, dtype=np.int64)
    active = np.logical_xor(context["long_active"], context["short_active"])
    positions = positions[period[positions] & active[positions]]
    trades: list[Trade] = []
    next_allowed = 0
    engine: ExecutionEngine = context["engine"]
    for signal in positions:
        signal = int(signal)
        if signal < next_allowed:
            continue
        side = 1 if bool(context["long_active"][signal]) else -1
        trade = engine.trade_at(
            signal,
            side,
            hold,
            int(CANDIDATE_SPEC["take_bps"]),
            int(CANDIDATE_SPEC["stop_bps"]),
        )
        if trade is None or not period[trade.exit_position]:
            continue
        trades.append(trade)
        next_allowed = trade.exit_position + 1
    return trades


def build_rank7_context(cfg: Config) -> dict[str, Any]:
    manifest, selection_result = validate_cadence_manifest()
    learner = LearnerSpec(**selection_result["learner"])
    policy = SelectionSpec(**selection_result["selection"])
    replay_cfg = Rank7Config(
        input_csv=cfg.input_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        exclude_from=cfg.exclude_from,
        output="/tmp/no_write_rank7.json",
        docs_output="",
    )
    base = build_replay_base(replay_cfg)
    folds = fit_cadence_folds(base, learner, "annual", start="2023-01-01", end=FULL_CUTOFF)
    active = activation_for(base, folds, policy)
    stats, hashes, _ = rank7_schedule_stats(base, active)
    assert_frozen_prefix(manifest, "annual", hashes)
    assert_annual_reference({"schedule_hashes": hashes, "stats": stats})
    return {
        "base": base,
        "active": active,
        "folds": folds,
        "reference_stats": stats,
        "reference_hashes": hashes,
    }


def rank7_schedule(context: dict[str, Any], *, start: str, end: str) -> list[Trade]:
    base = context["base"]
    return routed_schedule(
        base["context"],
        {"engine": base["engine"], "active": context["active"]},
        start=start,
        end=end,
    )


def trade_timing_overlap(
    candidate: Iterable[Trade], baseline: Iterable[Trade], *, total_bars: int, near_bars: int = 72
) -> dict[str, Any]:
    candidate = list(candidate)
    baseline = list(baseline)
    candidate_entries = np.asarray(sorted({trade.entry_position for trade in candidate}), dtype=np.int64)
    baseline_entries = np.asarray(sorted({trade.entry_position for trade in baseline}), dtype=np.int64)
    left, right = set(candidate_entries.tolist()), set(baseline_entries.tolist())
    union = left | right

    def near_fraction(source: np.ndarray, target: np.ndarray) -> float:
        if not len(source):
            return 0.0
        if not len(target):
            return 0.0
        return float(np.mean([np.min(np.abs(target - entry)) <= near_bars for entry in source]))

    def occupied(trades: list[Trade]) -> np.ndarray:
        mask = np.zeros(total_bars, dtype=bool)
        for trade in trades:
            start = max(0, int(trade.entry_position))
            end = min(total_bars, int(trade.exit_position))
            if end <= start:
                end = min(total_bars, start + 1)
            mask[start:end] = True
        return mask

    candidate_position = occupied(candidate)
    baseline_position = occupied(baseline)
    intersection = candidate_position & baseline_position
    position_union = candidate_position | baseline_position
    return {
        "candidate_entries": int(len(candidate_entries)),
        "baseline_entries": int(len(baseline_entries)),
        "shared_exact_entries": int(len(left & right)),
        "exact_entry_jaccard": float(len(left & right) / len(union)) if union else 0.0,
        "near_window_hours": float(near_bars * 5 / 60),
        "candidate_entries_near_6h_fraction": near_fraction(candidate_entries, baseline_entries),
        "baseline_entries_near_6h_fraction": near_fraction(baseline_entries, candidate_entries),
        "candidate_position_overlap_fraction": (
            float(intersection.sum() / candidate_position.sum()) if candidate_position.any() else 0.0
        ),
        "baseline_position_overlap_fraction": (
            float(intersection.sum() / baseline_position.sum()) if baseline_position.any() else 0.0
        ),
        "position_jaccard": float(intersection.sum() / position_union.sum()) if position_union.any() else 0.0,
    }


def daily_marked_returns(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    trades: Iterable[Trade],
    execution_cfg: ExecutionConfig,
    *,
    start: str,
    end: str,
) -> pd.Series:
    """Build zero-filled daily marked returns while preserving exact trade PnL.

    Open-to-open price factors and realized funding are placed on their actual
    calendar days.  Barrier fills differ from the open path, so the remaining
    price factor is booked on the exit day.  The product is asserted equal to
    the exact execution-engine factor for every trade and for the full window.
    """
    dates = pd.to_datetime(market["date"])
    opens = pd.to_numeric(market["open"], errors="raise").to_numpy(float)
    funding_dates = pd.to_datetime(funding["date"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    funding_rates = pd.to_numeric(funding["funding_rate"], errors="raise").to_numpy(float)
    day_index = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D", inclusive="left")
    factors = pd.Series(1.0, index=day_index, dtype=float)
    cost_factor = 1.0 - float(execution_cfg.leverage) * float(
        execution_cfg.fee_rate + execution_cfg.slippage_rate
    )
    exact_total = 1.0
    for trade in trades:
        entry_day = dates.iloc[trade.entry_position].normalize()
        exit_day = dates.iloc[trade.exit_position].normalize()
        factors.loc[entry_day] *= cost_factor
        price_path_factor = 1.0
        for position in range(int(trade.entry_position), int(trade.exit_position)):
            step = 1.0 + float(execution_cfg.leverage) * int(trade.side) * (
                opens[position + 1] / opens[position] - 1.0
            )
            if not np.isfinite(step) or step <= 0.0:
                raise ValueError("invalid open-to-open marked factor")
            factors.loc[dates.iloc[position + 1].normalize()] *= step
            price_path_factor *= step
        residual = float(trade.price_factor) / price_path_factor
        if not np.isfinite(residual) or residual <= 0.0:
            raise ValueError("invalid barrier residual factor")
        factors.loc[exit_day] *= residual

        entry_ns = int(dates.iloc[trade.entry_position].value)
        exit_ns = int(dates.iloc[trade.exit_position].value)
        left = int(np.searchsorted(funding_dates, entry_ns, side="left"))
        right = int(np.searchsorted(funding_dates, exit_ns, side="right"))
        realized = 1.0
        for timestamp, rate in zip(funding_dates[left:right], funding_rates[left:right], strict=True):
            funding_factor = 1.0 - float(execution_cfg.leverage) * int(trade.side) * float(rate)
            factors.loc[pd.Timestamp(timestamp).normalize()] *= funding_factor
            realized *= funding_factor
        if not np.isclose(realized, trade.funding_factor, rtol=1e-12, atol=1e-15):
            raise RuntimeError("marked funding path differs from execution engine")
        factors.loc[exit_day] *= cost_factor
        exact_trade = cost_factor * trade.price_factor * trade.funding_factor * cost_factor
        marked_trade = cost_factor * price_path_factor * residual * realized * cost_factor
        if not np.isclose(marked_trade, exact_trade, rtol=1e-12, atol=1e-15):
            raise RuntimeError("marked trade factor differs from exact execution factor")
        exact_total *= exact_trade
    if not np.isclose(float(factors.prod()), exact_total, rtol=1e-11, atol=1e-13):
        raise RuntimeError("daily marked path does not preserve exact compounded PnL")
    return factors - 1.0


def pnl_correlation(candidate: pd.Series, baseline: pd.Series) -> dict[str, float]:
    left, right = candidate.align(baseline, join="outer", fill_value=0.0)
    if len(left) < 2 or float(left.std(ddof=0)) == 0.0 or float(right.std(ddof=0)) == 0.0:
        return {"pearson": 0.0, "spearman": 0.0}
    return {
        "pearson": float(left.corr(right, method="pearson")),
        "spearman": float(left.corr(right, method="spearman")),
    }


def compact_stats(trades: list[Trade], *, start: str, end: str, cfg: ExecutionConfig) -> dict[str, Any]:
    stats = equity_stats(trades, start=start, end=end, cfg=cfg)
    stats["schedule_hash"] = _schedule_hash(trades)
    return stats


def render_docs(payload: dict[str, Any]) -> str:
    lines = [
        "# Fresh Kimchi/FX orthogonal alpha audit",
        "",
        "This audit pins the historical bidirectional policy and changes only one logically required rule: "
        "signals that depend on FX/Kimchi data fail closed when the current source bar is unavailable. "
        "No threshold was re-ranked on 2025/2026.",
        "",
        "## Candidate performance (realized funding included)",
        "",
        "| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "fit_2020_2023",
        "fit_2023",
        "selection_2024",
        "eval_2025",
        "holdout_2026h1",
        "future_2025_2026h1",
        "oos_2024_2026h1",
        "all_2023_2026h1",
    ):
        stats = payload["candidate_stats"][name]
        lines.append(
            f"| {name} | {stats['absolute_return_pct']:.4f}% | {stats['cagr_pct']:.4f}% | "
            f"{stats['strict_mdd_pct']:.4f}% | {stats['cagr_to_strict_mdd']:.4f} | "
            f"{stats['trades']} | {stats['longs']}/{stats['shorts']} |"
        )
    lines.extend(
        [
            "",
            "## Trade independence from frozen rank-7",
            "",
            "| Window | Exact entry Jaccard | Candidate entries within 6h | Position Jaccard | Candidate position overlap | Daily PnL Pearson | Spearman | Pass |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for name in ("selection_2023_2024", "future_2025_2026h1", "all_2023_2026h1"):
        row = payload["orthogonality"][name]
        lines.append(
            f"| {name} | {row['exact_entry_jaccard']:.4f} | "
            f"{row['candidate_entries_near_6h_fraction']:.4f} | {row['position_jaccard']:.4f} | "
            f"{row['candidate_position_overlap_fraction']:.4f} | {row['daily_pnl_correlation']['pearson']:.4f} | "
            f"{row['daily_pnl_correlation']['spearman']:.4f} | {row['passes_limits']} |"
        )
    fresh = payload["freshness"]
    lines.extend(
        [
            "",
            "## Integrity and interpretation",
            "",
            f"- Stale signal rows blocked: long `{fresh['blocked_stale_long_rows']}`, short "
            f"`{fresh['blocked_stale_short_rows']}`; post-gate availability violations are zero.",
            "- Entry is at the next 5-minute open; TP 4%, SL 2.5%, maximum hold 288 bars, "
            "0.5x, 6 bp/notional/side, realized funding, and split-contained exits.",
            "- Strict MDD includes intratrade adverse OHLC excursion and funding debit.",
            "- Full-calendar windows are used for every CAGR, including periods with no position.",
            "- Exact-entry Jaccard alone is not accepted as independence evidence; the audit also checks "
            "near entries, occupied bars, and zero-filled daily marked PnL.",
            "- 2025/2026 are not pristine discovery OOS because this family was viewed previously. "
            "The freshness repair was not tuned on those windows, but live promotion still requires new forward shadow data.",
            "",
            "## Verdict",
            "",
            payload["verdict"],
            "",
        ]
    )
    return "\n".join(lines)


def run(cfg: Config) -> dict[str, Any]:
    if cfg.exclude_from != FULL_CUTOFF:
        raise ValueError(f"exclude_from must equal frozen cutoff {FULL_CUTOFF}")
    candidate = build_candidate_context(cfg)
    rank7 = build_rank7_context(cfg)
    rank7_market = rank7["base"]["context"]["market"]
    if not np.array_equal(
        pd.to_datetime(candidate["market"]["date"]).to_numpy(),
        pd.to_datetime(rank7_market["date"]).to_numpy(),
    ):
        raise RuntimeError("candidate and rank-7 market grids differ")
    for column in ("open", "high", "low", "close"):
        if not np.allclose(
            candidate["market"][column].to_numpy(float),
            rank7_market[column].to_numpy(float),
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(f"candidate and rank-7 {column} differ")

    candidate_schedules: dict[str, list[Trade]] = {}
    rank7_schedules: dict[str, list[Trade]] = {}
    candidate_stats: dict[str, dict[str, Any]] = {}
    rank7_stats: dict[str, dict[str, Any]] = {}
    orthogonality: dict[str, dict[str, Any]] = {}
    for name, (start, end) in WINDOWS.items():
        candidate_trades = candidate_schedule(candidate, start=start, end=end)
        baseline_trades = rank7_schedule(rank7, start=start, end=end)
        candidate_schedules[name] = candidate_trades
        rank7_schedules[name] = baseline_trades
        candidate_stats[name] = compact_stats(
            candidate_trades, start=start, end=end, cfg=candidate["execution_cfg"]
        )
        rank7_stats[name] = compact_stats(
            baseline_trades, start=start, end=end, cfg=rank7["base"]["execution_cfg"]
        )
        if name in {"selection_2023_2024", "future_2025_2026h1", "all_2023_2026h1"}:
            overlap = trade_timing_overlap(
                candidate_trades,
                baseline_trades,
                total_bars=len(candidate["market"]),
            )
            candidate_daily = daily_marked_returns(
                candidate["market"],
                candidate["funding"],
                candidate_trades,
                candidate["execution_cfg"],
                start=start,
                end=end,
            )
            baseline_daily = daily_marked_returns(
                rank7_market,
                rank7["base"]["context"]["funding"],
                baseline_trades,
                rank7["base"]["execution_cfg"],
                start=start,
                end=end,
            )
            correlation = pnl_correlation(candidate_daily, baseline_daily)
            checks = {
                "exact_entry_jaccard": overlap["exact_entry_jaccard"]
                <= ORTHOGONALITY_LIMITS["exact_entry_jaccard"],
                "candidate_entries_near_6h_fraction": overlap["candidate_entries_near_6h_fraction"]
                <= ORTHOGONALITY_LIMITS["candidate_entries_near_6h_fraction"],
                "position_jaccard": overlap["position_jaccard"]
                <= ORTHOGONALITY_LIMITS["position_jaccard"],
                "absolute_daily_pnl_pearson": abs(correlation["pearson"])
                <= ORTHOGONALITY_LIMITS["absolute_daily_pnl_pearson"],
            }
            orthogonality[name] = {
                **overlap,
                "daily_pnl_correlation": correlation,
                "checks": checks,
                "passes_limits": all(checks.values()),
            }

    future_ratio = candidate_stats["future_2025_2026h1"]["cagr_to_strict_mdd"]
    future_independent = orthogonality["future_2025_2026h1"]["passes_limits"]
    verdict = (
        "Low-correlation regime-sleeve candidate: trade independence passes the declared audit limits, "
        f"but combined 2025-2026H1 CAGR/strict-MDD is {future_ratio:.4f}. "
        "Keep frozen in forward shadow; do not replace rank-7 or enable live until a new untouched window confirms it."
        if future_independent
        else "Rejected as an orthogonal sleeve because one or more trade-independence limits failed."
    )
    payload = {
        "schema_version": 1,
        "mode": "fresh_kimchi_fx_vs_frozen_rank7_trade_orthogonality_audit",
        "config": asdict(cfg),
        "candidate_spec": CANDIDATE_SPEC,
        "orthogonality_limits": ORTHOGONALITY_LIMITS,
        "future_is_pristine_discovery_oos": False,
        "freshness": candidate["freshness"],
        "candidate_stats": candidate_stats,
        "rank7_stats": rank7_stats,
        "orthogonality": orthogonality,
        "rank7_reference": {
            "folds": len(rank7["folds"]),
            "windows": [name for name, _, _ in RANK7_WINDOWS],
            "stats": rank7["reference_stats"],
            "schedule_hashes": rank7["reference_hashes"],
        },
        "verdict": verdict,
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if cfg.docs_output:
        docs = Path(cfg.docs_output)
        docs.parent.mkdir(parents=True, exist_ok=True)
        docs.write_text(render_docs(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=DEFAULT_INPUT)
    parser.add_argument("--funding-csv", default=DEFAULT_FUNDING)
    parser.add_argument("--premium-csv", default=DEFAULT_PREMIUM)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--docs-output", default=DEFAULT_DOCS)
    parser.add_argument("--exclude-from", default=FULL_CUTOFF)
    return parser.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    print(
        json.dumps(
            {
                "future": payload["candidate_stats"]["future_2025_2026h1"],
                "future_orthogonality": payload["orthogonality"]["future_2025_2026h1"],
                "verdict": payload["verdict"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
