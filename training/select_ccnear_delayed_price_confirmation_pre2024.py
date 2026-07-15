#!/usr/bin/env python3
"""Test one causal post-event price-confirmation direction on the CC-near clock.

The book event is treated as a directionless volatility alert.  After exactly
six completed five-minute bars, trade in the direction of the observed price
impulse and shorten the hold so the original event-horizon exit is unchanged.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.evaluate_metaorder_fragmentation_impact_curvature import weekly_cluster_sign_flip
from training.preregister_cross_collateral_liquidity_void_refill import lagged_robust_zscore
from training.search_inventory_purge_reclaim_alpha import (
    Config as ExecutionConfig,
    ExecutionEngine,
    Trade,
    _schedule_hash,
    equity_stats,
)
from training.select_cross_collateral_event_direction_pre2024 import exact_factor
from training.select_cross_collateral_near_pressure_pre2024 import (
    Config as ClockConfig,
    EXPECTED_SELECTED as CLOCK_SPEC,
    event_mask,
    load_sources,
    raw_pressure,
)


DEFAULT_OUTPUT = "results/ccnear_delayed_price_confirmation_pre2024_2026-07-16.json"
DEFAULT_DOCS = "docs/ccnear-delayed-price-confirmation-pre2024-2026-07-16.md"
CONFIRMATION_BARS = 6
HOLD_BARS = int(CLOCK_SPEC["hold_bars"]) - CONFIRMATION_BARS
PERMUTATIONS = 100_000
SEED = 20_260_716
WINDOWS = {
    "fit_2023h1": ("2023-01-01", "2023-07-01"),
    "confirm_2023h2": ("2023-07-01", "2024-01-01"),
    "q3_2023": ("2023-07-01", "2023-10-01"),
    "q4_2023": ("2023-10-01", "2024-01-01"),
    "full_2023": ("2023-01-01", "2024-01-01"),
}


@dataclass(frozen=True)
class Config:
    output: str = DEFAULT_OUTPUT
    docs_output: str = DEFAULT_DOCS


def build_context() -> dict[str, Any]:
    clock_cfg = ClockConfig(output="/tmp/no_write.json", manifest_output="/tmp/no_write.json", docs_output="")
    shells, credibility, market, funding, source = load_sources(clock_cfg)
    weights = (1.0, 0.5, 0.0, 0.0, 0.0)
    venue_scores = []
    for venue in ("um", "cm"):
        venue_scores.append(
            lagged_robust_zscore(
                raw_pressure(
                    shells,
                    credibility,
                    venue=venue,
                    weights=weights,
                    credibility_weighted=False,
                ),
                window=clock_cfg.robust_window_bars,
                minimum=clock_cfg.robust_min_periods,
            )
        )
    score = (venue_scores[0] + venue_scores[1]) / np.sqrt(2.0)
    score = score.where(shells["source_complete"].astype(bool) & score.notna())
    onset, _ = event_mask(score, float(CLOCK_SPEC["threshold"]))
    execution_cfg = ExecutionConfig(
        input_csv="",
        metrics_csv="",
        funding_csv="",
        output="/tmp/no_write_confirmation.json",
        manifest_output="/tmp/no_write_confirmation_manifest.json",
        exclude_from="2024-01-01",
        leverage=0.5,
        fee_rate=0.0005,
        slippage_rate=0.0001,
    )
    return {
        "market": market,
        "dates": pd.to_datetime(market["date"]),
        "onset": onset,
        "score": score,
        "engine": ExecutionEngine(market, funding, execution_cfg),
        "execution_cfg": execution_cfg,
        "source": source,
    }


def confirmation_schedule(
    context: dict[str, Any], *, start: str, end: str, reverse: bool = False
) -> list[Trade]:
    dates = context["dates"]
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    close = pd.to_numeric(context["market"]["close"], errors="raise").to_numpy(float)
    trades: list[Trade] = []
    next_allowed = 0
    for event in np.flatnonzero(context["onset"] & period):
        event = int(event)
        decision = event + CONFIRMATION_BARS
        if decision >= len(close) or decision < next_allowed or not period[decision]:
            continue
        impulse = close[decision] / close[event] - 1.0
        if not np.isfinite(impulse) or impulse == 0.0:
            continue
        side = 1 if impulse > 0.0 else -1
        if reverse:
            side *= -1
        trade = context["engine"].trade_at(
            decision, side, HOLD_BARS, 1_000_000, 1_000_000
        )
        if trade is None or not period[trade.exit_position]:
            continue
        # Original event policy entered at event+1 and exited at event+289.
        if trade.exit_position != event + int(CLOCK_SPEC["hold_bars"]) + 1:
            raise RuntimeError("price confirmation changed the frozen event horizon")
        trades.append(trade)
        next_allowed = trade.exit_position + 1
    return trades


def compact_stats(
    trades: list[Trade], context: dict[str, Any], *, start: str, end: str, permutations: int = 0
) -> dict[str, Any]:
    stats = {
        **equity_stats(trades, start=start, end=end, cfg=context["execution_cfg"]),
        "schedule_hash": _schedule_hash(trades),
    }
    if permutations:
        stats["weekly_cluster_sign_flip"] = weekly_cluster_sign_flip(
            [exact_factor(trade, context["execution_cfg"]) - 1.0 for trade in trades],
            [trade.entry_date for trade in trades],
            permutations=permutations,
            seed=SEED,
        )
    return stats


def render_docs(payload: dict[str, Any]) -> str:
    lines = [
        "# CC-near delayed price confirmation — pre-2024",
        "",
        "One fixed rule: wait 30 minutes after the directionless book event, follow the observed "
        "price impulse, and preserve the original event-horizon exit.",
        "",
        "| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in payload["stats"].items():
        lines.append(
            f"| {name} | {stats['absolute_return_pct']:.4f}% | {stats['cagr_pct']:.4f}% | "
            f"{stats['strict_mdd_pct']:.4f}% | {stats['cagr_to_strict_mdd']:.4f} | "
            f"{stats['trades']} | {stats['longs']}/{stats['shorts']} |"
        )
    lines.extend(["", f"Verdict: **{payload['verdict']}**", ""])
    return "\n".join(lines)


def run(cfg: Config) -> dict[str, Any]:
    context = build_context()
    stats: dict[str, dict[str, Any]] = {}
    reverse: dict[str, dict[str, Any]] = {}
    for name, (start, end) in WINDOWS.items():
        primary = confirmation_schedule(context, start=start, end=end)
        stats[name] = compact_stats(
            primary,
            context,
            start=start,
            end=end,
            permutations=PERMUTATIONS if name == "confirm_2023h2" else 0,
        )
        reverse[name] = compact_stats(
            confirmation_schedule(context, start=start, end=end, reverse=True),
            context,
            start=start,
            end=end,
        )
    h2 = stats["confirm_2023h2"]
    checks = {
        "h2_positive": h2["absolute_return_pct"] > 0.0,
        "h2_ratio_at_least_3": h2["cagr_to_strict_mdd"] >= 3.0,
        "h2_mdd_at_most_15": h2["strict_mdd_pct"] <= 15.0,
        "h2_long_support": h2["longs"] >= 15,
        "h2_short_support": h2["shorts"] >= 15,
        "q3_positive": stats["q3_2023"]["absolute_return_pct"] > 0.0,
        "q4_positive": stats["q4_2023"]["absolute_return_pct"] > 0.0,
        "weekly_cluster_p_below_0_10": h2["weekly_cluster_sign_flip"]["p_value_one_sided"] < 0.10,
        "beats_reverse_h2": h2["cagr_to_strict_mdd"] > reverse["confirm_2023h2"]["cagr_to_strict_mdd"],
    }
    future_ready = all(checks.values())
    payload = {
        "schema_version": 1,
        "mode": "ccnear_delayed_price_confirmation_pre2024",
        "candidate_count": 1,
        "post_2023_rows_opened": False,
        "clock_spec": CLOCK_SPEC,
        "candidate": {
            "name": "ccnear_price_confirmation_30m_v1",
            "confirmation_bars": CONFIRMATION_BARS,
            "confirmation_minutes": CONFIRMATION_BARS * 5,
            "side": "sign(close[event+6] / close[event] - 1)",
            "hold_bars_after_confirmation": HOLD_BARS,
            "original_event_horizon_preserved": True,
            "abstention": False,
        },
        "stats": stats,
        "reverse_control": reverse,
        "gate_checks": checks,
        "future_ready": future_ready,
        "verdict": "freeze before future replay" if future_ready else "reject before future replay",
        "source": context["source"],
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
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--docs-output", default=DEFAULT_DOCS)
    return parser.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    print(
        json.dumps(
            {
                "candidate": payload["candidate"],
                "stats": payload["stats"],
                "gate_checks": payload["gate_checks"],
                "future_ready": payload["future_ready"],
                "verdict": payload["verdict"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
