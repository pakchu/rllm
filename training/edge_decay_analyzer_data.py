"""Build analyzer records for regime-transition and edge-decay learning.

This dataset intentionally avoids gate-threshold optimization.  The prompt uses
only past market/macro context; the target is a future path-derived diagnosis of
whether the current regime's directional edge persists, decays, or reverses.
It is meant to train an analyzer/router that can warn the trader/RL layer about
edge decay instead of directly emitting TRADE/NO_TRADE thresholds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.path_outcome_dataset import PathOutcomeConfig, compute_trade_path_outcome
from training.text_analyzer_trader_data import (
    analyzer_summary_to_text,
    build_analyzer_summary,
    load_market_frame,
)


@dataclass(frozen=True)
class EdgeDecayConfig:
    window_size: int = 144
    stride_bars: int = 12
    entry_delay_bars: int = 1
    short_hold_bars: int = 72
    long_hold_bars: int = 432
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    leverage: float = 0.5
    min_edge: float = 0.001
    max_mae: float = 0.02
    trend_feature: str = "trend_96"
    trend_threshold: float = 0.0025


def _side_from_trend(value: float, threshold: float) -> str:
    if float(value) >= float(threshold):
        return "LONG"
    if float(value) <= -float(threshold):
        return "SHORT"
    return "NONE"


def _opposite(side: str) -> str:
    if side == "LONG":
        return "SHORT"
    if side == "SHORT":
        return "LONG"
    return "NONE"


def _outcome_map(market: pd.DataFrame, signal_pos: int, side: str, hold_bars: int, cfg: EdgeDecayConfig) -> dict[str, float] | None:
    if side not in {"LONG", "SHORT"}:
        return None
    path_cfg = PathOutcomeConfig(
        hold_bars=int(hold_bars),
        entry_delay_bars=int(cfg.entry_delay_bars),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
        leverage=float(cfg.leverage),
    )
    out = compute_trade_path_outcome(market, signal_pos, side, path_cfg)  # type: ignore[arg-type]
    if out is None:
        return None
    return {
        "net_return": float(out.net_return),
        "mae": float(out.mae),
        "mfe": float(out.mfe),
        "gross_return": float(out.gross_return),
    }


def classify_edge_decay(
    *,
    trend_side: str,
    short_same: dict[str, float] | None,
    long_same: dict[str, float] | None,
    long_opposite: dict[str, float] | None,
    cfg: EdgeDecayConfig,
) -> dict[str, str]:
    """Classify future path behavior relative to the current past trend."""
    if trend_side == "NONE" or short_same is None or long_same is None:
        return {
            "edge_decay_label": "NO_CLEAR_TREND",
            "transition_label": "RANGE_UNKNOWN",
            "risk_label": "UNKNOWN",
        }

    short_net = float(short_same["net_return"])
    long_net = float(long_same["net_return"])
    opp_net = float(long_opposite["net_return"]) if long_opposite is not None else 0.0
    long_mae = float(long_same["mae"])
    min_edge = float(cfg.min_edge)

    if long_mae > float(cfg.max_mae) and long_net <= 0.0:
        edge = "ADVERSE_STRESS"
    elif short_net >= min_edge and long_net <= 0.0:
        edge = "EDGE_DECAY"
    elif long_net >= min_edge and short_net >= 0.0:
        edge = "EDGE_PERSIST"
    elif long_net <= -min_edge and opp_net >= min_edge:
        edge = "REVERSAL_RISK"
    elif abs(long_net) < min_edge and abs(opp_net) < min_edge:
        edge = "NO_EDGE"
    elif long_net > 0.0:
        edge = "WEAK_PERSIST"
    else:
        edge = "WEAK_DECAY"

    if long_net >= min_edge and long_net >= opp_net:
        transition = "TREND_CONTINUATION"
    elif opp_net >= min_edge and opp_net > long_net:
        transition = "TREND_REVERSAL"
    else:
        transition = "CHOP_OR_DECAY"

    if long_mae <= 0.005:
        risk = "LOW_ADVERSE_EXCURSION"
    elif long_mae <= float(cfg.max_mae):
        risk = "MANAGEABLE_ADVERSE_EXCURSION"
    else:
        risk = "HIGH_ADVERSE_EXCURSION"

    return {"edge_decay_label": edge, "transition_label": transition, "risk_label": risk}


def build_edge_decay_record(
    market: pd.DataFrame,
    signal_pos: int,
    cfg: EdgeDecayConfig,
    *,
    feature_frame: pd.DataFrame,
) -> dict[str, Any] | None:
    feature_row = feature_frame.iloc[int(signal_pos)]
    trend_value = float(feature_row.get(cfg.trend_feature, 0.0))
    trend_side = _side_from_trend(trend_value, cfg.trend_threshold)
    opposite_side = _opposite(trend_side)

    short_same = _outcome_map(market, signal_pos, trend_side, cfg.short_hold_bars, cfg)
    long_same = _outcome_map(market, signal_pos, trend_side, cfg.long_hold_bars, cfg)
    long_opp = _outcome_map(market, signal_pos, opposite_side, cfg.long_hold_bars, cfg)
    if trend_side != "NONE" and (short_same is None or long_same is None or long_opp is None):
        return None

    labels = classify_edge_decay(
        trend_side=trend_side,
        short_same=short_same,
        long_same=long_same,
        long_opposite=long_opp,
        cfg=cfg,
    )
    summary = build_analyzer_summary(market, signal_pos, window_size=cfg.window_size, feature_frame=feature_frame)
    summary_text = analyzer_summary_to_text(summary)
    ts = str(pd.to_datetime(market.iloc[int(signal_pos)]["date"]))
    target = {
        "trend_side": trend_side,
        "trend_feature": cfg.trend_feature,
        "edge_decay_label": labels["edge_decay_label"],
        "transition_label": labels["transition_label"],
        "risk_label": labels["risk_label"],
        "recommended_router_hint": _router_hint(labels["edge_decay_label"], labels["transition_label"]),
    }
    return {
        "task": "edge_decay_analyzer",
        "date": ts,
        "signal_pos": int(signal_pos),
        "prompt": build_edge_decay_prompt(summary_text, cfg),
        "target": json.dumps(target, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        "past_summary": summary,
        "trend_value": trend_value,
        "path_diagnostics": {
            "short_same": short_same,
            "long_same": long_same,
            "long_opposite": long_opp,
        },
        "leakage_guard": {
            "prompt_uses_future_path": False,
            "target_uses_future_path": True,
            "input_bars_end_at_signal_pos": int(signal_pos),
            "external_features_joined_backward_asof": True,
            "not_gate_threshold_optimization": True,
        },
    }


def _router_hint(edge_label: str, transition_label: str) -> str:
    if edge_label == "EDGE_PERSIST" and transition_label == "TREND_CONTINUATION":
        return "ALLOW_TREND_SPECIALIST"
    if edge_label in {"EDGE_DECAY", "ADVERSE_STRESS"}:
        return "REDUCE_OR_SKIP_TREND_SPECIALIST"
    if transition_label == "TREND_REVERSAL":
        return "CONSIDER_REVERSAL_SPECIALIST"
    if edge_label == "NO_CLEAR_TREND":
        return "RANGE_ROUTER_ONLY"
    return "LOW_CONFIDENCE_ROUTER"


def build_edge_decay_prompt(summary_text: str, cfg: EdgeDecayConfig) -> str:
    return "\n".join(
        [
            "You are the analyzer/router stage for a BTCUSDT futures trading system.",
            "Use only the past-only market and macro summary.",
            "Do not emit TRADE/NO_TRADE. Diagnose whether the current edge is likely to persist, decay, or reverse.",
            f"Short horizon bars: {int(cfg.short_hold_bars)}; long horizon bars: {int(cfg.long_hold_bars)}.",
            "Return exactly one JSON object with keys trend_side, edge_decay_label, transition_label, risk_label, recommended_router_hint.",
            "",
            f"Past-only analyzer summary: {summary_text}",
        ]
    )


def iter_signal_positions(market: pd.DataFrame, cfg: EdgeDecayConfig, start_date: str | None, end_date: str | None):
    start_ts = pd.to_datetime(start_date) if start_date else None
    end_ts = pd.to_datetime(end_date) if end_date else None
    last_signal_pos = len(market) - max(1, int(cfg.entry_delay_bars)) - max(1, int(cfg.long_hold_bars)) - 1
    for pos in range(max(0, int(cfg.window_size) - 1), max(0, last_signal_pos) + 1, max(1, int(cfg.stride_bars))):
        ts = pd.to_datetime(market.iloc[pos]["date"])
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        yield pos


def build_edge_decay_records(
    market: pd.DataFrame,
    cfg: EdgeDecayConfig,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    feature_frame = build_market_feature_frame(market, window_size=cfg.window_size)
    records: list[dict[str, Any]] = []
    for pos in iter_signal_positions(market, cfg, start_date, end_date):
        rec = build_edge_decay_record(market, pos, cfg, feature_frame=feature_frame)
        if rec is None:
            continue
        records.append(rec)
        if max_records is not None and len(records) >= int(max_records):
            break
    return records


def summarize_edge_decay_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_records": 0}
    counts: dict[str, dict[str, int]] = {"edge_decay_label": {}, "transition_label": {}, "risk_label": {}, "router_hint": {}}
    for rec in records:
        target = json.loads(rec["target"])
        for src, key in [
            ("edge_decay_label", "edge_decay_label"),
            ("transition_label", "transition_label"),
            ("risk_label", "risk_label"),
            ("recommended_router_hint", "router_hint"),
        ]:
            val = str(target[src])
            counts[key][val] = counts[key].get(val, 0) + 1
    return {
        "num_records": len(records),
        "period": {"start": records[0]["date"], "end": records[-1]["date"]},
        **counts,
        "leakage_guard": {
            "prompts_are_past_only": True,
            "targets_are_future_path_labels": True,
            "not_gate_threshold_optimization": True,
        },
    }


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build edge-decay analyzer/router SFT data")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", default="data/edge_decay_analyzer.jsonl")
    p.add_argument("--summary-output", default="")
    p.add_argument("--start-date", default="")
    p.add_argument("--end-date", default="")
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--window-size", type=int, default=144)
    p.add_argument("--stride-bars", type=int, default=12)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--short-hold-bars", type=int, default=72)
    p.add_argument("--long-hold-bars", type=int, default=432)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--min-edge", type=float, default=0.001)
    p.add_argument("--max-mae", type=float, default=0.02)
    p.add_argument("--trend-feature", default="trend_96")
    p.add_argument("--trend-threshold", type=float, default=0.0025)
    p.add_argument("--wave-trading-root", default="", help="Optional wave_trading root for DXY/Kimchi cached external features")
    p.add_argument("--external-tolerance", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EdgeDecayConfig(
        window_size=args.window_size,
        stride_bars=args.stride_bars,
        entry_delay_bars=args.entry_delay_bars,
        short_hold_bars=args.short_hold_bars,
        long_hold_bars=args.long_hold_bars,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        leverage=args.leverage,
        min_edge=args.min_edge,
        max_mae=args.max_mae,
        trend_feature=args.trend_feature,
        trend_threshold=args.trend_threshold,
    )
    market = load_market_frame(
        args.market_csv,
        wave_trading_root=args.wave_trading_root or None,
        external_tolerance=args.external_tolerance or None,
    )
    records = build_edge_decay_records(
        market,
        cfg,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        max_records=args.max_records or None,
    )
    write_jsonl(args.output, records)
    summary = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "records": summarize_edge_decay_records(records),
        "outputs": {"records": args.output},
    }
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
