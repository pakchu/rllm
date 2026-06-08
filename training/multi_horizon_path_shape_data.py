"""Build multi-horizon path-shape analyzer SFT records.

Prior experiments showed that directly predicting one action label is too brittle:
static TREND/FADE edges are weak while horizon-wise oracle upper bounds are
strong.  This dataset retargets the LLM toward describing future path shape and
risk across horizons, leaving action selection to a trader/RL layer.

Prompts remain past-only.  Targets use future OHLC path outcomes and are for
supervised analyzer training/evaluation, not deployable labels by themselves.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.analyzer_state_edge_report import _opposite, _trend_side as _source_edge_trend_side
from training.decision_analyzer_data import _past_summary_text
from training.decision_feature_learnability import load_jsonl, parse_jsonish
from training.edge_decay_analyzer_data import write_jsonl
from training.multi_horizon_edge_report import parse_horizons
from training.path_outcome_dataset import PathOutcomeConfig, compute_trade_path_outcome
from training.strict_bar_backtest import load_market_bars


@dataclass(frozen=True)
class MultiHorizonShapeConfig:
    hold_bars_list: tuple[int, ...] = (36, 72, 144, 288, 432)
    entry_delay_bars: int = 1
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    leverage: float = 0.5
    edge_threshold: float = 0.001
    strong_edge_threshold: float = 0.004
    low_mae: float = 0.005
    medium_mae: float = 0.015
    high_mae: float = 0.03


def _return_bucket(value: float, cfg: MultiHorizonShapeConfig) -> str:
    v = float(value)
    if v >= float(cfg.strong_edge_threshold):
        return "STRONG_POSITIVE"
    if v >= float(cfg.edge_threshold):
        return "POSITIVE"
    if v > 0.0:
        return "WEAK_POSITIVE"
    if v <= -float(cfg.strong_edge_threshold):
        return "STRONG_NEGATIVE"
    if v <= -float(cfg.edge_threshold):
        return "NEGATIVE"
    return "FLAT_NEGATIVE"


def _mae_bucket(value: float, cfg: MultiHorizonShapeConfig) -> str:
    v = float(value)
    if v <= float(cfg.low_mae):
        return "LOW"
    if v <= float(cfg.medium_mae):
        return "MEDIUM"
    if v <= float(cfg.high_mae):
        return "HIGH"
    return "EXTREME"


def _path_for_side(market, signal_pos: int, side: str, hold_bars: int, cfg: MultiHorizonShapeConfig):
    if side not in {"LONG", "SHORT"}:
        return None
    path_cfg = PathOutcomeConfig(
        hold_bars=int(hold_bars),
        entry_delay_bars=int(cfg.entry_delay_bars),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
        leverage=float(cfg.leverage),
    )
    return compute_trade_path_outcome(market, int(signal_pos), side, path_cfg)  # type: ignore[arg-type]


def _trend_side(record: dict[str, Any]) -> str:
    """Read past trend side from both old and enriched analyzer schemas."""

    side = _source_edge_trend_side(record)
    if side in {"LONG", "SHORT"}:
        return side
    target = parse_jsonish(record.get("target"))
    side = str(target.get("trend_side", "NONE"))
    return side if side in {"LONG", "SHORT"} else "NONE"


def _one_horizon_shape(record: dict[str, Any], market, hold_bars: int, cfg: MultiHorizonShapeConfig) -> dict[str, Any]:
    trend_side = _trend_side(record)
    fade_side = _opposite(trend_side)
    trend = _path_for_side(market, int(record.get("signal_pos", 0)), trend_side, hold_bars, cfg)
    fade = _path_for_side(market, int(record.get("signal_pos", 0)), fade_side, hold_bars, cfg)
    if trend is None or fade is None:
        return {
            "trend_return_bucket": "UNAVAILABLE",
            "fade_return_bucket": "UNAVAILABLE",
            "trend_mae_bucket": "UNAVAILABLE",
            "fade_mae_bucket": "UNAVAILABLE",
            "relative_edge": "NO_TREND_SIDE",
            "best_path": "NONE",
            "tradable_path_count": 0,
        }
    trend_net = float(trend.net_return)
    fade_net = float(fade.net_return)
    margin = trend_net - fade_net
    if trend_net <= 0.0 and fade_net <= 0.0:
        best_path = "NONE"
    elif margin >= float(cfg.edge_threshold):
        best_path = "TREND"
    elif margin <= -float(cfg.edge_threshold):
        best_path = "FADE"
    else:
        best_path = "MIXED"
    if margin >= float(cfg.strong_edge_threshold):
        relative_edge = "TREND_STRONGER"
    elif margin >= float(cfg.edge_threshold):
        relative_edge = "TREND_SLIGHTLY_STRONGER"
    elif margin <= -float(cfg.strong_edge_threshold):
        relative_edge = "FADE_STRONGER"
    elif margin <= -float(cfg.edge_threshold):
        relative_edge = "FADE_SLIGHTLY_STRONGER"
    else:
        relative_edge = "NO_CLEAR_EDGE"
    return {
        "trend_return_bucket": _return_bucket(trend_net, cfg),
        "fade_return_bucket": _return_bucket(fade_net, cfg),
        "trend_mae_bucket": _mae_bucket(float(trend.mae), cfg),
        "fade_mae_bucket": _mae_bucket(float(fade.mae), cfg),
        "relative_edge": relative_edge,
        "best_path": best_path,
        "tradable_path_count": int(trend_net > 0.0) + int(fade_net > 0.0),
    }


def derive_path_shape_target(record: dict[str, Any], market, cfg: MultiHorizonShapeConfig) -> dict[str, Any]:
    horizons = {str(h): _one_horizon_shape(record, market, int(h), cfg) for h in cfg.hold_bars_list}
    best_paths = [h["best_path"] for h in horizons.values()]
    trend_wins = sum(1 for x in best_paths if x == "TREND")
    fade_wins = sum(1 for x in best_paths if x == "FADE")
    mixed = sum(1 for x in best_paths if x == "MIXED")
    none = sum(1 for x in best_paths if x == "NONE")
    stable_threshold = max(1, int(math.ceil(len(best_paths) * 0.6)))
    if trend_wins >= stable_threshold and fade_wins == 0:
        direction_stability = "TREND_STABLE"
    elif fade_wins >= stable_threshold and trend_wins == 0:
        direction_stability = "FADE_STABLE"
    elif trend_wins and fade_wins:
        direction_stability = "HORIZON_CONFLICT"
    elif mixed:
        direction_stability = "MIXED_WEAK"
    else:
        direction_stability = "NO_STABLE_EDGE"
    reversal_pressure = "HIGH" if fade_wins >= 2 else "MEDIUM" if fade_wins == 1 or mixed >= 2 else "LOW"
    mae_buckets = [v for h in horizons.values() for k, v in h.items() if k.endswith("mae_bucket") and v != "UNAVAILABLE"]
    if "EXTREME" in mae_buckets:
        risk_profile = "EXTREME_PATH_RISK"
    elif mae_buckets.count("HIGH") >= 2:
        risk_profile = "HIGH_PATH_RISK"
    elif mae_buckets.count("LOW") >= max(1, len(mae_buckets) // 2):
        risk_profile = "LOW_PATH_RISK"
    else:
        risk_profile = "MIXED_PATH_RISK"
    return {
        "trend_side": _trend_side(record),
        "direction_stability": direction_stability,
        "reversal_pressure": reversal_pressure,
        "risk_profile": risk_profile,
        "horizons": horizons,
        "summary_counts": {"trend_wins": trend_wins, "fade_wins": fade_wins, "mixed": mixed, "none": none},
    }


def build_path_shape_prompt(record: dict[str, Any], cfg: MultiHorizonShapeConfig) -> str:
    summary = _past_summary_text(record)
    return "\n".join(
        [
            "You are the path-shape analyzer for a BTCUSDT futures trading system.",
            "Use only the past-only analyzer summary JSON below.",
            "Do not choose a final trade. Describe multi-horizon trend/fade path shape, adverse-excursion risk, reversal pressure, and uncertainty.",
            f"Horizons in bars: {','.join(str(x) for x in cfg.hold_bars_list)}.",
            "Return exactly one JSON object with keys trend_side, direction_stability, reversal_pressure, risk_profile, horizons, summary_counts.",
            "Each horizons entry must include trend_return_bucket, fade_return_bucket, trend_mae_bucket, fade_mae_bucket, relative_edge, best_path, tradable_path_count.",
            "Allowed best_path values: TREND, FADE, MIXED, NONE.",
            "",
            f"Past-only analyzer summary: {summary}",
        ]
    )


def build_path_shape_record(record: dict[str, Any], market, cfg: MultiHorizonShapeConfig) -> dict[str, Any]:
    target = derive_path_shape_target(record, market, cfg)
    return {
        "task": "multi_horizon_path_shape_analyzer",
        "date": record.get("date"),
        "signal_pos": record.get("signal_pos"),
        "prompt": build_path_shape_prompt(record, cfg),
        "target": json.dumps(target, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        "source_edge_target": record.get("source_edge_target"),
        "leakage_guard": {
            "prompt_uses_future_path": False,
            "target_uses_future_ohlc_paths": True,
            "action_orientation_uses_past_trend_side_only": True,
            "target_is_path_shape_not_final_action": True,
        },
    }


def build_path_shape_records(rows: list[dict[str, Any]], market, cfg: MultiHorizonShapeConfig) -> list[dict[str, Any]]:
    return [build_path_shape_record(row, market, cfg) for row in rows]


def summarize_path_shape_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "direction_stability": Counter(),
        "reversal_pressure": Counter(),
        "risk_profile": Counter(),
        "trend_side": Counter(),
    }
    horizon_best: dict[str, Counter[str]] = {}
    for rec in records:
        target = json.loads(str(rec.get("target", "{}")))
        for key, counter in counts.items():
            counter[str(target.get(key, ""))] += 1
        for h, hdata in (target.get("horizons") or {}).items():
            horizon_best.setdefault(str(h), Counter())[str(hdata.get("best_path", ""))] += 1
    return {
        "num_records": len(records),
        "period": {"start": records[0].get("date") if records else None, "end": records[-1].get("date") if records else None},
        **{key: dict(counter) for key, counter in counts.items()},
        "horizon_best_path": {h: dict(counter) for h, counter in sorted(horizon_best.items(), key=lambda x: int(x[0]))},
        "leakage_guard": {
            "prompts_are_past_only": True,
            "targets_use_future_ohlc_paths": True,
            "target_is_path_shape_not_final_action": True,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build multi-horizon path-shape analyzer SFT data")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--records", required=True, help="Decision/edge analyzer records containing past summary and trend_side")
    p.add_argument("--output", default="data/multi_horizon_path_shape_analyzer.jsonl")
    p.add_argument("--summary-output", default="")
    p.add_argument("--hold-bars-list", default="36,72,144,288,432")
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--max-records", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MultiHorizonShapeConfig(
        hold_bars_list=parse_horizons(args.hold_bars_list),
        entry_delay_bars=int(args.entry_delay_bars),
        fee_rate=float(args.fee_rate),
        slippage_rate=float(args.slippage_rate),
        leverage=float(args.leverage),
    )
    market = load_market_bars(args.market_csv)
    rows = load_jsonl(args.records)
    if args.max_records:
        rows = rows[: int(args.max_records)]
    records = build_path_shape_records(rows, market, cfg)
    write_jsonl(args.output, records)
    summary = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "inputs": {"market_csv": args.market_csv, "records": args.records},
        "config": asdict(cfg),
        "records": summarize_path_shape_records(records),
        "outputs": {"records": args.output},
    }
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
