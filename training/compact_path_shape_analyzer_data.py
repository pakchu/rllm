"""Compress multi-horizon path-shape analyzer targets into router states.

The first Gemma4 path-shape SFT learned trend_side but struggled with the long,
nested horizon JSON.  This module distills those teacher records into a compact
state/action-orientation target: which path family looks best, which horizon
bucket to hand to the trader/RL layer, and how much risk budget to allow.

Inputs are existing path-shape analyzer records.  Prompts remain past-only by
reusing the source prompt context; targets are still supervised labels derived
from future OHLC path-shape labels, so they are not deployable oracle signals by
 themselves.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.decision_feature_learnability import load_jsonl
from training.edge_decay_analyzer_data import write_jsonl
from training.eval_multi_horizon_path_shape_analyzer import parse_path_shape_json
from training.multi_horizon_edge_report import parse_horizons

RETURN_SCORE = {
    "STRONG_POSITIVE": 4.0,
    "POSITIVE": 3.0,
    "WEAK_POSITIVE": 1.0,
    "FLAT_NEGATIVE": -0.5,
    "NEGATIVE": -2.0,
    "STRONG_NEGATIVE": -4.0,
    "UNAVAILABLE": -99.0,
}
MAE_PENALTY = {"LOW": 0.0, "MEDIUM": 0.75, "HIGH": 1.75, "EXTREME": 3.0, "UNAVAILABLE": 99.0}


@dataclass(frozen=True)
class CompactPathShapeConfig:
    hold_bars_list: tuple[int, ...] = (36, 72, 144, 288, 432)
    min_trade_score: float = 1.25
    strong_score: float = 3.0
    moderate_score: float = 2.0
    long_horizon_bars: int = 288
    short_horizon_bars: int = 72


def _horizon_policy(horizon: int, cfg: CompactPathShapeConfig) -> str:
    if horizon <= int(cfg.short_horizon_bars):
        return "SHORT_STEP"
    if horizon < int(cfg.long_horizon_bars):
        return "MID_STEP"
    return "LONG_STEP"


def _path_score(hdata: dict[str, Any], path: str) -> float:
    if path == "TREND":
        ret = str(hdata.get("trend_return_bucket", "UNAVAILABLE"))
        mae = str(hdata.get("trend_mae_bucket", "UNAVAILABLE"))
    elif path == "FADE":
        ret = str(hdata.get("fade_return_bucket", "UNAVAILABLE"))
        mae = str(hdata.get("fade_mae_bucket", "UNAVAILABLE"))
    else:
        return -99.0
    return float(RETURN_SCORE.get(ret, -99.0)) - float(MAE_PENALTY.get(mae, 99.0))


def _risk_budget(hdata: dict[str, Any], path: str, risk_profile: str) -> str:
    mae = "UNAVAILABLE"
    if path == "TREND":
        mae = str(hdata.get("trend_mae_bucket", "UNAVAILABLE"))
    elif path == "FADE":
        mae = str(hdata.get("fade_mae_bucket", "UNAVAILABLE"))
    if risk_profile == "EXTREME_PATH_RISK" or mae == "EXTREME":
        return "AVOID_OR_TINY"
    if risk_profile == "HIGH_PATH_RISK" or mae == "HIGH":
        return "SMALL"
    if mae == "MEDIUM" or risk_profile == "MIXED_PATH_RISK":
        return "NORMAL"
    return "AGGRESSIVE_OK"


def derive_compact_path_shape_target(source_target: str | dict[str, Any], cfg: CompactPathShapeConfig) -> dict[str, Any]:
    if isinstance(source_target, str):
        parsed = parse_path_shape_json(source_target, horizons=cfg.hold_bars_list)
    else:
        parsed = parse_path_shape_json(json.dumps(source_target), horizons=cfg.hold_bars_list)

    risk_profile = str(parsed.get("risk_profile", "MIXED_PATH_RISK"))
    candidates: list[dict[str, Any]] = []
    for h in cfg.hold_bars_list:
        hkey = str(int(h))
        hdata = dict((parsed.get("horizons") or {}).get(hkey) or {})
        for path in ("TREND", "FADE"):
            score = _path_score(hdata, path)
            candidates.append({"horizon_bars": int(h), "path": path, "score": score, "hdata": hdata})
    best = max(candidates, key=lambda x: (float(x["score"]), int(x["horizon_bars"]))) if candidates else None
    if best is None or float(best["score"]) < float(cfg.min_trade_score):
        action_path = "NONE"
        horizon_bars = 0
        horizon_policy = "SKIP_STEP"
        edge_quality = "NO_EDGE"
        risk_budget = "AVOID_OR_TINY"
        score_bucket = "NEGATIVE_OR_TOO_WEAK"
    else:
        action_path = str(best["path"])
        horizon_bars = int(best["horizon_bars"])
        horizon_policy = _horizon_policy(horizon_bars, cfg)
        score = float(best["score"])
        if score >= float(cfg.strong_score):
            edge_quality = "STRONG"
            score_bucket = "HIGH"
        elif score >= float(cfg.moderate_score):
            edge_quality = "MODERATE"
            score_bucket = "MEDIUM"
        else:
            edge_quality = "WEAK"
            score_bucket = "LOW"
        risk_budget = _risk_budget(dict(best["hdata"]), action_path, risk_profile)

    return {
        "trend_side": parsed.get("trend_side", "NONE"),
        "action_path": action_path,
        "horizon_bars": horizon_bars,
        "horizon_policy": horizon_policy,
        "edge_quality": edge_quality,
        "risk_budget": risk_budget,
        "score_bucket": score_bucket,
        "direction_stability": parsed.get("direction_stability", "NO_STABLE_EDGE"),
        "reversal_pressure": parsed.get("reversal_pressure", "LOW"),
    }


def build_compact_prompt(source_record: dict[str, Any], cfg: CompactPathShapeConfig) -> str:
    prompt = str(source_record.get("prompt", ""))
    if "Past-only analyzer summary:" in prompt:
        past_summary = prompt.split("Past-only analyzer summary:", 1)[1].strip()
    else:
        past_summary = prompt[-3000:]
    return "\n".join(
        [
            "You are the compact router-state analyzer for a BTCUSDT futures trading system.",
            "Use only the past-only analyzer summary below.",
            "Do not output a final order. Choose the most useful router state for a downstream trader/RL layer.",
            f"Allowed horizon_bars: 0,{','.join(str(x) for x in cfg.hold_bars_list)} where 0 means skip/no edge.",
            "Return exactly one compact JSON object with keys trend_side, action_path, horizon_bars, horizon_policy, edge_quality, risk_budget, score_bucket, direction_stability, reversal_pressure.",
            "Allowed action_path: TREND, FADE, NONE. Allowed horizon_policy: SHORT_STEP, MID_STEP, LONG_STEP, SKIP_STEP.",
            "Allowed edge_quality: STRONG, MODERATE, WEAK, NO_EDGE. Allowed risk_budget: AGGRESSIVE_OK, NORMAL, SMALL, AVOID_OR_TINY.",
            "",
            f"Past-only analyzer summary: {past_summary}",
        ]
    )


def build_compact_record(source_record: dict[str, Any], cfg: CompactPathShapeConfig) -> dict[str, Any]:
    target = derive_compact_path_shape_target(str(source_record.get("target", "{}")), cfg)
    return {
        "task": "compact_path_shape_analyzer",
        "date": source_record.get("date"),
        "signal_pos": source_record.get("signal_pos"),
        "prompt": build_compact_prompt(source_record, cfg),
        "target": json.dumps(target, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        "source_task": source_record.get("task"),
        "leakage_guard": {
            "prompt_uses_future_path": False,
            "target_is_compressed_from_future_path_shape_label": True,
            "target_is_final_order": False,
            "target_is_for_router_state_not_execution": True,
        },
    }


def build_compact_records(rows: list[dict[str, Any]], cfg: CompactPathShapeConfig) -> list[dict[str, Any]]:
    return [build_compact_record(row, cfg) for row in rows]


def summarize_compact_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    counters = {k: Counter() for k in ("action_path", "horizon_bars", "horizon_policy", "edge_quality", "risk_budget", "trend_side")}
    target_lens: list[int] = []
    prompt_lens: list[int] = []
    for rec in records:
        target_lens.append(len(str(rec.get("target", ""))))
        prompt_lens.append(len(str(rec.get("prompt", ""))))
        target = json.loads(str(rec.get("target", "{}")))
        for key, counter in counters.items():
            counter[str(target.get(key, ""))] += 1
    return {
        "num_records": len(records),
        "period": {"start": records[0].get("date") if records else None, "end": records[-1].get("date") if records else None},
        "prompt_chars": {"min": min(prompt_lens) if prompt_lens else 0, "max": max(prompt_lens) if prompt_lens else 0, "mean": sum(prompt_lens) / max(1, len(prompt_lens))},
        "target_chars": {"min": min(target_lens) if target_lens else 0, "max": max(target_lens) if target_lens else 0, "mean": sum(target_lens) / max(1, len(target_lens))},
        **{key: dict(counter) for key, counter in counters.items()},
        "leakage_guard": {
            "prompts_are_past_only": True,
            "targets_are_compressed_from_future_path_shape_labels": True,
            "targets_are_router_states_not_final_orders": True,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compress multi-horizon path-shape analyzer records into compact router-state SFT data")
    p.add_argument("--records", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--hold-bars-list", default="36,72,144,288,432")
    p.add_argument("--min-trade-score", type=float, default=1.25)
    p.add_argument("--strong-score", type=float, default=3.0)
    p.add_argument("--moderate-score", type=float, default=2.0)
    p.add_argument("--short-horizon-bars", type=int, default=72)
    p.add_argument("--long-horizon-bars", type=int, default=288)
    p.add_argument("--max-records", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CompactPathShapeConfig(
        hold_bars_list=parse_horizons(args.hold_bars_list),
        min_trade_score=float(args.min_trade_score),
        strong_score=float(args.strong_score),
        moderate_score=float(args.moderate_score),
        short_horizon_bars=int(args.short_horizon_bars),
        long_horizon_bars=int(args.long_horizon_bars),
    )
    rows = load_jsonl(args.records)
    if args.max_records:
        rows = rows[: int(args.max_records)]
    records = build_compact_records(rows, cfg)
    write_jsonl(args.output, records)
    summary = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "inputs": {"records": args.records},
        "config": asdict(cfg),
        "records": summarize_compact_records(records),
        "outputs": {"records": args.output},
    }
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
