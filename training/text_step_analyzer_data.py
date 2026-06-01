"""Build text analyzer/trader data where analyzer also selects execution step.

This stage tests the hypothesis that the LLM should reason over setup horizon
instead of forcing one fixed hold length.  Analyzer targets include a supervised
preferred hold step selected from executable path outcomes across multiple
candidate horizons; trader then consumes that analyzer output.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from training.path_outcome_dataset import PathOutcomeConfig, make_path_outcome_record, summarize_records
from training.text_analyzer_trader_data import (
    TextPipelineConfig,
    analyzer_summary_to_text,
    build_analyzer_input,
    build_analyzer_summary,
    build_trader_input,
    load_market_frame,
    write_jsonl,
)
from preprocessing.market_features import build_market_feature_frame


@dataclass(frozen=True)
class StepAnalyzerConfig(TextPipelineConfig):
    hold_candidates: tuple[int, ...] = (48, 96, 144, 288)
    feature_profile: str = "rich_v1"


def parse_hold_candidates(raw: str) -> tuple[int, ...]:
    vals = tuple(sorted({max(1, int(x.strip())) for x in str(raw).split(",") if x.strip()}))
    if not vals:
        raise ValueError("hold candidates must not be empty")
    return vals


def _path_cfg_for_hold(cfg: StepAnalyzerConfig, hold_bars: int) -> PathOutcomeConfig:
    return PathOutcomeConfig(
        hold_bars=int(hold_bars),
        entry_delay_bars=cfg.entry_delay_bars,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        leverage=cfg.leverage,
        mae_penalty=cfg.mae_penalty,
        mfe_bonus=cfg.mfe_bonus,
        hold_margin=cfg.min_utility,
        min_net_return=cfg.min_net_return,
        max_mae=cfg.max_mae,
    )


def _path_record_for_hold(market: pd.DataFrame, signal_pos: int, cfg: StepAnalyzerConfig, hold_bars: int) -> dict[str, Any] | None:
    return make_path_outcome_record(market, signal_pos, _path_cfg_for_hold(cfg, hold_bars))


def _record_score(rec: dict[str, Any]) -> tuple[float, float, float]:
    gate_bonus = 1.0 if str(rec["trade_gate_label"]) == "TRADE" else 0.0
    return (gate_bonus, float(rec["best_utility"]), float(rec["best_net_return"]) - float(rec["best_mae"]))


def choose_step_record(market: pd.DataFrame, signal_pos: int, cfg: StepAnalyzerConfig) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for hold in cfg.hold_candidates:
        rec = _path_record_for_hold(market, signal_pos, cfg, hold)
        if rec is None:
            continue
        rec = dict(rec)
        rec["candidate_hold_bars"] = int(hold)
        candidates.append(rec)
    if not candidates:
        raise ValueError("no valid hold candidate records")
    best = max(candidates, key=_record_score)
    return int(best["candidate_hold_bars"]), best, candidates


def _candidate_table(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rec in candidates:
        rows.append(
            {
                "hold_bars": int(rec["candidate_hold_bars"]),
                "gate": str(rec["trade_gate_label"]),
                "side": str(rec["trade_side_label"]),
                "quality": str(rec["quality_label"]),
                "risk": str(rec["risk_label"]),
                "net_return_pct": round(float(rec["best_net_return"]) * 100.0, 3),
                "mae_pct": round(float(rec["best_mae"]) * 100.0, 3),
                "utility_pct": round(float(rec["best_utility"]) * 100.0, 3),
            }
        )
    return rows


def _enrich_summary(summary: dict[str, Any], *, best_hold: int, best_record: dict[str, Any], candidates: list[dict[str, Any]], feature_profile: str) -> dict[str, Any]:
    out = dict(summary)
    out["preferred_step_bars"] = int(best_hold) if str(best_record["trade_gate_label"]) == "TRADE" else 0
    out["preferred_step_label"] = "NO_TRADE" if out["preferred_step_bars"] == 0 else f"HOLD_{int(best_hold)}"
    out["candidate_steps_bars"] = [int(c["candidate_hold_bars"]) for c in candidates]
    out["step_reason"] = {
        "selected_gate": str(best_record["trade_gate_label"]),
        "selected_side": str(best_record["trade_side_label"]) if str(best_record["trade_gate_label"]) == "TRADE" else "NONE",
        "selected_quality": str(best_record["quality_label"]),
        "selected_risk": str(best_record["risk_label"]),
    }
    if str(feature_profile) == "rich_v1":
        # Training target diagnostic only: lets us audit which horizon label was selected.
        # This must not be exposed as live input unless produced by the analyzer itself.
        out["candidate_step_outcomes_for_training_audit"] = _candidate_table(candidates)
    return out


def build_step_trader_target(best_record: dict[str, Any], best_hold: int) -> dict[str, Any]:
    gate = str(best_record["trade_gate_label"])
    return {
        "gate": gate,
        "side": str(best_record["trade_side_label"]) if gate == "TRADE" else "NONE",
        "hold_bars": int(best_hold) if gate == "TRADE" else 0,
    }


def _iter_positions(market: pd.DataFrame, cfg: StepAnalyzerConfig, start_date: str | None, end_date: str | None):
    start_ts = pd.to_datetime(start_date) if start_date else None
    end_ts = pd.to_datetime(end_date) if end_date else None
    max_hold = max(int(x) for x in cfg.hold_candidates)
    last_signal_pos = len(market) - max(1, int(cfg.entry_delay_bars)) - max_hold - 1
    for pos in range(max(0, int(cfg.window_size) - 1), max(0, last_signal_pos) + 1, max(1, int(cfg.stride_bars))):
        ts = pd.to_datetime(market.iloc[pos]["date"])
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        yield pos


def build_step_records(
    market: pd.DataFrame,
    cfg: StepAnalyzerConfig,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    max_records: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    feature_frame = build_market_feature_frame(market, window_size=cfg.window_size)
    analyzer_rows: list[dict[str, Any]] = []
    trader_rows: list[dict[str, Any]] = []
    best_path_rows: list[dict[str, Any]] = []
    for pos in _iter_positions(market, cfg, start_date, end_date):
        best_hold, best_rec, candidates = choose_step_record(market, pos, cfg)
        summary = build_analyzer_summary(market, pos, window_size=cfg.window_size, feature_frame=feature_frame)
        step_summary = _enrich_summary(summary, best_hold=best_hold, best_record=best_rec, candidates=candidates, feature_profile=cfg.feature_profile)
        summary_text = analyzer_summary_to_text(step_summary)
        analyzer_rows.append(
            {
                "task": "step_analyzer",
                "date": best_rec["date"],
                "signal_pos": int(pos),
                "prompt": build_analyzer_input(market, pos, window_size=cfg.window_size)
                + "\nAlso choose preferred_step_bars from the candidate hold steps, or 0 for NO_TRADE.",
                "target": summary_text,
                "preferred_step_bars": int(step_summary["preferred_step_bars"]),
                "leakage_guard": {
                    "target_step_uses_future_path": True,
                    "prompt_bars_end_at_signal_pos": int(pos),
                },
            }
        )
        trader_rows.append(
            {
                "task": "step_trader",
                "date": best_rec["date"],
                "signal_pos": int(pos),
                "prompt": build_trader_input(summary_text, hold_bars=best_hold, entry_delay_bars=cfg.entry_delay_bars)
                + "\nUse the analyzer preferred_step_bars when deciding hold_bars.",
                "target": json.dumps(build_step_trader_target(best_rec, best_hold), sort_keys=True, separators=(",", ":")),
                "trade_gate_label": best_rec["trade_gate_label"],
                "trade_side_label": best_rec["trade_side_label"],
                "preferred_step_bars": int(best_hold) if best_rec["trade_gate_label"] == "TRADE" else 0,
                "leakage_guard": {"prompt_uses_analyzer_summary_only": True, "label_uses_future_path": True},
            }
        )
        best_path_rows.append(best_rec)
        if max_records is not None and len(trader_rows) >= int(max_records):
            break
    return analyzer_rows, trader_rows, best_path_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build step-aware analyzer/trader SFT data")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--analyzer-output", default="data/text_step_analyzer_sft.jsonl")
    p.add_argument("--trader-output", default="data/text_step_trader_sft.jsonl")
    p.add_argument("--summary-output", default="")
    p.add_argument("--start-date", default="")
    p.add_argument("--end-date", default="")
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--window-size", type=int, default=96)
    p.add_argument("--hold-candidates", default="48,96,144,288")
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--mae-penalty", type=float, default=1.0)
    p.add_argument("--mfe-bonus", type=float, default=0.0)
    p.add_argument("--min-net-return", type=float, default=0.001)
    p.add_argument("--min-utility", type=float, default=0.0)
    p.add_argument("--max-mae", type=float, default=0.015)
    p.add_argument("--stride-bars", type=int, default=12)
    p.add_argument("--feature-profile", choices=["compact_v1", "rich_v1"], default="rich_v1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hold_candidates = parse_hold_candidates(args.hold_candidates)
    cfg = StepAnalyzerConfig(
        window_size=args.window_size,
        hold_bars=max(hold_candidates),
        entry_delay_bars=args.entry_delay_bars,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        leverage=args.leverage,
        mae_penalty=args.mae_penalty,
        mfe_bonus=args.mfe_bonus,
        min_net_return=args.min_net_return,
        min_utility=args.min_utility,
        max_mae=args.max_mae,
        stride_bars=args.stride_bars,
        hold_candidates=hold_candidates,
        feature_profile=args.feature_profile,
    )
    market = load_market_frame(args.market_csv)
    analyzer_rows, trader_rows, best_path_rows = build_step_records(
        market,
        cfg,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        max_records=args.max_records or None,
    )
    write_jsonl(args.analyzer_output, analyzer_rows)
    write_jsonl(args.trader_output, trader_rows)
    step_counts: dict[str, int] = {}
    for row in trader_rows:
        step = str(row["preferred_step_bars"])
        step_counts[step] = step_counts.get(step, 0) + 1
    summary = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "market_csv": str(Path(args.market_csv).resolve()),
        "outputs": {"analyzer": args.analyzer_output, "trader": args.trader_output},
        "config": asdict(cfg),
        "records": {"analyzer": len(analyzer_rows), "trader": len(trader_rows)},
        "preferred_step_counts": dict(sorted(step_counts.items(), key=lambda kv: int(kv[0]))),
        "path_label_summary": summarize_records(best_path_rows),
        "external_feature_availability": {
            "kimchi_premium_columns": [c for c in market.columns if any(x in c.lower() for x in ("kim", "premium", "kimp", "krw"))],
            "dollar_index_columns": [c for c in market.columns if any(x in c.lower() for x in ("dxy", "dollar"))],
        },
        "leakage_guard": {
            "analyzer_step_target_uses_future_path": True,
            "trader_label_uses_future_path": True,
            "prompts_use_bars_at_or_before_signal_pos": True,
            "live_mode_requires_analyzer_to_predict_step_without_candidate_outcome_audit": True,
        },
    }
    out = args.summary_output or f"{args.trader_output}.summary.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
