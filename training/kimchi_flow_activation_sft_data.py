"""Build Gemma SFT rows for Kimchi-flow regime activation decisions.

Targets are aligned to the discovered edge: activate/abstain and side context
around the audited Kimchi-change/trades-ratio rule, not generic direction labels.
Prompts are past-only edge-state hybrid prompts.
"""
from __future__ import annotations

import argparse, json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from training.compare_edge_state_prompt_modes import _load_market, _sample_rows
from training.vlm_trading_data import build_vlm_training_samples


@dataclass(frozen=True)
class KimchiFlowActivationConfig:
    input_csv: str
    fixed_report: str
    output: str
    summary_output: str = ""
    start: str = "2025-01-01"
    end: str = "2025-12-01 23:59:59"
    window_size: int = 144
    max_samples: int = 0
    sample_mode: str = "uniform"
    sample_seed: int = 42
    min_good_ret_pct: float = 0.10
    min_bad_ret_pct: float = -0.05
    include_no_trade_context: bool = True
    prompt_feature_mode: str = "edge_state_v5"
    trade_split: str = "eval"


def _load_trades(path: str, split: str = "eval") -> list[dict[str, Any]]:
    fixed = json.load(open(path))
    split_key = str(split).strip().lower()
    if split_key == "all":
        trades = []
        for key in ("train", "test", "eval"):
            trades.extend(fixed.get("monthly", {}).get(key, {}).get("trades", []))
    else:
        trades = fixed.get("monthly", {}).get(split_key, {}).get("trades", [])
    if not trades:
        raise ValueError(
            f"fixed_report lacks monthly.{split_key}.trades; "
            "rerun fixed-rule backtest with full trade export"
        )
    return trades


def _decision_target(trade: dict[str, Any], *, good: float, bad: float) -> dict[str, str]:
    side = str(trade.get("side", "NONE")).upper()
    ret = float(trade.get("ret_pct", 0.0))
    if ret >= good:
        decision = "ACTIVATE"
        quality = "GOOD"
    elif ret <= bad:
        decision = "ABSTAIN"
        quality = "BAD"
    else:
        decision = "ABSTAIN"
        quality = "MARGINAL"
    return {
        "regime": "KIMCHI_FLOW",
        "decision": decision,
        "side": side if decision == "ACTIVATE" else "NONE",
        "quality": quality,
        "confidence": "HIGH" if quality in {"GOOD", "BAD"} else "LOW",
    }


def _activation_prompt(base_prompt: str) -> str:
    context = str(base_prompt)
    if "Output format:" in context:
        context = context.split("Output format:", 1)[0].strip()
    return "\n".join([
        "You are a Gemma regime-activation policy for BTCUSDT futures.",
        "Use only the past-only market context below.",
        "Decide whether the audited Kimchi-flow opportunity should be activated now.",
        "Return compact JSON with keys: regime, decision, side, quality, confidence.",
        "Allowed decision: ACTIVATE, ABSTAIN.",
        "Allowed side: LONG, SHORT, NONE. If decision is ABSTAIN, side must be NONE.",
        "Allowed quality: GOOD, MARGINAL, BAD.",
        "Allowed confidence: LOW, MID, HIGH.",
        "Do not output exchange orders, size, leverage, or hold bars.",
        "",
        "Past-only context:",
        context,
    ])


def build_rows(cfg: KimchiFlowActivationConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trades = _load_trades(cfg.fixed_report, cfg.trade_split)
    market = _load_market(cfg.input_csv, cfg.start, cfg.end)
    sample_dates = [str(pd.to_datetime(t["signal_date"])) for t in trades]
    # Generate prompts exactly at fixed-rule signal dates.
    samples = build_vlm_training_samples(
        market,
        window_size=cfg.window_size,
        max_samples=None,
        sample_mode="sequential",
        sample_seed=cfg.sample_seed,
        target_horizon=1,
        label_mode="next_return",
        prompt_feature_mode=cfg.prompt_feature_mode,
        action_schema="buy_hold_sell",
        prompt_style="hybrid",
        modality="text_only",
        sample_dates=sample_dates,
        path_entry_delay_bars=1,
        utility_fee_rate=0.0004,
        utility_slippage_rate=0.0001,
        utility_leverage=0.5,
        path_mae_penalty=1.0,
        path_min_net_return=-1.0,
        path_max_mae=1.0,
    )
    by_date = {str(pd.to_datetime(s.date)): s for s in samples}
    rows=[]
    for trade in trades:
        date = str(pd.to_datetime(trade["signal_date"]))
        sample = by_date.get(date)
        if sample is None:
            continue
        target = _decision_target(trade, good=cfg.min_good_ret_pct, bad=cfg.min_bad_ret_pct)
        rows.append({
            "task": "kimchi_flow_activation_sft",
            "date": date,
            "prompt": _activation_prompt(sample.prompt),
            "target": json.dumps(target, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            "trade_ret_pct": float(trade.get("ret_pct", 0.0)),
            "source_trade": trade,
            "leakage_guard": {
                "prompt_uses_future_path": False,
                "target_uses_realized_trade_return_for_training_only": True,
                "activation_target_aligned_to_fixed_rule_trade_quality": True,
            },
        })
    if cfg.max_samples and cfg.max_samples < len(rows):
        if cfg.sample_mode == "sequential":
            rows = rows[: cfg.max_samples]
        else:
            rows = rows[:: max(1, len(rows)//cfg.max_samples)][: cfg.max_samples]
    summary = summarize(rows, cfg)
    return rows, summary


def summarize(rows: list[dict[str, Any]], cfg: KimchiFlowActivationConfig) -> dict[str, Any]:
    from collections import Counter
    counts = {"decision": Counter(), "side": Counter(), "quality": Counter(), "confidence": Counter()}
    rets=[]; prompt_lens=[]
    for row in rows:
        obj=json.loads(row["target"])
        for k in counts: counts[k][obj.get(k,"MISSING")]+=1
        rets.append(float(row.get("trade_ret_pct",0.0))); prompt_lens.append(len(str(row.get("prompt",""))))
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "period": {"start": rows[0]["date"] if rows else None, "end": rows[-1]["date"] if rows else None},
        "target_counts": {k: dict(v) for k,v in counts.items()},
        "trade_ret_pct": {"min": min(rets) if rets else 0, "max": max(rets) if rets else 0, "mean": sum(rets)/max(1,len(rets))},
        "prompt_chars": {"min": min(prompt_lens) if prompt_lens else 0, "max": max(prompt_lens) if prompt_lens else 0, "mean": sum(prompt_lens)/max(1,len(prompt_lens))},
        "config": asdict(cfg),
    }


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows)+("\n" if rows else ""))


def run(cfg: KimchiFlowActivationConfig) -> dict[str, Any]:
    rows, summary = build_rows(cfg)
    write_jsonl(cfg.output, rows)
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--input-csv', required=True); p.add_argument('--fixed-report', required=True); p.add_argument('--output', required=True); p.add_argument('--summary-output', default='')
    p.add_argument('--start', default='2025-01-01'); p.add_argument('--end', default='2025-12-01 23:59:59'); p.add_argument('--window-size', type=int, default=144)
    p.add_argument('--max-samples', type=int, default=0); p.add_argument('--sample-mode', default='uniform'); p.add_argument('--sample-seed', type=int, default=42)
    p.add_argument('--min-good-ret-pct', type=float, default=0.10); p.add_argument('--min-bad-ret-pct', type=float, default=-0.05)
    p.add_argument('--prompt-feature-mode', default='edge_state_v5', choices=['edge_state_v5', 'edge_state_v6', 'edge_state_v7'])
    p.add_argument('--trade-split', default='eval', choices=['train', 'test', 'eval', 'all'])
    return p.parse_args()


def main():
    print(json.dumps(run(KimchiFlowActivationConfig(**vars(parse_args()))), indent=2, ensure_ascii=False))

if __name__=='__main__': main()
