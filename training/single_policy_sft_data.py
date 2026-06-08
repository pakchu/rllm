"""Build single-LLM semantic policy SFT rows from economic action candidates.

This replaces the two-stage analyzer/trader imitation target with one compact
policy object.  The target still uses future OHLC utility for training labels,
but the output avoids raw hold-bar prediction by mapping holds into an
exit_profile abstraction.  Prompts remain past-only.
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
from training.economic_preference_data import EconomicPreferenceConfig, _candidate_actions
from training.eval_multi_horizon_path_shape_analyzer import parse_path_shape_json
from training.multi_horizon_edge_report import parse_horizons
from training.strict_bar_backtest import load_market_bars

ACTIONS = {"NO_TRADE", "LONG", "SHORT"}
EXIT_PROFILES = {"AVOID", "FAST", "NORMAL", "TRAIL"}


@dataclass(frozen=True)
class SinglePolicyConfig:
    hold_bars_list: tuple[int, ...] = (36, 72, 144, 288, 432)
    entry_delay_bars: int = 1
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    leverage: float = 0.5
    mae_penalty: float = 1.0
    no_trade_utility: float = 0.004
    min_trade_net_return: float = -1.0
    max_trade_mae: float = 1.0


def exit_profile_for_hold(hold_bars: int) -> str:
    hold = int(hold_bars or 0)
    if hold <= 0:
        return "AVOID"
    if hold <= 72:
        return "FAST"
    if hold <= 288:
        return "NORMAL"
    return "TRAIL"


def hold_bars_for_exit_profile(exit_profile: str) -> int:
    profile = str(exit_profile).upper()
    if profile == "FAST":
        return 72
    if profile == "NORMAL":
        return 288
    if profile == "TRAIL":
        return 432
    return 0


def _regime_from_target(target: dict[str, Any]) -> str:
    trend_side = str(target.get("trend_side", "NONE"))
    stability = str(target.get("direction_stability", ""))
    reversal = str(target.get("reversal_pressure", ""))
    if trend_side == "NONE":
        return "RANGE"
    if "CONFLICT" in stability:
        return "CHOP"
    if reversal == "HIGH":
        return "REVERSAL_RISK"
    if trend_side == "LONG":
        return "TREND_UP"
    if trend_side == "SHORT":
        return "TREND_DOWN"
    return "RANGE"


def _risk_from_action(action: dict[str, Any], target: dict[str, Any]) -> str:
    if str(action.get("gate")) != "TRADE":
        return "LOW"
    mae = float(action.get("mae", 0.0) or 0.0)
    risk_profile = str(target.get("risk_profile", ""))
    reversal = str(target.get("reversal_pressure", ""))
    if mae >= 0.015 or risk_profile == "HIGH_PATH_RISK" or reversal == "HIGH":
        return "HIGH"
    if mae >= 0.0075 or risk_profile == "MIXED_PATH_RISK":
        return "MID"
    return "LOW"


def _edge_quality(action: dict[str, Any]) -> str:
    if str(action.get("gate")) != "TRADE":
        return "NONE"
    rank_utility = float(action.get("rank_utility", action.get("utility", 0.0)) or 0.0)
    if rank_utility >= 0.015:
        return "STRONG"
    if rank_utility >= 0.008:
        return "MODERATE"
    return "WEAK"


def _confidence(edge_quality: str, risk: str) -> str:
    if edge_quality == "STRONG" and risk != "HIGH":
        return "HIGH"
    if edge_quality in {"MODERATE", "STRONG"} and risk in {"LOW", "MID"}:
        return "MID"
    return "LOW"


def single_policy_target(row: dict[str, Any], market, cfg: SinglePolicyConfig) -> dict[str, Any]:
    pref_cfg = EconomicPreferenceConfig(
        hold_bars_list=tuple(cfg.hold_bars_list),
        entry_delay_bars=int(cfg.entry_delay_bars),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
        leverage=float(cfg.leverage),
        mae_penalty=float(cfg.mae_penalty),
        no_trade_utility=float(cfg.no_trade_utility),
        min_trade_net_return=float(cfg.min_trade_net_return),
        max_trade_mae=float(cfg.max_trade_mae),
    )
    candidates = _candidate_actions(row, market, pref_cfg)
    chosen = candidates[0]
    path_target = parse_path_shape_json(str(row.get("target", "{}")))
    action = str(chosen.get("side", "NONE")) if str(chosen.get("gate")) == "TRADE" else "NO_TRADE"
    if action not in ACTIONS:
        action = "NO_TRADE"
    exit_profile = exit_profile_for_hold(int(chosen.get("hold_bars", 0) or 0)) if action != "NO_TRADE" else "AVOID"
    risk = _risk_from_action(chosen, path_target)
    edge_quality = _edge_quality(chosen)
    regime = _regime_from_target(path_target)
    return {
        "regime": regime,
        "edge_quality": edge_quality,
        "risk": risk,
        "action": action,
        "exit_profile": exit_profile,
        "confidence": _confidence(edge_quality, risk),
    }


def _policy_prompt(source_prompt: str) -> str:
    if "Past-only analyzer summary:" in source_prompt:
        past_summary = source_prompt.split("Past-only analyzer summary:", 1)[1].strip()
    else:
        past_summary = str(source_prompt)[-3000:]
    return "\n".join(
        [
            "You are a single LLM policy for BTCUSDT futures.",
            "Use only the past-only analyzer summary below.",
            "Return one compact JSON object with keys regime, edge_quality, risk, action, exit_profile, confidence.",
            "Allowed regime: TREND_UP, TREND_DOWN, RANGE, CHOP, REVERSAL_RISK.",
            "Allowed edge_quality: NONE, WEAK, MODERATE, STRONG.",
            "Allowed risk: LOW, MID, HIGH.",
            "Allowed action: NO_TRADE, LONG, SHORT.",
            "Allowed exit_profile: AVOID, FAST, NORMAL, TRAIL. If action is NO_TRADE, exit_profile must be AVOID.",
            "Do not output raw hold_bars or a final exchange order.",
            "",
            f"Past-only analyzer summary: {past_summary}",
        ]
    )


def build_single_policy_rows(rows: list[dict[str, Any]], market, cfg: SinglePolicyConfig) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        target = single_policy_target(row, market, cfg)
        out.append(
            {
                "task": "single_semantic_policy_sft",
                "date": row.get("date"),
                "signal_pos": row.get("signal_pos"),
                "prompt": _policy_prompt(str(row.get("prompt", ""))),
                "target": json.dumps(target, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                "leakage_guard": {
                    "prompt_uses_future_path": False,
                    "target_uses_future_ohlc_utility_for_training_only": True,
                    "single_llm_policy_not_analyzer_trader_chain": True,
                    "hold_bars_hidden_behind_exit_profile": True,
                },
            }
        )
    return out


def summarize_single_policy_rows(rows: list[dict[str, Any]], cfg: SinglePolicyConfig) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = {k: Counter() for k in ("regime", "edge_quality", "risk", "action", "exit_profile", "confidence")}
    combos: Counter[str] = Counter()
    prompts = []
    targets = []
    for row in rows:
        obj = json.loads(str(row["target"]))
        for k in counts:
            counts[k][str(obj.get(k))] += 1
        combos[f"action={obj.get('action')},exit={obj.get('exit_profile')},risk={obj.get('risk')}"] += 1
        prompts.append(len(str(row.get("prompt", ""))))
        targets.append(len(str(row.get("target", ""))))
    return {
        "rows": len(rows),
        "period": {"start": rows[0].get("date") if rows else None, "end": rows[-1].get("date") if rows else None},
        "field_counts": {k: dict(sorted(v.items())) for k, v in counts.items()},
        "top_combos": dict(combos.most_common(20)),
        "prompt_chars": {"min": min(prompts) if prompts else 0, "max": max(prompts) if prompts else 0, "mean": sum(prompts) / max(1, len(prompts))},
        "target_chars": {"min": min(targets) if targets else 0, "max": max(targets) if targets else 0, "mean": sum(targets) / max(1, len(targets))},
        "config": asdict(cfg),
        "leakage_guard": {
            "prompts_are_past_only": True,
            "targets_use_future_ohlc_utility": True,
            "not_a_backtest_result": True,
        },
    }


def build_single_policy_jsonl(
    *,
    records: str,
    market_csv: str,
    output: str,
    summary_output: str = "",
    hold_bars_list: str = "36,72,144,288,432",
    no_trade_utility: float = 0.004,
    min_trade_net_return: float = -1.0,
    max_trade_mae: float = 1.0,
    entry_delay_bars: int = 1,
    leverage: float = 0.5,
    fee_rate: float = 0.0004,
    slippage_rate: float = 0.0001,
    mae_penalty: float = 1.0,
    max_records: int = 0,
) -> dict[str, Any]:
    cfg = SinglePolicyConfig(
        hold_bars_list=parse_horizons(hold_bars_list),
        no_trade_utility=float(no_trade_utility),
        min_trade_net_return=float(min_trade_net_return),
        max_trade_mae=float(max_trade_mae),
        entry_delay_bars=int(entry_delay_bars),
        leverage=float(leverage),
        fee_rate=float(fee_rate),
        slippage_rate=float(slippage_rate),
        mae_penalty=float(mae_penalty),
    )
    source_rows = load_jsonl(records)
    if max_records:
        source_rows = source_rows[: int(max_records)]
    market = load_market_bars(market_csv)
    rows = build_single_policy_rows(source_rows, market, cfg)
    write_jsonl(output, rows)
    summary = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "inputs": {"records": records, "market_csv": market_csv},
        "outputs": {"sft_jsonl": output},
        "summary": summarize_single_policy_rows(rows, cfg),
    }
    if summary_output:
        Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build single semantic policy SFT rows")
    p.add_argument("--records", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--hold-bars-list", default="36,72,144,288,432")
    p.add_argument("--no-trade-utility", type=float, default=0.004)
    p.add_argument("--min-trade-net-return", type=float, default=-1.0)
    p.add_argument("--max-trade-mae", type=float, default=1.0)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--mae-penalty", type=float, default=1.0)
    p.add_argument("--max-records", type=int, default=0)
    return p.parse_args()


def main() -> None:
    print(json.dumps(build_single_policy_jsonl(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
