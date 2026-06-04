"""Build counterfactual economic preference pairs for LLM trader/RL stages.

For each past-only prompt timestamp, compare executable candidates:
SKIP, TREND at several hold horizons, and FADE at several hold horizons.  The
chosen/rejected responses are ranked by strict path utility computed from future
OHLC.  Prompts remain past-only; preferences are training-only labels.
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
from training.path_outcome_dataset import PathOutcomeConfig, compute_trade_path_outcome
from training.strict_bar_backtest import load_market_bars


@dataclass(frozen=True)
class EconomicPreferenceConfig:
    hold_bars_list: tuple[int, ...] = (36, 72, 144, 288, 432)
    entry_delay_bars: int = 1
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    leverage: float = 0.5
    mae_penalty: float = 1.0
    min_utility_gap: float = 0.001
    max_pairs_per_row: int = 3


def _opposite(side: str) -> str:
    if side == "LONG":
        return "SHORT"
    if side == "SHORT":
        return "LONG"
    return "NONE"


def _trend_side(row: dict[str, Any]) -> str:
    target = parse_path_shape_json(str(row.get("target", "{}")))
    side = str(target.get("trend_side", "NONE"))
    return side if side in {"LONG", "SHORT"} else "NONE"


def _action_response(action: dict[str, Any]) -> str:
    payload = {
        "gate": str(action["gate"]),
        "side": str(action["side"]),
        "hold_bars": int(action.get("hold_bars", 0) or 0),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _candidate_actions(row: dict[str, Any], market, cfg: EconomicPreferenceConfig) -> list[dict[str, Any]]:
    trend_side = _trend_side(row)
    candidates = [
        {
            "route": "SKIP",
            "gate": "NO_TRADE",
            "side": "NONE",
            "hold_bars": 0,
            "utility": 0.0,
            "net_return": 0.0,
            "mae": 0.0,
            "mfe": 0.0,
        }
    ]
    if trend_side == "NONE":
        return candidates
    routes = (("TREND", trend_side), ("FADE", _opposite(trend_side)))
    for route, side in routes:
        if side not in {"LONG", "SHORT"}:
            continue
        for hold in cfg.hold_bars_list:
            pcfg = PathOutcomeConfig(
                hold_bars=int(hold),
                entry_delay_bars=int(cfg.entry_delay_bars),
                fee_rate=float(cfg.fee_rate),
                slippage_rate=float(cfg.slippage_rate),
                leverage=float(cfg.leverage),
                mae_penalty=float(cfg.mae_penalty),
            )
            out = compute_trade_path_outcome(market, int(row.get("signal_pos", 0)), side, pcfg)  # type: ignore[arg-type]
            if out is None:
                continue
            candidates.append(
                {
                    "route": route,
                    "gate": "TRADE",
                    "side": side,
                    "hold_bars": int(hold),
                    "utility": float(out.utility),
                    "net_return": float(out.net_return),
                    "mae": float(out.mae),
                    "mfe": float(out.mfe),
                }
            )
    return sorted(candidates, key=lambda x: (float(x["utility"]), float(x["net_return"])), reverse=True)


def _preference_prompt(source_prompt: str) -> str:
    if "Past-only analyzer summary:" in source_prompt:
        past_summary = source_prompt.split("Past-only analyzer summary:", 1)[1].strip()
    else:
        past_summary = str(source_prompt)[-3000:]
    return "\n".join(
        [
            "You are a counterfactual economic trader for BTCUSDT futures.",
            "Use only the past-only analyzer summary below.",
            "Choose whether to trade with the current trend, fade it, or skip, including a hold_bars value.",
            "Return exactly one JSON object with keys gate, side, hold_bars.",
            "Allowed gate: TRADE, NO_TRADE. If NO_TRADE, side must be NONE and hold_bars 0.",
            "",
            f"Past-only analyzer summary: {past_summary}",
        ]
    )


def build_economic_preference_pairs(rows: list[dict[str, Any]], market, cfg: EconomicPreferenceConfig) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row in rows:
        candidates = _candidate_actions(row, market, cfg)
        if len(candidates) < 2:
            continue
        chosen = candidates[0]
        pair_count = 0
        for rejected in candidates[1:]:
            gap = float(chosen["utility"]) - float(rejected["utility"])
            if gap < float(cfg.min_utility_gap):
                continue
            pairs.append(
                {
                    "task": "economic_counterfactual_preference",
                    "date": row.get("date"),
                    "signal_pos": row.get("signal_pos"),
                    "prompt": _preference_prompt(str(row.get("prompt", ""))),
                    "chosen": _action_response(chosen),
                    "rejected": _action_response(rejected),
                    "chosen_action": {k: chosen[k] for k in ("route", "gate", "side", "hold_bars", "utility", "net_return", "mae")},
                    "rejected_action": {k: rejected[k] for k in ("route", "gate", "side", "hold_bars", "utility", "net_return", "mae")},
                    "utility_gap": gap,
                    "leakage_guard": {
                        "prompt_uses_future_path": False,
                        "chosen_rejected_use_future_ohlc_paths_for_training_only": True,
                        "preference_is_counterfactual_economic_not_path_classification": True,
                    },
                }
            )
            pair_count += 1
            if pair_count >= int(cfg.max_pairs_per_row):
                break
    return pairs


def summarize_pairs(pairs: list[dict[str, Any]], cfg: EconomicPreferenceConfig) -> dict[str, Any]:
    chosen_counts: Counter[str] = Counter()
    rejected_counts: Counter[str] = Counter()
    gaps = []
    prompt_lens = []
    for row in pairs:
        c = row["chosen_action"]
        r = row["rejected_action"]
        chosen_counts[f"route={c['route']},gate={c['gate']},side={c['side']},hold={c['hold_bars']}"] += 1
        rejected_counts[f"route={r['route']},gate={r['gate']},side={r['side']},hold={r['hold_bars']}"] += 1
        gaps.append(float(row.get("utility_gap", 0.0)))
        prompt_lens.append(len(str(row.get("prompt", ""))))
    return {
        "pairs": len(pairs),
        "period": {"start": pairs[0].get("date") if pairs else None, "end": pairs[-1].get("date") if pairs else None},
        "chosen_counts": dict(sorted(chosen_counts.items())),
        "rejected_counts": dict(sorted(rejected_counts.items())),
        "mean_utility_gap_pct": (sum(gaps) / max(1, len(gaps))) * 100.0,
        "prompt_chars": {"min": min(prompt_lens) if prompt_lens else 0, "max": max(prompt_lens) if prompt_lens else 0, "mean": sum(prompt_lens) / max(1, len(prompt_lens))},
        "config": asdict(cfg),
        "leakage_guard": {
            "prompts_are_past_only": True,
            "preferences_use_future_ohlc_paths": True,
            "not_a_backtest_result": True,
        },
    }


def build_economic_preference_jsonl(
    *,
    records: str,
    market_csv: str,
    output: str,
    summary_output: str = "",
    hold_bars_list: str = "36,72,144,288,432",
    min_utility_gap: float = 0.001,
    max_pairs_per_row: int = 3,
    entry_delay_bars: int = 1,
    leverage: float = 0.5,
    fee_rate: float = 0.0004,
    slippage_rate: float = 0.0001,
    mae_penalty: float = 1.0,
    max_records: int = 0,
) -> dict[str, Any]:
    cfg = EconomicPreferenceConfig(
        hold_bars_list=parse_horizons(hold_bars_list),
        min_utility_gap=float(min_utility_gap),
        max_pairs_per_row=int(max_pairs_per_row),
        entry_delay_bars=int(entry_delay_bars),
        leverage=float(leverage),
        fee_rate=float(fee_rate),
        slippage_rate=float(slippage_rate),
        mae_penalty=float(mae_penalty),
    )
    rows = load_jsonl(records)
    if max_records:
        rows = rows[: int(max_records)]
    market = load_market_bars(market_csv)
    pairs = build_economic_preference_pairs(rows, market, cfg)
    write_jsonl(output, pairs)
    summary = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "inputs": {"records": records, "market_csv": market_csv},
        "outputs": {"preferences": output},
        "preferences": summarize_pairs(pairs, cfg),
    }
    if summary_output:
        Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build counterfactual economic preference pairs from path-shape records")
    p.add_argument("--records", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--hold-bars-list", default="36,72,144,288,432")
    p.add_argument("--min-utility-gap", type=float, default=0.001)
    p.add_argument("--max-pairs-per-row", type=int, default=3)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--mae-penalty", type=float, default=1.0)
    p.add_argument("--max-records", type=int, default=0)
    return p.parse_args()


def main() -> None:
    print(json.dumps(build_economic_preference_jsonl(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
