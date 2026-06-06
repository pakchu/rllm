"""Build direct economic value-scoring rows from counterfactual action candidates.

Unlike DPO/classification, this dataset asks a model/regressor to score each
candidate action by strict future utility.  Prompts remain past-only; utility is
training/evaluation label only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.economic_preference_data import EconomicPreferenceConfig, _candidate_actions, _preference_prompt
from training.edge_decay_analyzer_data import write_jsonl
from training.multi_horizon_edge_report import parse_horizons
from training.decision_feature_learnability import load_jsonl
from training.strict_bar_backtest import load_market_bars


@dataclass(frozen=True)
class EconomicValueConfig:
    hold_bars_list: tuple[int, ...] = (36, 72, 144, 288, 432)
    entry_delay_bars: int = 1
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    leverage: float = 0.5
    mae_penalty: float = 1.0


def _action_json(action: dict[str, Any]) -> str:
    return json.dumps({"gate": action["gate"], "side": action["side"], "hold_bars": int(action.get("hold_bars", 0) or 0)}, sort_keys=True, separators=(",", ":"))


def build_value_rows(records: list[dict[str, Any]], market, cfg: EconomicValueConfig) -> list[dict[str, Any]]:
    pcfg = EconomicPreferenceConfig(**asdict(cfg), min_utility_gap=0.0, max_pairs_per_row=99)
    out: list[dict[str, Any]] = []
    for row in records:
        candidates = _candidate_actions(row, market, pcfg)
        if not candidates:
            continue
        best_utility = max(float(c["utility"]) for c in candidates)
        for cand in candidates:
            out.append(
                {
                    "task": "economic_value_scoring",
                    "date": row.get("date"),
                    "signal_pos": row.get("signal_pos"),
                    "prompt": _preference_prompt(str(row.get("prompt", ""))),
                    "action": _action_json(cand),
                    "utility": float(cand["utility"]),
                    "net_return": float(cand["net_return"]),
                    "mae": float(cand["mae"]),
                    "mfe": float(cand["mfe"]),
                    "is_best_action": abs(float(cand["utility"]) - best_utility) < 1e-12,
                    "route": cand.get("route"),
                    "leakage_guard": {
                        "prompt_uses_future_path": False,
                        "utility_uses_future_ohlc_for_training_or_eval_label_only": True,
                    },
                }
            )
    return out


def summarize_value_rows(rows: list[dict[str, Any]], cfg: EconomicValueConfig) -> dict[str, Any]:
    actions = Counter(str(r["action"]) for r in rows)
    utilities = [float(r["utility"]) for r in rows]
    n_signals = len({(r.get("date"), r.get("signal_pos")) for r in rows})
    return {
        "rows": len(rows),
        "signals": n_signals,
        "period": {"start": rows[0].get("date") if rows else None, "end": rows[-1].get("date") if rows else None},
        "action_counts": dict(sorted(actions.items())),
        "utility_pct": {
            "min": min(utilities) * 100.0 if utilities else 0.0,
            "max": max(utilities) * 100.0 if utilities else 0.0,
            "mean": sum(utilities) / max(1, len(utilities)) * 100.0,
        },
        "config": asdict(cfg),
        "leakage_guard": {"prompts_are_past_only": True, "utility_labels_use_future_ohlc": True, "not_a_backtest_result": True},
    }


def build_economic_value_jsonl(
    *,
    records: str,
    market_csv: str,
    output: str,
    summary_output: str = "",
    hold_bars_list: str = "36,72,144,288,432",
    entry_delay_bars: int = 1,
    leverage: float = 0.5,
    fee_rate: float = 0.0004,
    slippage_rate: float = 0.0001,
    mae_penalty: float = 1.0,
    max_records: int = 0,
) -> dict[str, Any]:
    cfg = EconomicValueConfig(
        hold_bars_list=parse_horizons(hold_bars_list),
        entry_delay_bars=int(entry_delay_bars),
        leverage=float(leverage),
        fee_rate=float(fee_rate),
        slippage_rate=float(slippage_rate),
        mae_penalty=float(mae_penalty),
    )
    recs = load_jsonl(records)
    if max_records:
        recs = recs[: int(max_records)]
    market = load_market_bars(market_csv)
    rows = build_value_rows(recs, market, cfg)
    write_jsonl(output, rows)
    summary = {"as_of": datetime.now(timezone.utc).isoformat(), "inputs": {"records": records, "market_csv": market_csv}, "outputs": {"value_jsonl": output}, "summary": summarize_value_rows(rows, cfg)}
    if summary_output:
        Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build economic value scoring rows")
    p.add_argument("--records", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--hold-bars-list", default="36,72,144,288,432")
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--mae-penalty", type=float, default=1.0)
    p.add_argument("--max-records", type=int, default=0)
    return p.parse_args()


def main() -> None:
    print(json.dumps(build_economic_value_jsonl(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
