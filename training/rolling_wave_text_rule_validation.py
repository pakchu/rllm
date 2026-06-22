"""Rolling validation for the fold-consistent wave text-rule selector.

Each split independently selects top-k wave policies on pre-eval selection folds,
builds bucketed text-state rows, fits a token-rule threshold on selection rows,
and replays the frozen selector on the held-out eval period.

The leverage grid is reported per split for diagnosis. A production leverage must
be fixed from pre-eval evidence; do not treat per-split ``best_by_rule`` as a
leak-free production choice.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.backtest_text_state_rule import run as run_text_rule
from training.build_wave_llm_state_dataset_multi import MultiCfg, run as build_multi
from training.sweep_wave_fold_consistency import (
    FoldConsistencyConfig,
    run_sweep as run_consistency,
)


@dataclass(frozen=True)
class RollingCfg:
    wave_root: str
    market_5m_csv: str
    output: str
    work_dir: str = "results/rolling_wave_text_rule"
    splits_json: str = ""
    top_k: int = 5
    quantile: float = 0.5
    leverages: str = "1.0,2.0,2.5,3.0"


def _default_splits() -> list[dict[str, str]]:
    return [
        {
            "name": "sel2021_2022_eval2023",
            "selection_folds": ",".join(
                [
                    "2021-01-01|2021-06-30 23:59:59",
                    "2021-07-01|2021-12-31 23:59:59",
                    "2022-01-01|2022-06-30 23:59:59",
                    "2022-07-01|2022-12-31 23:59:59",
                ]
            ),
            "eval_start": "2023-01-01",
            "eval_end": "2023-12-31 23:59:59",
        },
        {
            "name": "sel2021_2023_eval2024h1",
            "selection_folds": ",".join(
                [
                    "2021-01-01|2021-06-30 23:59:59",
                    "2021-07-01|2021-12-31 23:59:59",
                    "2022-01-01|2022-06-30 23:59:59",
                    "2022-07-01|2022-12-31 23:59:59",
                    "2023-01-01|2023-06-30 23:59:59",
                    "2023-07-01|2023-12-31 23:59:59",
                ]
            ),
            "eval_start": "2024-01-01",
            "eval_end": "2024-06-30 23:59:59",
        },
        {
            "name": "sel2021_2024h1_eval2024h2_2026",
            "selection_folds": ",".join(
                [
                    "2021-01-01|2021-06-30 23:59:59",
                    "2021-07-01|2021-12-31 23:59:59",
                    "2022-01-01|2022-06-30 23:59:59",
                    "2022-07-01|2022-12-31 23:59:59",
                    "2023-01-01|2023-06-30 23:59:59",
                    "2023-07-01|2023-12-31 23:59:59",
                    "2024-01-01|2024-06-30 23:59:59",
                ]
            ),
            "eval_start": "2024-07-01",
            "eval_end": "2026-06-01 00:00:00",
        },
    ]


def _parse_leverages(raw: str) -> list[float]:
    return [float(x) for x in str(raw).split(",") if x.strip()]


def _score(sim: dict[str, Any]) -> float:
    cagr = float(sim.get("cagr_pct", 0))
    mdd = float(sim.get("strict_mdd_pct", 999))
    ratio = float(sim.get("cagr_to_strict_mdd", 0))
    trades = int(sim.get("trade_entries", 0))
    if mdd > 15 or trades < 20 or cagr <= 0:
        return -1000 + cagr - mdd + trades / 1000
    return ratio * 100 + cagr + trades / 10


def run(cfg: RollingCfg) -> dict[str, Any]:
    splits = json.loads(cfg.splits_json) if cfg.splits_json else _default_splits()
    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    results = []

    for split in splits:
        split_dir = work / split["name"]
        split_dir.mkdir(parents=True, exist_ok=True)
        fold_consistency_path = split_dir / "fold_consistency.json"
        consistency = run_consistency(
            FoldConsistencyConfig(
                wave_root=cfg.wave_root,
                market_5m_csv=cfg.market_5m_csv,
                output=str(fold_consistency_path),
                selection_folds=split["selection_folds"],
                eval_start=split["eval_start"],
                eval_end=split["eval_end"],
                top_k=int(cfg.top_k),
                vote_k=2,
            )
        )

        train_state = split_dir / "state_train.jsonl"
        eval_state = split_dir / "state_eval.jsonl"
        state_summary = split_dir / "state_summary.json"
        multi_summary = build_multi(
            MultiCfg(
                wave_root=cfg.wave_root,
                market_5m_csv=cfg.market_5m_csv,
                fold_consistency_report=str(fold_consistency_path),
                train_output=str(train_state),
                eval_output=str(eval_state),
                summary_output=str(state_summary),
                top_k=int(cfg.top_k),
                selection_folds=split["selection_folds"],
                eval_start=split["eval_start"],
                eval_end=split["eval_end"],
            )
        )

        leverage_results = []
        q_label = str(cfg.quantile).replace(".", "p")
        for leverage in _parse_leverages(cfg.leverages):
            lev_label = str(leverage).replace(".", "p")
            out = split_dir / f"text_rule_q{q_label}_lev{lev_label}.json"
            predictions = split_dir / f"text_rule_q{q_label}_lev{lev_label}.predictions.jsonl"
            backtest = run_text_rule(
                train_jsonl=str(train_state),
                eval_jsonl=str(eval_state),
                market_csv=cfg.market_5m_csv,
                output=str(out),
                predictions_output=str(predictions),
                quantile=float(cfg.quantile),
                leverage=float(leverage),
                aggregate_duplicates=True,
            )
            leverage_results.append(
                {
                    "leverage": leverage,
                    "sim": backtest["sim"],
                    "trade_stats": backtest["trade_stats"],
                    "score": _score(backtest["sim"]),
                    "path": str(out),
                }
            )

        leverage_results.sort(key=lambda row: row["score"], reverse=True)
        results.append(
            {
                "split": split,
                "fold_consistency_top": consistency["top10"][:5],
                "dataset": {"train": multi_summary["train"], "eval": multi_summary["eval"]},
                "leverage_results": leverage_results,
                "best_by_rule": leverage_results[0] if leverage_results else None,
            }
        )

    summary = {
        "config": asdict(cfg),
        "splits": results,
        "leakage_guard": {
            "each_split_selects_policy_and_token_rule_before_eval": True,
            "leverage_grid_reported_per_split_not_global_production_choice": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave-root", default="/home/pakchu/workspace/wave_trading")
    parser.add_argument("--market-5m-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", default="results/rolling_wave_text_rule")
    parser.add_argument("--splits-json", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--leverages", default="1.0,2.0,2.5,3.0")
    return parser.parse_args()


def main() -> None:
    report = run(RollingCfg(**vars(parse_args())))
    print(
        json.dumps(
            {
                "splits": [
                    {
                        "name": row["split"]["name"],
                        "dataset": row["dataset"],
                        "best": row["best_by_rule"],
                    }
                    for row in report["splits"]
                ]
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
