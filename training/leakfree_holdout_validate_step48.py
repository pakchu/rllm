"""Leakage-free holdout validation for step48 policy search.

Protocol:
1) Train cutoff assumption: 2025-06-30 (checkpoint name: 2023to2025h1).
2) Parameter selection window: 2025-07..2025-12.
3) Final holdout window: 2026-01..2026-02.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from training.search_significant_cagr_mdd_pool import (
    SimConfig,
    _pass_relaxed,
    _pass_strict,
    _simulate,
)


def _load_rows(files: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in files:
        payload = json.loads(Path(p).read_text())
        arr = payload.get("action_scores", [])
        if isinstance(arr, list):
            rows.extend(arr)
    rows.sort(key=lambda x: str(x.get("date", "")))
    if not rows:
        raise ValueError("No rows loaded.")
    return rows


def _month_from_path(p: str, file_suffix: str = "") -> str:
    name = Path(p).name
    # step48_nobias_YYYY-MM.json
    month = name.replace("step48_nobias_", "").replace(".json", "")
    if file_suffix and month.endswith(file_suffix):
        month = month[: -len(file_suffix)]
    return month


def _split_files(file_suffix: str = "") -> tuple[list[str], list[str]]:
    all_files = sorted(
        glob.glob("results/qrdqn_seq_eval_unbiased_balanced_2023to2025h1/step48_nobias_*.json")
    )
    sel_months = {"2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"}
    holdout_months = {"2026-01", "2026-02"}
    if file_suffix:
        all_files = [p for p in all_files if Path(p).name.replace(".json", "").endswith(file_suffix)]
    sel_files = [p for p in all_files if _month_from_path(p, file_suffix=file_suffix) in sel_months]
    holdout_files = [p for p in all_files if _month_from_path(p, file_suffix=file_suffix) in holdout_months]
    return sel_files, holdout_files


def run_holdout_validation(
    *,
    output: str,
    file_suffix: str,
    alpha: float,
    min_trades: int,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
) -> dict[str, Any]:
    sel_files, holdout_files = _split_files(file_suffix=file_suffix)
    sel_rows = _load_rows(sel_files)
    holdout_rows = _load_rows(holdout_files)

    cutoff = datetime.fromisoformat("2025-06-30")
    leak_guard = {
        "assumed_train_end_date": "2025-06-30",
        "selection_earliest_date": str(sel_rows[0]["date"]),
        "holdout_earliest_date": str(holdout_rows[0]["date"]),
        "selection_strictly_post_train_end": bool(datetime.fromisoformat(str(sel_rows[0]["date"])) > cutoff),
        "holdout_strictly_post_train_end": bool(
            datetime.fromisoformat(str(holdout_rows[0]["date"])) > cutoff
        ),
    }

    spread_modes = ["max_minus_hold", "abs_dir", "max_minus_min3", "max_minus_min_bs"]
    spread_thresholds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    hold_bars = [6, 12, 18, 24, 36, 48, 72]
    cooldown_bars = [0, 1, 2, 3, 6, 12]
    inverse_opts = [False, True]

    candidates: list[dict[str, Any]] = []
    for inv in inverse_opts:
        for sm in spread_modes:
            for th in spread_thresholds:
                for hb in hold_bars:
                    for cd in cooldown_bars:
                        cfg = SimConfig(
                            inverse=inv,
                            spread_mode=sm,
                            spread_threshold=float(th),
                            hold_bars=int(hb),
                            cooldown_bars=int(cd),
                        )
                        rep = _simulate(
                            sel_rows,
                            cfg,
                            leverage=leverage,
                            fee_rate=fee_rate,
                            slippage_rate=slippage_rate,
                        )
                        row = {"params": cfg.__dict__, **rep}
                        row["significance"] = {
                            "relaxed_pass": _pass_relaxed(row, alpha=alpha, min_trades=min_trades),
                            "strict_pass": _pass_strict(row, alpha=alpha, min_trades=min_trades),
                        }
                        candidates.append(row)

    strict = [x for x in candidates if x["significance"]["strict_pass"]]
    relaxed = [x for x in candidates if x["significance"]["relaxed_pass"]]

    def _rank_key(x: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(x["sim"]["cagr_to_strict_mdd"]),
            float(x["sim"]["cagr_pct"]),
            -float(x["sim"]["strict_mdd_pct"]),
        )

    strict_sorted = sorted(strict, key=_rank_key, reverse=True)
    relaxed_sorted = sorted(relaxed, key=_rank_key, reverse=True)

    chosen = strict_sorted[0] if strict_sorted else (relaxed_sorted[0] if relaxed_sorted else None)
    holdout_eval = None
    if chosen is not None:
        p = chosen["params"]
        cfg = SimConfig(
            inverse=bool(p["inverse"]),
            spread_mode=str(p["spread_mode"]),
            spread_threshold=float(p["spread_threshold"]),
            hold_bars=int(p["hold_bars"]),
            cooldown_bars=int(p["cooldown_bars"]),
        )
        holdout_eval = _simulate(
            holdout_rows,
            cfg,
            leverage=leverage,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
        holdout_eval["significance"] = {
            "relaxed_pass": _pass_relaxed(holdout_eval, alpha=alpha, min_trades=min_trades),
            "strict_pass": _pass_strict(holdout_eval, alpha=alpha, min_trades=min_trades),
        }

    out = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "setup": {
            "checkpoint": "checkpoints/vlm_grpo_qrdqn_5m_unbiased_balanced_2023to2025h1_step48",
            "file_suffix": file_suffix,
            "alpha": float(alpha),
            "min_trades": int(min_trades),
            "leverage": float(leverage),
            "fee_rate": float(fee_rate),
            "slippage_rate": float(slippage_rate),
        },
        "leakage_guard": leak_guard,
        "selection_pool": {
            "files": sel_files,
            "num_files": int(len(sel_files)),
            "samples": int(len(sel_rows)),
            "period": {"start": str(sel_rows[0]["date"]), "end": str(sel_rows[-1]["date"])},
        },
        "holdout_pool": {
            "files": holdout_files,
            "num_files": int(len(holdout_files)),
            "samples": int(len(holdout_rows)),
            "period": {"start": str(holdout_rows[0]["date"]), "end": str(holdout_rows[-1]["date"])},
        },
        "selection_summary": {
            "num_candidates": int(len(candidates)),
            "relaxed_pass_count": int(len(relaxed)),
            "strict_pass_count": int(len(strict)),
        },
        "selected_params": chosen["params"] if chosen is not None else None,
        "selected_from": "strict" if strict_sorted else ("relaxed" if relaxed_sorted else None),
        "selected_selection_metrics": chosen if chosen is not None else None,
        "holdout_eval": holdout_eval,
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leakage-free holdout validation for step48")
    p.add_argument(
        "--output",
        type=str,
        default="results/vlm_qrdqn_leakfree_holdout_step48_2026-03-07.json",
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-trades", type=int, default=60)
    p.add_argument("--leverage", type=float, default=2.0)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument(
        "--file-suffix",
        type=str,
        default="",
        help="Optional monthly file suffix (e.g. '_seq1000').",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = run_holdout_validation(
        output=args.output,
        file_suffix=args.file_suffix,
        alpha=args.alpha,
        min_trades=args.min_trades,
        leverage=args.leverage,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    print("[done]", args.output)
    print(
        json.dumps(
            {
                "leakage_guard": out.get("leakage_guard", {}),
                "selected_params": out.get("selected_params"),
                "selected_from": out.get("selected_from"),
                "holdout_sim": (out.get("holdout_eval") or {}).get("sim"),
                "holdout_trade_stats": (out.get("holdout_eval") or {}).get("trade_stats"),
                "holdout_significance": (out.get("holdout_eval") or {}).get("significance"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
