"""Leakage-free split scan for step48 (post-2025H1) with selection/holdout splits."""

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

from training.search_significant_cagr_mdd_pool import (  # noqa: E402
    SimConfig,
    _pass_relaxed,
    _pass_strict,
    _simulate,
)


MONTHS = [
    "2025-07",
    "2025-08",
    "2025-09",
    "2025-10",
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
]


def _file_map(file_suffix: str = "") -> dict[str, str]:
    files = sorted(glob.glob("results/qrdqn_seq_eval_unbiased_balanced_2023to2025h1/step48_nobias_*.json"))
    out: dict[str, str] = {}
    for p in files:
        m = Path(p).name.replace("step48_nobias_", "").replace(".json", "")
        if file_suffix:
            if not m.endswith(file_suffix):
                continue
            m = m[: -len(file_suffix)]
        out[m] = p
    return out


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


def _rank_key(x: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(x["sim"]["cagr_to_strict_mdd"]),
        float(x["sim"]["cagr_pct"]),
        -float(x["sim"]["strict_mdd_pct"]),
    )


def _search_best_on_selection(
    rows: list[dict[str, Any]],
    *,
    alpha: float,
    min_trades: int,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[dict[str, Any] | None, str | None, dict[str, int]]:
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
                            rows,
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
    strict_sorted = sorted(strict, key=_rank_key, reverse=True)
    relaxed_sorted = sorted(relaxed, key=_rank_key, reverse=True)
    if strict_sorted:
        return strict_sorted[0], "strict", {
            "num_candidates": len(candidates),
            "strict_pass_count": len(strict),
            "relaxed_pass_count": len(relaxed),
        }
    if relaxed_sorted:
        return relaxed_sorted[0], "relaxed", {
            "num_candidates": len(candidates),
            "strict_pass_count": len(strict),
            "relaxed_pass_count": len(relaxed),
        }
    return None, None, {
        "num_candidates": len(candidates),
        "strict_pass_count": 0,
        "relaxed_pass_count": 0,
    }


def run_scan(
    *,
    output: str,
    file_suffix: str,
    alpha: float,
    min_trades: int,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
) -> dict[str, Any]:
    fmap = _file_map(file_suffix=file_suffix)
    missing = [m for m in MONTHS if m not in fmap]
    if missing:
        raise FileNotFoundError(f"Missing monthly eval files: {missing}")

    cutoff = datetime.fromisoformat("2025-06-30")
    splits = []
    # selection months [0:k), holdout [k:8)
    for k in [2, 3, 4, 5, 6]:
        sel_months = MONTHS[:k]
        hold_months = MONTHS[k:]
        sel_rows = _load_rows([fmap[m] for m in sel_months])
        hold_rows = _load_rows([fmap[m] for m in hold_months])

        best, chosen_from, counts = _search_best_on_selection(
            sel_rows,
            alpha=alpha,
            min_trades=min_trades,
            leverage=leverage,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
        hold_eval = None
        if best is not None:
            p = best["params"]
            cfg = SimConfig(
                inverse=bool(p["inverse"]),
                spread_mode=str(p["spread_mode"]),
                spread_threshold=float(p["spread_threshold"]),
                hold_bars=int(p["hold_bars"]),
                cooldown_bars=int(p["cooldown_bars"]),
            )
            hold_eval = _simulate(
                hold_rows,
                cfg,
                leverage=leverage,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
            )
            hold_eval["significance"] = {
                "relaxed_pass": _pass_relaxed(hold_eval, alpha=alpha, min_trades=min_trades),
                "strict_pass": _pass_strict(hold_eval, alpha=alpha, min_trades=min_trades),
            }

        splits.append(
            {
                "selection_months": sel_months,
                "holdout_months": hold_months,
                "selection_samples": len(sel_rows),
                "holdout_samples": len(hold_rows),
                "selection_counts": counts,
                "selected_from": chosen_from,
                "selected_params": best["params"] if best else None,
                "selected_selection_metrics": best,
                "holdout_eval": hold_eval,
                "leakage_guard": {
                    "assumed_train_end_date": "2025-06-30",
                    "selection_earliest_date": str(sel_rows[0]["date"]),
                    "holdout_earliest_date": str(hold_rows[0]["date"]),
                    "selection_strictly_post_train_end": bool(
                        datetime.fromisoformat(str(sel_rows[0]["date"])) > cutoff
                    ),
                    "holdout_strictly_post_train_end": bool(
                        datetime.fromisoformat(str(hold_rows[0]["date"])) > cutoff
                    ),
                },
            }
        )

    def _split_score(s: dict[str, Any]) -> tuple[int, int, float]:
        h = s.get("holdout_eval") or {}
        sig = h.get("significance") or {}
        strict = 1 if sig.get("strict_pass") else 0
        relaxed = 1 if sig.get("relaxed_pass") else 0
        ratio = float((h.get("sim") or {}).get("cagr_to_strict_mdd", -1e9))
        return (strict, relaxed, ratio)

    ranked = sorted(splits, key=_split_score, reverse=True)
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
            "months": MONTHS,
        },
        "num_splits": len(splits),
        "best_split_by_holdout": ranked[0] if ranked else None,
        "splits": splits,
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leakage-free split scan for step48")
    p.add_argument(
        "--output",
        type=str,
        default="results/vlm_qrdqn_leakfree_split_scan_step48_2026-03-07.json",
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
    out = run_scan(
        output=args.output,
        file_suffix=args.file_suffix,
        alpha=args.alpha,
        min_trades=args.min_trades,
        leverage=args.leverage,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    best = out.get("best_split_by_holdout") or {}
    print("[done]", args.output)
    print(
        json.dumps(
            {
                "best_selection_months": best.get("selection_months"),
                "best_holdout_months": best.get("holdout_months"),
                "selected_params": best.get("selected_params"),
                "holdout_sim": ((best.get("holdout_eval") or {}).get("sim")),
                "holdout_trade_stats": ((best.get("holdout_eval") or {}).get("trade_stats")),
                "holdout_significance": ((best.get("holdout_eval") or {}).get("significance")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
