"""Leakage-safe direct split search on two action-score files.

Usage:
  - test file: parameter selection only
  - eval file: untouched holdout for final validation
"""

from __future__ import annotations

import argparse
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


def _load_rows(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    rows = payload.get("action_scores", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No action_scores in {path}")
    rows = sorted(rows, key=lambda x: str(x.get("date", "")))
    return rows


def _rank_key(x: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(x["sim"]["cagr_to_strict_mdd"]),
        float(x["sim"]["cagr_pct"]),
        -float(x["sim"]["strict_mdd_pct"]),
    )


def run_search(
    *,
    test_file: str,
    eval_file: str,
    output: str,
    min_spread_threshold: float,
    alpha: float,
    min_trades: int,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
) -> dict[str, Any]:
    test_rows = _load_rows(test_file)
    eval_rows = _load_rows(eval_file)

    spread_modes = ["max_minus_hold", "abs_dir", "max_minus_min3", "max_minus_min_bs"]
    spread_thresholds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    hold_bars = [6, 12, 18, 24, 36, 48, 72]
    cooldown_bars = [0, 1, 2, 3, 6, 12]
    inverse_opts = [False, True]

    candidates: list[dict[str, Any]] = []
    for inv in inverse_opts:
        for sm in spread_modes:
            for th in spread_thresholds:
                if float(th) < float(min_spread_threshold):
                    continue
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
                            test_rows,
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

    strict = sorted([x for x in candidates if x["significance"]["strict_pass"]], key=_rank_key, reverse=True)
    relaxed = sorted([x for x in candidates if x["significance"]["relaxed_pass"]], key=_rank_key, reverse=True)
    selected = strict[0] if strict else (relaxed[0] if relaxed else None)
    selected_from = "strict" if strict else ("relaxed" if relaxed else None)

    eval_rep = None
    if selected is not None:
        p = selected["params"]
        cfg = SimConfig(
            inverse=bool(p["inverse"]),
            spread_mode=str(p["spread_mode"]),
            spread_threshold=float(p["spread_threshold"]),
            hold_bars=int(p["hold_bars"]),
            cooldown_bars=int(p["cooldown_bars"]),
        )
        eval_rep = _simulate(
            eval_rows,
            cfg,
            leverage=leverage,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
        eval_rep["significance"] = {
            "relaxed_pass": _pass_relaxed(eval_rep, alpha=alpha, min_trades=min_trades),
            "strict_pass": _pass_strict(eval_rep, alpha=alpha, min_trades=min_trades),
        }

    out = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "setup": {
            "alpha": float(alpha),
            "min_trades": int(min_trades),
            "min_spread_threshold": float(min_spread_threshold),
            "leverage": float(leverage),
            "fee_rate": float(fee_rate),
            "slippage_rate": float(slippage_rate),
        },
        "files": {
            "test_file": str(Path(test_file).resolve()),
            "eval_file": str(Path(eval_file).resolve()),
        },
        "coverage": {
            "test_samples": int(len(test_rows)),
            "eval_samples": int(len(eval_rows)),
            "test_period": {"start": str(test_rows[0]["date"]), "end": str(test_rows[-1]["date"])},
            "eval_period": {"start": str(eval_rows[0]["date"]), "end": str(eval_rows[-1]["date"])},
        },
        "search_summary": {
            "num_candidates": int(len(candidates)),
            "strict_pass_count_test": int(len(strict)),
            "relaxed_pass_count_test": int(len(relaxed)),
        },
        "selected_from": selected_from,
        "selected_params": selected["params"] if selected else None,
        "selected_test_metrics": selected,
        "eval_metrics": eval_rep,
        "leakage_guard": {
            "test_end": str(test_rows[-1]["date"]),
            "eval_start": str(eval_rows[0]["date"]),
            "eval_strictly_after_test": bool(
                datetime.fromisoformat(str(eval_rows[0]["date"])) > datetime.fromisoformat(str(test_rows[-1]["date"]))
            ),
        },
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct leakage-safe test/eval split search")
    p.add_argument("--test-file", type=str, required=True)
    p.add_argument("--eval-file", type=str, required=True)
    p.add_argument("--output", type=str, default="results/vlm_qrdqn_direct_split_search.json")
    p.add_argument("--min-spread-threshold", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-trades", type=int, default=60)
    p.add_argument("--leverage", type=float, default=2.0)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = run_search(
        test_file=args.test_file,
        eval_file=args.eval_file,
        output=args.output,
        min_spread_threshold=args.min_spread_threshold,
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
                "selected_from": out.get("selected_from"),
                "selected_params": out.get("selected_params"),
                "eval_sim": ((out.get("eval_metrics") or {}).get("sim")),
                "eval_trade_stats": ((out.get("eval_metrics") or {}).get("trade_stats")),
                "eval_significance": ((out.get("eval_metrics") or {}).get("significance")),
                "leakage_guard": out.get("leakage_guard"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
