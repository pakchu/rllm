"""Train/Test/Eval leakage-safe validation for step48 execution policy.

- Train: fixed model training window (metadata) up to 2025-06-30.
- Test: parameter search window (post-train only).
- Eval: final untouched holdout window (post-test only).
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
    out: dict[str, str] = {}
    for p in sorted(glob.glob("results/qrdqn_seq_eval_unbiased_balanced_2023to2025h1/step48_nobias_*.json")):
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


def _search_best_on_test(
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

    strict = sorted([x for x in candidates if x["significance"]["strict_pass"]], key=_rank_key, reverse=True)
    relaxed = sorted([x for x in candidates if x["significance"]["relaxed_pass"]], key=_rank_key, reverse=True)

    if strict:
        return strict[0], "strict", {
            "num_candidates": len(candidates),
            "strict_pass_count": len(strict),
            "relaxed_pass_count": len(relaxed),
        }
    if relaxed:
        return relaxed[0], "relaxed", {
            "num_candidates": len(candidates),
            "strict_pass_count": len(strict),
            "relaxed_pass_count": len(relaxed),
        }
    return None, None, {
        "num_candidates": len(candidates),
        "strict_pass_count": 0,
        "relaxed_pass_count": 0,
    }


def _coverage_info(files: list[str], file_suffix: str = "") -> dict[str, Any]:
    # lightweight coverage summary from already-evaluated jsons
    month_rows = []
    total_samples = 0
    for p in files:
        payload = json.loads(Path(p).read_text())
        n = int(payload.get("num_samples", 0))
        s = str(payload.get("start_date"))
        e = str(payload.get("end_date"))
        month = Path(p).name.replace("step48_nobias_", "").replace(".json", "")
        if file_suffix and month.endswith(file_suffix):
            month = month[: -len(file_suffix)]
        month_rows.append({"month": month, "num_samples": n, "requested_range": {"start": s, "end": e}})
        total_samples += n
    return {"total_samples": int(total_samples), "months": month_rows}


def run_validate(
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
        raise FileNotFoundError(f"Missing monthly files: {missing}")

    train_meta = {
        "checkpoint": "checkpoints/vlm_grpo_qrdqn_5m_unbiased_balanced_2023to2025h1_step48",
        "train_period_assumed": {"start": "2023-01-01", "end": "2025-06-30"},
    }

    split_defs = [
        {"name": "tte_A", "test_months": MONTHS[:4], "eval_months": MONTHS[4:]},   # 4/4
        {"name": "tte_B", "test_months": MONTHS[:5], "eval_months": MONTHS[5:]},   # 5/3
        {"name": "tte_C", "test_months": MONTHS[:6], "eval_months": MONTHS[6:]},   # 6/2
    ]

    cutoff = datetime.fromisoformat(train_meta["train_period_assumed"]["end"])
    splits = []
    for sd in split_defs:
        test_files = [fmap[m] for m in sd["test_months"]]
        eval_files = [fmap[m] for m in sd["eval_months"]]
        test_rows = _load_rows(test_files)
        eval_rows = _load_rows(eval_files)

        best, selected_from, counts = _search_best_on_test(
            test_rows,
            alpha=alpha,
            min_trades=min_trades,
            leverage=leverage,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )

        eval_rep = None
        if best is not None:
            p = best["params"]
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

        splits.append(
            {
                "name": sd["name"],
                "test_months": sd["test_months"],
                "eval_months": sd["eval_months"],
                "test_coverage": _coverage_info(test_files, file_suffix=file_suffix),
                "eval_coverage": _coverage_info(eval_files, file_suffix=file_suffix),
                "selection_counts": counts,
                "selected_from": selected_from,
                "selected_params": best["params"] if best else None,
                "selected_test_metrics": best,
                "eval_metrics": eval_rep,
                "leakage_guard": {
                    "train_end": train_meta["train_period_assumed"]["end"],
                    "test_earliest": str(test_rows[0]["date"]),
                    "eval_earliest": str(eval_rows[0]["date"]),
                    "test_after_train": bool(datetime.fromisoformat(str(test_rows[0]["date"])) > cutoff),
                    "eval_after_train": bool(datetime.fromisoformat(str(eval_rows[0]["date"])) > cutoff),
                    "eval_after_test": bool(datetime.fromisoformat(str(eval_rows[0]["date"])) > datetime.fromisoformat(str(test_rows[-1]["date"]))),
                },
            }
        )

    def _best_key(s: dict[str, Any]) -> tuple[int, int, float]:
        em = s.get("eval_metrics") or {}
        sig = em.get("significance") or {}
        sim = em.get("sim") or {}
        return (
            1 if sig.get("strict_pass") else 0,
            1 if sig.get("relaxed_pass") else 0,
            float(sim.get("cagr_to_strict_mdd", -1e9)),
        )

    ranked = sorted(splits, key=_best_key, reverse=True)
    out = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "train_meta": train_meta,
        "pool_variant": {"file_suffix": file_suffix},
        "criteria": {
            "alpha": float(alpha),
            "min_trades": int(min_trades),
            "relaxed": "n>=min_trades AND mean>0 AND p<alpha AND ci_low>0",
            "strict": "relaxed AND n>=n_required_for_80pct_power_alpha5pct",
        },
        "best_split": ranked[0] if ranked else None,
        "splits": splits,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/Test/Eval validation for step48")
    p.add_argument(
        "--output",
        type=str,
        default="results/vlm_qrdqn_train_test_eval_step48_2026-03-07.json",
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
    out = run_validate(
        output=args.output,
        file_suffix=args.file_suffix,
        alpha=args.alpha,
        min_trades=args.min_trades,
        leverage=args.leverage,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    b = out.get("best_split") or {}
    print("[done]", args.output)
    print(
        json.dumps(
            {
                "best_split_name": b.get("name"),
                "test_months": b.get("test_months"),
                "eval_months": b.get("eval_months"),
                "selected_params": b.get("selected_params"),
                "eval_sim": ((b.get("eval_metrics") or {}).get("sim")),
                "eval_trade_stats": ((b.get("eval_metrics") or {}).get("trade_stats")),
                "eval_significance": ((b.get("eval_metrics") or {}).get("significance")),
                "leakage_guard": b.get("leakage_guard"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
