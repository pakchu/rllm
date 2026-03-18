"""Expand evaluation pool and search statistically significant CAGR/MDD cases.

Pool (default):
  - validation: 2024H2 sequential eval
  - monthly sequential evals: 2025-03 .. 2026-02

Significance levels:
  - relaxed: n>=min_trades, mean>0, p<alpha, CI_low>0
  - strict: relaxed + n>=required_n_for_80pct_power
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


@dataclass(frozen=True)
class SimConfig:
    inverse: bool
    spread_mode: str
    spread_threshold: float
    hold_bars: int
    cooldown_bars: int


def _load_pool_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        payload = json.loads(Path(p).read_text())
        arr = payload.get("action_scores", [])
        if not isinstance(arr, list):
            continue
        rows.extend(arr)
    rows.sort(key=lambda x: str(x.get("date", "")))
    if not rows:
        raise ValueError("No action_scores loaded from pool files.")
    return rows


def _spread_value(mode: str, b: float, h: float, s: float) -> float:
    dir_score = b - s
    if mode == "max_minus_hold":
        return max(b, s) - h
    if mode == "abs_dir":
        return abs(dir_score)
    if mode == "max_minus_min3":
        return max(b, s) - min(b, h, s)
    if mode == "max_minus_min_bs":
        return max(b, s) - min(b, s)
    raise ValueError(f"Unsupported spread_mode: {mode}")


def _simulate(
    rows: list[dict[str, Any]],
    cfg: SimConfig,
    *,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
) -> dict[str, Any]:
    eq = 1.0
    peak = 1.0
    max_dd = 0.0

    side = 0
    bars_left = 0
    cooldown = 0
    trade_entry_eq: float | None = None

    entries = 0
    turnover_legs = 0
    trade_returns: list[float] = []
    equity_curve: list[float] = []
    gap_flatten_events = 0

    row_dts = [datetime.fromisoformat(str(row["date"])) for row in rows]
    deltas_sec = [
        max(0.0, float((b - a).total_seconds()))
        for a, b in zip(row_dts[:-1], row_dts[1:])
        if b > a
    ]
    inferred_bar_sec = 300.0
    if deltas_sec:
        deltas_sec_sorted = sorted(deltas_sec)
        inferred_bar_sec = float(deltas_sec_sorted[len(deltas_sec_sorted) // 2])
    gap_sec_threshold = max(600.0, inferred_bar_sec * 1.5)

    prev_dt: datetime | None = None
    for row, row_dt in zip(rows, row_dts):
        if prev_dt is not None and float((row_dt - prev_dt).total_seconds()) > gap_sec_threshold:
            if side != 0 and trade_entry_eq is not None:
                leg = abs(int(side))
                turnover_legs += leg
                eq *= max(
                    0.0,
                    1.0 - (float(fee_rate) + float(slippage_rate)) * float(leg) * float(leverage),
                )
                trade_returns.append(eq / trade_entry_eq - 1.0)
                trade_entry_eq = None
                side = 0
                bars_left = 0
                cooldown = 0
                gap_flatten_events += 1

        next_ret = float(row.get("next_return", 0.0))
        if side != 0:
            eq *= max(0.0, 1.0 + float(side) * float(next_ret) * float(leverage))

        scores = row.get("adjusted_scores", {})
        b = float(scores.get("BUY", 0.0))
        h = float(scores.get("HOLD", 0.0))
        s = float(scores.get("SELL", 0.0))
        dir_score = b - s
        spread = _spread_value(cfg.spread_mode, b, h, s)

        target = side
        if side != 0:
            bars_left -= 1
            if bars_left <= 0:
                target = 0
        if side == 0 and cooldown > 0:
            cooldown -= 1

        if side == 0 and cooldown <= 0 and spread >= float(cfg.spread_threshold):
            if dir_score > 0.0:
                target = -1 if cfg.inverse else 1
            elif dir_score < 0.0:
                target = 1 if cfg.inverse else -1
            else:
                target = 0
            if target != 0:
                bars_left = int(cfg.hold_bars)

        if target != side:
            leg = abs(int(target) - int(side))
            turnover_legs += leg
            eq *= max(0.0, 1.0 - (float(fee_rate) + float(slippage_rate)) * float(leg) * float(leverage))

            if side != 0 and trade_entry_eq is not None:
                trade_returns.append(eq / trade_entry_eq - 1.0)
                trade_entry_eq = None
                if target == 0:
                    cooldown = int(cfg.cooldown_bars)

            if target != 0:
                entries += 1
                trade_entry_eq = eq

        side = target
        peak = max(peak, eq)
        if peak > 0.0:
            max_dd = max(max_dd, 1.0 - eq / peak)
        equity_curve.append(eq)
        prev_dt = row_dt

    if side != 0 and trade_entry_eq is not None:
        # terminal close cost
        eq *= max(0.0, 1.0 - (float(fee_rate) + float(slippage_rate)) * float(leverage))
        trade_returns.append(eq / trade_entry_eq - 1.0)

    n = len(trade_returns)
    if n >= 2:
        mean = sum(trade_returns) / float(n)
        var = sum((x - mean) ** 2 for x in trade_returns) / float(n - 1)
        std = math.sqrt(max(0.0, var))
    elif n == 1:
        mean = float(trade_returns[0])
        std = 0.0
    else:
        mean = 0.0
        std = 0.0

    se = (std / math.sqrt(float(n))) if n > 0 else 0.0
    t_like = (mean / se) if se > 0.0 else 0.0
    p_two = 2.0 * (1.0 - _norm_cdf(abs(t_like))) if se > 0.0 else 1.0
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se
    effect_d = (mean / std) if std > 1e-12 else 0.0

    n_required = None
    n_gap = None
    if abs(effect_d) > 1e-12:
        z_alpha_over_2 = 1.959963984540054
        z_beta_80 = 0.8416212335729143
        n_required = int(math.ceil(((z_alpha_over_2 + z_beta_80) / abs(effect_d)) ** 2))
        n_gap = int(max(0, n_required - n))

    start_dt = datetime.fromisoformat(str(rows[0]["date"]))
    end_dt = datetime.fromisoformat(str(rows[-1]["date"]))
    years = max(1.0 / 365.25, float((end_dt - start_dt).days) / 365.25)
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr_pct = float((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0.0 else -100.0
    mdd_pct = float(max_dd * 100.0)
    cagr_to_mdd = float(cagr_pct / mdd_pct) if mdd_pct > 1e-12 else float("inf")

    return {
        "period": {"start": str(rows[0]["date"]), "end": str(rows[-1]["date"]), "years": float(years)},
        "sim": {
            "ret_pct": float(ret_pct),
            "cagr_pct": float(cagr_pct),
            "strict_mdd_pct": float(mdd_pct),
            "cagr_to_strict_mdd": float(cagr_to_mdd),
            "trade_entries": int(entries),
            "turnover_legs": int(turnover_legs),
            "samples": int(len(rows)),
            "gap_flatten_events": int(gap_flatten_events),
            "inferred_bar_minutes": float(inferred_bar_sec / 60.0),
        },
        "trade_stats": {
            "n_trades": int(n),
            "mean_trade_ret_pct": float(mean * 100.0),
            "std_trade_ret_pct": float(std * 100.0),
            "t_stat_like": float(t_like),
            "p_value_mean_ret_approx": float(p_two),
            "ci95_mean_trade_ret_pct": [float(ci_low * 100.0), float(ci_high * 100.0)],
            "effect_size_d": float(effect_d),
            "n_required_for_80pct_power_alpha5pct": n_required,
            "n_gap_to_power_rule": n_gap,
        },
    }


def _pass_relaxed(
    rep: dict[str, Any],
    *,
    alpha: float,
    min_trades: int,
) -> bool:
    ts = rep["trade_stats"]
    return bool(
        int(ts["n_trades"]) >= int(min_trades)
        and float(ts["mean_trade_ret_pct"]) > 0.0
        and float(ts["p_value_mean_ret_approx"]) < float(alpha)
        and float(ts["ci95_mean_trade_ret_pct"][0]) > 0.0
    )


def _pass_strict(
    rep: dict[str, Any],
    *,
    alpha: float,
    min_trades: int,
) -> bool:
    if not _pass_relaxed(rep, alpha=alpha, min_trades=min_trades):
        return False
    n_req = rep["trade_stats"].get("n_required_for_80pct_power_alpha5pct")
    if n_req is None:
        return False
    return bool(int(rep["trade_stats"]["n_trades"]) >= int(n_req))


def _default_pool_files() -> list[str]:
    files = ["results/vlm_qrdqn_unbiased_balanced_val_2024h2_step36.json"]
    files += sorted(glob.glob("results/qrdqn_seq_eval_unbiased_balanced/step36_nobias_2025-*.json"))
    files += sorted(glob.glob("results/qrdqn_seq_eval_unbiased_balanced/step36_nobias_2026-*.json"))
    # keep only 2025-03..2026-02 + val
    keep: list[str] = []
    for p in files:
        if "val_2024h2" in p:
            keep.append(p)
            continue
        if any(
            m in p
            for m in [
                "2025-03",
                "2025-04",
                "2025-05",
                "2025-06",
                "2025-07",
                "2025-08",
                "2025-09",
                "2025-10",
                "2025-11",
                "2025-12",
                "2026-01",
                "2026-02",
            ]
        ):
            keep.append(p)
    return keep


def _leakfree_step48_pool_files() -> list[str]:
    files = sorted(glob.glob("results/qrdqn_seq_eval_unbiased_balanced_2023to2025h1/step48_nobias_*.json"))
    keep: list[str] = []
    for p in files:
        if any(
            m in p
            for m in [
                "2025-07",
                "2025-08",
                "2025-09",
                "2025-10",
                "2025-11",
                "2025-12",
                "2026-01",
                "2026-02",
            ]
        ):
            keep.append(p)
    return keep


def _leakfree_step48_seq1000_pool_files() -> list[str]:
    files = sorted(
        glob.glob(
            "results/qrdqn_seq_eval_unbiased_balanced_2023to2025h1/step48_nobias_*_seq1000.json"
        )
    )
    keep: list[str] = []
    for p in files:
        if any(
            m in p
            for m in [
                "2025-07",
                "2025-08",
                "2025-09",
                "2025-10",
                "2025-11",
                "2025-12",
                "2026-01",
                "2026-02",
            ]
        ):
            keep.append(p)
    return keep


def _pool_files_by_mode(pool_mode: str) -> tuple[list[str], dict[str, Any]]:
    mode = str(pool_mode).strip().lower()
    if mode == "leakfree_step48_post2025h1_seq1000":
        files = _leakfree_step48_seq1000_pool_files()
        meta = {
            "pool_mode": mode,
            "checkpoint": "checkpoints/vlm_grpo_qrdqn_5m_unbiased_balanced_2023to2025h1_step48",
            "assumed_train_end_date": "2025-06-30",
            "note": "Same leakage-safe post-2025H1 pool, but with sequential max_samples=1000 per month.",
        }
        return files, meta
    if mode == "leakfree_step48_post2025h1":
        files = _leakfree_step48_pool_files()
        meta = {
            "pool_mode": mode,
            "checkpoint": "checkpoints/vlm_grpo_qrdqn_5m_unbiased_balanced_2023to2025h1_step48",
            "assumed_train_end_date": "2025-06-30",
            "note": "Checkpoint name indicates train span through 2025-H1; pool starts 2025-07-01.",
        }
        return files, meta
    # default
    files = _default_pool_files()
    meta = {
        "pool_mode": "default_step36",
        "checkpoint": "checkpoints/vlm_grpo_qrdqn_5m_unbiased_balanced_step36",
        "assumed_train_end_date": None,
        "note": "Legacy mixed pool used previously (may include potential train overlap uncertainty).",
    }
    return files, meta


def run_search(
    *,
    output: str,
    pool_mode: str,
    alpha: float,
    min_trades: int,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
) -> dict[str, Any]:
    pool_files, pool_meta = _pool_files_by_mode(pool_mode)
    rows = _load_pool_rows(pool_files)
    leakage_guard: dict[str, Any] = {}
    if pool_meta.get("assumed_train_end_date"):
        cutoff = datetime.fromisoformat(str(pool_meta["assumed_train_end_date"]))
        min_dt = datetime.fromisoformat(str(rows[0]["date"]))
        leakage_guard = {
            "assumed_train_end_date": str(pool_meta["assumed_train_end_date"]),
            "earliest_eval_sample_date": str(rows[0]["date"]),
            "strictly_post_train_end": bool(min_dt > cutoff),
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
                            rows,
                            cfg,
                            leverage=leverage,
                            fee_rate=fee_rate,
                            slippage_rate=slippage_rate,
                        )
                        row = {
                            "params": {
                                "inverse": bool(inv),
                                "spread_mode": sm,
                                "spread_threshold": float(th),
                                "hold_bars": int(hb),
                                "cooldown_bars": int(cd),
                            },
                            **rep,
                        }
                        row["significance"] = {
                            "relaxed_pass": _pass_relaxed(row, alpha=alpha, min_trades=min_trades),
                            "strict_pass": _pass_strict(row, alpha=alpha, min_trades=min_trades),
                        }
                        candidates.append(row)

    relaxed = [x for x in candidates if x["significance"]["relaxed_pass"]]
    strict = [x for x in candidates if x["significance"]["strict_pass"]]
    relaxed_mdd15 = [x for x in relaxed if float(x["sim"]["strict_mdd_pct"]) <= 15.0]
    relaxed_sorted = sorted(
        relaxed,
        key=lambda x: (
            float(x["sim"]["cagr_to_strict_mdd"]),
            float(x["sim"]["cagr_pct"]),
            -float(x["sim"]["strict_mdd_pct"]),
        ),
        reverse=True,
    )
    relaxed_mdd15_sorted = sorted(
        relaxed_mdd15,
        key=lambda x: (
            float(x["sim"]["cagr_to_strict_mdd"]),
            float(x["sim"]["cagr_pct"]),
            -float(x["sim"]["strict_mdd_pct"]),
        ),
        reverse=True,
    )
    strict_sorted = sorted(
        strict,
        key=lambda x: (
            float(x["sim"]["cagr_to_strict_mdd"]),
            float(x["sim"]["cagr_pct"]),
            -float(x["sim"]["strict_mdd_pct"]),
        ),
        reverse=True,
    )

    out = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "pool": {
            **pool_meta,
            "files": pool_files,
            "num_files": int(len(pool_files)),
            "samples": int(len(rows)),
            "period": {"start": str(rows[0]["date"]), "end": str(rows[-1]["date"])},
            "leakage_guard": leakage_guard,
        },
        "search_space": {
            "inverse": inverse_opts,
            "spread_mode": spread_modes,
            "spread_thresholds": spread_thresholds,
            "hold_bars": hold_bars,
            "cooldown_bars": cooldown_bars,
        },
        "criteria": {
            "alpha": float(alpha),
            "min_trades": int(min_trades),
            "relaxed": "n>=min_trades AND mean>0 AND p<alpha AND ci_low>0",
            "strict": "relaxed AND n>=n_required_for_80pct_power_alpha5pct",
        },
        "summary": {
            "num_candidates": int(len(candidates)),
            "relaxed_pass_count": int(len(relaxed)),
            "strict_pass_count": int(len(strict)),
            "relaxed_pass_count_mdd_le_15": int(len(relaxed_mdd15)),
        },
        "best_relaxed": relaxed_sorted[0] if relaxed_sorted else None,
        "best_relaxed_mdd_le_15": relaxed_mdd15_sorted[0] if relaxed_mdd15_sorted else None,
        "best_strict": strict_sorted[0] if strict_sorted else None,
        "top10_relaxed": relaxed_sorted[:10],
        "top10_relaxed_mdd_le_15": relaxed_mdd15_sorted[:10],
        "top10_strict": strict_sorted[:10],
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search significant CAGR/MDD cases on expanded pool")
    p.add_argument(
        "--output",
        type=str,
        default="results/vlm_qrdqn_significant_cagr_mdd_pool_2026-03-07.json",
    )
    p.add_argument(
        "--pool-mode",
        type=str,
        default="default_step36",
        choices=[
            "default_step36",
            "leakfree_step48_post2025h1",
            "leakfree_step48_post2025h1_seq1000",
        ],
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-trades", type=int, default=60)
    p.add_argument("--leverage", type=float, default=2.0)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = run_search(
        output=args.output,
        pool_mode=args.pool_mode,
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
                "summary": out.get("summary", {}),
                "best_relaxed_params": (out.get("best_relaxed") or {}).get("params"),
                "best_relaxed_sim": (out.get("best_relaxed") or {}).get("sim"),
                "best_relaxed_trade_stats": (out.get("best_relaxed") or {}).get("trade_stats"),
                "best_strict_params": (out.get("best_strict") or {}).get("params"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
