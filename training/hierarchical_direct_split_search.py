"""Leakage-safe direct split search specialized for hierarchical gate+side reports."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.search_significant_cagr_mdd_pool import _pass_relaxed, _pass_strict


@dataclass(frozen=True)
class HierSimConfig:
    inverse: bool
    gate_margin_threshold: float
    side_margin_threshold: float
    hold_bars: int
    cooldown_bars: int


def _load_rows(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    rows = payload.get("action_scores", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No action_scores in {path}")
    rows = sorted(rows, key=lambda x: str(x.get("date", "")))
    return rows


def _pair_rows(
    gate_rows: list[dict[str, Any]],
    side_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    side_by_date = {str(row["date"]): row for row in side_rows}
    paired: list[dict[str, Any]] = []
    for gate_row in gate_rows:
        date = str(gate_row["date"])
        side_row = side_by_date.get(date)
        if side_row is None:
            continue
        paired.append(
            {
                "date": date,
                "next_return": float(gate_row.get("next_return", 0.0)),
                "gate_scores": gate_row.get("adjusted_scores") or gate_row.get("scores") or {},
                "side_scores": side_row.get("adjusted_scores") or side_row.get("scores") or {},
            }
        )
    if not paired:
        raise ValueError("No overlapping dates between gate and side reports.")
    return paired


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _signal_from_row(row: dict[str, Any], cfg: HierSimConfig) -> int:
    gate_scores = row["gate_scores"]
    side_scores = row["side_scores"]
    trade_score = float(gate_scores.get("TRADE", float("-inf")))
    no_trade_score = float(gate_scores.get("NO_TRADE", float("-inf")))
    gate_margin = trade_score - no_trade_score
    if gate_margin < float(cfg.gate_margin_threshold):
        return 0
    long_score = float(side_scores.get("LONG", float("-inf")))
    short_score = float(side_scores.get("SHORT", float("-inf")))
    side_margin = abs(long_score - short_score)
    if side_margin < float(cfg.side_margin_threshold):
        return 0
    if long_score >= short_score:
        return -1 if cfg.inverse else 1
    return 1 if cfg.inverse else -1


def _simulate_hier(
    rows: list[dict[str, Any]],
    cfg: HierSimConfig,
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

        target = side
        if side != 0:
            bars_left -= 1
            if bars_left <= 0:
                target = 0
        if side == 0 and cooldown > 0:
            cooldown -= 1

        signal = _signal_from_row(row, cfg)
        if side == 0 and cooldown <= 0 and signal != 0:
            target = signal
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
        prev_dt = row_dt

    if side != 0 and trade_entry_eq is not None:
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


def _rank_key(x: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(x["sim"]["cagr_to_strict_mdd"]),
        float(x["sim"]["cagr_pct"]),
        -float(x["sim"]["strict_mdd_pct"]),
    )


def run_search(
    *,
    gate_test_file: str,
    side_test_file: str,
    gate_eval_file: str,
    side_eval_file: str,
    output: str,
    alpha: float,
    min_trades: int,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
) -> dict[str, Any]:
    test_rows = _pair_rows(_load_rows(gate_test_file), _load_rows(side_test_file))
    eval_rows = _pair_rows(_load_rows(gate_eval_file), _load_rows(side_eval_file))

    gate_margin_thresholds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    side_margin_thresholds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    hold_bars = [6, 12, 18, 24, 36, 48, 72]
    cooldown_bars = [0, 1, 2, 3, 6, 12]
    inverse_opts = [False, True]

    candidates: list[dict[str, Any]] = []
    for inv in inverse_opts:
        for gm in gate_margin_thresholds:
            for sm in side_margin_thresholds:
                for hb in hold_bars:
                    for cd in cooldown_bars:
                        cfg = HierSimConfig(
                            inverse=inv,
                            gate_margin_threshold=float(gm),
                            side_margin_threshold=float(sm),
                            hold_bars=int(hb),
                            cooldown_bars=int(cd),
                        )
                        rep = _simulate_hier(
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
        cfg = HierSimConfig(
            inverse=bool(p["inverse"]),
            gate_margin_threshold=float(p["gate_margin_threshold"]),
            side_margin_threshold=float(p["side_margin_threshold"]),
            hold_bars=int(p["hold_bars"]),
            cooldown_bars=int(p["cooldown_bars"]),
        )
        eval_rep = _simulate_hier(
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
        "files": {
            "gate_test_file": str(Path(gate_test_file).resolve()),
            "side_test_file": str(Path(side_test_file).resolve()),
            "gate_eval_file": str(Path(gate_eval_file).resolve()),
            "side_eval_file": str(Path(side_eval_file).resolve()),
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
    parser = argparse.ArgumentParser(description="Hierarchical leakage-safe direct split search")
    parser.add_argument("--gate-test-file", type=str, required=True)
    parser.add_argument("--side-test-file", type=str, required=True)
    parser.add_argument("--gate-eval-file", type=str, required=True)
    parser.add_argument("--side-eval-file", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/hierarchical_direct_split_search.json")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-trades", type=int, default=60)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    parser.add_argument("--slippage-rate", type=float, default=0.0001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = run_search(
        gate_test_file=args.gate_test_file,
        side_test_file=args.side_test_file,
        gate_eval_file=args.gate_eval_file,
        side_eval_file=args.side_eval_file,
        output=args.output,
        alpha=args.alpha,
        min_trades=args.min_trades,
        leverage=args.leverage,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    print(
        json.dumps(
            {
                "selected_from": out.get("selected_from"),
                "selected_params": out.get("selected_params"),
                "eval_sim": ((out.get("eval_metrics") or {}).get("sim")),
                "eval_significance": ((out.get("eval_metrics") or {}).get("significance")),
                "leakage_guard": out.get("leakage_guard"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
