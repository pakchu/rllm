"""Search past-only risk overlays for a fixed hierarchical regime policy.

The overlay is selected on a train split, then reported on validation, OOS, and
combined rows without re-selection.  It models practical kill-switch rules:
strategy drawdown pause and monthly loss pause, both based only on realized
strategy equity up to the current row.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.hierarchical_direct_split_search import HierSimConfig, _load_rows, _norm_cdf, _pair_rows
from training.hierarchical_regime_filter_search import (
    RegimeFilter,
    _attach_features,
    _load_market_features,
    _regime_ok,
    _row_signal,
)


def _parse_float_list(raw: str, default: list[float]) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()] if raw else default


def _parse_int_list(raw: str, default: list[int]) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()] if raw else default


def _split_paths(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _load_policy_rows(gate_files: str, side_files: str, features: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    gates = _split_paths(gate_files)
    sides = _split_paths(side_files)
    if len(gates) != len(sides):
        raise ValueError("gate and side file counts must match")
    rows: list[dict[str, Any]] = []
    for gate, side in zip(gates, sides):
        rows.extend(_pair_rows(_load_rows(gate), _load_rows(side)))
    return _attach_features(sorted(rows, key=lambda r: str(r["date"])), features)


def _trade_stats(trade_returns: list[float]) -> dict[str, Any]:
    n = len(trade_returns)
    mean = sum(trade_returns) / n if n else 0.0
    std = math.sqrt(sum((x - mean) ** 2 for x in trade_returns) / (n - 1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n else 0.0
    t_like = mean / se if se > 0 else 0.0
    p_two = 2.0 * (1.0 - _norm_cdf(abs(t_like))) if se > 0 else 1.0
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se
    effect_d = mean / std if std > 1e-12 else 0.0
    n_required = int(math.ceil(((1.959963984540054 + 0.8416212335729143) / abs(effect_d)) ** 2)) if abs(effect_d) > 1e-12 else None
    return {
        "n_trades": n,
        "mean_trade_ret_pct": mean * 100.0,
        "std_trade_ret_pct": std * 100.0,
        "t_stat_like": t_like,
        "p_value_mean_ret_approx": p_two,
        "ci95_mean_trade_ret_pct": [ci_low * 100.0, ci_high * 100.0],
        "effect_size_d": effect_d,
        "n_required_for_80pct_power_alpha5pct": n_required,
        "n_gap_to_power_rule": max(0, n_required - n) if n_required is not None else None,
    }


def simulate_overlay(
    rows: list[dict[str, Any]],
    cfg: HierSimConfig,
    filt: RegimeFilter,
    *,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
    drawdown_stop: float,
    pause_bars: int,
    monthly_loss_stop: float,
) -> dict[str, Any]:
    eq = peak = 1.0
    max_dd = 0.0
    trade_returns: list[float] = []
    entries = 0
    i = 0
    step = max(1, int(cfg.hold_bars) + int(cfg.cooldown_bars))
    paused_until = -1
    month_key: str | None = None
    month_start_eq = 1.0
    month_paused = False
    cost = (float(fee_rate) + float(slippage_rate)) * float(leverage)

    while i < len(rows):
        dt = datetime.fromisoformat(str(rows[i]["date"]))
        current_month = f"{dt.year}-{dt.month:02d}"
        if current_month != month_key:
            month_key = current_month
            month_start_eq = eq
            month_paused = False

        if i < paused_until or month_paused:
            i += 1
            continue
        signal = _row_signal(rows[i], cfg)
        if signal == 0 or not _regime_ok(rows[i], cfg, filt):
            i += 1
            continue

        entry_eq = eq
        entries += 1
        eq *= max(0.0, 1.0 - cost)
        eq *= max(0.0, 1.0 + float(signal) * float(rows[i].get("next_return", 0.0)) * float(leverage))
        eq *= max(0.0, 1.0 - cost)
        trade_returns.append(eq / entry_eq - 1.0)

        peak = max(peak, eq)
        dd = 1.0 - eq / peak if peak > 0.0 else 0.0
        max_dd = max(max_dd, dd)
        if drawdown_stop < 1.0 and dd >= drawdown_stop:
            paused_until = i + max(1, int(pause_bars))
        if monthly_loss_stop < 1.0 and eq / month_start_eq - 1.0 <= -monthly_loss_stop:
            month_paused = True
        i += step

    start_dt = datetime.fromisoformat(str(rows[0]["date"]))
    end_dt = datetime.fromisoformat(str(rows[-1]["date"]))
    years = max(1.0 / 365.25, float((end_dt - start_dt).days) / 365.25)
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr_pct = ((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0.0 else -100.0
    mdd_pct = max_dd * 100.0
    ratio = cagr_pct / mdd_pct if mdd_pct > 1e-12 else float("inf")
    return {
        "period": {"start": str(rows[0]["date"]), "end": str(rows[-1]["date"]), "years": years},
        "sim": {
            "ret_pct": ret_pct,
            "cagr_pct": cagr_pct,
            "strict_mdd_pct": mdd_pct,
            "cagr_to_strict_mdd": ratio,
            "trade_entries": entries,
            "turnover_legs": entries * 2,
            "samples": len(rows),
            "return_application": "entry_forward_return_non_overlap_regime_filtered_risk_overlay",
        },
        "trade_stats": _trade_stats(trade_returns),
    }


def _rank(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
    sim = row["train"]["sim"]
    stats = row["train"]["trade_stats"]
    return (
        1.0 if float(sim["strict_mdd_pct"]) <= 15.0 else 0.0,
        float(sim["cagr_to_strict_mdd"]),
        float(sim["cagr_pct"]),
        float(stats["ci95_mean_trade_ret_pct"][0]),
        int(sim["trade_entries"]),
    )


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    features = _load_market_features(args.market_csv)
    train_rows = _load_policy_rows(args.train_gate_file, args.train_side_file, features)
    val_rows = _load_policy_rows(args.val_gate_file, args.val_side_file, features)
    oos_rows = _load_policy_rows(args.oos_gate_file, args.oos_side_file, features)
    all_rows = sorted([*train_rows, *val_rows, *oos_rows], key=lambda r: str(r["date"]))

    cfg = HierSimConfig(False, args.gate_margin, args.side_margin, args.hold_bars, args.cooldown_bars)
    filt = RegimeFilter(args.filter_name, abs_trend_min=args.abs_trend_min, align_mode=args.align_mode, trend_col=args.trend_col)

    top: list[dict[str, Any]] = []
    count = 0
    for lev in _parse_float_list(args.leverages, [0.5, 0.75, 1.0, 1.25, 1.5]):
        for dd_stop in _parse_float_list(args.drawdown_stops, [0.075, 0.10, 0.125, 0.15, 1.0]):
            for pause in _parse_int_list(args.pause_bars, [864, 2016, 4032, 8064]):
                for monthly_stop in _parse_float_list(args.monthly_loss_stops, [0.03, 0.05, 0.075, 0.10, 1.0]):
                    count += 1
                    train = simulate_overlay(
                        train_rows,
                        cfg,
                        filt,
                        leverage=lev,
                        fee_rate=args.fee_rate,
                        slippage_rate=args.slippage_rate,
                        drawdown_stop=dd_stop,
                        pause_bars=pause,
                        monthly_loss_stop=monthly_stop,
                    )
                    if int(train["sim"]["trade_entries"]) < args.min_train_trades:
                        continue
                    row = {
                        "overlay": {
                            "leverage": lev,
                            "drawdown_stop": dd_stop,
                            "pause_bars": pause,
                            "monthly_loss_stop": monthly_stop,
                        },
                        "policy": {"hierarchical": cfg.__dict__, "regime_filter": filt.__dict__},
                        "train": train,
                    }
                    if len(top) < args.keep_top or _rank(row) > _rank(top[-1]):
                        top.append(row)
                        top.sort(key=_rank, reverse=True)
                        del top[args.keep_top :]

    for row in top:
        ov = row["overlay"]
        kwargs = {
            "leverage": float(ov["leverage"]),
            "fee_rate": args.fee_rate,
            "slippage_rate": args.slippage_rate,
            "drawdown_stop": float(ov["drawdown_stop"]),
            "pause_bars": int(ov["pause_bars"]),
            "monthly_loss_stop": float(ov["monthly_loss_stop"]),
        }
        row["val"] = simulate_overlay(val_rows, cfg, filt, **kwargs)
        row["oos"] = simulate_overlay(oos_rows, cfg, filt, **kwargs)
        row["all"] = simulate_overlay(all_rows, cfg, filt, **kwargs)

    out = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "num_candidates": count,
        "files": vars(args),
        "top": top,
        "leakage_guard": {
            "overlay_selected_on_train_only": True,
            "validation_and_oos_report_only": True,
            "features_are_past_only": True,
            "train_end": str(train_rows[-1]["date"]),
            "val_start": str(val_rows[0]["date"]),
            "oos_start": str(oos_rows[0]["date"]),
            "chronological_order_ok": datetime.fromisoformat(str(train_rows[-1]["date"])) < datetime.fromisoformat(str(val_rows[0]["date"])) < datetime.fromisoformat(str(oos_rows[0]["date"])),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search fixed-policy risk overlay on train and report val/OOS")
    p.add_argument("--train-gate-file", required=True)
    p.add_argument("--train-side-file", required=True)
    p.add_argument("--val-gate-file", required=True)
    p.add_argument("--val-side-file", required=True)
    p.add_argument("--oos-gate-file", required=True)
    p.add_argument("--oos-side-file", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", default="results/hierarchical_risk_overlay_search.json")
    p.add_argument("--gate-margin", type=float, default=3.0)
    p.add_argument("--side-margin", type=float, default=1.0)
    p.add_argument("--hold-bars", type=int, default=288)
    p.add_argument("--cooldown-bars", type=int, default=24)
    p.add_argument("--filter-name", default="tf_trend_48_0p01")
    p.add_argument("--abs-trend-min", type=float, default=0.01)
    p.add_argument("--align-mode", default="trend_follow", choices=["any", "trend_follow", "mean_revert"])
    p.add_argument("--trend-col", default="trend_48")
    p.add_argument("--leverages", default="")
    p.add_argument("--drawdown-stops", default="")
    p.add_argument("--pause-bars", default="")
    p.add_argument("--monthly-loss-stops", default="")
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--min-train-trades", type=int, default=300)
    p.add_argument("--keep-top", type=int, default=50)
    return p.parse_args()


def main() -> None:
    out = run_search(parse_args())
    best = out["top"][0] if out["top"] else None
    print(json.dumps({
        "num_candidates": out["num_candidates"],
        "best_overlay": None if best is None else best["overlay"],
        "best_train": None if best is None else best["train"]["sim"],
        "best_val": None if best is None else best["val"]["sim"],
        "best_oos": None if best is None else best["oos"]["sim"],
        "best_all": None if best is None else best["all"]["sim"],
        "leakage_guard": out["leakage_guard"],
    }, indent=2))


if __name__ == "__main__":
    main()
