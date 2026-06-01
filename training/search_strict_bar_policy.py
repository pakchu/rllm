"""Grid-search hierarchical LLM policies under strict OHLC bar execution.

This is the replacement search surface after the forward-return simulator proved
optimistic.  It selects only on the train split, then evaluates the selected
candidates on validation/OOS/all with the same exact bar-by-bar simulator.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.hierarchical_direct_split_search import HierSimConfig
from training.hierarchical_regime_filter_search import RegimeFilter, _load_market_features
from training.strict_bar_backtest import BarExecutionConfig, load_market_bars, load_policy_rows, simulate_bar_by_bar


def _parse_float_list(raw: str, default: list[float]) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()] if raw else default


def _parse_int_list(raw: str, default: list[int]) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()] if raw else default


def _parse_bool_list(raw: str, default: list[bool]) -> list[bool]:
    if not raw:
        return default
    return [x.strip().lower() in {"1", "true", "yes", "y"} for x in raw.split(",") if x.strip()]


def _make_filters(names: str) -> list[RegimeFilter]:
    if not names:
        names = "tf48_001,mr48_001,none"
    out: list[RegimeFilter] = []
    for name in [x.strip() for x in names.split(",") if x.strip()]:
        if name == "none":
            out.append(RegimeFilter("none"))
        elif name in {"tf48_001", "tf_trend_48_0p01"}:
            out.append(RegimeFilter("tf_trend_48_0p01", abs_trend_min=0.01, align_mode="trend_follow", trend_col="trend_48"))
        elif name in {"mr48_001", "mr_trend_48_0p01"}:
            out.append(RegimeFilter("mr_trend_48_0p01", abs_trend_min=0.01, align_mode="mean_revert", trend_col="trend_48"))
        elif name in {"tf144_0005", "tf_trend_144_0p005"}:
            out.append(RegimeFilter("tf_trend_144_0p005", abs_trend_min=0.005, align_mode="trend_follow", trend_col="trend_144"))
        elif name in {"mr144_0005", "mr_trend_144_0p005"}:
            out.append(RegimeFilter("mr_trend_144_0p005", abs_trend_min=0.005, align_mode="mean_revert", trend_col="trend_144"))
        else:
            raise ValueError(f"unknown filter name: {name}")
    return out


def _load_selection_files(selection_file: str) -> dict[str, Any]:
    payload = json.loads(Path(selection_file).read_text())
    files = payload.get("files") or {}
    if not files:
        raise ValueError(f"selection file lacks files block: {selection_file}")
    return files


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
    files = _load_selection_files(args.selection_file)
    market_csv = args.market_csv or files["market_csv"]
    features = _load_market_features(market_csv)
    market = load_market_bars(market_csv)
    train_rows = load_policy_rows(files["train_gate_file"], files["train_side_file"], features)
    val_rows = load_policy_rows(files["val_gate_file"], files["val_side_file"], features)
    oos_rows = load_policy_rows(files["oos_gate_file"], files["oos_side_file"], features)
    all_rows = sorted([*train_rows, *val_rows, *oos_rows], key=lambda r: str(r["date"]))

    top: list[dict[str, Any]] = []
    count = 0
    for inverse in _parse_bool_list(args.inverse_opts, [False, True]):
        for gate_margin in _parse_float_list(args.gate_margins, [2.0, 3.0, 4.0, 5.0]):
            for side_margin in _parse_float_list(args.side_margins, [0.5, 1.0, 2.0]):
                for hold_bars in _parse_int_list(args.hold_bars, [144, 192]):
                    for cooldown_bars in _parse_int_list(args.cooldown_bars, [12, 24]):
                        cfg = HierSimConfig(inverse, gate_margin, side_margin, hold_bars, cooldown_bars)
                        for filt in _make_filters(args.filters):
                            for leverage in _parse_float_list(args.leverages, [0.25, 0.5]):
                                count += 1
                                exec_cfg = BarExecutionConfig(
                                    leverage=leverage,
                                    fee_rate=float(args.fee_rate if args.fee_rate is not None else files.get("fee_rate", 0.0004)),
                                    slippage_rate=float(args.slippage_rate if args.slippage_rate is not None else files.get("slippage_rate", 0.0001)),
                                    drawdown_stop=float(args.drawdown_stop),
                                    pause_bars=int(args.pause_bars),
                                    monthly_loss_stop=float(args.monthly_loss_stop),
                                    entry_delay_bars=int(args.entry_delay_bars),
                                )
                                train = simulate_bar_by_bar(train_rows, market, cfg, filt, exec_cfg)
                                if int(train["sim"]["trade_entries"]) < int(args.min_train_trades):
                                    continue
                                row = {
                                    "policy": {"hierarchical": cfg.__dict__, "regime_filter": filt.__dict__},
                                    "execution": exec_cfg.__dict__,
                                    "train": train,
                                }
                                if len(top) < args.keep_top or _rank(row) > _rank(top[-1]):
                                    top.append(row)
                                    top.sort(key=_rank, reverse=True)
                                    del top[int(args.keep_top) :]
    for row in top:
        cfg = HierSimConfig(**row["policy"]["hierarchical"])
        filt = RegimeFilter(**row["policy"]["regime_filter"])
        exec_cfg = BarExecutionConfig(**row["execution"])
        row["val"] = simulate_bar_by_bar(val_rows, market, cfg, filt, exec_cfg)
        row["oos"] = simulate_bar_by_bar(oos_rows, market, cfg, filt, exec_cfg)
        row["all"] = simulate_bar_by_bar(all_rows, market, cfg, filt, exec_cfg)
    out = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "selection_file": args.selection_file,
        "source_files": files,
        "num_candidates": count,
        "top": top,
        "search_space": {
            "inverse_opts": args.inverse_opts,
            "gate_margins": args.gate_margins,
            "side_margins": args.side_margins,
            "hold_bars": args.hold_bars,
            "cooldown_bars": args.cooldown_bars,
            "filters": args.filters,
            "leverages": args.leverages,
        },
        "leakage_guard": {
            "selected_on_train_only": True,
            "validation_oos_report_only": True,
            "strict_bar_by_bar": True,
            "uses_forward_return_column": False,
            "features_are_past_only": True,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict bar-by-bar grid search for hierarchical LLM policy scores")
    p.add_argument("--selection-file", default="results/h144_candidate_trainselected_risk_overlay_search.json")
    p.add_argument("--market-csv", default="")
    p.add_argument("--output", default="results/h144_strict_bar_policy_search.json")
    p.add_argument("--inverse-opts", default="false,true")
    p.add_argument("--gate-margins", default="2,3,4,5")
    p.add_argument("--side-margins", default="0.5,1,2")
    p.add_argument("--hold-bars", default="144,192")
    p.add_argument("--cooldown-bars", default="12,24")
    p.add_argument("--filters", default="tf48_001,mr48_001,none")
    p.add_argument("--leverages", default="0.25,0.5")
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--fee-rate", type=float, default=None)
    p.add_argument("--slippage-rate", type=float, default=None)
    p.add_argument("--drawdown-stop", type=float, default=1.0)
    p.add_argument("--pause-bars", type=int, default=864)
    p.add_argument("--monthly-loss-stop", type=float, default=1.0)
    p.add_argument("--min-train-trades", type=int, default=100)
    p.add_argument("--keep-top", type=int, default=20)
    return p.parse_args()


def main() -> None:
    out = run_search(parse_args())
    best = out["top"][0] if out["top"] else None
    print(json.dumps({
        "num_candidates": out["num_candidates"],
        "best_policy": None if best is None else best["policy"],
        "best_execution": None if best is None else best["execution"],
        "best_train": None if best is None else best["train"]["sim"],
        "best_val": None if best is None else best["val"]["sim"],
        "best_oos": None if best is None else best["oos"]["sim"],
        "best_all": None if best is None else best["all"]["sim"],
        "leakage_guard": out["leakage_guard"],
    }, indent=2))


if __name__ == "__main__":
    main()
