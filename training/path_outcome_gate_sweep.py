"""Sweep executable path-outcome gate labels and report oracle strict-bar upper bounds.

This is an analysis tool, not a deployable policy. It deliberately uses future
path outcomes to decide whether/which side would trade so we can find label
conditions that are worth asking an LLM gate to learn.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from training.path_outcome_dataset import PathOutcomeConfig, compute_trade_path_outcome
from training.strict_bar_backtest import _trade_stats, load_market_bars


@dataclass(frozen=True)
class GateSweepConfig:
    hold_bars: int
    entry_delay_bars: int
    mae_penalty: float
    mfe_bonus: float
    min_net_return: float
    min_utility: float
    max_mae: float
    cooldown_bars: int = 0


def _years(start: str, end: str) -> float:
    a = datetime.fromisoformat(str(start))
    b = datetime.fromisoformat(str(end))
    return max(1.0 / 365.25, float((b - a).total_seconds()) / (365.25 * 24 * 3600))


def _compute_signal_table(
    market: pd.DataFrame,
    *,
    hold_bars: int,
    entry_delay_bars: int,
    fee_rate: float,
    slippage_rate: float,
    leverage: float,
    mae_penalty_values: list[float],
    mfe_bonus_values: list[float],
    window_size: int,
) -> list[dict[str, Any]]:
    """Precompute long/short outcomes for each mae/mfe utility variant."""
    last_signal_pos = len(market) - max(1, int(entry_delay_bars)) - max(1, int(hold_bars)) - 1
    rows: list[dict[str, Any]] = []
    for pos in range(max(0, int(window_size) - 1), max(0, last_signal_pos) + 1):
        base: dict[str, Any] = {
            "date": str(pd.to_datetime(market.iloc[pos]["date"])),
            "signal_pos": int(pos),
            "variants": {},
        }
        for mae_penalty in mae_penalty_values:
            for mfe_bonus in mfe_bonus_values:
                cfg = PathOutcomeConfig(
                    hold_bars=hold_bars,
                    entry_delay_bars=entry_delay_bars,
                    fee_rate=fee_rate,
                    slippage_rate=slippage_rate,
                    leverage=leverage,
                    mae_penalty=mae_penalty,
                    mfe_bonus=mfe_bonus,
                )
                long = compute_trade_path_outcome(market, pos, "LONG", cfg)
                short = compute_trade_path_outcome(market, pos, "SHORT", cfg)
                if long is None or short is None:
                    continue
                best = long if long.utility >= short.utility else short
                base["variants"][f"{mae_penalty:g}|{mfe_bonus:g}"] = {
                    "side": best.side,
                    "net_return": float(best.net_return),
                    "mae": float(best.mae),
                    "mfe": float(best.mfe),
                    "utility": float(best.utility),
                }
        if base["variants"]:
            rows.append(base)
    return rows


def _simulate_oracle_strict(
    rows: list[dict[str, Any]],
    market: pd.DataFrame,
    cfg: GateSweepConfig,
    *,
    fee_rate: float,
    slippage_rate: float,
    leverage: float,
) -> dict[str, Any]:
    """Strict OHLC simulation where signal/side come from future path labels."""
    if not rows:
        return {
            "period": {"start": None, "end": None, "years": 0.0},
            "sim": {"ret_pct": 0.0, "cagr_pct": 0.0, "strict_mdd_pct": 0.0, "cagr_to_strict_mdd": 0.0, "trade_entries": 0, "samples": 0},
            "trade_stats": _trade_stats([]),
        }
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    variant_key = f"{cfg.mae_penalty:g}|{cfg.mfe_bonus:g}"
    cost = (float(fee_rate) + float(slippage_rate)) * float(leverage)
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    trade_returns: list[float] = []
    entries = 0
    candidate_trades = 0
    next_allowed_pos = 0

    for row in rows:
        v = row["variants"].get(variant_key)
        if v is None:
            continue
        if (
            float(v["net_return"]) <= float(cfg.min_net_return)
            or float(v["utility"]) <= float(cfg.min_utility)
            or float(v["mae"]) > float(cfg.max_mae)
        ):
            continue
        candidate_trades += 1
        signal_pos = int(row["signal_pos"])
        if signal_pos < next_allowed_pos:
            continue
        entry_pos = signal_pos + max(0, int(cfg.entry_delay_bars))
        exit_pos = entry_pos + max(1, int(cfg.hold_bars))
        if entry_pos >= len(market) - 1 or exit_pos >= len(market):
            continue
        signal = 1 if str(v["side"]).upper() == "LONG" else -1
        entry_eq = eq
        entries += 1
        eq *= max(0.0, 1.0 - cost)
        if peak > 0.0:
            max_dd = max(max_dd, 1.0 - max(0.0, eq) / peak)
        for j in range(entry_pos, exit_pos):
            open_j = float(opens[j])
            if open_j <= 0.0:
                continue
            if signal > 0:
                adverse_ret = (float(lows[j]) - open_j) / open_j
                close_ret = (float(opens[j + 1]) - open_j) / open_j
            else:
                adverse_ret = (open_j - float(highs[j])) / open_j
                close_ret = (open_j - float(opens[j + 1])) / open_j
            adverse_eq = eq * (1.0 + float(leverage) * adverse_ret)
            if peak > 0.0:
                max_dd = max(max_dd, 1.0 - max(0.0, adverse_eq) / peak)
            eq *= max(0.0, 1.0 + float(leverage) * close_ret)
            peak = max(peak, eq)
            if eq <= 0.0:
                break
        eq *= max(0.0, 1.0 - cost)
        if peak > 0.0:
            max_dd = max(max_dd, 1.0 - max(0.0, eq) / peak)
        peak = max(peak, eq)
        trade_returns.append(eq / entry_eq - 1.0)
        next_allowed_pos = exit_pos + max(0, int(cfg.cooldown_bars))
        if eq <= 0.0:
            break

    start = str(rows[0]["date"])
    end = str(rows[-1]["date"])
    years = _years(start, end)
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr_pct = ((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0.0 else -100.0
    mdd_pct = max_dd * 100.0
    ratio = cagr_pct / mdd_pct if mdd_pct > 1e-12 else float("inf")
    return {
        "period": {"start": start, "end": end, "years": years},
        "sim": {
            "ret_pct": float(ret_pct),
            "cagr_pct": float(cagr_pct),
            "strict_mdd_pct": float(mdd_pct),
            "cagr_to_strict_mdd": float(ratio),
            "trade_entries": int(entries),
            "candidate_trades_before_overlap_filter": int(candidate_trades),
            "samples": int(len(rows)),
            "return_application": "oracle_future_path_signal_actual_ohlc_bar_by_bar_strict_mdd",
        },
        "trade_stats": _trade_stats(trade_returns),
    }


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
    sim = row["train"]["sim"]
    return (
        float(sim["cagr_to_strict_mdd"]),
        float(sim["cagr_pct"]),
        -float(sim["strict_mdd_pct"]),
        int(sim["trade_entries"]),
    )


def run_sweep(
    *,
    market_csv: str,
    output: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    eval_start: str,
    eval_end: str,
    window_size: int,
    hold_bars: int,
    entry_delay_bars: int,
    fee_rate: float,
    slippage_rate: float,
    leverage: float,
    mae_penalty_values: list[float],
    mfe_bonus_values: list[float],
    min_net_return_values: list[float],
    min_utility_values: list[float],
    max_mae_values: list[float],
    cooldown_bars_values: list[int],
    min_train_trades: int,
) -> dict[str, Any]:
    market = load_market_bars(market_csv)

    def slice_rows(start: str, end: str) -> list[dict[str, Any]]:
        df = market[(market["date"] >= pd.to_datetime(start)) & (market["date"] <= pd.to_datetime(end))].reset_index(drop=True)
        return _compute_signal_table(
            df,
            hold_bars=hold_bars,
            entry_delay_bars=entry_delay_bars,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            leverage=leverage,
            mae_penalty_values=mae_penalty_values,
            mfe_bonus_values=mfe_bonus_values,
            window_size=window_size,
        ), df

    train_rows, train_market = slice_rows(train_start, train_end)
    test_rows, test_market = slice_rows(test_start, test_end)
    eval_rows, eval_market = slice_rows(eval_start, eval_end)

    candidates: list[dict[str, Any]] = []
    for mae_penalty in mae_penalty_values:
        for mfe_bonus in mfe_bonus_values:
            for min_net_return in min_net_return_values:
                for min_utility in min_utility_values:
                    for max_mae in max_mae_values:
                        for cooldown in cooldown_bars_values:
                            cfg = GateSweepConfig(
                                hold_bars=hold_bars,
                                entry_delay_bars=entry_delay_bars,
                                mae_penalty=mae_penalty,
                                mfe_bonus=mfe_bonus,
                                min_net_return=min_net_return,
                                min_utility=min_utility,
                                max_mae=max_mae,
                                cooldown_bars=cooldown,
                            )
                            train_rep = _simulate_oracle_strict(
                                train_rows,
                                train_market,
                                cfg,
                                fee_rate=fee_rate,
                                slippage_rate=slippage_rate,
                                leverage=leverage,
                            )
                            row = {"params": asdict(cfg), "train": train_rep}
                            if int(train_rep["sim"]["trade_entries"]) >= int(min_train_trades):
                                row["test"] = _simulate_oracle_strict(
                                    test_rows,
                                    test_market,
                                    cfg,
                                    fee_rate=fee_rate,
                                    slippage_rate=slippage_rate,
                                    leverage=leverage,
                                )
                                row["eval"] = _simulate_oracle_strict(
                                    eval_rows,
                                    eval_market,
                                    cfg,
                                    fee_rate=fee_rate,
                                    slippage_rate=slippage_rate,
                                    leverage=leverage,
                                )
                            candidates.append(row)

    eligible = [r for r in candidates if "test" in r]
    top = sorted(eligible, key=_rank_key, reverse=True)[:25]
    out = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "files": {"market_csv": str(Path(market_csv).resolve())},
        "setup": {
            "window_size": int(window_size),
            "hold_bars": int(hold_bars),
            "entry_delay_bars": int(entry_delay_bars),
            "fee_rate": float(fee_rate),
            "slippage_rate": float(slippage_rate),
            "leverage": float(leverage),
            "min_train_trades": int(min_train_trades),
        },
        "periods": {
            "train": [train_start, train_end],
            "test": [test_start, test_end],
            "eval": [eval_start, eval_end],
        },
        "search_summary": {
            "num_candidates": int(len(candidates)),
            "eligible_candidates": int(len(eligible)),
        },
        "top": top,
        "leakage_note": {
            "oracle_future_labels": True,
            "deployable_policy": False,
            "purpose": "upper-bound label-target search before LLM gate training",
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep path-outcome gate labels with oracle strict-bar upper bounds")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", default="results/path_outcome_gate_sweep.json")
    p.add_argument("--train-start", default="2023-01-01")
    p.add_argument("--train-end", default="2023-03-31")
    p.add_argument("--test-start", default="2023-04-01")
    p.add_argument("--test-end", default="2023-06-30")
    p.add_argument("--eval-start", default="2023-07-01")
    p.add_argument("--eval-end", default="2023-09-30")
    p.add_argument("--window-size", type=int, default=96)
    p.add_argument("--hold-bars", type=int, default=144)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--mae-penalty-values", default="0.5,1.0,1.5,2.0")
    p.add_argument("--mfe-bonus-values", default="0.0,0.25")
    p.add_argument("--min-net-return-values", default="0.0,0.001,0.002,0.003,0.005")
    p.add_argument("--min-utility-values", default="0.0,0.001,0.002,0.003")
    p.add_argument("--max-mae-values", default="0.01,0.015,0.02,0.03")
    p.add_argument("--cooldown-bars-values", default="0,6,12,24")
    p.add_argument("--min-train-trades", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = run_sweep(
        market_csv=args.market_csv,
        output=args.output,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        window_size=args.window_size,
        hold_bars=args.hold_bars,
        entry_delay_bars=args.entry_delay_bars,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        leverage=args.leverage,
        mae_penalty_values=_parse_float_list(args.mae_penalty_values),
        mfe_bonus_values=_parse_float_list(args.mfe_bonus_values),
        min_net_return_values=_parse_float_list(args.min_net_return_values),
        min_utility_values=_parse_float_list(args.min_utility_values),
        max_mae_values=_parse_float_list(args.max_mae_values),
        cooldown_bars_values=_parse_int_list(args.cooldown_bars_values),
        min_train_trades=args.min_train_trades,
    )
    preview = [
        {
            "params": row["params"],
            "train_sim": row["train"]["sim"],
            "test_sim": row.get("test", {}).get("sim"),
            "eval_sim": row.get("eval", {}).get("sim"),
        }
        for row in out["top"][:5]
    ]
    print(json.dumps({"search_summary": out["search_summary"], "top": preview, "leakage_note": out["leakage_note"]}, indent=2))


if __name__ == "__main__":
    main()
