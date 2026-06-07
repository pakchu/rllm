"""Val-selected action subset sweep for generated trader prediction JSONL.

This is a post-model risk/action filter: it may select allowed generated sides
and holds on validation predictions, then applies the frozen filter to OOS
predictions.  It never reads future returns while choosing from OOS.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from training.economic_action_backtest import EconomicActionBacktestConfig, load_prediction_rows, strict_backtest_actions
from training.eval_text_trader import parse_trader_json
from training.strict_bar_backtest import load_market_bars


@dataclass(frozen=True)
class ActionFilterConfig:
    sides: tuple[str, ...]
    holds: tuple[int, ...]
    cooldown_bars: int = 0


def _csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _action_key(row: dict[str, Any]) -> tuple[str, int]:
    action = parse_trader_json(json.dumps(row.get("prediction", {})))
    return str(action.get("side", "NONE")), int(action.get("hold_bars", 0) or 0)


def observed_actions(rows: list[dict[str, Any]]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    sides = sorted({s for s, h in (_action_key(r) for r in rows) if s in {"LONG", "SHORT"}})
    holds = sorted({h for s, h in (_action_key(r) for r in rows) if h > 0})
    return tuple(sides), tuple(holds)


def apply_filter(rows: list[dict[str, Any]], cfg: ActionFilterConfig) -> list[dict[str, Any]]:
    allowed_sides = set(cfg.sides)
    allowed_holds = {int(h) for h in cfg.holds}
    out: list[dict[str, Any]] = []
    for row in rows:
        side, hold = _action_key(row)
        new_row = dict(row)
        if side not in allowed_sides or int(hold) not in allowed_holds:
            new_row["prediction"] = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}
        out.append(new_row)
    return out


def run_filtered_backtest(rows: list[dict[str, Any]], market, cfg: ActionFilterConfig, *, leverage: float, fee_rate: float, slippage_rate: float, entry_delay_bars: int, max_hold_bars: int) -> dict[str, Any]:
    filtered = apply_filter(rows, cfg)
    bt_cfg = EconomicActionBacktestConfig(
        leverage=float(leverage),
        fee_rate=float(fee_rate),
        slippage_rate=float(slippage_rate),
        entry_delay_bars=int(entry_delay_bars),
        cooldown_bars=int(cfg.cooldown_bars),
        max_hold_bars=int(max_hold_bars),
    )
    return strict_backtest_actions(filtered, market, bt_cfg)


def powerset_nonempty(items: tuple[Any, ...]) -> list[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = []
    for n in range(1, len(items) + 1):
        out.extend(tuple(x) for x in itertools.combinations(items, n))
    return out


def objective(sim: dict[str, Any], *, min_trades: int, max_mdd: float) -> float:
    s = sim["sim"]
    if int(s.get("trade_entries", 0)) < int(min_trades):
        return -1e9
    if float(s.get("strict_mdd_pct", 0.0)) > float(max_mdd):
        return -1e9
    return float(s.get("cagr_to_strict_mdd", -1e9))


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    val_rows = load_prediction_rows(args.val_predictions)
    oos_rows = load_prediction_rows(args.oos_predictions)
    market = load_market_bars(args.market_csv)
    sides, holds = observed_actions(val_rows + oos_rows)
    if args.sides:
        sides = tuple(x.strip().upper() for x in args.sides.split(",") if x.strip())
    if args.holds:
        holds = tuple(_csv_ints(args.holds))
    candidates: list[dict[str, Any]] = []
    for side_set, hold_set, cooldown in itertools.product(powerset_nonempty(sides), powerset_nonempty(holds), _csv_ints(args.cooldown_bars_list)):
        cfg = ActionFilterConfig(sides=tuple(side_set), holds=tuple(int(h) for h in hold_set), cooldown_bars=int(cooldown))
        val_bt = run_filtered_backtest(
            val_rows,
            market,
            cfg,
            leverage=args.leverage,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            entry_delay_bars=args.entry_delay_bars,
            max_hold_bars=args.max_hold_bars,
        )
        score = objective(val_bt, min_trades=args.min_trades, max_mdd=args.max_mdd)
        candidates.append({"config": asdict(cfg), "score": score, "val": val_bt})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    selected = candidates[0]
    cfg = ActionFilterConfig(**selected["config"])
    oos_bt = run_filtered_backtest(
        oos_rows,
        market,
        cfg,
        leverage=args.leverage,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        entry_delay_bars=args.entry_delay_bars,
        max_hold_bars=args.max_hold_bars,
    )
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "val_predictions": str(Path(args.val_predictions).resolve()),
            "oos_predictions": str(Path(args.oos_predictions).resolve()),
            "market_csv": str(Path(args.market_csv).resolve()),
        },
        "selection": {
            "selected_on": "val_only",
            "objective": "max cagr_to_strict_mdd subject to min_trades and max_mdd",
            "min_trades": int(args.min_trades),
            "max_mdd": float(args.max_mdd),
            "observed_sides": list(sides),
            "observed_holds": list(holds),
        },
        "selected": {"config": selected["config"], "val": selected["val"], "oos": oos_bt},
        "top_val": candidates[: int(args.top_k)],
        "leakage_guard": {
            "filter_selected_on_val_only": True,
            "oos_not_used_for_selection": True,
            "uses_model_predictions_only": True,
            "uses_forward_return_column": False,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep side/hold filters on generated action predictions")
    p.add_argument("--val-predictions", required=True)
    p.add_argument("--oos-predictions", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", default="results/prediction_action_filter_sweep.json")
    p.add_argument("--sides", default="")
    p.add_argument("--holds", default="")
    p.add_argument("--cooldown-bars-list", default="0,12,36")
    p.add_argument("--min-trades", type=int, default=30)
    p.add_argument("--max-mdd", type=float, default=15.0)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--max-hold-bars", type=int, default=432)
    p.add_argument("--top-k", type=int, default=10)
    return p.parse_args()


def main() -> None:
    report = run_sweep(parse_args())
    selected = report["selected"]
    print(json.dumps({"config": selected["config"], "val": selected["val"]["sim"], "oos": selected["oos"]["sim"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
