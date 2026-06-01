"""Strict bar-by-bar backtest for fixed hierarchical LLM trading policies.

This evaluator is intentionally more conservative than the legacy h-horizon
policy simulators.  It does not apply a precomputed forward return.  Instead,
it enters on market bars, marks equity through each held 5-minute bar, includes
intrabar adverse excursions in strict MDD, and only uses policy/regime inputs
known at or before the signal timestamp.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from training.hierarchical_direct_split_search import HierSimConfig, _load_rows, _norm_cdf, _pair_rows
from training.hierarchical_regime_filter_search import (
    RegimeFilter,
    _attach_features,
    _load_market_features,
    _regime_ok,
    _row_signal,
)


@dataclass(frozen=True)
class BarExecutionConfig:
    leverage: float
    fee_rate: float
    slippage_rate: float
    drawdown_stop: float
    pause_bars: int
    monthly_loss_stop: float
    entry_delay_bars: int = 1


def _split_paths(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def load_policy_rows(gate_files: str, side_files: str, features: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    gates = _split_paths(gate_files)
    sides = _split_paths(side_files)
    if len(gates) != len(sides):
        raise ValueError("gate and side file counts must match")
    rows: list[dict[str, Any]] = []
    for gate, side in zip(gates, sides):
        rows.extend(_pair_rows(_load_rows(gate), _load_rows(side)))
    return _attach_features(sorted(rows, key=lambda r: str(r["date"])), features)


def load_market_bars(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"market csv lacks required columns: {sorted(missing)}")
    df = df[["date", "open", "high", "low", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    return df


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


def _drawdown_from_trough(peak: float, trough_eq: float) -> float:
    if peak <= 0.0:
        return 0.0
    return max(0.0, 1.0 - max(0.0, trough_eq) / peak)


def simulate_bar_by_bar(
    rows: list[dict[str, Any]],
    market: pd.DataFrame,
    cfg: HierSimConfig,
    filt: RegimeFilter,
    exec_cfg: BarExecutionConfig,
) -> dict[str, Any]:
    """Run fixed-policy strict MDD simulation on actual OHLC bars.

    Entry uses ``entry_delay_bars`` after the signal timestamp by default, so a
    signal produced from bar t is filled at bar t+1 open.  While held, equity is
    marked open-to-open and strict MDD includes each bar's adverse high/low move
    against the current position before the scheduled exit.
    """

    if not rows:
        raise ValueError("rows must not be empty")
    if len(market) < 2:
        raise ValueError("market must contain at least two bars")

    date_to_pos = {ts.to_pydatetime().replace(tzinfo=None): int(i) for i, ts in enumerate(market["date"])}
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)

    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    trade_returns: list[float] = []
    entries = 0
    skipped_missing_bars = 0
    forced_liquidations = 0
    next_allowed_market_pos = 0
    paused_until_market_pos = -1
    month_key: str | None = None
    month_start_eq = 1.0
    month_paused = False
    cost = (float(exec_cfg.fee_rate) + float(exec_cfg.slippage_rate)) * float(exec_cfg.leverage)
    hold_bars = max(1, int(cfg.hold_bars))
    cooldown_bars = max(0, int(cfg.cooldown_bars))
    entry_delay = max(0, int(exec_cfg.entry_delay_bars))

    for row in rows:
        dt = datetime.fromisoformat(str(row["date"]))
        pos = date_to_pos.get(dt.replace(tzinfo=None))
        if pos is None:
            skipped_missing_bars += 1
            continue
        current_month = f"{dt.year}-{dt.month:02d}"
        if current_month != month_key:
            month_key = current_month
            month_start_eq = eq
            month_paused = False

        if pos < next_allowed_market_pos or pos < paused_until_market_pos or month_paused:
            continue

        signal = _row_signal(row, cfg)
        if signal == 0 or not _regime_ok(row, cfg, filt):
            continue

        entry_pos = pos + entry_delay
        exit_pos = entry_pos + hold_bars
        if entry_pos >= len(market) - 1 or exit_pos >= len(market):
            skipped_missing_bars += 1
            continue

        entry_eq = eq
        entries += 1
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))

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
            adverse_eq = eq * (1.0 + float(exec_cfg.leverage) * adverse_ret)
            max_dd = max(max_dd, _drawdown_from_trough(peak, adverse_eq))
            eq *= max(0.0, 1.0 + float(exec_cfg.leverage) * close_ret)
            peak = max(peak, eq)
            if eq <= 0.0:
                forced_liquidations += 1
                break

        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        peak = max(peak, eq)
        trade_returns.append(eq / entry_eq - 1.0)

        current_dd = 1.0 - eq / peak if peak > 0.0 else 0.0
        if exec_cfg.drawdown_stop < 1.0 and current_dd >= exec_cfg.drawdown_stop:
            paused_until_market_pos = exit_pos + max(1, int(exec_cfg.pause_bars))
        if exec_cfg.monthly_loss_stop < 1.0 and eq / month_start_eq - 1.0 <= -float(exec_cfg.monthly_loss_stop):
            month_paused = True
        next_allowed_market_pos = exit_pos + cooldown_bars
        if eq <= 0.0:
            break

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
            "skipped_missing_bars": skipped_missing_bars,
            "forced_liquidations": forced_liquidations,
            "entry_delay_bars": entry_delay,
            "return_application": "actual_ohlc_bar_by_bar_strict_mdd",
        },
        "trade_stats": _trade_stats(trade_returns),
    }


def _selection_to_args(selection: dict[str, Any], top_index: int) -> dict[str, Any]:
    top = selection.get("top") or []
    if not top:
        raise ValueError("selection file has no top candidates")
    row = top[top_index]
    files = selection.get("files") or {}
    policy = row.get("policy") or {}
    hierarchical = policy.get("hierarchical") or row.get("params") or {}
    regime_filter = policy.get("regime_filter") or row.get("regime_filter") or {"name": "none"}
    overlay = row.get("overlay") or {}
    return {"files": files, "hierarchical": hierarchical, "regime_filter": regime_filter, "overlay": overlay}


def run_from_selection(args: argparse.Namespace) -> dict[str, Any]:
    selection = json.loads(Path(args.selection_file).read_text())
    selected = _selection_to_args(selection, args.top_index)
    files = selected["files"]
    features = _load_market_features(files.get("market_csv") or args.market_csv)
    market = load_market_bars(files.get("market_csv") or args.market_csv)
    cfg = HierSimConfig(**selected["hierarchical"])
    filt = RegimeFilter(**selected["regime_filter"])
    overlay = selected["overlay"]
    exec_cfg = BarExecutionConfig(
        leverage=float(overlay.get("leverage", args.leverage)),
        fee_rate=float(files.get("fee_rate", args.fee_rate)),
        slippage_rate=float(files.get("slippage_rate", args.slippage_rate)),
        drawdown_stop=float(overlay.get("drawdown_stop", args.drawdown_stop)),
        pause_bars=int(overlay.get("pause_bars", args.pause_bars)),
        monthly_loss_stop=float(overlay.get("monthly_loss_stop", args.monthly_loss_stop)),
        entry_delay_bars=int(args.entry_delay_bars),
    )

    split_specs = {
        "train": (files.get("train_gate_file"), files.get("train_side_file")),
        "val": (files.get("val_gate_file"), files.get("val_side_file")),
        "oos": (files.get("oos_gate_file"), files.get("oos_side_file")),
    }
    out: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "selection_file": args.selection_file,
        "top_index": args.top_index,
        "policy": {"hierarchical": cfg.__dict__, "regime_filter": filt.__dict__},
        "execution": exec_cfg.__dict__,
        "source_files": files,
        "splits": {},
        "leakage_guard": {
            "uses_forward_return_column": False,
            "entry_after_signal_by_bars": int(args.entry_delay_bars),
            "features_are_past_only": True,
            "fixed_policy_loaded_from_selection": True,
        },
    }
    all_rows: list[dict[str, Any]] = []
    for split, (gate, side) in split_specs.items():
        if not gate or not side:
            continue
        rows = load_policy_rows(gate, side, features)
        all_rows.extend(rows)
        out["splits"][split] = simulate_bar_by_bar(rows, market, cfg, filt, exec_cfg)
    if all_rows:
        out["splits"]["all"] = simulate_bar_by_bar(sorted(all_rows, key=lambda r: str(r["date"])), market, cfg, filt, exec_cfg)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict OHLC bar-by-bar evaluator for selected hierarchical LLM policies")
    p.add_argument("--selection-file", default="results/h144_candidate_trainselected_risk_overlay_search.json")
    p.add_argument("--top-index", type=int, default=0)
    p.add_argument("--market-csv", default="")
    p.add_argument("--output", default="results/h144_candidate_bar_by_bar_strict_eval.json")
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--drawdown-stop", type=float, default=1.0)
    p.add_argument("--pause-bars", type=int, default=864)
    p.add_argument("--monthly-loss-stop", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    out = run_from_selection(parse_args())
    summary = {split: payload["sim"] for split, payload in out["splits"].items()}
    print(json.dumps({"summary": summary, "leakage_guard": out["leakage_guard"]}, indent=2))


if __name__ == "__main__":
    main()
