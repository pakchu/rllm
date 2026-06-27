"""Sweep causal focus-score thresholds and strict-backtest the resulting policies."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from training.eval_single_policy import policy_to_action
from training.focus_score_policy import PATH_OPTIONS, TARGET_PATH, TARGET_UTILITY, UTILITY_OPTIONS, _margin, _softmax
from training.single_policy_sft_data import exit_profile_for_hold
from training.strict_bar_backtest import BarExecutionConfig, _drawdown_from_trough, _trade_stats, load_market_bars


@dataclass(frozen=True)
class FocusScoreSweepCfg:
    focus_predictions_jsonl: str
    market_csv: str
    output_json: str
    clean_probs: str = "0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.45"
    high_probs: str = "0.0,0.2,0.3,0.333,0.4,0.5,0.6"
    clean_margins: str = "-999,0"
    high_margins: str = "-999,0"
    min_trades: int = 1
    top_n: int = 30
    entry_delay_bars: int = 1
    cooldown_bars: int = 0
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001


def _floats(csv: str) -> list[float]:
    return [float(x.strip()) for x in str(csv).split(",") if x.strip()]


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _score_diag(row: dict[str, Any]) -> dict[str, Any]:
    scores = row.get("focus_scores") or row.get("scores") or {}
    path_scores = {k: float(v) for k, v in dict(scores.get("path_shape") or {}).items()}
    utility_scores = {k: float(v) for k, v in dict(scores.get("utility_bucket") or {}).items()}
    path_probs = _softmax(path_scores, PATH_OPTIONS)
    utility_probs = _softmax(utility_scores, UTILITY_OPTIONS)
    return {
        "clean_prob": float(path_probs.get(TARGET_PATH, 0.0)),
        "high_prob": float(utility_probs.get(TARGET_UTILITY, 0.0)),
        "clean_margin": _margin(path_scores, TARGET_PATH),
        "high_margin": _margin(utility_scores, TARGET_UTILITY),
        "has_scores": bool(path_scores and utility_scores),
    }


def _policy(row: dict[str, Any], trade: bool) -> dict[str, str]:
    cand = dict(row.get("candidate") or {})
    side = str(cand.get("side", "")).upper()
    horizon = int(cand.get("horizon", 288) or 288)
    if trade and side in {"LONG", "SHORT"}:
        return {
            "regime": "TREND_UP" if side == "LONG" else "TREND_DOWN",
            "edge_quality": "STRONG",
            "risk": "LOW",
            "action": side,
            "exit_profile": exit_profile_for_hold(horizon),
            "confidence": "HIGH",
        }
    return {
        "regime": "RANGE",
        "edge_quality": "NONE",
        "risk": "LOW",
        "action": "NO_TRADE",
        "exit_profile": "AVOID",
        "confidence": "LOW",
    }


def _passes(diag: dict[str, Any], cp: float, hp: float, cm: float, hm: float) -> bool:
    if not diag["has_scores"]:
        return False
    clean_margin = float(diag["clean_margin"]) if math.isfinite(float(diag["clean_margin"])) else -999999.0
    high_margin = float(diag["high_margin"]) if math.isfinite(float(diag["high_margin"])) else -999999.0
    return (
        float(diag["clean_prob"]) >= cp
        and float(diag["high_prob"]) >= hp
        and clean_margin >= cm
        and high_margin >= hm
    )


def _simulate_fast(
    rows: list[dict[str, Any]],
    policies: list[dict[str, str]],
    market: Any,
    exec_cfg: BarExecutionConfig,
    cooldown_bars: int,
) -> dict[str, Any]:
    date_to_pos = {ts.to_pydatetime().replace(tzinfo=None): int(i) for i, ts in enumerate(market["date"])}
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    ordered = sorted(zip(rows, policies), key=lambda rp: str(rp[0].get("date", "")))

    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    trade_returns: list[float] = []
    entries = 0
    skipped_missing_bars = 0
    next_allowed_market_pos = 0
    cost = (float(exec_cfg.fee_rate) + float(exec_cfg.slippage_rate)) * float(exec_cfg.leverage)

    for row, policy in ordered:
        dt = datetime.fromisoformat(str(row["date"]))
        pos = date_to_pos.get(dt.replace(tzinfo=None))
        if pos is None:
            skipped_missing_bars += 1
            continue
        if pos < next_allowed_market_pos:
            continue
        action = policy_to_action(policy)
        side = str(action.get("side", "NONE"))
        hold_bars = int(action.get("hold_bars", 0) or 0)
        signal = 1 if side == "LONG" else -1 if side == "SHORT" else 0
        if signal == 0 or hold_bars <= 0:
            continue
        entry_pos = pos + int(exec_cfg.entry_delay_bars)
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
                break
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        peak = max(peak, eq)
        trade_returns.append(eq / entry_eq - 1.0)
        next_allowed_market_pos = exit_pos + max(0, int(cooldown_bars))
        if eq <= 0.0:
            break

    sorted_rows = [r for r, _ in ordered]
    start_dt = datetime.fromisoformat(str(sorted_rows[0]["date"])) if sorted_rows else datetime.now()
    end_dt = datetime.fromisoformat(str(sorted_rows[-1]["date"])) if sorted_rows else start_dt
    years = max(1.0 / 365.25, float((end_dt - start_dt).days) / 365.25)
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr_pct = ((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0.0 else -100.0
    mdd_pct = max_dd * 100.0
    return {
        "period": {
            "start": str(sorted_rows[0].get("date")) if sorted_rows else None,
            "end": str(sorted_rows[-1].get("date")) if sorted_rows else None,
            "years": years,
        },
        "sim": {
            "ret_pct": ret_pct,
            "cagr_pct": cagr_pct,
            "strict_mdd_pct": mdd_pct,
            "cagr_to_strict_mdd": cagr_pct / mdd_pct if mdd_pct > 1e-12 else float("inf"),
            "trade_entries": entries,
            "turnover_legs": entries * 2,
            "samples": len(sorted_rows),
            "skipped_missing_bars": skipped_missing_bars,
            "entry_delay_bars": int(exec_cfg.entry_delay_bars),
            "return_application": "actual_ohlc_bar_by_bar_variable_hold_strict_mdd",
            "target_echo_oracle_mode": False,
        },
        "trade_stats": _trade_stats(trade_returns),
    }


def run(cfg: FocusScoreSweepCfg) -> dict[str, Any]:
    rows = _load_jsonl(cfg.focus_predictions_jsonl)
    diags = [_score_diag(r) for r in rows]
    market = load_market_bars(cfg.market_csv)
    exec_cfg = BarExecutionConfig(
        leverage=float(cfg.leverage),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
        drawdown_stop=1.0,
        pause_bars=0,
        monthly_loss_stop=1.0,
        entry_delay_bars=int(cfg.entry_delay_bars),
    )
    results = []
    total_grid = 0
    for cp in _floats(cfg.clean_probs):
        for hp in _floats(cfg.high_probs):
            for cm in _floats(cfg.clean_margins):
                for hm in _floats(cfg.high_margins):
                    total_grid += 1
                    policies = [_policy(row, _passes(diag, cp, hp, cm, hm)) for row, diag in zip(rows, diags)]
                    action_counts = Counter(p["action"] for p in policies)
                    if (len(rows) - action_counts.get("NO_TRADE", 0)) < int(cfg.min_trades):
                        continue
                    sim_result = _simulate_fast(rows, policies, market, exec_cfg, int(cfg.cooldown_bars))
                    sim = sim_result["sim"]
                    if int(sim["trade_entries"]) < int(cfg.min_trades):
                        continue
                    results.append(
                        {
                            "thresholds": {
                                "min_clean_prob": cp,
                                "min_high_prob": hp,
                                "min_clean_margin": cm,
                                "min_high_margin": hm,
                            },
                            "actions": dict(action_counts),
                            "sim": sim,
                            "trade_stats": sim_result["trade_stats"],
                        }
                    )
    results.sort(
        key=lambda r: (
            float(r["sim"]["cagr_to_strict_mdd"]),
            float(r["sim"]["cagr_pct"]),
            int(r["sim"]["trade_entries"]),
        ),
        reverse=True,
    )
    report = {
        "config": asdict(cfg),
        "rows": len(rows),
        "grid_size": total_grid,
        "tested": len(results),
        "top": results[: int(cfg.top_n)],
    }
    Path(cfg.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--focus-predictions-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--clean-probs", default=FocusScoreSweepCfg.clean_probs)
    p.add_argument("--high-probs", default=FocusScoreSweepCfg.high_probs)
    p.add_argument("--clean-margins", default=FocusScoreSweepCfg.clean_margins)
    p.add_argument("--high-margins", default=FocusScoreSweepCfg.high_margins)
    p.add_argument("--min-trades", type=int, default=FocusScoreSweepCfg.min_trades)
    p.add_argument("--top-n", type=int, default=FocusScoreSweepCfg.top_n)
    p.add_argument("--entry-delay-bars", type=int, default=FocusScoreSweepCfg.entry_delay_bars)
    p.add_argument("--cooldown-bars", type=int, default=FocusScoreSweepCfg.cooldown_bars)
    p.add_argument("--leverage", type=float, default=FocusScoreSweepCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=FocusScoreSweepCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=FocusScoreSweepCfg.slippage_rate)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(FocusScoreSweepCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
