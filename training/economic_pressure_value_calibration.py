"""Train-only cost-aware calibration for pressure analyzer trade decisions.

This module deliberately separates:
- analyzer: past-only compact/teacher/model pressure context
- trader: train-fitted table of realized stop/target returns by context and side

Validation may rank configurations, but OOS is evaluated only after selecting a fixed
configuration externally or by this script's val-first selection report.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

from training.economic_pressure_backtest import PressureBacktestConfig, run_pressure_backtest, strict_pressure_backtest
from training.strict_bar_backtest import load_market_bars

LABEL_TO_SIDE = {"LONG_FAVORED": "LONG", "SHORT_FAVORED": "SHORT"}
SIDE_TO_LABEL = {"LONG": "LONG_FAVORED", "SHORT": "SHORT_FAVORED", "NONE": "NO_TRADE_FAVORED"}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n")


def compact_features(row: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row.get("prompt", ""))
    marker = "Compact features: "
    start = prompt.find(marker)
    if start < 0:
        return {}
    start += len(marker)
    end = prompt.find("\n\n", start)
    raw = prompt[start:] if end < 0 else prompt[start:end]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def pred_pressure(row: dict[str, Any]) -> str:
    p = row.get("prediction", {})
    if isinstance(p, dict):
        return str(p.get("direction_pressure", "NO_TRADE_FAVORED"))
    try:
        return str(json.loads(str(p)).get("direction_pressure", "NO_TRADE_FAVORED"))
    except Exception:
        return "NO_TRADE_FAVORED"


def target_pressure(row: dict[str, Any]) -> str:
    try:
        return str(json.loads(str(row.get("target", "{}"))).get("direction_pressure", row.get("pressure", "NO_TRADE_FAVORED")))
    except Exception:
        return str(row.get("pressure", "NO_TRADE_FAVORED"))


def teacher(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("teacher", {}) if isinstance(row.get("teacher"), dict) else {}


def teacher_pressure(row: dict[str, Any]) -> str:
    return str(teacher(row).get("teacher_pressure", "NO_TRADE_FAVORED"))


def teacher_conf(row: dict[str, Any]) -> float:
    try:
        return float(teacher(row).get("teacher_confidence", 0.0) or 0.0)
    except Exception:
        return 0.0


def teacher_margin(row: dict[str, Any]) -> float:
    probs = teacher(row).get("teacher_probs", {}) if isinstance(teacher(row).get("teacher_probs"), dict) else {}
    vals = sorted((float(v) for v in probs.values()), reverse=True)
    if len(vals) < 2:
        return 0.0
    return vals[0] - vals[1]


def bucket(x: float, cuts: Iterable[float]) -> str:
    for c in cuts:
        if x < c:
            return f"lt{str(c).replace('.', 'p')}"
    return "hi"


def context_values(row: dict[str, Any]) -> dict[str, str]:
    cached = row.get("_context_values")
    if isinstance(cached, dict):
        return cached
    cf = compact_features(row)
    state = cf.get("state", {}) if isinstance(cf.get("state"), dict) else {}
    sym = cf.get("symbolic", {}) if isinstance(cf.get("symbolic"), dict) else {}
    evidence = cf.get("evidence", {}) if isinstance(cf.get("evidence"), dict) else {}
    seq = cf.get("sequence", {}) if isinstance(cf.get("sequence"), dict) else {}
    tags = cf.get("tags", []) if isinstance(cf.get("tags"), list) else []
    return {
        "teacher_pressure": teacher_pressure(row),
        "teacher_conf_bucket": bucket(teacher_conf(row), [0.34, 0.42, 0.50, 0.60]),
        "teacher_margin_bucket": bucket(teacher_margin(row), [0.05, 0.12, 0.20, 0.35]),
        "regime": str(state.get("regime", "NA")),
        "trend_alignment": str(state.get("trend_alignment", "NA")),
        "trend_strength": str(state.get("trend_strength", "NA")),
        "momentum": str(state.get("momentum", "NA")),
        "oscillator": str(state.get("oscillator", "NA")),
        "location": str(state.get("location", "NA")),
        "volatility_level": str(state.get("volatility_level", "NA")),
        "volume_state": str(state.get("volume_state", "NA")),
        "risk_state": str(state.get("risk_state", "NA")),
        "order_flow": str(sym.get("Order Flow", "NA")),
        "kimchi": str(sym.get("Korea Premium State", "NA")),
        "macro_dollar": str(sym.get("Macro Dollar State", "NA")),
        "range_pos_bucket": bucket(float(evidence.get("range_position", 0.0) or 0.0), [-0.6, -0.2, 0.2, 0.6]),
        "vol_z_bucket": bucket(float(evidence.get("volume_zscore", 0.0) or 0.0), [-1.0, 0.0, 1.0, 2.0]),
        "drawdown_bucket": bucket(float(evidence.get("window_drawdown_pct", 0.0) or 0.0), [0.3, 0.8, 1.5, 3.0]),
        "seq_bias": "up" if int(seq.get("rally_or_up", 0) or 0) > int(seq.get("drop_or_down", 0) or 0) else "down" if int(seq.get("drop_or_down", 0) or 0) > int(seq.get("rally_or_up", 0) or 0) else "flat",
        "tags": "+".join(str(t) for t in tags[:4]),
    }


def attach_context_cache(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        if not isinstance(row.get("_context_values"), dict):
            row["_context_values"] = context_values(row)
    return rows


KEY_LEVELS = {
    "coarse": ["teacher_pressure", "regime", "trend_alignment"],
    "state": ["teacher_pressure", "regime", "trend_alignment", "momentum", "location"],
    "micro": ["teacher_pressure", "regime", "trend_alignment", "order_flow", "oscillator", "volume_state"],
    "risk": ["teacher_pressure", "risk_state", "volatility_level", "drawdown_bucket", "range_pos_bucket"],
    "macro": ["teacher_pressure", "kimchi", "macro_dollar", "risk_state", "trend_alignment"],
    "teacher_only": ["teacher_pressure", "teacher_conf_bucket", "teacher_margin_bucket"],
}
FALLBACK_LEVELS = ["coarse", "teacher_only"]


def key_for(row: dict[str, Any], level: str, side: str) -> tuple[str, ...]:
    vals = context_values(row)
    return tuple([side, level] + [vals.get(k, "NA") for k in KEY_LEVELS[level]])


def simulate_trade_return(row: dict[str, Any], market, *, side: str, horizon_bars: int, target_pct: float, stop_pct: float, leverage: float, fee_rate: float, slippage_rate: float, entry_delay_bars: int) -> float | None:
    signal_pos = int(row.get("signal_pos", -1))
    entry_pos = signal_pos + max(0, int(entry_delay_bars))
    last_pos = entry_pos + max(1, int(horizon_bars))
    if entry_pos >= len(market) - 1 or last_pos >= len(market):
        return None
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    entry = float(opens[entry_pos])
    if entry <= 0:
        return None
    target = float(target_pct) / 100.0
    stop = float(stop_pct) / 100.0
    signal = 1 if side == "LONG" else -1
    gross = None
    for j in range(entry_pos, last_pos):
        if signal > 0:
            hit_target = (float(highs[j]) / entry - 1.0) >= target
            hit_stop = (float(lows[j]) / entry - 1.0) <= -stop
        else:
            hit_target = (entry / float(lows[j]) - 1.0) >= target if float(lows[j]) > 0 else False
            hit_stop = (entry / float(highs[j]) - 1.0) <= -stop if float(highs[j]) > 0 else False
        if hit_target or hit_stop:
            gross = -stop if hit_stop else target
            break
    if gross is None:
        exit_open = float(opens[last_pos])
        gross = (exit_open / entry - 1.0) if side == "LONG" else (entry / exit_open - 1.0)
    cost = 2.0 * (float(fee_rate) + float(slippage_rate)) * float(leverage)
    return float(leverage) * gross - cost


@dataclass(frozen=True)
class Stats:
    n: int
    mean: float
    std: float
    lower95: float


def make_stats(vals: list[float]) -> Stats:
    n = len(vals)
    mu = mean(vals) if vals else 0.0
    sd = pstdev(vals) if n > 1 else 0.0
    lower = mu - 1.96 * sd / math.sqrt(n) if n > 1 else mu
    return Stats(n=n, mean=mu, std=sd, lower95=lower)


def fit_tables(rows: list[dict[str, Any]], market, *, horizon_bars: int, target_pct: float, stop_pct: float, leverage: float, fee_rate: float, slippage_rate: float, entry_delay_bars: int) -> dict[tuple[str, ...], Stats]:
    buckets: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        for side in ["LONG", "SHORT"]:
            ret = simulate_trade_return(row, market, side=side, horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars)
            if ret is None:
                continue
            for level in KEY_LEVELS:
                buckets[key_for(row, level, side)].append(ret)
    return {k: make_stats(v) for k, v in buckets.items()}


def lookup_score(row: dict[str, Any], side: str, tables: dict[tuple[str, ...], Stats], *, level: str, min_n: int, score_mode: str) -> tuple[float, Stats | None, str | None]:
    for candidate in [level] + [x for x in FALLBACK_LEVELS if x != level]:
        st = tables.get(key_for(row, candidate, side))
        if st and st.n >= min_n:
            score = st.lower95 if score_mode == "lower95" else st.mean
            return score, st, candidate
    return -999.0, None, None


def choose_action(row: dict[str, Any], tables: dict[tuple[str, ...], Stats], *, level: str, min_n: int, min_score: float, score_mode: str, side_gate: str) -> tuple[str, dict[str, Any]]:
    allowed = ["LONG", "SHORT"]
    if side_gate == "model":
        allowed = [LABEL_TO_SIDE[pred_pressure(row)]] if pred_pressure(row) in LABEL_TO_SIDE else []
    elif side_gate == "teacher":
        allowed = [LABEL_TO_SIDE[teacher_pressure(row)]] if teacher_pressure(row) in LABEL_TO_SIDE else []
    scored = []
    for side in allowed:
        score, st, used = lookup_score(row, side, tables, level=level, min_n=min_n, score_mode=score_mode)
        scored.append((score, side, st, used))
    scored.sort(reverse=True, key=lambda x: x[0])
    if not scored or scored[0][0] < min_score:
        return "NONE", {"score": scored[0][0] if scored else None, "used_level": scored[0][3] if scored else None}
    score, side, st, used = scored[0]
    return side, {"score": score, "used_level": used, "bucket_n": st.n if st else 0, "bucket_mean": st.mean if st else None, "bucket_lower95": st.lower95 if st else None}


def write_policy_predictions(path: str | Path, rows: list[dict[str, Any]], tables: dict[tuple[str, ...], Stats], *, level: str, min_n: int, min_score: float, score_mode: str, side_gate: str) -> dict[str, Any]:
    out = []
    counts = defaultdict(int)
    for row in rows:
        r = dict(row)
        side, info = choose_action(r, tables, level=level, min_n=min_n, min_score=min_score, score_mode=score_mode, side_gate=side_gate)
        r["prediction"] = {"direction_pressure": SIDE_TO_LABEL[side], "policy_score": info.get("score"), "policy_level": info.get("used_level")}
        r["policy_calibration"] = info
        counts[side] += 1
        out.append(r)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n")
    return dict(counts)


def policy_rows(rows: list[dict[str, Any]], tables: dict[tuple[str, ...], Stats], *, level: str, min_n: int, min_score: float, score_mode: str, side_gate: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out = []
    counts = defaultdict(int)
    for row in rows:
        r = dict(row)
        side, info = choose_action(r, tables, level=level, min_n=min_n, min_score=min_score, score_mode=score_mode, side_gate=side_gate)
        r["prediction"] = {"direction_pressure": SIDE_TO_LABEL[side], "policy_score": info.get("score"), "policy_level": info.get("used_level")}
        r["policy_calibration"] = info
        counts[side] += 1
        out.append(r)
    return out, dict(counts)


def evaluate_config(rows: list[dict[str, Any]], tables: dict[tuple[str, ...], Stats], *, market, market_csv: str, prefix: str, split: str, level: str, min_n: int, min_score: float, score_mode: str, side_gate: str, horizon_bars: int, target_pct: float, stop_pct: float, leverage: float, fee_rate: float, slippage_rate: float, entry_delay_bars: int, write_artifacts: bool = False) -> dict[str, Any]:
    tag = f"{split}_{level}_n{min_n}_score{str(min_score).replace('-', 'm').replace('.', 'p')}_{score_mode}_{side_gate}"
    pred_path = f"{prefix}_{tag}_predictions.jsonl"
    bt_path = f"{prefix}_{tag}_backtest.json"
    rows2, action_counts = policy_rows(rows, tables, level=level, min_n=min_n, min_score=min_score, score_mode=score_mode, side_gate=side_gate)
    cfg = PressureBacktestConfig(horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars)
    bt_core = strict_pressure_backtest(rows2, market, cfg)
    result = {"config": {"level": level, "min_n": min_n, "min_score": min_score, "score_mode": score_mode, "side_gate": side_gate}, "prediction_rows": pred_path if write_artifacts else None, "backtest_path": bt_path if write_artifacts else None, "action_counts": action_counts, "sim": bt_core["sim"], "trade_stats": bt_core["trade_stats"]}
    if write_artifacts:
        write_jsonl(pred_path, rows2)
        run_pressure_backtest(predictions_jsonl=pred_path, market_csv=market_csv, output=bt_path, horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars)
    return result


def run_calibration(*, train_jsonl: str, val_predictions_jsonl: str, oos_predictions_jsonl: str, market_csv: str, output: str, prefix: str, horizon_bars: int = 36, target_pct: float = 0.5, stop_pct: float = 0.6, leverage: float = 0.5, fee_rate: float = 0.0004, slippage_rate: float = 0.0001, entry_delay_bars: int = 1, min_trades: int = 50) -> dict[str, Any]:
    market = load_market_bars(market_csv)
    train_rows = attach_context_cache(load_jsonl(train_jsonl))
    val_rows = attach_context_cache(load_jsonl(val_predictions_jsonl))
    oos_rows = attach_context_cache(load_jsonl(oos_predictions_jsonl))
    tables = fit_tables(train_rows, market, horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars)
    configs = []
    for level in KEY_LEVELS:
        for min_n in [10, 20, 35, 50, 80, 120]:
            for min_score in [-0.0005, 0.0, 0.0002, 0.0005, 0.0010, 0.0015]:
                for score_mode in ["mean", "lower95"]:
                    for side_gate in ["free", "model", "teacher"]:
                        configs.append((level, min_n, min_score, score_mode, side_gate))
    val_results = []
    for level, min_n, min_score, score_mode, side_gate in configs:
        val_results.append(evaluate_config(val_rows, tables, market=market, market_csv=market_csv, prefix=prefix, split="val", level=level, min_n=min_n, min_score=min_score, score_mode=score_mode, side_gate=side_gate, horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars))
    eligible = [r for r in val_results if r["sim"]["trade_entries"] >= min_trades]
    ranked = sorted(eligible or val_results, key=lambda r: (r["sim"]["cagr_to_strict_mdd"], r["sim"]["cagr_pct"], r["sim"]["trade_entries"]), reverse=True)
    selected = ranked[0]
    cfg = selected["config"]
    selected = evaluate_config(val_rows, tables, market=market, market_csv=market_csv, prefix=prefix, split="selected_val", horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars, write_artifacts=True, **cfg)
    oos = evaluate_config(oos_rows, tables, market=market, market_csv=market_csv, prefix=prefix, split="oos", horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars, write_artifacts=True, **cfg)
    report = {"train_jsonl": train_jsonl, "val_predictions_jsonl": val_predictions_jsonl, "oos_predictions_jsonl": oos_predictions_jsonl, "table_count": len(tables), "selection_rule": f"rank val configs with trade_entries >= {min_trades}; evaluate selected once on OOS", "selected_val": selected, "selected_oos": oos, "top_val": ranked[:20], "leakage_guard": {"calibration_fit_split": "train only", "config_selection_split": "validation only", "oos_used_for_selection": False, "trade_returns_computed_from_market_after_entry_only": True}}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-predictions-jsonl", required=True)
    p.add_argument("--oos-predictions-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--horizon-bars", type=int, default=36)
    p.add_argument("--target-pct", type=float, default=0.5)
    p.add_argument("--stop-pct", type=float, default=0.6)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--min-trades", type=int, default=50)
    return p.parse_args()


def main() -> None:
    report = run_calibration(**vars(parse_args()))
    print(json.dumps({"selected_val": report["selected_val"], "selected_oos": report["selected_oos"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
