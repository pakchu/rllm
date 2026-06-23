"""Nested selection for score-geometry side-map transforms.

This is a stricter follow-up to month-level inversion: it can pass, invert, or
block an existing generated trade using only prior month validation health and
current prediction-score geometry (`score_long`, `score_short`, `score_wait`).
Candidate transform/overlay configs are selected on one period and replayed on
an untouched eval period.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

NO_TRADE = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "SCORE_GEOMETRY_TRANSFORM", "confidence": "HIGH"}
ACTIONS = {
    "pass",
    "invert",
    "block",
    "invert_low_gap",
    "invert_high_gap",
    "block_low_gap",
    "block_high_gap",
    "invert_low_gap_block_high",
    "invert_high_gap_block_low",
}


@dataclass(frozen=True)
class NestedScoreGeometryTransformCfg:
    predictions_jsonl: str
    rolling_summary_json: str
    market_csv: str
    output: str
    work_dir: str
    selection_start: str = "2024-01"
    selection_end: str = "2025-12"
    eval_start: str = "2026-01"
    eval_end: str = "2026-05"
    thresholds: str = "-1000,-500,-100,0,0.5,1,2,3,5"
    below_actions: str = "pass,invert,block,invert_low_gap,invert_high_gap,block_low_gap,block_high_gap"
    above_actions: str = "pass,invert,block_low_gap,block_high_gap"
    score_gaps: str = "0,0.025,0.05,0.10,0.20"
    stop_losses: str = "0,2,3,4"
    take_profits: str = "0,2,3,4,6"
    rolling_loss_specs: str = "0:0,10:5,20:8"
    min_selection_trades: int = 30
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    pause_bars: int = 864
    top_k: int = 20
    keep_artifacts: bool = False


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _actions(raw: str) -> list[str]:
    vals = [x.strip().lower() for x in str(raw).split(",") if x.strip()]
    bad = sorted(set(vals) - ACTIONS)
    if bad:
        raise ValueError(f"unknown action(s): {bad}; expected one of {sorted(ACTIONS)}")
    return vals


def _loss_specs(raw: str) -> list[tuple[int, float]]:
    out = []
    for part in str(raw).split(","):
        if not part.strip():
            continue
        n, loss = part.split(":", 1)
        out.append((int(n), float(loss)))
    return out


def _month_scores(summary: dict[str, Any]) -> dict[str, float]:
    out = {}
    for m in summary.get("months", []):
        try:
            out[str(m.get("month"))] = float((m.get("selected") or {}).get("score", float("-inf")))
        except Exception:
            out[str(m.get("month"))] = float("-inf")
    return out


def _period_rows(rows: list[dict[str, Any]], start_month: str, end_month: str) -> list[dict[str, Any]]:
    return [r for r in rows if start_month <= str(r.get("date", ""))[:7] <= end_month]


def _side_gap(row: dict[str, Any]) -> float:
    try:
        return abs(float(row.get("score_long", 0.0) or 0.0) - float(row.get("score_short", 0.0) or 0.0))
    except Exception:
        return 0.0


def _invert_prediction(pred: dict[str, Any]) -> dict[str, Any]:
    out = dict(pred)
    if out.get("gate") == "TRADE":
        side = str(out.get("side", "NONE")).upper()
        if side == "LONG":
            out["side"] = "SHORT"
        elif side == "SHORT":
            out["side"] = "LONG"
        out["family"] = str(out.get("family", "")) + "+SCORE_INV"
    return out


def _resolve_action(action: str, gap: float, threshold: float) -> str:
    action = str(action).lower()
    if action in {"pass", "invert", "block"}:
        return action
    if action == "invert_low_gap":
        return "invert" if gap <= threshold else "pass"
    if action == "invert_high_gap":
        return "invert" if gap >= threshold else "pass"
    if action == "block_low_gap":
        return "block" if gap <= threshold else "pass"
    if action == "block_high_gap":
        return "block" if gap >= threshold else "pass"
    if action == "invert_low_gap_block_high":
        return "invert" if gap <= threshold else "block"
    if action == "invert_high_gap_block_low":
        return "invert" if gap >= threshold else "block"
    raise ValueError(f"unknown action: {action}")


def _apply_transform(rows: list[dict[str, Any]], scores: dict[str, float], month_threshold: float, below_action: str, above_action: str, score_gap: float) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts = {"below_rows": 0, "above_rows": 0, "passed_trades": 0, "inverted_trades": 0, "blocked_trades": 0}
    out: list[dict[str, Any]] = []
    for row in rows:
        month = str(row.get("date", ""))[:7]
        month_score = float(scores.get(month, float("-inf")))
        raw_action = below_action if month_score < float(month_threshold) else above_action
        if month_score < float(month_threshold):
            counts["below_rows"] += 1
        else:
            counts["above_rows"] += 1
        pred = dict(row.get("prediction", {})) if isinstance(row.get("prediction"), dict) else {}
        was_trade = pred.get("gate") == "TRADE"
        action = _resolve_action(raw_action, _side_gap(row), float(score_gap))
        nr = dict(row)
        nr["month_validation_score"] = month_score
        nr["score_geometry_action"] = action
        nr["score_geometry_gap"] = _side_gap(row)
        if action == "block":
            nr["prediction"] = dict(NO_TRADE)
            if was_trade:
                counts["blocked_trades"] += 1
        elif action == "invert":
            nr["prediction"] = _invert_prediction(pred)
            if was_trade:
                counts["inverted_trades"] += 1
        else:
            nr["prediction"] = pred
            if was_trade:
                counts["passed_trades"] += 1
        out.append(nr)
    return out, counts


def _score_selection(bt: dict[str, Any], min_trades: int) -> float:
    sim = bt.get("sim", {})
    stats = bt.get("trade_stats", {})
    trades = int(sim.get("trade_entries", 0) or 0)
    if trades < int(min_trades):
        return -999.0 + trades / 1000.0
    cagr = float(sim.get("cagr_pct", -100.0))
    mdd = float(sim.get("strict_mdd_pct", 999.0))
    ratio = float(sim.get("cagr_to_strict_mdd", -999.0))
    p = float(stats.get("p_value_mean_ret_approx", 1.0))
    if cagr <= 0:
        return -500.0 + trades / 1000.0 + cagr / 100.0
    return ratio + 0.01 * cagr - 0.05 * max(0.0, mdd - 15.0) - p + min(1.0, trades / 100.0)


def _run_bt(rows: list[dict[str, Any]], cfg: NestedScoreGeometryTransformCfg, path: Path, *, stop: float, take: float, rolling_n: int, rolling_loss: float) -> dict[str, Any]:
    pred = path.with_suffix(".jsonl")
    out = path.with_suffix(".bt.json")
    _write_jsonl(pred, rows)
    result = run_overlay(OnlineRiskOverlayConfig(
        predictions_jsonl=str(pred),
        market_csv=cfg.market_csv,
        output=str(out),
        leverage=float(cfg.leverage),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
        entry_delay_bars=int(cfg.entry_delay_bars),
        trade_stop_loss_pct=float(stop),
        trade_take_profit_pct=float(take),
        rolling_window_trades=int(rolling_n),
        rolling_loss_stop_pct=float(rolling_loss),
        pause_bars=int(cfg.pause_bars),
    ))
    if not bool(cfg.keep_artifacts):
        pred.unlink(missing_ok=True)
        out.unlink(missing_ok=True)
    return result


def run(cfg: NestedScoreGeometryTransformCfg) -> dict[str, Any]:
    rows = _read_jsonl(cfg.predictions_jsonl)
    summary = json.loads(Path(cfg.rolling_summary_json).read_text())
    scores = _month_scores(summary)
    select_base = _period_rows(rows, cfg.selection_start, cfg.selection_end)
    eval_base = _period_rows(rows, cfg.eval_start, cfg.eval_end)
    work = Path(cfg.work_dir); work.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    i = 0
    for threshold in _floats(cfg.thresholds):
        for below_action in _actions(cfg.below_actions):
            for above_action in _actions(cfg.above_actions):
                for score_gap in _floats(cfg.score_gaps):
                    transformed, select_counts = _apply_transform(select_base, scores, threshold, below_action, above_action, score_gap)
                    for stop in _floats(cfg.stop_losses):
                        for take in _floats(cfg.take_profits):
                            for rolling_n, rolling_loss in _loss_specs(cfg.rolling_loss_specs):
                                bt = _run_bt(transformed, cfg, work / f"select_{i}", stop=stop, take=take, rolling_n=rolling_n, rolling_loss=rolling_loss)
                                conf = {"threshold": threshold, "below_action": below_action, "above_action": above_action, "score_gap": score_gap, "stop": stop, "take": take, "rolling_window_trades": rolling_n, "rolling_loss_stop_pct": rolling_loss}
                                candidates.append({"config": conf, "selection_counts": dict(select_counts), "selection_score": _score_selection(bt, cfg.min_selection_trades), "selection": {"sim": bt["sim"], "trade_stats": bt["trade_stats"]}})
                                i += 1
    ranked = sorted(candidates, key=lambda r: (float(r["selection_score"]), float(r["selection"]["sim"].get("cagr_to_strict_mdd", -999)), float(r["selection"]["sim"].get("cagr_pct", -999))), reverse=True)
    for j, row in enumerate(ranked[: int(cfg.top_k)]):
        c = row["config"]
        transformed_eval, eval_counts = _apply_transform(eval_base, scores, float(c["threshold"]), str(c["below_action"]), str(c["above_action"]), float(c["score_gap"]))
        bt = _run_bt(transformed_eval, cfg, work / f"eval_top{j}", stop=float(c["stop"]), take=float(c["take"]), rolling_n=int(c["rolling_window_trades"]), rolling_loss=float(c["rolling_loss_stop_pct"]))
        row["eval_counts"] = eval_counts
        row["eval"] = {"sim": bt["sim"], "trade_stats": bt["trade_stats"]}
        if bool(cfg.keep_artifacts):
            row["eval"].update({"predictions_jsonl": str((work / f"eval_top{j}").with_suffix(".jsonl")), "backtest_json": str((work / f"eval_top{j}").with_suffix(".bt.json"))})
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "selection_rows": len(select_base),
        "eval_rows": len(eval_base),
        "top": ranked[: int(cfg.top_k)],
        "all_count": len(ranked),
        "leakage_guard": {
            "configs_ranked_on_selection_period_only": True,
            "eval_period_not_used_for_selection": True,
            "month_scores_are_prior_validation_scores": True,
            "row_transform_uses_only_current_prediction_scores": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nested score-geometry side-map transform selection")
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--rolling-summary-json", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--selection-start", default=NestedScoreGeometryTransformCfg.selection_start)
    p.add_argument("--selection-end", default=NestedScoreGeometryTransformCfg.selection_end)
    p.add_argument("--eval-start", default=NestedScoreGeometryTransformCfg.eval_start)
    p.add_argument("--eval-end", default=NestedScoreGeometryTransformCfg.eval_end)
    p.add_argument("--thresholds", default=NestedScoreGeometryTransformCfg.thresholds)
    p.add_argument("--below-actions", default=NestedScoreGeometryTransformCfg.below_actions)
    p.add_argument("--above-actions", default=NestedScoreGeometryTransformCfg.above_actions)
    p.add_argument("--score-gaps", default=NestedScoreGeometryTransformCfg.score_gaps)
    p.add_argument("--stop-losses", default=NestedScoreGeometryTransformCfg.stop_losses)
    p.add_argument("--take-profits", default=NestedScoreGeometryTransformCfg.take_profits)
    p.add_argument("--rolling-loss-specs", default=NestedScoreGeometryTransformCfg.rolling_loss_specs)
    p.add_argument("--min-selection-trades", type=int, default=NestedScoreGeometryTransformCfg.min_selection_trades)
    p.add_argument("--leverage", type=float, default=NestedScoreGeometryTransformCfg.leverage)
    p.add_argument("--top-k", type=int, default=NestedScoreGeometryTransformCfg.top_k)
    p.add_argument("--keep-artifacts", action="store_true")
    return p.parse_args()


def main() -> None:
    report = run(NestedScoreGeometryTransformCfg(**vars(parse_args())))
    compact = []
    for row in report["top"][:10]:
        compact.append({"config": row["config"], "selection": row["selection"]["sim"], "eval": row.get("eval", {}).get("sim", {}), "eval_counts": row.get("eval_counts", {})})
    print(json.dumps({"output": report["config"]["output"], "selection_rows": report["selection_rows"], "eval_rows": report["eval_rows"], "top": compact}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
