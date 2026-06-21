"""Scan past-only feature regime filters for an existing prediction stream.

The intended use is to take a leak-safe alpha signal that is profitable in some
windows but drawdown-prone in others, then test simple interpretable veto rules
on a historical selector/validation split before touching final eval.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.alpha_linear_combo_scan import _feature_groups, _load_market, _parse_list
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class FeatureRegimeFilterConfig:
    input_csv: str
    base_predictions: str
    eval_predictions: str
    output: str
    work_dir: str
    selector_start: str = "2024-07-01"
    selector_end: str = "2024-12-31 23:59:59"
    validation_start: str = "2025-01-01"
    validation_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01 00:00:00"
    feature_groups: str = "external_plus_market"
    quantiles: str = "0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90"
    scopes: str = "ALL,LONG,SHORT"
    preserve_other_side: bool = True
    leverage: float = 0.30
    pause_after_losses: int = 4
    pause_bars: int = 288
    monthly_loss_stop_pct: float = 0.0
    trade_take_profit_pct: float = 0.0
    window_size: int = 144
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    min_selector_trades: int = 40
    min_validation_trades: int = 120
    max_validation_mdd: float = 18.0
    top_k: int = 40


NO_TRADE = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "FEATURE_REGIME_FILTER", "confidence": "HIGH"}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n")


def _is_trade(row: dict[str, Any]) -> bool:
    p = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
    return p.get("gate") == "TRADE"


def _side(row: dict[str, Any]) -> str:
    p = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
    return str(p.get("side", "NONE")).upper()


def _slice_rows(rows: list[dict[str, Any]], start: str, end: str) -> list[dict[str, Any]]:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    for r in rows:
        d = pd.Timestamp(str(r.get("date")))
        if s <= d <= e:
            out.append(r)
    return out


def _apply_filter(rows: list[dict[str, Any]], values: np.ndarray, *, feature: str, direction: str, threshold: float, scope: str, preserve_other_side: bool) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out: list[dict[str, Any]] = []
    stats = {"trade_rows": 0, "passed": 0, "blocked": 0, "missing_feature": 0, "other_side_preserved": 0}
    scope = scope.upper()
    for r in rows:
        if not _is_trade(r):
            # Dense NO_TRADE rows are execution no-ops. Omitting them keeps broad
            # filter scans fast and avoids writing gigabytes of scratch JSONL.
            continue
        stats["trade_rows"] += 1
        row_side = _side(r)
        if scope in {"LONG", "SHORT"} and row_side != scope:
            if preserve_other_side:
                stats["other_side_preserved"] += 1
                stats["passed"] += 1
                out.append(r)
            else:
                stats["blocked"] += 1
                out.append({**r, "blocked_prediction": r.get("prediction"), "prediction": dict(NO_TRADE), "regime_filter": {"feature": feature, "scope": scope, "reason": "side_scope"}})
            continue
        pos = int(r.get("signal_pos", -1) or -1)
        v = float(values[pos]) if 0 <= pos < len(values) else float("nan")
        if not np.isfinite(v):
            stats["missing_feature"] += 1
            ok = False
        elif direction == "ge":
            ok = v >= threshold
        elif direction == "le":
            ok = v <= threshold
        else:
            raise ValueError(direction)
        if ok:
            stats["passed"] += 1
            out.append({**r, "regime_filter": {"feature": feature, "direction": direction, "threshold": threshold, "scope": scope, "value": v}})
        else:
            stats["blocked"] += 1
            out.append({**r, "blocked_prediction": r.get("prediction"), "prediction": dict(NO_TRADE), "regime_filter": {"feature": feature, "direction": direction, "threshold": threshold, "scope": scope, "value": v}})
    return out, stats


def _bt(pred_path: str, cfg: FeatureRegimeFilterConfig) -> dict[str, Any]:
    return run_overlay(
        OnlineRiskOverlayConfig(
            predictions_jsonl=pred_path,
            market_csv=cfg.input_csv,
            output=str(Path(cfg.work_dir) / (Path(pred_path).stem + ".bt.json")),
            leverage=float(cfg.leverage),
            pause_after_losses=int(cfg.pause_after_losses),
            pause_bars=int(cfg.pause_bars),
            monthly_loss_stop_pct=float(cfg.monthly_loss_stop_pct),
            trade_take_profit_pct=float(cfg.trade_take_profit_pct),
        )
    )


def _score(sel: dict[str, Any], val: dict[str, Any], *, min_sel: int, min_val: int, max_mdd: float) -> float:
    ss, vs = sel["sim"], val["sim"]
    sel_trades, val_trades = int(ss["trade_entries"]), int(vs["trade_entries"])
    val_cagr, val_mdd = float(vs["cagr_pct"]), float(vs["strict_mdd_pct"])
    sel_cagr, sel_mdd = float(ss["cagr_pct"]), float(ss["strict_mdd_pct"])
    val_ratio = float(vs["cagr_to_strict_mdd"])
    sel_ratio = float(ss["cagr_to_strict_mdd"])
    if sel_trades < min_sel or val_trades < min_val or val_cagr <= 0 or sel_cagr <= 0 or val_mdd > max_mdd:
        return -1000.0 + min(sel_trades, val_trades) / 10000.0 + val_cagr / 1000.0 - max(0.0, val_mdd - max_mdd) / 100.0
    return min(sel_ratio, val_ratio) + 0.25 * max(0.0, val_ratio) + min(1.0, val_trades / 300.0) - float(val["trade_stats"].get("p_value_mean_ret_approx", 1.0))


def run_scan(cfg: FeatureRegimeFilterConfig) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    dates = pd.to_datetime(market["date"])
    columns = [c for c in features.columns if np.nanstd(features[c].to_numpy(dtype=float)) > 1e-12]
    groups = _feature_groups(columns)
    selected_cols: list[str] = []
    for group in _parse_list(cfg.feature_groups, str):
        selected_cols.extend(groups.get(group, []))
    selected_cols = sorted(set(selected_cols))
    if not selected_cols:
        raise ValueError(f"no columns selected for groups={cfg.feature_groups}")

    base_rows = _read_jsonl(cfg.base_predictions)
    eval_rows = _read_jsonl(cfg.eval_predictions)
    selector_rows = _slice_rows(base_rows, cfg.selector_start, cfg.selector_end)
    validation_rows = _slice_rows(base_rows, cfg.validation_start, cfg.validation_end)
    final_eval_rows = _slice_rows(eval_rows, cfg.eval_start, cfg.eval_end)
    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    # Baselines for the exact same split and overlay.
    _write_jsonl(work / "baseline_selector.jsonl", [r for r in selector_rows if _is_trade(r)])
    _write_jsonl(work / "baseline_validation.jsonl", [r for r in validation_rows if _is_trade(r)])
    _write_jsonl(work / "baseline_eval.jsonl", [r for r in final_eval_rows if _is_trade(r)])
    baseline = {
        "selector": _bt(str(work / "baseline_selector.jsonl"), cfg),
        "validation": _bt(str(work / "baseline_validation.jsonl"), cfg),
        "eval": _bt(str(work / "baseline_eval.jsonl"), cfg),
    }

    q_values = _parse_list(cfg.quantiles, float)
    scopes = [str(s).upper() for s in _parse_list(cfg.scopes, str)]
    candidates: list[dict[str, Any]] = []
    for col in selected_cols:
        vals = features[col].to_numpy(dtype=float)
        selector_positions = [int(r.get("signal_pos", -1) or -1) for r in selector_rows if _is_trade(r)]
        ref = vals[[p for p in selector_positions if 0 <= p < len(vals)]] if selector_positions else vals
        ref = ref[np.isfinite(ref)]
        if len(ref) < 10:
            continue
        for q in q_values:
            thr = float(np.quantile(ref, float(q)))
            for direction in ("ge", "le"):
                for scope in scopes:
                    # Reuse scratch files so broad scans do not fill WSL with
                    # thousands of full prediction streams. Metrics and filter
                    # metadata are captured in the JSON report.
                    sel_pred, sel_stats = _apply_filter(selector_rows, vals, feature=col, direction=direction, threshold=thr, scope=scope, preserve_other_side=bool(cfg.preserve_other_side))
                    val_pred, val_stats = _apply_filter(validation_rows, vals, feature=col, direction=direction, threshold=thr, scope=scope, preserve_other_side=bool(cfg.preserve_other_side))
                    sel_path = work / "candidate_selector.jsonl"
                    val_path = work / "candidate_validation.jsonl"
                    _write_jsonl(sel_path, sel_pred)
                    _write_jsonl(val_path, val_pred)
                    sel_bt = _bt(str(sel_path), cfg)
                    val_bt = _bt(str(val_path), cfg)
                    score = _score(sel_bt, val_bt, min_sel=int(cfg.min_selector_trades), min_val=int(cfg.min_validation_trades), max_mdd=float(cfg.max_validation_mdd))
                    eval_bt = None
                    eval_stats = None
                    if score > -999.0 or (float(val_bt["sim"]["cagr_pct"]) > 0 and float(val_bt["sim"]["strict_mdd_pct"]) <= 25.0):
                        ev_pred, eval_stats = _apply_filter(final_eval_rows, vals, feature=col, direction=direction, threshold=thr, scope=scope, preserve_other_side=bool(cfg.preserve_other_side))
                        ev_path = work / "candidate_eval.jsonl"
                        _write_jsonl(ev_path, ev_pred)
                        eval_bt = _bt(str(ev_path), cfg)
                    candidates.append({
                        "filter": {"feature": col, "direction": direction, "threshold": thr, "quantile": float(q), "scope": scope, "preserve_other_side": bool(cfg.preserve_other_side)},
                        "stats": {"selector": sel_stats, "validation": val_stats, "eval": eval_stats},
                        "selector": {"period": sel_bt["period"], "sim": sel_bt["sim"], "trade_stats": sel_bt["trade_stats"]},
                        "validation": {"period": val_bt["period"], "sim": val_bt["sim"], "trade_stats": val_bt["trade_stats"]},
                        "eval": None if eval_bt is None else {"period": eval_bt["period"], "sim": eval_bt["sim"], "trade_stats": eval_bt["trade_stats"]},
                        "selection_score": score,
                    })
    ranked = sorted(candidates, key=lambda r: (float(r["selection_score"]), float((r.get("eval") or {"sim": {"cagr_to_strict_mdd": -999}})["sim"].get("cagr_to_strict_mdd", -999))), reverse=True)
    report = {
        "config": asdict(cfg),
        "baseline": baseline,
        "feature_columns": selected_cols,
        "top_by_selection": ranked[: int(cfg.top_k)],
        "all_count": len(candidates),
        "selection_protocol": "feature thresholds are fit on selector-window trade rows only; candidates are ranked by selector+validation; eval is reported but not used for selection",
        "leakage_guard": {"threshold_fit_window": "selector only", "validation_not_used_for_threshold_fit": True, "eval_not_used_for_selection": True, "external_join": "backward_asof_no_future" if cfg.wave_trading_root else "disabled"},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan feature regime filters for an existing prediction stream")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--base-predictions", required=True)
    p.add_argument("--eval-predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--selector-start", default=FeatureRegimeFilterConfig.selector_start)
    p.add_argument("--selector-end", default=FeatureRegimeFilterConfig.selector_end)
    p.add_argument("--validation-start", default=FeatureRegimeFilterConfig.validation_start)
    p.add_argument("--validation-end", default=FeatureRegimeFilterConfig.validation_end)
    p.add_argument("--eval-start", default=FeatureRegimeFilterConfig.eval_start)
    p.add_argument("--eval-end", default=FeatureRegimeFilterConfig.eval_end)
    p.add_argument("--feature-groups", default=FeatureRegimeFilterConfig.feature_groups)
    p.add_argument("--quantiles", default=FeatureRegimeFilterConfig.quantiles)
    p.add_argument("--scopes", default=FeatureRegimeFilterConfig.scopes)
    p.add_argument("--block-other-side", dest="preserve_other_side", action="store_false")
    p.add_argument("--leverage", type=float, default=FeatureRegimeFilterConfig.leverage)
    p.add_argument("--pause-after-losses", type=int, default=FeatureRegimeFilterConfig.pause_after_losses)
    p.add_argument("--pause-bars", type=int, default=FeatureRegimeFilterConfig.pause_bars)
    p.add_argument("--monthly-loss-stop-pct", type=float, default=FeatureRegimeFilterConfig.monthly_loss_stop_pct)
    p.add_argument("--trade-take-profit-pct", type=float, default=FeatureRegimeFilterConfig.trade_take_profit_pct)
    p.add_argument("--window-size", type=int, default=FeatureRegimeFilterConfig.window_size)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default=FeatureRegimeFilterConfig.external_tolerance)
    p.add_argument("--min-selector-trades", type=int, default=FeatureRegimeFilterConfig.min_selector_trades)
    p.add_argument("--min-validation-trades", type=int, default=FeatureRegimeFilterConfig.min_validation_trades)
    p.add_argument("--max-validation-mdd", type=float, default=FeatureRegimeFilterConfig.max_validation_mdd)
    p.add_argument("--top-k", type=int, default=FeatureRegimeFilterConfig.top_k)
    return p.parse_args()


def main() -> None:
    rep = run_scan(FeatureRegimeFilterConfig(**vars(parse_args())))
    for row in rep["top_by_selection"][:20]:
        ss, vs = row["selector"]["sim"], row["validation"]["sim"]
        es = (row.get("eval") or {"sim": {}})["sim"]
        print(json.dumps({
            "filter": row["filter"], "score": row["selection_score"],
            "selector": {"cagr": ss.get("cagr_pct"), "mdd": ss.get("strict_mdd_pct"), "ratio": ss.get("cagr_to_strict_mdd"), "trades": ss.get("trade_entries")},
            "validation": {"cagr": vs.get("cagr_pct"), "mdd": vs.get("strict_mdd_pct"), "ratio": vs.get("cagr_to_strict_mdd"), "trades": vs.get("trade_entries")},
            "eval": {"cagr": es.get("cagr_pct"), "mdd": es.get("strict_mdd_pct"), "ratio": es.get("cagr_to_strict_mdd"), "trades": es.get("trade_entries")},
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
