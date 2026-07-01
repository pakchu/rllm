"""Fit simple causal risk filters for pairwise prediction rows on one period and test on another."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.sparse_setup_ensemble_audit import EnsembleCfg, _load_market, _simulate_events


@dataclass(frozen=True)
class PairwiseRiskFilterSweepCfg:
    predictions_jsonl: str
    candidate_jsonl: str
    market_csv: str
    output: str
    fit_end: str = "2025-12-31 23:59:59"
    score_quantile: float = 0.80
    min_fit_trades: int = 20
    min_test_trades: int = 10
    max_filters: int = 50
    leverage: float = 1.0
    entry_delay_bars: int = 1
    max_same_bar_signals: int = 1


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _candidate_key(source: dict[str, Any]) -> tuple[int, str, int, str]:
    cand = source.get("candidate") if isinstance(source.get("candidate"), dict) else {}
    return (
        int(source.get("signal_pos", -1) or -1),
        str(cand.get("side", source.get("side", ""))).upper(),
        int(cand.get("hold_bars", cand.get("horizon", 0)) or 0),
        str(cand.get("family", "unknown")),
    )


def _candidate_map(path: str) -> dict[tuple[int, str, int, str], dict[str, Any]]:
    out: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    for row in _load_jsonl(path):
        cand = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
        key = (int(row.get("signal_pos", -1) or -1), str(cand.get("side", row.get("side", ""))).upper(), int(cand.get("hold_bars", cand.get("horizon", 0)) or 0), str(cand.get("family", "unknown")))
        out[key] = row
    return out


def _pred_choice(row: dict[str, Any]) -> tuple[str, float]:
    scores = row.get("scores") or {}
    margin = float(scores.get("A", 0.0)) - float(scores.get("B", 0.0))
    return str(row.get("prediction", "A")), abs(margin)


def _selected_source(row: dict[str, Any]) -> dict[str, Any]:
    pred, _ = _pred_choice(row)
    return (row.get("candidates") or {}).get(pred, {})


def _enrich_predictions(preds: list[dict[str, Any]], candidates: dict[tuple[int, str, int, str], dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in preds:
        source = _selected_source(row)
        key = _candidate_key(source)
        cand_row = candidates.get(key)
        if not cand_row:
            continue
        nr = dict(row)
        nr["selected_source"] = source
        nr["selected_candidate_row"] = cand_row
        nr["score_margin_abs"] = _pred_choice(row)[1]
        nr["selected_date"] = str(source.get("date", row.get("date")))
        out.append(nr)
    return out


def _event(row: dict[str, Any]) -> dict[str, Any] | None:
    src = row.get("selected_source") or {}
    cand = src.get("candidate") if isinstance(src.get("candidate"), dict) else {}
    side_txt = str(cand.get("side", src.get("side", ""))).upper()
    side = 1 if side_txt == "LONG" else -1 if side_txt == "SHORT" else 0
    hold = int(cand.get("hold_bars", cand.get("horizon", 0)) or 0)
    pos = int(src.get("signal_pos", -1) or -1)
    if side == 0 or hold <= 0 or pos < 0:
        return None
    return {
        "signal_pos": pos,
        "date": str(src.get("date")),
        "side": side,
        "horizon": hold,
        "source_horizon": hold,
        "candidate_index": 0,
        "candidate_key": f"pairwise_filter|{side_txt}|h{hold}|m{float(row.get('score_margin_abs',0)):.4f}",
        "fold": str(src.get("date", ""))[:7],
        "prior_mean_ret": max(0.0, float(row.get("score_margin_abs", 0.0))),
        "prior_std_ret": 1.0,
        "prior_n": 100,
        "score": float(row.get("score_margin_abs", 0.0)),
    }


def _dedupe(events: list[dict[str, Any]], max_same_bar: int) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for ev in events:
        grouped.setdefault(int(ev["signal_pos"]), []).append(ev)
    out = []
    for pos in sorted(grouped):
        out.extend(sorted(grouped[pos], key=lambda e: float(e.get("score", 0.0)), reverse=True)[: max(1, int(max_same_bar))])
    return out


def _simulate(rows: list[dict[str, Any]], market: pd.DataFrame, cfg: PairwiseRiskFilterSweepCfg) -> dict[str, Any]:
    events = _dedupe([e for r in rows if (e := _event(r)) is not None], int(cfg.max_same_bar_signals))
    sim_cfg = EnsembleCfg(sparse_report="", market_csv=cfg.market_csv, output=cfg.output, leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars, max_same_bar_signals=cfg.max_same_bar_signals)
    bt = _simulate_events(events, dates=pd.to_datetime(market["date"]), market=market, cfg=sim_cfg)
    return {"events": len(events), "sim": bt.get("sim", {}), "trade_stats": bt.get("trade_stats", {})}


def _feature_value(row: dict[str, Any], name: str) -> str:
    src = row.get("selected_source") or {}
    cand = src.get("candidate") if isinstance(src.get("candidate"), dict) else {}
    cand_row = row.get("selected_candidate_row") or {}
    tokens = cand_row.get("state_tokens") if isinstance(cand_row.get("state_tokens"), dict) else {}
    if name == "side":
        return str(cand.get("side", src.get("side", ""))).upper()
    if name == "hold":
        return str(int(cand.get("hold_bars", cand.get("horizon", 0)) or 0))
    if name == "family":
        return str(cand.get("family", "unknown"))
    return str(tokens.get(name, "na"))


def _filter_rows(rows: list[dict[str, Any]], feature: str, value: str, threshold: float) -> list[dict[str, Any]]:
    return [r for r in rows if float(r.get("score_margin_abs", 0.0)) >= threshold and _feature_value(r, feature) == value]


def run(cfg: PairwiseRiskFilterSweepCfg) -> dict[str, Any]:
    preds = _enrich_predictions(_load_jsonl(cfg.predictions_jsonl), _candidate_map(cfg.candidate_jsonl))
    fit = [r for r in preds if str(r.get("selected_date", "")) <= str(cfg.fit_end)]
    test = [r for r in preds if str(r.get("selected_date", "")) > str(cfg.fit_end)]
    fit_margins = sorted(float(r.get("score_margin_abs", 0.0)) for r in fit)
    threshold = fit_margins[int(float(cfg.score_quantile) * (len(fit_margins) - 1))] if fit_margins else 999.0
    market = _load_market(cfg.market_csv)
    feature_names = [
        "side", "hold", "family", "range_location", "drawdown_state", "trend_24", "trend_96", "side_trend_24", "side_trend_96", "htf_4h", "htf_1d", "htf_1w", "taker_flow", "volume_state", "tok:rex_144_loc", "tok:rex_2016_loc", "tok:rex_144_lower_gap", "tok:rex_144_upper_gap",
    ]
    baseline_fit = _simulate([r for r in fit if float(r.get("score_margin_abs", 0.0)) >= threshold], market, cfg)
    baseline_test = _simulate([r for r in test if float(r.get("score_margin_abs", 0.0)) >= threshold], market, cfg)
    candidates = []
    for feat in feature_names:
        values = Counter(_feature_value(r, feat) for r in fit)
        for val, count in values.items():
            if val == "na" or count < 5:
                continue
            frows = _filter_rows(fit, feat, val, threshold)
            fit_res = _simulate(frows, market, cfg)
            if int(fit_res["sim"].get("trade_entries", 0) or 0) < int(cfg.min_fit_trades):
                continue
            trows = _filter_rows(test, feat, val, threshold)
            test_res = _simulate(trows, market, cfg)
            candidates.append({"feature": feat, "value": val, "fit_rows": len(frows), "test_rows": len(trows), "fit": fit_res, "test": test_res, "score": float(fit_res["sim"].get("cagr_to_strict_mdd", -999) or -999)})
    candidates.sort(key=lambda r: (float(r["score"]), int(r["fit"]["sim"].get("trade_entries", 0) or 0)), reverse=True)
    report = {
        "config": asdict(cfg),
        "rows": {"predictions": len(preds), "fit": len(fit), "test": len(test)},
        "threshold": threshold,
        "baseline_fit": baseline_fit,
        "baseline_test": baseline_test,
        "top_filters": candidates[: int(cfg.max_filters)],
        "leakage_guard": {"threshold_fit_on_fit_period_only": True, "filters_ranked_on_fit_period_only": True, "test_period_not_used_for_selection": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--candidate-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fit-end", default=PairwiseRiskFilterSweepCfg.fit_end)
    p.add_argument("--score-quantile", type=float, default=PairwiseRiskFilterSweepCfg.score_quantile)
    p.add_argument("--min-fit-trades", type=int, default=PairwiseRiskFilterSweepCfg.min_fit_trades)
    p.add_argument("--min-test-trades", type=int, default=PairwiseRiskFilterSweepCfg.min_test_trades)
    p.add_argument("--max-filters", type=int, default=PairwiseRiskFilterSweepCfg.max_filters)
    p.add_argument("--leverage", type=float, default=PairwiseRiskFilterSweepCfg.leverage)
    p.add_argument("--entry-delay-bars", type=int, default=PairwiseRiskFilterSweepCfg.entry_delay_bars)
    p.add_argument("--max-same-bar-signals", type=int, default=PairwiseRiskFilterSweepCfg.max_same_bar_signals)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(PairwiseRiskFilterSweepCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
