"""Train-only descriptor veto discovery with test selection and untouched eval.

This promotes failure-cluster mining into a leakage-safer protocol: descriptor
thresholds are discovered from train rows only, test only selects among fixed
veto/quantile candidates, and eval is never used until the final report.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import MISSING, asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.sparse_setup_ensemble_audit import EnsembleCfg, _load_market, _simulate_events
from training.sparse_setup_failure_cluster_miner import _descriptor_rows, _side_rows
from training.sparse_setup_regime_gate_tte import (
    RegimeGateTTECfg,
    _build_events,
    _feature_frame,
    _feature_names,
    _fit_ridge,
    _fold_names,
    _load_folds,
    _matrix,
    _parse_floats,
    _predict,
    _selection_score,
    _standardize_apply,
    _standardize_fit,
    _target,
)


@dataclass(frozen=True)
class DescriptorVetoTTECfg:
    sparse_report: str
    market_csv: str
    output: str
    train_folds_json: str
    test_folds_json: str
    eval_folds_json: str
    candidate_limit: int = 80
    ridge_alpha: float = 300.0
    quantiles: str = "0.90,0.925,0.95,0.975"
    min_test_trades: int = 20
    max_veto_size: int = 1
    top_descriptors_per_scope: int = 12
    good_min_utility_pct: float = 0.25
    good_max_mae_pct: float = 2.5
    bad_max_utility_pct: float = -0.25
    bad_min_mae_pct: float = 0.0
    min_cluster_rows: int = 80
    min_coverage_edge: float = 0.08
    max_good_block_rate: float = 0.65
    leverage: float = 1.0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    trade_stop_loss_pct: float = 6.0
    trade_take_profit_pct: float = 6.0
    setup_sizing: str = "prior_sharpe"
    window_size: int = 144
    include_price_action_extremes: bool = True
    include_failure_regime_classes: bool = True
    price_action_lookbacks: str = "36,72,144,288,576,2016"
    feature_include_regex: str = "^(fr__|mkt__(dxy_|kimchi_|usdkrw_|htf_1d|htf_3d|range_|trend_|volume_)|wave__(mom_|cvd_|flow_|vol_)|pa__pa_ext_(144|288|576)_)"
    max_features: int = 140
    target: str = "utility"


def _gate_cfg(cfg: DescriptorVetoTTECfg) -> RegimeGateTTECfg:
    return RegimeGateTTECfg(**{k: getattr(cfg, k) for k in RegimeGateTTECfg.__dataclass_fields__ if hasattr(cfg, k)})


def _label_rows(rows: list[dict[str, Any]], cfg: DescriptorVetoTTECfg) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    good: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    for row in rows:
        reward = row.get("reward", {})
        utility = float(reward.get("utility", 0.0) or 0.0)
        mae = float(reward.get("mae_pct", 0.0) or 0.0)
        if utility >= float(cfg.good_min_utility_pct) and mae <= float(cfg.good_max_mae_pct):
            good.append(row)
        elif utility <= float(cfg.bad_max_utility_pct) and mae >= float(cfg.bad_min_mae_pct):
            bad.append(row)
    return good, bad


def _descriptor_candidates(features: pd.DataFrame, names: list[str], train_rows: list[dict[str, Any]], cfg: DescriptorVetoTTECfg) -> list[dict[str, Any]]:
    good, bad = _label_rows(train_rows, cfg)
    scoped: list[dict[str, Any]] = []
    specs = [
        ("overall", good, bad),
        ("long", _side_rows(good, 1), _side_rows(bad, 1)),
        ("short", _side_rows(good, -1), _side_rows(bad, -1)),
    ]
    for scope, g_rows, b_rows in specs:
        rows = _descriptor_rows(features, names, g_rows, b_rows, cfg)[: int(cfg.top_descriptors_per_scope) * 3]
        kept = []
        for row in rows:
            rule = row["veto_rule"]
            if float(rule["coverage_edge"]) < float(cfg.min_coverage_edge):
                continue
            if float(rule["good_block_rate"]) > float(cfg.max_good_block_rate):
                continue
            kept.append({"scope": scope, **row})
            if len(kept) >= int(cfg.top_descriptors_per_scope):
                break
        scoped.extend(kept)
    return scoped


def _rule_fires(row: dict[str, Any], features: pd.DataFrame, desc: dict[str, Any]) -> bool:
    scope = str(desc.get("scope", "overall"))
    side = int(row.get("side", 0) or 0)
    if scope == "long" and side <= 0:
        return False
    if scope == "short" and side >= 0:
        return False
    rule = desc["veto_rule"]
    val = float(features[str(desc["feature"])].iloc[int(row["signal_pos"])])
    if str(rule["direction"]) == "ge":
        return val >= float(rule["threshold"])
    if str(rule["direction"]) == "le":
        return val <= float(rule["threshold"])
    raise ValueError(f"unsupported direction: {rule['direction']}")


def _veto_mask(rows: list[dict[str, Any]], features: pd.DataFrame, veto: tuple[dict[str, Any], ...]) -> np.ndarray:
    if not veto:
        return np.ones(len(rows), dtype=bool)
    keep = np.ones(len(rows), dtype=bool)
    for i, row in enumerate(rows):
        for desc in veto:
            if _rule_fires(row, features, desc):
                keep[i] = False
                break
    return keep


def _select_events(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float, features: pd.DataFrame, veto: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    keep = _veto_mask(rows, features, veto)
    best: dict[int, tuple[float, dict[str, Any]]] = {}
    for row, score, ok in zip(rows, scores, keep):
        if not ok or float(score) < float(threshold):
            continue
        pos = int(row["signal_pos"])
        cur = best.get(pos)
        if cur is None or float(score) > cur[0]:
            best[pos] = (float(score), row)
    return [dict(row, pred_score=float(score)) for _pos, (score, row) in sorted(best.items())]


def _simulate_selected(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float, features: pd.DataFrame, veto: tuple[dict[str, Any], ...], market: pd.DataFrame, cfg: DescriptorVetoTTECfg) -> dict[str, Any]:
    selected = _select_events(rows, scores, threshold, features, veto)
    ecfg = EnsembleCfg(cfg.sparse_report, cfg.market_csv, cfg.output, leverage=cfg.leverage, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate, entry_delay_bars=cfg.entry_delay_bars, trade_stop_loss_pct=cfg.trade_stop_loss_pct, trade_take_profit_pct=cfg.trade_take_profit_pct, setup_sizing=cfg.setup_sizing)
    sim = _simulate_events(selected, dates=pd.to_datetime(market["date"]), market=market, cfg=ecfg)
    return {k: v for k, v in sim.items() if k != "executed"} | {"selected_events": len(selected)}


def _descriptor_id(desc: dict[str, Any]) -> str:
    rule = desc["veto_rule"]
    return f"{desc.get('scope','overall')}:{desc['feature']}:{rule['direction']}:{float(rule['threshold']):.12g}"


def _veto_sets(descriptors: list[dict[str, Any]], cfg: DescriptorVetoTTECfg) -> list[tuple[dict[str, Any], ...]]:
    out: list[tuple[dict[str, Any], ...]] = [tuple()]
    max_size = max(1, int(cfg.max_veto_size))
    for k in range(1, max_size + 1):
        out.extend(tuple(combo) for combo in itertools.combinations(descriptors, k))
    return out


def run(cfg: DescriptorVetoTTECfg) -> dict[str, Any]:
    gate_cfg = _gate_cfg(cfg)
    train_folds = _load_folds(cfg.train_folds_json)
    test_folds = _load_folds(cfg.test_folds_json)
    eval_folds = _load_folds(cfg.eval_folds_json)
    sparse = json.loads(Path(cfg.sparse_report).read_text())
    sparse = dict(sparse)
    sparse["folds"] = sorted(train_folds + test_folds + eval_folds, key=lambda f: f["eval_start"])
    market = _load_market(cfg.market_csv)
    dates = pd.to_datetime(market["date"])
    features = _feature_frame(market, gate_cfg)
    names = _feature_names(features, gate_cfg)
    events = _build_events(gate_cfg, sparse, market, dates)
    train_names, test_names, eval_names = _fold_names(train_folds), _fold_names(test_folds), _fold_names(eval_folds)
    train_rows = [e for e in events if str(e.get("fold")) in train_names]
    test_rows = [e for e in events if str(e.get("fold")) in test_names]
    eval_rows = [e for e in events if str(e.get("fold")) in eval_names]

    descriptors = _descriptor_candidates(features, names, train_rows, cfg)
    vetoes = _veto_sets(descriptors, cfg)

    xtr, _ = _matrix(train_rows, features, names)
    ytr = _target(train_rows, gate_cfg)
    xtrz, mu, sd = _standardize_fit(xtr)
    w = _fit_ridge(xtrz, ytr, cfg.ridge_alpha)
    train_scores = _predict(xtrz, w)
    xtest, _ = _matrix(test_rows, features, names)
    test_scores = _predict(_standardize_apply(xtest, mu, sd), w)

    candidates: list[dict[str, Any]] = []
    for q in _parse_floats(cfg.quantiles):
        threshold = float(np.quantile(train_scores, q)) if len(train_scores) else 1e9
        for veto in vetoes:
            train_res = _simulate_selected(train_rows, train_scores, threshold, features, veto, market, cfg)
            test_res = _simulate_selected(test_rows, test_scores, threshold, features, veto, market, cfg)
            candidates.append({
                "q": q,
                "threshold": threshold,
                "veto": [_descriptor_id(d) for d in veto],
                "score": _selection_score(test_res["sim"], int(cfg.min_test_trades)),
                "train": train_res,
                "test": test_res,
            })
    candidates.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = candidates[0] if candidates else {"q": 1.0, "threshold": 1e9, "veto": [], "score": -1e9}
    selected_veto_ids = set(selected.get("veto", []))
    selected_veto = tuple(d for d in descriptors if _descriptor_id(d) in selected_veto_ids)

    tt_rows = train_rows + test_rows
    xtt, _ = _matrix(tt_rows, features, names)
    ytt = _target(tt_rows, gate_cfg)
    xttz, mu2, sd2 = _standardize_fit(xtt)
    w2 = _fit_ridge(xttz, ytt, cfg.ridge_alpha)
    tt_scores = _predict(xttz, w2)
    eval_threshold = float(np.quantile(tt_scores, float(selected["q"]))) if len(tt_scores) else 1e9
    xeval, _ = _matrix(eval_rows, features, names)
    eval_scores = _predict(_standardize_apply(xeval, mu2, sd2), w2)
    eval_res = _simulate_selected(eval_rows, eval_scores, eval_threshold, features, selected_veto, market, cfg)

    train_good, train_bad = _label_rows(train_rows, cfg)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "rows": {"events": len(events), "train": len(train_rows), "test": len(test_rows), "eval": len(eval_rows), "train_good": len(train_good), "train_bad": len(train_bad)},
        "features": {"numeric": len(names), "expanded": len(names) * 2 + 8},
        "descriptor_candidates": [{"id": _descriptor_id(d), **d} for d in descriptors],
        "veto_candidates": len(vetoes),
        "selected_by_test": {"q": selected.get("q"), "threshold": selected.get("threshold"), "veto": selected.get("veto"), "score": selected.get("score")},
        "top_test_candidates": candidates[:20],
        "final_eval_threshold": eval_threshold,
        "periods": {"train": selected.get("train"), "test": selected.get("test"), "eval": eval_res},
        "leakage_guard": {
            "descriptor_thresholds_discovered_from_train_only": True,
            "ridge_fit_uses_train_rows_only_for_test_selection": True,
            "test_selects_quantile_and_fixed_veto_only": True,
            "eval_not_used_for_descriptor_or_quantile_or_veto_selection": True,
            "final_eval_refits_scoring_model_on_train_plus_test_but_keeps_selected_train_descriptor_veto": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train-only descriptor-veto sparse setup TTE validation")
    for field in DescriptorVetoTTECfg.__dataclass_fields__.values():
        name = "--" + field.name.replace("_", "-")
        required = field.default is MISSING and field.default_factory is MISSING
        p.add_argument(name, default=None if required else field.default, required=required)
    ns = vars(p.parse_args())
    for k in {"candidate_limit", "min_test_trades", "max_veto_size", "top_descriptors_per_scope", "entry_delay_bars", "window_size", "max_features", "min_cluster_rows"}:
        ns[k] = int(ns[k])
    for k in {"ridge_alpha", "good_min_utility_pct", "good_max_mae_pct", "bad_max_utility_pct", "bad_min_mae_pct", "min_coverage_edge", "max_good_block_rate", "leverage", "fee_rate", "slippage_rate", "trade_stop_loss_pct", "trade_take_profit_pct"}:
        ns[k] = float(ns[k])
    for k in {"include_price_action_extremes", "include_failure_regime_classes"}:
        ns[k] = str(ns[k]).lower() not in {"false", "0", "no"} if not isinstance(ns[k], bool) else ns[k]
    return argparse.Namespace(**ns)


def main() -> None:
    rep = run(DescriptorVetoTTECfg(**vars(parse_args())))
    print(json.dumps({"selected_by_test": rep["selected_by_test"], "periods": {k: v["sim"] for k, v in rep["periods"].items()}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
