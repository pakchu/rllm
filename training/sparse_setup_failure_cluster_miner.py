"""Mine signal-time descriptors separating train-good from test-bad sparse events.

This is diagnostic feature discovery for failure-regime label design.  It uses
realized event outcomes to form clusters, so descriptors must be promoted through
a later nested train/test/eval protocol before deployment.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import MISSING, asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.sparse_setup_regime_gate_tte import RegimeGateTTECfg, _build_events, _feature_frame, _feature_names, _fold_names, _load_folds
from training.sparse_setup_ensemble_audit import _load_market


@dataclass(frozen=True)
class FailureClusterMinerCfg:
    sparse_report: str
    market_csv: str
    output: str
    train_folds_json: str
    test_folds_json: str
    eval_folds_json: str
    candidate_limit: int = 80
    window_size: int = 144
    include_price_action_extremes: bool = True
    include_failure_regime_classes: bool = True
    price_action_lookbacks: str = "36,72,144,288,576,2016"
    feature_include_regex: str = "^(fr__|mkt__(dxy_|kimchi_|usdkrw_|htf_1d|htf_3d|range_|trend_|volume_)|wave__(mom_|cvd_|flow_|vol_)|pa__pa_ext_(144|288|576)_)"
    max_features: int = 140
    good_min_utility_pct: float = 0.25
    good_max_mae_pct: float = 2.5
    bad_max_utility_pct: float = -0.25
    bad_min_mae_pct: float = 0.0
    min_cluster_rows: int = 30
    top_k: int = 60


def _gate_cfg(cfg: FailureClusterMinerCfg) -> RegimeGateTTECfg:
    data: dict[str, Any] = {}
    for name in RegimeGateTTECfg.__dataclass_fields__:
        if hasattr(cfg, name):
            data[name] = getattr(cfg, name)
    data.setdefault("ridge_alpha", 100.0)
    data.setdefault("quantiles", "0.9")
    data.setdefault("min_test_trades", 20)
    data.setdefault("output", cfg.output)
    return RegimeGateTTECfg(**data)


def _rows_by_cluster(events: list[dict[str, Any]], train_names: set[str], test_names: set[str], cfg: FailureClusterMinerCfg) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    train_good: list[dict[str, Any]] = []
    test_bad: list[dict[str, Any]] = []
    other = 0
    for ev in events:
        reward = ev.get("reward", {})
        utility = float(reward.get("utility", 0.0) or 0.0)
        mae = float(reward.get("mae_pct", 0.0) or 0.0)
        fold = str(ev.get("fold"))
        if fold in train_names and utility >= float(cfg.good_min_utility_pct) and mae <= float(cfg.good_max_mae_pct):
            train_good.append(ev)
        elif fold in test_names and utility <= float(cfg.bad_max_utility_pct) and mae >= float(cfg.bad_min_mae_pct):
            test_bad.append(ev)
        else:
            other += 1
    return train_good, test_bad, {"train_good": len(train_good), "test_bad": len(test_bad), "other": other}


def _values(features: pd.DataFrame, rows: list[dict[str, Any]], name: str) -> np.ndarray:
    if not rows:
        return np.zeros(0, dtype=float)
    pos = [int(r["signal_pos"]) for r in rows]
    return features[name].iloc[pos].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)


def _effect(good: np.ndarray, bad: np.ndarray) -> dict[str, float]:
    gm, bm = float(np.mean(good)), float(np.mean(bad))
    gs, bs = float(np.std(good)), float(np.std(bad))
    pooled = float(np.sqrt((gs * gs + bs * bs) / 2.0)) if gs + bs > 0 else 0.0
    d_bad_minus_good = (bm - gm) / pooled if pooled > 1e-12 else 0.0
    return {
        "good_mean": gm,
        "bad_mean": bm,
        "good_median": float(np.median(good)),
        "bad_median": float(np.median(bad)),
        "good_p10": float(np.percentile(good, 10)),
        "good_p25": float(np.percentile(good, 25)),
        "good_p75": float(np.percentile(good, 75)),
        "good_p90": float(np.percentile(good, 90)),
        "bad_p10": float(np.percentile(bad, 10)),
        "bad_p25": float(np.percentile(bad, 25)),
        "bad_p75": float(np.percentile(bad, 75)),
        "bad_p90": float(np.percentile(bad, 90)),
        "effect_d_bad_minus_good": float(d_bad_minus_good),
        "abs_effect": abs(float(d_bad_minus_good)),
    }


def _coverage_rule(row: dict[str, Any], good: np.ndarray, bad: np.ndarray) -> dict[str, Any]:
    # If bad values are higher, a high-threshold veto is natural; if lower, low-threshold veto.
    if float(row["effect_d_bad_minus_good"]) >= 0.0:
        threshold = float(row["bad_p25"])
        direction = "ge"
        bad_cov = float(np.mean(bad >= threshold))
        good_block = float(np.mean(good >= threshold))
    else:
        threshold = float(row["bad_p75"])
        direction = "le"
        bad_cov = float(np.mean(bad <= threshold))
        good_block = float(np.mean(good <= threshold))
    return {
        "direction": direction,
        "threshold": threshold,
        "bad_coverage": bad_cov,
        "good_block_rate": good_block,
        "coverage_edge": bad_cov - good_block,
    }


def _descriptor_rows(features: pd.DataFrame, feature_names: list[str], good_rows: list[dict[str, Any]], bad_rows: list[dict[str, Any]], cfg: FailureClusterMinerCfg) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in feature_names:
        good = _values(features, good_rows, name)
        bad = _values(features, bad_rows, name)
        if len(good) < int(cfg.min_cluster_rows) or len(bad) < int(cfg.min_cluster_rows):
            continue
        row = {"feature": name, "good_n": int(len(good)), "bad_n": int(len(bad)), **_effect(good, bad)}
        row["veto_rule"] = _coverage_rule(row, good, bad)
        out.append(row)
    out.sort(key=lambda r: (float(r["veto_rule"]["coverage_edge"]), float(r["abs_effect"])), reverse=True)
    return out


def _side_rows(rows: list[dict[str, Any]], side: int) -> list[dict[str, Any]]:
    return [r for r in rows if int(r.get("side", 0)) == int(side)]


def run(cfg: FailureClusterMinerCfg) -> dict[str, Any]:
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
    train_good, test_bad, counts = _rows_by_cluster(events, _fold_names(train_folds), _fold_names(test_folds), cfg)
    overall = _descriptor_rows(features, names, train_good, test_bad, cfg)
    long_rows = _descriptor_rows(features, names, _side_rows(train_good, 1), _side_rows(test_bad, 1), cfg)
    short_rows = _descriptor_rows(features, names, _side_rows(train_good, -1), _side_rows(test_bad, -1), cfg)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "rows": {"events": len(events), **counts},
        "feature_count": len(names),
        "overall_descriptors": overall[: int(cfg.top_k)],
        "long_descriptors": long_rows[: int(cfg.top_k)],
        "short_descriptors": short_rows[: int(cfg.top_k)],
        "guidance": {
            "use": "diagnostic only; promote descriptors through nested TTE before deployment",
            "interpretation": "high coverage_edge means the rule captures many 2025 bad events while blocking fewer train-good events",
        },
        "leakage_note": "test_bad labels use 2025 realized outcomes for mining; selected rules are not deployment-valid until retested in a fresh nested split",
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mine descriptors separating train-good and test-bad sparse setup events")
    for field in FailureClusterMinerCfg.__dataclass_fields__.values():
        name = "--" + field.name.replace("_", "-")
        required = field.default is MISSING and field.default_factory is MISSING
        p.add_argument(name, default=None if required else field.default, required=required)
    ns = vars(p.parse_args())
    for k in {"candidate_limit", "window_size", "max_features", "min_cluster_rows", "top_k"}:
        ns[k] = int(ns[k])
    for k in {"good_min_utility_pct", "good_max_mae_pct", "bad_max_utility_pct", "bad_min_mae_pct"}:
        ns[k] = float(ns[k])
    for k in {"include_price_action_extremes", "include_failure_regime_classes"}:
        ns[k] = str(ns[k]).lower() not in {"false", "0", "no"} if not isinstance(ns[k], bool) else ns[k]
    return argparse.Namespace(**ns)


def main() -> None:
    rep = run(FailureClusterMinerCfg(**vars(parse_args())))
    print(json.dumps({"rows": rep["rows"], "top_overall": rep["overall_descriptors"][:10], "top_long": rep["long_descriptors"][:5], "top_short": rep["short_descriptors"][:5]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
