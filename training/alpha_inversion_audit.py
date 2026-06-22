"""Audit whether failed rolling alpha candidates become viable when inverted.

The prior alpha reports often show sign-flipping behavior. This audit replays the
same strict folds for top rolling candidates with the fitted side mapping inverted
inside each fold. It is diagnostic only: the inversion is evaluated per candidate
using already-selected report rows, so it is not a deployable selection rule by
itself unless a separate train/test/eval selector later chooses inversion without
looking at eval.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.alpha_candidate_gate import AlphaGateConfig, score_candidate
from training.alpha_feature_backtest import FeatureRuleConfig, _forward_return, fit_rule, simulate_rule
from training.wave_feature_ridge_policy import build_wave_feature_frame


@dataclass(frozen=True)
class InversionAuditConfig:
    input_report: str
    input_csv: str
    output: str
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    window_size: int = 144
    max_candidates: int = 12
    leverage: float = 1.0
    min_cagr_to_mdd: float = 3.0
    max_strict_mdd_pct: float = 15.0
    min_fold_trades: int = 30
    min_total_trades: int = 300
    min_positive_folds: int = 5


def _load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _build_features(market: pd.DataFrame, cfg: InversionAuditConfig) -> pd.DataFrame:
    base = build_market_feature_frame(market, window_size=int(cfg.window_size)).add_prefix("mkt__")
    wave = build_wave_feature_frame(market, window=int(cfg.window_size)).add_prefix("wave__")
    features = pd.concat([base, wave], axis=1)
    return features.loc[:, ~features.columns.duplicated(keep="last")].replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _invert_rule(rule: dict[str, Any]) -> dict[str, Any]:
    out = dict(rule)
    out["high_side"] = "SHORT" if str(rule.get("high_side")) == "LONG" else "LONG"
    out["low_side"] = "SHORT" if str(rule.get("low_side")) == "LONG" else "LONG"
    return out


def _replay_candidate(candidate: dict[str, Any], *, market: pd.DataFrame, features: pd.DataFrame, dates: pd.Series, folds: list[dict[str, str]], cfg: InversionAuditConfig, inverted: bool) -> dict[str, Any]:
    feature = str(candidate.get("feature"))
    if feature not in features.columns:
        raise ValueError(f"missing feature in rebuilt frame: {feature}")
    horizon = int(candidate.get("horizon"))
    quantile = float(candidate.get("quantile"))
    values = features[feature].to_numpy(dtype=float)
    fwd = _forward_return(market["open"].astype(float), horizon=horizon, entry_delay_bars=1)
    strict_folds = []
    for fold in folds:
        fit_end = str(pd.Timestamp(fold["eval_start"]) - pd.Timedelta(seconds=1))
        fold_cfg = FeatureRuleConfig(
            input_csv=cfg.input_csv,
            output="",
            feature=feature,
            horizon=horizon,
            fit_start=str(dates.iloc[0]),
            fit_end=fit_end,
            eval_start=fold["eval_start"],
            eval_end=fold["eval_end"],
            quantile=quantile,
            window_size=int(cfg.window_size),
            entry_delay_bars=1,
            leverage=float(cfg.leverage),
            wave_trading_root=cfg.wave_trading_root,
            external_tolerance=cfg.external_tolerance,
        )
        try:
            rule = fit_rule(dates=dates, feature_values=values, forward_returns=fwd, cfg=fold_cfg)
            if inverted:
                rule = _invert_rule(rule)
            result = simulate_rule(market=market, feature_values=values, dates=dates, rule=rule, cfg=fold_cfg)
        except Exception as exc:
            strict_folds.append({"fold": fold.get("name"), "error": str(exc)})
            continue
        strict_folds.append({"fold": fold.get("name"), "rule": rule, "result": result})
    out = {k: candidate.get(k) for k in ("feature", "horizon", "quantile", "event_score", "strict_score")}
    out["variant"] = "inverted" if inverted else "original_replay"
    out["strict_folds"] = strict_folds
    return out


def run(cfg: InversionAuditConfig) -> dict[str, Any]:
    source = json.loads(Path(cfg.input_report).read_text())
    candidates = list(source.get("top_strict", []))[: int(cfg.max_candidates)]
    market = _load_market(cfg.input_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    features = _build_features(market, cfg)
    dates = pd.to_datetime(market["date"])
    folds = source.get("folds", [])
    gate_cfg = AlphaGateConfig(
        input_report=cfg.input_report,
        output=cfg.output,
        min_cagr_to_mdd=float(cfg.min_cagr_to_mdd),
        max_strict_mdd_pct=float(cfg.max_strict_mdd_pct),
        min_fold_trades=int(cfg.min_fold_trades),
        min_total_trades=int(cfg.min_total_trades),
        min_positive_folds=int(cfg.min_positive_folds),
        require_all_folds_mdd=True,
    )
    rows = []
    for cand in candidates:
        original = _replay_candidate(cand, market=market, features=features, dates=dates, folds=folds, cfg=cfg, inverted=False)
        inverted = _replay_candidate(cand, market=market, features=features, dates=dates, folds=folds, cfg=cfg, inverted=True)
        rows.append({
            "candidate": {k: cand.get(k) for k in ("feature", "horizon", "quantile", "event_score", "strict_score")},
            "original": score_candidate(original, gate_cfg),
            "inverted": score_candidate(inverted, gate_cfg),
        })
    inverted_pass = [r for r in rows if r["inverted"]["passed"]]
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": int(len(market)), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1]), "feature_count": int(len(features.columns))},
        "decision": "GO" if inverted_pass else "NO_GO",
        "inverted_passed_count": len(inverted_pass),
        "candidate_count": len(rows),
        "rows": rows,
        "interpretation": "Diagnostic inversion only; a deployable inversion selector would need separate no-leak selection." if inverted_pass else "Simple sign inversion does not rescue the current rolling alpha candidates.",
        "leakage_guard": {
            "fold_rules_fit_before_each_eval_start": True,
            "inversion_is_diagnostic_not_deployable_selector": True,
            "strict_replay_uses_actual_ohlc_bar_by_bar": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit inverted strict replay for rolling alpha candidates")
    p.add_argument("--input-report", required=True)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default=InversionAuditConfig.external_tolerance)
    p.add_argument("--window-size", type=int, default=InversionAuditConfig.window_size)
    p.add_argument("--max-candidates", type=int, default=InversionAuditConfig.max_candidates)
    p.add_argument("--leverage", type=float, default=InversionAuditConfig.leverage)
    p.add_argument("--min-cagr-to-mdd", type=float, default=InversionAuditConfig.min_cagr_to_mdd)
    p.add_argument("--max-strict-mdd-pct", type=float, default=InversionAuditConfig.max_strict_mdd_pct)
    p.add_argument("--min-fold-trades", type=int, default=InversionAuditConfig.min_fold_trades)
    p.add_argument("--min-total-trades", type=int, default=InversionAuditConfig.min_total_trades)
    p.add_argument("--min-positive-folds", type=int, default=InversionAuditConfig.min_positive_folds)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(InversionAuditConfig(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
