"""Apply a prior-fold-selected regime abstain gate to walk-forward predictions.

For each fold, candidate threshold rules are scored only on earlier traded folds
with known outcomes. The selected rule is then applied to the current fold using
only pre-test/validation regime metrics known before current test trading.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class OnlineRegimeGateCfg:
    fold_regime_audit: str
    predictions_root: str
    market_csv: str
    output: str
    work_dir: str = "results/event_candidate_online_regime_gate"
    min_prior_folds: int = 4
    candidate_features: str = "pretest_range_pos,val_full_range_pos,val_tail_range_pos"
    candidate_thresholds: str = "0.70,0.75,0.80,0.85"
    min_prior_improvement: float = 0.0
    leverage: float = 1.0
    entry_delay_bars: int = 1


def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _parse_strings(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _rule_applies(row: dict[str, Any], rule: dict[str, Any]) -> bool:
    feature = str(rule["feature"])
    threshold = float(rule["threshold"])
    direction = str(rule.get("direction", "above"))
    value = float(row.get(feature, 0.0) or 0.0)
    return value >= threshold if direction == "above" else value <= threshold


def _score_rule(prior: list[dict[str, Any]], rule: dict[str, Any]) -> dict[str, Any]:
    kept = [r for r in prior if not _rule_applies(r, rule)]
    skipped = [r for r in prior if _rule_applies(r, rule)]
    base = sum(float(r.get("test_ratio", 0.0) or 0.0) for r in prior)
    score = sum(float(r.get("test_ratio", 0.0) or 0.0) for r in kept)
    bad_skipped = sum(1 for r in skipped if bool(r.get("is_bad")))
    good_skipped = sum(1 for r in skipped if not bool(r.get("is_bad")))
    return {
        "rule": rule,
        "prior_folds": len(prior),
        "kept_folds": len(kept),
        "skipped_folds": len(skipped),
        "bad_skipped": bad_skipped,
        "good_skipped": good_skipped,
        "base_ratio_sum": base,
        "kept_ratio_sum": score,
        "improvement": score - base,
    }


def _select_rule(prior: list[dict[str, Any]], cfg: OnlineRegimeGateCfg) -> dict[str, Any] | None:
    if len(prior) < int(cfg.min_prior_folds):
        return None
    candidates: list[dict[str, Any]] = []
    for feature in _parse_strings(cfg.candidate_features):
        for threshold in _parse_floats(cfg.candidate_thresholds):
            candidates.append(_score_rule(prior, {"feature": feature, "threshold": threshold, "direction": "above"}))
    # Prefer rules that improve prior ratio without skipping good folds; then larger improvement.
    candidates.sort(key=lambda x: (int(x["good_skipped"]) == 0, float(x["improvement"]), int(x["bad_skipped"]), -int(x["skipped_folds"])), reverse=True)
    best = candidates[0] if candidates else None
    if not best or float(best["improvement"]) < float(cfg.min_prior_improvement):
        return None
    return best


def _no_trade_copy(src: Path, dst: Path, reason: str) -> None:
    rows = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["prediction"] = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "confidence": "LOW", "family": "online_regime_gate", "reason": reason}
        row["position_scale"] = 0.0
        rows.append(row)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _fold_prediction_path(root: Path, fold_id: int) -> Path:
    return root / f"fold{fold_id:03d}" / f"fold{fold_id:03d}_test_predictions.jsonl"


def run(cfg: OnlineRegimeGateCfg) -> dict[str, Any]:
    audit = json.loads(Path(cfg.fold_regime_audit).read_text())
    audited_folds = {int(r["fold_id"]): r for r in audit.get("folds", [])}
    root = Path(cfg.predictions_root)
    all_fold_ids = sorted(int(p.parent.name.replace("fold", "")) for p in root.glob("fold*/fold*_test_predictions.jsonl"))
    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    prior: list[dict[str, Any]] = []
    output_files: list[str] = []
    decisions: list[dict[str, Any]] = []
    for fid in all_fold_ids:
        row = audited_folds.get(fid)
        src = _fold_prediction_path(root, fid)
        dst = work / f"fold{fid:03d}_test_predictions.jsonl"
        selected = _select_rule(prior, cfg) if row is not None else None
        abstain = bool(row is not None and selected and _rule_applies(row, selected["rule"]))
        if abstain:
            _no_trade_copy(src, dst, reason=f"prior_selected_{selected['rule']['feature']}_above_{selected['rule']['threshold']}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src.read_text())
        output_files.append(str(dst))
        decisions.append({"fold_id": fid, "test_start": None if row is None else row.get("test_start"), "selected_rule": selected, "abstained": abstain, "row_metrics": {} if row is None else {k: row.get(k) for k in _parse_strings(cfg.candidate_features)}})
        if row is not None:
            prior.append(row)
    bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=",".join(output_files), market_csv=cfg.market_csv, output=str(work / "aggregate_backtest.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars))
    out = {
        "config": asdict(cfg),
        "decisions": decisions,
        "prediction_files": output_files,
        "backtest": {"sim": bt["sim"], "trade_stats": bt["trade_stats"]},
        "leakage_guard": {"current_fold_rule_selected_from_prior_completed_folds_only": True, "current_fold_metrics_known_before_test": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Online prior-fold-selected regime gate")
    p.add_argument("--fold-regime-audit", required=True)
    p.add_argument("--predictions-root", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", default=OnlineRegimeGateCfg.work_dir)
    p.add_argument("--min-prior-folds", type=int, default=OnlineRegimeGateCfg.min_prior_folds)
    p.add_argument("--candidate-features", default=OnlineRegimeGateCfg.candidate_features)
    p.add_argument("--candidate-thresholds", default=OnlineRegimeGateCfg.candidate_thresholds)
    p.add_argument("--min-prior-improvement", type=float, default=OnlineRegimeGateCfg.min_prior_improvement)
    p.add_argument("--leverage", type=float, default=OnlineRegimeGateCfg.leverage)
    p.add_argument("--entry-delay-bars", type=int, default=OnlineRegimeGateCfg.entry_delay_bars)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(OnlineRegimeGateCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
