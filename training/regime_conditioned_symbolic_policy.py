"""Regime-conditioned symbolic action policy.

This is a leak-safe alternative to repeatedly adding fixed gates to one global
ridge signal.  For each eval month it fits only on rows before that month, builds
multiple symbolic ridge experts on predefined past-only regime slices, scores the
current candidates with each eligible expert, and selects the best action from
the highest-confidence expert.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay
from training.symbolic_action_ridge import FeatureSpace, load_jsonl, row_tokens, target_value, write_jsonl, _candidate_rows_from_preds


NO_TRADE = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "REGIME_POLICY", "confidence": "HIGH"}


def _date(row: dict[str, Any]) -> pd.Timestamp:
    return pd.Timestamp(str(row.get("date")))


def prompt_plain_tokens(row: dict[str, Any]) -> set[str]:
    toks: set[str] = set()
    for tok in row_tokens(row):
        if tok.startswith("regime:"):
            toks.add(tok.removeprefix("regime:"))
        elif tok.startswith("book:"):
            val = tok.removeprefix("book:")
            toks.add("book_" + val.split(":", 1)[0])
        elif tok.startswith("action:"):
            toks.add(tok.removeprefix("action:"))
        elif tok.startswith("raw_action:"):
            toks.add(tok.removeprefix("raw_action:"))
    return toks


def action_key(row: dict[str, Any]) -> tuple[str, str, int]:
    a = row.get("action", {}) if isinstance(row.get("action"), dict) else {}
    return (str(a.get("family", "UNKNOWN")), str(a.get("side", "NONE")).upper(), int(a.get("hold_bars", 0) or 0))


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    require_any: tuple[str, ...]
    require_all: tuple[str, ...] = ()
    forbid_any: tuple[str, ...] = ()

    def matches(self, row: dict[str, Any]) -> bool:
        toks = prompt_plain_tokens(row)
        if self.require_all and not set(self.require_all).issubset(toks):
            return False
        if self.require_any and not any(t in toks for t in self.require_any):
            return False
        if self.forbid_any and any(t in toks for t in self.forbid_any):
            return False
        return True


EXPERTS: tuple[ExpertSpec, ...] = (
    ExpertSpec("global", ()),
    ExpertSpec("trend_up", ("daily_context=up", "weekly_context=up", "three_day_context=up")),
    ExpertSpec("trend_down", ("daily_context=down", "weekly_context=down", "three_day_context=down")),
    ExpertSpec("strong_down", ("short_trend=strong_down", "medium_trend=strong_down", "four_hour_context=strong_down")),
    ExpertSpec("drawdown", ("recent_drawdown=large_drawdown", "higher_tf_drawdown=daily_dd_medium", "higher_tf_drawdown=daily_dd_high")),
    ExpertSpec("low_drawdown", ("recent_drawdown=no_recent_drawdown", "higher_tf_drawdown=daily_dd_low")),
    ExpertSpec("orderflow_buy", ("orderflow=buy_aggression_strong",)),
    ExpertSpec("orderflow_sell", ("orderflow=sell_aggression_strong",)),
    ExpertSpec("range_low", ("location=near_range_low", "weekly_location=weekly_lower")),
    ExpertSpec("range_high", ("location=near_range_high", "weekly_location=weekly_upper")),
    ExpertSpec("dxy_pressure", ("dollar_pressure=dxy_up", "dollar_pressure=dxy_strong_up")),
    ExpertSpec("kimchi_premium", ("kimchi_context=kimchi_soft_premium", "kimchi_context=kimchi_hard_premium")),
    ExpertSpec("book_trend", ("book_higher_tf_momentum", "book_htf_pullback_resume")),
    ExpertSpec("book_reversal", ("book_drawdown_reversal", "book_mean_reversion_stretch", "book_micro_exhaustion_reversal")),
)



def _fit_ridge_stable(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    xtx = x.T @ x
    reg = np.eye(x.shape[1], dtype=np.float64) * float(alpha)
    reg[0, 0] = 1e-6
    rhs = x.T @ y
    a = np.nan_to_num(xtx + reg, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        return np.linalg.solve(a, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a, rhs, rcond=1e-8)[0]

@dataclass
class FittedExpert:
    spec: ExpertSpec
    w: np.ndarray
    train_rows: int
    train_corr: float
    train_mean: float


def _fit_expert_from_matrix(x_all: np.ndarray, y_all: np.ndarray, indexes: list[int], spec: ExpertSpec, *, alpha: float, min_train_rows: int) -> FittedExpert | None:
    if len(indexes) < int(min_train_rows):
        return None
    idx = np.asarray(indexes, dtype=np.int64)
    x = x_all[idx]
    y = y_all[idx]
    w = _fit_ridge_stable(x, y, alpha=float(alpha))
    pred = x @ w
    corr = 0.0 if len(y) < 3 or np.std(y) < 1e-12 or np.std(pred) < 1e-12 else float(np.corrcoef(y, pred)[0, 1])
    return FittedExpert(spec=spec, w=w, train_rows=len(indexes), train_corr=corr, train_mean=float(np.mean(y)))


def _eligible(expert: FittedExpert, candidate_token_sets: list[set[str]]) -> bool:
    if expert.spec.name == "global":
        return True
    return any(_spec_matches_tokens(expert.spec, toks) for toks in candidate_token_sets)


def _spec_matches_tokens(spec: ExpertSpec, toks: set[str]) -> bool:
    if spec.require_all and not set(spec.require_all).issubset(toks):
        return False
    if spec.require_any and not any(t in toks for t in spec.require_any):
        return False
    if spec.forbid_any and any(t in toks for t in spec.forbid_any):
        return False
    return True


def _candidate_pred_rows(rows: list[dict[str, Any]], preds: np.ndarray, expert: FittedExpert) -> list[dict[str, Any]]:
    out = _candidate_rows_from_preds(rows, preds)
    for r in out:
        r["expert"] = expert.spec.name
        r["expert_train_rows"] = expert.train_rows
        r["expert_train_corr"] = expert.train_corr
    return out


def _choose_expert_best(candidates: list[dict[str, Any]], candidate_x: np.ndarray, candidate_token_sets: list[set[str]], experts: list[FittedExpert], *, threshold: float, min_gap: float, expert_margin: float) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for ex in experts:
        if not _eligible(ex, candidate_token_sets):
            continue
        preds = candidate_x @ ex.w
        rows = sorted(_candidate_pred_rows(candidates, preds, ex), key=lambda r: float(r["predicted_utility"]), reverse=True)
        if rows:
            best = rows[0]
            second = float(rows[1]["predicted_utility"]) if len(rows) > 1 else -1e9
            best = {**best, "runner_up_gap": float(best["predicted_utility"] - second)}
            scored.append(best)
    if not scored:
        return {"prediction": dict(NO_TRADE), "predicted_utility": 0.0, "runner_up_gap": 0.0, "selected_action": {}, "expert": "none"}
    scored.sort(key=lambda r: (float(r["predicted_utility"]), float(r.get("runner_up_gap", 0.0)), int(r.get("expert_train_rows", 0))), reverse=True)
    best = scored[0]
    second_score = float(scored[1]["predicted_utility"]) if len(scored) > 1 else -1e9
    if float(best["predicted_utility"]) >= float(threshold) and float(best.get("runner_up_gap", 0.0)) >= float(min_gap) and float(best["predicted_utility"]) - second_score >= float(expert_margin):
        a = best["action"]
        pred = {"gate": "TRADE", "family": a.get("family", "UNKNOWN"), "side": a.get("side", "NONE"), "hold_bars": int(a.get("hold_bars", 0) or 0), "confidence": "HIGH", "expert": best.get("expert")}
    else:
        pred = dict(NO_TRADE)
    return {"prediction": pred, "predicted_utility": float(best["predicted_utility"]), "runner_up_gap": float(best.get("runner_up_gap", 0.0)), "selected_action": best["action"], "actual_utility": float(best.get("actual_utility", 0.0)), "expert": best.get("expert"), "expert_second_gap": float(best["predicted_utility"] - second_score), "expert_train_rows": int(best.get("expert_train_rows", 0)), "expert_train_corr": float(best.get("expert_train_corr", 0.0))}


def rolling_predict(*, history_jsonl: str, eval_jsonl: str, predictions_output: str, summary_output: str, start_date: str, end_date: str, alpha: float = 10000.0, threshold: float = 0.003, min_gap: float = 0.0, expert_margin: float = 0.0, target: str = "net_return", min_feature_count: int = 5, min_train_rows: int = 2000) -> dict[str, Any]:
    history = load_jsonl(history_jsonl)
    ev = load_jsonl(eval_jsonl)
    all_rows = sorted(history + ev, key=_date)
    eval_rows = [r for r in ev if pd.Timestamp(start_date) <= _date(r) < pd.Timestamp(end_date)]
    months = pd.date_range(pd.Timestamp(start_date).replace(day=1), pd.Timestamp(end_date), freq="MS")
    out: list[dict[str, Any]] = []
    month_summaries: list[dict[str, Any]] = []
    for mstart in months:
        mend = mstart + pd.offsets.MonthBegin(1)
        if mstart >= pd.Timestamp(end_date):
            continue
        train = [r for r in all_rows if _date(r) < mstart]
        test = [r for r in eval_rows if mstart <= _date(r) < min(mend, pd.Timestamp(end_date))]
        if not test:
            continue
        fs = FeatureSpace.fit(train, min_count=int(min_feature_count))
        x_train = fs.matrix(train)
        y_train = np.asarray([target_value(r, target=target) for r in train], dtype=np.float64)
        train_tokens = [prompt_plain_tokens(r) for r in train]
        fitted: list[FittedExpert] = []
        for spec in EXPERTS:
            if spec.name == "global":
                idxs = list(range(len(train)))
            else:
                idxs = [i for i, toks in enumerate(train_tokens) if _spec_matches_tokens(spec, toks)]
            ex = _fit_expert_from_matrix(x_train, y_train, idxs, spec, alpha=alpha, min_train_rows=min_train_rows)
            if ex is not None:
                fitted.append(ex)
        groups: dict[tuple[str, int], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for i, r in enumerate(test):
            groups[(str(r.get("date")), int(r.get("signal_pos", -1) or -1))].append((i, r))
        x_test = fs.matrix(test)
        test_tokens = [prompt_plain_tokens(r) for r in test]
        month_rows: list[dict[str, Any]] = []
        for key in sorted(groups):
            idxs = [i for i, _ in groups[key]]
            candidates = [r for _, r in groups[key]]
            chosen = _choose_expert_best(candidates, x_test[np.asarray(idxs, dtype=np.int64)], [test_tokens[i] for i in idxs], fitted, threshold=threshold, min_gap=min_gap, expert_margin=expert_margin)
            month_rows.append({"date": key[0], "signal_pos": key[1], **chosen})
        out.extend(month_rows)
        month_summaries.append({
            "month": str(mstart.date())[:7],
            "train_rows": len(train),
            "candidate_rows": len(test),
            "signals": len(month_rows),
            "trade_signals": sum(1 for r in month_rows if r["prediction"].get("gate") == "TRADE"),
            "features": len(fs.vocab),
            "experts": {ex.spec.name: {"train_rows": ex.train_rows, "train_corr": ex.train_corr, "train_mean": ex.train_mean} for ex in fitted},
            "chosen_experts": dict(Counter(str(r.get("expert")) for r in month_rows if r["prediction"].get("gate") == "TRADE")),
        })
    write_jsonl(predictions_output, out)
    report = {"config": {"alpha": alpha, "threshold": threshold, "min_gap": min_gap, "expert_margin": expert_margin, "target": target, "min_feature_count": min_feature_count, "min_train_rows": min_train_rows, "expert_names": [e.name for e in EXPERTS]}, "history_jsonl": history_jsonl, "eval_jsonl": eval_jsonl, "predictions_output": predictions_output, "period": {"start": start_date, "end": end_date}, "rows": len(out), "months": month_summaries, "leakage_guard": {"each_month_fit_uses_rows_before_month_start_only": True, "current_month_labels_not_used_for_current_month": True, "expert_slices_use_past_only_prompt_tokens": True, "config_fixed_before_eval": True}}
    Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monthly prior-only regime-conditioned symbolic ridge policy")
    p.add_argument("--history-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--predictions-output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--alpha", type=float, default=10000.0)
    p.add_argument("--threshold", type=float, default=0.003)
    p.add_argument("--min-gap", type=float, default=0.0)
    p.add_argument("--expert-margin", type=float, default=0.0)
    p.add_argument("--target", choices=["utility", "net_return", "risk_adjusted"], default="net_return")
    p.add_argument("--min-feature-count", type=int, default=5)
    p.add_argument("--min-train-rows", type=int, default=2000)
    p.add_argument("--market-csv", default="")
    p.add_argument("--backtest-output", default="")
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--trade-take-profit-pct", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    rep = rolling_predict(history_jsonl=a.history_jsonl, eval_jsonl=a.eval_jsonl, predictions_output=a.predictions_output, summary_output=a.summary_output, start_date=a.start_date, end_date=a.end_date, alpha=a.alpha, threshold=a.threshold, min_gap=a.min_gap, expert_margin=a.expert_margin, target=a.target, min_feature_count=a.min_feature_count, min_train_rows=a.min_train_rows)
    out: dict[str, Any] = {"summary": rep}
    if a.market_csv and a.backtest_output:
        bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=a.predictions_output, market_csv=a.market_csv, output=a.backtest_output, leverage=float(a.leverage), trade_take_profit_pct=float(a.trade_take_profit_pct)))
        out = {"summary": rep, "backtest": {"period": bt["period"], "sim": bt["sim"], "trade_stats": bt["trade_stats"]}}
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
