"""Rolling utility ranker for event-context RLLM rows.

This replaces exact context matching with a lightweight generalizing utility
model over symbolic event/regime tokens.  For each target month:
- fit categorical ridge on rows before the validation window;
- choose threshold/gap on the immediately previous validation window;
- predict the target month without using target labels;
- evaluate all target-month predictions through the standard strict overlay.

This is still a diagnostic/ranker, not an LLM inference result.  It is the
pre-Gemma check for whether event-context tokens carry learnable utility.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

NO_TRADE = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "EVENT_CONTEXT_RIDGE", "confidence": "HIGH"}


@dataclass(frozen=True)
class RollingEventContextUtilityCfg:
    input_jsonl: str
    market_csv: str
    predictions_output: str
    summary_output: str
    backtest_output: str
    start_date: str = "2024-01-01"
    end_date: str = "2026-06-01"
    train_days: int = 1095
    validation_days: int = 180
    alpha: float = 100.0
    min_feature_count: int = 5
    thresholds: str = "0.05,0.10,0.15,0.20,0.30"
    min_gaps: str = "0.00,0.05,0.10,0.20"
    min_validation_trades: int = 3
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    cooldown_bars: int = 0


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: (str(r.get("date")), int(r.get("signal_pos", -1) or -1)))
    return rows


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _dt(row: dict[str, Any]) -> pd.Timestamp:
    return pd.Timestamp(str(row["date"]))


def _month_starts(start: str, end: str) -> list[pd.Timestamp]:
    start_ts = pd.Timestamp(start).normalize().replace(day=1)
    end_ts = pd.Timestamp(end)
    return [m for m in pd.date_range(start_ts, end_ts, freq="MS") if m < end_ts]


def _base_tokens(row: dict[str, Any]) -> list[str]:
    toks = ["bias"]
    state = row.get("state_tokens") if isinstance(row.get("state_tokens"), dict) else {}
    for k, v in sorted(state.items()):
        toks.append(f"state:{k}={v}")
    # Coarser interactions that should generalize better than exact context IDs.
    for a, b in (
        ("trend_alignment", "risk_state"),
        ("pa_event_pressure", "risk_state"),
        ("pa_long_window_event", "trend_alignment"),
        ("pa_downside_reclaim", "funding_zscore"),
        ("pa_upside_rejection", "funding_zscore"),
        ("range_pos", "window_drawdown"),
    ):
        if a in state and b in state:
            toks.append(f"x:{a}={state[a]}|{b}={state[b]}")
    return toks


def _candidate_tokens(row: dict[str, Any], side: str) -> list[str]:
    side = str(side).upper()
    toks = list(_base_tokens(row))
    toks.append(f"action:side={side}")
    for tok in list(toks):
        if tok.startswith("state:pa_") or tok.startswith("state:trend_alignment") or tok.startswith("state:risk_state") or tok.startswith("state:htf_") or tok.startswith("x:"):
            toks.append(f"side_x:{side}|{tok}")
    return toks


def _candidate_value_pct(row: dict[str, Any], side: str) -> float:
    audit = row.get("reward_audit") if isinstance(row.get("reward_audit"), dict) else {}
    val = audit.get(str(side).upper()) if isinstance(audit.get(str(side).upper()), dict) else {}
    try:
        return float(val.get("net_return_pct", 0.0))
    except Exception:
        return 0.0


@dataclass
class FeatureSpace:
    vocab: dict[str, int]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, rows: list[dict[str, Any]], *, min_count: int) -> "FeatureSpace":
        counts: Counter[str] = Counter()
        for row in rows:
            for side in ("LONG", "SHORT"):
                counts.update(_candidate_tokens(row, side))
        vocab = {"bias": 0}
        for tok, cnt in sorted(counts.items()):
            if tok == "bias":
                continue
            if cnt >= int(min_count):
                vocab[tok] = len(vocab)
        dummy = cls(vocab=vocab, mean=np.zeros(len(vocab)), std=np.ones(len(vocab)))
        x = dummy.matrix_candidates(rows, scale=False)
        mean = x.mean(axis=0) if len(x) else np.zeros(len(vocab))
        std = x.std(axis=0) if len(x) else np.ones(len(vocab))
        std[std < 1e-9] = 1.0
        mean[0] = 0.0
        std[0] = 1.0
        return cls(vocab=vocab, mean=mean, std=std)

    def matrix_for(self, rows: list[dict[str, Any]], sides: Iterable[str], *, scale: bool = True) -> np.ndarray:
        pairs = list(zip(rows, sides))
        x = np.zeros((len(pairs), len(self.vocab)), dtype=np.float64)
        for i, (row, side) in enumerate(pairs):
            for tok in _candidate_tokens(row, side):
                j = self.vocab.get(tok)
                if j is not None:
                    x[i, j] = 1.0
        if scale:
            x = (x - self.mean) / self.std
        return x

    def matrix_candidates(self, rows: list[dict[str, Any]], *, scale: bool = True) -> np.ndarray:
        pair_rows: list[dict[str, Any]] = []
        sides: list[str] = []
        for row in rows:
            pair_rows.extend([row, row])
            sides.extend(["LONG", "SHORT"])
        return self.matrix_for(pair_rows, sides, scale=scale)


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    if x.shape[0] == 0:
        raise ValueError("no train candidates")
    reg = np.eye(x.shape[1], dtype=np.float64) * float(alpha)
    reg[0, 0] = 1e-9
    return np.linalg.pinv(x.T @ x + reg) @ x.T @ y


def _fit_model(rows: list[dict[str, Any]], cfg: RollingEventContextUtilityCfg) -> tuple[FeatureSpace, np.ndarray, dict[str, Any]]:
    fs = FeatureSpace.fit(rows, min_count=int(cfg.min_feature_count))
    x = fs.matrix_candidates(rows)
    y_vals: list[float] = []
    for row in rows:
        y_vals.append(_candidate_value_pct(row, "LONG"))
        y_vals.append(_candidate_value_pct(row, "SHORT"))
    y = np.asarray(y_vals, dtype=np.float64)
    w = _fit_ridge(x, y, float(cfg.alpha))
    pred = x @ w
    corr = 0.0 if len(y) < 2 or float(np.std(y)) < 1e-12 or float(np.std(pred)) < 1e-12 else float(np.corrcoef(y, pred)[0, 1])
    rmse = math.sqrt(float(np.mean((pred - y) ** 2))) if len(y) else 0.0
    return fs, w, {"train_rows": len(rows), "candidate_rows": int(len(y)), "features": len(fs.vocab), "train_corr": corr, "train_rmse_pct": rmse}


def _predict_rows(rows: list[dict[str, Any]], fs: FeatureSpace, w: np.ndarray, *, threshold: float, min_gap: float) -> list[dict[str, Any]]:
    if not rows:
        return []
    x_long = fs.matrix_for(rows, ["LONG"] * len(rows))
    x_short = fs.matrix_for(rows, ["SHORT"] * len(rows))
    p_long = x_long @ w
    p_short = x_short @ w
    out: list[dict[str, Any]] = []
    for row, long_score, short_score in zip(rows, p_long, p_short):
        if float(long_score) >= float(short_score):
            side = "LONG"; best = float(long_score); second = float(short_score)
        else:
            side = "SHORT"; best = float(short_score); second = float(long_score)
        gap = best - second
        if best >= float(threshold) and gap >= float(min_gap):
            pred = {"gate": "TRADE", "side": side, "hold_bars": 288, "family": "EVENT_CONTEXT_RIDGE", "confidence": "HIGH"}
        else:
            pred = dict(NO_TRADE)
        out.append({
            "date": row.get("date"),
            "signal_pos": int(row.get("signal_pos", -1) or -1),
            "prediction": pred,
            "predicted_long_pct": float(long_score),
            "predicted_short_pct": float(short_score),
            "predicted_best_pct": float(best),
            "runner_up_gap_pct": float(gap),
            "actual_long_pct": _candidate_value_pct(row, "LONG"),
            "actual_short_pct": _candidate_value_pct(row, "SHORT"),
            "split": row.get("split"),
        })
    return out


def _score_bt(bt: dict[str, Any], min_trades: int) -> float:
    sim = bt.get("sim", {})
    trades = int(sim.get("trade_entries", 0) or 0)
    if trades < int(min_trades):
        return -999.0 + trades / 1000.0
    cagr = float(sim.get("cagr_pct", -100.0))
    ratio = float(sim.get("cagr_to_strict_mdd", -999.0))
    mdd = float(sim.get("strict_mdd_pct", 999.0))
    p = float(bt.get("trade_stats", {}).get("p_value_mean_ret_approx", 1.0))
    if cagr <= 0:
        return -500.0 + trades / 1000.0 + cagr / 100.0
    return ratio + 0.01 * cagr - 0.02 * max(0.0, mdd - 15.0) - p + min(1.0, trades / 50.0)


def _run_temp_backtest(rows: list[dict[str, Any]], market_csv: str, cfg: RollingEventContextUtilityCfg, path: Path) -> dict[str, Any]:
    pred_path = path.with_suffix(".jsonl")
    out_path = path.with_suffix(".bt.json")
    _write_jsonl(pred_path, rows)
    return run_overlay(OnlineRiskOverlayConfig(
        predictions_jsonl=str(pred_path),
        market_csv=market_csv,
        output=str(out_path),
        leverage=float(cfg.leverage),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
        entry_delay_bars=int(cfg.entry_delay_bars),
        cooldown_bars=int(cfg.cooldown_bars),
    ))


def run(cfg: RollingEventContextUtilityCfg) -> dict[str, Any]:
    rows = _read_jsonl(cfg.input_jsonl)
    start = pd.Timestamp(cfg.start_date)
    end = pd.Timestamp(cfg.end_date)
    thresholds = _parse_floats(cfg.thresholds)
    min_gaps = _parse_floats(cfg.min_gaps)
    all_predictions: list[dict[str, Any]] = []
    months: list[dict[str, Any]] = []
    work = Path(cfg.summary_output).with_suffix("").parent / "rolling_event_context_utility_work"
    work.mkdir(parents=True, exist_ok=True)

    for mstart in _month_starts(cfg.start_date, cfg.end_date):
        mend = min(mstart + pd.offsets.MonthBegin(1), end)
        val_start = mstart - pd.Timedelta(days=int(cfg.validation_days))
        train_start = val_start - pd.Timedelta(days=int(cfg.train_days))
        train = [r for r in rows if train_start <= _dt(r) < val_start]
        val = [r for r in rows if val_start <= _dt(r) < mstart]
        target = [r for r in rows if max(start, mstart) <= _dt(r) < mend]
        if not target or len(train) < 100 or len(val) < 10:
            continue
        fs, w, fit = _fit_model(train, cfg)
        candidates: list[dict[str, Any]] = []
        for th in thresholds:
            for gap in min_gaps:
                val_preds = _predict_rows(val, fs, w, threshold=th, min_gap=gap)
                tag = f"{mstart:%Y%m}_th{th}_gap{gap}".replace(".", "p")
                bt = _run_temp_backtest(val_preds, cfg.market_csv, cfg, work / tag)
                candidates.append({"threshold": th, "min_gap": gap, "score": _score_bt(bt, cfg.min_validation_trades), "validation": {"sim": bt["sim"], "trade_stats": bt["trade_stats"]}})
        selected = max(candidates, key=lambda r: (float(r["score"]), int(r["validation"]["sim"].get("trade_entries", 0))))
        target_preds = _predict_rows(target, fs, w, threshold=float(selected["threshold"]), min_gap=float(selected["min_gap"]))
        all_predictions.extend(target_preds)
        months.append({
            "month": f"{mstart:%Y-%m}",
            "train_start": str(train_start),
            "validation_start": str(val_start),
            "selection_cutoff_exclusive": str(mstart),
            "target_start": str(mstart),
            "target_end_exclusive": str(mend),
            "fit": fit,
            "validation_rows": len(val),
            "target_rows": len(target),
            "selected": selected,
            "target_trade_signals": sum(1 for r in target_preds if r["prediction"].get("gate") == "TRADE"),
        })
        print(json.dumps({"month": f"{mstart:%Y-%m}", "selected": {"threshold": selected["threshold"], "min_gap": selected["min_gap"], "score": selected["score"]}, "target_trades_raw": months[-1]["target_trade_signals"], "fit_features": fit["features"]}, ensure_ascii=False), flush=True)

    _write_jsonl(cfg.predictions_output, all_predictions)
    final_bt = run_overlay(OnlineRiskOverlayConfig(
        predictions_jsonl=cfg.predictions_output,
        market_csv=cfg.market_csv,
        output=cfg.backtest_output,
        leverage=float(cfg.leverage),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
        entry_delay_bars=int(cfg.entry_delay_bars),
        cooldown_bars=int(cfg.cooldown_bars),
    ))
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "rows": len(all_predictions),
        "months": months,
        "backtest": {"period": final_bt["period"], "sim": final_bt["sim"], "trade_stats": final_bt["trade_stats"]},
        "leakage_guard": {
            "each_month_fit_uses_rows_before_validation_only": True,
            "threshold_selected_on_prior_validation_only": True,
            "target_month_not_used_for_fit_or_selection": True,
            "prompts_and_state_tokens_are_causal": True,
            "not_llm_inference_result": True,
        },
    }
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rolling event-context utility ridge ranker")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--predictions-output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--backtest-output", required=True)
    p.add_argument("--start-date", default=RollingEventContextUtilityCfg.start_date)
    p.add_argument("--end-date", default=RollingEventContextUtilityCfg.end_date)
    p.add_argument("--train-days", type=int, default=RollingEventContextUtilityCfg.train_days)
    p.add_argument("--validation-days", type=int, default=RollingEventContextUtilityCfg.validation_days)
    p.add_argument("--alpha", type=float, default=RollingEventContextUtilityCfg.alpha)
    p.add_argument("--min-feature-count", type=int, default=RollingEventContextUtilityCfg.min_feature_count)
    p.add_argument("--thresholds", default=RollingEventContextUtilityCfg.thresholds)
    p.add_argument("--min-gaps", default=RollingEventContextUtilityCfg.min_gaps)
    p.add_argument("--min-validation-trades", type=int, default=RollingEventContextUtilityCfg.min_validation_trades)
    p.add_argument("--leverage", type=float, default=RollingEventContextUtilityCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=RollingEventContextUtilityCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=RollingEventContextUtilityCfg.slippage_rate)
    p.add_argument("--entry-delay-bars", type=int, default=RollingEventContextUtilityCfg.entry_delay_bars)
    p.add_argument("--cooldown-bars", type=int, default=RollingEventContextUtilityCfg.cooldown_bars)
    return p.parse_args()


def main() -> None:
    report = run(RollingEventContextUtilityCfg(**vars(parse_args())))
    sim = report["backtest"]["sim"]
    stats = report["backtest"]["trade_stats"]
    print(json.dumps({
        "predictions_output": report["config"]["predictions_output"],
        "months": len(report["months"]),
        "rows": report["rows"],
        "result": {
            "cagr_pct": sim["cagr_pct"],
            "strict_mdd_pct": sim["strict_mdd_pct"],
            "cagr_to_strict_mdd": sim["cagr_to_strict_mdd"],
            "trade_entries": sim["trade_entries"],
            "mean_trade_ret_pct": stats.get("mean_trade_ret_pct"),
            "p_value_mean_ret_approx": stats.get("p_value_mean_ret_approx"),
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
