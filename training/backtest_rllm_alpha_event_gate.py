"""Backtest OOS RLLM event-gate predictions at fixed 0.5 gross."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from training import evaluate_high_volatility_spot_adverse_underwater_duration_relay_economics as econ
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


STAGES = {
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def train_scores(results: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for path in results.glob("high_volatility_*_train_economics_*.json"):
        slug = path.name.split("_train_economics_")[0]
        value = json.loads(path.read_text())
        primary = value.get("primary", {}).get("base", {})
        score = primary.get("cagr_to_strict_mdd")
        if isinstance(score, (int, float)):
            scores[slug] = float(score)
    return scores


def attach_predictions(events: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(events) != len(predictions):
        raise RuntimeError("prediction/event length mismatch")
    out = []
    for event, pred in zip(events, predictions):
        row = dict(event)
        row["prediction"] = str(pred["prediction"])
        out.append(row)
    return out


def select_clock(rows: list[dict[str, Any]], scores: dict[str, float], *, gated: bool) -> pd.DataFrame:
    eligible = []
    for row in rows:
        if gated and row.get("prediction") != "TRADE":
            continue
        item = dict(row)
        item["entry_time"] = pd.Timestamp(item["entry_time"])
        item["exit_time"] = pd.Timestamp(item["exit_time"])
        item["train_score"] = float(scores.get(str(item["slug"]), float("-inf")))
        eligible.append(item)
    eligible.sort(key=lambda x: (x["entry_time"], -x["train_score"], str(x["policy_id"])))
    selected = []
    reserved_until: pd.Timestamp | None = None
    index = 0
    while index < len(eligible):
        entry = eligible[index]["entry_time"]
        same = []
        while index < len(eligible) and eligible[index]["entry_time"] == entry:
            same.append(eligible[index]); index += 1
        if reserved_until is not None and entry < reserved_until:
            continue
        winner = max(same, key=lambda x: (x["train_score"], str(x["policy_id"])))
        reserved_until = winner["exit_time"]
        selected.append({
            "candidate": winner["policy_id"], "control": "rllm_gate" if gated else "ungated",
            "split": winner["stage"], "entry_time": winner["entry_time"], "exit_time": winner["exit_time"],
            "side": int(winner["side"]), "train_score": winner["train_score"], "slug": winner["slug"],
        })
    return pd.DataFrame(selected, columns=("candidate","control","split","entry_time","exit_time","side","train_score","slug"))


def _metrics(clock: pd.DataFrame, market: pd.DataFrame, funding: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    bare = clock[["entry_time", "exit_time", "side"]]
    return econ.evaluate_primary(bare, market, funding, start, end)


def run(data_dir: Path, results_dir: Path, output: Path, clock_dir: Path) -> dict[str, Any]:
    scores = train_scores(results_dir)
    stage_reports = {}
    selected_by_stage = {}
    baseline_by_stage = {}
    markets, funds = [], []
    for stage, (start, end) in STAGES.items():
        events = _rows(data_dir / f"rllm_alpha_event_gate_{stage}_trainpassed_2026-08-19.jsonl")
        preds = _rows(results_dir / f"rllm_alpha_event_gate_{stage}_predictions_2026-08-19.jsonl")
        attached = attach_predictions(events, preds)
        selected = select_clock(attached, scores, gated=True)
        baseline = select_clock(attached, scores, gated=False)
        market, funding, source = econ.load_sources(stage, start, end)
        selected_by_stage[stage], baseline_by_stage[stage] = selected, baseline
        markets.append(market); funds.append(funding)
        stage_reports[stage] = {
            "source": source,
            "event_rows": len(events),
            "predicted_trade_rows": sum(r["prediction"] == "TRADE" for r in attached),
            "selected_policy_counts": dict(Counter(selected["candidate"])),
            "selected": _metrics(selected, market, funding, start, end),
            "ungated": _metrics(baseline, market, funding, start, end),
        }
        clock_dir.mkdir(parents=True, exist_ok=True)
        _write_gzip_csv(selected, clock_dir / f"rllm_alpha_event_gate_{stage}_clock_2026-08-19.csv.gz")

    all_selected = pd.concat(selected_by_stage.values(), ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    all_baseline = pd.concat(baseline_by_stage.values(), ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    market = pd.concat(markets, ignore_index=True).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    funding = pd.concat(funds, ignore_index=True).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    start, end = STAGES["test"][0], STAGES["final"][1]
    report = {
        "protocol_version": "rllm_alpha_event_gate_oos_backtest_v1",
        "policy": "earliest available event; same-entry highest frozen 2023 train CAGR/MDD; 0.5 gross; skip overlaps",
        "stages": stage_reports,
        "combined_oos": {
            "window": [str(start), str(end)],
            "selected": _metrics(all_selected, market, funding, start, end),
            "ungated": _metrics(all_baseline, market, funding, start, end),
            "selected_policy_counts": dict(Counter(all_selected["candidate"])),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n")
    return report


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--data-dir",type=Path,default=Path("data"));p.add_argument("--results-dir",type=Path,default=Path("results"));p.add_argument("--output",type=Path,required=True);p.add_argument("--clock-dir",type=Path,default=Path("data/rllm_alpha_event_gate_oos_clocks_2026-08-19"));print(json.dumps(run(**vars(p.parse_args())),indent=2,ensure_ascii=False,default=str))

if __name__=="__main__":main()
