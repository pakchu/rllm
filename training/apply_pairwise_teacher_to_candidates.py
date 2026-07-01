"""Apply past-only pairwise teacher to choose one trade candidate per timestamp.

This turns the weak pairwise teacher into live-style prediction rows so the idea
can be audited with portfolio/backtest tooling.  For each eval period, teacher
stats are built from previous pairwise rows only, then pairwise majority votes
select a candidate among same-timestamp frozen alpha candidates.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.linear_alpha_candidate_pairwise_teacher import (
    _build_stats,
    _candidate_id_from_prompt,
    _load,
    _periods,
    _score_candidate,
)


@dataclass(frozen=True)
class ApplyTeacherConfig:
    pairwise_inputs: str
    output_jsonl: str
    summary_output: str = ""
    period: str = "halfyear"
    min_train_rows: int = 1000
    smoothing: float = 8.0
    train_window_periods: int = 0
    min_vote_margin: float = 0.0


def _pred(row: dict[str, Any]) -> dict[str, Any]:
    pred = row.get("prediction")
    return pred if isinstance(pred, dict) else {}


def _candidate_key_from_pair(row: dict[str, Any], label: str) -> str:
    return _candidate_id_from_prompt(str(row.get("prompt", "")), label)


def _candidate_payload(row: dict[str, Any], label: str) -> dict[str, Any] | None:
    meta = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    key = "candidate_a" if label == "A" else "candidate_b"
    cand = meta.get(key)
    if not isinstance(cand, dict):
        return None
    pred = cand.get("prediction") if isinstance(cand.get("prediction"), dict) else {}
    return {"id": cand.get("id"), "prediction": pred, "score": cand.get("score"), "path": cand.get("path"), "utility": cand.get("utility")}


def run(cfg: ApplyTeacherConfig) -> dict[str, Any]:
    rows = _load(cfg.pairwise_inputs)
    by_period = _periods(rows, cfg.period)
    periods = list(by_period)
    out_rows: list[dict[str, Any]] = []
    period_reports: list[dict[str, Any]] = []
    for idx, period in enumerate(periods):
        train_periods = periods[:idx]
        if cfg.train_window_periods > 0:
            train_periods = train_periods[-int(cfg.train_window_periods):]
        train_rows = [r for p in train_periods for r in by_period[p]]
        eval_rows = by_period[period]
        if len(train_rows) < int(cfg.min_train_rows):
            continue
        stats = _build_stats(train_rows)
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in eval_rows:
            grouped[int(row.get("signal_pos", -1) or -1)].append(row)
        selected = 0
        skipped = Counter()
        pred_counts = Counter()
        for pos, pair_rows in sorted(grouped.items()):
            votes: Counter[str] = Counter()
            payloads: dict[str, dict[str, Any]] = {}
            for row in pair_rows:
                sa = _score_candidate(row, "A", stats, float(cfg.smoothing))
                sb = _score_candidate(row, "B", stats, float(cfg.smoothing))
                winner = "A" if sa >= sb else "B"
                loser = "B" if winner == "A" else "A"
                margin = abs(sa - sb)
                if margin < float(cfg.min_vote_margin):
                    skipped["low_pair_margin"] += 1
                    continue
                winner_id = _candidate_key_from_pair(row, winner)
                loser_id = _candidate_key_from_pair(row, loser)
                votes[winner_id] += 1
                votes[loser_id] -= 1
                payload = _candidate_payload(row, winner)
                if payload is not None:
                    payloads[str(payload.get("id"))] = payload
            if not votes:
                skipped["no_votes"] += 1
                continue
            best_id, best_vote = votes.most_common(1)[0]
            if best_vote <= 0:
                skipped["non_positive_vote"] += 1
                continue
            payload = payloads.get(best_id)
            if payload is None:
                skipped["missing_payload"] += 1
                continue
            pred = dict(payload["prediction"])
            if str(pred.get("gate", "")).upper() != "TRADE":
                skipped["winner_not_trade"] += 1
                continue
            # Use the first row timestamp for this signal_pos.
            base = pair_rows[0]
            out_rows.append({
                "date": base.get("date"),
                "signal_pos": int(pos),
                "prediction": {**pred, "family": "pairwise_teacher_candidate_selector", "confidence": "MEDIUM"},
                "position_scale": 1.0,
                "score": float(best_vote),
                "selected_candidate_id": best_id,
                "period": period,
                "teacher_train_periods": train_periods,
            })
            pred_counts[str(pred.get("side", "NONE"))] += 1
            selected += 1
        period_reports.append({"period": period, "train_periods": train_periods, "train_rows": len(train_rows), "pair_rows": len(eval_rows), "selected": selected, "pred_counts": dict(pred_counts), "skipped": dict(skipped)})
    Path(cfg.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_jsonl).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out_rows) + ("\n" if out_rows else ""))
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "rows_out": len(out_rows),
        "periods": period_reports,
        "leakage_guard": {"each_period_teacher_uses_only_previous_period_pairwise_rows": True},
    }
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply pairwise teacher to select one candidate per timestamp")
    p.add_argument("--pairwise-inputs", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--period", choices=["year", "halfyear", "quarter"], default=ApplyTeacherConfig.period)
    p.add_argument("--min-train-rows", type=int, default=ApplyTeacherConfig.min_train_rows)
    p.add_argument("--smoothing", type=float, default=ApplyTeacherConfig.smoothing)
    p.add_argument("--train-window-periods", type=int, default=ApplyTeacherConfig.train_window_periods)
    p.add_argument("--min-vote-margin", type=float, default=ApplyTeacherConfig.min_vote_margin)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(ApplyTeacherConfig(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
