"""Deductive symbolic selector for linear-alpha candidate pairs.

This module uses explicit signal-time premises instead of learned numeric
classification.  The goal is to test whether LLM-style deductive reasoning can be
made useful: convert market state + candidate descriptors into symbolic premises,
apply transparent rules, and select a trade candidate only when the conclusion is
supported.

No future labels are used for selection.  Future path labels may exist in the
pairwise rows only for offline audit/backtest metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from training.linear_alpha_meta_stability_diagnostic import _date, _period_key


@dataclass(frozen=True)
class DeductiveSelectorConfig:
    pairwise_inputs: str
    output_jsonl: str
    summary_output: str = ""
    period: str = "halfyear"
    min_score: float = 1.0
    min_vote: int = 1
    max_candidates_per_timestamp: int = 1


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _load(inputs: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in str(inputs).split(","):
        path = raw.strip()
        if path:
            rows.extend(_read_jsonl(path))
    return sorted(rows, key=lambda r: (str(r.get("date", "")), int(r.get("signal_pos", 0) or 0)))


def _candidate_from_pair(row: dict[str, Any], label: str) -> dict[str, Any] | None:
    meta = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    key = "candidate_a" if label == "A" else "candidate_b"
    cand = meta.get(key)
    if not isinstance(cand, dict):
        return None
    pred = cand.get("prediction") if isinstance(cand.get("prediction"), dict) else {}
    if str(pred.get("gate", "")).upper() != "TRADE":
        return None
    return {
        "id": str(cand.get("id", "unknown")),
        "prediction": pred,
        "score": float(cand.get("score", 0.0) or 0.0),
        "path": cand.get("path"),
        "utility": cand.get("utility"),
    }


def _numeric_state(prompt: str) -> dict[str, float]:
    out: dict[str, float] = {}
    in_state = False
    for raw in str(prompt).splitlines():
        line = raw.strip()
        if line == "state_context:":
            in_state = True
            continue
        if not in_state or not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        try:
            out[key.strip()] = float(value.strip())
        except Exception:
            out[key.strip()] = 0.0
    return out


def _sign_for_side(side: str) -> int:
    side = str(side).upper()
    if side == "LONG":
        return 1
    if side == "SHORT":
        return -1
    return 0


def _safe(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return 0.0
    return x if math.isfinite(x) else 0.0


def _candidate_premises(state: dict[str, float], cand: dict[str, Any]) -> tuple[list[str], float]:
    pred = cand["prediction"]
    side = str(pred.get("side", "NONE")).upper()
    s = _sign_for_side(side)
    score = 0.0
    premises: list[str] = []
    if s == 0:
        return ["reject: candidate has no valid side"], -999.0

    trend96 = _safe(state.get("trend_96"))
    htf4 = _safe(state.get("htf_4h_return_4"))
    htf1d = _safe(state.get("htf_1d_return_4"))
    htf1w = _safe(state.get("htf_1w_return_4"))
    range_pos = _safe(state.get("range_pos"))
    rex2016 = _safe(state.get("rex_2016_range_pos"))
    rex8640 = _safe(state.get("rex_8640_range_pos"))
    range_vol = _safe(state.get("range_vol"))
    drawdown = _safe(state.get("window_drawdown"))
    dxy = _safe(state.get("dxy_zscore"))
    kimchi = _safe(state.get("kimchi_premium_zscore"))
    kimchi_chg = _safe(state.get("kimchi_premium_change"))
    usdkrw = _safe(state.get("usdkrw_zscore"))

    # Rule family 1: trend alignment.  Trade with multi-timeframe direction; fade only at extremes.
    trend_vote = s * (1.2 * trend96 + 0.9 * htf4 + 0.8 * htf1d + 0.5 * htf1w)
    if trend_vote > 0.035:
        score += 1.4
        premises.append("support: candidate aligns with multi-timeframe trend")
    elif trend_vote < -0.035:
        score -= 1.4
        premises.append("reject: candidate fights multi-timeframe trend")
    else:
        premises.append("neutral: trend evidence is mixed")

    # Rule family 2: location.  Long is better near lower range; short is better near upper range.
    location = -s * (0.55 * range_pos + 0.30 * rex2016 + 0.15 * rex8640)
    if location > 0.35:
        score += 1.1
        premises.append("support: candidate has favorable range location")
    elif location < -0.35:
        score -= 1.1
        premises.append("reject: candidate enters into unfavorable range extreme")
    else:
        premises.append("neutral: range location is not decisive")

    # Rule family 3: volatility/drawdown.  Avoid very wide/noisy states unless location supports mean reversion.
    if range_vol > 0.10 and location < 0.2:
        score -= 0.9
        premises.append("reject: high range volatility without enough location edge")
    elif 0.025 <= range_vol <= 0.08:
        score += 0.35
        premises.append("support: volatility is tradable rather than chaotic")
    if drawdown > 0.08 and side == "SHORT":
        score -= 0.7
        premises.append("reject: short after large drawdown has squeeze risk")
    elif drawdown > 0.05 and side == "LONG":
        score += 0.45
        premises.append("support: long has drawdown-recovery premise")

    # Rule family 4: macro pressure.  Strong DXY/USDKRW is BTC headwind; kimchi premium expansion is local risk-on.
    macro = 0.0
    macro += -0.45 * dxy
    macro += -0.25 * usdkrw
    macro += 0.30 * kimchi
    macro += 0.20 * kimchi_chg
    macro_vote = s * macro
    if macro_vote > 0.45:
        score += 0.9
        premises.append("support: macro/kimchi pressure agrees with side")
    elif macro_vote < -0.45:
        score -= 0.9
        premises.append("reject: macro/kimchi pressure conflicts with side")
    else:
        premises.append("neutral: macro pressure is weak or mixed")

    # Rule family 5: alpha-source prior, deliberately small.  Avoid making source identity dominate.
    cid = str(cand.get("id", ""))
    if "market_derivatives" in cid:
        score += 0.20
        premises.append("support: market-derivatives source has diversified context")
    if "external|h288" in cid and side == "LONG" and dxy > 1.0:
        score -= 0.25
        premises.append("reject: external long conflicts with high DXY")

    return premises, float(score)


def _deduction_text(cand: dict[str, Any], premises: list[str], score: float) -> str:
    pred = cand["prediction"]
    conclusion = "ACCEPT" if score >= 1.0 else "REJECT"
    return json.dumps(
        {
            "candidate": cand.get("id"),
            "side": pred.get("side"),
            "score": round(float(score), 4),
            "premises": premises[:8],
            "conclusion": conclusion,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def run(cfg: DeductiveSelectorConfig) -> dict[str, Any]:
    pair_rows = _load(cfg.pairwise_inputs)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[int(row.get("signal_pos", -1) or -1)].append(row)

    out_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    score_values: list[float] = []
    for pos, rows in sorted(grouped.items()):
        state = _numeric_state(str(rows[0].get("prompt", "")))
        candidates: dict[str, dict[str, Any]] = {}
        for row in rows:
            for label in ("A", "B"):
                cand = _candidate_from_pair(row, label)
                if cand is not None:
                    candidates[str(cand["id"])] = cand
        if not candidates:
            skipped["no_candidates"] += 1
            continue
        scored = []
        for cand in candidates.values():
            premises, score = _candidate_premises(state, cand)
            scored.append((score, cand, premises))
        scored.sort(key=lambda x: x[0], reverse=True)
        emitted = 0
        for score, cand, premises in scored:
            if score < float(cfg.min_score):
                skipped["below_min_score"] += 1
                continue
            pred = dict(cand["prediction"])
            base = rows[0]
            out_rows.append(
                {
                    "date": base.get("date"),
                    "signal_pos": int(pos),
                    "prediction": {**pred, "family": "deductive_symbolic_candidate_selector", "confidence": "HIGH" if score >= 2.0 else "MEDIUM"},
                    "position_scale": min(1.0, max(0.25, float(score) / 3.0)),
                    "score": float(score),
                    "selected_candidate_id": cand.get("id"),
                    "deduction": _deduction_text(cand, premises, score),
                    "period": _period_key(_date(base), cfg.period),
                }
            )
            pred_counts[str(pred.get("side", "NONE"))] += 1
            score_values.append(float(score))
            emitted += 1
            if emitted >= int(cfg.max_candidates_per_timestamp):
                break
        if emitted == 0:
            skipped["no_candidate_emitted"] += 1
    Path(cfg.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_jsonl).write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in out_rows) + ("\n" if out_rows else ""))
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "rows_out": len(out_rows),
        "pred_counts": dict(sorted(pred_counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "score_summary": {
            "min": min(score_values) if score_values else 0.0,
            "max": max(score_values) if score_values else 0.0,
            "mean": float(np.mean(score_values)) if score_values else 0.0,
        },
        "leakage_guard": {
            "uses_signal_time_state_and_candidate_descriptors_only": True,
            "does_not_use_pairwise_future_target_for_selection": True,
        },
    }
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deductive symbolic candidate selector")
    p.add_argument("--pairwise-inputs", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--period", choices=["year", "halfyear", "quarter"], default=DeductiveSelectorConfig.period)
    p.add_argument("--min-score", type=float, default=DeductiveSelectorConfig.min_score)
    p.add_argument("--min-vote", type=int, default=DeductiveSelectorConfig.min_vote)
    p.add_argument("--max-candidates-per-timestamp", type=int, default=DeductiveSelectorConfig.max_candidates_per_timestamp)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(DeductiveSelectorConfig(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
