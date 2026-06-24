"""Audit yearly reward drift for event candidate/ranker rows."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EventCandidateDriftAuditCfg:
    input_jsonl: str
    output: str
    min_count: int = 50
    top_n: int = 30


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _year(row: dict[str, Any]) -> str:
    return str(row.get("date", ""))[:4]


def _utility(row: dict[str, Any]) -> float:
    reward = row.get("reward", {}) if isinstance(row.get("reward"), dict) else {}
    return float(reward.get("rank_utility", reward.get("net_return_pct", 0.0)) or 0.0)


def _candidate_key(row: dict[str, Any], field: str) -> str:
    cand = row.get("candidate", {}) if isinstance(row.get("candidate"), dict) else {}
    return str(cand.get(field, "UNKNOWN"))


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 30 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _stats(xs: list[float]) -> dict[str, Any]:
    arr = np.asarray(xs, dtype=float)
    if len(arr) == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "positive_frac": 0.0, "p90": 0.0, "p10": 0.0}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "positive_frac": float(np.mean(arr > 0.0)),
        "p90": float(np.quantile(arr, 0.9)),
        "p10": float(np.quantile(arr, 0.1)),
    }


def _yearly_group_stats(rows: list[dict[str, Any]], field: str, min_count: int) -> list[dict[str, Any]]:
    vals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        vals[(_year(row), _candidate_key(row, field))].append(_utility(row))
    out = []
    for (year, key), xs in vals.items():
        if len(xs) >= min_count:
            out.append({"year": year, "field": field, "value": key, **_stats(xs)})
    return sorted(out, key=lambda r: (r["field"], r["value"], r["year"]))


def _token_yearly_stats(rows: list[dict[str, Any]], min_count: int) -> list[dict[str, Any]]:
    vals: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        toks = row.get("state_tokens", {}) if isinstance(row.get("state_tokens"), dict) else {}
        for k, v in toks.items():
            vals[(_year(row), str(k), str(v))].append(_utility(row))
    out = []
    for (year, key, val), xs in vals.items():
        if len(xs) >= min_count:
            out.append({"year": year, "token": key, "value": val, **_stats(xs)})
    return sorted(out, key=lambda r: (r["token"], r["value"], r["year"]))


def _feature_ic(rows: list[dict[str, Any]], min_count: int) -> list[dict[str, Any]]:
    by_year: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_year[_year(row)].append(row)
    names = sorted({str(k) for row in rows for k in (row.get("feature_snapshot", {}) if isinstance(row.get("feature_snapshot"), dict) else {}).keys()})
    out = []
    for year, yr_rows in sorted(by_year.items()):
        if len(yr_rows) < min_count:
            continue
        y = np.asarray([_utility(r) for r in yr_rows], dtype=float)
        for name in names:
            x = np.asarray([float((r.get("feature_snapshot", {}) if isinstance(r.get("feature_snapshot"), dict) else {}).get(name, 0.0) or 0.0) for r in yr_rows], dtype=float)
            ic = _spearman(x, y)
            if ic is not None:
                out.append({"year": year, "feature": name, "ic": ic, "n": len(yr_rows)})
    return sorted(out, key=lambda r: (r["feature"], r["year"]))


def _drift_summary(yearly: list[dict[str, Any]], *, key_fields: tuple[str, ...], top_n: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in yearly:
        grouped[tuple(str(row[k]) for k in key_fields)].append(row)
    out = []
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda r: r["year"])
        means = {r["year"]: float(r["mean"]) for r in rows}
        if "2026" not in means or len(means) < 2:
            continue
        prior = [v for y, v in means.items() if y < "2026"]
        if not prior:
            continue
        prior_mean = float(np.mean(prior))
        delta = means["2026"] - prior_mean
        out.append({"key": dict(zip(key_fields, key)), "years": means, "prior_mean": prior_mean, "delta_2026_vs_prior": delta, "abs_delta": abs(delta)})
    return sorted(out, key=lambda r: r["abs_delta"], reverse=True)[:top_n]


def _ic_drift(feature_ic: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_ic:
        grouped[str(row["feature"])].append(row)
    out = []
    for feature, rows in grouped.items():
        vals = {r["year"]: float(r["ic"]) for r in rows}
        if "2026" not in vals:
            continue
        prior = [v for y, v in vals.items() if y < "2026"]
        if not prior:
            continue
        prior_mean = float(np.mean(prior))
        delta = vals["2026"] - prior_mean
        flip = (prior_mean * vals["2026"]) < 0.0
        out.append({"feature": feature, "ics": vals, "prior_mean": prior_mean, "delta_2026_vs_prior": delta, "sign_flip": bool(flip), "abs_delta": abs(delta)})
    return sorted(out, key=lambda r: (not r["sign_flip"], -r["abs_delta"]))[:top_n]


def run(cfg: EventCandidateDriftAuditCfg) -> dict[str, Any]:
    rows = _load(cfg.input_jsonl)
    by_year: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_year[_year(row)].append(_utility(row))
    group_yearly = []
    for field in ["family", "side", "hold_bars"]:
        group_yearly.extend(_yearly_group_stats(rows, field, cfg.min_count))
    token_yearly = _token_yearly_stats(rows, cfg.min_count)
    fic = _feature_ic(rows, cfg.min_count)
    report = {
        "config": asdict(cfg),
        "overall_by_year": {year: _stats(xs) for year, xs in sorted(by_year.items())},
        "candidate_group_yearly": group_yearly,
        "token_yearly": token_yearly,
        "feature_ic_yearly": fic,
        "largest_candidate_group_drifts": _drift_summary(group_yearly, key_fields=("field", "value"), top_n=cfg.top_n),
        "largest_token_drifts": _drift_summary(token_yearly, key_fields=("token", "value"), top_n=cfg.top_n),
        "largest_feature_ic_drifts": _ic_drift(fic, cfg.top_n),
        "leakage_guard": {"uses_reward_for_audit_only": True, "does_not_train_or_select_policy": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit yearly candidate reward and feature drift")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-count", type=int, default=EventCandidateDriftAuditCfg.min_count)
    p.add_argument("--top-n", type=int, default=EventCandidateDriftAuditCfg.top_n)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(EventCandidateDriftAuditCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
