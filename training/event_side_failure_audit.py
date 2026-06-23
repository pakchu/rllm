"""Audit event-context prediction side failures by period and event tokens."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EventSideFailureAuditCfg:
    predictions_jsonl: str
    source_context_jsonl: str
    output: str
    periods: str = "selection:2024-01:2025-12,eval:2026-01:2026-05"
    top_k: int = 40


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _periods(raw: str) -> list[tuple[str, str, str]]:
    out = []
    for part in str(raw).split(","):
        if not part.strip():
            continue
        name, start, end = part.split(":", 2)
        out.append((name, start, end))
    return out


def _month(row: dict[str, Any]) -> str:
    return str(row.get("date", ""))[:7]


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("date")), int(row.get("signal_pos", -1) or -1))


def _actual_for(row: dict[str, Any], side: str) -> float:
    side = str(side).upper()
    if side == "LONG":
        return float(row.get("actual_long_pct", 0.0) or 0.0)
    if side == "SHORT":
        return float(row.get("actual_short_pct", 0.0) or 0.0)
    return 0.0


def _source_actual(src: dict[str, Any], side: str) -> float:
    audit = src.get("reward_audit") if isinstance(src.get("reward_audit"), dict) else {}
    val = audit.get(str(side).upper()) if isinstance(audit.get(str(side).upper()), dict) else {}
    try:
        return float(val.get("net_return_pct", 0.0))
    except Exception:
        return 0.0


def _summary(vals: list[float]) -> dict[str, Any]:
    arr = np.asarray(vals, dtype=float)
    if len(arr) == 0:
        return {"n": 0, "mean_pct": 0.0, "win_rate": 0.0, "sum_pct": 0.0}
    return {"n": int(len(arr)), "mean_pct": float(np.mean(arr)), "median_pct": float(np.median(arr)), "win_rate": float(np.mean(arr > 0.0)), "sum_pct": float(np.sum(arr))}


def _trade_action(row: dict[str, Any]) -> str:
    pred = row.get("prediction") if isinstance(row.get("prediction"), dict) else {}
    if pred.get("gate") != "TRADE":
        return "WAIT"
    side = str(pred.get("side", "NONE")).upper()
    return side if side in {"LONG", "SHORT"} else "WAIT"


def _event_tokens(src: dict[str, Any]) -> dict[str, str]:
    state = src.get("state_tokens") if isinstance(src.get("state_tokens"), dict) else {}
    return {k: str(state.get(k, "missing")) for k in ("pa_event_pressure", "pa_downside_reclaim", "pa_upside_rejection", "pa_long_window_event", "trend_alignment", "risk_state", "htf_trend_stack", "htf_risk_state")}


def run(cfg: EventSideFailureAuditCfg) -> dict[str, Any]:
    preds = _read_jsonl(cfg.predictions_jsonl)
    src_rows = _read_jsonl(cfg.source_context_jsonl)
    src_by_key = {_key(r): r for r in src_rows}
    periods = _periods(cfg.periods)
    report: dict[str, Any] = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "periods": {}, "leakage_guard": {"audit_only_uses_realized_labels": True, "not_a_selector": True}}
    for name, start, end in periods:
        rows = [r for r in preds if start <= _month(r) <= end]
        trade_rows = [r for r in rows if _trade_action(r) in {"LONG", "SHORT"}]
        chosen = []
        inverted = []
        oracle_best = []
        by_side: dict[str, list[float]] = defaultdict(list)
        by_event: dict[str, list[float]] = defaultdict(list)
        by_event_side: dict[str, list[float]] = defaultdict(list)
        pred_vs_best = {"matched_best": 0, "opposite_best": 0, "wait_best": 0, "trades": len(trade_rows)}
        for r in trade_rows:
            src = src_by_key.get(_key(r), {})
            action = _trade_action(r)
            other = "SHORT" if action == "LONG" else "LONG"
            actual = _actual_for(r, action)
            inv = _actual_for(r, other)
            long_v = _source_actual(src, "LONG") if src else float(r.get("actual_long_pct", 0.0) or 0.0)
            short_v = _source_actual(src, "SHORT") if src else float(r.get("actual_short_pct", 0.0) or 0.0)
            best_side = "LONG" if long_v >= short_v and long_v > 0 else "SHORT" if short_v > long_v and short_v > 0 else "WAIT"
            if best_side == action:
                pred_vs_best["matched_best"] += 1
            elif best_side == other:
                pred_vs_best["opposite_best"] += 1
            else:
                pred_vs_best["wait_best"] += 1
            chosen.append(actual); inverted.append(inv); oracle_best.append(max(0.0, long_v, short_v)); by_side[action].append(actual)
            toks = _event_tokens(src)
            for k, v in toks.items():
                key = f"{k}={v}"
                by_event[key].append(actual)
                by_event_side[f"{key}|side={action}"].append(actual)
        event_rows = sorted(({"token": k, **_summary(v)} for k, v in by_event.items()), key=lambda x: (x["mean_pct"], x["n"]), reverse=True)
        event_side_rows = sorted(({"token_side": k, **_summary(v)} for k, v in by_event_side.items()), key=lambda x: (x["mean_pct"], x["n"]), reverse=True)
        report["periods"][name] = {
            "rows": len(rows),
            "trade_rows": len(trade_rows),
            "chosen": _summary(chosen),
            "inverted_same_entries": _summary(inverted),
            "oracle_best_same_entries": _summary(oracle_best),
            "by_side": {k: _summary(v) for k, v in sorted(by_side.items())},
            "pred_vs_best": pred_vs_best,
            "top_event_tokens": event_rows[: int(cfg.top_k)],
            "bottom_event_tokens": list(reversed(event_rows[-int(cfg.top_k):])),
            "top_event_side_tokens": event_side_rows[: int(cfg.top_k)],
            "bottom_event_side_tokens": list(reversed(event_side_rows[-int(cfg.top_k):])),
        }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit event/side failures for prediction rows")
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--source-context-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--periods", default=EventSideFailureAuditCfg.periods)
    p.add_argument("--top-k", type=int, default=EventSideFailureAuditCfg.top_k)
    return p.parse_args()


def main() -> None:
    report = run(EventSideFailureAuditCfg(**vars(parse_args())))
    compact = {}
    for k, v in report["periods"].items():
        compact[k] = {kk: v[kk] for kk in ("trade_rows", "chosen", "inverted_same_entries", "oracle_best_same_entries", "by_side", "pred_vs_best")}
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
