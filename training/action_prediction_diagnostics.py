"""Diagnostics and family filters for generated action predictions.

The script uses only already-generated predictions and strict realized executed
trade records from the backtester.  It is for post-hoc diagnosis and for testing
fixed family allow/block sets chosen on a prior period.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _month(date: Any) -> str:
    return str(date)[:7]


def _family(row: dict[str, Any]) -> str:
    pred = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
    if pred.get("gate") == "TRADE":
        return str(pred.get("family") or row.get("selected_action", {}).get("family") or "UNKNOWN")
    act = row.get("selected_action", {}) if isinstance(row.get("selected_action"), dict) else {}
    return str(act.get("family") or "NO_TRADE")


def _side(row: dict[str, Any]) -> str:
    pred = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
    if pred.get("gate") == "TRADE":
        return str(pred.get("side", "NONE")).upper()
    act = row.get("selected_action", {}) if isinstance(row.get("selected_action"), dict) else {}
    return str(act.get("side", "NONE")).upper()


def _hold(row: dict[str, Any]) -> int:
    pred = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
    if pred.get("gate") == "TRADE":
        return int(pred.get("hold_bars", 0) or 0)
    act = row.get("selected_action", {}) if isinstance(row.get("selected_action"), dict) else {}
    return int(act.get("hold_bars", 0) or 0)


def _summ(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0, "sum_pct": 0.0, "mean_pct": 0.0, "win_rate": 0.0}
    return {"n": len(xs), "sum_pct": sum(xs), "mean_pct": sum(xs) / len(xs), "win_rate": sum(1 for x in xs if x > 0) / len(xs)}


def summarize_predictions(*, predictions_jsonl: str, backtest_json: str, output: str) -> dict[str, Any]:
    preds = _read_jsonl(predictions_jsonl)
    pred_by_key = {(str(r.get("date")), int(r.get("signal_pos", -1) or -1)): r for r in preds}
    bt = json.loads(Path(backtest_json).read_text())
    executed = bt.get("executed", [])
    buckets: dict[str, list[float]] = defaultdict(list)
    enriched = []
    for tr in executed:
        key = (str(tr.get("date")), int(tr.get("signal_pos", -1) or -1))
        pr = pred_by_key.get(key, {})
        ret = float(tr.get("trade_ret_pct", 0.0) or 0.0)
        fam = _family(pr)
        side = _side(pr)
        hold = _hold(pr)
        for b in [f"month={_month(tr.get('date'))}", f"family={fam}", f"side={side}", f"hold={hold}", f"family={fam}|side={side}", f"family={fam}|hold={hold}", f"month={_month(tr.get('date'))}|family={fam}"]:
            buckets[b].append(ret)
        enriched.append({"date": tr.get("date"), "signal_pos": tr.get("signal_pos"), "family": fam, "side": side, "hold_bars": hold, "trade_ret_pct": ret, "equity": tr.get("equity")})
    report = {
        "predictions_jsonl": predictions_jsonl,
        "backtest_json": backtest_json,
        "sim": bt.get("sim", {}),
        "trade_stats": bt.get("trade_stats", {}),
        "by_bucket": {k: _summ(v) for k, v in sorted(buckets.items())},
        "worst_buckets": sorted(({"bucket": k, **_summ(v)} for k, v in buckets.items() if len(v) >= 5), key=lambda r: r["sum_pct"])[:30],
        "best_buckets": sorted(({"bucket": k, **_summ(v)} for k, v in buckets.items() if len(v) >= 5), key=lambda r: r["sum_pct"], reverse=True)[:30],
        "executed_enriched": enriched,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def filter_predictions(*, predictions_jsonl: str, output: str, allowed_families: str = "", blocked_families: str = "") -> dict[str, Any]:
    rows = _read_jsonl(predictions_jsonl)
    allow = {x.strip() for x in allowed_families.split(",") if x.strip()}
    block = {x.strip() for x in blocked_families.split(",") if x.strip()}
    out = []
    blocked = allowed = no_trade = 0
    for row in rows:
        pred = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
        if pred.get("gate") != "TRADE":
            no_trade += 1
            out.append(row)
            continue
        fam = str(pred.get("family") or row.get("selected_action", {}).get("family") or "UNKNOWN")
        keep = True
        if allow and fam not in allow:
            keep = False
        if fam in block:
            keep = False
        if keep:
            allowed += 1
            out.append(row)
        else:
            blocked += 1
            out.append({**row, "prediction": {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "FAMILY_FILTER", "confidence": "HIGH"}, "blocked_family": fam})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n")
    return {"input": predictions_jsonl, "output": output, "allowed_families": sorted(allow), "blocked_families": sorted(block), "rows": len(rows), "allowed_trade_rows": allowed, "blocked_trade_rows": blocked, "input_no_trade_rows": no_trade}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Action prediction diagnostics")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--predictions-jsonl", required=True)
    s.add_argument("--backtest-json", required=True)
    s.add_argument("--output", required=True)
    f = sub.add_parser("filter")
    f.add_argument("--predictions-jsonl", required=True)
    f.add_argument("--output", required=True)
    f.add_argument("--allowed-families", default="")
    f.add_argument("--blocked-families", default="")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.cmd == "summarize":
        print(json.dumps({k: v for k, v in summarize_predictions(predictions_jsonl=a.predictions_jsonl, backtest_json=a.backtest_json, output=a.output).items() if k not in {"executed_enriched", "by_bucket"}}, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(filter_predictions(predictions_jsonl=a.predictions_jsonl, output=a.output, allowed_families=a.allowed_families, blocked_families=a.blocked_families), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
