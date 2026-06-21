"""Join executed trades to symbolic prompt tokens and summarize token-conditioned returns."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _action_key(date: Any, signal_pos: Any, action: dict[str, Any]) -> tuple[str, int, str, str, int]:
    return (str(date), int(signal_pos or -1), str(action.get("family", "")), str(action.get("side", "")).upper(), int(action.get("hold_bars", 0) or 0))


def _pred_action(row: dict[str, Any]) -> dict[str, Any]:
    pred = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
    if pred.get("gate") == "TRADE":
        return {"family": pred.get("family"), "side": str(pred.get("side", "")).upper(), "hold_bars": int(pred.get("hold_bars", 0) or 0)}
    act = row.get("selected_action", {}) if isinstance(row.get("selected_action"), dict) else {}
    return {"family": act.get("family"), "side": str(act.get("side", "")).upper(), "hold_bars": int(act.get("hold_bars", 0) or 0)}


def _tokens(prompt: str) -> list[str]:
    out: list[str] = []
    for line in str(prompt).splitlines():
        if line.startswith("Regime tokens:"):
            for part in line.split(":", 1)[1].split(";"):
                t = part.strip()
                if t:
                    out.append(t)
        elif line.startswith("Candidate book tokens:"):
            # Keep only family presence, not full strength, to avoid tiny buckets.
            for part in line.split(":", 1)[1].split(";"):
                t = part.strip()
                if t:
                    out.append("book_family=" + t.split(":", 1)[0])
        elif line.startswith("Selected action tokens:"):
            for part in line.split(":", 1)[1].split(";"):
                t = part.strip()
                if t:
                    out.append("action_" + t)
    return out


def _summ(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0, "sum_pct": 0.0, "mean_pct": 0.0, "win_rate": 0.0}
    return {"n": len(xs), "sum_pct": sum(xs), "mean_pct": sum(xs) / len(xs), "win_rate": sum(1 for x in xs if x > 0) / len(xs)}


def run(*, candidate_jsonl: str, predictions_jsonl: str, backtest_json: str, output: str, family: str = "") -> dict[str, Any]:
    candidates = _read_jsonl(candidate_jsonl)
    cand_by_key = {_action_key(r.get("date"), r.get("signal_pos"), r.get("action", {}) if isinstance(r.get("action"), dict) else {}): r for r in candidates}
    preds = _read_jsonl(predictions_jsonl)
    pred_by_key = {(str(r.get("date")), int(r.get("signal_pos", -1) or -1)): r for r in preds}
    bt = json.loads(Path(backtest_json).read_text())
    by_token: dict[str, list[float]] = defaultdict(list)
    enriched = []
    misses = 0
    for tr in bt.get("executed", []):
        pr = pred_by_key.get((str(tr.get("date")), int(tr.get("signal_pos", -1) or -1)), {})
        action = _pred_action(pr)
        fam = str(action.get("family"))
        if family and fam != family:
            continue
        cand = cand_by_key.get(_action_key(tr.get("date"), tr.get("signal_pos"), action))
        if cand is None:
            misses += 1
            continue
        ret = float(tr.get("trade_ret_pct", 0.0) or 0.0)
        toks = _tokens(str(cand.get("prompt", "")))
        for tok in toks:
            by_token[tok].append(ret)
        enriched.append({"date": tr.get("date"), "signal_pos": tr.get("signal_pos"), "family": fam, "side": action.get("side"), "hold_bars": action.get("hold_bars"), "trade_ret_pct": ret, "tokens": toks})
    rows = [{"token": k, **_summ(v)} for k, v in by_token.items()]
    report = {
        "candidate_jsonl": candidate_jsonl,
        "predictions_jsonl": predictions_jsonl,
        "backtest_json": backtest_json,
        "family_filter": family,
        "matched_trades": len(enriched),
        "misses": misses,
        "worst_tokens": sorted([r for r in rows if r["n"] >= 3], key=lambda r: r["sum_pct"])[:40],
        "best_tokens": sorted([r for r in rows if r["n"] >= 3], key=lambda r: r["sum_pct"], reverse=True)[:40],
        "by_token": {r["token"]: {k: r[k] for k in ("n", "sum_pct", "mean_pct", "win_rate")} for r in rows},
        "enriched": enriched,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regime token trade diagnostics")
    p.add_argument("--candidate-jsonl", required=True)
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--backtest-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--family", default="")
    return p.parse_args()


def main() -> None:
    r = run(**vars(parse_args()))
    print(json.dumps({k: r[k] for k in ("family_filter", "matched_trades", "misses", "worst_tokens", "best_tokens")}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
