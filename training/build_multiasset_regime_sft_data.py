"""Build LLM SFT rows for monthly multi-asset regime/policy selection.

This keeps the quant harness inside the reset RLLM framing: the LLM sees a
past-only textual regime brief plus trailing candidate-policy evidence, then emits
one compact policy JSON selecting the next month's policy or CASH.

Labels use the online bandit decision made from prior months only.  Future month
returns are included only in metadata for audit, not in the prompt or target.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def load_report(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _fmt_pct(v: float) -> str:
    return f"{float(v):+.2f}%"


def _bucket(v: float, lo: float, hi: float) -> str:
    if v <= lo:
        return "low"
    if v >= hi:
        return "high"
    return "neutral"


def _month_market_summary(month: str, price_dir: str, aux_dir: str, symbols: list[str]) -> dict[str, Any]:
    start = pd.Timestamp(month)
    end = start + pd.offsets.MonthBegin(1)
    rows = []
    for sym in symbols:
        p = sorted(Path(price_dir).glob(f"{sym}_5m_*.csv.gz"))[-1]
        df = pd.read_csv(p, usecols=["date", "close", "volume"], parse_dates=["date"])
        sub = df[(df["date"] >= start) & (df["date"] < end)]
        if len(sub) < 2:
            continue
        ret = float(sub["close"].iloc[-1] / sub["close"].iloc[0] - 1.0)
        vol = float(sub["close"].pct_change().std() * (12 * 24 * 30) ** 0.5)
        rows.append({"symbol": sym, "ret": ret, "vol": vol})
    funding_vals = []
    premium_vals = []
    for sym in symbols:
        fps = sorted(Path(aux_dir).glob(f"{sym}_funding_*.csv.gz"))
        pps = sorted(Path(aux_dir).glob(f"{sym}_premium_*.csv.gz"))
        if fps:
            f = pd.read_csv(fps[-1], parse_dates=["date"])
            fs = f[(f["date"] >= start) & (f["date"] < end)]
            if len(fs): funding_vals.append(float(pd.to_numeric(fs["funding_rate"], errors="coerce").mean()))
        if pps:
            pr = pd.read_csv(pps[-1])
            pr["dt"] = pd.to_datetime(pr["close_time"].astype("int64"), unit="ms")
            ps = pr[(pr["dt"] >= start) & (pr["dt"] < end)]
            if len(ps): premium_vals.append(float(pd.to_numeric(ps["close"], errors="coerce").mean()))
    if not rows:
        return {"market_return": "unknown", "dispersion": "unknown", "volatility": "unknown", "funding_pressure": "unknown", "premium_basis": "unknown"}
    rets = pd.Series([r["ret"] for r in rows])
    vols = pd.Series([r["vol"] for r in rows])
    return {
        "market_return": _bucket(float(rets.mean()), -0.03, 0.03),
        "market_return_pct": round(float(rets.mean() * 100), 2),
        "cross_asset_dispersion": _bucket(float(rets.std()), 0.05, 0.15),
        "dispersion_pct": round(float(rets.std() * 100), 2),
        "volatility": _bucket(float(vols.mean()), 0.45, 0.95),
        "ann_vol_pct": round(float(vols.mean() * 100), 2),
        "funding_pressure": _bucket(float(pd.Series(funding_vals).mean()) if funding_vals else 0.0, -0.00005, 0.00008),
        "avg_funding_bps": round(float(pd.Series(funding_vals).mean() * 10000), 3) if funding_vals else 0.0,
        "premium_basis": _bucket(float(pd.Series(premium_vals).mean()) if premium_vals else 0.0, -0.0003, 0.0003),
        "avg_premium_bps": round(float(pd.Series(premium_vals).mean() * 10000), 3) if premium_vals else 0.0,
    }


def _prompt(decision: dict[str, Any], prior: list[dict[str, Any]], market: dict[str, Any]) -> str:
    lines = [
        "You are a single compact RLLM monthly policy for Binance USD-M alt futures.",
        "Use only past evidence shown here. Choose the next-month policy or CASH.",
        "Candidate policies: excess_spread, utility1_pos, utility1_inv, utility3_pos, utility3_inv, cash.",
        "Return compact JSON with keys: policy, allow_trade, evidence_strength, score_margin, risk_note, reason_code.",
        f"decision_month: {decision['month']}",
        "current_past_month_market_state:",
    ]
    for k in sorted(market):
        lines.append(f"- {k}: {market[k]}")
    lines.append("trailing_policy_evidence:")
    for item in prior:
        scores = item.get("scores", {})
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top = ", ".join(f"{k}={v:+.2f}" for k, v in ordered[:4])
        lines.append(f"- month {item['month']}: selected={item['selected']} realized_ret={_fmt_pct(item['sim']['ret_pct'])} strict_mdd={_fmt_pct(item['sim']['strict_mdd_pct'])} trailing_scores[{top}]")
    lines.append("Do not mention future realized return. Choose exactly one policy.")
    return "\n".join(lines)


def policy_response(policy: str, scores: dict[str, Any], *, reason_code: str) -> str:
    """Return the reset single-policy JSON response for a monthly selector row."""

    score_map = {str(k): float(v) for k, v in dict(scores or {}).items()}
    selected = str(policy)
    sorted_scores = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    best_policy = sorted_scores[0][0] if sorted_scores else selected
    best_score = float(score_map.get(selected, 0.0))
    if len(sorted_scores) > 1:
        raw_margin = float(sorted_scores[0][1] - sorted_scores[1][1])
    else:
        raw_margin = 0.0
    if selected != best_policy:
        raw_margin = -abs(float(score_map.get(best_policy, 0.0)) - best_score)
    obj = {
        "policy": selected,
        "allow_trade": selected != "cash",
        "evidence_strength": "high" if best_score > 3 and raw_margin > 1 else "medium" if best_score > 0 else "low",
        "score_margin": round(raw_margin, 4),
        "risk_note": "use_cash_when_all_recent_scores_negative" if selected == "cash" else "candidate_selected_from_trailing_only",
        "reason_code": reason_code,
    }
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _target(decision: dict[str, Any]) -> str:
    return policy_response(decision["selected"], decision.get("scores", {}), reason_code="trailing_regime_bandit_label")


def run(args: argparse.Namespace) -> dict[str, Any]:
    report = load_report(args.bandit_report)
    decisions = report["decisions"]
    symbols = ["ADAUSDT", "BNBUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    rows = []
    by_month = {d["month"]: d for d in decisions}
    for i, d in enumerate(decisions):
        prior = decisions[max(0, i - args.lookback_months):i]
        if len(prior) < args.min_prior_months:
            continue
        market = _month_market_summary(prior[-1]["month"], args.price_dir, args.aux_dir, symbols)
        prompt = _prompt(d, prior, market)
        target = _target(d)
        rows.append({
            "task": "multiasset_monthly_regime_policy_sft",
            "messages": [
                {"role": "system", "content": "You emit one no-leak compact policy JSON for monthly futures policy selection."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ],
            "prompt": prompt,
            "target": target,
            "metadata": {
                "month": d["month"],
                "selected_policy": d["selected"],
                "future_month_ret_pct_audit_only": d["sim"]["ret_pct"],
                "future_month_mdd_pct_audit_only": d["sim"]["strict_mdd_pct"],
                "leakage_guard": "prompt and target use only prior-month bandit evidence; future month outcome is metadata audit only",
            },
        })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))
    counts = Counter(r["metadata"]["selected_policy"] for r in rows)
    chars = [len(r["prompt"]) + len(r["target"]) for r in rows]
    summary = {"rows": len(rows), "policy_counts": dict(counts), "chars": {"min": min(chars) if chars else 0, "max": max(chars) if chars else 0, "mean": sum(chars)/max(1,len(chars))}, "source_bandit_sim": report.get("sim"), "schema": "single_policy_no_analyzer_trader_cascade", "leakage_guard": "SFT labels are bandit decisions from trailing evidence only; not future oracle labels"}
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bandit-report", required=True)
    p.add_argument("--price-dir", required=True)
    p.add_argument("--aux-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--lookback-months", type=int, default=3)
    p.add_argument("--min-prior-months", type=int, default=3)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
