"""Mine stable analyzer-conditioned action templates.

This is a diagnostic/template search, not a live trader.  Rules are conjunctions
of past-only analyzer facts paired with a fixed action.  Candidate ranking is
computed on train; val/OOS are reported separately so unstable hindsight labels
are visible instead of hidden behind an oracle best-action target.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

from training.economic_drift_diagnostic import utility_summary
from training.economic_event_memory_policy import _action_obj, _action_text
from training.economic_value_baseline import _summary_obj, load_jsonl

DEFAULT_FIELDS = (
    "regime",
    "trend_alignment",
    "trend_strength",
    "location",
    "momentum",
    "oscillator",
    "risk_state",
    "volatility_level",
    "volume_state",
    "candle_pattern",
)
DEFAULT_SYMBOLIC_FIELDS = (
    "Macro Dollar State",
    "Korea Premium State",
    "Order Flow",
)


def row_facts(row: dict[str, Any], *, fields: tuple[str, ...] = DEFAULT_FIELDS, symbolic_fields: tuple[str, ...] = DEFAULT_SYMBOLIC_FIELDS, include_context: bool = True) -> tuple[str, ...]:
    s = _summary_obj(str(row.get("prompt", "")))
    sym = s.get("symbolic_features", {}) if isinstance(s.get("symbolic_features"), dict) else {}
    facts: set[str] = set()
    for f in fields:
        v = s.get(f)
        if v is not None:
            facts.add(f"{f}={v}")
    for f in symbolic_fields:
        v = sym.get(f)
        if v is not None:
            facts.add(f"symbolic.{f}={v}")
    if include_context:
        tags = s.get("context_tags", [])
        if isinstance(tags, list):
            for t in tags:
                facts.add(f"tag={t}")
    return tuple(sorted(facts))


def rule_key(action: str, facts: tuple[str, ...]) -> str:
    return action + " | " + " & ".join(facts)


def iter_rule_keys_for_row(row: dict[str, Any], *, max_terms: int, allowed_facts: set[str] | None = None) -> Iterable[str]:
    action = _action_text(_action_obj(str(row.get("action", "{}"))))
    if '"NO_TRADE"' in action:
        return []
    facts = row_facts(row)
    if allowed_facts is not None:
        facts = tuple(f for f in facts if f in allowed_facts)
    keys: list[str] = []
    for k in range(1, max_terms + 1):
        for combo in itertools.combinations(facts, k):
            keys.append(rule_key(action, combo))
    return keys


def collect_rule_values(rows: list[dict[str, Any]], *, max_terms: int, allowed_rules: set[str] | None = None, allowed_facts: set[str] | None = None) -> dict[str, list[float]]:
    vals: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        utility = float(row.get("utility", 0.0))
        for key in iter_rule_keys_for_row(row, max_terms=max_terms, allowed_facts=allowed_facts):
            if allowed_rules is None or key in allowed_rules:
                vals[key].append(utility)
    return vals


def add_rule_quality(summary: dict[str, Any]) -> dict[str, Any]:
    n = int(summary.get("n", 0) or 0)
    if n <= 0:
        return {**summary, "quality": -999.0}
    lo = float(summary.get("ci95_mean_pct", [-999.0, -999.0])[0])
    mu = float(summary.get("mean_pct", 0.0))
    win = float(summary.get("win_rate", 0.0))
    # Conservative quality: prefer lower-bound edge and sample size, not just high mean.
    return {**summary, "quality": lo * (n ** 0.5) + max(0.0, win - 0.5) * 10.0 + mu}


def summarize_rule_values(vals: dict[str, list[float]], *, min_train_n: int, min_train_ci_pct: float) -> dict[str, dict[str, Any]]:
    out = {}
    for key, xs in vals.items():
        s = add_rule_quality(utility_summary(xs))
        if s["n"] >= min_train_n and s["ci95_mean_pct"][0] >= min_train_ci_pct:
            out[key] = s
    return out


def eval_rules(rows: list[dict[str, Any]], rules: set[str], *, max_terms: int, allowed_facts: set[str] | None = None) -> dict[str, dict[str, Any]]:
    vals = collect_rule_values(rows, max_terms=max_terms, allowed_rules=rules, allowed_facts=allowed_facts)
    return {k: add_rule_quality(utility_summary(vals.get(k, []))) for k in rules}


def split_rule(rule: str) -> dict[str, Any]:
    action, _, facts = rule.partition(" | ")
    return {"action": json.loads(action), "facts": facts.split(" & ") if facts else []}



def frequent_facts(rows: list[dict[str, Any]], *, min_fact_count: int) -> set[str]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if '"NO_TRADE"' in _action_text(_action_obj(str(row.get("action", "{}")))):
            continue
        for fact in row_facts(row):
            counts[fact] += 1
    return {fact for fact, n in counts.items() if n >= int(min_fact_count)}

def run_template_miner(
    *,
    train_jsonl: str,
    val_jsonl: str,
    oos_jsonl: str,
    output: str,
    max_terms: int = 2,
    min_train_n: int = 80,
    min_train_ci_pct: float = 0.0,
    min_val_n: int = 20,
    min_val_mean_pct: float = 0.0,
    top_k: int = 100,
    min_fact_count: int = 1,
) -> dict[str, Any]:
    train = load_jsonl(train_jsonl)
    val = load_jsonl(val_jsonl)
    oos = load_jsonl(oos_jsonl)
    allowed_facts = frequent_facts(train, min_fact_count=min_fact_count) if min_fact_count > 1 else None
    train_vals = collect_rule_values(train, max_terms=max_terms, allowed_facts=allowed_facts)
    train_rules = summarize_rule_values(train_vals, min_train_n=min_train_n, min_train_ci_pct=min_train_ci_pct)
    ranked_train = sorted(train_rules.items(), key=lambda kv: kv[1]["quality"], reverse=True)[:top_k]
    rule_set = {k for k, _ in ranked_train}
    val_eval = eval_rules(val, rule_set, max_terms=max_terms, allowed_facts=allowed_facts)
    oos_eval = eval_rules(oos, rule_set, max_terms=max_terms, allowed_facts=allowed_facts)
    rows = []
    for key, train_s in ranked_train:
        val_s = val_eval.get(key, {"n": 0})
        oos_s = oos_eval.get(key, {"n": 0})
        stable_val = int(val_s.get("n", 0) or 0) >= min_val_n and float(val_s.get("mean_pct", -999.0)) >= min_val_mean_pct
        rows.append({"rule": key, **split_rule(key), "train": train_s, "val": val_s, "oos": oos_s, "val_stable": stable_val})
    val_stable = [r for r in rows if r["val_stable"]]
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": {
            "max_terms": max_terms,
            "min_train_n": min_train_n,
            "min_train_ci_pct": min_train_ci_pct,
            "min_val_n": min_val_n,
            "min_val_mean_pct": min_val_mean_pct,
            "top_k": top_k,
            "min_fact_count": min_fact_count,
            "allowed_fact_count": len(allowed_facts) if allowed_facts is not None else None,
        },
        "counts": {"train_candidate_rules": len(train_vals), "train_survivors": len(train_rules), "reported": len(rows), "val_stable": len(val_stable)},
        "top_train_ranked": rows[:25],
        "val_stable_ranked": sorted(val_stable, key=lambda r: (r["val"].get("mean_pct", -999), r["train"].get("quality", -999)), reverse=True)[:25],
        "leakage_guard": {
            "rules_mined_on_train_only": True,
            "val_used_for_stability_filter_only": True,
            "oos_report_only": True,
            "facts_are_analyzer_prompt_summary_only": True,
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--oos-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-terms", type=int, default=2)
    p.add_argument("--min-train-n", type=int, default=80)
    p.add_argument("--min-train-ci-pct", type=float, default=0.0)
    p.add_argument("--min-val-n", type=int, default=20)
    p.add_argument("--min-val-mean-pct", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--min-fact-count", type=int, default=1)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_template_miner(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
