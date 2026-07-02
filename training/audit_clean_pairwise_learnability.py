"""Audit whether clean pairwise labels are learnable from visible prompt features."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class PairwiseLearnabilityAuditConfig:
    jsonl: str
    output: str = ""


def _num(x: Any) -> float:
    try:
        return float(x or 0.0)
    except Exception:
        return 0.0


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(str(row["prompt"]).split("\n")[-1])


def _target(row: dict[str, Any]) -> str:
    raw = row.get("target", row.get("completion", "A"))
    try:
        obj = json.loads(str(raw))
        return str(obj.get("choice", raw)).upper()
    except Exception:
        return str(raw).strip().upper()[:1] or "A"


def _manual_score(opt: dict[str, Any]) -> float:
    latest = opt.get("latest_evidence") or {}
    return (
        _num(opt.get("pre_fold_score"))
        + 0.05 * _num(opt.get("evidence_count"))
        + 0.5 * _num(latest.get("cagr_to_mdd"))
        + 0.5 * _num(latest.get("weighted_score"))
        + 0.2 * _num(latest.get("raw_score"))
        + 0.01 * _num(latest.get("trades"))
        - 0.2 * _num(latest.get("p_value"))
    )


def _rules() -> dict[str, Callable[[dict[str, Any], dict[str, Any]], str]]:
    return {
        "higher_pre_fold_score": lambda a, b: "A" if _num(a.get("pre_fold_score")) >= _num(b.get("pre_fold_score")) else "B",
        "lower_pre_fold_score": lambda a, b: "B" if _num(a.get("pre_fold_score")) >= _num(b.get("pre_fold_score")) else "A",
        "higher_evidence_count": lambda a, b: "A" if _num(a.get("evidence_count")) >= _num(b.get("evidence_count")) else "B",
        "prefer_abstain": lambda a, b: "A" if a.get("family") == "ABSTAIN" else ("B" if b.get("family") == "ABSTAIN" else "A"),
        "avoid_abstain": lambda a, b: "B" if a.get("family") == "ABSTAIN" else ("A" if b.get("family") == "ABSTAIN" else "A"),
        "manual_combo": lambda a, b: "A" if _manual_score(a) >= _manual_score(b) else "B",
    }


def run(cfg: PairwiseLearnabilityAuditConfig) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(cfg.jsonl).read_text().splitlines() if line.strip()]
    rule_counts = {name: 0 for name in _rules()}
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    by_order: dict[str, Counter[str]] = defaultdict(Counter)
    target_counts: Counter[str] = Counter()
    for row in rows:
        pl = _payload(row)
        a, b = pl["option_a"], pl["option_b"]
        target = _target(row)
        target_counts[target] += 1
        for name, fn in _rules().items():
            rule_counts[name] += int(fn(a, b) == target)
        fam = str(row.get("target_family"))
        by_family[fam]["n"] += 1
        by_family[fam]["a_targets"] += int(target == "A")
        by_order[str(row.get("order_variant"))]["n"] += 1
        by_order[str(row.get("order_variant"))]["a_targets"] += int(target == "A")
    n = max(1, len(rows))
    report = {
        "config": asdict(cfg),
        "rows": len(rows),
        "target_counts": dict(target_counts),
        "rule_accuracy": {name: rule_counts[name] / n for name in sorted(rule_counts)},
        "family_target_a_rate": {k: v["a_targets"] / max(1, v["n"]) for k, v in sorted(by_family.items())},
        "order_target_a_rate": {k: v["a_targets"] / max(1, v["n"]) for k, v in sorted(by_order.items())},
    }
    if cfg.output:
        Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--output", default="")
    print(json.dumps(run(PairwiseLearnabilityAuditConfig(**vars(p.parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
