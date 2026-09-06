"""Build leakage-ordered SFT rows from frozen alpha formulas and gate artifacts.

Each prompt contains the preregistered signal contract plus evidence available at
exactly one scientific gate.  Later-stage artifacts are never included.  The
target is TRADE when the candidate may advance and NO_TRADE at a terminal gate.
All rows for one preregistration stay in the same deterministic split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


STAGE_SPECS = (
    ("source", "support", "support_passed"),
    ("gross9", "gross9_novelty", "advance_to_economic_outcomes"),
    ("train", "train_economics", "passed"),
    ("test", "test_economics", "passed"),
    ("eval", "eval_economics", "passed"),
    ("final", "final_economics", "passed"),
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _compact(value: Any, limit: int = 5000) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return raw if len(raw) <= limit else raw[: limit - 24] + '..."<truncated>"}'


def _formula(prereg: dict[str, Any]) -> dict[str, Any]:
    mechanism = prereg.get("mechanism", {})
    return {
        "policy_id": prereg.get("policy_id"),
        "protocol_version": prereg.get("protocol_version"),
        "mechanism": {
            key: mechanism.get(key)
            for key in ("claim", "side", "why_distinct", "why_suited_to_volatile_regimes", "volatile_market_target", "why_low_gross9_overlap_is_plausible")
            if mechanism.get(key) is not None
        },
        "features": prereg.get("features", prereg.get("construction", {})),
        "policy": prereg.get("policy", {}),
        "clock": prereg.get("clock", {}),
        "source_support_gates": prereg.get("source_support_gates", {}),
        "gross9_novelty_gates": prereg.get("gross9_novelty_gates", prereg.get("novelty_gates", {})),
        "economic_gates": prereg.get("economic_gates", {}),
        "stopping_rule": prereg.get("stopping_rule"),
    }


def _evidence(stage: str, artifact: dict[str, Any]) -> dict[str, Any]:
    common = {"policy_id": artifact.get("policy_id"), "stage": stage}
    if stage == "source":
        return {
            **common,
            "support": artifact.get("support", {}),
            "activation": artifact.get("activation", artifact.get("active_veto_accounting", {})),
        }
    if stage == "gross9":
        sleeves = artifact.get("gross9_sleeves", {})
        return {
            **common,
            "limits": artifact.get("limits", {}),
            "sleeves": {name: {"metrics": row.get("metrics", {})} for name, row in sleeves.items()},
        }
    return {
        **common,
        "window": artifact.get("window"),
        "primary": artifact.get("primary", {}),
    }


def _target(stage: str, artifact: dict[str, Any], pass_key: str) -> str:
    value = artifact.get(pass_key)
    if value is None and stage in {"train", "test", "eval", "final"}:
        value = artifact.get("advance_to_next_stage")
        if stage == "final" and value is None:
            value = artifact.get("advance_to_post_stage_volatility_audit")
    return "TRADE" if value is True else "NO_TRADE"


def _prompt(formula: dict[str, Any], stage: str, evidence: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are the scientific gate controller for RLLM BTC alpha research.",
            "Apply the frozen contract exactly. Do not repair, tune, substitute, invert, or promote controls.",
            "Use only the formula and evidence shown for the current stage; later-stage evidence is unavailable.",
            "Return exactly one token: TRADE if the candidate may advance, otherwise NO_TRADE.",
            f"current_stage: {stage}",
            f"frozen_formula: {_compact(formula)}",
            f"current_stage_evidence: {_compact(evidence)}",
        ]
    )


def _split(group: str, eval_percent: int) -> str:
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return "eval" if bucket < int(eval_percent) else "train"


def _artifact_for(results_dir: Path, slug: str, suffix: str) -> Path | None:
    rows = sorted(results_dir.glob(f"{slug}_{suffix}_*.json"))
    return rows[-1] if rows else None


def build(results_dir: Path, train_output: Path, eval_output: Path, summary_output: Path, *, eval_percent: int = 15) -> dict[str, Any]:
    preregs = sorted(results_dir.glob("high_volatility_*_preregistration_*.json"))
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for prereg_path in preregs:
        slug = prereg_path.name.split("_preregistration_")[0]
        try:
            prereg = _read(prereg_path)
        except Exception:
            skipped["invalid_preregistration"] += 1
            continue
        policy = str(prereg.get("policy_id") or slug)
        group = f"{slug}|{policy}"
        formula = _formula(prereg)
        predecessor_passed = True
        for stage, suffix, pass_key in STAGE_SPECS:
            artifact_path = _artifact_for(results_dir, slug, suffix)
            if artifact_path is None:
                skipped[f"missing_{stage}"] += 1
                break
            if not predecessor_passed:
                skipped[f"sealed_after_failure_{stage}"] += 1
                break
            try:
                artifact = _read(artifact_path)
            except Exception:
                skipped[f"invalid_{stage}"] += 1
                break
            target = _target(stage, artifact, pass_key)
            evidence = _evidence(stage, artifact)
            rows.append(
                {
                    "task": f"alpha_formula_gate_{stage}",
                    "prompt": _prompt(formula, stage, evidence),
                    "target": target,
                    "policy_id": policy,
                    "group_id": group,
                    "stage": stage,
                    "metadata": {
                        "preregistration": str(prereg_path),
                        "artifact": str(artifact_path),
                        "split_guard": "all stages for one preregistration share one hash split",
                        "leakage_guard": "prompt contains no evidence from stages after current_stage",
                    },
                }
            )
            predecessor_passed = target == "TRADE"

    train_rows, eval_rows = [], []
    for row in rows:
        (eval_rows if _split(str(row["group_id"]), eval_percent) == "eval" else train_rows).append(row)
    for path, selected in ((train_output, train_rows), (eval_output, eval_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected))
    train_groups = {row["group_id"] for row in train_rows}
    eval_groups = {row["group_id"] for row in eval_rows}
    report = {
        "results_dir": str(results_dir),
        "preregistrations": len(preregs),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_groups": len(train_groups),
        "eval_groups": len(eval_groups),
        "group_overlap": sorted(train_groups & eval_groups),
        "targets": dict(Counter(row["target"] for row in rows)),
        "stages": dict(Counter(row["stage"] for row in rows)),
        "tasks": dict(Counter(row["task"] for row in rows)),
        "skipped": dict(sorted(skipped.items())),
        "eval_percent": int(eval_percent),
        "leakage_guard": {
            "group_overlap_zero": not bool(train_groups & eval_groups),
            "future_stage_evidence_in_prompt": False,
            "terminal_first_failure_order_enforced": True,
        },
        "sha256": {
            "train": hashlib.sha256(train_output.read_bytes()).hexdigest(),
            "eval": hashlib.sha256(eval_output.read_bytes()).hexdigest(),
        },
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--train-output", type=Path, required=True)
    p.add_argument("--eval-output", type=Path, required=True)
    p.add_argument("--summary-output", type=Path, required=True)
    p.add_argument("--eval-percent", type=int, default=15)
    print(json.dumps(build(**vars(p.parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
