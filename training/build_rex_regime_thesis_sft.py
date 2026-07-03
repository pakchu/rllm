"""Build compact regime-thesis SFT rows for the REX pullback gate.

The current profitable surface is not a free-form analyzer/trader stack; it is a
fixed REX candidate generator plus an abstain regime.  This exporter turns a
validated symbolic gate into a small LLM target so the model can learn the
reasoning-shaped policy: trade only when the signal-time regime supports the REX
pullback/reclaim thesis, otherwise abstain.

Targets are generated only from signal-time features and fixed gates. Future path
reward is copied into metadata for audit, never used to create the rule label.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Gate:
    feature: str
    op: str
    threshold: float

    def match(self, snapshot: dict[str, Any]) -> bool:
        value = snapshot.get(self.feature)
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            return False
        x = float(value)
        return x >= self.threshold if self.op == ">=" else x <= self.threshold

    def text(self) -> str:
        return f"{self.feature} {self.op} {self.threshold:.6f}"


@dataclass(frozen=True)
class RexRegimeThesisSftCfg:
    train_jsonl: str
    test_jsonl: str
    eval_jsonl: str
    train_output: str
    test_output: str
    eval_output: str
    summary_output: str
    gates_json: str = '[{"feature":"range_vol","op":">=","threshold":0.023959233645008706},{"feature":"kimchi_premium_change","op":"<=","threshold":0.0}]'
    system_prompt: str = "You are a compact BTCUSDT futures RLLM regime-thesis policy. Return exactly one label: TRADE or ABSTAIN."
    target_format: str = "decision_label"


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _parse_gates(raw: str) -> tuple[Gate, ...]:
    data = json.loads(raw)
    if not isinstance(data, list) or not data:
        raise ValueError("gates_json must be a non-empty JSON list")
    gates = []
    for item in data:
        gates.append(Gate(feature=str(item["feature"]), op=str(item["op"]), threshold=float(item["threshold"])))
    for gate in gates:
        if gate.op not in {">=", "<="}:
            raise ValueError(f"unsupported gate op: {gate.op}")
    return tuple(gates)


def _bucket(value: float, cuts: tuple[float, ...], labels: tuple[str, ...]) -> str:
    for cut, label in zip(cuts, labels):
        if float(value) < cut:
            return label
    return labels[-1]


def _num(snapshot: dict[str, Any], key: str) -> float:
    value = snapshot.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) and np.isfinite(float(value)) else 0.0


def _context(snapshot: dict[str, Any]) -> dict[str, str]:
    return {
        "range_vol": _bucket(_num(snapshot, "range_vol"), (0.010, 0.018, 0.024, 0.035), ("quiet", "normal", "expanded", "high", "extreme")),
        "window_drawdown": _bucket(_num(snapshot, "window_drawdown"), (0.005, 0.012, 0.025, 0.050), ("none", "small", "pullback", "deep", "stress")),
        "rex_144_width": _bucket(_num(snapshot, "rex_144_range_width_pct"), (0.010, 0.018, 0.024, 0.040), ("tight", "normal", "wide", "very_wide", "extreme")),
        "rex_2016_from_min": _bucket(_num(snapshot, "rex_2016_cur_to_min_pct"), (0.02, 0.05, 0.10, 0.20), ("near_low", "lifted", "reclaimed", "extended", "overextended")),
        "dxy_momentum": "nonnegative_or_flat" if _num(snapshot, "dxy_momentum") >= -0.000254 else "negative",
        "kimchi_premium_change": "non_positive" if _num(snapshot, "kimchi_premium_change") <= 0.0 else "positive",
        "htf_1w_position": _bucket(_num(snapshot, "htf_1w_range_pos"), (0.25, 0.50, 0.75), ("lower_quartile", "lower_mid", "upper_mid", "upper_quartile")),
    }


def _prompt(row: dict[str, Any], gates: tuple[Gate, ...]) -> str:
    snap = row.get("feature_snapshot", {}) if isinstance(row.get("feature_snapshot"), dict) else {}
    action = row.get("action", {}) if isinstance(row.get("action"), dict) else {}
    ctx = _context(snap)
    compact_nums = {
        "range_vol": _num(snap, "range_vol"),
        "kimchi_premium_change": _num(snap, "kimchi_premium_change"),
        "rex_144_range_width_pct": _num(snap, "rex_144_range_width_pct"),
        "rex_2016_cur_to_min_pct": _num(snap, "rex_2016_cur_to_min_pct"),
        "window_drawdown": _num(snap, "window_drawdown"),
        "dxy_momentum": _num(snap, "dxy_momentum"),
        "rex_threshold_ratio": _num(snap, "rex_threshold_ratio"),
    }
    return "\n".join(
        [
            "Task: decide whether to trade a fixed REX pullback/reclaim BTCUSDT futures candidate.",
            "Use only signal-time context. Prefer abstain unless the regime supports the REX thesis.",
            "Return exactly one label: TRADE or ABSTAIN. The thesis and risk context are in the prompt.",
            f"Fixed regime prior: {' AND '.join(g.text() for g in gates)}.",
            f"Date: {row.get('date')}",
            f"Candidate: family={action.get('family')}; side={str(action.get('side', '')).upper()}; hold_bars={action.get('hold_bars')}",
            "Categorical regime: " + "; ".join(f"{k}={v}" for k, v in sorted(ctx.items())),
            "Compact numeric snapshot: " + "; ".join(f"{k}={v:+.5f}" for k, v in compact_nums.items()),
        ]
    )


def _target(row: dict[str, Any], gates: tuple[Gate, ...]) -> dict[str, Any]:
    snap = row.get("feature_snapshot", {}) if isinstance(row.get("feature_snapshot"), dict) else {}
    action = row.get("action", {}) if isinstance(row.get("action"), dict) else {}
    active = all(g.match(snap) for g in gates)
    side = str(action.get("side", "NONE")).upper() if active else "NONE"
    if active:
        return {
            "decision": "TRADE",
            "action_side": side,
            "size_bucket": "base_0p5x_or_scale_if_portfolio_mdd_allows",
            "confidence": "medium",
            "rationale_class": "expanded_range_with_nonpositive_kimchi_premium_change_rex_pullback",
        }
    failed = [g.feature for g in gates if not g.match(snap)]
    return {
        "decision": "ABSTAIN",
        "action_side": "NONE",
        "size_bucket": "zero",
        "confidence": "medium",
        "rationale_class": "regime_prior_failed_" + "_and_".join(failed[:2]),
    }


def _target_text(target: dict[str, Any], target_format: str) -> str:
    fmt = str(target_format).strip().lower()
    if fmt == "decision_label":
        return str(target["decision"])
    if fmt == "label_then_json":
        return str(target["decision"]) + "\n" + json.dumps(target, ensure_ascii=False, sort_keys=True)
    if fmt == "json":
        return json.dumps(target, ensure_ascii=False, sort_keys=True)
    raise ValueError("target_format must be one of {'decision_label', 'label_then_json', 'json'}")


def _message_row(row: dict[str, Any], split: str, gates: tuple[Gate, ...], cfg: RexRegimeThesisSftCfg) -> dict[str, Any]:
    target = _target(row, gates)
    target_text = _target_text(target, cfg.target_format)
    prompt = _prompt(row, gates)
    return {
        "task": "rex_regime_thesis_policy_sft",
        "split": split,
        "date": row.get("date"),
        "signal_pos": row.get("signal_pos"),
        "prompt": prompt,
        "target": target_text,
        "messages": [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target_text},
        ],
        "metadata": {
            "action": row.get("action", {}),
            "reward": row.get("reward", {}),
            "target": target,
            "gate_matches": {g.feature: g.match(row.get("feature_snapshot", {}) if isinstance(row.get("feature_snapshot"), dict) else {}) for g in gates},
            "leakage_guard": {
                "target_generated_from_signal_time_fixed_gates_only": True,
                "future_reward_copied_for_audit_not_labeling": True,
            },
        },
    }


def _write(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + ("\n" if rows else ""))


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets = [r.get("metadata", {}).get("target", {}) for r in rows]
    return {
        "rows": len(rows),
        "decision_counts": dict(Counter(t["decision"] for t in targets)),
        "side_counts": dict(Counter(t["action_side"] for t in targets)),
        "rationale_counts": dict(Counter(t["rationale_class"] for t in targets)),
        "prompt_chars": {
            "min": min((len(r["prompt"]) for r in rows), default=0),
            "max": max((len(r["prompt"]) for r in rows), default=0),
            "mean": sum(len(r["prompt"]) for r in rows) / max(1, len(rows)),
        },
    }


def run(cfg: RexRegimeThesisSftCfg) -> dict[str, Any]:
    gates = _parse_gates(cfg.gates_json)
    raw = {"train": _load_jsonl(cfg.train_jsonl), "test": _load_jsonl(cfg.test_jsonl), "eval": _load_jsonl(cfg.eval_jsonl)}
    out = {split: [_message_row(row, split, gates, cfg) for row in rows] for split, rows in raw.items()}
    _write(cfg.train_output, out["train"])
    _write(cfg.test_output, out["test"])
    _write(cfg.eval_output, out["eval"])
    report = {
        "config": asdict(cfg),
        "gates": [g.__dict__ for g in gates],
        "train": _summary(out["train"]),
        "test": _summary(out["test"]),
        "eval": _summary(out["eval"]),
        "outputs": {"train": cfg.train_output, "test": cfg.test_output, "eval": cfg.eval_output},
        "leakage_guard": {
            "uses_fixed_gate_supplied_by_prior_no_leak_sweep": True,
            "targets_do_not_read_future_reward": True,
            "test_eval_untouched_chronological_splits": True,
            "target_format": cfg.target_format,
        },
    }
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build compact REX regime-thesis SFT rows")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--test-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--train-output", required=True)
    p.add_argument("--test-output", required=True)
    p.add_argument("--eval-output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--gates-json", default=RexRegimeThesisSftCfg.gates_json)
    p.add_argument("--system-prompt", default=RexRegimeThesisSftCfg.system_prompt)
    p.add_argument("--target-format", choices=["decision_label", "label_then_json", "json"], default=RexRegimeThesisSftCfg.target_format)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(RexRegimeThesisSftCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
