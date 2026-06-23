"""Build month-level side-map reliability SFT rows.

Targets come from monthly pass/invert/block audit labels. Prompts include only
state that would be known before the target month starts: prior rolling
validation score for the target month and realized side-map history from earlier
months.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SideMapReliabilitySftCfg:
    monthly_audit_json: str
    rolling_summary_json: str
    output_jsonl: str
    summary_output: str = ""
    train_end_month: str = "2024-12"
    val_end_month: str = "2025-12"
    history_months: int = 6


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _month_scores(summary: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in summary.get("months", []):
        try:
            out[str(m.get("month"))] = float((m.get("selected") or {}).get("score", float("-inf")))
        except Exception:
            out[str(m.get("month"))] = float("-inf")
    return out


def _score_bucket(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 0.5:
        return "positive"
    if score < -500.0:
        return "severe_decay"
    if score < 0.0:
        return "negative"
    return "weak_positive"


def _split(month: str, train_end: str, val_end: str) -> str:
    if month <= train_end:
        return "train"
    if month <= val_end:
        return "val"
    return "eval"


def _prompt(month: str, score: float | None, history: list[dict[str, Any]]) -> str:
    lines = [
        "You are a BTCUSDT side-map reliability classifier for an RLLM trading policy.",
        "Use only information known before the target month starts.",
        "Classify whether the base side map should be trusted, inverted, or avoided for the target month.",
        "Return one JSON object with keys: side_map, confidence, reason_code.",
        "Allowed side_map: normal, inverse, unreliable.",
        "",
        f"target_month: {month}",
        f"prior_validation_score_bucket: {_score_bucket(score)}",
        "prior_side_map_history:",
    ]
    if not history:
        lines.append("- none")
    for h in history:
        lines.append(
            "- month={month} label={label} pass_cagr_bucket={pass_bucket} invert_cagr_bucket={invert_bucket}".format(
                month=h["month"],
                label=h["label"],
                pass_bucket=_ret_bucket(float(h["variants"]["pass"]["sim"].get("cagr_pct", 0.0) or 0.0)),
                invert_bucket=_ret_bucket(float(h["variants"]["invert"]["sim"].get("cagr_pct", 0.0) or 0.0)),
            )
        )
    lines.append("Policy intent: choose normal only when side mapping is stable, inverse only when reversal is clear, otherwise unreliable.")
    return "\n".join(lines)


def _ret_bucket(x: float) -> str:
    if x >= 100:
        return "very_positive"
    if x >= 20:
        return "positive"
    if x > 0:
        return "slightly_positive"
    if x <= -50:
        return "very_negative"
    if x < 0:
        return "negative"
    return "flat"


def build(cfg: SideMapReliabilitySftCfg) -> dict[str, Any]:
    audit = json.loads(Path(cfg.monthly_audit_json).read_text())
    summary = json.loads(Path(cfg.rolling_summary_json).read_text())
    scores = _month_scores(summary)
    months = sorted(audit.get("months", []), key=lambda r: str(r.get("month")))
    rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for i, row in enumerate(months):
        month = str(row["month"])
        label = str(row["label"])
        history = months[max(0, i - int(cfg.history_months)) : i]
        split = _split(month, cfg.train_end_month, cfg.val_end_month)
        counts.setdefault(split, {})[label] = counts.setdefault(split, {}).get(label, 0) + 1
        target = {"side_map": label, "confidence": "HIGH", "reason_code": f"monthly_audit_{label}"}
        rows.append({
            "task": "side_map_reliability_sft",
            "split": split,
            "month": month,
            "prompt": _prompt(month, scores.get(month), history),
            "target": json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "source": {
                "prior_validation_score_bucket": _score_bucket(scores.get(month)),
                "history_months": [h["month"] for h in history],
                "audit_label_source": cfg.monthly_audit_json,
            },
            "leakage_guard": {
                "prompt_uses_target_month_outcome": False,
                "target_uses_target_month_audit_label_for_training": True,
                "history_uses_only_prior_month_labels": True,
                "not_a_live_selector_without_rolling_model_eval": True,
            },
        })
    _write_jsonl(cfg.output_jsonl, rows)
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "rows": len(rows), "counts": counts, "leakage_guard": {"prompts_are_prior_only": True, "targets_are_audit_labels": True}}
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build side-map reliability SFT rows")
    p.add_argument("--monthly-audit-json", required=True)
    p.add_argument("--rolling-summary-json", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--train-end-month", default=SideMapReliabilitySftCfg.train_end_month)
    p.add_argument("--val-end-month", default=SideMapReliabilitySftCfg.val_end_month)
    p.add_argument("--history-months", type=int, default=SideMapReliabilitySftCfg.history_months)
    return p.parse_args()


def main() -> None:
    report = build(SideMapReliabilitySftCfg(**vars(parse_args())))
    print(json.dumps({"output": report["config"]["output_jsonl"], "rows": report["rows"], "counts": report["counts"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
