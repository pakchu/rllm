"""Apply additive action biases to a stored VLM likelihood report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from models.option_b_vlm import get_action_labels
from training.eval_vlm_policy import select_action_from_scores, summarize_action_metrics


def _parse_label_value_specs(specs: list[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for spec in specs or []:
        text = str(spec).strip()
        if "=" not in text:
            raise ValueError(f"bias spec must look like LABEL=VALUE, got {spec}")
        label, value = text.split("=", 1)
        out[str(label).strip().upper()] = float(value)
    return out


def _load_biases(bias_specs: list[str] | None, calibration_report: str | None) -> dict[str, float]:
    biases: dict[str, float] = {}
    if calibration_report:
        payload = json.loads(Path(calibration_report).read_text())
        best = payload.get("best") or {}
        loaded = best.get("biases") or payload.get("biases") or {}
        biases.update({str(k).upper(): float(v) for k, v in loaded.items()})
    biases.update(_parse_label_value_specs(bias_specs))
    if not biases:
        raise ValueError("No biases provided. Use --bias LABEL=VALUE or --calibration-report.")
    return biases


def apply_vlm_bias_report(
    *,
    input_report: str,
    output: str,
    action_schema: str | None = None,
    biases: dict[str, float],
) -> dict[str, Any]:
    """Rewrite predictions/adjusted_scores in an eval-vlm report using additive biases."""
    report = json.loads(Path(input_report).read_text())
    schema = str(action_schema or report.get("action_schema") or "buy_hold_sell")
    labels = get_action_labels(schema)
    rows = report.get("action_scores")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"report has no action_scores: {input_report}")

    effective_biases = {label: float(biases.get(label, 0.0)) for label in labels}
    targets: list[str] = []
    preds: list[str] = []
    adjusted_rows: list[dict[str, Any]] = []
    for row in rows:
        scores = row.get("scores")
        if not isinstance(scores, dict):
            raise ValueError("action_score row has no scores dict")
        pred, adjusted = select_action_from_scores(
            scores={str(k): float(v) for k, v in scores.items()},
            action_biases=effective_biases,
            labels=labels,
        )
        new_row = dict(row)
        new_row["pred"] = pred
        new_row["adjusted_scores"] = adjusted
        adjusted_rows.append(new_row)
        targets.append(str(row.get("target", "")))
        preds.append(pred)

    report["action_scores"] = adjusted_rows
    report["metrics"] = summarize_action_metrics(
        targets=targets,
        predictions=preds,
        labels=labels,
    )
    report["bias_application"] = {
        "input_report": str(Path(input_report).resolve()),
        "action_schema": schema,
        "biases": effective_biases,
    }
    decision = dict(report.get("decision") or {})
    decision["action_biases"] = effective_biases
    report["decision"] = decision

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply additive biases to VLM action scores.")
    parser.add_argument("--input-report", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--action-schema",
        type=str,
        default="",
        choices=["", "buy_hold_sell", "trade_gate", "trade_side"],
    )
    parser.add_argument("--calibration-report", type=str, default="")
    parser.add_argument(
        "--bias",
        type=str,
        action="append",
        default=[],
        help="Repeatable LABEL=VALUE additive bias. Overrides calibration values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = apply_vlm_bias_report(
        input_report=args.input_report,
        output=args.output,
        action_schema=args.action_schema or None,
        biases=_load_biases(args.bias, args.calibration_report or None),
    )
    print(f"[apply-vlm-bias] saved={Path(args.output).resolve()}")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
