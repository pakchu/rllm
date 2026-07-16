"""Re-optimize the previous best portfolio with leak-safe state ensembles.

This experiment extends ``portfolio_opt_added_alpha_update`` without changing
its historical defaults or artifacts.  Allocation ranking remains train+2024
only.  Eval-2025 and YTD-2026 are attached after the rank order is frozen and
may only veto rank one.
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from dataclasses import asdict, fields, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import training.portfolio_opt_added_alpha_update as base
from training.state_model_top10_ensemble import (
    SCAN_PATHS,
    SLEEVE_NAMES,
    STRICT_MAJORITY_COUNT,
    TOP_N,
    build_state_model_top10_ensembles,
)


OUTPUT = "results/portfolio_state_ensemble_update_2026-07-16.json"
DOCS_OUTPUT = "docs/portfolio-state-ensemble-update-2026-07-16.md"
CANDIDATE_CONFIG = "configs/shadow/portfolio_state_ensemble_candidate_2026-07-16.json"
PREVIOUS_RESULT = Path("results/portfolio_added_alpha_update_2026-07-16.json")
ENSEMBLE_SLEEVES = tuple(SLEEVE_NAMES.values())
NEW_SLEEVES = base.NEW_SLEEVES + ENSEMBLE_SLEEVES
SLEEVES = tuple(base.LIVE_WEIGHTS) + NEW_SLEEVES
FAMILIES = {
    **base.FAMILIES,
    # Every state ensemble gates the same funding/premium base setup.  Keep
    # them in the existing family cap instead of pretending diversification.
    **{name: "funding_premium" for name in ENSEMBLE_SLEEVES},
}
DEFAULT_CONFIG = replace(
    base.Config(),
    output=OUTPUT,
    docs_output=DOCS_OUTPUT,
    candidate_config=CANDIDATE_CONFIG,
)


@contextmanager
def patched_portfolio_universe(
    cfg: base.Config,
) -> Iterator[dict[str, Any]]:
    """Inject ensemble events while preserving the historical optimizer."""
    originals = {
        "sleeves": base.SLEEVES,
        "new_sleeves": base.NEW_SLEEVES,
        "families": base.FAMILIES,
        "feature_frame": base.feature_frame,
        "split_arrays": base.split_arrays,
    }
    state: dict[str, Any] = {}

    def capture_feature_frame(market):
        frame = originals["feature_frame"](market)
        state["features"] = frame
        return frame

    def split_arrays_with_state_ensembles(events, market, masks):
        features = state.get("features")
        if features is None:
            raise RuntimeError("portfolio feature frame was not captured")
        signals, audit = build_state_model_top10_ensembles(market, features)
        event_counts: dict[str, dict[str, int]] = {}
        inactive = np.zeros(len(market), dtype=bool)
        for sleeve in ENSEMBLE_SLEEVES:
            event_counts[sleeve] = base.append_mask_policy(
                events,
                market,
                masks,
                name=sleeve,
                long_active=signals[sleeve],
                short_active=inactive,
                hold=576,
                stride=12,
                cost_rate=cfg.cost_rate,
            )
        audit["event_counts"] = event_counts
        state["audit"] = audit
        state.pop("features", None)
        return originals["split_arrays"](events, market, masks)

    base.SLEEVES = SLEEVES
    base.NEW_SLEEVES = NEW_SLEEVES
    base.FAMILIES = FAMILIES
    base.feature_frame = capture_feature_frame
    base.split_arrays = split_arrays_with_state_ensembles
    try:
        yield state
    finally:
        base.SLEEVES = originals["sleeves"]
        base.NEW_SLEEVES = originals["new_sleeves"]
        base.FAMILIES = originals["families"]
        base.feature_frame = originals["feature_frame"]
        base.split_arrays = originals["split_arrays"]


def _metric_cell(metric: dict[str, Any]) -> str:
    return (
        f"{metric['absolute_return_pct']:.2f}/{metric['cagr_pct']:.2f}/"
        f"{metric['strict_mdd_pct']:.2f}/{metric['cagr_to_strict_mdd']:.2f}/"
        f"{metric['trades']}"
    )


def _selection_score(row: dict[str, Any]) -> tuple[float, float, float]:
    train = row["stats"]["train"]
    test = row["stats"]["test2024"]
    train_ratio = max(0.0, float(train["cagr_to_strict_mdd"]))
    test_ratio = max(0.0, float(test["cagr_to_strict_mdd"]))
    return (
        min(train_ratio, test_ratio),
        float(np.sqrt(train_ratio * test_ratio)),
        test_ratio,
    )


def render_docs(report: dict[str, Any]) -> str:
    selected = report["frozen_pre2025_top1"]
    previous = report["previous_added_alpha_best"]["frozen_pre2025_top1"]
    lines = [
        "# State-ensemble portfolio allocation update (2026-07-16)",
        "",
        "Metric cells: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.",
        "",
        "## Selection contract",
        "",
        "- Portfolio ranking uses train and 2024 only.",
        "- 2025 and 2026 performance metrics are evaluated after rank order is frozen; they may veto rank 1 but never select rank 2+.",
        f"- Each new state sleeve is a predeclared strict majority: >= {STRICT_MAJORITY_COUNT} of the pre-evaluation Top-{TOP_N} family.",
        "- No exact Kalman/BOCPD/Semi-Markov representative was chosen from later-window passers.",
        f"- Gross <= {report['config']['gross_cap']}; family gross <= {report['config']['family_gross_cap']}; non-zero >= {report['config']['min_nonzero_weight']}; step {report['config']['weight_step']}.",
        "- All state sleeves share the funding/premium family cap because they gate the same base setup.",
        "",
        "## Decision",
        "",
        f"- Frozen rank-1 weights: `{selected['weights']}` (gross {selected['gross']:.2f}).",
        f"- Frozen rank-1 future veto: **{'PASS' if selected['future_veto_passed'] else 'FAIL'}**.",
        f"- Replaces previous added-alpha candidate: **{'YES' if report['replace_previous_candidate'] else 'NO'}**.",
        f"- Disposition: **{report['deployment_disposition']}**.",
        "",
        "| Portfolio | Train | 2024 selection | 2025 report | 2026H1 report |",
        "|---|---:|---:|---:|---:|",
        f"| Previous added-alpha best | {_metric_cell(previous['stats']['train'])} | {_metric_cell(previous['stats']['test2024'])} | {_metric_cell(previous['stats']['eval2025'])} | {_metric_cell(previous['stats']['ytd2026'])} |",
        f"| State-ensemble frozen rank 1 | {_metric_cell(selected['stats']['train'])} | {_metric_cell(selected['stats']['test2024'])} | {_metric_cell(selected['stats']['eval2025'])} | {_metric_cell(selected['stats']['ytd2026'])} |",
        "",
        "## Top frozen pre-2025 ranks",
        "",
        "| # | Gross | Weights | Train | 2024 | 2025 report | 2026H1 report | Veto |",
        "|---:|---:|---|---:|---:|---:|---:|:---:|",
    ]
    for index, row in enumerate(report["top_pre2025"][:20], start=1):
        stats = row["stats"]
        lines.append(
            f"| {index} | {row['gross']:.2f} | `{row['weights']}` | "
            f"{_metric_cell(stats['train'])} | {_metric_cell(stats['test2024'])} | "
            f"{_metric_cell(stats['eval2025'])} | {_metric_cell(stats['ytd2026'])} | "
            f"{'PASS' if row['future_veto_passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- A future-veto failure retains the previous shadow candidate; lower-ranked future passers are not promoted.",
        "- The search is deterministic seeded sampling plus exact 0.05-grid refinement, not a proof of the global combinatorial optimum.",
        "- Reported later windows have prior research exposure, so even a pass remains forward-shadow only.",
        "- The current live config is not modified by this experiment.",
    ]
    return "\n".join(lines) + "\n"


def build_candidate_config(report: dict[str, Any], cfg: base.Config) -> dict[str, Any]:
    """Keep rejected research weights out of the deployable ``weights`` field."""
    selected = report["frozen_pre2025_top1"]
    previous = report["previous_added_alpha_best"]["frozen_pre2025_top1"]
    replace_previous = bool(report["replace_previous_candidate"])
    active = selected if replace_previous else previous
    active_state_sleeves = sorted(set(active["weights"]) & set(ENSEMBLE_SLEEVES))
    rejected_state_sleeves = sorted(set(selected["weights"]) & set(ENSEMBLE_SLEEVES))
    return {
        "name": "portfolio_state_ensemble_candidate_2026_07_16",
        "status": report["deployment_disposition"],
        "as_of": "2026-07-16",
        "weights": active["weights"],
        "gross_weight": active["gross"],
        "selected_state_ensemble_sleeves": active_state_sleeves,
        "selection": "frozen train+2024 rank 1; 2025/2026 veto only; no reranking",
        "future_veto_passed": selected["future_veto_passed"],
        "replace_previous_candidate": replace_previous,
        "rejected_frozen_rank1": None
        if replace_previous
        else {
            "weights": selected["weights"],
            "gross_weight": selected["gross"],
            "state_ensemble_sleeves": rejected_state_sleeves,
            "reason": "frozen rank 1 failed the 2025/2026 future veto",
        },
        "retained_previous_result": None if replace_previous else str(PREVIOUS_RESULT),
        "research_contaminated": True,
        "source_result": cfg.output,
        "protocol_hash": report["protocol_hash"],
        "accounting_version": base.ACCOUNTING_VERSION,
    }


def run(cfg: base.Config) -> dict[str, Any]:
    if not PREVIOUS_RESULT.exists():
        raise FileNotFoundError(PREVIOUS_RESULT)
    previous = json.loads(PREVIOUS_RESULT.read_text())
    # The base runner writes all three artifacts itself.  Isolate those writes
    # so a wrapper/audit failure cannot leave a rejected row at a final path.
    with TemporaryDirectory(prefix="rllm-state-ensemble-") as temporary:
        inner_cfg = replace(
            cfg,
            output=str(Path(temporary) / "result.json"),
            docs_output=str(Path(temporary) / "report.md"),
            candidate_config=str(Path(temporary) / "candidate.json"),
        )
        with patched_portfolio_universe(inner_cfg) as extension_state:
            report = base.run(inner_cfg)
    audit = extension_state.get("audit")
    if not audit:
        raise RuntimeError("state-model ensemble audit was not produced")

    state_inputs = {
        f"state_model_{family}_scan": base.file_record(path)
        for family, path in SCAN_PATHS.items()
    }
    previous_input = base.file_record(PREVIOUS_RESULT)
    state_inputs["previous_added_alpha_result"] = previous_input
    report["schema_version"] = 2
    report["mode"] = "pre2025_state_ensemble_allocation_rank_future_veto_only"
    report["config"] = asdict(cfg)
    report["input_provenance"].update(state_inputs)
    report["source_validation"]["state_model_top10_ensembles"] = audit
    report["candidate_universe"]["state_ensemble_rule"] = {
        "members": TOP_N,
        "required_votes": STRICT_MAJORITY_COUNT,
        "threshold_tuned": False,
        "future_fields_used": False,
        "sleeves": list(ENSEMBLE_SLEEVES),
        "family_cap": "shared funding_premium cap",
    }
    report["candidate_universe"]["excluded"][
        "kalman_bocpd_semimarkov_representatives"
    ] = "future-passer representatives remain excluded; only fixed Top-10 majority ensembles were added"
    old_protocol_hash = report["protocol_hash"]
    report["base_protocol_hash"] = old_protocol_hash
    report["protocol_hash"] = base.json_hash(
        {
            "base_protocol_hash": old_protocol_hash,
            "state_ensemble_rule": report["candidate_universe"]["state_ensemble_rule"],
            "state_scan_sha256": {
                name: record["sha256"]
                for name, record in state_inputs.items()
                if name.startswith("state_model_")
            },
            "previous_added_alpha_result_sha256": previous_input["sha256"],
            "ensemble_signal_hashes": {
                family: family_audit["ensemble_signal_hash"]
                for family, family_audit in audit["families"].items()
            },
        }
    )
    report["previous_added_alpha_best"] = {
        "source_result": str(PREVIOUS_RESULT),
        "protocol_hash": previous["protocol_hash"],
        "frozen_pre2025_top1": previous["frozen_pre2025_top1"],
    }
    selected = report["frozen_pre2025_top1"]
    prior_selected = previous["frozen_pre2025_top1"]
    improved_pre2025 = _selection_score(selected) > _selection_score(prior_selected)
    selected_ensemble = sorted(set(selected["weights"]) & set(ENSEMBLE_SLEEVES))
    replace_previous = bool(
        selected["future_veto_passed"] and improved_pre2025 and selected_ensemble
    )
    report["selected_state_ensemble_sleeves"] = selected_ensemble
    report["improved_pre2025_selection_score"] = improved_pre2025
    report["replace_previous_candidate"] = replace_previous
    report["deployment_disposition"] = (
        "state_ensemble_forward_shadow_candidate_not_live"
        if replace_previous
        else "retain_previous_added_alpha_shadow_candidate"
    )

    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    docs_path = Path(cfg.docs_output)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(render_docs(report))
    candidate = build_candidate_config(report, cfg)
    candidate_path = Path(cfg.candidate_config)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps(candidate, indent=2, ensure_ascii=False) + "\n")
    return report


def parse_args() -> base.Config:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in fields(DEFAULT_CONFIG):
        name = "--" + field.name.replace("_", "-")
        default = getattr(DEFAULT_CONFIG, field.name)
        parser.add_argument(name, type=type(default), default=default)
    return base.Config(**vars(parser.parse_args()))


def main() -> None:
    report = run(parse_args())
    selected = report["frozen_pre2025_top1"]
    print(
        json.dumps(
            {
                "weights": selected["weights"],
                "gross": selected["gross"],
                "stats": selected["stats"],
                "future_veto_passed": selected["future_veto_passed"],
                "replace_previous_candidate": report["replace_previous_candidate"],
                "deployment_disposition": report["deployment_disposition"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
