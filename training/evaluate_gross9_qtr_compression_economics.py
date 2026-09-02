"""Sequential economics evaluator for G9QTR-COMPRESS-8.

This reuses the already frozen G9QTR-DISTILL-8 clock package and simulator, but
has independent replacement/compression authorization.  It does not require the
Gross9 additive near-6h novelty gate to pass; that overlap is disclosed and all
non-near structural overlap checks must pass before economics opens.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import evaluate_gross9_qtr_distill_economics as distill_eval
from training import preregister_gross9_qtr_compression as prereg

POLICY_ID = prereg.POLICY_ID
SOURCE_POLICY_ID = prereg.SOURCE_POLICY_ID
PROTOCOL_VERSION = "gross9_qtr_compression_fixed_quantity_portfolio_economics_v1"
PREREGISTRATION = prereg.DEFAULT_OUTPUT
CLOCK_PACKAGE = prereg.SOURCE_CLOCK_PACKAGE
TERMINAL_NOVELTY = prereg.TERMINAL_ADDITIVE_NOVELTY
STAGES = dict(distill_eval.STAGES)
PREDECESSOR = dict(distill_eval.PREDECESSOR)
OUTPUTS = {stage: Path(path) for stage, path in prereg.OUTPUTS.items()}
MIN_NONZERO_SIGNED_EPISODES = dict(prereg.MIN_NONZERO_SIGNED_EPISODES)
OOS_CLUSTER_P_MAX = prereg.SINGLE_HYPOTHESIS_OOS_P_MAX
TRAIN_LEGACY_BONFERRONI_P_MAX = distill_eval.TRAIN_LEGACY_BONFERRONI_P_MAX
SleeveSpec = distill_eval.SleeveSpec


@dataclass(frozen=True)
class FrozenAuthorization:
    preregistration: dict[str, Any]
    source_clock_package: dict[str, Any]
    terminal_additive_novelty: dict[str, Any]
    sleeves: list[SleeveSpec]
    source_signed_episodes_by_split: dict[str, int]
    preliminary_train_receipt_support: Any = None
    overlap_disclosure: dict[str, Any] | None = None


canonical_hash = distill_eval.canonical_hash
sha256_file = distill_eval.sha256_file
_utc = distill_eval._utc
_iso_z = distill_eval._iso_z
simulate_portfolio = distill_eval.simulate_portfolio
cluster_signflip = distill_eval.cluster_signflip
evaluate_primary = distill_eval.evaluate_primary
load_sources = distill_eval.load_sources
validate_market = distill_eval.validate_market
validate_funding = distill_eval.validate_funding
load_portfolio_clock = distill_eval.load_portfolio_clock


def _load_json_object(path: str | Path) -> dict[str, Any]:
    return distill_eval._load_json_object(path)


def _verify_manifest(value: Mapping[str, Any], label: str) -> None:
    distill_eval._verify_manifest(value, label)


def _assert_hash_bound_file(record: Mapping[str, Any], label: str, count_rows: bool = True) -> None:
    distill_eval._assert_hash_bound_file(record, label, count_rows=count_rows)


def _validate_preregistration(path: Path = PREREGISTRATION) -> dict[str, Any]:
    report = _load_json_object(path)
    _verify_manifest(report, "compression preregistration")
    if report != prereg.build():
        raise RuntimeError(f"{POLICY_ID} preregistration artifact does not match build()")
    prereg.validate(report)
    return report


def _validate_terminal_overlap(novelty: Mapping[str, Any]) -> dict[str, Any]:
    disclosure = prereg._novelty_disclosure(novelty)
    if novelty.get("advance_to_economic_outcomes") is not False:
        raise RuntimeError(f"{POLICY_ID} additive novelty artifact unexpectedly authorized additive economics")
    return disclosure


def load_frozen_authorization(preregistration_path: Path = PREREGISTRATION, clock_package_path: Path = CLOCK_PACKAGE) -> FrozenAuthorization:
    registration = _validate_preregistration(preregistration_path)
    clock_package = _load_json_object(clock_package_path)
    _verify_manifest(clock_package, "source clock package")
    if clock_package.get("policy_id") != SOURCE_POLICY_ID or clock_package.get("decision") != "materialized_shadow_distilled_clock_package":
        raise RuntimeError(f"{POLICY_ID} source clock package identity drift")
    reuse = registration.get("source_clock_reuse", {})
    expected_source_prereg = reuse.get("source_preregistration", {})
    expected_clock_package = reuse.get("source_clock_package", {})
    source_prereg_path = Path(str(expected_source_prereg.get("path", "")))
    if (
        not source_prereg_path.is_file()
        or sha256_file(source_prereg_path) != expected_source_prereg.get("sha256")
        or clock_package.get("preregistration", {}).get("sha256") != expected_source_prereg.get("sha256")
        or clock_package.get("preregistration", {}).get("manifest_hash") != expected_source_prereg.get("manifest_hash")
    ):
        raise RuntimeError(f"{POLICY_ID} source preregistration receipt drift")
    if (
        str(clock_package_path) != expected_clock_package.get("path")
        or sha256_file(clock_package_path) != expected_clock_package.get("sha256")
        or clock_package.get("manifest_hash") != expected_clock_package.get("manifest_hash")
    ):
        raise RuntimeError(f"{POLICY_ID} source clock package receipt drift")
    builder = clock_package.get("implementation", {}).get("builder", {})
    _assert_hash_bound_file(builder, "source clock package builder", count_rows=False)

    sleeves: list[SleeveSpec] = []
    for base in clock_package.get("components", {}).get("base_order", []):
        record = clock_package.get("sleeves", {}).get(base)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"{POLICY_ID} missing source sleeve record: {base}")
        clock = record.get("clock", {})
        if not isinstance(clock, Mapping):
            raise RuntimeError(f"{POLICY_ID} missing source sleeve clock record: {base}")
        _assert_hash_bound_file(clock, f"source sleeve clock {base}")
        sleeves.append(SleeveSpec(name=str(record["sleeve_id"]), weight=float(record["weight"]), clock_path=Path(str(clock["path"])), clock_sha256=str(clock["sha256"])))
    distill_eval.validate_sleeves(sleeves)
    base_order = list(clock_package.get("components", {}).get("base_order", []))
    expected_weights = reuse.get("sleeve_weights", {})
    if base_order != list(prereg.distill.DISTILLED_BASES):
        raise RuntimeError(f"{POLICY_ID} source base order drift")
    for base, sleeve in zip(base_order, sleeves):
        expected_candidate = f"{base}__{prereg.distill.ACTIVE_VETO_OPERATOR}__{prereg.distill.DISTILLATION_VETO}"
        if expected_candidate not in expected_weights or sleeve.weight != float(expected_weights[expected_candidate]):
            raise RuntimeError(f"{POLICY_ID} source sleeve weight drift: {base}")
    for name, record in clock_package.get("portfolio_schedules", {}).items():
        if not isinstance(record, Mapping):
            raise RuntimeError(f"{POLICY_ID} portfolio schedule record drift: {name}")
        _assert_hash_bound_file(record, f"portfolio schedule {name}")

    novelty = _load_json_object(TERMINAL_NOVELTY)
    _verify_manifest(novelty, "terminal additive novelty")
    novelty_receipt = registration.get("terminal_additive_novelty_binding", {})
    if (
        str(TERMINAL_NOVELTY) != novelty_receipt.get("path")
        or sha256_file(TERMINAL_NOVELTY) != novelty_receipt.get("sha256")
        or novelty.get("manifest_hash") != novelty_receipt.get("manifest_hash")
    ):
        raise RuntimeError(f"{POLICY_ID} terminal additive novelty receipt drift")
    expected_source = {
        "path": str(clock_package_path),
        "sha256": sha256_file(clock_package_path),
        "manifest_hash": clock_package["manifest_hash"],
        "predecessor_mutated": False,
    }
    if novelty.get("policy_id") != SOURCE_POLICY_ID or novelty.get("source_package") != expected_source:
        raise RuntimeError(f"{POLICY_ID} terminal additive novelty source binding drift")
    if novelty.get("preregistration") != expected_source_prereg:
        raise RuntimeError(f"{POLICY_ID} terminal additive novelty preregistration binding drift")
    overlap = _validate_terminal_overlap(novelty)
    if registration.get("terminal_additive_novelty_binding", {}).get("near_6h_failures") != overlap["near_6h_failures"]:
        raise RuntimeError(f"{POLICY_ID} preregistered overlap disclosure drift")

    stats = clock_package.get("portfolio_source_stats", {}).get("splits", {})
    source_signed_episodes_by_split = {split: int(row.get("signed_episodes", 0)) for split, row in stats.items() if isinstance(row, Mapping)}
    return FrozenAuthorization(
        preregistration=registration,
        source_clock_package=clock_package,
        terminal_additive_novelty=novelty,
        sleeves=sleeves,
        source_signed_episodes_by_split=source_signed_episodes_by_split,
        preliminary_train_receipt_support=registration.get("preliminary_train_diagnostic_binding", {}).get("preliminary_train_receipt"),
        overlap_disclosure=overlap,
    )


def stage_checks(stage: str, primary: Mapping[str, Any], source_signed_episodes: int | None = None) -> dict[str, bool]:
    checks = distill_eval.stage_checks(stage, primary, source_signed_episodes)
    if stage != "train":
        checks["source_min_nonzero_signed_episodes"] = int(source_signed_episodes or 0) >= MIN_NONZERO_SIGNED_EPISODES[stage]
        checks["oos_cluster_signflip_p_max_0_1"] = float(primary["cluster_signflip"]["pvalue"]) <= OOS_CLUSTER_P_MAX
    return checks


def verify_predecessor(stage: str, outputs: Mapping[str, Path] = OUTPUTS) -> dict[str, Any] | None:
    if stage == "train":
        return None
    predecessor_stage = PREDECESSOR[stage]
    predecessor_path = Path(outputs[predecessor_stage])
    if not predecessor_path.is_file():
        raise RuntimeError(f"{POLICY_ID} missing predecessor {predecessor_stage}: {predecessor_path}")
    report = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise RuntimeError(f"{POLICY_ID} predecessor is not a JSON object")
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    if report.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError(f"{POLICY_ID} predecessor manifest hash drift")
    if report.get("policy_id") != POLICY_ID or report.get("stage") != predecessor_stage:
        raise RuntimeError(f"{POLICY_ID} predecessor identity drift")
    if report.get("passed") is not True or report.get("advance_to_next_stage") is not True:
        raise RuntimeError(f"{POLICY_ID} predecessor did not pass")
    return {"stage": predecessor_stage, "path": str(predecessor_path), "sha256": sha256_file(predecessor_path), "manifest_hash": report["manifest_hash"]}


def run(stage: str, output: str | Path | None = None, sleeves: Sequence[SleeveSpec] | None = None, outputs: Mapping[str, Path] = OUTPUTS) -> dict[str, Any]:
    if stage not in STAGES:
        raise RuntimeError(f"{POLICY_ID} unknown stage: {stage}")
    authorization = load_frozen_authorization()
    resolved_sleeves = list(sleeves) if sleeves is not None else list(authorization.sleeves)
    predecessor = verify_predecessor(stage, outputs)
    split, start_s, end_s = STAGES[stage]
    start = _utc(start_s); end = _utc(end_s)
    portfolio_clock = load_portfolio_clock(resolved_sleeves, split, start, end)
    market, funding, source = load_sources(stage, start, end)
    validate_market(market, start, end)
    validate_funding(funding, start, end)
    primary = evaluate_primary(portfolio_clock, market, funding, start, end)
    checks = stage_checks(stage, primary, authorization.source_signed_episodes_by_split.get(split))
    passed = all(checks.values())
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "source_policy_id": SOURCE_POLICY_ID,
        "stage": stage,
        "window": [_iso_z(start), _iso_z(end)],
        "predecessor": predecessor,
        "frozen_authorization": {
            "compression_preregistration": {"path": str(PREREGISTRATION), "sha256": sha256_file(PREREGISTRATION), "manifest_hash": authorization.preregistration["manifest_hash"]},
            "source_clock_package": {"path": str(CLOCK_PACKAGE), "sha256": sha256_file(CLOCK_PACKAGE), "manifest_hash": authorization.source_clock_package["manifest_hash"], "source_policy_id": SOURCE_POLICY_ID},
            "terminal_additive_novelty": {"path": str(TERMINAL_NOVELTY), "sha256": sha256_file(TERMINAL_NOVELTY), "manifest_hash": authorization.terminal_additive_novelty["manifest_hash"], "decision": authorization.terminal_additive_novelty.get("decision")},
            "overlap_disclosure": authorization.overlap_disclosure,
            "preliminary_train_receipt_support": authorization.preliminary_train_receipt_support,
        },
        "source": source,
        "accounting": {
            "ledger": "cash plus fixed sleeve quantities; aggregate q delta netted for execution cost",
            "entry_quantity": "q=side*weight*pre_transition_portfolio_equity/open for simultaneous entries",
            "transition_order": "mark at open, remove exits, add simultaneous entries from same pre-transition equity, charge abs(net_delta_q)*open*cost",
            "funding": "post-transition aggregate q receives funding cash=-aggregate_q*settlement_mark*rate for entry<=funding<exit",
            "strict_mdd": "global HWM; every held 5m favorable then adverse OHLC on aggregate net q with virtual adverse liquidation cost",
            "final_exit": "mandatory liquidation at stage end open",
        },
        "costs": {"base_each_notional_side_bp": 6, "stress_each_notional_side_bp": 10},
        "fixed_sleeves": [{"name": s.name, "weight": s.weight, "clock_path": str(s.clock_path) if s.clock_path else None, "clock_sha256": s.clock_sha256} for s in resolved_sleeves],
        "clock_package_source_signed_episodes": authorization.source_signed_episodes_by_split,
        "physical_rows_opened": {"market": len(market), "funding": len(funding), "portfolio_clock": len(portfolio_clock)},
        "later_stage_outcomes_opened": False,
        "primary": primary,
        "checks": checks,
        "train_legacy_cluster_diagnostic": {
            "reported_not_pass_authorizing": stage == "train",
            "legacy_p_max_0_1_over_72": TRAIN_LEGACY_BONFERRONI_P_MAX,
            "observed_pvalue": primary["cluster_signflip"]["pvalue"],
            "would_pass_legacy_gate": primary["cluster_signflip"]["pvalue"] <= TRAIN_LEGACY_BONFERRONI_P_MAX,
        } if stage == "train" else None,
        "passed": passed,
        "status": "post_selection_train_shape_shadow" if stage == "train" and passed else ("oos_pass" if passed else "terminal_reject_no_repair"),
        "formal_legacy_train_pass": False if stage == "train" else None,
        "advance_to_next_stage": passed and stage != "final",
        "decision": "post_selection_train_shape_shadow" if stage == "train" and passed else ("pass" if passed else "terminal_reject_no_repair"),
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    destination = Path(output) if output is not None else Path(outputs[stage])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        load_frozen_authorization()
        predecessor = verify_predecessor(args.stage)
        print(json.dumps({"stage": args.stage, "verified": True, "predecessor": predecessor, "outcomes_opened": False}, ensure_ascii=False))
        return
    result = run(args.stage, args.output)
    print(json.dumps({"stage": args.stage, "passed": result["passed"], "output": str(args.output or OUTPUTS[args.stage])}, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
