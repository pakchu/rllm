"""Freeze the G9 overlap net-position frequency-waived revalidation family.

This is a diagnostic successor to G9-OVERLAP-NET-PORT-1.  It preserves all 64
frozen train-finalist weights and adds 14 outcome-blind constituent standalones
at one newly frozen canonical weight.  The previously terminal 2024 stop is a
user-authorized diagnostic barrier waiver so the 78-candidate family can be
measured on test2024 and eval2025.  Final2026 remains unopened.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import build_gross9_overlap_net_position_config as net_config
from training import evaluate_gross9_overlap_net_position_eval2025_override as normalized_eval_source
from training import evaluate_gross9_overlap_net_position_portfolio as original_evaluator
from training import optimize_gross9_overlap_portfolio as optimizer
from training import preregister_gross9_overlap_net_position_validation as original_freeze

POLICY_ID = "G9-OVERLAP-NET-PORT-1-FREQ-WAIVED-REVALIDATION"
SOURCE_POLICY_ID = net_config.POLICY_ID
PROTOCOL_VERSION = "gross9_frequency_waived_revalidation_freeze_v2"
AS_OF_DATE = "2026-09-04"
SELECTION = Path("results/gross9_overlap_portfolio_train_selection_2026-09-03.json")
CONFIG = net_config.CONFIG_OUTPUT
UNIVERSE = net_config.UNIVERSE
ORIGINAL_FREEZE = original_freeze.DEFAULT_OUTPUT
HOLDOUT = Path("results/gross9_overlap_net_position_holdout_dec2023_2026-09-03.json")
TEST2024 = Path("results/gross9_overlap_net_position_test2024_2026-09-03.json")
CURRENT_RANK1_EVAL2025 = Path(
    "results/gross9_overlap_net_position_eval2025_diagnostic_v2_2026-09-04.json"
)
EVALUATOR = Path("training/evaluate_gross9_frequency_waived_revalidation.py")
FREEZER = Path("training/preregister_gross9_frequency_waived_revalidation.py")
DEFAULT_OUTPUT = Path("results/gross9_frequency_waived_revalidation_freeze_v2_2026-09-04.json")
V1_FREEZE = Path("results/gross9_frequency_waived_revalidation_freeze_2026-09-04.json")
V1_PREFLIGHT_FAILURE = Path(
    "results/gross9_frequency_waived_revalidation_preflight_failure_2026-09-04.json"
)
STAGES: dict[str, tuple[str, str, str]] = {
    "test2024": ("test", "2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
    "eval2025": ("eval", "2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
}
MAX_T_DRAWS = 100_000
MAX_T_SEED = 20260904
MAX_T_P_MAX = 0.10
STANDALONE_WEIGHT = 0.25
EXPECTED_EXACT_FINALISTS = 64
EXPECTED_STANDALONES = 14
EXPECTED_CANDIDATES = EXPECTED_EXACT_FINALISTS + EXPECTED_STANDALONES


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode()
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hashed_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{POLICY_ID} JSON object required: {path}")
    hash_key = "protocol_hash" if "protocol_hash" in value else "manifest_hash"
    core = {key: item for key, item in value.items() if key != hash_key}
    if value.get(hash_key) != canonical_hash(core):
        raise RuntimeError(f"{POLICY_ID} artifact hash drift: {path}")
    return value


def receipt(path: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
    hash_key = "protocol_hash" if "protocol_hash" in value else "manifest_hash"
    return {"path": str(path), "sha256": sha256_file(path), hash_key: value[hash_key]}


def count_gzip_csv_rows(path: str | Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _candidate_id(prefix: str, index: int, weights: Mapping[str, Any]) -> str:
    digest = canonical_hash({key: float(value) for key, value in sorted(weights.items())})[:12]
    return f"{prefix}-{index:03d}-{digest}"


def build_candidate_family(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    exact = selection.get("exact_finalists")
    if not isinstance(exact, list) or len(exact) != EXPECTED_EXACT_FINALISTS:
        raise RuntimeError(f"{POLICY_ID} expected exactly {EXPECTED_EXACT_FINALISTS} exact finalists")
    family: list[dict[str, Any]] = []
    seen_weights: set[str] = set()
    constituent_ids: set[str] = set()
    waived_train_failures = {"turnover_cap", "sleeve_turnover_share_cap"}
    for index, row in enumerate(exact, start=1):
        weights = row.get("sleeve_weights", {}) if isinstance(row, Mapping) else {}
        if not isinstance(weights, Mapping) or not weights:
            raise RuntimeError(f"{POLICY_ID} exact finalist missing weights")
        normalized = {str(key): float(value) for key, value in weights.items()}
        failed_gates = {
            str(name)
            for name, passed in row.get("gates", {}).items()
            if not bool(passed)
        }
        if not failed_gates or not failed_gates.issubset(waived_train_failures):
            raise RuntimeError(
                f"{POLICY_ID} exact finalist is not a frequency-waiver-only train reject"
            )
        signature = canonical_hash(normalized)
        if signature in seen_weights:
            raise RuntimeError(f"{POLICY_ID} duplicate exact-finalist weight vector")
        seen_weights.add(signature)
        constituent_ids.update(normalized)
        family.append(
            {
                "candidate_id": _candidate_id("EXACT64", index, normalized),
                "kind": "frozen_exact_finalist",
                "source_exact_finalist_proxy_rank": int(row.get("proxy_rank", index)),
                "weights": normalized,
                "weight_sum": round(sum(abs(v) for v in normalized.values()), 12),
                "preexisting_frozen_candidate": True,
                "derived_constituent_candidate": False,
                "weights_changed": False,
            }
        )
    if len(constituent_ids) != EXPECTED_STANDALONES:
        raise RuntimeError(f"{POLICY_ID} expected {EXPECTED_STANDALONES} unique constituent sleeves")
    for index, sleeve_id in enumerate(sorted(constituent_ids), start=1):
        weights = {sleeve_id: STANDALONE_WEIGHT}
        family.append(
            {
                "candidate_id": _candidate_id("STANDALONE025", index, weights),
                "kind": "constituent_standalone_weight_0_25",
                "source_sleeve_id": sleeve_id,
                "weights": weights,
                "weight_sum": STANDALONE_WEIGHT,
                "preexisting_frozen_candidate": False,
                "derived_constituent_candidate": True,
                "new_fixed_weight_assignment": STANDALONE_WEIGHT,
            }
        )
    ids = [row["candidate_id"] for row in family]
    if len(family) != EXPECTED_CANDIDATES or len(set(ids)) != EXPECTED_CANDIDATES:
        raise RuntimeError(f"{POLICY_ID} candidate family cardinality drift")
    return family


def selected_clock_receipts(universe: Mapping[str, Any], candidate_family: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    universe_records = {row["sleeve_id"]: row for row in universe.get("sleeves", [])}
    sleeve_ids = sorted({sleeve for candidate in candidate_family for sleeve in candidate["weights"]})
    out: list[dict[str, Any]] = []
    for sleeve_id in sleeve_ids:
        record = universe_records.get(sleeve_id)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"{POLICY_ID} candidate sleeve absent from universe: {sleeve_id}")
        clock = record.get("clock", {})
        path = Path(str(clock.get("path", "")))
        if not path.is_file() or sha256_file(path) != clock.get("sha256") or count_gzip_csv_rows(path) != int(clock.get("rows", -1)):
            raise RuntimeError(f"{POLICY_ID} clock receipt drift: {sleeve_id}")
        out.append(
            {
                "sleeve_id": sleeve_id,
                "path": str(path),
                "sha256": clock["sha256"],
                "rows": int(clock["rows"]),
                "split_counts": record.get("split_counts"),
            }
        )
    return out


def validate_original_terminal_chain(freeze: Mapping[str, Any], holdout: Mapping[str, Any], test2024: Mapping[str, Any]) -> None:
    freeze_receipt = receipt(ORIGINAL_FREEZE, freeze)
    implementation = freeze.get("implementation")
    if holdout.get("policy_id") != SOURCE_POLICY_ID or holdout.get("stage") != "holdout_dec2023" or holdout.get("passed") is not True or holdout.get("advance_to_next_stage") is not True:
        raise RuntimeError(f"{POLICY_ID} holdout artifact does not authorize test2024")
    if holdout.get("validation_freeze") != freeze_receipt or holdout.get("implementation") != implementation or holdout.get("predecessor") is not None:
        raise RuntimeError(f"{POLICY_ID} holdout chain binding drift")
    expected_holdout_receipt = {"stage": "holdout_dec2023", **receipt(HOLDOUT, holdout)}
    if test2024.get("policy_id") != SOURCE_POLICY_ID or test2024.get("stage") != "test2024" or test2024.get("passed") is not False or test2024.get("advance_to_next_stage") is not False:
        raise RuntimeError(f"{POLICY_ID} test2024 is not the terminal stop being overridden")
    if test2024.get("validation_freeze") != freeze_receipt or test2024.get("implementation") != implementation or test2024.get("predecessor") != expected_holdout_receipt:
        raise RuntimeError(f"{POLICY_ID} test2024 terminal chain binding drift")


def build() -> dict[str, Any]:
    selection = load_hashed_json(SELECTION)
    config = load_hashed_json(CONFIG)
    universe = load_hashed_json(UNIVERSE)
    freeze = original_evaluator.load_validation_freeze(ORIGINAL_FREEZE)
    holdout = load_hashed_json(HOLDOUT)
    test2024 = load_hashed_json(TEST2024)
    current_rank1_eval2025 = load_hashed_json(CURRENT_RANK1_EVAL2025)
    current_rank1_eval_freeze = normalized_eval_source.load_freeze(
        normalized_eval_source.FREEZE
    )
    v1_freeze = load_hashed_json(V1_FREEZE)
    v1_preflight_failure = load_hashed_json(V1_PREFLIGHT_FAILURE)
    if (
        v1_preflight_failure.get("attempt_freeze") != receipt(V1_FREEZE, v1_freeze)
        or v1_preflight_failure.get("market_or_funding_rows_opened") != 0
        or v1_preflight_failure.get("candidate_economic_metrics_computed") != 0
        or v1_preflight_failure.get("candidate_weight_vectors_changed") is not False
        or v1_preflight_failure.get("disposition")
        != "pre_outcome_infrastructure_failure_successor_allowed"
    ):
        raise RuntimeError(f"{POLICY_ID} V1 preflight failure receipt drift")
    if selection.get("policy_id") != optimizer.POLICY_ID or config.get("policy_id") != SOURCE_POLICY_ID:
        raise RuntimeError(f"{POLICY_ID} source policy identity drift")
    if config.get("sleeve_weights") != selection.get("authoritative_rank1", {}).get("sleeve_weights"):
        raise RuntimeError(f"{POLICY_ID} train-selected config weight drift")
    validate_original_terminal_chain(freeze, holdout, test2024)
    if (
        current_rank1_eval2025.get("policy_id") != normalized_eval_source.POLICY_ID
        or current_rank1_eval2025.get("protocol_version")
        != normalized_eval_source.PROTOCOL_VERSION
        or current_rank1_eval2025.get("stage") != "eval2025_diagnostic"
        or current_rank1_eval2025.get("fixed_portfolio", {}).get("sleeve_weights")
        != config.get("sleeve_weights")
        or current_rank1_eval2025.get("override_freeze")
        != {
            "path": str(normalized_eval_source.FREEZE),
            "sha256": sha256_file(normalized_eval_source.FREEZE),
            "manifest_hash": current_rank1_eval_freeze["manifest_hash"],
        }
        or current_rank1_eval2025.get("original_chain")
        != current_rank1_eval_freeze["original_chain"]
        or current_rank1_eval2025.get("decision")
        != "diagnostic_only_no_further_advance"
        or current_rank1_eval2025.get("advance_beyond_eval2025") is not False
        or current_rank1_eval2025.get("original_protocol_advance") is not False
        or current_rank1_eval2025.get("original_test2024_relabelled_as_pass") is not False
        or current_rank1_eval2025.get("final2026_outcomes_opened") is not False
        or current_rank1_eval2025.get("live_capital_authorized") is not False
        or current_rank1_eval2025.get("order_submission_enabled") is not False
    ):
        raise RuntimeError(f"{POLICY_ID} current rank1 eval diagnostic receipt drift")
    family = build_candidate_family(selection)
    known_rank1_candidate = next(
        row
        for row in family
        if row["weights"] == config["sleeve_weights"]
    )
    clocks = selected_clock_receipts(universe, family)
    retained_checks = [
        "absolute_return_positive",
        "cagr_to_strict_mdd_min",
        "strict_mdd_max",
        "mean_exposure_weighted_gross_edge_min",
        "stress_absolute_return_positive",
        "stress_cagr_to_strict_mdd_min",
        "each_calendar_half_positive",
        "max_t_aligned_week_p_max_0_10",
        "mean_abs_net_position_cap",
        "max_abs_net_position_cap",
        "minimum_intervals",
        "minimum_active_iso_weeks",
        "minimum_aggregate_net_signed_episodes",
    ]
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "source_policy_id": SOURCE_POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "objective": "diagnostically revalidate 64 unchanged frozen finalists plus 14 outcome-blind derived constituent standalones on test2024 and eval2025",
        "explicit_user_override": {
            "override_scope": "test2024 terminal stop may be bypassed for diagnostic eval2025 measurement only",
            "permits_test2024_and_eval2025_for_every_candidate_regardless_stage_failure": True,
            "final2026_open_authorized": False,
            "repair_or_weight_change_authorized": False,
            "diagnostic_post_outcome_ordering_authorized": True,
            "selection_or_promotion_from_ordering_authorized": False,
        },
        "frozen_inputs": {
            "config": receipt(CONFIG, config),
            "train_selection": receipt(SELECTION, selection),
            "universe": receipt(UNIVERSE, universe),
            "original_validation_freeze": receipt(ORIGINAL_FREEZE, freeze),
            "holdout_dec2023_artifact": receipt(HOLDOUT, holdout),
            "test2024_terminal_artifact": receipt(TEST2024, test2024),
            "current_rank1_eval2025_diagnostic": receipt(
                CURRENT_RANK1_EVAL2025,
                current_rank1_eval2025,
            ),
            "current_rank1_eval2025_override_freeze": receipt(
                normalized_eval_source.FREEZE,
                current_rank1_eval_freeze,
            ),
            "selected_clocks": clocks,
        },
        "known_outcome_boundary": {
            "current_rank1_test2024_and_eval2025_known_before_family_freeze": True,
            "known_current_rank1_candidate_id": known_rank1_candidate["candidate_id"],
            "other_77_candidate_test2024_outcomes_known_before_family_freeze": False,
            "other_77_candidate_eval2025_outcomes_known_before_family_freeze": False,
            "classification": "retrospective diagnostic revalidation, not clean model selection",
            "prospective_fwer_claim": False,
            "final2026_remains_unopened_for_every_candidate": True,
        },
        "v1_preflight_failure": {
            "freeze": receipt(V1_FREEZE, v1_freeze),
            "failure": receipt(V1_PREFLIGHT_FAILURE, v1_preflight_failure),
            "market_or_funding_rows_opened": 0,
            "candidate_economic_metrics_computed": 0,
            "repair": "derived weight_sum rounded to 12 decimal places",
        },
        "candidate_family": family,
        "candidate_counts": {
            "frozen_exact_finalists": EXPECTED_EXACT_FINALISTS,
            "constituent_standalone_weight_0_25": EXPECTED_STANDALONES,
            "total": EXPECTED_CANDIDATES,
        },
        "stages": {stage: {"split": split, "start": start, "end": end} for stage, (split, start, end) in STAGES.items()},
        "gate_policy": {
            "waived_rejection_gates": ["turnover_cap", "sleeve_turnover_share_cap", "max_trade_frequency"],
            "waived_metrics_remain_disclosures": True,
            "retained_checks": retained_checks,
            "qualifier_rule": "candidate must pass all retained checks in both test2024 and eval2025",
            "minimum_aggregate_net_signed_episodes": 12,
            "minimum_active_iso_weeks": 4,
            "max_t": {"method": "shared aligned UTC-week sign-flip max-T", "candidate_count": EXPECTED_CANDIDATES, "draws": MAX_T_DRAWS, "seed": MAX_T_SEED, "p_max": MAX_T_P_MAX, "prospective_fwer_claim": False},
        },
        "ranking_rule": [
            "qualifiers first, then nonqualifiers for disclosure",
            "descending min(test2024_cagr_to_strict_mdd, eval2025_cagr_to_strict_mdd)",
            "descending summed log final-equity return across test2024 and eval2025",
            "ascending worst strict_mdd_pct across test2024 and eval2025",
            "ascending candidate_id",
        ],
        "implementation": {
            "freezer": {"path": str(FREEZER), "sha256": sha256_file(FREEZER)},
            "evaluator": {"path": str(EVALUATOR), "sha256": sha256_file(EVALUATOR) if EVALUATOR.is_file() else None},
            "fixed_ledger": {"path": str(Path(original_evaluator.fixed_ledger.__file__).relative_to(Path.cwd())), "sha256": sha256_file(original_evaluator.fixed_ledger.__file__)},
            "source_loader": {"path": str(Path(original_evaluator.__file__).relative_to(Path.cwd())), "sha256": sha256_file(original_evaluator.__file__)},
            "normalized_eval_source": {
                "path": str(Path(normalized_eval_source.__file__).relative_to(Path.cwd())),
                "sha256": sha256_file(normalized_eval_source.__file__),
            },
            "net_risk_metrics": {"path": str(Path(net_config.__file__).relative_to(Path.cwd())), "sha256": sha256_file(net_config.__file__)},
        },
        "evidence_boundary": {"market_or_funding_rows_opened_by_freeze": 0, "test2024_reopened_by_freeze": False, "eval2025_opened_by_freeze": False, "final2026_opened": False},
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: Mapping[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError(f"{POLICY_ID} manifest drift")
    if value.get("policy_id") != POLICY_ID or value.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"{POLICY_ID} identity drift")
    override = value.get("explicit_user_override", {})
    if override.get("permits_test2024_and_eval2025_for_every_candidate_regardless_stage_failure") is not True or override.get("final2026_open_authorized") is not False or override.get("repair_or_weight_change_authorized") is not False or override.get("selection_or_promotion_from_ordering_authorized") is not False:
        raise RuntimeError(f"{POLICY_ID} override boundary drift")
    counts = value.get("candidate_counts", {})
    if counts != {"frozen_exact_finalists": EXPECTED_EXACT_FINALISTS, "constituent_standalone_weight_0_25": EXPECTED_STANDALONES, "total": EXPECTED_CANDIDATES}:
        raise RuntimeError(f"{POLICY_ID} candidate count drift")
    family = value.get("candidate_family")
    if not isinstance(family, list) or len(family) != EXPECTED_CANDIDATES or len({row.get("candidate_id") for row in family if isinstance(row, Mapping)}) != EXPECTED_CANDIDATES:
        raise RuntimeError(f"{POLICY_ID} candidate family drift")
    for row in family:
        if not isinstance(row, Mapping) or not isinstance(row.get("weights"), Mapping) or not row["weights"]:
            raise RuntimeError(f"{POLICY_ID} candidate family malformed")
        if row.get("kind") == "frozen_exact_finalist":
            if row.get("preexisting_frozen_candidate") is not True or row.get("weights_changed") is not False:
                raise RuntimeError(f"{POLICY_ID} frozen finalist provenance drift")
        elif row.get("kind") == "constituent_standalone_weight_0_25":
            if row.get("derived_constituent_candidate") is not True or row.get("new_fixed_weight_assignment") != STANDALONE_WEIGHT:
                raise RuntimeError(f"{POLICY_ID} derived standalone provenance drift")
        else:
            raise RuntimeError(f"{POLICY_ID} candidate kind drift")
    gates = value.get("gate_policy", {})
    if gates.get("waived_rejection_gates") != ["turnover_cap", "sleeve_turnover_share_cap", "max_trade_frequency"] or gates.get("max_t", {}).get("draws") != MAX_T_DRAWS or gates.get("max_t", {}).get("seed") != MAX_T_SEED:
        raise RuntimeError(f"{POLICY_ID} gate policy drift")
    if set(value.get("stages", {})) != set(STAGES) or "final2026" in value.get("stages", {}):
        raise RuntimeError(f"{POLICY_ID} stage boundary drift")
    boundary = value.get("evidence_boundary", {})
    if boundary.get("eval2025_opened_by_freeze") is not False or boundary.get("final2026_opened") is not False:
        raise RuntimeError(f"{POLICY_ID} freeze opened forbidden outcomes")
    if dict(value) != build():
        raise RuntimeError(f"{POLICY_ID} freeze no longer matches exact bound inputs/code")


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    value = build()
    validate(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    value = run(args.output)
    print(json.dumps({"policy_id": POLICY_ID, "output": str(args.output), "manifest_hash": value["manifest_hash"], "outcomes_opened": False}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
