"""Preregister G9QTR-COMPRESS-8 as a standalone/replacement candidate.

This protocol deliberately reuses the frozen G9QTR-DISTILL-8 clock package and
four fixed sleeves, but changes the objective from Gross9-additive alpha to a
standalone/replacement/compression validation.  The terminal Gross9 additive
novelty reject remains bound as a disclosure: only the inherited near-6h entry
share gate is non-authorizing for this objective; exact-entry Jaccard,
occupied-bar Jaccard, and absolute signed-exposure Pearson must still pass for
every Gross9 comparator before any economics stage can run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import preregister_gross9_qtr_distill as distill

POLICY_ID = "G9QTR-COMPRESS-8"
SOURCE_POLICY_ID = distill.POLICY_ID
PROTOCOL_VERSION = "gross9_qtr_compression_replacement_preregistration_v1"
AS_OF_DATE = distill.AS_OF_DATE
DEFAULT_OUTPUT = Path("results/gross9_qtr_compression_shadow_preregistration_2026-09-02.json")
SOURCE_PREREGISTRATION = distill.DEFAULT_OUTPUT
SOURCE_CLOCK_PACKAGE = Path("results/gross9_qtr_distill_split_clock_source_support_2026-09-02.json")
TERMINAL_ADDITIVE_NOVELTY = Path("results/gross9_qtr_distill_train_gross9_novelty_2026-09-02.json")
OUTPUTS = {
    stage: f"results/gross9_qtr_compression_{stage}_economics_2026-09-02.json"
    for stage in ("train", "test", "eval", "final")
}
SINGLE_HYPOTHESIS_OOS_P_MAX = 0.10
MIN_NONZERO_SIGNED_EPISODES = {"test": 12, "eval": 12, "final": 8}
SOURCE_DISTILL_PREREG_RECEIPT = {
    "path": str(SOURCE_PREREGISTRATION),
    "sha256": "2cc6ce9aad4d46be0f6baf82afc2e4fd98f2c67e44b150f26021a6a26cde38b9",
    "manifest_hash": "1a112ca2a43014e852e4985b17636d3e08b2c3ab94b17cd2e37e0b77b7317d7a",
}
SOURCE_CLOCK_PACKAGE_RECEIPT = {
    "path": str(SOURCE_CLOCK_PACKAGE),
    "sha256": "5b9ea50790d37f6daa2ccb7eaa351022b40088c503be8c0d8f39111c740270c7",
    "manifest_hash": "734be40990a92c03e6bd0f70e0f0e2fbf177545bf936bb2eb6b56820f0045e70",
}
TERMINAL_ADDITIVE_NOVELTY_RECEIPT = {
    "path": str(TERMINAL_ADDITIVE_NOVELTY),
    "sha256": "e8781c156c3dfff29478f59ac7eb5d825387e256c7878ac53208eb7bee5ce5b4",
    "manifest_hash": "f66dc7ea1af3b1dd16dedacd084b96785d598a0962a1bd5e8e0d3bd322a307c4",
}
EXPECTED_NEAR_6H_FAILURES = {
    "cand_rex_veto_7": 0.4444444444444444,
    "markov_transition_long": 0.4166666666666667,
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{POLICY_ID} expected JSON object: {path}")
    return value


def _manifest(value: Mapping[str, Any]) -> str | None:
    return value.get("manifest_hash") if isinstance(value, Mapping) else None


def _novelty_disclosure(novelty: Mapping[str, Any]) -> dict[str, Any]:
    sleeves = novelty.get("gross9_sleeves", {})
    if not isinstance(sleeves, Mapping) or not sleeves:
        raise RuntimeError(f"{POLICY_ID} missing terminal additive novelty sleeves")
    required_pass_checks = ("exact_entry_jaccard", "occupied_5m_bar_jaccard", "absolute_signed_exposure_pearson")
    near_failures: dict[str, Any] = {}
    for sleeve, row in sleeves.items():
        checks = row.get("checks", {}) if isinstance(row, Mapping) else {}
        if not all(checks.get(name) is True for name in required_pass_checks):
            raise RuntimeError(f"{POLICY_ID} non-near6h Gross9 overlap gate failed for {sleeve}")
        failing = {name for name, passed in checks.items() if passed is not True}
        if failing - {"one_to_one_6h_max_matched_share"}:
            raise RuntimeError(f"{POLICY_ID} unexpected Gross9 overlap failure for {sleeve}: {sorted(failing)}")
        if "one_to_one_6h_max_matched_share" in failing:
            near_failures[str(sleeve)] = row.get("metrics", {}).get("one_to_one_6h_max_matched_share")
    if not near_failures:
        raise RuntimeError(f"{POLICY_ID} expected terminal additive novelty reject to be near-6h only")
    if novelty.get("decision") != "terminal_gross9_novelty_reject" or novelty.get("gross9_pass") is not False:
        raise RuntimeError(f"{POLICY_ID} terminal additive novelty reject binding drift")
    return {
        "terminal_additive_decision": novelty.get("decision"),
        "terminal_additive_gross9_pass": novelty.get("gross9_pass"),
        "required_pass_checks_for_replacement": list(required_pass_checks),
        "near_6h_overlap_is_disclosure_not_authorization_gate": True,
        "near_6h_failures": near_failures,
        "all_exact_entry_occupied_and_abs_pearson_passed": True,
    }


def build() -> dict[str, Any]:
    source_prereg = _load_json_object(SOURCE_PREREGISTRATION)
    source_package = _load_json_object(SOURCE_CLOCK_PACKAGE)
    terminal_novelty = _load_json_object(TERMINAL_ADDITIVE_NOVELTY)
    for value, receipt, label in (
        (source_prereg, SOURCE_DISTILL_PREREG_RECEIPT, "source distill preregistration"),
        (source_package, SOURCE_CLOCK_PACKAGE_RECEIPT, "source clock package"),
        (terminal_novelty, TERMINAL_ADDITIVE_NOVELTY_RECEIPT, "terminal additive novelty"),
    ):
        if sha256_file(receipt["path"]) != receipt["sha256"] or _manifest(value) != receipt["manifest_hash"]:
            raise RuntimeError(f"{POLICY_ID} immutable {label} receipt drift")
    disclosure = _novelty_disclosure(terminal_novelty)
    if disclosure["near_6h_failures"] != EXPECTED_NEAR_6H_FAILURES:
        raise RuntimeError(f"{POLICY_ID} immutable near-6h disclosure drift")
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "source_policy_id": SOURCE_POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "objective": "validate the frozen G9QTR-DISTILL-8 four-sleeve clock package as a standalone/replacement/compression policy, not as Gross9-additive alpha",
        "research_status": "one frozen post-selection hypothesis; sequential no-repair OOS required",
        "hypothesis_count": 1,
        "source_clock_reuse": {
            "source_policy_id": SOURCE_POLICY_ID,
            "identical_sleeves_weights_and_rule": True,
            "source_preregistration": copy.deepcopy(SOURCE_DISTILL_PREREG_RECEIPT),
            "source_clock_package": copy.deepcopy(SOURCE_CLOCK_PACKAGE_RECEIPT),
            "component_sleeves": list(distill.DISTILLED_SLEEVES),
            "sleeve_weights": copy.deepcopy(distill.SLEEVE_WEIGHTS),
            "gross_exposure_sum": sum(distill.SLEEVE_WEIGHTS.values()),
            "per_sleeve_rule": "exact active-veto strict-lower6h candidate schedule with 8 elapsed hour hold",
        },
        "terminal_additive_novelty_binding": {
            **copy.deepcopy(TERMINAL_ADDITIVE_NOVELTY_RECEIPT),
            **disclosure,
            "additive_gross9_alpha_authorized": False,
            "standalone_replacement_compression_authorized_to_test": True,
        },
        "preliminary_train_diagnostic_binding": {
            "legacy_active_veto_familywise_p_non_authorizing": True,
            "preliminary_train_receipt": copy.deepcopy(distill.PRELIMINARY_SEQUENCING_RECEIPT),
            "distilled_train_diagnostics": copy.deepcopy(distill.DISTILLED_TRAIN_DIAGNOSTICS),
            "train_diagnostic_must_be_rerun_under_compression_policy_id": True,
        },
        "stages": {
            "train_diagnostic": list(distill.TRAIN_WINDOW),
            "test2024": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval2025": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final2026": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "oos_gate_rule": {
            "sequence": ["test2024", "eval2025", "final2026"],
            "single_hypothesis_weekly_signflip_one_sided_p_max": SINGLE_HYPOTHESIS_OOS_P_MAX,
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "mean_gross_underlying_min_bp": 20.0,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "source_min_nonzero_signed_episodes": copy.deepcopy(MIN_NONZERO_SIGNED_EPISODES),
            "gross9_additive_near_6h_gate_removed_by_user_objective_change": True,
            "repair_authorized_after_failure": False,
            "stop_on_first_failure": True,
        },
        "outputs": copy.deepcopy(OUTPUTS),
        "implementation": {
            "preregister": {
                "path": "training/preregister_gross9_qtr_compression.py",
                "sha256": sha256_file(__file__),
            },
            "economics_evaluator": {
                "path": "training/evaluate_gross9_qtr_compression_economics.py",
                "sha256": sha256_file("training/evaluate_gross9_qtr_compression_economics.py"),
            },
        },
        "evidence_boundary": {
            "oos_outcomes_opened_by_this_preregistration": False,
            "oos_schedule_rows_opened_by_this_preregistration": False,
            "market_rows_opened": False,
            "funding_opened": False,
            "returns_or_pnl_opened": False,
        },
    }
    core["manifest_hash"] = canonical_hash(core)
    return core


def validate(value: Mapping[str, Any]) -> None:
    core = dict(value)
    manifest = core.pop("manifest_hash", None)
    if manifest != canonical_hash(core):
        raise RuntimeError(f"{POLICY_ID} preregistration manifest drift")
    if value.get("protocol_version") != PROTOCOL_VERSION or value.get("policy_id") != POLICY_ID:
        raise RuntimeError(f"{POLICY_ID} identity drift")
    reuse = value.get("source_clock_reuse", {})
    if reuse.get("source_policy_id") != SOURCE_POLICY_ID or reuse.get("identical_sleeves_weights_and_rule") is not True:
        raise RuntimeError(f"{POLICY_ID} source clock reuse drift")
    if tuple(reuse.get("component_sleeves", ())) != distill.DISTILLED_SLEEVES or reuse.get("sleeve_weights") != distill.SLEEVE_WEIGHTS:
        raise RuntimeError(f"{POLICY_ID} sleeve/weight drift")
    if not math.isclose(float(reuse.get("gross_exposure_sum", -1)), 0.5):
        raise RuntimeError(f"{POLICY_ID} gross exposure drift")
    if reuse.get("source_preregistration") != SOURCE_DISTILL_PREREG_RECEIPT or reuse.get("source_clock_package") != SOURCE_CLOCK_PACKAGE_RECEIPT:
        raise RuntimeError(f"{POLICY_ID} immutable source receipt drift")
    novelty = value.get("terminal_additive_novelty_binding", {})
    if (
        novelty.get("terminal_additive_decision") != "terminal_gross9_novelty_reject"
        or novelty.get("terminal_additive_gross9_pass") is not False
        or novelty.get("near_6h_overlap_is_disclosure_not_authorization_gate") is not True
        or novelty.get("all_exact_entry_occupied_and_abs_pearson_passed") is not True
        or novelty.get("additive_gross9_alpha_authorized") is not False
        or novelty.get("standalone_replacement_compression_authorized_to_test") is not True
        or novelty.get("near_6h_failures") != EXPECTED_NEAR_6H_FAILURES
    ):
        raise RuntimeError(f"{POLICY_ID} replacement overlap disclosure drift")
    for key, expected in TERMINAL_ADDITIVE_NOVELTY_RECEIPT.items():
        if novelty.get(key) != expected:
            raise RuntimeError(f"{POLICY_ID} immutable terminal novelty receipt drift")
    prelim = value.get("preliminary_train_diagnostic_binding", {})
    if prelim.get("legacy_active_veto_familywise_p_non_authorizing") is not True or prelim.get("preliminary_train_receipt") != distill.PRELIMINARY_SEQUENCING_RECEIPT:
        raise RuntimeError(f"{POLICY_ID} preliminary train diagnostic binding drift")
    gates = value.get("oos_gate_rule", {})
    if gates.get("sequence") != ["test2024", "eval2025", "final2026"] or gates.get("repair_authorized_after_failure") is not False:
        raise RuntimeError(f"{POLICY_ID} OOS sequence/no-repair drift")
    if gates.get("source_min_nonzero_signed_episodes") != MIN_NONZERO_SIGNED_EPISODES:
        raise RuntimeError(f"{POLICY_ID} signed episode gate drift")
    if value.get("hypothesis_count") != 1:
        raise RuntimeError(f"{POLICY_ID} hypothesis count drift")
    for binding in value.get("implementation", {}).values():
        if not isinstance(binding, Mapping) or sha256_file(str(binding.get("path", ""))) != binding.get("sha256"):
            raise RuntimeError(f"{POLICY_ID} implementation binding drift")
    if value.get("evidence_boundary", {}).get("oos_outcomes_opened_by_this_preregistration") is not False:
        raise RuntimeError(f"{POLICY_ID} OOS boundary drift")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
