"""Evaluate the three source-supported HVSOF-8 candidates against Gross9."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import (
    evaluate_high_volatility_multi_condition_pairwise_concordance_gross9_novelty
    as hvmcpac,
)
from training import preregister_high_volatility_state_ordered_filter as prereg


POLICY = "HVSOF-8"
PROTOCOL = "hvsof_8_gross9_novelty_v1"
PREREG = Path(
    "results/high_volatility_state_ordered_filter_preregistration_2026-08-14.json"
)
PREREG_SHA = "53e78737a357eb183c8247bbca847a8319334400be5390e4bf01861a368e3484"
SUPPORT = Path("results/high_volatility_state_ordered_filter_support_2026-08-14.json")
SUPPORT_SHA = "035ad63c5b75f3a143fa9337e09bea65ac2cc3c5db6e4eb1054b585bdc351762"
OUTPUT = Path(
    "results/high_volatility_state_ordered_filter_gross9_novelty_2026-08-14.json"
)
ELIGIBLE_CANDIDATES = (
    "HVRSSR-8__FILTERED_BY__HVTCCR-8",
    "HVRSSR-8__FILTERED_BY__HVLZC-8",
    "HVSVF-8__FILTERED_BY__HVLZC-8",
)
REJECTED_CANDIDATES = tuple(
    candidate for candidate in prereg.CANDIDATE_FAMILY if candidate not in ELIGIBLE_CANDIDATES
)
CLOCKS = {
    "HVRSSR-8__FILTERED_BY__HVTCCR-8": {
        "path": "data/high_volatility_state_ordered_filter_clocks_2023_2026/HVRSSR-8__FILTERED_BY__HVTCCR-8.csv.gz",
        "sha256": "7bc2d40ee74a82c04e862a523d3e0139b43808d003b0ee00c5d16e0fbb7f3663",
        "rows": 63,
    },
    "HVRSSR-8__FILTERED_BY__HVLZC-8": {
        "path": "data/high_volatility_state_ordered_filter_clocks_2023_2026/HVRSSR-8__FILTERED_BY__HVLZC-8.csv.gz",
        "sha256": "6dd088834f0ca59f09098ea723fbe1e2d3ba6866c9056ed2c09279280d0f54ef",
        "rows": 116,
    },
    "HVSVF-8__FILTERED_BY__HVLZC-8": {
        "path": "data/high_volatility_state_ordered_filter_clocks_2023_2026/HVSVF-8__FILTERED_BY__HVLZC-8.csv.gz",
        "sha256": "5603943f49cd936889a7b71f1d17019489a1a3ea4600ee0c70fa983a3f9eee47",
        "rows": 121,
    },
}

# The structural metric implementation and limits are the unchanged HVMCPAC contract.
gross9 = hvmcpac.gross9
metric = hvmcpac.metric
LIMITS = hvmcpac.LIMITS
evaluate_pair = hvmcpac.evaluate_pair


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hvmcpac.canonical_hash(value)


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError(f"manifest drift: {path}")
    return value


def _load_gross9_authority() -> dict[str, Any]:
    manifest = load_manifest(gross9.DEFAULT_MANIFEST)
    authority = manifest.get("authority", {})
    if (
        manifest.get("protocol_version") != gross9.PROTOCOL_VERSION
        or manifest.get("all_authoritative_counts_verified") is not True
        or authority.get("sha256") != gross9.ANCHOR_SHA256
        or authority.get("weights") != gross9.EXPECTED_WEIGHTS
        or set(manifest.get("clocks", {})) != set(gross9.EXPECTED_WEIGHTS)
    ):
        raise RuntimeError("Gross9 authority drift")
    return manifest


def load_frozen_controls() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify preregistration, source eligibility, eligible clocks, and Gross9 authority."""
    if sha(PREREG) != PREREG_SHA:
        raise RuntimeError("HVSOF-8 preregistration hash drift")
    registration = load_manifest(PREREG)
    expected_limits = {
        "exact_entry_jaccard_max": LIMITS["exact_entry_jaccard"],
        "candidate_near_6h_share_max": LIMITS["one_to_one_6h_max_matched_share"],
        "occupied_5m_jaccard_max": LIMITS["occupied_5m_bar_jaccard"],
        "absolute_signed_exposure_pearson_max": LIMITS[
            "absolute_signed_exposure_pearson"
        ],
        "must_pass_before_economics": True,
    }
    if (
        registration.get("policy_id") != POLICY
        or tuple(registration.get("candidate_family", ())) != prereg.CANDIDATE_FAMILY
        or registration.get("gross9_novelty_gates") != expected_limits
    ):
        raise RuntimeError("HVSOF-8 preregistration state drift")

    if sha(SUPPORT) != SUPPORT_SHA:
        raise RuntimeError("HVSOF-8 source-support artifact hash drift")
    support = load_manifest(SUPPORT)
    candidates = support.get("candidates", {})
    if (
        support.get("policy_id") != POLICY
        or support.get("preregistration", {}).get("sha256") != PREREG_SHA
        or tuple(candidates) != prereg.CANDIDATE_FAMILY
        or support.get("eligible_candidates_for_combination_gross9")
        != list(ELIGIBLE_CANDIDATES)
        or support.get("advance_to_combination_gross9") is not True
        or support.get("advance_to_economic_outcomes") is not False
    ):
        raise RuntimeError("HVSOF-8 source eligibility drift")
    for candidate in ELIGIBLE_CANDIDATES:
        record = candidates.get(candidate, {})
        if (
            record.get("support_passed") is not True
            or record.get("advance_to_combination_gross9") is not True
            or record.get("advance_to_economic_outcomes") is not False
            or record.get("clock") != CLOCKS[candidate]
            or sha(CLOCKS[candidate]["path"]) != CLOCKS[candidate]["sha256"]
        ):
            raise RuntimeError(f"HVSOF-8 eligible candidate drift: {candidate}")
    for candidate in REJECTED_CANDIDATES:
        record = candidates.get(candidate, {})
        if (
            record.get("support_passed") is not False
            or record.get("advance_to_combination_gross9") is not False
            or record.get("advance_to_economic_outcomes") is not False
            or record.get("decision") != "terminal_source_support_reject"
        ):
            raise RuntimeError(f"HVSOF-8 rejected candidate drift: {candidate}")
    if any(
        support.get(field) is not False
        for field in (
            "filter_postentry_returns_or_pnl_opened",
            "entry_exit_prices_opened",
            "returns_opened",
            "funding_opened",
            "pnl_opened",
            "gross9_comparator_rows_opened",
        )
    ):
        raise RuntimeError("HVSOF-8 economic evidence boundary drift")
    return registration, support, _load_gross9_authority()


def run(output: str | Path = OUTPUT) -> dict[str, Any]:
    registration, support, manifest = load_frozen_controls()
    candidate_clocks = {}
    for candidate in ELIGIBLE_CANDIDATES:
        clock = metric.load_clock(Path(CLOCKS[candidate]["path"]), label=f"HVSOF-8 {candidate}")
        if len(clock) != CLOCKS[candidate]["rows"]:
            raise RuntimeError(f"HVSOF-8 candidate clock row-count drift: {candidate}")
        candidate_clocks[candidate] = clock

    comparator_clocks = {}
    for sleeve in gross9.EXPECTED_WEIGHTS:
        record = manifest["clocks"][sleeve]
        path = Path(record["path"])
        if sha(path) != record["sha256"]:
            raise RuntimeError(f"Gross9 clock hash drift: {sleeve}")
        comparator = metric.load_clock(path, label=f"Gross9 {sleeve}")
        if len(comparator) != record["rows"] or len(comparator) != sum(
            gross9.EXPECTED_COUNTS[sleeve].values()
        ):
            raise RuntimeError(f"Gross9 count drift: {sleeve}")
        comparator_clocks[sleeve] = comparator

    candidate_results: dict[str, Any] = {}
    economics_eligible: list[str] = []
    for candidate in ELIGIBLE_CANDIDATES:
        sleeves = {
            sleeve: evaluate_pair(candidate_clocks[candidate], comparator_clocks[sleeve])
            for sleeve in gross9.EXPECTED_WEIGHTS
        }
        passed = all(result["passed"] for result in sleeves.values())
        if passed:
            economics_eligible.append(candidate)
        candidate_results[candidate] = {
            "source_support_passed": True,
            "candidate_clock": {**CLOCKS[candidate]},
            "gross9_sleeves": sleeves,
            "every_gross9_sleeve_passed": passed,
            "gross9_novelty_status": "passed" if passed else "failed",
            "advance_to_economic_outcomes": passed,
            "failure_action": None if passed else f"reject {candidate} unchanged before economics",
        }

    rejected = {
        candidate: {
            "source_support_passed": False,
            "gross9_evaluated": False,
            "advance_to_economic_outcomes": False,
            "decision": "terminal_source_support_reject",
        }
        for candidate in REJECTED_CANDIDATES
    }
    core = {
        "protocol_version": PROTOCOL,
        "policy_id": POLICY,
        "preregistration": {
            "path": PREREG.as_posix(),
            "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_support": {
            "path": SUPPORT.as_posix(),
            "sha256": SUPPORT_SHA,
            "manifest_hash": support["manifest_hash"],
            "eligible_candidates": list(ELIGIBLE_CANDIDATES),
            "rejected_candidates": list(REJECTED_CANDIDATES),
        },
        "gross9_structural_clocks": {
            "path": gross9.DEFAULT_MANIFEST.as_posix(),
            "sha256": sha(gross9.DEFAULT_MANIFEST),
            "manifest_hash": manifest["manifest_hash"],
            "authority_sha256": gross9.ANCHOR_SHA256,
            "complete_roster": list(gross9.EXPECTED_WEIGHTS),
        },
        "evidence_boundary": {
            "eligible_candidate_clock_rows_opened": sum(
                record["rows"] for record in CLOCKS.values()
            ),
            "rejected_candidate_clock_rows_opened": 0,
            "gross9_structural_clock_rows_opened": sum(
                record["rows"] for record in manifest["clocks"].values()
            ),
            "price_or_return_rows_opened": 0,
            "funding_rows_opened": 0,
            "economic_outcome_rows_opened": 0,
            "portfolio_return_or_pnl_metrics_computed": False,
            "outcomes_opened": False,
        },
        "limits": LIMITS,
        "candidate_results": candidate_results,
        "source_rejected_candidates": rejected,
        "eligible_candidates_for_economics": economics_eligible,
        "all_source_eligible_candidates_passed": len(economics_eligible)
        == len(ELIGIBLE_CANDIDATES),
        "advance_to_economic_outcomes": bool(economics_eligible),
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = run(args.output)
    print(
        json.dumps(
            {
                "eligible_candidates_for_economics": report[
                    "eligible_candidates_for_economics"
                ],
                "all_source_eligible_candidates_passed": report[
                    "all_source_eligible_candidates_passed"
                ],
            }
        )
    )
