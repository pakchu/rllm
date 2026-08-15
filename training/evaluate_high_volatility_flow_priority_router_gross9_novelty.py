"""Evaluate both source-supported HVFPR-6 routers against Gross9."""

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
    evaluate_high_volatility_state_ordered_filter_gross9_novelty as hvsof,
)
from training import preregister_high_volatility_flow_priority_router as prereg

POLICY = "HVFPR-6"
PROTOCOL = "hvfpr_6_gross9_novelty_v1"
PREREG = Path(
    "results/high_volatility_flow_priority_router_preregistration_2026-08-16.json"
)
PREREG_SHA = "c9a6c2799155fa89bf6fdecdfd66a97e5777a468efe5ad290e643eb8704a21c8"
SUPPORT = Path("results/high_volatility_flow_priority_router_support_2026-08-16.json")
SUPPORT_SHA = "5f0a95c151f12ac19ecc4b20fb0e3bc49ea49d47ff9e8b160379bf7ca42d93e6"
OUTPUT = Path(
    "results/high_volatility_flow_priority_router_gross9_novelty_2026-08-16.json"
)
ELIGIBLE_ROUTERS = prereg.CANDIDATE_FAMILY
CLOCKS = {
    "HVAFC-6__THEN__HVELR-6__THEN__RIVSCR-6__ELIGIBLE_BY__HVTCCR-8": {
        "path": "data/high_volatility_flow_priority_router_clocks_2023_2026/HVAFC-6__THEN__HVELR-6__THEN__RIVSCR-6__ELIGIBLE_BY__HVTCCR-8.csv.gz",
        "sha256": "dcd2aaaf153d6e299cbcf3abf9236995d526cd4b33c44cb95178446ec0034a6a",
        "rows": 160,
    },
    "RIVSCR-6__THEN__HVELR-6__THEN__HVAFC-6__ELIGIBLE_BY__HVTCCR-8": {
        "path": "data/high_volatility_flow_priority_router_clocks_2023_2026/RIVSCR-6__THEN__HVELR-6__THEN__HVAFC-6__ELIGIBLE_BY__HVTCCR-8.csv.gz",
        "sha256": "56c79c267dff57c38aeb972fdfeb07ce7c269d06e6dea36e24c40fe8a8a90b94",
        "rows": 160,
    },
}

# Reuse without substitution the complete HVSOF Gross9 authority and metric contract.
gross9 = hvsof.gross9
metric = hvsof.metric
LIMITS = hvsof.LIMITS
evaluate_pair = hvsof.evaluate_pair


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hvsof.canonical_hash(value)


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact is not an object: {path}")
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError(f"manifest drift: {path}")
    return value


def load_frozen_controls() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify preregistration, source eligibility, router clocks, and Gross9 authority."""
    if sha(PREREG) != PREREG_SHA:
        raise RuntimeError("HVFPR-6 preregistration hash drift")
    registration = load_manifest(PREREG)
    expected_limits = {
        "exact_entry_jaccard_max": LIMITS["exact_entry_jaccard"],
        "candidate_near_6h_share_max": LIMITS["one_to_one_6h_max_matched_share"],
        "occupied_5m_bar_jaccard_max": LIMITS["occupied_5m_bar_jaccard"],
        "absolute_signed_exposure_pearson_max": LIMITS[
            "absolute_signed_exposure_pearson"
        ],
        "must_pass_before_economics": True,
    }
    if (
        registration.get("policy_id") != POLICY
        or tuple(registration.get("candidate_family", ())) != prereg.CANDIDATE_FAMILY
        or registration.get("candidate_family_size") != len(prereg.CANDIDATE_FAMILY)
        or registration.get("gross9_novelty_gates") != expected_limits
    ):
        raise RuntimeError("HVFPR-6 preregistration state drift")

    if sha(SUPPORT) != SUPPORT_SHA:
        raise RuntimeError("HVFPR-6 source-support artifact hash drift")
    support = load_manifest(SUPPORT)
    routers = support.get("candidates", {})
    if (
        support.get("policy_id") != POLICY
        or support.get("preregistration", {}).get("sha256") != PREREG_SHA
        or tuple(routers) != prereg.CANDIDATE_FAMILY
        or support.get("eligible_routers_for_combination_gross9")
        != list(ELIGIBLE_ROUTERS)
        or support.get("eligible_router_count") != len(ELIGIBLE_ROUTERS)
        or support.get("advance_to_combination_gross9") is not True
        or support.get("advance_to_economic_outcomes") is not False
    ):
        raise RuntimeError("HVFPR-6 source eligibility drift")
    for router in ELIGIBLE_ROUTERS:
        record = routers.get(router, {})
        if (
            record.get("support_passed") is not True
            or record.get("advance_to_combination_gross9") is not True
            or record.get("advance_to_economic_outcomes") is not False
            or record.get("clock") != CLOCKS[router]
            or sha(CLOCKS[router]["path"]) != CLOCKS[router]["sha256"]
        ):
            raise RuntimeError(f"HVFPR-6 eligible router drift: {router}")
    if any(
        support.get(field) is not False
        for field in (
            "router_postentry_returns_or_pnl_opened",
            "entry_exit_prices_opened",
            "returns_opened",
            "funding_opened",
            "pnl_opened",
            "gross9_comparator_rows_opened",
        )
    ):
        raise RuntimeError("HVFPR-6 economic evidence boundary drift")
    return registration, support, hvsof._load_gross9_authority()


def run(output: str | Path = OUTPUT) -> dict[str, Any]:
    registration, support, manifest = load_frozen_controls()
    router_clocks = {}
    for router in ELIGIBLE_ROUTERS:
        clock = metric.load_clock(
            Path(CLOCKS[router]["path"]), label=f"HVFPR-6 {router}"
        )
        if len(clock) != CLOCKS[router]["rows"]:
            raise RuntimeError(f"HVFPR-6 router clock row-count drift: {router}")
        router_clocks[router] = clock

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

    router_results: dict[str, Any] = {}
    economics_eligible: list[str] = []
    for router in ELIGIBLE_ROUTERS:
        sleeves = {
            sleeve: evaluate_pair(router_clocks[router], comparator_clocks[sleeve])
            for sleeve in gross9.EXPECTED_WEIGHTS
        }
        passed = all(result["passed"] for result in sleeves.values())
        if passed:
            economics_eligible.append(router)
        router_results[router] = {
            "source_support_passed": True,
            "router_clock": {**CLOCKS[router]},
            "gross9_sleeves": sleeves,
            "every_gross9_sleeve_passed": passed,
            "gross9_novelty_status": "passed" if passed else "failed",
            "advance_to_economic_outcomes": passed,
            "failure_action": None
            if passed
            else f"reject {router} unchanged before economics",
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
            "eligible_routers": list(ELIGIBLE_ROUTERS),
        },
        "gross9_structural_clocks": {
            "path": gross9.DEFAULT_MANIFEST.as_posix(),
            "sha256": sha(gross9.DEFAULT_MANIFEST),
            "manifest_hash": manifest["manifest_hash"],
            "authority_sha256": gross9.ANCHOR_SHA256,
            "complete_roster": list(gross9.EXPECTED_WEIGHTS),
        },
        "evidence_boundary": {
            "eligible_router_clock_rows_opened": sum(
                record["rows"] for record in CLOCKS.values()
            ),
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
        "router_results": router_results,
        "eligible_routers_for_economics": economics_eligible,
        "all_source_supported_routers_passed": len(economics_eligible)
        == len(ELIGIBLE_ROUTERS),
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
                "eligible_routers_for_economics": report[
                    "eligible_routers_for_economics"
                ],
                "all_source_supported_routers_passed": report[
                    "all_source_supported_routers_passed"
                ],
            }
        )
    )
