"""Evaluate all four source-supported HVSPR-8 routers against Gross9."""

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
from training import preregister_high_volatility_state_priority_router as prereg

POLICY = "HVSPR-8"
PROTOCOL = "hvspr_8_gross9_novelty_v1"
PREREG = Path(
    "results/high_volatility_state_priority_router_preregistration_2026-08-14.json"
)
PREREG_SHA = "80b8865a9b3b44af04e682f5179fac4482c5f839a9a96e6c10daa9f396304c3b"
SUPPORT = Path("results/high_volatility_state_priority_router_support_2026-08-14.json")
SUPPORT_SHA = "57e4b0658fadcf508b6855ea91b7c5014b67554c98902404532dc331d50e7d48"
OUTPUT = Path(
    "results/high_volatility_state_priority_router_gross9_novelty_2026-08-14.json"
)
ELIGIBLE_ROUTERS = prereg.CANDIDATE_FAMILY
CLOCKS = {
    "HVTFR-8__THEN__HVSVF-8__THEN__HVRSSR-8__ELIGIBLE_BY__HVTCCR-8": {
        "path": "data/high_volatility_state_priority_router_clocks_2023_2026/HVTFR-8__THEN__HVSVF-8__THEN__HVRSSR-8__ELIGIBLE_BY__HVTCCR-8.csv.gz",
        "sha256": "d15453de52f76eaf4e0610f9b84cac70e83c580ab732749820258ab902d83d81",
        "rows": 181,
    },
    "HVTFR-8__THEN__HVSVF-8__THEN__HVRSSR-8__ELIGIBLE_BY__HVLZC-8": {
        "path": "data/high_volatility_state_priority_router_clocks_2023_2026/HVTFR-8__THEN__HVSVF-8__THEN__HVRSSR-8__ELIGIBLE_BY__HVLZC-8.csv.gz",
        "sha256": "d73262d5e6da57549a13fd585f0f8eefacbc860327dfd5f7cc93c0a5251833e4",
        "rows": 268,
    },
    "HVRSSR-8__THEN__HVSVF-8__THEN__HVTFR-8__ELIGIBLE_BY__HVTCCR-8": {
        "path": "data/high_volatility_state_priority_router_clocks_2023_2026/HVRSSR-8__THEN__HVSVF-8__THEN__HVTFR-8__ELIGIBLE_BY__HVTCCR-8.csv.gz",
        "sha256": "10c91f140758a1f9b15ea7d55509ebac7664ec0f79a7636b9c5dbf403a862852",
        "rows": 181,
    },
    "HVRSSR-8__THEN__HVSVF-8__THEN__HVTFR-8__ELIGIBLE_BY__HVLZC-8": {
        "path": "data/high_volatility_state_priority_router_clocks_2023_2026/HVRSSR-8__THEN__HVSVF-8__THEN__HVTFR-8__ELIGIBLE_BY__HVLZC-8.csv.gz",
        "sha256": "c78d66c504d926f08d7b934213f939d875dafe82ec3adacd447e5395e4129c6d",
        "rows": 268,
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
        raise RuntimeError("HVSPR-8 preregistration hash drift")
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
        or tuple(registration.get("candidate_family", ())) != ELIGIBLE_ROUTERS
        or registration.get("candidate_family_size") != len(ELIGIBLE_ROUTERS)
        or registration.get("gross9_novelty_gates") != expected_limits
    ):
        raise RuntimeError("HVSPR-8 preregistration state drift")

    if sha(SUPPORT) != SUPPORT_SHA:
        raise RuntimeError("HVSPR-8 source-support artifact hash drift")
    support = load_manifest(SUPPORT)
    routers = support.get("candidates", {})
    if (
        support.get("policy_id") != POLICY
        or support.get("preregistration", {}).get("sha256") != PREREG_SHA
        or tuple(routers) != ELIGIBLE_ROUTERS
        or support.get("eligible_routers_for_combination_gross9")
        != list(ELIGIBLE_ROUTERS)
        or support.get("eligible_router_count") != len(ELIGIBLE_ROUTERS)
        or support.get("advance_to_combination_gross9") is not True
        or support.get("advance_to_economic_outcomes") is not False
    ):
        raise RuntimeError("HVSPR-8 source eligibility drift")
    for router in ELIGIBLE_ROUTERS:
        record = routers.get(router, {})
        if (
            record.get("support_passed") is not True
            or record.get("advance_to_combination_gross9") is not True
            or record.get("advance_to_economic_outcomes") is not False
            or record.get("clock") != CLOCKS[router]
            or sha(CLOCKS[router]["path"]) != CLOCKS[router]["sha256"]
        ):
            raise RuntimeError(f"HVSPR-8 eligible router drift: {router}")
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
        raise RuntimeError("HVSPR-8 economic evidence boundary drift")
    return registration, support, hvsof._load_gross9_authority()


def run(output: str | Path = OUTPUT) -> dict[str, Any]:
    registration, support, manifest = load_frozen_controls()
    router_clocks = {}
    for router in ELIGIBLE_ROUTERS:
        clock = metric.load_clock(
            Path(CLOCKS[router]["path"]), label=f"HVSPR-8 {router}"
        )
        if len(clock) != CLOCKS[router]["rows"]:
            raise RuntimeError(f"HVSPR-8 router clock row-count drift: {router}")
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
