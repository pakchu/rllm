"""Evaluate frozen SPVIR-6 structural novelty against every Gross9 sleeve."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from training import export_gross9_structural_clocks as gross9
from training import evaluate_options_led_volatility_expansion_premium_relay_gross9_novelty as metric


POLICY = "SPVIR-6"
PROTOCOL = "spvir_6_gross9_novelty_v1"
PREREG = Path("results/spot_participation_volatility_ignition_relay_preregistration_2026-08-08.json")
PREREG_SHA = "826a457dc93ca340abc08f67e44d02be39bb47047704ccf4350f9bfb03cd9b4f"
SUPPORT = Path("results/spot_participation_volatility_ignition_relay_support_2026-08-08.json")
SUPPORT_SHA = "ac1315456d38ff3ed2926d8c9a1dd398fe7e8db9de47d1d261029f7d64811fb0"
CLOCK = Path("data/spot_participation_volatility_ignition_relay_clocks_2023_2026.csv.gz")
CLOCK_SHA = "a87045fbc0aa00e7852bd6eaf6b4ae6b87d7349d5a78cc79af07679a8284f6dc"
OUTPUT = Path("results/spot_participation_volatility_ignition_relay_gross9_novelty_2026-08-08.json")
LIMITS = {
    "exact_entry_jaccard": 0.10,
    "one_to_one_6h_max_matched_share": 0.35,
    "occupied_5m_bar_jaccard": 0.25,
    "absolute_signed_exposure_pearson": 0.35,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return gross9.canonical_hash(payload)


def load_manifest(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    if report.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError(f"manifest drift: {path}")
    return report


def evaluate_pair(candidate, comparator) -> dict[str, Any]:
    result = metric.evaluate_pair(candidate, comparator)
    result["checks"] = {
        name: result["metrics"][name] <= limit for name, limit in LIMITS.items()
    }
    result["passed"] = all(result["checks"].values())
    return result


def run(output: Path = OUTPUT) -> dict[str, Any]:
    if sha256(PREREG) != PREREG_SHA or sha256(SUPPORT) != SUPPORT_SHA or sha256(CLOCK) != CLOCK_SHA:
        raise RuntimeError("SPVIR predecessor hash drift")
    registration = load_manifest(PREREG)
    support = load_manifest(SUPPORT)
    expected_limits = {
        "exact_entry_jaccard_max": 0.10,
        "candidate_near_6h_share_max": 0.35,
        "occupied_5m_jaccard_max": 0.25,
        "absolute_signed_exposure_pearson_max": 0.35,
        "must_pass_before_economics": True,
    }
    if registration.get("novelty_gates") != expected_limits:
        raise RuntimeError("SPVIR novelty limits drift")
    if (
        support.get("policy_id") != POLICY
        or support.get("support_passed") is not True
        or support.get("advance_to_gross9_novelty") is not True
        or support.get("advance_to_economic_outcomes") is not False
        or support.get("clock", {}).get("sha256") != CLOCK_SHA
    ):
        raise RuntimeError("SPVIR predecessor state drift")
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
    candidate = metric.load_clock(CLOCK, label="SPVIR primary")
    results: dict[str, Any] = {}
    for sleeve in gross9.EXPECTED_WEIGHTS:
        record = manifest["clocks"][sleeve]
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"Gross9 clock hash drift: {sleeve}")
        comparator = metric.load_clock(path, label=f"Gross9 {sleeve}")
        if len(comparator) != record["rows"] or len(comparator) != sum(gross9.EXPECTED_COUNTS[sleeve].values()):
            raise RuntimeError(f"Gross9 count drift: {sleeve}")
        results[sleeve] = evaluate_pair(candidate, comparator)
    passed = all(result["passed"] for result in results.values())
    advance = support["support_passed"] and passed
    core = {
        "protocol_version": PROTOCOL,
        "policy_id": POLICY,
        "preregistration": {
            "path": str(PREREG), "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_support": {
            "path": str(SUPPORT), "sha256": SUPPORT_SHA,
            "manifest_hash": support["manifest_hash"], "predecessor_mutated": False,
        },
        "gross9_structural_clocks": {
            "path": str(gross9.DEFAULT_MANIFEST), "sha256": sha256(gross9.DEFAULT_MANIFEST),
            "manifest_hash": manifest["manifest_hash"],
            "authority_sha256": gross9.ANCHOR_SHA256,
            "complete_roster": list(gross9.EXPECTED_WEIGHTS),
        },
        "evidence_boundary": {
            "pviar_clock_rows_opened": len(candidate),
            "gross9_structural_clock_rows_opened": sum(record["rows"] for record in manifest["clocks"].values()),
            "btc_execution_rows_opened": 0, "btc_price_or_return_rows_opened": 0,
            "funding_rows_opened": 0, "economic_outcome_rows_opened": 0,
            "portfolio_return_or_pnl_metrics_computed": False, "outcomes_opened": False,
        },
        "limits": LIMITS, "gross9_sleeves": results,
        "source_support_passed": support["support_passed"],
        "every_gross9_sleeve_passed": passed,
        "gross9_novelty_status": "passed" if passed else "failed",
        "advance_to_economic_outcomes": advance,
        "failure_action": None if advance else "reject SPVIR-6 unchanged before economic outcomes",
    }
    report = {**core, "manifest_hash": canonical_hash(core)}
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({
        "status": report["gross9_novelty_status"],
        "advance": report["advance_to_economic_outcomes"],
    }))
