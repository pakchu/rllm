"""Evaluate frozen HVCHV-12 structural novelty against every Gross9 sleeve."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import export_gross9_structural_clocks as gross9
from training import evaluate_options_led_volatility_expansion_premium_relay_gross9_novelty as metric

POLICY = "HVCHV-12"
PROTOCOL = "hvchv_12_gross9_novelty_v1"
PREREG = Path("results/high_volatility_chaikin_volatility_expansion_trend_relay_preregistration_2026-08-11.json")
PREREG_SHA = "dbb582c2b01f009c76b7d3fa2b51714252ff82c2e25e0bb55372d0787365dfed"
SUPPORT = Path("results/high_volatility_chaikin_volatility_expansion_trend_relay_support_2026-08-11.json")
SUPPORT_SHA = "6e6856920352f15c551e8437ba090f259af11890d8e9e2f33cd9dc3ec479af61"
CLOCK = Path("data/high_volatility_chaikin_volatility_expansion_trend_relay_clocks_2023_2026.csv.gz")
CLOCK_SHA = "916d2e333eb94aa78eec7673da246aa046ba82db12b217e4fab670cffc3673e6"
OUTPUT = Path("results/high_volatility_chaikin_volatility_expansion_trend_relay_gross9_novelty_2026-08-11.json")
LIMITS = {
    "exact_entry_jaccard": 0.10,
    "one_to_one_6h_max_matched_share": 0.35,
    "occupied_5m_bar_jaccard": 0.25,
    "absolute_signed_exposure_pearson": 0.35,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return gross9.canonical_hash(value)


def load_manifest(path: Path) -> dict:
    value = json.loads(path.read_text())
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError(f"manifest drift: {path}")
    return value


def evaluate_pair(candidate, comparator):
    result = metric.evaluate_pair(candidate, comparator)
    result["checks"] = {name: result["metrics"][name] <= limit for name, limit in LIMITS.items()}
    result["passed"] = all(result["checks"].values())
    return result


def run(output: Path = OUTPUT) -> dict:
    if sha(PREREG) != PREREG_SHA or sha(SUPPORT) != SUPPORT_SHA or sha(CLOCK) != CLOCK_SHA:
        raise RuntimeError("HVCHV predecessor hash drift")
    preregistration = load_manifest(PREREG)
    support = load_manifest(SUPPORT)
    expected_limits = {
        "exact_entry_jaccard_max": 0.1,
        "candidate_near_6h_share_max": 0.35,
        "occupied_5m_jaccard_max": 0.25,
        "absolute_signed_exposure_pearson_max": 0.35,
        "must_pass_before_economics": True,
    }
    if preregistration.get("novelty_gates") != expected_limits:
        raise RuntimeError("HVCHV novelty limits drift")
    if (
        support.get("policy_id") != POLICY
        or support.get("support_passed") is not True
        or support.get("advance_to_gross9_novelty") is not True
        or support.get("advance_to_economic_outcomes") is not False
        or support.get("clock", {}).get("sha256") != CLOCK_SHA
    ):
        raise RuntimeError("HVCHV predecessor state drift")

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

    candidate = metric.load_clock(CLOCK, label="HVCHV primary")
    sleeve_results = {}
    for sleeve in gross9.EXPECTED_WEIGHTS:
        record = manifest["clocks"][sleeve]
        path = Path(record["path"])
        if sha(path) != record["sha256"]:
            raise RuntimeError(f"Gross9 clock hash drift: {sleeve}")
        comparator = metric.load_clock(path, label=f"Gross9 {sleeve}")
        if len(comparator) != record["rows"] or len(comparator) != sum(gross9.EXPECTED_COUNTS[sleeve].values()):
            raise RuntimeError(f"Gross9 count drift: {sleeve}")
        sleeve_results[sleeve] = evaluate_pair(candidate, comparator)

    passed = all(result["passed"] for result in sleeve_results.values())
    advance = support["support_passed"] and passed
    core = {
        "protocol_version": PROTOCOL,
        "policy_id": POLICY,
        "preregistration": {"path": str(PREREG), "sha256": PREREG_SHA, "manifest_hash": preregistration["manifest_hash"]},
        "source_support": {"path": str(SUPPORT), "sha256": SUPPORT_SHA, "manifest_hash": support["manifest_hash"], "predecessor_mutated": False},
        "gross9_structural_clocks": {
            "path": str(gross9.DEFAULT_MANIFEST),
            "sha256": sha(gross9.DEFAULT_MANIFEST),
            "manifest_hash": manifest["manifest_hash"],
            "authority_sha256": gross9.ANCHOR_SHA256,
            "complete_roster": list(gross9.EXPECTED_WEIGHTS),
        },
        "evidence_boundary": {
            "hvchv_clock_rows_opened": len(candidate),
            "gross9_structural_clock_rows_opened": sum(item["rows"] for item in manifest["clocks"].values()),
            "btc_execution_rows_opened": 0,
            "btc_price_or_return_rows_opened": 0,
            "funding_rows_opened": 0,
            "economic_outcome_rows_opened": 0,
            "portfolio_return_or_pnl_metrics_computed": False,
            "outcomes_opened": False,
        },
        "limits": LIMITS,
        "gross9_sleeves": sleeve_results,
        "source_support_passed": support["support_passed"],
        "every_gross9_sleeve_passed": passed,
        "gross9_novelty_status": "passed" if passed else "failed",
        "advance_to_economic_outcomes": advance,
        "failure_action": None if advance else "reject HVCHV-12 unchanged before economic outcomes",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    evaluation = run(args.output)
    print(json.dumps({"status": evaluation["gross9_novelty_status"], "advance": evaluation["advance_to_economic_outcomes"]}))
