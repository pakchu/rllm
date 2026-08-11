"""Evaluate frozen HVCVD-24 structural novelty against every Gross9 sleeve."""
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

POLICY = "HVCVD-24"
PROTOCOL = "hvcvd_24_gross9_novelty_v1"
PREREG = Path("results/high_volatility_cumulative_volume_delta_signal_crossover_relay_preregistration_2026-08-11.json")
PREREG_SHA = "686591ecdd900ea022bd61dfc4bdb9e0d01307897864ec9c74b388dc7e3b6a6e"
SUPPORT = Path("results/high_volatility_cumulative_volume_delta_signal_crossover_relay_support_2026-08-11.json")
SUPPORT_SHA = "a7e7bdbfa5239ac5c89b7977f881cc90ef0e81c87e339108c6d5a20970a3ad99"
CLOCK = Path("data/high_volatility_cumulative_volume_delta_signal_crossover_relay_clocks_2023_2026.csv.gz")
CLOCK_SHA = "14055168be9923a968a6dfa19364388ceaf7ebe862723c0693ee2b36bb6d4df8"
OUTPUT = Path("results/high_volatility_cumulative_volume_delta_signal_crossover_relay_gross9_novelty_2026-08-11.json")
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
        raise RuntimeError("HVCVD predecessor hash drift")
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
        raise RuntimeError("HVCVD novelty limits drift")
    if (
        support.get("policy_id") != POLICY
        or support.get("support_passed") is not True
        or support.get("advance_to_gross9_novelty") is not True
        or support.get("advance_to_economic_outcomes") is not False
        or support.get("clock", {}).get("sha256") != CLOCK_SHA
    ):
        raise RuntimeError("HVCVD predecessor state drift")

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

    candidate = metric.load_clock(CLOCK, label="HVCVD primary")
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
            "hvcvd_clock_rows_opened": len(candidate),
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
        "failure_action": None if advance else "reject HVCVD-24 unchanged before economic outcomes",
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
