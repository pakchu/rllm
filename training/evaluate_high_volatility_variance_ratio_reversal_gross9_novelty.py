"""Evaluate frozen HVVRR-12 structural novelty against every Gross9 sleeve."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from training import export_gross9_structural_clocks as gross9
from training import (
    evaluate_options_led_volatility_expansion_premium_relay_gross9_novelty as metric,
)


POLICY = "HVVRR-12"
PROTOCOL = "hvvrr_12_gross9_novelty_v1"
PREREG = Path(
    "results/high_volatility_variance_ratio_reversal_preregistration_2026-08-09.json"
)
PREREG_SHA = "bf205158df3dcd057459fa2cb692e0ccec0ca256d59870e4fd15287a8b98bcb3"
SUPPORT = Path("results/high_volatility_variance_ratio_reversal_support_2026-08-09.json")
SUPPORT_SHA = "ce1a882ae02dc72d163460c4dc7003b06d1f797bee5fefd6210900d26ab5fdc2"
CLOCK = Path("data/high_volatility_variance_ratio_reversal_clocks_2023_2026.csv.gz")
CLOCK_SHA = "bfedce7712ee9cf6e0c92c1071baf736b2a1f9f3c795d6551d7aab31d113d1d0"
OUTPUT = Path(
    "results/high_volatility_variance_ratio_reversal_gross9_novelty_2026-08-09.json"
)
LIMITS = {
    "exact_entry_jaccard": 0.10,
    "one_to_one_6h_max_matched_share": 0.35,
    "occupied_5m_bar_jaccard": 0.25,
    "absolute_signed_exposure_pearson": 0.35,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chash(payload: Any) -> str:
    return gross9.canonical_hash(payload)


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != chash(core):
        raise RuntimeError(f"manifest drift: {path}")
    return payload


def pair(candidate: Any, comparator: Any) -> dict[str, Any]:
    result = metric.evaluate_pair(candidate, comparator)
    result["checks"] = {
        name: result["metrics"][name] <= limit for name, limit in LIMITS.items()
    }
    result["passed"] = all(result["checks"].values())
    return result


def run(output: Path = OUTPUT) -> dict[str, Any]:
    if sha(PREREG) != PREREG_SHA or sha(SUPPORT) != SUPPORT_SHA or sha(CLOCK) != CLOCK_SHA:
        raise RuntimeError("HVVRR predecessor hash drift")
    registration = load(PREREG)
    support = load(SUPPORT)
    expected = {
        "exact_entry_jaccard_max": 0.1,
        "candidate_near_6h_share_max": 0.35,
        "occupied_5m_bar_jaccard_max": 0.25,
        "absolute_signed_exposure_pearson_max": 0.35,
        "must_pass_before_economics": True,
    }
    if registration.get("novelty_gates") != expected:
        raise RuntimeError("HVVRR novelty limits drift")
    if (
        support.get("policy_id") != POLICY
        or support.get("support_passed") is not True
        or support.get("advance_to_gross9_novelty") is not True
        or support.get("advance_to_economic_outcomes") is not False
        or support.get("clock", {}).get("sha256") != CLOCK_SHA
    ):
        raise RuntimeError("HVVRR predecessor state drift")

    manifest = load(gross9.DEFAULT_MANIFEST)
    authority = manifest.get("authority", {})
    if (
        manifest.get("protocol_version") != gross9.PROTOCOL_VERSION
        or manifest.get("all_authoritative_counts_verified") is not True
        or authority.get("sha256") != gross9.ANCHOR_SHA256
        or authority.get("weights") != gross9.EXPECTED_WEIGHTS
        or set(manifest.get("clocks", {})) != set(gross9.EXPECTED_WEIGHTS)
    ):
        raise RuntimeError("Gross9 authority drift")

    candidate = metric.load_clock(CLOCK, label="HVVRR primary")
    results: dict[str, Any] = {}
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
        results[sleeve] = pair(candidate, comparator)

    passed = all(item["passed"] for item in results.values())
    advance = support["support_passed"] and passed
    core = {
        "protocol_version": PROTOCOL,
        "policy_id": POLICY,
        "preregistration": {
            "path": str(PREREG),
            "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_support": {
            "path": str(SUPPORT),
            "sha256": SUPPORT_SHA,
            "manifest_hash": support["manifest_hash"],
            "predecessor_mutated": False,
        },
        "gross9_structural_clocks": {
            "path": str(gross9.DEFAULT_MANIFEST),
            "sha256": sha(gross9.DEFAULT_MANIFEST),
            "manifest_hash": manifest["manifest_hash"],
            "authority_sha256": gross9.ANCHOR_SHA256,
            "complete_roster": list(gross9.EXPECTED_WEIGHTS),
        },
        "evidence_boundary": {
            "hvvrr_clock_rows_opened": len(candidate),
            "gross9_structural_clock_rows_opened": sum(
                item["rows"] for item in manifest["clocks"].values()
            ),
            "btc_execution_rows_opened": 0,
            "btc_price_or_return_rows_opened": 0,
            "funding_rows_opened": 0,
            "economic_outcome_rows_opened": 0,
            "portfolio_return_or_pnl_metrics_computed": False,
            "outcomes_opened": False,
        },
        "limits": LIMITS,
        "gross9_sleeves": results,
        "source_support_passed": support["support_passed"],
        "every_gross9_sleeve_passed": passed,
        "gross9_novelty_status": "passed" if passed else "failed",
        "advance_to_economic_outcomes": advance,
        "failure_action": (
            None if advance else "reject HVVRR-12 unchanged before economic outcomes"
        ),
    }
    result = {**core, "manifest_hash": chash(core)}
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
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
                "status": report["gross9_novelty_status"],
                "advance": report["advance_to_economic_outcomes"],
            }
        )
    )
