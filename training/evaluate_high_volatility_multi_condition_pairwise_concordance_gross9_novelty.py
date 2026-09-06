"""Evaluate the sole source-supported HVMCPAC-8 pair against Gross9."""
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
from training import (
    evaluate_options_led_volatility_expansion_premium_relay_gross9_novelty as metric,
)


POLICY = "HVMCPAC-8"
PROTOCOL = "hvmcpac_8_gross9_novelty_v1"
PAIR = "CARSC-8__AND__HVTFR-8"
PREREG = Path(
    "results/high_volatility_multi_condition_pairwise_concordance_"
    "preregistration_2026-08-14.json"
)
PREREG_SHA = "3cdd3edbedfda4e581bb95b9fac2db7309a7b54fe72c37ff5e5004ce8bba8d14"
SUPPORT = Path(
    "results/high_volatility_multi_condition_pairwise_concordance_support_2026-08-14.json"
)
SUPPORT_SHA = "dd2e185fd924ce60eda3bd9c0c4fb1813ec79ff696cf3e6011182ff5c09293c6"
PAIR_CLOCK = Path(
    "data/high_volatility_multi_condition_pairwise_concordance_clocks_2023_2026/"
    f"{PAIR}.csv.gz"
)
PAIR_CLOCK_SHA = "463bd2b9aa72eb292078387635471480688d6265749a51f12fc2c40ef77b0da7"
PAIR_CLOCK_ROWS = 85
OUTPUT = Path(
    "results/high_volatility_multi_condition_pairwise_concordance_"
    "gross9_novelty_2026-08-14.json"
)
LIMITS = {
    "exact_entry_jaccard": 0.10,
    "one_to_one_6h_max_matched_share": 0.35,
    "occupied_5m_bar_jaccard": 0.25,
    "absolute_signed_exposure_pearson": 0.35,
}


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return gross9.canonical_hash(value)


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError(f"manifest drift: {path}")
    return value


def evaluate_pair(candidate: Any, comparator: Any) -> dict[str, Any]:
    """Apply the frozen Gross9 structural metrics and HVMCPAC-8 limits."""
    result = metric.evaluate_pair(candidate, comparator)
    result["checks"] = {
        name: result["metrics"][name] <= limit for name, limit in LIMITS.items()
    }
    result["passed"] = all(result["checks"].values())
    return result


def load_frozen_controls() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha(PREREG) != PREREG_SHA:
        raise RuntimeError("HVMCPAC-8 preregistration hash drift")
    registration = load_manifest(PREREG)
    expected_limits = {
        "exact_entry_jaccard_max": 0.1,
        "candidate_near_6h_share_max": 0.35,
        "occupied_5m_jaccard_max": 0.25,
        "absolute_signed_exposure_pearson_max": 0.35,
        "must_pass_before_economics": True,
    }
    if (
        registration.get("policy_id") != POLICY
        or registration.get("gross9_novelty_gates") != expected_limits
    ):
        raise RuntimeError("HVMCPAC-8 preregistration state drift")

    if sha(SUPPORT) != SUPPORT_SHA:
        raise RuntimeError("HVMCPAC-8 source-support artifact hash drift")
    support = load_manifest(SUPPORT)
    pair = support.get("pairs", {}).get(PAIR, {})
    clock = pair.get("clock", {})
    rejected_pairs = [
        candidate
        for candidate, record in support.get("pairs", {}).items()
        if candidate != PAIR and record.get("support_passed") is not False
    ]
    if (
        support.get("policy_id") != POLICY
        or support.get("preregistration", {}).get("sha256") != PREREG_SHA
        or support.get("eligible_pairs_for_combination_gross9") != [PAIR]
        or support.get("advance_to_combination_gross9") is not True
        or support.get("advance_to_combination_economic_outcomes") is not False
        or pair.get("support_passed") is not True
        or pair.get("advance_to_combination_gross9") is not True
        or pair.get("advance_to_combination_economic_outcomes") is not False
        or clock.get("path") != PAIR_CLOCK.as_posix()
        or clock.get("sha256") != PAIR_CLOCK_SHA
        or clock.get("rows") != PAIR_CLOCK_ROWS
        or rejected_pairs
    ):
        raise RuntimeError("HVMCPAC-8 source eligibility drift")
    if (
        support.get("combination_outcomes_opened") is not False
        or support.get("combination_postentry_returns_or_pnl_opened") is not False
        or support.get("entry_exit_prices_opened") is not False
        or support.get("funding_opened") is not False
    ):
        raise RuntimeError("HVMCPAC-8 economic evidence boundary drift")
    if sha(PAIR_CLOCK) != PAIR_CLOCK_SHA:
        raise RuntimeError("HVMCPAC-8 pair-clock hash drift")

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
    return registration, support, manifest


def run(output: str | Path = OUTPUT) -> dict[str, Any]:
    registration, support, manifest = load_frozen_controls()
    candidate = metric.load_clock(PAIR_CLOCK, label=f"HVMCPAC-8 {PAIR}")
    if len(candidate) != PAIR_CLOCK_ROWS:
        raise RuntimeError("HVMCPAC-8 pair-clock row-count drift")

    sleeve_results: dict[str, Any] = {}
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
        sleeve_results[sleeve] = evaluate_pair(candidate, comparator)

    passed = all(result["passed"] for result in sleeve_results.values())
    source_eligible = support["pairs"][PAIR]["support_passed"] is True
    advance = source_eligible and passed
    core = {
        "protocol_version": PROTOCOL,
        "policy_id": POLICY,
        "candidate": PAIR,
        "preregistration": {
            "path": PREREG.as_posix(),
            "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_support": {
            "path": SUPPORT.as_posix(),
            "sha256": SUPPORT_SHA,
            "manifest_hash": support["manifest_hash"],
            "sole_eligible_pair": PAIR,
            "predecessor_mutated": False,
        },
        "pair_clock": {
            "path": PAIR_CLOCK.as_posix(),
            "sha256": PAIR_CLOCK_SHA,
            "rows": len(candidate),
        },
        "gross9_structural_clocks": {
            "path": gross9.DEFAULT_MANIFEST.as_posix(),
            "sha256": sha(gross9.DEFAULT_MANIFEST),
            "manifest_hash": manifest["manifest_hash"],
            "authority_sha256": gross9.ANCHOR_SHA256,
            "complete_roster": list(gross9.EXPECTED_WEIGHTS),
        },
        "evidence_boundary": {
            "pair_clock_rows_opened": len(candidate),
            "gross9_structural_clock_rows_opened": sum(
                record["rows"] for record in manifest["clocks"].values()
            ),
            "btc_execution_rows_opened": 0,
            "btc_price_or_return_rows_opened": 0,
            "funding_rows_opened": 0,
            "economic_outcome_rows_opened": 0,
            "portfolio_return_or_pnl_metrics_computed": False,
            "outcomes_opened": False,
        },
        "limits": LIMITS,
        "gross9_sleeves": sleeve_results,
        "source_support_passed": source_eligible,
        "every_gross9_sleeve_passed": passed,
        "gross9_novelty_status": "passed" if passed else "failed",
        "advance_to_economic_outcomes": advance,
        "failure_action": (
            None
            if advance
            else f"reject {PAIR} unchanged before economic outcomes"
        ),
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
                "status": report["gross9_novelty_status"],
                "advance": report["advance_to_economic_outcomes"],
            }
        )
    )
