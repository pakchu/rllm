"""Evaluate the frozen HVCAV-8 action-vote clock against Gross9."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from training import evaluate_high_volatility_state_ordered_filter_gross9_novelty as hvsof
from training import preregister_high_volatility_cross_structure_action_vote as prereg

POLICY = "HVCAV-8"
PROTOCOL = "hvcav_8_gross9_novelty_v1"
PREREG = Path("results/high_volatility_cross_structure_action_vote_preregistration_2026-08-16.json")
PREREG_SHA = "340627ccd4928acb6297f0959fd001cc07066e05e00bfde98db88ae0cb0c550e"
SUPPORT = Path("results/high_volatility_cross_structure_action_vote_support_2026-08-16.json")
SUPPORT_SHA = "881b16adab6c0b646cbeca6cf3341a3921b8b6792a7790828b1ad65eb85fa0df"
CLOCK = {
    "path": "data/high_volatility_cross_structure_action_vote_clocks_2023_2026.csv.gz",
    "sha256": "4a30d75dbb9c0efe73f2ac929299a7413e97c8812c2b404d22f8328f26bf657d",
    "rows": 117,
}
OUTPUT = Path("results/high_volatility_cross_structure_action_vote_gross9_novelty_2026-08-16.json")

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
    if sha(PREREG) != PREREG_SHA:
        raise RuntimeError("HVCAV-8 preregistration hash drift")
    registration = load_manifest(PREREG)
    expected_limits = {
        "exact_entry_jaccard_max": LIMITS["exact_entry_jaccard"],
        "candidate_near_6h_share_max": LIMITS["one_to_one_6h_max_matched_share"],
        "occupied_5m_jaccard_max": LIMITS["occupied_5m_bar_jaccard"],
        "absolute_signed_exposure_pearson_max": LIMITS["absolute_signed_exposure_pearson"],
        "must_pass_before_economics": True,
    }
    if (
        registration.get("policy_id") != POLICY
        or registration.get("candidate_family") != [POLICY]
        or registration.get("candidate_family_size") != 1
        or registration.get("single_candidate_only") is not True
        or registration.get("gross9_novelty_gates") != expected_limits
    ):
        raise RuntimeError("HVCAV-8 preregistration state drift")
    if sha(SUPPORT) != SUPPORT_SHA:
        raise RuntimeError("HVCAV-8 source-support artifact hash drift")
    support = load_manifest(SUPPORT)
    if (
        support.get("policy_id") != POLICY
        or support.get("preregistration", {}).get("sha256") != PREREG_SHA
        or support.get("support_passed") is not True
        or support.get("advance_to_gross9_novelty") is not True
        or support.get("advance_to_economic_outcomes") is not False
        or support.get("clock") != CLOCK
        or sha(CLOCK["path"]) != CLOCK["sha256"]
    ):
        raise RuntimeError("HVCAV-8 source eligibility drift")
    sealed = (
        "action_vote_postentry_returns_or_pnl_opened",
        "entry_exit_prices_opened",
        "returns_opened",
        "funding_opened",
        "pnl_opened",
        "gross9_comparator_rows_opened",
    )
    if any(support.get(field) is not False for field in sealed):
        raise RuntimeError("HVCAV-8 economic evidence boundary drift")
    return registration, support, hvsof._load_gross9_authority()


def run(output: str | Path = OUTPUT) -> dict[str, Any]:
    registration, support, manifest = load_frozen_controls()
    candidate = metric.load_clock(Path(CLOCK["path"]), label=POLICY)
    if len(candidate) != CLOCK["rows"]:
        raise RuntimeError("HVCAV-8 clock row-count drift")
    comparators = {}
    for sleeve in gross9.EXPECTED_WEIGHTS:
        record = manifest["clocks"][sleeve]
        if sha(record["path"]) != record["sha256"]:
            raise RuntimeError(f"Gross9 clock hash drift: {sleeve}")
        clock = metric.load_clock(Path(record["path"]), label=f"Gross9 {sleeve}")
        if len(clock) != record["rows"] or len(clock) != sum(gross9.EXPECTED_COUNTS[sleeve].values()):
            raise RuntimeError(f"Gross9 count drift: {sleeve}")
        comparators[sleeve] = clock
    sleeves = {sleeve: evaluate_pair(candidate, clock) for sleeve, clock in comparators.items()}
    passed = all(result["passed"] for result in sleeves.values())
    core = {
        "protocol_version": PROTOCOL,
        "policy_id": POLICY,
        "preregistration": {"path": PREREG.as_posix(), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_support": {"path": SUPPORT.as_posix(), "sha256": SUPPORT_SHA, "manifest_hash": support["manifest_hash"]},
        "candidate_clock": {**CLOCK},
        "gross9_structural_clocks": {
            "path": gross9.DEFAULT_MANIFEST.as_posix(),
            "sha256": sha(gross9.DEFAULT_MANIFEST),
            "manifest_hash": manifest["manifest_hash"],
            "authority_sha256": gross9.ANCHOR_SHA256,
            "complete_roster": list(gross9.EXPECTED_WEIGHTS),
        },
        "evidence_boundary": {
            "candidate_clock_rows_opened": CLOCK["rows"],
            "gross9_structural_clock_rows_opened": sum(record["rows"] for record in manifest["clocks"].values()),
            "price_or_return_rows_opened": 0,
            "funding_rows_opened": 0,
            "economic_outcome_rows_opened": 0,
            "portfolio_return_or_pnl_metrics_computed": False,
            "outcomes_opened": False,
        },
        "limits": LIMITS,
        "gross9_sleeves": sleeves,
        "source_support_passed": True,
        "every_gross9_sleeve_passed": passed,
        "gross9_novelty_status": "passed" if passed else "failed",
        "advance_to_economic_outcomes": passed,
        "failure_action": None if passed else "reject HVCAV-8 unchanged before economics",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({"every_gross9_sleeve_passed": report["every_gross9_sleeve_passed"], "gross9_novelty_status": report["gross9_novelty_status"]}))
