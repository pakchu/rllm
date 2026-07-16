"""Preregister the one-time support-only feasibility repair for CVTT v2."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training import preregister_cross_venue_temporal_torsion_alpha as v1


DEFAULT_OUTPUT = (
    "results/cross_venue_temporal_torsion_v2_preregistration_2026-07-16.json"
)
PARENT_SUPPORT = "results/cross_venue_temporal_torsion_support_v1_2026-07-16.json"
PARENT_SUPPORT_FILE_SHA256 = (
    "f4dcb224fce534622a31c2139ebcf356e42dea50cd80ec04130f8e1fd450a8b6"
)
PARENT_SUPPORT_MANIFEST_HASH = (
    "bb4b3bfc6dd4c127c4429c14c0dc4bf21bd8f817ed768d0793051acd88dbed00"
)
MINIMUM_CLEAN_CALENDAR_BARS = 2_016
MINIMUM_PRIOR_ROUTE_EVENTS = 256

Policy = v1.Policy
policy_grid = v1.policy_grid
canonical_hash = v1.canonical_hash


def build_manifest() -> dict[str, Any]:
    base = v1.build_manifest()
    core = {
        key: value
        for key, value in base.items()
        if key not in {"manifest_hash", "created_at"}
    }
    core["protocol_version"] = "cross_venue_temporal_torsion_v2"
    feature = core["feature_contract"]
    feature["route_threshold"] = (
        "route score at or above its strictly-prior rolling 30-day 95th "
        "percentile over clean directionally-confirmed crossed-clock bars; "
        "requires at least 2,016 clean prior calendar bars and 256 eligible "
        "prior route events in the window"
    )
    feature.pop("rolling_minimum_bars")
    feature["rolling_minimum_clean_calendar_bars"] = MINIMUM_CLEAN_CALENDAR_BARS
    feature["rolling_minimum_prior_route_events"] = MINIMUM_PRIOR_ROUTE_EVENTS
    core["support_freeze_before_returns"][
        "monthly_missing_or_quarantined_fraction_max"
    ] = 0.05
    core["support_only_feasibility_repair"] = {
        "parent_v1_support": PARENT_SUPPORT,
        "parent_v1_support_file_sha256": PARENT_SUPPORT_FILE_SHA256,
        "parent_v1_support_manifest_hash": PARENT_SUPPORT_MANIFEST_HASH,
        "parent_v1_outcomes_opened": False,
        "evidence_seen": {
            "spot_preload_um_echo_confirmed_rows": 29_176,
            "spot_preload_um_echo_max_prior_30d_events": 1_028,
            "um_preload_spot_echo_confirmed_rows": 29_701,
            "um_preload_spot_echo_max_prior_30d_events": 979,
            "global_quarantine_fraction": 0.004390967153284671,
            "maximum_monthly_quarantine_fraction": 0.033602150537634407,
        },
        "fields_changed_from_v1": [
            "protocol_version",
            "feature_contract.route_threshold",
            "feature_contract.rolling_minimum_bars replaced by separate clean-calendar and eligible-event minima",
            "support_freeze_before_returns.monthly_missing_or_quarantined_fraction_max",
            "support_only_feasibility_repair added",
        ],
        "unchanged_fields": [
            "economic hypothesis",
            "source columns and hashes",
            "route and direction rules",
            "four policies and holds",
            "execution delay, leverage, costs, funding, and strict MDD",
            "selection and 2023 holdout gates",
            "controls and orthogonality gates",
        ],
        "repair_limit": (
            "one support-only feasibility repair; no further protocol change is "
            "allowed after any trade return is opened"
        ),
    }
    return {
        **core,
        "manifest_hash": canonical_hash(core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_manifest(payload: dict[str, Any]) -> None:
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("CVTT v2 preregistration hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("CVTT v2 preregistration cannot open outcomes")
    if payload.get("policies") != [
        {"policy_id": p.policy_id, "route": p.route, "hold_bars": p.hold_bars}
        for p in policy_grid()
    ]:
        raise RuntimeError("CVTT v2 policy family differs from v1")
    repair = payload.get("support_only_feasibility_repair", {})
    if repair.get("parent_v1_outcomes_opened") is not False:
        raise RuntimeError("CVTT v2 repair boundary is invalid")


def write_manifest_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite CVTT v2 preregistration")
        return "verified_existing"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return "created"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_manifest()
    validate_manifest(payload)
    status = write_manifest_once(args.output, payload)
    print(
        json.dumps(
            {
                "output": args.output,
                "manifest_hash": payload["manifest_hash"],
                "policies": len(payload["policies"]),
                "status": status,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
