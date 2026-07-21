"""Audit outcome-blind USDC mint/burn role topology.

This module is deliberately descriptive.  It may establish whether a directed
mint-recipient/burn-caller graph exists, but it must not form trading pairs or
open market outcomes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_VERSION = "usdc_role_topology_audit_v1"
AS_OF_DATE = "2026-07-21"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = Path(
    "results/ethereum_stablecoin_issuance_redemption_source_manifest_2026-07-21.json"
)
SOURCE_MANIFEST_SHA256 = (
    "8ec9ab08c413bf6f5f8170fb800b05105522d4cf1a7932943c214288701e31fe"
)
SOURCE_CSV = Path(
    "data/ethereum_stablecoin_issuance_redemption_2020_2023/"
    "ethereum_usdt_usdc_issuance_redemption_2020_2023.csv.gz"
)
SOURCE_CSV_SHA256 = "70ba3799ba84dc671051623a8d167b1731f043cf84a686b9878a67fcd52e5901"
EVALUATOR_SOURCE = Path("training/audit_usdc_role_topology.py")
DEFAULT_OUTPUT = Path("results/usdc_role_topology_audit_2026-07-21.json")
ADDRESS = re.compile(r"^0x[0-9a-f]{40}$")

OUTCOME_BOUNDARY = {
    "source_csv_rows_read": 266_362,
    "eligible_usdc_rows_read": 265_585,
    "post_2023_event_rows_read": 0,
    "comparator_clock_rows_read": 0,
    "btc_market_rows_read": 0,
    "funding_rows_read": 0,
    "future_return_rows_read": 0,
    "return_or_pnl_fields_read": 0,
    "network_calls": 0,
    "subprocess_calls": 0,
}


@dataclass(frozen=True)
class RoleEvent:
    event: str
    actor: str
    recipient: str
    amount_raw: int
    year: str


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _share(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _concentration(counter: Counter[str]) -> dict[str, Any]:
    total = sum(counter.values())
    values = sorted(counter.values(), reverse=True)
    return {
        "distinct_roles": len(counter),
        "total": total,
        "largest_role_share": _share(values[0], total) if values else 0.0,
        "top_two_role_share": _share(sum(values[:2]), total),
        "herfindahl": sum((_share(value, total)) ** 2 for value in values),
    }


def _validate_source_manifest(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("protocol_version")
        != "ethereum_stablecoin_issuance_redemption_source_v1"
    ):
        raise RuntimeError("unexpected source protocol")
    output = manifest.get("output", {})
    if output.get("sha256") != SOURCE_CSV_SHA256 or output.get("rows") != 266_362:
        raise RuntimeError("source output identity drift")
    if not manifest.get("dual_replay", {}).get("canonical_replay_equal"):
        raise RuntimeError("source lacks independent canonical replay equality")
    if not manifest.get("header_materialization", {}).get(
        "event_block_hash_cross_checked"
    ):
        raise RuntimeError("source lacks event-block header cross-check")
    if manifest.get("source_contract", {}).get("confirmation_blocks") != 64:
        raise RuntimeError("source is not N+64 causal")
    finalized = manifest.get("source_audit", {}).get("finalized_coverage", {})
    if not finalized.get("observed_finalized_block_at_least_required"):
        raise RuntimeError("source lacks finalized coverage")
    if not manifest.get("outcome_boundary", {}).get("source_only"):
        raise RuntimeError("source manifest crossed the outcome boundary")


def load_events(path: str | Path) -> tuple[list[RoleEvent], int]:
    source_rows = 0
    events: list[RoleEvent] = []
    with gzip.open(_path(path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_rows += 1
            if row["asset"] != "usdc_eth" or row["event"] not in {"mint", "burn"}:
                continue
            actor = row["indexed_address_1"]
            recipient = row["indexed_address_2"] if row["event"] == "mint" else ""
            if not ADDRESS.fullmatch(actor):
                raise RuntimeError("malformed USDC actor address")
            if row["event"] == "mint" and not ADDRESS.fullmatch(recipient):
                raise RuntimeError("malformed USDC mint recipient")
            amount_raw = int(row["amount_raw"])
            if amount_raw <= 0:
                raise RuntimeError("non-positive USDC event amount")
            events.append(
                RoleEvent(
                    event=row["event"],
                    actor=actor,
                    recipient=recipient,
                    amount_raw=amount_raw,
                    year=row["block_timestamp"][:4],
                )
            )
    return events, source_rows


def audit_events(events: Iterable[RoleEvent]) -> dict[str, Any]:
    materialized = list(events)
    mint_callers: Counter[str] = Counter()
    burn_callers: Counter[str] = Counter()
    mint_recipients: Counter[str] = Counter()
    mint_caller_amount: Counter[str] = Counter()
    burn_caller_amount: Counter[str] = Counter()
    mint_recipient_amount: Counter[str] = Counter()

    for event in materialized:
        if event.event == "mint":
            mint_callers[event.actor] += 1
            mint_recipients[event.recipient] += 1
            mint_caller_amount[event.actor] += event.amount_raw
            mint_recipient_amount[event.recipient] += event.amount_raw
        elif event.event == "burn":
            burn_callers[event.actor] += 1
            burn_caller_amount[event.actor] += event.amount_raw
        else:
            raise RuntimeError(f"unexpected role event: {event.event}")

    caller_overlap = set(mint_callers) & set(burn_callers)
    recipient_burner = set(mint_recipients) & set(burn_callers)
    recipient_minter = set(mint_recipients) & set(mint_callers)
    all_three = set(mint_callers) & set(burn_callers) & set(mint_recipients)

    eligible_mint_events: Counter[str] = Counter()
    eligible_burn_events: Counter[str] = Counter()
    eligible_mint_amount: Counter[str] = Counter()
    eligible_burn_amount: Counter[str] = Counter()
    eligible_mint_years: Counter[str] = Counter()
    eligible_burn_years: Counter[str] = Counter()
    eligible_mint_callers: set[str] = set()
    directed_edges: set[tuple[str, str]] = set()

    for event in materialized:
        if event.event == "mint" and event.recipient in recipient_burner:
            eligible_mint_events[event.recipient] += 1
            eligible_mint_amount[event.recipient] += event.amount_raw
            eligible_mint_years[event.year] += 1
            eligible_mint_callers.add(event.actor)
            directed_edges.add((event.actor, event.recipient))
        elif event.event == "burn" and event.actor in recipient_burner:
            eligible_burn_events[event.actor] += 1
            eligible_burn_amount[event.actor] += event.amount_raw
            eligible_burn_years[event.year] += 1

    all_mint_events = sum(mint_callers.values())
    all_burn_events = sum(burn_callers.values())
    all_mint_amount = sum(mint_caller_amount.values())
    all_burn_amount = sum(burn_caller_amount.values())

    return {
        "role_cardinality": {
            "mint_callers": len(mint_callers),
            "burn_callers": len(burn_callers),
            "mint_recipients": len(mint_recipients),
        },
        "role_overlap": {
            "mint_caller_and_burn_caller": len(caller_overlap),
            "mint_recipient_and_burn_caller": len(recipient_burner),
            "mint_recipient_and_mint_caller": len(recipient_minter),
            "all_three": len(all_three),
        },
        "role_concentration": {
            "mint_caller_events": _concentration(mint_callers),
            "burn_caller_events": _concentration(burn_callers),
            "mint_recipient_events": _concentration(mint_recipients),
            "mint_caller_amount": _concentration(mint_caller_amount),
            "burn_caller_amount": _concentration(burn_caller_amount),
            "mint_recipient_amount": _concentration(mint_recipient_amount),
        },
        "directed_recipient_burner_topology": {
            "recipient_burner_roles": len(recipient_burner),
            "distinct_mint_callers_into_roles": len(eligible_mint_callers),
            "distinct_minter_recipient_edges": len(directed_edges),
            "mint_leg_events": sum(eligible_mint_events.values()),
            "mint_leg_event_share": _share(
                sum(eligible_mint_events.values()), all_mint_events
            ),
            "mint_leg_amount_share": _share(
                sum(eligible_mint_amount.values()), all_mint_amount
            ),
            "burn_leg_events": sum(eligible_burn_events.values()),
            "burn_leg_event_share": _share(
                sum(eligible_burn_events.values()), all_burn_events
            ),
            "burn_leg_amount_share": _share(
                sum(eligible_burn_amount.values()), all_burn_amount
            ),
            "mint_leg_role_concentration": _concentration(eligible_mint_events),
            "burn_leg_role_concentration": _concentration(eligible_burn_events),
            "mint_leg_year_counts": dict(sorted(eligible_mint_years.items())),
            "burn_leg_year_counts": dict(sorted(eligible_burn_years.items())),
        },
        "interpretation": {
            "directed_graph_exists": bool(recipient_burner),
            "full_period_membership_is_descriptive_only": True,
            "future_membership_authorized_for_features": False,
            "pair_incidence_calculated": False,
            "address_identities_exported": False,
            "economic_support_established": False,
        },
    }


def build_report(
    source_csv: str | Path = SOURCE_CSV,
    source_manifest: str | Path = SOURCE_MANIFEST,
) -> dict[str, Any]:
    if sha256_file(source_csv) != SOURCE_CSV_SHA256:
        raise RuntimeError("source CSV hash drift")
    if sha256_file(source_manifest) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("source manifest file hash drift")
    with _path(source_manifest).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    _validate_source_manifest(manifest)
    events, source_rows = load_events(source_csv)
    if source_rows != OUTCOME_BOUNDARY["source_csv_rows_read"]:
        raise RuntimeError("source row count drift")
    if len(events) != OUTCOME_BOUNDARY["eligible_usdc_rows_read"]:
        raise RuntimeError("eligible USDC row count drift")

    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "as_of_date": AS_OF_DATE,
        "source": {
            "csv_path": str(SOURCE_CSV),
            "csv_sha256": SOURCE_CSV_SHA256,
            "manifest_path": str(SOURCE_MANIFEST),
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "manifest_hash": manifest["manifest_hash"],
        },
        "evaluator": {
            "path": str(EVALUATOR_SOURCE),
            "sha256": sha256_file(EVALUATOR_SOURCE),
        },
        "outcome_boundary": dict(OUTCOME_BOUNDARY),
        "audit": audit_events(events),
        "decision": {
            "source_topology_audited": True,
            "mechanism_frozen": False,
            "candidate_clock_opened": False,
            "novelty_opened": False,
            "economic_outcomes_opened": False,
            "candidate_authorized": False,
            "status": "retired_before_temporal_pairing",
            "reason": (
                "only two recipient-burner roles and greater than 99 percent "
                "eligible-leg concentration make a post-AMTR graph carveout "
                "non-independent"
            ),
            "next_action": "new independently frozen source or mechanism axis",
        },
    }
    report["manifest_hash"] = canonical_hash(report)
    return report


def write_report(report: Mapping[str, Any], output: str | Path) -> None:
    target = _path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", default=str(SOURCE_CSV))
    parser.add_argument("--source-manifest", default=str(SOURCE_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args.source_csv, args.source_manifest)
    write_report(report, args.output)
    print(json.dumps(report["audit"], indent=2, sort_keys=True))
    print(f"wrote {_path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
