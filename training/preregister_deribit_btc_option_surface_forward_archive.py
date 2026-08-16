"""Preregister the forward-only Deribit BTC option-surface source archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "DERIBIT-BTC-OPTION-SURFACE-FORWARD-V1"
DEFAULT_OUTPUT = Path(
    "results/deribit_btc_option_surface_forward_archive_preregistration_2026-08-16.json"
)
COLLECTOR = Path("training/collect_deribit_btc_option_surface_snapshot.py")
DIAGNOSTIC_SNAPSHOTS = (
    Path(
        "data/forward_deribit_btc_option_surface/2026-08-16/"
        "20260816T004554_026968Z.json.gz"
    ),
    Path(
        "data/forward_deribit_btc_option_surface/2026-08-16/"
        "20260816T004843_310530Z.json.gz"
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "deribit_btc_option_surface_forward_archive_preregistration_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-16",
        "purpose": "independent point-in-time source accumulation; no economic candidate is authorized",
        "source": {
            "provider": "Deribit",
            "currency": "BTC",
            "kind": "option",
            "instruments_endpoint": "https://www.deribit.com/api/v2/public/get_instruments",
            "summaries_endpoint": "https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
            "official_instruments_docs": "https://docs.deribit.com/api-reference/market-data/public-get_instruments",
            "official_summaries_docs": "https://docs.deribit.com/api-reference/market-data/public-get_book_summary_by_currency",
        },
        "collector": {
            "path": str(COLLECTOR),
            "sha256": sha256(COLLECTOR),
            "protocol_version": "deribit_btc_option_surface_forward_snapshot_v1",
        },
        "collection_contract": {
            "eligible_start": "2026-08-16T08:02:00Z",
            "cadence": "exact 00:02, 08:02, and 16:02 UTC every day",
            "timer_expression": "*-*-* 00,08,16:02:00 UTC",
            "feature_available_time": "later local UTC receipt time of active-instrument and book-summary responses",
            "join": "exact instrument_name intersection",
            "minimum_joined_share": 0.95,
            "duplicate_or_nonincreasing_time": "hard failure with no snapshot",
            "archive": "immutable gzip JSON snapshots chained by predecessor file SHA256 and manifest hash",
            "historical_backfill": "forbidden",
            "missed_run_imputation": "forbidden",
            "provider_substitution": "forbidden",
            "failed_or_partial_response": "no snapshot; wait for next scheduled observation",
        },
        "diagnostic_snapshots": [
            {
                "path": str(path),
                "sha256": sha256(path),
                "economic_or_candidate_eligibility": False,
            }
            for path in DIAGNOSTIC_SNAPSHOTS
        ],
        "research_boundary": {
            "diagnostic_snapshot_rows_opened": 1636,
            "eligible_forward_snapshot_rows_opened": 0,
            "option_surface_candidate_incidence_opened": False,
            "post_snapshot_btc_returns_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "economic_candidate_authorized": False,
            "2024_test_reusable_as_confirmatory_oos": False,
            "future_candidate_requirement": "separate singleton preregistration before incidence or outcomes",
        },
        "unblock_rule": (
            "accumulate sufficient eligible untouched forward history, then preregister exactly one physical "
            "option-surface state machine and new train/test/eval/final split before opening its incidence or outcomes"
        ),
        "stopping_rule": (
            "No alpha claim, source-support pass, Gross9 pass, or economic evaluation may be inferred from archive "
            "collection alone. Collector/schema/cadence changes terminate this exact source protocol."
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("forward archive preregistration manifest drift")
    if value.get("policy_id") != POLICY_ID:
        raise RuntimeError("forward archive policy drift")
    if sha256(COLLECTOR) != value["collector"]["sha256"]:
        raise RuntimeError("forward archive collector drift")
    for record, path in zip(value["diagnostic_snapshots"], DIAGNOSTIC_SNAPSHOTS, strict=True):
        if record["sha256"] != sha256(path):
            raise RuntimeError(f"diagnostic snapshot drift: {path}")
        if record["economic_or_candidate_eligibility"] is not False:
            raise RuntimeError("diagnostic snapshot eligibility drift")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
