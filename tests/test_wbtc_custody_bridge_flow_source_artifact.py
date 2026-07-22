from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


SOURCE = Path(
    "data/wbtc_custody_bridge_flow_2020_2023/"
    "wbtc_mint_burn_2020_2023.csv.gz"
)
MANIFEST = Path(
    "results/wbtc_custody_bridge_flow_source_manifest_2026-07-23.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_source_artifact_matches_hash_bound_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    frozen_hash = manifest.pop("manifest_hash")
    assert _canonical_hash(manifest) == frozen_hash
    assert manifest["output"]["path"] == str(SOURCE)
    assert SOURCE.stat().st_size == manifest["output"]["bytes"]
    assert _sha256(SOURCE) == manifest["output"]["sha256"]
    assert manifest["dual_replay"] == {
        "canonical_log_hashes": [
            "b8bcb672126e2668ef09a706eb68c6fed0e8a1ac19b31cdbc7a68ba40ed1245e",
            "b8bcb672126e2668ef09a706eb68c6fed0e8a1ac19b31cdbc7a68ba40ed1245e",
        ],
        "canonical_replay_equal": True,
        "independent_transport_count": 2,
        "provider_urls_embedded": False,
    }
    assert manifest["source_audit"]["receipt_pairing"] == {
        "canonical_pair_hash": (
            "3718a1d1b21ba572f3d58a0f76a318eb059d7e0808e67bd3b627aed9390f5b5c"
        ),
        "semantic_events_verified": 993,
        "unique_receipts": 993,
        "zero_transfer_pairs_verified": 993,
    }
    assert manifest["source_audit"]["header_materialization"] == {
        "event_block_hash_cross_checked": True,
        "finalized_tag_checked": True,
        "observed_finalized_block_at_least_required": True,
        "required_through_block": 18886189,
    }
    serialized = json.dumps(manifest, sort_keys=True)
    for provider_host in (
        "eth.drpc.org",
        "rpc.mevblocker.io",
        "ethereum-rpc.publicnode.com",
    ):
        assert provider_host not in serialized


def test_source_artifact_preserves_causal_event_contract() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    identities: set[tuple[str, str, int]] = set()
    event_counts: Counter[str] = Counter()
    year_event_counts: Counter[tuple[str, str]] = Counter()
    previous_key: tuple[int, int, int] | None = None
    rows = 0

    with gzip.open(SOURCE, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == manifest["output"]["columns"]
        for row in reader:
            rows += 1
            block_number = int(row["block_number"])
            transaction_index = int(row["transaction_index"])
            semantic_log_index = int(row["semantic_log_index"])
            identity = (
                row["block_hash"],
                row["transaction_hash"],
                semantic_log_index,
            )
            assert identity not in identities
            identities.add(identity)
            assert row["asset"] == "wbtc_eth"
            assert row["contract_address"] == (
                "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"
            )
            assert row["event"] in {"mint", "burn"}
            assert int(row["event_sign"]) == (1 if row["event"] == "mint" else -1)
            assert int(row["amount_raw"]) > 0
            assert int(row["decimals"]) == 8
            assert int(row["companion_transfer_log_index"]) == semantic_log_index + 1
            assert int(row["confirmation_block_number"]) == block_number + 64
            assert _parse_utc(row["available_at"]) > _parse_utc(row["block_timestamp"])
            key = (block_number, transaction_index, semantic_log_index)
            if previous_key is not None:
                assert key >= previous_key
            previous_key = key
            event_counts[row["event"]] += 1
            year_event_counts[(row["block_timestamp"][:4], row["event"])] += 1

    support = manifest["source_support"]
    assert rows == manifest["output"]["rows"] == 993
    assert rows == manifest["source_audit"]["log_source"]["semantic_event_rows"]
    assert dict(sorted(event_counts.items())) == support["event_counts"]
    assert {
        year: {
            event: year_event_counts[(year, event)]
            for event in ("mint", "burn")
        }
        for year in ("2020", "2021", "2022", "2023")
    } == support["year_event_counts"]
    assert support["decision"] == "PASS_SOURCE"
    assert support["mint_and_burn_present_in_each_calendar_year"] is True
    assert manifest["outcome_boundary"] == {
        "btc_market_rows_read": 0,
        "funding_rows_read": 0,
        "future_return_rows_read": 0,
        "labels_opened": False,
        "mechanism_features_opened": False,
        "pnl_cagr_mdd_opened": False,
        "post_2023_confirmation_headers_may_be_read": True,
        "post_2023_contract_event_rows_read": 0,
        "source_only": True,
    }
