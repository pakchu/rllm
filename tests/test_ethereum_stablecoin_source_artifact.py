from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE = Path(
    "data/ethereum_stablecoin_issuance_redemption_2020_2023/"
    "ethereum_usdt_usdc_issuance_redemption_2020_2023.csv.gz"
)
MANIFEST = Path(
    "results/ethereum_stablecoin_issuance_redemption_source_manifest_2026-07-21.json"
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


def test_source_artifact_matches_hash_bound_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    frozen_hash = manifest.pop("manifest_hash")
    assert _canonical_hash(manifest) == frozen_hash
    assert manifest["output"]["path"] == str(SOURCE)
    assert SOURCE.stat().st_size == manifest["output"]["bytes"]
    assert _sha256(SOURCE) == manifest["output"]["sha256"]
    assert manifest["dual_replay"]["canonical_replay_equal"] is True
    assert len(set(manifest["dual_replay"]["canonical_log_hashes"])) == 1
    assert manifest["header_materialization"] == {
        "event_block_hash_cross_checked": True,
        "provider_url_embedded": False,
        "transport_independent_from_primary_logs": True,
    }
    serialized = json.dumps(manifest, sort_keys=True)
    assert "rpc.mevblocker.io" not in serialized
    assert "tenderly.co" not in serialized
    assert "publicnode.com" not in serialized


def test_source_artifact_preserves_causal_event_contract() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    identities: set[tuple[str, str, int]] = set()
    event_counts: Counter[str] = Counter()
    year_counts: Counter[str] = Counter()
    previous_key: tuple[int, int, int, str, str] | None = None
    rows = 0
    with gzip.open(SOURCE, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == manifest["output"]["columns"]
        for row in reader:
            rows += 1
            block_number = int(row["block_number"])
            transaction_index = int(row["transaction_index"])
            log_index = int(row["log_index"])
            identity = (row["block_hash"], row["transaction_hash"], log_index)
            assert identity not in identities
            identities.add(identity)
            assert int(row["confirmation_block_number"]) == block_number + 64
            assert row["available_at"] > row["block_timestamp"]
            assert row["event"] != "deprecate"
            key = (
                block_number,
                transaction_index,
                log_index,
                row["asset"],
                row["event"],
            )
            if previous_key is not None:
                assert key >= previous_key
            previous_key = key
            event_counts[f"{row['asset']}:{row['event']}"] += 1
            year_counts[row["block_timestamp"][:4]] += 1

    assert rows == manifest["output"]["rows"]
    assert rows == manifest["source_audit"]["materialized_rows"]
    assert dict(sorted(event_counts.items())) == manifest["event_counts"]
    assert dict(sorted(year_counts.items())) == manifest["year_counts"]
    assert manifest["outcome_boundary"] == {
        "btc_market_rows_read": 0,
        "funding_rows_read": 0,
        "future_return_rows_read": 0,
        "pnl_cagr_mdd_opened": False,
        "post_2023_confirmation_headers_may_be_read": True,
        "post_2023_contract_event_rows_read": 0,
        "source_only": True,
    }
