from __future__ import annotations

import json
from typing import Any

import pytest

from training import audit_bitcoin_coinbase_payout_transport as audit


P2PKH_HASH = "11" * 20
P2PKH_ASM = f"OP_DUP OP_HASH160 OP_PUSHBYTES_20 {P2PKH_HASH} OP_EQUALVERIFY OP_CHECKSIG"
P2PKH_RAW = "76a914" + P2PKH_HASH + "88ac"
ADDRESS = "bc1qtransportcheck"


def _hash(height: int) -> str:
    return f"{height:064x}"


def _references() -> dict[int, audit.ReferenceRow]:
    return {
        height: {
            "height": height,
            "id": _hash(height),
            "previousblockhash": _hash(height - 1),
            "timestamp": 1_600_000_000 + height,
        }
        for height in range(audit.PROBE_START_HEIGHT, audit.PROBE_END_HEIGHT + 1)
    }


def _fetch(
    references: dict[int, audit.ReferenceRow], missing_height: int | None
) -> audit.Fetch:
    def fetch(url: str) -> Any:
        if "/v1/blocks/" in url:
            cursor = int(url.rsplit("/", 1)[1])
            rows = []
            for height in range(cursor, cursor - audit.MEMPOOL_PAGE_SIZE, -1):
                extras: dict[str, Any] = {
                    "coinbaseSignature": P2PKH_ASM,
                    "coinbaseAddresses": [ADDRESS],
                }
                if height == missing_height:
                    extras = {"coinbaseSignature": None, "coinbaseAddresses": []}
                rows.append(
                    {
                        **references[height],
                        "stale": 0,
                        "extras": extras,
                    }
                )
            return rows
        block_hash = url.split("/block/", 1)[1].split("/", 1)[0]
        return [
            {
                "status": {"block_hash": block_hash},
                "vin": [{"is_coinbase": True}],
                "vout": [
                    {
                        "scriptpubkey": P2PKH_RAW,
                        "scriptpubkey_address": ADDRESS,
                    }
                ],
            }
        ]

    return fetch


def test_transport_audit_retires_incomplete_summary_before_full_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references = _references()
    monkeypatch.setattr(audit, "load_reference", lambda: references)
    report = audit.build_report(_fetch(references, audit.EXPECTED_MISSING_HEIGHT))

    assert report["probe"]["unique_blocks"] == 20
    assert report["probe"]["missing_heights"] == [800_015]
    assert report["checks"]["missing_summary_is_transport_specific"] is True
    assert report["decision"]["status"] == "retired_before_full_source_build"
    assert report["decision"]["rejection_reproduced"] is True
    assert report["decision"]["fallback_repair_authorized"] is False
    assert report["decision"]["full_source_build_authorized"] is False
    assert report["outcome_boundary"]["artifact_network_calls"] == 4
    encoded = json.dumps(report)
    assert ADDRESS not in encoded
    assert P2PKH_RAW not in encoded


def test_complete_live_probe_cannot_authorize_full_source_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references = _references()
    monkeypatch.setattr(audit, "load_reference", lambda: references)
    report = audit.build_report(_fetch(references, None))
    assert report["checks"]["all_summary_topology_fields_nonempty"] is True
    assert report["checks"]["expected_missing_height_reproduced"] is False
    assert report["checks"]["missing_summary_is_transport_specific"] is False
    assert report["decision"]["status"] == "audit_not_reproduced"
    assert report["decision"]["rejection_reproduced"] is False
    assert report["decision"]["source_contract_passed"] is False
    assert report["decision"]["full_source_build_authorized"] is False


def test_wrong_missing_height_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    references = _references()
    monkeypatch.setattr(audit, "load_reference", lambda: references)
    report = audit.build_report(_fetch(references, 800_016))
    assert report["probe"]["missing_heights"] == [800_016]
    assert report["checks"]["expected_missing_height_reproduced"] is False
    assert report["checks"]["missing_summary_is_transport_specific"] is True
    assert report["decision"]["status"] == "audit_not_reproduced"
    assert report["decision"]["full_source_build_authorized"] is False
