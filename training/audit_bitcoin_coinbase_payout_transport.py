"""Reproduce the bounded Bitcoin coinbase-payout transport rejection."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypedDict


PROTOCOL_VERSION = "bitcoin_coinbase_payout_transport_audit_v1"
AS_OF_DATE = "2026-07-21"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DECISION = Path(
    "docs/bitcoin-coinbase-payout-topology-source-feasibility-2026-07-21.md"
)
SOURCE_DECISION_SHA256 = (
    "29b5855d89d727d4bb14ab43b311fb933a2eab34e90dcfb298bb353ac3d53f8f"
)
SOURCE_REFERENCE = Path("data/bitcoin_utxo_fee_block_stats_2020_2023.csv.gz")
SOURCE_REFERENCE_SHA256 = (
    "8d5160b59bd104819adb34752f2bd5b01e9c07d7aa108188abcd9e6d323d102f"
)
AUDITOR_SOURCE = Path("training/audit_bitcoin_coinbase_payout_transport.py")
DEFAULT_REPORT = Path(
    "results/bitcoin_coinbase_payout_transport_rejection_2026-07-21.json"
)
MEMPOOL_BASE_URL = "https://mempool.space/api"
BLOCKSTREAM_BASE_URL = "https://blockstream.info/api"
USER_AGENT = "rllm-private-research/1.0"
MEMPOOL_PAGE_SIZE = 15
PROBE_START_HEIGHT = 800_000
PROBE_END_HEIGHT = 800_019
PROBE_PAGE_CURSORS = (800_019, 800_014)
EXPECTED_MISSING_HEIGHT = 800_015
FROZEN_START_HEIGHT = 610_691
FROZEN_END_HEIGHT = 823_785
FROZEN_REFERENCE_ROWS = FROZEN_END_HEIGHT - FROZEN_START_HEIGHT + 1
FIRST_2024_TIMESTAMP = 1_704_067_200
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

REFERENCE_COLUMNS = (
    "height",
    "id",
    "previousblockhash",
    "timestamp",
    "mediantime",
    "tx_count",
    "size",
    "weight",
    "total_fees",
    "total_inputs",
    "total_outputs",
    "utxo_set_change",
)
REFERENCE_SELECTED_COLUMNS = (
    "height",
    "id",
    "previousblockhash",
    "timestamp",
)
REFERENCE_SELECTED_INDEXES = {
    REFERENCE_COLUMNS.index(column): column for column in REFERENCE_SELECTED_COLUMNS
}
MEMPOOL_REQUIRED_BLOCK_KEYS = frozenset(
    {"height", "id", "previousblockhash", "timestamp", "stale", "extras"}
)
MEMPOOL_REQUIRED_EXTRA_KEYS = frozenset({"coinbaseSignature", "coinbaseAddresses"})


class ReferenceRow(TypedDict):
    height: int
    id: str
    previousblockhash: str
    timestamp: int


Fetch = Callable[[str], Any]


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character hash")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _project_unquoted_record(raw_record: bytes) -> dict[str, str]:
    if not raw_record.endswith(b"\n"):
        raise RuntimeError("reference CSV record lacks a newline terminator")
    if b'"' in raw_record:
        raise RuntimeError("reference CSV unexpectedly requires quoted parsing")
    end = len(raw_record) - (2 if raw_record.endswith(b"\r\n") else 1)
    field_index = 0
    selected = REFERENCE_SELECTED_INDEXES.get(field_index)
    buffer = bytearray() if selected is not None else None
    projected: dict[str, str] = {}

    def finish() -> None:
        if selected is not None and buffer is not None:
            projected[selected] = buffer.decode("utf-8")

    for position in range(end):
        value = raw_record[position]
        if value == ord(","):
            finish()
            field_index += 1
            selected = REFERENCE_SELECTED_INDEXES.get(field_index)
            buffer = bytearray() if selected is not None else None
        elif value in (ord("\r"), ord("\n")):
            raise RuntimeError("reference CSV contains an embedded line ending")
        elif buffer is not None:
            buffer.append(value)
    finish()
    if field_index + 1 != len(REFERENCE_COLUMNS):
        raise RuntimeError("reference CSV data-column count drift")
    return projected


def load_reference() -> dict[int, ReferenceRow]:
    if sha256_file(SOURCE_REFERENCE) != SOURCE_REFERENCE_SHA256:
        raise RuntimeError("frozen Bitcoin best-chain reference hash drift")
    selected: dict[int, ReferenceRow] = {}
    physical_rows = 0
    prior: ReferenceRow | None = None
    with gzip.open(_path(SOURCE_REFERENCE), "rb") as handle:
        header = tuple(handle.readline().decode("ascii").rstrip("\r\n").split(","))
        if header != REFERENCE_COLUMNS:
            raise RuntimeError("Bitcoin best-chain reference schema drift")
        for raw_record in handle:
            physical_rows += 1
            raw = _project_unquoted_record(raw_record)
            row: ReferenceRow = {
                "height": int(raw["height"]),
                "id": _validate_hash(raw["id"], "reference id"),
                "previousblockhash": _validate_hash(
                    raw["previousblockhash"], "reference previousblockhash"
                ),
                "timestamp": int(raw["timestamp"]),
            }
            if row["height"] != FROZEN_START_HEIGHT + physical_rows - 1:
                raise RuntimeError("Bitcoin reference height continuity drift")
            if row["timestamp"] >= FIRST_2024_TIMESTAMP:
                raise RuntimeError("Bitcoin reference crossed the pre-2024 boundary")
            if prior is not None and row["previousblockhash"] != prior["id"]:
                raise RuntimeError("Bitcoin reference hash-chain linkage drift")
            if PROBE_START_HEIGHT <= row["height"] <= PROBE_END_HEIGHT:
                selected[row["height"]] = row
            prior = row
    if physical_rows != FROZEN_REFERENCE_ROWS:
        raise RuntimeError("Bitcoin reference physical-row count drift")
    if prior is None or prior["height"] != FROZEN_END_HEIGHT:
        raise RuntimeError("Bitcoin reference terminal height drift")
    if sorted(selected) != list(range(PROBE_START_HEIGHT, PROBE_END_HEIGHT + 1)):
        raise RuntimeError("Bitcoin reference did not cover the exact probe range")
    return selected


def parse_standard_output_asm(asm: str) -> tuple[str, bytes]:
    if not isinstance(asm, str) or not asm or asm.strip() != asm:
        raise ValueError("coinbaseSignature must be a non-empty canonical ASM string")
    tokens = asm.split(" ")

    def data(index: int, length: int) -> bytes:
        value = tokens[index]
        if len(value) != length * 2 or re.fullmatch(r"[0-9a-f]+", value) is None:
            raise ValueError("coinbaseSignature contains malformed pushed bytes")
        return bytes.fromhex(value)

    if tokens[:3] == ["OP_DUP", "OP_HASH160", "OP_PUSHBYTES_20"] and tokens[4:] == [
        "OP_EQUALVERIFY",
        "OP_CHECKSIG",
    ]:
        return "p2pkh", bytes.fromhex("76a914") + data(3, 20) + bytes.fromhex("88ac")
    if tokens[:2] == ["OP_HASH160", "OP_PUSHBYTES_20"] and tokens[3:] == ["OP_EQUAL"]:
        return "p2sh", bytes.fromhex("a914") + data(2, 20) + bytes.fromhex("87")
    if tokens[:2] == ["OP_0", "OP_PUSHBYTES_20"] and len(tokens) == 3:
        return "p2wpkh", bytes.fromhex("0014") + data(2, 20)
    if tokens[:2] == ["OP_0", "OP_PUSHBYTES_32"] and len(tokens) == 3:
        return "p2wsh", bytes.fromhex("0020") + data(2, 32)
    if (
        tokens[:2]
        in (
            ["OP_PUSHNUM_1", "OP_PUSHBYTES_32"],
            ["OP_1", "OP_PUSHBYTES_32"],
        )
        and len(tokens) == 3
    ):
        return "p2tr", bytes.fromhex("5120") + data(2, 32)
    if tokens[0:1] == ["OP_PUSHBYTES_33"] and tokens[2:] == ["OP_CHECKSIG"]:
        return "p2pk_compressed", bytes.fromhex("21") + data(1, 33) + bytes.fromhex(
            "ac"
        )
    if tokens[0:1] == ["OP_PUSHBYTES_65"] and tokens[2:] == ["OP_CHECKSIG"]:
        return "p2pk_uncompressed", bytes.fromhex("41") + data(1, 65) + bytes.fromhex(
            "ac"
        )
    raise ValueError("unsupported first coinbase output script ASM")


def address_set_fingerprint(addresses: Any) -> tuple[int, str]:
    if not isinstance(addresses, list) or not addresses:
        raise ValueError("coinbase address set must be a non-empty list")
    if any(
        not isinstance(value, str) or not value or not value.isascii()
        for value in addresses
    ):
        raise ValueError("coinbase address set contains an invalid address")
    ordered = sorted(set(addresses))
    payload = b"coinbase-address-set-v1\0" + b"\0".join(
        address.encode("ascii") for address in ordered
    )
    return len(ordered), hashlib.sha256(payload).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"HTTP response used non-standard JSON constant {value}")


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"HTTP response contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _decode_json(raw: bytes) -> Any:
    return json.loads(
        raw.decode("utf-8"),
        parse_float=Decimal,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_unique_object,
    )


def _http_json(url: str, *, timeout: float = 30.0, retries: int = 8) -> Any:
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise RuntimeError("HTTP JSON response exceeded two megabytes")
                return _decode_json(raw)
        except urllib.error.HTTPError as exc:
            if not (exc.code == 429 or 500 <= exc.code <= 599) or attempt >= retries:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt >= retries:
                raise
        time.sleep(min(60.0, 2.0**attempt))
    raise AssertionError("unreachable HTTP retry loop")


def _project_block(raw: Any, references: Mapping[int, ReferenceRow]) -> dict[str, Any]:
    if not isinstance(raw, dict) or not MEMPOOL_REQUIRED_BLOCK_KEYS <= raw.keys():
        raise ValueError("Mempool probe block is malformed")
    height = _positive_int(raw["height"], "probe height")
    reference = references.get(height)
    if reference is None:
        raise RuntimeError("Mempool probe opened an unbound height")
    block_hash = _validate_hash(raw["id"], "probe id")
    previous = _validate_hash(raw["previousblockhash"], "probe previousblockhash")
    timestamp = _positive_int(raw["timestamp"], "probe timestamp")
    stale = raw["stale"]
    if not (stale is False or (type(stale) is int and stale == 0)):
        raise RuntimeError("Mempool probe returned a stale block")
    if (
        block_hash != reference["id"]
        or previous != reference["previousblockhash"]
        or timestamp != reference["timestamp"]
    ):
        raise RuntimeError("Mempool probe disagrees with bound best chain")
    extras = raw["extras"]
    if not isinstance(extras, dict) or not MEMPOOL_REQUIRED_EXTRA_KEYS <= extras.keys():
        raise ValueError("Mempool probe lacks topology field names")
    signature = extras["coinbaseSignature"]
    addresses = extras["coinbaseAddresses"]
    if signature is None and addresses == []:
        return {
            "height": height,
            "id": block_hash,
            "topology_complete": False,
            "script_type": None,
            "script_sha256": None,
            "address_count": 0,
            "address_set_sha256": None,
        }
    if not isinstance(signature, str) or not isinstance(addresses, list):
        raise ValueError("Mempool probe has partially missing topology values")
    script_type, raw_script = parse_standard_output_asm(signature)
    address_count, address_hash = address_set_fingerprint(addresses)
    return {
        "height": height,
        "id": block_hash,
        "topology_complete": True,
        "script_type": script_type,
        "script_sha256": hashlib.sha256(raw_script).hexdigest(),
        "address_count": address_count,
        "address_set_sha256": address_hash,
    }


def _normalise_coinbase_transaction(
    payload: Any, block_hash: str
) -> tuple[str, int, str]:
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise RuntimeError("coinbase transaction page is empty or malformed")
    transaction = payload[0]
    status = transaction.get("status")
    if not isinstance(status, dict) or status.get("block_hash") != block_hash:
        raise RuntimeError("coinbase transaction block identity drift")
    inputs = transaction.get("vin")
    if (
        not isinstance(inputs, list)
        or len(inputs) != 1
        or not isinstance(inputs[0], dict)
        or inputs[0].get("is_coinbase") is not True
    ):
        raise RuntimeError("first transaction is not coinbase")
    outputs = transaction.get("vout")
    if not isinstance(outputs, list) or not outputs or not isinstance(outputs[0], dict):
        raise RuntimeError("coinbase outputs are missing")
    first_script = outputs[0].get("scriptpubkey")
    if (
        not isinstance(first_script, str)
        or re.fullmatch(r"[0-9a-f]+", first_script) is None
    ):
        raise ValueError("first coinbase output script is not canonical hex")
    addresses = sorted(
        {
            output["scriptpubkey_address"]
            for output in outputs
            if isinstance(output, dict)
            and isinstance(output.get("scriptpubkey_address"), str)
        }
    )
    count, address_hash = address_set_fingerprint(addresses)
    return hashlib.sha256(bytes.fromhex(first_script)).hexdigest(), count, address_hash


def build_report(fetch: Fetch | None = None) -> dict[str, Any]:
    if sha256_file(SOURCE_DECISION) != SOURCE_DECISION_SHA256:
        raise RuntimeError("coinbase topology source decision hash drift")
    references = load_reference()
    resolved_fetch = fetch or _http_json
    projected: dict[int, dict[str, Any]] = {}
    network_calls = 0
    for cursor in PROBE_PAGE_CURSORS:
        payload = resolved_fetch(f"{MEMPOOL_BASE_URL}/v1/blocks/{cursor}")
        network_calls += 1
        if not isinstance(payload, list) or len(payload) != MEMPOOL_PAGE_SIZE:
            raise RuntimeError("Mempool probe page-size drift")
        for raw in payload:
            row = _project_block(raw, references)
            if not PROBE_START_HEIGHT <= row["height"] <= PROBE_END_HEIGHT:
                raise RuntimeError("Mempool probe opened a height outside its bound")
            prior = projected.get(row["height"])
            if prior is not None and prior != row:
                raise RuntimeError("overlapping Mempool probe pages disagree")
            projected[row["height"]] = row
    if sorted(projected) != list(range(PROBE_START_HEIGHT, PROBE_END_HEIGHT + 1)):
        raise RuntimeError("Mempool probe did not cover its exact height range")

    missing = [row for row in projected.values() if not row["topology_complete"]]
    fallback_checks: list[dict[str, Any]] = []
    for row in missing:
        block_hash = row["id"]
        blockstream = _normalise_coinbase_transaction(
            resolved_fetch(f"{BLOCKSTREAM_BASE_URL}/block/{block_hash}/txs/0"),
            block_hash,
        )
        mempool_tx = _normalise_coinbase_transaction(
            resolved_fetch(f"{MEMPOOL_BASE_URL}/block/{block_hash}/txs/0"),
            block_hash,
        )
        network_calls += 2
        if blockstream != mempool_tx:
            raise RuntimeError("transaction transports disagree on missing summary row")
        fallback_checks.append(
            {
                "height": row["height"],
                "id": block_hash,
                "transaction_endpoints_agree": True,
                "first_output_script_present": bool(blockstream[0]),
                "address_bearing_output_count": blockstream[1],
                "clear_script_or_address_persisted": False,
            }
        )

    missing_heights = sorted(row["height"] for row in missing)
    checks = {
        "exact_probe_height_coverage": len(projected) == 20,
        "all_block_identities_match_bound_best_chain": True,
        "all_summary_topology_fields_nonempty": not missing,
        "expected_missing_height_reproduced": missing_heights
        == [EXPECTED_MISSING_HEIGHT],
        "missing_summary_is_transport_specific": bool(fallback_checks)
        and all(
            check["transaction_endpoints_agree"]
            and check["first_output_script_present"]
            and check["address_bearing_output_count"] >= 1
            for check in fallback_checks
        ),
    }
    rejection_reproduced = (
        checks["expected_missing_height_reproduced"]
        and checks["missing_summary_is_transport_specific"]
    )
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "as_of_date": AS_OF_DATE,
        "source_decision": {
            "path": str(SOURCE_DECISION),
            "sha256": SOURCE_DECISION_SHA256,
        },
        "source_reference": {
            "path": str(SOURCE_REFERENCE),
            "sha256": SOURCE_REFERENCE_SHA256,
            "physical_rows": FROZEN_REFERENCE_ROWS,
        },
        "auditor": {
            "path": str(AUDITOR_SOURCE),
            "sha256": sha256_file(AUDITOR_SOURCE),
        },
        "probe": {
            "start_height": PROBE_START_HEIGHT,
            "end_height": PROBE_END_HEIGHT,
            "unique_blocks": len(projected),
            "summary_complete_blocks": len(projected) - len(missing),
            "summary_missing_blocks": len(missing),
            "missing_heights": missing_heights,
            "missing_block_checks": fallback_checks,
        },
        "checks": checks,
        "outcome_boundary": {
            "artifact_network_calls": network_calls,
            "prefreeze_manual_source_calls_disclosed_minimum": 9,
            "prefreeze_clear_coinbase_addresses_printed": 1,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_source_rows_read": 0,
            "candidate_signal_rows_created": 0,
            "economic_outcomes_opened": False,
        },
        "decision": {
            "status": (
                "retired_before_full_source_build"
                if rejection_reproduced
                else "audit_not_reproduced"
            ),
            "rejection_reproduced": rejection_reproduced,
            "source_contract_passed": False,
            "full_source_build_authorized": False,
            "fallback_repair_authorized": False,
            "candidate_authorized": False,
            "next_action": "new independent source axis",
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def write_report(report: Mapping[str, Any], output: str | Path) -> None:
    target = _path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report()
    write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
