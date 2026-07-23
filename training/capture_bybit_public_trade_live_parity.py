"""Capture the frozen prospective Bybit REST/WebSocket parity window.

This is a source-transport audit.  It must not load a Binance comparator,
construct a BSEA clock, or inspect any market outcome.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gzip
import json
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

from training.probe_bybit_public_trade_sequence_source import (
    DISK_LIMIT_GIB,
    REPO_ROOT,
    canonical_hash,
    canonical_json,
    sha256_bytes,
    sha256_file,
    used_gib,
)


PROTOCOL_VERSION = "bybit_public_trade_live_parity_capture_v2"
SCRIPT_PATH = Path("training/capture_bybit_public_trade_live_parity.py")
CONTRACT_PATH = Path(
    "docs/bybit-public-trade-live-parity-capture-contract-2026-07-23.md"
)
CONTRACT_SHA256 = (
    "50cca9c3e103e8978bb260c65b103dd90361615f6e69443181350ae560622b6c"
)
SOURCE_AUDIT_PATH = Path(
    "docs/bybit-public-trade-sequence-source-audit-2026-07-23.md"
)
SOURCE_AUDIT_SHA256 = (
    "fe324cccfb0c3f66963c142b9a6c0237489420313750de873622cadb10e8c112"
)
SOURCE_RESULT_PATH = Path(
    "results/bybit_public_trade_sequence_source_feasibility_v2_2026-07-23.json"
)
SOURCE_RESULT_SHA256 = (
    "916a55f7cd957eff39e84b2ac383c2b49cb342e2012a0f8bc15c3af98b3b3cb0"
)
SOURCE_RESULT_MANIFEST_HASH = (
    "c36f46c8399692b62d202a7331c9215fc3a5684cc3b2d57ca04d7fc7c83a5f84"
)
V1_INVALIDATION_PATH = Path(
    "docs/bybit-public-trade-live-parity-capture-v1-invalidation-2026-07-23.md"
)
V1_INVALIDATION_SHA256 = (
    "d43cb2eeec60bf3866ffe740c820c7b5e8ffe495143be3304560960f039ff15a"
)
V1_INVALID_ARTIFACT_PATH = Path(
    "results/bybit_public_trade_live_parity_capture_v1_invalid_2026-07-23.json"
)
V1_INVALID_ARTIFACT_SHA256 = (
    "d7eb515b4571a767d5efebc18ea179ebdcdd0f345c425bf2840755807f32dcbf"
)

REST_ENDPOINT = "https://api.bybit.com/v5/market/recent-trade"
REST_QUERY = "category=linear&symbol=BTCUSDT&limit=1000"
REST_URL = f"{REST_ENDPOINT}?{REST_QUERY}"
WS_URL = "wss://stream.bybit.com/v5/public/linear"
WS_TOPIC = "publicTrade.BTCUSDT"
SYMBOL = "BTCUSDT"

CAPTURE_SECONDS = 600
REST_INTERVAL_SECONDS = 1
REST_LIMIT = 1000
HEARTBEAT_SECONDS = 20
HTTP_TIMEOUT_SECONDS = 10
WS_OPEN_TIMEOUT_SECONDS = 10
MAX_REST_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_WS_MESSAGE_BYTES = 16 * 1024 * 1024
MAX_CAPTURE_RAW_BYTES = 2 * 1024 * 1024 * 1024
MAX_WS_TRADES = 5_000_000
MIN_UNIQUE_WS_TRADES = 1_000
MIN_REST_SNAPSHOTS = 10
MIN_COMMON_IDS = 1_000
USER_AGENT = "rllm-bybit-live-parity/1.0"
TICK_DIRECTIONS = frozenset(
    {"PlusTick", "ZeroPlusTick", "MinusTick", "ZeroMinusTick"}
)


class ParityCaptureError(RuntimeError):
    """The frozen source-transport contract failed."""


@dataclass(frozen=True)
class NormalizedTrade:
    source_id: str
    symbol: str
    side: Literal["Buy", "Sell"]
    price: str
    size: str
    time_ms: int
    seq: int
    block_trade: bool
    rpi_trade: bool

    def comparison_fields(self) -> tuple[object, ...]:
        return (
            self.symbol,
            self.side,
            self.price,
            self.size,
            self.time_ms,
            self.seq,
            self.block_trade,
            self.rpi_trade,
        )


@dataclass(frozen=True)
class ObservedWsTrade:
    trade: NormalizedTrade
    receipt_monotonic_ns: int


@dataclass(frozen=True)
class RestSnapshot:
    ordinal: int
    request_start_monotonic_ns: int
    response_end_monotonic_ns: int
    final: bool
    trades: tuple[NormalizedTrade, ...]


@dataclass(frozen=True)
class RestTransportResponse:
    status: int
    final_url: str
    content_type: str | None
    location: str | None
    raw: bytes


@dataclass(frozen=True)
class ClockSample:
    source: str
    ordinal: int
    monotonic_ns: int
    utc_ns: int
    uncertainty_ns: int


@dataclass
class CaptureState:
    capture_day: str
    started_at_utc: str
    output_dir: Path
    ws_trades: list[ObservedWsTrade] = field(default_factory=list)
    rest_snapshots: list[RestSnapshot] = field(default_factory=list)
    ws_messages: int = 0
    ws_subscription_acks: int = 0
    ws_pongs: int = 0
    websocket_sessions: int = 0
    clock_samples: list[ClockSample] = field(default_factory=list)
    raw_bytes: int = 0
    first_trade_monotonic_ns: int | None = None
    last_trade_monotonic_ns: int | None = None
    ended_at_utc: str | None = None


FetchRest = Callable[[], RestTransportResponse]


class _AuditNoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Return the original 3xx response for audit without contacting Location."""

    def _return_original(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
    ) -> Any:
        del req, code, msg, headers
        return fp

    http_error_301 = _return_original
    http_error_302 = _return_original
    http_error_303 = _return_original
    http_error_307 = _return_original
    http_error_308 = _return_original


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _local_clock_sample(source: str, ordinal: int) -> ClockSample:
    before = time.monotonic_ns()
    utc_ns = time.time_ns()
    after = time.monotonic_ns()
    if after < before:
        raise ParityCaptureError("monotonic clock reversed during UTC sampling")
    return ClockSample(
        source=source,
        ordinal=ordinal,
        monotonic_ns=before + (after - before) // 2,
        utc_ns=utc_ns,
        uncertainty_ns=after - before,
    )


def validate_bindings() -> None:
    if sha256_file(CONTRACT_PATH) != CONTRACT_SHA256:
        raise ParityCaptureError("live-parity contract hash differs")
    if sha256_file(SOURCE_AUDIT_PATH) != SOURCE_AUDIT_SHA256:
        raise ParityCaptureError("source-audit document hash differs")
    if sha256_file(SOURCE_RESULT_PATH) != SOURCE_RESULT_SHA256:
        raise ParityCaptureError("source-feasibility result hash differs")
    if sha256_file(V1_INVALIDATION_PATH) != V1_INVALIDATION_SHA256:
        raise ParityCaptureError("v1 live-capture invalidation hash differs")
    if sha256_file(V1_INVALID_ARTIFACT_PATH) != V1_INVALID_ARTIFACT_SHA256:
        raise ParityCaptureError("v1 invalid capture artifact hash differs")
    source_result = json.loads(_repo_path(SOURCE_RESULT_PATH).read_text())
    if (
        source_result.get("decision") != "SOURCE_FEASIBILITY_PASS"
        or source_result.get("manifest_hash_without_self")
        != SOURCE_RESULT_MANIFEST_HASH
    ):
        raise ParityCaptureError("source-feasibility pass binding differs")


def enforce_disk_guard() -> int:
    current = used_gib(REPO_ROOT)
    if current >= DISK_LIMIT_GIB:
        raise ParityCaptureError(
            f"disk guard rejected {current} GiB used at {DISK_LIMIT_GIB} GiB"
        )
    return current


def _strict_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ParityCaptureError(f"{label} must be a nonnegative integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.isascii() and value.isdigit():
        parsed = int(value)
    else:
        raise ParityCaptureError(f"{label} must be a nonnegative integer")
    if parsed < 0:
        raise ParityCaptureError(f"{label} must be a nonnegative integer")
    return parsed


def _canonical_positive_decimal(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ParityCaptureError(f"{label} must be a nonempty decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ParityCaptureError(f"{label} is not decimal") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ParityCaptureError(f"{label} must be finite positive")
    normalized = format(parsed, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _strict_source_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParityCaptureError(f"{label} must be a nonempty string")
    return value


def _strict_tick_direction(value: object) -> str:
    if not isinstance(value, str) or value not in TICK_DIRECTIONS:
        raise ParityCaptureError("WebSocket L is not a documented tick direction")
    return value


def _strict_side(value: object, label: str) -> Literal["Buy", "Sell"]:
    if value not in ("Buy", "Sell"):
        raise ParityCaptureError(f"{label} must be Buy or Sell")
    return value


def _strict_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ParityCaptureError(f"{label} must be boolean")
    return value


def _load_json_object(raw: bytes, label: str) -> dict[str, Any]:
    if not raw or len(raw) > MAX_REST_RESPONSE_BYTES:
        raise ParityCaptureError(f"{label} bytes outside frozen bound")
    try:
        payload = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParityCaptureError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ParityCaptureError(f"{label} must be a JSON object")
    return payload


def parse_rest_response(raw: bytes) -> tuple[NormalizedTrade, ...]:
    payload = _load_json_object(raw, "REST response")
    if payload.get("retCode") != 0:
        raise ParityCaptureError("REST retCode is not zero")
    result = payload.get("result")
    if not isinstance(result, dict) or result.get("category") != "linear":
        raise ParityCaptureError("REST result category is not linear")
    rows = result.get("list")
    if not isinstance(rows, list) or not 1 <= len(rows) <= REST_LIMIT:
        raise ParityCaptureError("REST trade list count outside frozen bound")
    parsed: list[NormalizedTrade] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ParityCaptureError("REST trade row is not an object")
        if row.get("symbol") != SYMBOL:
            raise ParityCaptureError("REST trade symbol differs")
        parsed.append(
            NormalizedTrade(
                source_id=_strict_source_id(row.get("execId"), "REST execId"),
                symbol=SYMBOL,
                side=_strict_side(row.get("side"), "REST side"),
                price=_canonical_positive_decimal(row.get("price"), "REST price"),
                size=_canonical_positive_decimal(row.get("size"), "REST size"),
                time_ms=_strict_nonnegative_int(row.get("time"), "REST time"),
                seq=_strict_nonnegative_int(row.get("seq"), "REST seq"),
                block_trade=_strict_bool(
                    row.get("isBlockTrade"), "REST isBlockTrade"
                ),
                rpi_trade=_strict_bool(row.get("isRPITrade"), "REST isRPITrade"),
            )
        )
    return tuple(parsed)


def parse_ws_payload(
    raw: bytes,
) -> tuple[Literal["subscribe", "pong", "trade"], tuple[NormalizedTrade, ...]]:
    if not raw or len(raw) > MAX_WS_MESSAGE_BYTES:
        raise ParityCaptureError("WebSocket message bytes outside frozen bound")
    payload = _load_json_object(raw, "WebSocket message")
    if payload.get("op") == "subscribe":
        if payload.get("success") is not True:
            raise ParityCaptureError("WebSocket subscription was not successful")
        return "subscribe", ()
    if payload.get("op") == "pong" or payload.get("ret_msg") == "pong":
        return "pong", ()
    if payload.get("topic") != WS_TOPIC or payload.get("type") != "snapshot":
        raise ParityCaptureError("unexpected WebSocket message kind")
    _strict_nonnegative_int(payload.get("ts"), "WebSocket message ts")
    rows = payload.get("data")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 1024:
        raise ParityCaptureError("WebSocket trade count outside documented bound")
    parsed: list[NormalizedTrade] = []
    previous_time: int | None = None
    for row in rows:
        if not isinstance(row, dict):
            raise ParityCaptureError("WebSocket trade row is not an object")
        if row.get("s") != SYMBOL:
            raise ParityCaptureError("WebSocket trade symbol differs")
        trade_time = _strict_nonnegative_int(row.get("T"), "WebSocket trade T")
        if previous_time is not None and trade_time < previous_time:
            raise ParityCaptureError("WebSocket message trade times are decreasing")
        previous_time = trade_time
        _strict_tick_direction(row.get("L"))
        parsed.append(
            NormalizedTrade(
                source_id=_strict_source_id(row.get("i"), "WebSocket i"),
                symbol=SYMBOL,
                side=_strict_side(row.get("S"), "WebSocket S"),
                price=_canonical_positive_decimal(row.get("p"), "WebSocket p"),
                size=_canonical_positive_decimal(row.get("v"), "WebSocket v"),
                time_ms=trade_time,
                seq=_strict_nonnegative_int(row.get("seq"), "WebSocket seq"),
                block_trade=_strict_bool(row.get("BT"), "WebSocket BT"),
                rpi_trade=_strict_bool(row.get("RPI"), "WebSocket RPI"),
            )
        )
    return "trade", tuple(parsed)


def _merge_unique(
    records: Sequence[NormalizedTrade],
) -> tuple[dict[str, NormalizedTrade], int, set[str]]:
    unique: dict[str, NormalizedTrade] = {}
    identical_duplicates = 0
    conflicts: set[str] = set()
    for trade in records:
        previous = unique.get(trade.source_id)
        if previous is None:
            unique[trade.source_id] = trade
        elif previous == trade:
            identical_duplicates += 1
        else:
            conflicts.add(trade.source_id)
    return unique, identical_duplicates, conflicts


def evaluate_rest_ws_parity(
    ws_observations: Sequence[ObservedWsTrade],
    rest_snapshots: Sequence[RestSnapshot],
) -> dict[str, Any]:
    ws_records = [observed.trade for observed in ws_observations]
    rest_records = [trade for snapshot in rest_snapshots for trade in snapshot.trades]
    ws_unique, ws_duplicates, ws_conflicts = _merge_unique(ws_records)
    rest_unique, rest_duplicates, rest_conflicts = _merge_unique(rest_records)

    adjacency_overlaps = [
        len(
            {trade.source_id for trade in left.trades}
            & {trade.source_id for trade in right.trades}
        )
        for left, right in zip(rest_snapshots, rest_snapshots[1:])
    ]
    common_ids = set(ws_unique) & set(rest_unique)
    mismatched_ids = {
        source_id
        for source_id in common_ids
        if ws_unique[source_id].comparison_fields()
        != rest_unique[source_id].comparison_fields()
    }
    eligible_ws_ids: set[str] = set()
    eligible_rest_ids: set[str] = set()
    if ws_observations and rest_snapshots:
        first_rest_complete = rest_snapshots[0].response_end_monotonic_ns
        final_rest_start = rest_snapshots[-1].request_start_monotonic_ns
        eligible_ws_ids = {
            observed.trade.source_id
            for observed in ws_observations
            if first_rest_complete < observed.receipt_monotonic_ns < final_rest_start
        }
        first_ws_time = min(observed.trade.time_ms for observed in ws_observations)
        last_ws_time = max(observed.trade.time_ms for observed in ws_observations)
        eligible_rest_ids = {
            trade.source_id
            for trade in rest_records
            if first_ws_time < trade.time_ms < last_ws_time
        }

    missing_from_rest = eligible_ws_ids - set(rest_unique)
    missing_from_ws = eligible_rest_ids - set(ws_unique)
    checks = {
        "minimum_unique_ws_trades": len(ws_unique) >= MIN_UNIQUE_WS_TRADES,
        "minimum_rest_snapshots": len(rest_snapshots) >= MIN_REST_SNAPSHOTS,
        "one_final_rest_snapshot_last": bool(rest_snapshots)
        and sum(snapshot.final for snapshot in rest_snapshots) == 1
        and rest_snapshots[-1].final,
        "adjacent_rest_windows_overlap": bool(adjacency_overlaps)
        and min(adjacency_overlaps) >= 1,
        "minimum_exact_id_intersection": len(common_ids) >= MIN_COMMON_IDS,
        "common_fields_exact": not mismatched_ids,
        "eligible_ws_complete_in_rest": not missing_from_rest,
        "interior_rest_complete_in_ws": not missing_from_ws,
        "no_conflicting_duplicate_ids": not ws_conflicts and not rest_conflicts,
    }
    failures = [name for name, passed in checks.items() if not passed]

    def _set_hash(values: set[str]) -> str:
        return sha256_bytes("\n".join(sorted(values)).encode())

    return {
        "decision": (
            "REST_WS_PARITY_PASS_PENDING_ARCHIVE"
            if not failures
            else "REJECT_NO_REPAIR"
        ),
        "checks": checks,
        "failures": failures,
        "counts": {
            "ws_observations": len(ws_observations),
            "ws_unique_ids": len(ws_unique),
            "ws_identical_duplicates": ws_duplicates,
            "rest_snapshots": len(rest_snapshots),
            "rest_observations": len(rest_records),
            "rest_unique_ids": len(rest_unique),
            "rest_identical_duplicates": rest_duplicates,
            "minimum_adjacent_rest_overlap": (
                min(adjacency_overlaps) if adjacency_overlaps else 0
            ),
            "common_ids": len(common_ids),
            "common_field_mismatches": len(mismatched_ids),
            "eligible_ws_ids": len(eligible_ws_ids),
            "missing_eligible_ws_ids_in_rest": len(missing_from_rest),
            "eligible_interior_rest_ids": len(eligible_rest_ids),
            "missing_interior_rest_ids_in_ws": len(missing_from_ws),
            "ws_conflicting_ids": len(ws_conflicts),
            "rest_conflicting_ids": len(rest_conflicts),
        },
        "diagnostic_set_hashes": {
            "common_field_mismatch_ids": _set_hash(mismatched_ids),
            "missing_eligible_ws_ids_in_rest": _set_hash(missing_from_rest),
            "missing_interior_rest_ids_in_ws": _set_hash(missing_from_ws),
            "ws_conflicting_ids": _set_hash(ws_conflicts),
            "rest_conflicting_ids": _set_hash(rest_conflicts),
        },
    }


def evaluate_clock_integrity(samples: Sequence[ClockSample]) -> dict[str, Any]:
    ordered = sorted(samples, key=lambda sample: sample.monotonic_ns)
    reversals: list[tuple[ClockSample, ClockSample]] = []
    nonincreasing_monotonic = 0
    maximum_disagreement_ns = 0
    for previous, current in zip(ordered, ordered[1:]):
        monotonic_delta = current.monotonic_ns - previous.monotonic_ns
        utc_delta = current.utc_ns - previous.utc_ns
        if monotonic_delta <= 0:
            nonincreasing_monotonic += 1
        if utc_delta < 0:
            reversals.append((previous, current))
        maximum_disagreement_ns = max(
            maximum_disagreement_ns,
            abs(utc_delta - monotonic_delta),
        )
    if len(ordered) >= 2:
        monotonic_elapsed_ns = ordered[-1].monotonic_ns - ordered[0].monotonic_ns
        utc_elapsed_ns = ordered[-1].utc_ns - ordered[0].utc_ns
    else:
        monotonic_elapsed_ns = 0
        utc_elapsed_ns = 0
    return {
        "clock_contract_passed": (
            len(ordered) >= 2
            and not reversals
            and nonincreasing_monotonic == 0
        ),
        "samples": len(ordered),
        "utc_reversal_count": len(reversals),
        "nonincreasing_monotonic_count": nonincreasing_monotonic,
        "maximum_adjacent_clock_disagreement_ns": maximum_disagreement_ns,
        "monotonic_elapsed_ns": monotonic_elapsed_ns,
        "utc_elapsed_ns": utc_elapsed_ns,
        "elapsed_disagreement_ns": utc_elapsed_ns - monotonic_elapsed_ns,
        "maximum_sampling_uncertainty_ns": max(
            (sample.uncertainty_ns for sample in ordered), default=0
        ),
        "reversal_transition_hash": sha256_bytes(
            "\n".join(
                (
                    f"{left.source}:{left.ordinal}:{left.monotonic_ns}:"
                    f"{right.source}:{right.ordinal}:{right.monotonic_ns}"
                )
                for left, right in reversals
            ).encode()
        ),
    }


def apply_clock_gate(
    parity: Mapping[str, Any], clock_integrity: Mapping[str, Any]
) -> dict[str, Any]:
    output = json.loads(json.dumps(parity))
    output["clock_integrity"] = dict(clock_integrity)
    passed = bool(clock_integrity["clock_contract_passed"])
    output.setdefault("checks", {})["local_utc_nonreversing"] = passed
    if not passed:
        output["decision"] = "REJECT_NO_REPAIR"
        failures = output.setdefault("failures", [])
        if "local_utc_nonreversing" not in failures:
            failures.append("local_utc_nonreversing")
    return output


def _validate_rest_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (
        url != REST_URL
        or parsed.scheme != "https"
        or parsed.hostname != "api.bybit.com"
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ParityCaptureError("REST URL differs from frozen Bybit route")


def fetch_rest() -> RestTransportResponse:
    _validate_rest_url(REST_URL)
    request = urllib.request.Request(
        REST_URL,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
        method="GET",
    )
    opener = urllib.request.build_opener(
        _AuditNoRedirectHandler(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    try:
        with opener.open(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_REST_RESPONSE_BYTES + 1)
            transport = RestTransportResponse(
                status=int(response.status),
                final_url=str(response.geturl()),
                content_type=response.headers.get("Content-Type"),
                location=response.headers.get("Location"),
                raw=raw,
            )
    except urllib.error.HTTPError as exc:
        raw = exc.read(MAX_REST_RESPONSE_BYTES + 1)
        transport = RestTransportResponse(
            status=int(exc.code),
            final_url=str(exc.geturl()),
            content_type=exc.headers.get("Content-Type"),
            location=exc.headers.get("Location"),
            raw=raw,
        )
    except urllib.error.URLError as exc:
        raise ParityCaptureError("frozen Bybit REST transport failed") from exc
    if len(transport.raw) > MAX_REST_RESPONSE_BYTES:
        raise ParityCaptureError("REST response exceeds frozen byte bound")
    return transport


def _write_raw_line(handle: Any, payload: Mapping[str, Any]) -> None:
    handle.write(canonical_json(dict(payload)))
    handle.flush()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _capture_output_dir(now: datetime) -> Path:
    stamp = now.strftime("%Y-%m-%dT%H-%M-%S-%fZ")
    return REPO_ROOT / "data" / f"bybit_public_trade_parity_capture_{stamp}"


def _record_ws_message(
    state: CaptureState,
    handle: Any,
    raw: bytes,
    receipt_utc_ns: int,
    receipt_monotonic_ns: int,
    receipt_uncertainty_ns: int = 0,
) -> tuple[str, tuple[NormalizedTrade, ...]]:
    state.raw_bytes += len(raw)
    if state.raw_bytes > MAX_CAPTURE_RAW_BYTES:
        raise ParityCaptureError("capture raw bytes exceed frozen bound")
    state.ws_messages += 1
    state.clock_samples.append(
        ClockSample(
            source="websocket_receipt",
            ordinal=state.ws_messages,
            monotonic_ns=receipt_monotonic_ns,
            utc_ns=receipt_utc_ns,
            uncertainty_ns=receipt_uncertainty_ns,
        )
    )
    _write_raw_line(
        handle,
        {
            "ordinal": state.ws_messages,
            "receipt_utc_ns": receipt_utc_ns,
            "receipt_monotonic_ns": receipt_monotonic_ns,
            "receipt_clock_uncertainty_ns": receipt_uncertainty_ns,
            "raw_json_sha256": sha256_bytes(raw),
            "raw_json_base64": base64.b64encode(raw).decode("ascii"),
        },
    )
    if receipt_utc_ns // 1_000_000_000 < 0:
        raise ParityCaptureError("local UTC receipt time is invalid")
    if datetime.fromtimestamp(receipt_utc_ns / 1e9, timezone.utc).date().isoformat() != (
        state.capture_day
    ):
        raise ParityCaptureError("capture crossed UTC midnight")
    kind, trades = parse_ws_payload(raw)
    if kind == "subscribe":
        state.ws_subscription_acks += 1
    elif kind == "pong":
        state.ws_pongs += 1
    else:
        if len(state.ws_trades) + len(trades) > MAX_WS_TRADES:
            raise ParityCaptureError("WebSocket trades exceed frozen bound")
        state.ws_trades.extend(
            ObservedWsTrade(trade, receipt_monotonic_ns) for trade in trades
        )
        state.last_trade_monotonic_ns = receipt_monotonic_ns
    return kind, trades


def _record_rest_response(
    state: CaptureState,
    handle: Any,
    response: RestTransportResponse,
    *,
    request_start_utc_ns: int,
    request_start_monotonic_ns: int,
    response_end_utc_ns: int,
    response_end_monotonic_ns: int,
    final: bool,
    request_start_uncertainty_ns: int = 0,
    response_end_uncertainty_ns: int = 0,
) -> RestSnapshot:
    state.raw_bytes += len(response.raw)
    if state.raw_bytes > MAX_CAPTURE_RAW_BYTES:
        raise ParityCaptureError("capture raw bytes exceed frozen bound")
    ordinal = len(state.rest_snapshots) + 1
    state.clock_samples.extend(
        (
            ClockSample(
                source="rest_request_start",
                ordinal=ordinal,
                monotonic_ns=request_start_monotonic_ns,
                utc_ns=request_start_utc_ns,
                uncertainty_ns=request_start_uncertainty_ns,
            ),
            ClockSample(
                source="rest_response_end",
                ordinal=ordinal,
                monotonic_ns=response_end_monotonic_ns,
                utc_ns=response_end_utc_ns,
                uncertainty_ns=response_end_uncertainty_ns,
            ),
        )
    )
    _write_raw_line(
        handle,
        {
            "ordinal": ordinal,
            "final": final,
            "request_start_utc_ns": request_start_utc_ns,
            "request_start_monotonic_ns": request_start_monotonic_ns,
            "response_end_utc_ns": response_end_utc_ns,
            "response_end_monotonic_ns": response_end_monotonic_ns,
            "request_start_clock_uncertainty_ns": request_start_uncertainty_ns,
            "response_end_clock_uncertainty_ns": response_end_uncertainty_ns,
            "http_status": response.status,
            "final_url": response.final_url,
            "content_type": response.content_type,
            "location": response.location,
            "raw_json_sha256": sha256_bytes(response.raw),
            "raw_json_base64": base64.b64encode(response.raw).decode("ascii"),
        },
    )
    if response_end_monotonic_ns < request_start_monotonic_ns:
        raise ParityCaptureError("REST monotonic clock reversed")
    for utc_ns in (request_start_utc_ns, response_end_utc_ns):
        if datetime.fromtimestamp(utc_ns / 1e9, timezone.utc).date().isoformat() != (
            state.capture_day
        ):
            raise ParityCaptureError("capture crossed UTC midnight")
    if response.status != 200 or response.final_url != REST_URL:
        raise ParityCaptureError("unexpected REST status or redirect")
    trades = parse_rest_response(response.raw)
    snapshot = RestSnapshot(
        ordinal=ordinal,
        request_start_monotonic_ns=request_start_monotonic_ns,
        response_end_monotonic_ns=response_end_monotonic_ns,
        final=final,
        trades=trades,
    )
    state.rest_snapshots.append(snapshot)
    return snapshot


async def _fetch_and_record_rest(
    state: CaptureState,
    handle: Any,
    *,
    final: bool,
    fetch: FetchRest,
) -> RestSnapshot:
    enforce_disk_guard()
    ordinal = len(state.rest_snapshots) + 1
    start = _local_clock_sample("rest_request_start", ordinal)
    response = await asyncio.to_thread(fetch)
    end = _local_clock_sample("rest_response_end", ordinal)
    return _record_rest_response(
        state,
        handle,
        response,
        request_start_utc_ns=start.utc_ns,
        request_start_monotonic_ns=start.monotonic_ns,
        response_end_utc_ns=end.utc_ns,
        response_end_monotonic_ns=end.monotonic_ns,
        final=final,
        request_start_uncertainty_ns=start.uncertainty_ns,
        response_end_uncertainty_ns=end.uncertainty_ns,
    )


async def _heartbeat(socket: Any, deadline_ns: int) -> None:
    while True:
        remaining = (deadline_ns - time.monotonic_ns()) / 1e9
        if remaining <= 0:
            return
        await asyncio.sleep(min(HEARTBEAT_SECONDS, remaining))
        if time.monotonic_ns() < deadline_ns:
            await socket.send(json.dumps({"op": "ping"}, separators=(",", ":")))


async def _rest_poller(
    state: CaptureState,
    handle: Any,
    deadline_ns: int,
    *,
    fetch: FetchRest,
) -> None:
    next_due_ns = time.monotonic_ns()
    interval_ns = REST_INTERVAL_SECONDS * 1_000_000_000
    while next_due_ns < deadline_ns:
        remaining = (next_due_ns - time.monotonic_ns()) / 1e9
        if remaining > 0:
            await asyncio.sleep(remaining)
        await _fetch_and_record_rest(state, handle, final=False, fetch=fetch)
        next_due_ns += interval_ns


async def _capture_network(
    state: CaptureState,
    ws_handle: Any,
    rest_handle: Any,
    *,
    fetch: FetchRest = fetch_rest,
) -> None:
    import websockets
    from websockets.asyncio.client import connect
    from websockets.exceptions import SecurityError

    class NoRedirectConnect(connect):
        def process_redirect(self, exc: Exception) -> Exception | str:
            del exc
            return SecurityError("WebSocket redirect rejected before target access")

    async with NoRedirectConnect(
        WS_URL,
        open_timeout=WS_OPEN_TIMEOUT_SECONDS,
        ping_interval=None,
        max_size=MAX_WS_MESSAGE_BYTES,
        max_queue=1024,
        compression=None,
        proxy=None,
        user_agent_header=USER_AGENT,
    ) as socket:
        state.websocket_sessions += 1
        if state.websocket_sessions != 1:
            raise ParityCaptureError("more than one WebSocket session opened")
        await socket.send(
            json.dumps({"op": "subscribe", "args": [WS_TOPIC]}, separators=(",", ":"))
        )
        first_trade_seen = False
        while not first_trade_seen:
            message = await asyncio.wait_for(
                socket.recv(), timeout=WS_OPEN_TIMEOUT_SECONDS
            )
            if not isinstance(message, str):
                raise ParityCaptureError("WebSocket message is not a text frame")
            raw = message.encode("utf-8")
            receipt = _local_clock_sample("websocket_receipt", state.ws_messages + 1)
            kind, _ = _record_ws_message(
                state,
                ws_handle,
                raw,
                receipt.utc_ns,
                receipt.monotonic_ns,
                receipt.uncertainty_ns,
            )
            if kind == "trade":
                if state.ws_subscription_acks != 1:
                    raise ParityCaptureError(
                        "first trade arrived without exactly one subscription ack"
                    )
                state.first_trade_monotonic_ns = receipt.monotonic_ns
                first_trade_seen = True

        assert state.first_trade_monotonic_ns is not None
        deadline_ns = state.first_trade_monotonic_ns + CAPTURE_SECONDS * 1_000_000_000
        rest_task = asyncio.create_task(
            _rest_poller(state, rest_handle, deadline_ns, fetch=fetch)
        )
        heartbeat_task = asyncio.create_task(_heartbeat(socket, deadline_ns))
        try:
            while True:
                remaining = (deadline_ns - time.monotonic_ns()) / 1e9
                if remaining <= 0:
                    break
                try:
                    message = await asyncio.wait_for(socket.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if not isinstance(message, str):
                    raise ParityCaptureError("WebSocket message is not a text frame")
                raw = message.encode("utf-8")
                receipt = _local_clock_sample(
                    "websocket_receipt", state.ws_messages + 1
                )
                kind, _ = _record_ws_message(
                    state,
                    ws_handle,
                    raw,
                    receipt.utc_ns,
                    receipt.monotonic_ns,
                    receipt.uncertainty_ns,
                )
                if kind == "subscribe":
                    raise ParityCaptureError("duplicate WebSocket subscription ack")
            await rest_task
            await heartbeat_task
        finally:
            for task in (rest_task, heartbeat_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(rest_task, heartbeat_task, return_exceptions=True)

    await _fetch_and_record_rest(state, rest_handle, final=True, fetch=fetch)


def _artifact_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_manifest(
    state: CaptureState,
    parity: Mapping[str, Any],
    *,
    disk_used_gib_before_capture: int,
    ws_path: Path,
    rest_path: Path,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "decision": parity["decision"],
        "capture_day_utc": state.capture_day,
        "started_at_utc": state.started_at_utc,
        "ended_at_utc": state.ended_at_utc,
        "transport": {
            "rest_url": REST_URL,
            "rest_order_assumed": False,
            "rest_limit": REST_LIMIT,
            "rest_interval_seconds": REST_INTERVAL_SECONDS,
            "websocket_url": WS_URL,
            "websocket_topic": WS_TOPIC,
            "websocket_sessions": state.websocket_sessions,
            "reconnects": 0,
            "capture_seconds": CAPTURE_SECONDS,
            "heartbeat_seconds": HEARTBEAT_SECONDS,
        },
        "capture": {
            "ws_messages": state.ws_messages,
            "ws_subscription_acks": state.ws_subscription_acks,
            "ws_pongs": state.ws_pongs,
            "raw_json_bytes": state.raw_bytes,
            "clock_samples": len(state.clock_samples),
            "first_trade_monotonic_ns": state.first_trade_monotonic_ns,
            "last_trade_monotonic_ns": state.last_trade_monotonic_ns,
        },
        "parity": dict(parity),
        "raw_artifacts": {
            "websocket": _artifact_metadata(ws_path),
            "rest": _artifact_metadata(rest_path),
            "committed_to_git": False,
        },
        "archive_parity": {
            "status": "PENDING" if parity["decision"].startswith("REST_WS_PARITY_PASS") else "NOT_AUTHORIZED",
            "publication_cadence_assumed": False,
            "archive_rows_opened": 0,
        },
        "disk": {
            "used_gib_before_capture": disk_used_gib_before_capture,
            "limit_gib": DISK_LIMIT_GIB,
            "guard_filesystem": str(REPO_ROOT),
            "guard_enforced_before_each_http_request": True,
        },
        "bindings": {
            "contract_path": str(CONTRACT_PATH),
            "contract_sha256": CONTRACT_SHA256,
            "source_audit_path": str(SOURCE_AUDIT_PATH),
            "source_audit_sha256": SOURCE_AUDIT_SHA256,
            "source_result_path": str(SOURCE_RESULT_PATH),
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "source_result_manifest_hash": SOURCE_RESULT_MANIFEST_HASH,
            "v1_invalidation_path": str(V1_INVALIDATION_PATH),
            "v1_invalidation_sha256": V1_INVALIDATION_SHA256,
            "v1_invalid_artifact_path": str(V1_INVALID_ARTIFACT_PATH),
            "v1_invalid_artifact_sha256": V1_INVALID_ARTIFACT_SHA256,
            "capture_script_path": str(SCRIPT_PATH),
            "capture_script_sha256": sha256_file(SCRIPT_PATH),
        },
        "outcome_boundary": {
            "bsea_clock_built": False,
            "candidate_incidence_opened": False,
            "binance_comparator_opened": False,
            "market_outcomes_opened": False,
            "returns_or_pnl_opened": False,
        },
        "manifest_hash_scope": "canonical_json_excluding_manifest_hash_without_self",
    }
    manifest["manifest_hash_without_self"] = canonical_hash(manifest)
    return manifest


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable manifest already exists: {path}")
    core = dict(manifest)
    observed = core.pop("manifest_hash_without_self", None)
    if observed != canonical_hash(core):
        raise ParityCaptureError("manifest self-hash differs")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(dict(manifest)))
    temporary.replace(path)


def run_capture() -> tuple[Path, dict[str, Any]]:
    validate_bindings()
    disk_before = enforce_disk_guard()
    now = _utc_now()
    output_dir = _capture_output_dir(now)
    output_dir.mkdir(parents=True, exist_ok=False)
    state = CaptureState(
        capture_day=now.date().isoformat(),
        started_at_utc=_utc_iso(now),
        output_dir=output_dir,
    )
    ws_path = output_dir / "websocket_messages.ndjson.gz"
    rest_path = output_dir / "rest_responses.ndjson.gz"
    runtime_error: Exception | None = None
    with gzip.open(ws_path, "wb", compresslevel=6) as ws_handle, gzip.open(
        rest_path, "wb", compresslevel=6
    ) as rest_handle:
        try:
            asyncio.run(_capture_network(state, ws_handle, rest_handle))
        except Exception as exc:  # The immutable failure manifest is the audit trail.
            runtime_error = exc
    state.ended_at_utc = _utc_iso(_utc_now())
    clock_integrity = evaluate_clock_integrity(state.clock_samples)
    if runtime_error is None:
        parity = apply_clock_gate(
            evaluate_rest_ws_parity(state.ws_trades, state.rest_snapshots),
            clock_integrity,
        )
    else:
        parity = apply_clock_gate({
            "decision": "REJECT_NO_REPAIR",
            "checks": {"capture_runtime_completed": False},
            "failures": [f"capture_runtime:{type(runtime_error).__name__}"],
            "runtime_error": {
                "type": type(runtime_error).__name__,
                "message": str(runtime_error)[:1_000],
            },
        }, clock_integrity)
    manifest = build_manifest(
        state,
        parity,
        disk_used_gib_before_capture=disk_before,
        ws_path=ws_path,
        rest_path=rest_path,
    )
    manifest_path = output_dir / "manifest.json"
    write_manifest(manifest_path, manifest)
    return manifest_path, manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    manifest_path, manifest = run_capture()
    print(manifest_path)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if manifest["decision"] == "REST_WS_PARITY_PASS_PENDING_ARCHIVE" else 2


if __name__ == "__main__":
    sys.exit(main())
