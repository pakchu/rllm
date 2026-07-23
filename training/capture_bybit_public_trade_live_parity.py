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
import os
import select
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, Protocol

from training.probe_bybit_public_trade_sequence_source import (
    DISK_LIMIT_GIB,
    REPO_ROOT,
    canonical_hash,
    canonical_json,
    sha256_bytes,
    sha256_file,
    used_gib,
)


PROTOCOL_VERSION = "bybit_public_trade_live_parity_capture_v3"
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
CLOCK_CORRECTION_PATH = Path(
    "docs/bybit-capture-clock-source-correction-2026-07-23.md"
)
CLOCK_CORRECTION_SHA256 = (
    "95300035e07a8d57927916284f0a69e7855c12fe50e050599a548a038bcb82e9"
)
CLOCK_PREFLIGHT_RESULT_PATH = Path(
    "results/bybit_capture_clock_source_preflight_v1_2026-07-23.json"
)
CLOCK_PREFLIGHT_RESULT_SHA256 = (
    "9868e49ad722b5cf5d557efe88fcf1d6a24fc318117a71dc01d7a1680a28614e"
)
CLOCK_PREFLIGHT_RESULT_MANIFEST_HASH = (
    "40cce365242abade2aa79802f57167488f48b20319244a43af53826ecf1e28be"
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
CLOCK_PREFLIGHT_SECONDS = 60
CLOCK_PREFLIGHT_SAMPLE_INTERVAL_SECONDS = 0.1
RAW_WAIT_QUANTUM_SECONDS = 0.25
HOST_CLOCK_READ_TIMEOUT_SECONDS = 2
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
POWERSHELL_PATH = Path(
    "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe"
)
HOST_CLOCK_SCRIPT = """while (($line = [Console]::In.ReadLine()) -ne $null) {
  if ($line -eq 'q') { break }
  if ($line -ne 't') { exit 64 }
  $ticks = [DateTime]::UtcNow.Ticks
  [Console]::Out.WriteLine($ticks)
  [Console]::Out.Flush()
}"""
HOST_CLOCK_SCRIPT_SHA256 = sha256_bytes(HOST_CLOCK_SCRIPT.encode("utf-8"))
WINDOWS_UNIX_EPOCH_TICKS = 621_355_968_000_000_000
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


class ClockReader(Protocol):
    provider_id: str
    monotonic_source: str
    utc_source: str

    def monotonic_ns(self) -> int: ...

    def utc_ns(self) -> int: ...


class ProcessClockReader:
    """Test-only/default process clocks; production capture never selects these."""

    provider_id = "python_process_clock"
    monotonic_source = "CLOCK_MONOTONIC"
    utc_source = "CLOCK_REALTIME"

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def utc_ns(self) -> int:
        return time.time_ns()


PROCESS_CLOCK = ProcessClockReader()


def _pipe_readable(stream: Any, timeout_seconds: float) -> bool:
    readable, _, _ = select.select([stream], [], [], timeout_seconds)
    return bool(readable)


def _read_pipe_chunk(stream: Any, maximum_bytes: int) -> bytes:
    return os.read(stream.fileno(), maximum_bytes)


def _kernel_release() -> str:
    return Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8")


class WindowsHostRawClock:
    """Frozen WSL clock: Windows host UTC bracketed by CLOCK_MONOTONIC_RAW."""

    provider_id = "windows_host_utc_raw_monotonic_v1"
    monotonic_source = "CLOCK_MONOTONIC_RAW"
    utc_source = "windows_powershell_datetime_utcnow_ticks"

    def __init__(
        self,
        *,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        readable: Callable[[Any, float], bool] = _pipe_readable,
        read_chunk: Callable[[Any, int], bytes] = _read_pipe_chunk,
        raw_monotonic_ns: Callable[[], int] | None = None,
        powershell_path: Path = POWERSHELL_PATH,
        release_reader: Callable[[], str] = _kernel_release,
    ) -> None:
        self._popen_factory = popen_factory
        self._readable = readable
        self._read_chunk = read_chunk
        self._raw_monotonic_ns = raw_monotonic_ns or (
            lambda: time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        )
        self._powershell_path = powershell_path
        self._release_reader = release_reader
        self._process: Any | None = None
        self._warmup_sample: ClockSample | None = None
        self._utc_reads = 0
        self._pid: int | None = None
        self._closed_cleanly = False

    def monotonic_ns(self) -> int:
        return self._raw_monotonic_ns()

    def utc_ns(self) -> int:
        process = self._process
        if process is None or process.poll() is not None:
            raise ParityCaptureError("Windows host clock process is not running")
        if process.stdin is None or process.stdout is None:
            raise ParityCaptureError("Windows host clock pipes are unavailable")
        try:
            process.stdin.write(b"t\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ParityCaptureError("Windows host clock request failed") from exc
        deadline_ns = (
            self.monotonic_ns() + HOST_CLOCK_READ_TIMEOUT_SECONDS * 1_000_000_000
        )
        raw = bytearray()
        while b"\n" not in raw:
            remaining_seconds = (deadline_ns - self.monotonic_ns()) / 1e9
            if remaining_seconds <= 0 or not self._readable(
                process.stdout, remaining_seconds
            ):
                raise ParityCaptureError("Windows host clock response timed out")
            try:
                chunk = self._read_chunk(process.stdout, 128 - len(raw))
            except OSError as exc:
                raise ParityCaptureError(
                    "Windows host clock response read failed"
                ) from exc
            if not chunk:
                raise ParityCaptureError("Windows host clock response reached EOF")
            raw.extend(chunk)
            if len(raw) >= 128 and b"\n" not in raw:
                raise ParityCaptureError("Windows host clock response is oversized")
        line, separator, suffix = bytes(raw).partition(b"\n")
        if separator != b"\n" or suffix:
            raise ParityCaptureError("Windows host clock response framing differs")
        token = line.strip()
        if not token.isascii() or not token.isdigit() or len(token) > 32:
            raise ParityCaptureError("Windows host clock response is malformed")
        ticks = int(token)
        if ticks <= WINDOWS_UNIX_EPOCH_TICKS:
            raise ParityCaptureError("Windows host clock response is nonpositive")
        value = (ticks - WINDOWS_UNIX_EPOCH_TICKS) * 100
        self._utc_reads += 1
        return value

    def start(self) -> None:
        if self._process is not None:
            raise ParityCaptureError("Windows host clock process started twice")
        if not hasattr(time, "CLOCK_MONOTONIC_RAW"):
            raise ParityCaptureError("CLOCK_MONOTONIC_RAW is unavailable")
        release = self._release_reader().lower()
        if "microsoft" not in release or not self._powershell_path.is_file():
            raise ParityCaptureError("frozen WSL Windows host clock is unavailable")
        process = self._popen_factory(
            [
                str(self._powershell_path),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                HOST_CLOCK_SCRIPT,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=False,
            bufsize=0,
        )
        self._process = process
        self._pid = process.pid
        self._warmup_sample = _local_clock_sample("clock_provider_warmup", 1, self)
        if self._warmup_sample.uncertainty_ns < 0:
            raise ParityCaptureError("Windows host clock warmup was invalid")
        _utc_datetime_from_ns(self._warmup_sample.utc_ns)

    def close(self) -> None:
        process = self._process
        if process is None:
            raise ParityCaptureError("Windows host clock process was never started")
        error: Exception | None = None
        try:
            if process.poll() is None:
                if process.stdin is None:
                    raise ParityCaptureError("Windows host clock stdin is unavailable")
                process.stdin.write(b"q\n")
                process.stdin.flush()
            return_code = process.wait(timeout=5)
            if return_code != 0:
                raise ParityCaptureError(
                    f"Windows host clock exited with status {return_code}"
                )
        except Exception as exc:
            error = exc
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        finally:
            self._closed_cleanly = error is None
            for stream in (process.stdin, process.stdout):
                close_stream = getattr(stream, "close", None)
                if close_stream is not None:
                    try:
                        close_stream()
                    except OSError as exc:
                        self._closed_cleanly = False
                        if error is None:
                            error = exc
        if error is not None:
            raise ParityCaptureError("Windows host clock close failed") from error

    def metadata(self) -> dict[str, Any]:
        warmup = self._warmup_sample
        return {
            "provider_id": self.provider_id,
            "monotonic_source": self.monotonic_source,
            "utc_source": self.utc_source,
            "powershell_path": str(self._powershell_path),
            "powershell_script_sha256": HOST_CLOCK_SCRIPT_SHA256,
            "shell": False,
            "process_pid": self._pid,
            "utc_reads": self._utc_reads,
            "warmup_sample": (
                {
                    "monotonic_ns": warmup.monotonic_ns,
                    "utc_ns": warmup.utc_ns,
                    "uncertainty_ns": warmup.uncertainty_ns,
                }
                if warmup is not None
                else None
            ),
            "closed_cleanly": self._closed_cleanly,
            "fallback_used": False,
        }


@dataclass(frozen=True)
class ClockLedgerExpectation:
    websocket_messages: int
    rest_attempts: int
    rest_attempts_completed: int
    rest_responses_audited: int


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
    rest_attempts_started: int = 0
    rest_attempts_completed: int = 0
    rest_responses_audited: int = 0
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


def _utc_datetime_from_ns(utc_ns: int) -> datetime:
    seconds, nanoseconds = divmod(utc_ns, 1_000_000_000)
    return datetime.fromtimestamp(seconds, timezone.utc).replace(
        microsecond=nanoseconds // 1_000
    )


def _utc_iso_from_ns(utc_ns: int) -> str:
    seconds, nanoseconds = divmod(utc_ns, 1_000_000_000)
    prefix = datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{prefix}.{nanoseconds:09d}Z"


def _local_clock_sample(
    source: str,
    ordinal: int,
    clock: ClockReader = PROCESS_CLOCK,
) -> ClockSample:
    before = clock.monotonic_ns()
    utc_ns = clock.utc_ns()
    after = clock.monotonic_ns()
    if after < before:
        return ClockSample(
            source=source,
            ordinal=ordinal,
            monotonic_ns=before,
            utc_ns=utc_ns,
            uncertainty_ns=-1,
        )
    return ClockSample(
        source=source,
        ordinal=ordinal,
        monotonic_ns=before + (after - before) // 2,
        utc_ns=utc_ns,
        uncertainty_ns=after - before,
    )


def evaluate_clock_preflight(
    samples: Sequence[ClockSample],
    *,
    probe_started_monotonic_ns: int,
    probe_ended_monotonic_ns: int,
    required_duration_ns: int,
) -> dict[str, Any]:
    ordered = sorted(samples, key=lambda sample: sample.monotonic_ns)
    reversals = 0
    nonincreasing = 0
    maximum_disagreement_ns = 0
    for previous, current in zip(ordered, ordered[1:]):
        monotonic_delta = current.monotonic_ns - previous.monotonic_ns
        utc_delta = current.utc_ns - previous.utc_ns
        reversals += utc_delta < 0
        nonincreasing += monotonic_delta <= 0
        maximum_disagreement_ns = max(
            maximum_disagreement_ns,
            abs(utc_delta - monotonic_delta),
        )
    ordinals_complete = [sample.ordinal for sample in samples] == list(
        range(1, len(samples) + 1)
    ) and all(sample.source == "clock_provider_preflight" for sample in samples)
    invalid_uncertainty = sum(sample.uncertainty_ns < 0 for sample in samples)
    try:
        utc_days = {
            _utc_datetime_from_ns(sample.utc_ns).date().isoformat()
            for sample in samples
        }
    except (OSError, OverflowError, ValueError):
        utc_days = set()
    probe_elapsed_ns = probe_ended_monotonic_ns - probe_started_monotonic_ns
    sample_elapsed_ns = (
        ordered[-1].monotonic_ns - ordered[0].monotonic_ns
        if len(ordered) >= 2
        else 0
    )
    utc_elapsed_ns = (
        ordered[-1].utc_ns - ordered[0].utc_ns if len(ordered) >= 2 else 0
    )
    checks = {
        "minimum_two_samples": len(samples) >= 2,
        "required_raw_duration_elapsed": probe_elapsed_ns >= required_duration_ns,
        "sample_ordinals_complete": ordinals_complete,
        "raw_monotonic_strictly_increasing": nonincreasing == 0,
        "host_utc_nonreversing": reversals == 0,
        "sampling_uncertainty_valid": invalid_uncertainty == 0,
        "single_utc_day": len(utc_days) == 1,
    }
    ledger_rows = [
        {
            "source": sample.source,
            "ordinal": sample.ordinal,
            "monotonic_ns": sample.monotonic_ns,
            "utc_ns": sample.utc_ns,
            "uncertainty_ns": sample.uncertainty_ns,
        }
        for sample in samples
    ]
    return {
        "decision": "PASS" if all(checks.values()) else "REJECT_NO_NETWORK",
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "samples": len(samples),
        "probe_elapsed_ns": probe_elapsed_ns,
        "required_duration_ns": required_duration_ns,
        "sample_elapsed_ns": sample_elapsed_ns,
        "utc_elapsed_ns": utc_elapsed_ns,
        "elapsed_disagreement_ns": utc_elapsed_ns - sample_elapsed_ns,
        "utc_reversal_count": reversals,
        "nonincreasing_monotonic_count": nonincreasing,
        "invalid_sampling_uncertainty_count": invalid_uncertainty,
        "maximum_adjacent_clock_disagreement_ns": maximum_disagreement_ns,
        "maximum_sampling_uncertainty_ns": max(
            (sample.uncertainty_ns for sample in samples), default=0
        ),
        "utc_days": sorted(utc_days),
        "clock_ledger_hash": sha256_bytes(
            canonical_json(ledger_rows).rstrip(b"\n")
        ),
    }


def run_clock_preflight(
    clock: ClockReader,
    *,
    duration_ns: int = CLOCK_PREFLIGHT_SECONDS * 1_000_000_000,
    sample_interval_seconds: float = CLOCK_PREFLIGHT_SAMPLE_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if duration_ns <= 0 or sample_interval_seconds <= 0:
        raise ParityCaptureError("clock preflight bounds must be positive")
    probe_started = clock.monotonic_ns()
    deadline = probe_started + duration_ns
    samples: list[ClockSample] = []
    while clock.monotonic_ns() < deadline:
        samples.append(
            _local_clock_sample(
                "clock_provider_preflight",
                len(samples) + 1,
                clock,
            )
        )
        remaining_seconds = (deadline - clock.monotonic_ns()) / 1e9
        if remaining_seconds > 0:
            sleep(min(sample_interval_seconds, remaining_seconds))
    probe_ended = clock.monotonic_ns()
    report = evaluate_clock_preflight(
        samples,
        probe_started_monotonic_ns=probe_started,
        probe_ended_monotonic_ns=probe_ended,
        required_duration_ns=duration_ns,
    )
    if report["decision"] != "PASS":
        raise ParityCaptureError(
            "Windows host/raw-monotonic clock preflight rejected before network"
        )
    return report


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
    if sha256_file(CLOCK_CORRECTION_PATH) != CLOCK_CORRECTION_SHA256:
        raise ParityCaptureError("clock-source correction hash differs")
    if sha256_file(CLOCK_PREFLIGHT_RESULT_PATH) != CLOCK_PREFLIGHT_RESULT_SHA256:
        raise ParityCaptureError("clock-source preflight result hash differs")
    source_result = json.loads(_repo_path(SOURCE_RESULT_PATH).read_text())
    if (
        source_result.get("decision") != "SOURCE_FEASIBILITY_PASS"
        or source_result.get("manifest_hash_without_self")
        != SOURCE_RESULT_MANIFEST_HASH
    ):
        raise ParityCaptureError("source-feasibility pass binding differs")
    clock_result = json.loads(_repo_path(CLOCK_PREFLIGHT_RESULT_PATH).read_text())
    if (
        clock_result.get("decision")
        != "PROCESS_REALTIME_REJECT_HOST_RAW_BRIDGE_FEASIBLE"
        or clock_result.get("manifest_hash_without_self")
        != CLOCK_PREFLIGHT_RESULT_MANIFEST_HASH
    ):
        raise ParityCaptureError("clock-source preflight binding differs")


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


def evaluate_clock_integrity(
    samples: Sequence[ClockSample], expectation: ClockLedgerExpectation
) -> dict[str, Any]:
    ordered = sorted(samples, key=lambda sample: sample.monotonic_ns)
    reversals: list[tuple[ClockSample, ClockSample]] = []
    nonincreasing_monotonic = 0
    invalid_sampling_uncertainty = sum(
        sample.uncertainty_ns < 0 for sample in ordered
    )
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
    observed_ledger = Counter((sample.source, sample.ordinal) for sample in samples)
    expected_ledger = Counter(
        [("capture_start", 1), ("capture_end", 1)]
        + [
            ("websocket_receipt", ordinal)
            for ordinal in range(1, expectation.websocket_messages + 1)
        ]
        + [
            (source, ordinal)
            for ordinal in range(1, expectation.rest_attempts + 1)
            for source in ("rest_request_start", "rest_response_end")
        ]
    )
    missing_ledger = expected_ledger - observed_ledger
    excess_ledger = observed_ledger - expected_ledger
    ordered_identities = [(sample.source, sample.ordinal) for sample in ordered]
    websocket_ledger = [
        identity
        for identity in ordered_identities
        if identity[0] == "websocket_receipt"
    ]
    rest_ledger = [
        identity
        for identity in ordered_identities
        if identity[0] in {"rest_request_start", "rest_response_end"}
    ]
    expected_websocket_ledger = [
        ("websocket_receipt", ordinal)
        for ordinal in range(1, expectation.websocket_messages + 1)
    ]
    expected_rest_ledger = [
        (source, ordinal)
        for ordinal in range(1, expectation.rest_attempts + 1)
        for source in ("rest_request_start", "rest_response_end")
    ]
    boundary_ledger_complete = (
        bool(ordered_identities)
        and ordered_identities[0] == ("capture_start", 1)
        and ordered_identities[-1] == ("capture_end", 1)
    )
    ledger_sequence_complete = (
        boundary_ledger_complete
        and websocket_ledger == expected_websocket_ledger
        and rest_ledger == expected_rest_ledger
    )
    rest_ledger_complete = (
        expectation.rest_attempts
        == expectation.rest_attempts_completed
        == expectation.rest_responses_audited
    )
    ledger_complete = (
        not missing_ledger
        and not excess_ledger
        and ledger_sequence_complete
        and rest_ledger_complete
    )
    capture_start_samples = [
        sample
        for sample in samples
        if (sample.source, sample.ordinal) == ("capture_start", 1)
    ]
    try:
        capture_utc_day = (
            _utc_datetime_from_ns(capture_start_samples[0].utc_ns).date().isoformat()
            if len(capture_start_samples) == 1
            else None
        )
        utc_day_consistent = capture_utc_day is not None and all(
            _utc_datetime_from_ns(sample.utc_ns).date().isoformat()
            == capture_utc_day
            for sample in samples
        )
    except (OSError, OverflowError, ValueError):
        capture_utc_day = None
        utc_day_consistent = False
    ledger_rows = [
        {
            "source": sample.source,
            "ordinal": sample.ordinal,
            "monotonic_ns": sample.monotonic_ns,
            "utc_ns": sample.utc_ns,
            "uncertainty_ns": sample.uncertainty_ns,
        }
        for sample in ordered
    ]

    def _counter_hash(values: Counter[tuple[str, int]]) -> str:
        serialized = "\n".join(
            f"{source}:{ordinal}:{count}"
            for (source, ordinal), count in sorted(values.items())
        )
        return sha256_bytes(serialized.encode())

    return {
        "clock_contract_passed": (
            len(ordered) >= 2
            and not reversals
            and nonincreasing_monotonic == 0
            and invalid_sampling_uncertainty == 0
            and ledger_complete
            and utc_day_consistent
        ),
        "samples": len(ordered),
        "utc_reversal_count": len(reversals),
        "nonincreasing_monotonic_count": nonincreasing_monotonic,
        "invalid_sampling_uncertainty_count": invalid_sampling_uncertainty,
        "maximum_adjacent_clock_disagreement_ns": maximum_disagreement_ns,
        "monotonic_elapsed_ns": monotonic_elapsed_ns,
        "utc_elapsed_ns": utc_elapsed_ns,
        "elapsed_disagreement_ns": utc_elapsed_ns - monotonic_elapsed_ns,
        "maximum_sampling_uncertainty_ns": max(
            (sample.uncertainty_ns for sample in ordered), default=0
        ),
        "ledger_complete": ledger_complete,
        "boundary_ledger_complete": boundary_ledger_complete,
        "ledger_sequence_complete": ledger_sequence_complete,
        "rest_attempt_ledger_complete": rest_ledger_complete,
        "capture_utc_day": capture_utc_day,
        "utc_day_consistent": utc_day_consistent,
        "expected_samples": sum(expected_ledger.values()),
        "missing_ledger_entries": sum(missing_ledger.values()),
        "excess_ledger_entries": sum(excess_ledger.values()),
        "missing_ledger_hash": _counter_hash(missing_ledger),
        "excess_ledger_hash": _counter_hash(excess_ledger),
        "clock_ledger_hash": sha256_bytes(canonical_json(ledger_rows).rstrip(b"\n")),
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
    frame_type: Literal["text", "binary"] = "text",
) -> tuple[str, tuple[NormalizedTrade, ...]]:
    state.raw_bytes += len(raw)
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
            "frame_type": frame_type,
            "raw_frame_sha256": sha256_bytes(raw),
            "raw_frame_base64": base64.b64encode(raw).decode("ascii"),
        },
    )
    if state.raw_bytes > MAX_CAPTURE_RAW_BYTES:
        raise ParityCaptureError("capture raw bytes exceed frozen bound")
    if frame_type != "text":
        raise ParityCaptureError("WebSocket binary frame is forbidden")
    if receipt_utc_ns // 1_000_000_000 < 0:
        raise ParityCaptureError("local UTC receipt time is invalid")
    if _utc_datetime_from_ns(receipt_utc_ns).date().isoformat() != state.capture_day:
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


def _record_ws_clock_error(
    state: CaptureState,
    handle: Any,
    raw: bytes,
    *,
    frame_type: Literal["text", "binary"],
    error: Exception,
) -> None:
    state.raw_bytes += len(raw)
    state.ws_messages += 1
    _write_raw_line(
        handle,
        {
            "ordinal": state.ws_messages,
            "receipt_utc_ns": None,
            "receipt_monotonic_ns": None,
            "receipt_clock_uncertainty_ns": None,
            "frame_type": frame_type,
            "raw_frame_sha256": sha256_bytes(raw),
            "raw_frame_base64": base64.b64encode(raw).decode("ascii"),
            "clock_error_type": type(error).__name__,
            "clock_error_message": str(error)[:1_000],
        },
    )
    if state.raw_bytes > MAX_CAPTURE_RAW_BYTES:
        raise ParityCaptureError("capture raw bytes exceed frozen bound")


def _sample_and_record_ws_message(
    state: CaptureState,
    handle: Any,
    raw: bytes,
    frame_type: Literal["text", "binary"],
    clock: ClockReader,
) -> tuple[str, tuple[NormalizedTrade, ...], ClockSample]:
    try:
        receipt = _local_clock_sample(
            "websocket_receipt", state.ws_messages + 1, clock
        )
    except Exception as exc:
        _record_ws_clock_error(
            state,
            handle,
            raw,
            frame_type=frame_type,
            error=exc,
        )
        raise
    kind, trades = _record_ws_message(
        state,
        handle,
        raw,
        receipt.utc_ns,
        receipt.monotonic_ns,
        receipt.uncertainty_ns,
        frame_type,
    )
    return kind, trades, receipt


def _begin_rest_attempt(
    state: CaptureState,
    handle: Any,
    start: ClockSample,
) -> int:
    ordinal = state.rest_attempts_started + 1
    state.rest_attempts_started = ordinal
    state.clock_samples.append(start)
    _write_raw_line(
        handle,
        {
            "record_type": "request_start",
            "ordinal": ordinal,
            "clock_source": start.source,
            "clock_ordinal": start.ordinal,
            "request_start_utc_ns": start.utc_ns,
            "request_start_monotonic_ns": start.monotonic_ns,
            "request_start_clock_uncertainty_ns": start.uncertainty_ns,
        },
    )
    if start.source != "rest_request_start" or start.ordinal != ordinal:
        raise ParityCaptureError("REST request-start clock identity differs")
    if _utc_datetime_from_ns(start.utc_ns).date().isoformat() != state.capture_day:
        raise ParityCaptureError("capture crossed UTC midnight")
    return ordinal


def _complete_rest_clock(
    state: CaptureState,
    ordinal: int,
    end: ClockSample,
) -> None:
    state.clock_samples.append(end)
    if end.source != "rest_response_end" or end.ordinal != ordinal:
        raise ParityCaptureError("REST response-end clock identity differs")
    if ordinal != state.rest_attempts_completed + 1:
        raise ParityCaptureError("REST attempt completion order differs")
    state.rest_attempts_completed = ordinal


def _record_rest_transport_error(
    state: CaptureState,
    handle: Any,
    *,
    ordinal: int,
    end: ClockSample,
    error: Exception,
) -> None:
    _write_raw_line(
        handle,
        {
            "record_type": "transport_error",
            "ordinal": ordinal,
            "clock_source": end.source,
            "clock_ordinal": end.ordinal,
            "response_end_utc_ns": end.utc_ns,
            "response_end_monotonic_ns": end.monotonic_ns,
            "response_end_clock_uncertainty_ns": end.uncertainty_ns,
            "error_type": type(error).__name__,
            "error_message": str(error)[:1_000],
        },
    )
    _complete_rest_clock(state, ordinal, end)


def _record_rest_transport_clock_error(
    state: CaptureState,
    handle: Any,
    *,
    ordinal: int,
    transport_error: Exception,
    clock_error: Exception,
) -> None:
    _write_raw_line(
        handle,
        {
            "record_type": "transport_error_clock_error",
            "ordinal": ordinal,
            "response_end_utc_ns": None,
            "response_end_monotonic_ns": None,
            "response_end_clock_uncertainty_ns": None,
            "error_type": type(transport_error).__name__,
            "error_message": str(transport_error)[:1_000],
            "clock_error_type": type(clock_error).__name__,
            "clock_error_message": str(clock_error)[:1_000],
        },
    )


def _record_rest_response_clock_error(
    state: CaptureState,
    handle: Any,
    response: RestTransportResponse,
    *,
    ordinal: int,
    start: ClockSample,
    final: bool,
    clock_error: Exception,
) -> None:
    state.raw_bytes += len(response.raw)
    _write_raw_line(
        handle,
        {
            "record_type": "response_clock_error",
            "ordinal": ordinal,
            "final": final,
            "request_clock_source": start.source,
            "request_clock_ordinal": start.ordinal,
            "response_clock_source": None,
            "response_clock_ordinal": None,
            "request_start_utc_ns": start.utc_ns,
            "request_start_monotonic_ns": start.monotonic_ns,
            "response_end_utc_ns": None,
            "response_end_monotonic_ns": None,
            "request_start_clock_uncertainty_ns": start.uncertainty_ns,
            "response_end_clock_uncertainty_ns": None,
            "http_status": response.status,
            "final_url": response.final_url,
            "content_type": response.content_type,
            "location": response.location,
            "raw_body_complete": len(response.raw) <= MAX_REST_RESPONSE_BYTES,
            "raw_json_sha256": sha256_bytes(response.raw),
            "raw_json_base64": base64.b64encode(response.raw).decode("ascii"),
            "clock_error_type": type(clock_error).__name__,
            "clock_error_message": str(clock_error)[:1_000],
        },
    )
    state.rest_responses_audited += 1
    if len(response.raw) > MAX_REST_RESPONSE_BYTES:
        raise ParityCaptureError("REST response exceeds frozen byte bound")
    if state.raw_bytes > MAX_CAPTURE_RAW_BYTES:
        raise ParityCaptureError("capture raw bytes exceed frozen bound")


def _record_rest_response(
    state: CaptureState,
    handle: Any,
    response: RestTransportResponse,
    *,
    ordinal: int,
    start: ClockSample,
    end: ClockSample,
    final: bool,
) -> RestSnapshot:
    state.raw_bytes += len(response.raw)
    _write_raw_line(
        handle,
        {
            "record_type": "response",
            "ordinal": ordinal,
            "final": final,
            "request_clock_source": start.source,
            "request_clock_ordinal": start.ordinal,
            "response_clock_source": end.source,
            "response_clock_ordinal": end.ordinal,
            "request_start_utc_ns": start.utc_ns,
            "request_start_monotonic_ns": start.monotonic_ns,
            "response_end_utc_ns": end.utc_ns,
            "response_end_monotonic_ns": end.monotonic_ns,
            "request_start_clock_uncertainty_ns": start.uncertainty_ns,
            "response_end_clock_uncertainty_ns": end.uncertainty_ns,
            "http_status": response.status,
            "final_url": response.final_url,
            "content_type": response.content_type,
            "location": response.location,
            "raw_body_complete": len(response.raw) <= MAX_REST_RESPONSE_BYTES,
            "raw_json_sha256": sha256_bytes(response.raw),
            "raw_json_base64": base64.b64encode(response.raw).decode("ascii"),
        },
    )
    state.rest_responses_audited += 1
    _complete_rest_clock(state, ordinal, end)
    if len(response.raw) > MAX_REST_RESPONSE_BYTES:
        raise ParityCaptureError("REST response exceeds frozen byte bound")
    if state.raw_bytes > MAX_CAPTURE_RAW_BYTES:
        raise ParityCaptureError("capture raw bytes exceed frozen bound")
    if end.monotonic_ns < start.monotonic_ns:
        raise ParityCaptureError("REST monotonic clock reversed")
    for sample in (start, end):
        if _utc_datetime_from_ns(sample.utc_ns).date().isoformat() != state.capture_day:
            raise ParityCaptureError("capture crossed UTC midnight")
    if response.status != 200 or response.final_url != REST_URL:
        raise ParityCaptureError("unexpected REST status or redirect")
    trades = parse_rest_response(response.raw)
    snapshot = RestSnapshot(
        ordinal=ordinal,
        request_start_monotonic_ns=start.monotonic_ns,
        response_end_monotonic_ns=end.monotonic_ns,
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
    clock: ClockReader = PROCESS_CLOCK,
) -> RestSnapshot:
    enforce_disk_guard()
    ordinal = state.rest_attempts_started + 1
    start = _local_clock_sample("rest_request_start", ordinal, clock)
    _begin_rest_attempt(state, handle, start)
    try:
        response = await asyncio.to_thread(fetch)
    except Exception as exc:
        try:
            end = _local_clock_sample("rest_response_end", ordinal, clock)
        except Exception as clock_exc:
            _record_rest_transport_clock_error(
                state,
                handle,
                ordinal=ordinal,
                transport_error=exc,
                clock_error=clock_exc,
            )
            raise clock_exc from exc
        _record_rest_transport_error(
            state,
            handle,
            ordinal=ordinal,
            end=end,
            error=exc,
        )
        raise
    try:
        end = _local_clock_sample("rest_response_end", ordinal, clock)
    except Exception as exc:
        _record_rest_response_clock_error(
            state,
            handle,
            response,
            ordinal=ordinal,
            start=start,
            final=final,
            clock_error=exc,
        )
        raise
    return _record_rest_response(
        state,
        handle,
        response,
        ordinal=ordinal,
        start=start,
        end=end,
        final=final,
    )


async def _wait_until_clock(
    clock: ClockReader,
    deadline_ns: int,
    *,
    stop: asyncio.Event | None = None,
) -> bool:
    """Wait on asyncio's clock while gating completion on the frozen clock."""

    while True:
        remaining = (deadline_ns - clock.monotonic_ns()) / 1e9
        if remaining <= 0:
            return False
        timeout = min(remaining, RAW_WAIT_QUANTUM_SECONDS)
        if stop is None:
            await asyncio.sleep(timeout)
            continue
        try:
            await asyncio.wait_for(stop.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            continue


async def _recv_until_clock(socket: Any, clock: ClockReader, deadline_ns: int) -> Any:
    receive_task = asyncio.create_task(socket.recv())
    try:
        while True:
            remaining = (deadline_ns - clock.monotonic_ns()) / 1e9
            if remaining <= 0:
                return None
            done, _ = await asyncio.wait(
                {receive_task},
                timeout=min(remaining, RAW_WAIT_QUANTUM_SECONDS),
            )
            if receive_task in done:
                return receive_task.result()
    finally:
        if not receive_task.done():
            receive_task.cancel()
        await asyncio.gather(receive_task, return_exceptions=True)


async def _heartbeat(
    socket: Any,
    deadline_ns: int,
    clock: ClockReader = PROCESS_CLOCK,
) -> None:
    interval_ns = HEARTBEAT_SECONDS * 1_000_000_000
    next_due_ns = clock.monotonic_ns() + interval_ns
    while next_due_ns < deadline_ns:
        await _wait_until_clock(clock, next_due_ns)
        if clock.monotonic_ns() >= deadline_ns:
            return
        await socket.send(json.dumps({"op": "ping"}, separators=(",", ":")))
        next_due_ns += interval_ns


async def _rest_poller(
    state: CaptureState,
    handle: Any,
    deadline_ns: int,
    stop: asyncio.Event,
    *,
    fetch: FetchRest,
    clock: ClockReader = PROCESS_CLOCK,
) -> None:
    next_due_ns = clock.monotonic_ns()
    interval_ns = REST_INTERVAL_SECONDS * 1_000_000_000
    while next_due_ns < deadline_ns and not stop.is_set():
        if await _wait_until_clock(clock, next_due_ns, stop=stop):
            return
        if stop.is_set():
            return
        await _fetch_and_record_rest(
            state,
            handle,
            final=False,
            fetch=fetch,
            clock=clock,
        )
        next_due_ns += interval_ns


async def _capture_network(
    state: CaptureState,
    ws_handle: Any,
    rest_handle: Any,
    *,
    fetch: FetchRest = fetch_rest,
    clock: ClockReader = PROCESS_CLOCK,
) -> None:
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
            if isinstance(message, str):
                raw = message.encode("utf-8")
                frame_type: Literal["text", "binary"] = "text"
            elif isinstance(message, bytes):
                raw = message
                frame_type = "binary"
            else:
                raise ParityCaptureError("WebSocket frame type is unsupported")
            kind, _, receipt = _sample_and_record_ws_message(
                state,
                ws_handle,
                raw,
                frame_type,
                clock,
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
        stop_rest = asyncio.Event()
        rest_task = asyncio.create_task(
            _rest_poller(
                state,
                rest_handle,
                deadline_ns,
                stop_rest,
                fetch=fetch,
                clock=clock,
            )
        )
        heartbeat_task = asyncio.create_task(_heartbeat(socket, deadline_ns, clock))
        try:
            while True:
                message = await _recv_until_clock(socket, clock, deadline_ns)
                if message is None:
                    break
                if isinstance(message, str):
                    raw = message.encode("utf-8")
                    frame_type = "text"
                elif isinstance(message, bytes):
                    raw = message
                    frame_type = "binary"
                else:
                    raise ParityCaptureError("WebSocket frame type is unsupported")
                kind, _, _ = _sample_and_record_ws_message(
                    state,
                    ws_handle,
                    raw,
                    frame_type,
                    clock,
                )
                if kind == "subscribe":
                    raise ParityCaptureError("duplicate WebSocket subscription ack")
            await rest_task
            await heartbeat_task
        finally:
            stop_rest.set()
            if not rest_task.done():
                await asyncio.gather(rest_task, return_exceptions=True)
            if not heartbeat_task.done():
                heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)

    await _fetch_and_record_rest(
        state,
        rest_handle,
        final=True,
        fetch=fetch,
        clock=clock,
    )


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
    clock_metadata: Mapping[str, Any],
    clock_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    provider_binding_passed = (
        clock_metadata.get("provider_id")
        == WindowsHostRawClock.provider_id
        and clock_metadata.get("monotonic_source")
        == WindowsHostRawClock.monotonic_source
        and clock_metadata.get("utc_source") == WindowsHostRawClock.utc_source
        and clock_metadata.get("powershell_path") == str(POWERSHELL_PATH)
        and clock_metadata.get("powershell_script_sha256")
        == HOST_CLOCK_SCRIPT_SHA256
        and clock_metadata.get("shell") is False
        and clock_metadata.get("fallback_used") is False
        and clock_metadata.get("warmup_sample") is not None
    )
    if not provider_binding_passed or clock_preflight.get("decision") != "PASS":
        raise ParityCaptureError("capture clock provider manifest binding differs")
    if parity["decision"].startswith("REST_WS_PARITY_PASS") and not clock_metadata.get(
        "closed_cleanly"
    ):
        raise ParityCaptureError("passing manifest requires clean clock-provider close")
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
            "deadline_clock": clock_metadata["monotonic_source"],
        },
        "clock_provider": dict(clock_metadata),
        "clock_preflight": dict(clock_preflight),
        "capture": {
            "ws_messages": state.ws_messages,
            "ws_subscription_acks": state.ws_subscription_acks,
            "ws_pongs": state.ws_pongs,
            "raw_json_bytes": state.raw_bytes,
            "clock_samples": len(state.clock_samples),
            "rest_attempts_started": state.rest_attempts_started,
            "rest_attempts_completed": state.rest_attempts_completed,
            "rest_responses_audited": state.rest_responses_audited,
            "first_trade_monotonic_ns": state.first_trade_monotonic_ns,
            "last_trade_monotonic_ns": state.last_trade_monotonic_ns,
            "boundary_clock_samples": [
                {
                    "source": sample.source,
                    "ordinal": sample.ordinal,
                    "monotonic_ns": sample.monotonic_ns,
                    "utc_ns": sample.utc_ns,
                    "uncertainty_ns": sample.uncertainty_ns,
                }
                for sample in state.clock_samples
                if sample.source in {"capture_start", "capture_end"}
            ],
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
            "clock_correction_path": str(CLOCK_CORRECTION_PATH),
            "clock_correction_sha256": CLOCK_CORRECTION_SHA256,
            "clock_preflight_result_path": str(CLOCK_PREFLIGHT_RESULT_PATH),
            "clock_preflight_result_sha256": CLOCK_PREFLIGHT_RESULT_SHA256,
            "clock_preflight_result_manifest_hash": (
                CLOCK_PREFLIGHT_RESULT_MANIFEST_HASH
            ),
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


def _merge_runtime_error(
    current: Exception | None, additional: Exception
) -> Exception:
    if current is None:
        return additional
    return ParityCaptureError(
        f"{type(current).__name__}: {str(current)[:400]} | "
        f"{type(additional).__name__}: {str(additional)[:400]}"
    )


def run_capture() -> tuple[Path, dict[str, Any]]:
    validate_bindings()
    enforce_disk_guard()
    clock = WindowsHostRawClock()
    try:
        clock.start()
        clock_preflight = run_clock_preflight(clock)
        disk_before = enforce_disk_guard()
        capture_start = _local_clock_sample("capture_start", 1, clock)
        now = _utc_datetime_from_ns(capture_start.utc_ns)
        output_dir = _capture_output_dir(now)
        output_dir.mkdir(parents=True, exist_ok=False)
    except Exception:
        try:
            clock.close()
        except Exception:
            pass
        raise
    state = CaptureState(
        capture_day=now.date().isoformat(),
        started_at_utc=_utc_iso_from_ns(capture_start.utc_ns),
        output_dir=output_dir,
    )
    state.clock_samples.append(capture_start)
    ws_path = output_dir / "websocket_messages.ndjson.gz"
    rest_path = output_dir / "rest_responses.ndjson.gz"
    runtime_error: Exception | None = None
    try:
        with gzip.open(ws_path, "wb", compresslevel=6) as ws_handle, gzip.open(
            rest_path, "wb", compresslevel=6
        ) as rest_handle:
            asyncio.run(
                _capture_network(
                    state,
                    ws_handle,
                    rest_handle,
                    clock=clock,
                )
            )
    except Exception as exc:  # The immutable failure manifest is the audit trail.
        runtime_error = exc
    try:
        capture_end = _local_clock_sample("capture_end", 1, clock)
        state.clock_samples.append(capture_end)
        state.ended_at_utc = _utc_iso_from_ns(capture_end.utc_ns)
    except Exception as exc:
        runtime_error = _merge_runtime_error(runtime_error, exc)
    try:
        clock.close()
    except Exception as exc:
        runtime_error = _merge_runtime_error(runtime_error, exc)
    clock_metadata = clock.metadata()
    clock_integrity = evaluate_clock_integrity(
        state.clock_samples,
        ClockLedgerExpectation(
            websocket_messages=state.ws_messages,
            rest_attempts=state.rest_attempts_started,
            rest_attempts_completed=state.rest_attempts_completed,
            rest_responses_audited=state.rest_responses_audited,
        ),
    )
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
        clock_metadata=clock_metadata,
        clock_preflight=clock_preflight,
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
