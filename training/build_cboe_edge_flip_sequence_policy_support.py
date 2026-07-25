"""Build the frozen outcome-blind CEFS-D1 source-language support result."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from training import preregister_cboe_edge_flip_sequence_policy as prereg


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = "training/build_cboe_edge_flip_sequence_policy_support.py"
TEST_PATH = "tests/test_build_cboe_edge_flip_sequence_policy_support.py"
CONTRACT_PATH = "docs/cefs-d1-source-support-implementation-contract-2026-07-25.md"
CONTRACT_SHA256 = (
    "a109e256cf9742c1a6d90c9762455690bd54e789bfe3afe0e695db67f442f9d2"
)
CONTRACT_COMMIT = "3a27658e81409e97ae007178981256c46f68b9aa"
PREREGISTRATION_PATH = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = (
    "5e515663e99ef4aa322cae25cfb2c07f69b3e24f289bc2f0f79463aca64a8878"
)
PREREGISTRATION_MANIFEST_HASH = (
    "9aa7c891ec241d4733db215068bed3507f41c03cbae7198c906a079ddb6467bf"
)
PREREGISTRATION_COMMIT = "d60d9744e82f48350be506f1da63bcb2e706cf03"
PREREGISTRATION_PRODUCER_COMMIT = (
    "ec8eae23226d39f0c62b6c5711d6080f2bf990a4"
)
PREREGISTRATION_PRODUCER_SHA256 = (
    "1c4668e72846eadf66011c582c62e8574af3679204500c1ff9631101ecbb7ac1"
)
EXECUTION_SEAL_PATH = (
    "results/cefs_d1_source_support_execution_seal_2026-07-25.json"
)
SOURCE_OUTPUT = "data/cboe_edge_flip_sequence_policy_source_2020_2023.csv.gz"
CONTROL_OUTPUT = "data/cboe_edge_flip_sequence_policy_controls_2020_2023.csv.gz"
PASS_REPORT = (
    "results/cboe_edge_flip_sequence_policy_source_support_2026-07-25.json"
)
REJECTION_REPORT = (
    "results/cboe_edge_flip_sequence_policy_source_rejection_2026-07-25.json"
)
SEAL_PROTOCOL = "cefs_d1_source_support_execution_seal_v1"
RESULT_PROTOCOL = "cefs_d1_source_support_result_v1"
GATE_NAMES = (
    "authority_forbidden_access",
    "schema_chronology",
    "schedule_support",
    "primitive_edge_support",
    "state_diversity_stability",
    "source_only_controls",
    "determinism_append_replay",
)
DETAIL_KEYS_BY_GATE = {
    2: "parser",
    3: "schedule",
    4: "edge_support",
    5: "diversity_stability",
    6: "controls",
    7: "determinism_append_replay",
}

PREFIX_CUTOFFS = (
    date(2021, 1, 1),
    date(2022, 1, 1),
    date(2023, 1, 1),
    date(2024, 1, 1),
)
SOURCE_START = date(2020, 1, 1)
SOURCE_END = date(2024, 1, 1)
HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
DATE_TEXT = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z", re.ASCII)
DECIMAL_TEXT = re.compile(r"[0-9]+(?:\.[0-9]+)?\Z", re.ASCII)
INTEGER_TEXT = re.compile(r"[0-9]+\Z", re.ASCII)
UTC_TEXT = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z",
    re.ASCII,
)

SOURCE_OUTPUT_COLUMNS = (
    "observation_date",
    "available_utc",
    "entry_utc",
    "exit_utc",
    "reservation_state",
    "role",
    "model_eligible",
    "current_signature",
    "sequence_signature",
    "prompt_target_flat",
    "prompt_target_long",
    "prompt_target_short",
)
CONTROL_OUTPUT_COLUMNS = (
    "observation_date",
    "position_context",
    "control_id",
    "control_prompt",
    "control_prompt_sha256",
    "semantic_difference",
)
ROLE_WINDOWS = {
    "TRAIN": (
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2022, 1, 1, tzinfo=timezone.utc),
    ),
    "TEST": (
        datetime(2022, 1, 1, tzinfo=timezone.utc),
        datetime(2023, 1, 1, tzinfo=timezone.utc),
    ),
    "EVAL": (
        datetime(2023, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, tzinfo=timezone.utc),
    ),
}
FORBIDDEN_COUNTER_NAMES = (
    "post_2023_source_non_date_rows_opened",
    "market_rows_opened",
    "funding_rows_opened",
    "future_return_rows_built",
    "reward_rows_built",
    "model_rows_built",
    "selected_action_rows_built",
    "trade_rows_built",
    "pnl_cagr_mdd_values_computed",
    "comparator_action_rows_opened",
)


@dataclass(frozen=True)
class ParseAudit:
    physical_rows: int
    date_only_rows: int
    value_rows: int
    first_date: str
    last_date: str
    prefix_rows: Mapping[str, int]
    post_2023_non_date_rows: int = 0


@dataclass(frozen=True)
class TermRow:
    observation_date: date
    vix9d: Decimal
    vix: Decimal
    vix3m: Decimal


@dataclass(frozen=True)
class TailRow:
    observation_date: date
    skew: Decimal
    vvix: Decimal
    vix: Decimal


@dataclass(frozen=True)
class FlowRow:
    observation_date: date
    total_pcr: Decimal
    index_pcr: Decimal
    equity_pcr: Decimal
    vix_pcr: Decimal
    spx_pcr: Decimal
    index_volume: int
    vix_volume: int


@dataclass(frozen=True)
class CommonRow:
    observation_date: date
    vix9d: Decimal
    vix: Decimal
    vix3m: Decimal
    skew: Decimal
    vvix: Decimal
    total_pcr: Decimal
    index_pcr: Decimal
    equity_pcr: Decimal
    vix_pcr: Decimal
    spx_pcr: Decimal
    index_volume: int
    vix_volume: int


@dataclass(frozen=True)
class EdgeState:
    observation_date: date
    levels: tuple[str, ...]


@dataclass(frozen=True)
class ScheduleRow:
    observation_date: date
    available_utc: datetime
    entry_utc: datetime
    exit_utc: datetime
    reservation_state: str
    role: str
    model_eligible: bool
    states: tuple[tuple[str, ...], ...]
    current_signature: str
    sequence_signature: str
    prompts: tuple[str, ...]


@dataclass(frozen=True)
class ControlRow:
    observation_date: date
    position_context: str
    control_id: str
    prompt: str
    prompt_sha256: str
    semantic_difference: bool


@dataclass(frozen=True)
class PanelResult:
    rows: Mapping[date, Any]
    prefix_rows: Mapping[str, Mapping[date, Any]]
    audit: ParseAudit


@dataclass(frozen=True)
class SourceInputs:
    term: PanelResult
    tail: PanelResult
    flow: PanelResult


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(repository_path(path).read_bytes()).hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _assert_tracked_clean(path: str) -> str:
    _git_output("ls-files", "--error-unmatch", "--", path)
    for args in (
        ("diff", "--quiet", "--", path),
        ("diff", "--cached", "--quiet", "--", path),
    ):
        if subprocess.run(
            ("git", *args),
            cwd=REPOSITORY_ROOT,
            check=False,
        ).returncode:
            raise RuntimeError(f"CEFS sealed file is dirty: {path}")
    return _git_output("log", "-1", "--format=%H", "--", path)


def _worktree_clean() -> bool:
    return not _git_output("status", "--porcelain", "--untracked-files=all")


def _safe_output_path(path: str | Path) -> Path:
    candidate = Path(path)
    if (
        str(path).startswith("~")
        or candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError("CEFS output path must be repository-relative")
    target = REPOSITORY_ROOT / candidate
    root = REPOSITORY_ROOT.resolve(strict=True)
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise RuntimeError("CEFS output path escapes repository") from error
    current = REPOSITORY_ROOT
    for part in candidate.parent.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError("CEFS output parent contains a symlink")
    return target


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stage_bytes(output: Path, content: bytes) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.cefs-stage-",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        handle = os.fdopen(descriptor, "wb")
        descriptor = -1
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _publish_staged_file(temporary: Path, output: Path) -> None:
    os.link(temporary, output, follow_symlinks=False)


def write_once_bytes(path: str | Path, content: bytes) -> str:
    output = _safe_output_path(path)
    digest = hashlib.sha256(content).hexdigest()
    if output.exists():
        if output.is_symlink() or not output.is_file():
            raise RuntimeError("CEFS output is not a regular file")
        if output.read_bytes() != content:
            raise RuntimeError(f"CEFS write-once artifact drift: {output}")
        return digest
    temporary = _stage_bytes(output, content)
    published = False
    try:
        try:
            _publish_staged_file(temporary, output)
            published = True
            _fsync_directory(output.parent)
        except FileExistsError:
            if (
                output.is_symlink()
                or not output.is_file()
                or output.read_bytes() != content
            ):
                raise RuntimeError(
                    f"CEFS concurrent write-once artifact drift: {output}"
                )
    except BaseException:
        if published:
            output.unlink(missing_ok=True)
            _fsync_directory(output.parent)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_once_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    return write_once_bytes(path, _json_bytes(payload))


def publish_write_once_transaction(
    artifacts: Sequence[tuple[str | Path, bytes]],
) -> dict[str, str]:
    if not artifacts:
        raise ValueError("CEFS publication transaction is empty")
    outputs = [_safe_output_path(path) for path, _ in artifacts]
    if len(set(outputs)) != len(outputs):
        raise RuntimeError("CEFS publication transaction has duplicate outputs")
    existing = [output for output in outputs if output.exists()]
    if existing:
        raise RuntimeError(
            "CEFS publication transaction requires absent outputs: "
            + ", ".join(str(path) for path in existing)
        )
    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for output, (_, content) in zip(outputs, artifacts):
            staged.append((_stage_bytes(output, content), output))
        for temporary, output in staged:
            _publish_staged_file(temporary, output)
            published.append(output)
            _fsync_directory(output.parent)
    except BaseException:
        for output in reversed(published):
            output.unlink(missing_ok=True)
        for parent in {output.parent for output in published}:
            _fsync_directory(parent)
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return {
        str(path): hashlib.sha256(content).hexdigest()
        for path, content in artifacts
    }


def build_execution_seal() -> dict[str, Any]:
    runner_commit = _assert_tracked_clean(RUNNER_PATH)
    test_commit = _assert_tracked_clean(TEST_PATH)
    if runner_commit != test_commit:
        raise RuntimeError("CEFS runner and tests must share one commit")
    if not _worktree_clean():
        raise RuntimeError("CEFS worktree must be clean before seal creation")
    prereg_payload = json.loads(repository_path(PREREGISTRATION_PATH).read_text())
    preregistration_producer = _validate_current_preregistration_producer(
        prereg_payload
    )
    core = {
        "protocol_version": SEAL_PROTOCOL,
        "policy_id": prereg.POLICY_ID,
        "contract": {
            "path": CONTRACT_PATH,
            "commit": CONTRACT_COMMIT,
            "sha256": CONTRACT_SHA256,
        },
        "preregistration": {
            "path": PREREGISTRATION_PATH,
            "commit": PREREGISTRATION_COMMIT,
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "preregistration_producer": preregistration_producer,
        "runner": {
            "path": RUNNER_PATH,
            "commit": runner_commit,
            "sha256": sha256_file(RUNNER_PATH),
        },
        "tests": {
            "path": TEST_PATH,
            "commit": test_commit,
            "sha256": sha256_file(TEST_PATH),
        },
        "source_values_opened": False,
        "outcomes_opened": False,
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def create_execution_seal() -> dict[str, Any]:
    payload = build_execution_seal()
    write_once_json(EXECUTION_SEAL_PATH, payload)
    return payload


def validate_execution_seal() -> dict[str, Any]:
    path = repository_path(EXECUTION_SEAL_PATH)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("CEFS execution seal is missing")
    payload = json.loads(path.read_text())
    expected = build_execution_seal()
    if payload != expected:
        raise RuntimeError("CEFS execution seal differs from committed runner")
    for key in ("source_values_opened", "outcomes_opened"):
        if payload.get(key) is not False:
            raise RuntimeError("CEFS execution seal opened forbidden data")
    return payload


def _validate_current_preregistration_producer(
    prereg_payload: Mapping[str, Any],
) -> dict[str, str]:
    sealed = prereg_payload.get("authority", {}).get("producer")
    expected = {
        "path": prereg.PRODUCER_SCRIPT,
        "commit": PREREGISTRATION_PRODUCER_COMMIT,
        "sha256": PREREGISTRATION_PRODUCER_SHA256,
    }
    if sealed != expected:
        raise RuntimeError("CEFS preregistration producer binding missing")
    current = prereg.producer_binding()
    if current != sealed:
        raise RuntimeError(
            "CEFS current preregistration producer differs from sealed artifact"
        )
    return current


def validate_frozen_authority() -> dict[str, Any]:
    prereg_payload = json.loads(repository_path(PREREGISTRATION_PATH).read_text())
    preregistration_producer = _validate_current_preregistration_producer(
        prereg_payload
    )
    bindings = {
        CONTRACT_PATH: (CONTRACT_SHA256, CONTRACT_COMMIT),
        prereg.BOUNDARY_DOCUMENT: (
            prereg.BOUNDARY_DOCUMENT_SHA256,
            prereg.BOUNDARY_COMMIT,
        ),
        prereg.PRODUCER_SCRIPT: (
            preregistration_producer["sha256"],
            preregistration_producer["commit"],
        ),
        PREREGISTRATION_PATH: (
            PREREGISTRATION_SHA256,
            PREREGISTRATION_COMMIT,
        ),
    }
    for path, (digest, commit) in bindings.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"CEFS frozen authority hash mismatch: {path}")
        if _git_output("log", "-1", "--format=%H", "--", path) != commit:
            raise RuntimeError(f"CEFS frozen authority commit mismatch: {path}")
    prereg.validate_manifest(prereg_payload)
    if prereg_payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("CEFS preregistration manifest hash mismatch")
    prereg._validate_sealed_producer(prereg_payload)
    prereg._validate_source_anchors()
    return {
        "boundary": {
            "path": prereg.BOUNDARY_DOCUMENT,
            "commit": prereg.BOUNDARY_COMMIT,
            "sha256": prereg.BOUNDARY_DOCUMENT_SHA256,
        },
        "contract": {
            "path": CONTRACT_PATH,
            "commit": CONTRACT_COMMIT,
            "sha256": CONTRACT_SHA256,
        },
        "preregistration": {
            "path": PREREGISTRATION_PATH,
            "commit": PREREGISTRATION_COMMIT,
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "preregistration_producer": preregistration_producer,
    }


def _parse_date(token: str) -> date:
    if not DATE_TEXT.fullmatch(token):
        raise ValueError("CEFS observation date grammar failed")
    parsed = date.fromisoformat(token)
    if parsed.isoformat() != token:
        raise ValueError("CEFS observation date round-trip failed")
    return parsed


def _decimal(token: str, *, field: str) -> Decimal:
    if not DECIMAL_TEXT.fullmatch(token):
        raise ValueError(f"CEFS decimal grammar failed: {field}")
    try:
        value = Decimal(token)
    except InvalidOperation as error:
        raise ValueError(f"CEFS decimal parse failed: {field}") from error
    if not value.is_finite() or value <= 0:
        raise ValueError(f"CEFS decimal positivity failed: {field}")
    return value


def _integer(token: str, *, field: str) -> int:
    if not INTEGER_TEXT.fullmatch(token):
        raise ValueError(f"CEFS integer grammar failed: {field}")
    value = int(token)
    if value <= 0:
        raise ValueError(f"CEFS integer positivity failed: {field}")
    return value


def _open_csv(path: str | Path) -> Iterable[list[str]]:
    source = repository_path(path)
    if source.suffix == ".gz":
        handle = gzip.open(source, "rt", encoding="utf-8", newline="")
    else:
        handle = source.open("rt", encoding="utf-8", newline="")
    with handle:
        yield from csv.reader(handle, strict=True)


def _seal_due_prefixes(
    raw_date: str,
    rows: Mapping[date, Any],
    snapshots: dict[str, Mapping[date, Any]],
) -> None:
    try:
        encoded = raw_date.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("CEFS observation date grammar failed") from error
    if len(encoded) != 10 or not DATE_TEXT.fullmatch(raw_date):
        raise ValueError("CEFS observation date grammar failed")
    for cutoff in PREFIX_CUTOFFS:
        key = cutoff.isoformat()
        if key not in snapshots and raw_date >= key:
            snapshots[key] = MappingProxyType(dict(rows))


def _read_panel(
    path: str | Path,
    *,
    expected_header: Sequence[str],
    parser: Any,
) -> PanelResult:
    reader = iter(_open_csv(path))
    try:
        header = next(reader)
    except StopIteration as error:
        raise ValueError("CEFS source is empty") from error
    if tuple(header) != tuple(expected_header):
        raise ValueError("CEFS physical header mismatch")
    rows: dict[date, Any] = {}
    snapshots: dict[str, Mapping[date, Any]] = {}
    previous: date | None = None
    physical_rows = 0
    date_only_rows = 0
    value_rows = 0
    first: date | None = None
    last: date | None = None
    cutoffs = list(PREFIX_CUTOFFS)
    for physical in reader:
        if len(physical) != len(header):
            raise ValueError("CEFS physical row width mismatch")
        _seal_due_prefixes(physical[0], rows, snapshots)
        day = _parse_date(physical[0])
        if previous is not None and day <= previous:
            raise ValueError("CEFS source dates are not strictly increasing")
        previous = day
        first = first or day
        last = day
        physical_rows += 1
        if day < SOURCE_START:
            date_only_rows += 1
            continue
        if day >= SOURCE_END:
            raise ValueError("CEFS post-2023 source row is forbidden")
        rows[day] = parser(header, physical, day)
        value_rows += 1
    for cutoff in cutoffs:
        snapshots.setdefault(
            cutoff.isoformat(),
            MappingProxyType(dict(rows)),
        )
    if first is None or last is None:
        raise ValueError("CEFS source has no physical rows")
    audit = ParseAudit(
        physical_rows=physical_rows,
        date_only_rows=date_only_rows,
        value_rows=value_rows,
        first_date=first.isoformat(),
        last_date=last.isoformat(),
        prefix_rows={key: len(value) for key, value in snapshots.items()},
    )
    return PanelResult(rows=rows, prefix_rows=snapshots, audit=audit)


def _column_map(header: Sequence[str]) -> dict[str, int]:
    return {column: index for index, column in enumerate(header)}


def _parse_term(header: Sequence[str], row: Sequence[str], day: date) -> TermRow:
    index = _column_map(header)
    return TermRow(
        observation_date=day,
        vix9d=_decimal(row[index["VIX9D_close"]], field="VIX9D_close"),
        vix=_decimal(row[index["VIX_close"]], field="VIX_close"),
        vix3m=_decimal(row[index["VIX3M_close"]], field="VIX3M_close"),
    )


def _parse_tail(header: Sequence[str], row: Sequence[str], day: date) -> TailRow:
    index = _column_map(header)
    return TailRow(
        observation_date=day,
        skew=_decimal(row[index["SKEW_close"]], field="SKEW_close"),
        vvix=_decimal(row[index["VVIX_close"]], field="VVIX_close"),
        vix=_decimal(row[index["VIX_close"]], field="VIX_close"),
    )


def _parse_flow(header: Sequence[str], row: Sequence[str], day: date) -> FlowRow:
    index = _column_map(header)
    response_hash = row[index["response_sha256"]]
    if not HEX64.fullmatch(response_hash):
        raise ValueError("CEFS response_sha256 grammar failed")
    return FlowRow(
        observation_date=day,
        total_pcr=_decimal(row[index["total_pcr"]], field="total_pcr"),
        index_pcr=_decimal(row[index["index_pcr"]], field="index_pcr"),
        equity_pcr=_decimal(row[index["equity_pcr"]], field="equity_pcr"),
        vix_pcr=_decimal(row[index["vix_pcr"]], field="vix_pcr"),
        spx_pcr=_decimal(row[index["spx_pcr"]], field="spx_pcr"),
        index_volume=_integer(
            row[index["index_volume"]],
            field="index_volume",
        ),
        vix_volume=_integer(row[index["vix_volume"]], field="vix_volume"),
    )


def load_source_inputs() -> SourceInputs:
    return SourceInputs(
        term=_read_panel(
            prereg.TERM_SOURCE,
            expected_header=prereg.TERM_HEADER,
            parser=_parse_term,
        ),
        tail=_read_panel(
            prereg.TAIL_SOURCE,
            expected_header=prereg.TAIL_HEADER,
            parser=_parse_tail,
        ),
        flow=_read_panel(
            prereg.FLOW_SOURCE,
            expected_header=prereg.FLOW_HEADER,
            parser=_parse_flow,
        ),
    )


def _prefix_inputs(inputs: SourceInputs, cutoff: str) -> SourceInputs:
    def panel(source: PanelResult) -> PanelResult:
        rows = source.prefix_rows[cutoff]
        return PanelResult(rows=rows, prefix_rows={}, audit=source.audit)

    return SourceInputs(
        term=panel(inputs.term),
        tail=panel(inputs.tail),
        flow=panel(inputs.flow),
    )


def join_common_rows(inputs: SourceInputs) -> list[CommonRow]:
    common_dates = sorted(
        set(inputs.term.rows)
        .intersection(inputs.tail.rows)
        .intersection(inputs.flow.rows)
    )
    joined: list[CommonRow] = []
    for day in common_dates:
        term = inputs.term.rows[day]
        tail = inputs.tail.rows[day]
        flow = inputs.flow.rows[day]
        if term.vix != tail.vix:
            raise ValueError("CEFS term/tail VIX identity failed")
        joined.append(
            CommonRow(
                observation_date=day,
                vix9d=term.vix9d,
                vix=term.vix,
                vix3m=term.vix3m,
                skew=tail.skew,
                vvix=tail.vvix,
                total_pcr=flow.total_pcr,
                index_pcr=flow.index_pcr,
                equity_pcr=flow.equity_pcr,
                vix_pcr=flow.vix_pcr,
                spx_pcr=flow.spx_pcr,
                index_volume=flow.index_volume,
                vix_volume=flow.vix_volume,
            )
        )
    return joined


def compare(left: Decimal | int, right: Decimal | int) -> str:
    if left < right:
        return "LOWER"
    if left > right:
        return "HIGHER"
    return "EQUAL"


def compare_ratio(
    a: Decimal | int,
    b: Decimal | int,
    c: Decimal | int,
    d: Decimal | int,
) -> str:
    return compare(a * d, c * b)


def edge_state(previous: CommonRow, current: CommonRow) -> EdgeState:
    levels = (
        compare(current.vix9d, current.vix),
        compare(current.vix, current.vix3m),
        compare_ratio(current.vix9d, current.vix, previous.vix9d, previous.vix),
        compare_ratio(current.vix, current.vix3m, previous.vix, previous.vix3m),
        compare(current.skew, previous.skew),
        compare_ratio(current.vvix, current.vix, previous.vvix, previous.vix),
        compare(current.total_pcr, previous.total_pcr),
        compare(current.index_pcr, previous.index_pcr),
        compare(current.equity_pcr, previous.equity_pcr),
        compare(current.vix_pcr, previous.vix_pcr),
        compare(current.spx_pcr, previous.spx_pcr),
        compare_ratio(
            current.vix_volume,
            current.index_volume,
            previous.vix_volume,
            previous.index_volume,
        ),
    )
    if len(levels) != len(prereg.EDGE_NAMES):
        raise RuntimeError("CEFS edge count drift")
    return EdgeState(observation_date=current.observation_date, levels=levels)


def build_edge_states(rows: Sequence[CommonRow]) -> list[EdgeState]:
    return [edge_state(rows[index - 1], rows[index]) for index in range(1, len(rows))]


def _state_mapping(
    states: Sequence[tuple[str, ...]],
) -> dict[str, dict[str, str]]:
    if len(states) != len(prereg.STATE_LABELS):
        raise ValueError("CEFS sequence state count changed")
    return {
        state_label: {
            edge: level for edge, level in zip(prereg.EDGE_NAMES, levels)
        }
        for state_label, levels in zip(prereg.STATE_LABELS, states)
    }


def _role(entry: datetime, exit_at: datetime) -> str:
    for role, (start, end) in ROLE_WINDOWS.items():
        if start <= entry and exit_at <= end:
            return role
    return "ROLE_CROSSING"


def build_schedules(rows: Sequence[CommonRow]) -> list[ScheduleRow]:
    edges = build_edge_states(rows)
    candidates: list[tuple[EdgeState, tuple[tuple[str, ...], ...]]] = []
    for index in range(4, len(edges)):
        sequence = tuple(item.levels for item in edges[index - 4 : index + 1])
        candidates.append((edges[index], sequence))
    schedules: list[ScheduleRow] = []
    prior_accepted_exit: datetime | None = None
    for current, states in candidates:
        clock = prereg.fixed_clock(current.observation_date)
        entry = clock["entry_utc"]
        exit_at = clock["exit_utc"]
        accepted = prior_accepted_exit is None or entry >= prior_accepted_exit
        if accepted:
            prior_accepted_exit = exit_at
        role = _role(entry, exit_at) if accepted else "SUPPRESSED"
        model_eligible = accepted and role in ROLE_WINDOWS
        mapping = _state_mapping(states)
        prompts = tuple(
            prereg.serialize_prompt(mapping, position)
            for position in prereg.POSITION_CONTEXTS
        )
        current_signature = "|".join(states[-1])
        sequence_signature = "/".join("|".join(state) for state in states)
        schedules.append(
            ScheduleRow(
                observation_date=current.observation_date,
                available_utc=clock["available_utc"],
                entry_utc=entry,
                exit_utc=exit_at,
                reservation_state="ACCEPTED" if accepted else "SUPPRESSED_OVERLAP",
                role=role,
                model_eligible=model_eligible,
                states=states,
                current_signature=current_signature,
                sequence_signature=sequence_signature,
                prompts=prompts,
            )
        )
    return schedules


def _masked(states: Sequence[tuple[str, ...]], keep: set[int]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(value if index in keep else "MASKED" for index, value in enumerate(state))
        for state in states
    )


def _control_states(
    states: tuple[tuple[str, ...], ...],
    control_id: str,
) -> tuple[tuple[str, ...], ...]:
    if control_id == "reverse_sequence":
        return tuple(reversed(states))
    if control_id == "stale_current":
        return (*states[:-1], states[-2])
    if control_id == "within_group_value_rotation":
        transformed = []
        for state in states:
            values = list(state)
            rotated = (
                values[1],
                values[0],
                values[3],
                values[2],
                values[5],
                values[4],
                *values[7:12],
                values[6],
            )
            transformed.append(tuple(rotated))
        return tuple(transformed)
    if control_id == "term_only":
        return _masked(states, set(range(0, 4)))
    if control_id == "tail_only":
        return _masked(states, set(range(4, 6)))
    if control_id == "flow_only":
        return _masked(states, set(range(6, 12)))
    if control_id == "current_only":
        return tuple(
            state if index == 4 else tuple("MASKED" for _ in state)
            for index, state in enumerate(states)
        )
    if control_id == "group_order_rotation":
        return states
    raise ValueError(f"unknown CEFS control: {control_id}")


def _group_order_prompt(
    states: Sequence[tuple[str, ...]],
    position: str,
) -> str:
    order = (*range(6, 12), *range(0, 4), *range(4, 6))
    lines: list[str] = []
    for state_label, values in zip(prereg.STATE_LABELS, states):
        for index in order:
            lines.append(
                f"{state_label}.{prereg.EDGE_NAMES[index]}={values[index]}"
            )
    lines.append(f"POSITION={position}")
    return "\n".join(lines) + "\n"


def build_controls(schedules: Sequence[ScheduleRow]) -> list[ControlRow]:
    controls: list[ControlRow] = []
    for schedule in schedules:
        if not schedule.model_eligible:
            continue
        for position_index, position in enumerate(prereg.POSITION_CONTEXTS):
            primary = schedule.prompts[position_index]
            for control_id in prereg.CONTROL_IDS:
                transformed = _control_states(schedule.states, control_id)
                if control_id == "group_order_rotation":
                    prompt = _group_order_prompt(transformed, position)
                else:
                    prompt = prereg.serialize_prompt(
                        _state_mapping(transformed),
                        position,
                        control=True,
                    )
                controls.append(
                    ControlRow(
                        observation_date=schedule.observation_date,
                        position_context=position,
                        control_id=control_id,
                        prompt=prompt,
                        prompt_sha256=hashlib.sha256(
                            prompt.encode("utf-8")
                        ).hexdigest(),
                        semantic_difference=transformed != schedule.states,
                    )
                )
                if prompt == primary and control_id != "group_order_rotation":
                    # Equality is measured and may fail a frozen aggregate gate;
                    # it is not silently repaired here.
                    pass
    return controls


def format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("CEFS timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def schedule_record(row: ScheduleRow) -> dict[str, str]:
    return {
        "observation_date": row.observation_date.isoformat(),
        "available_utc": format_utc(row.available_utc),
        "entry_utc": format_utc(row.entry_utc),
        "exit_utc": format_utc(row.exit_utc),
        "reservation_state": row.reservation_state,
        "role": row.role,
        "model_eligible": str(row.model_eligible).lower(),
        "current_signature": row.current_signature,
        "sequence_signature": row.sequence_signature,
        "prompt_target_flat": row.prompts[0],
        "prompt_target_long": row.prompts[1],
        "prompt_target_short": row.prompts[2],
    }


def control_record(row: ControlRow) -> dict[str, str]:
    return {
        "observation_date": row.observation_date.isoformat(),
        "position_context": row.position_context,
        "control_id": row.control_id,
        "control_prompt": row.prompt,
        "control_prompt_sha256": row.prompt_sha256,
        "semantic_difference": str(row.semantic_difference).lower(),
    }


def _canonical_records(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"\n".join(canonical_bytes(record) for record in records)


def parser_metrics(inputs: SourceInputs, common: Sequence[CommonRow]) -> dict[str, Any]:
    gaps = [
        (right.observation_date - left.observation_date).days
        for left, right in zip(common, common[1:])
    ]
    return {
        "panels": {
            "term": asdict(inputs.term.audit),
            "tail": asdict(inputs.tail.audit),
            "flow": asdict(inputs.flow.audit),
        },
        "common_dates": len(common),
        "common_first": common[0].observation_date.isoformat() if common else None,
        "common_last": common[-1].observation_date.isoformat() if common else None,
        "maximum_common_gap_days": max(gaps) if gaps else None,
        "vix_identity_rows": len(common),
        "post_2023_non_date_rows": sum(
            panel.audit.post_2023_non_date_rows
            for panel in (inputs.term, inputs.tail, inputs.flow)
        ),
    }


def parser_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "common_date_count_exact": metrics["common_dates"] == 1006,
        "common_first_exact": metrics["common_first"] == "2020-01-02",
        "common_last_exact": metrics["common_last"] == "2023-12-29",
        "maximum_common_gap_at_most_10": (
            metrics["maximum_common_gap_days"] is not None
            and metrics["maximum_common_gap_days"] <= 10
        ),
        "vix_identity_complete": metrics["vix_identity_rows"] == 1006,
        "post_2023_non_date_zero": metrics["post_2023_non_date_rows"] == 0,
    }


def _quarter(value: datetime) -> str:
    return f"{value.year}Q{((value.month - 1) // 3) + 1}"


def schedule_metrics(schedules: Sequence[ScheduleRow]) -> dict[str, Any]:
    eligible = [row for row in schedules if row.model_eligible]
    accepted = [row for row in schedules if row.reservation_state == "ACCEPTED"]
    by_year = Counter(str(row.entry_utc.year) for row in eligible)
    by_quarter = Counter(_quarter(row.entry_utc) for row in eligible)
    overlap = sum(
        right.entry_utc < left.exit_utc
        for left, right in zip(accepted, accepted[1:])
    )
    wrong_hold = sum(
        row.exit_utc - row.entry_utc
        != timedelta(
            minutes=(
                prereg.Policy().hold_bars * prereg.Policy().bar_minutes
            )
        )
        for row in accepted
    )
    return {
        "sequence_ready": len(schedules),
        "accepted": len(accepted),
        "suppressed_overlap": sum(
            row.reservation_state == "SUPPRESSED_OVERLAP" for row in schedules
        ),
        "role_crossing_audit_rows": sum(
            row.role == "ROLE_CROSSING" for row in schedules
        ),
        "complete_primary": len(eligible),
        "by_entry_year": dict(sorted(by_year.items())),
        "by_entry_quarter": dict(sorted(by_quarter.items())),
        "accepted_overlap_count": overlap,
        "wrong_hold_count": wrong_hold,
        "complete_primary_role_crossing": sum(
            row.model_eligible and row.role == "ROLE_CROSSING" for row in schedules
        ),
    }


def schedule_replay_metrics(
    inputs: SourceInputs,
    common: Sequence[CommonRow],
    schedules: Sequence[ScheduleRow],
) -> dict[str, Any]:
    full_records = [schedule_record(row) for row in schedules]
    prefixes: dict[str, bool] = {}
    for cutoff_day in PREFIX_CUTOFFS:
        cutoff = cutoff_day.isoformat()
        rebuilt = build_schedules(
            join_common_rows(_prefix_inputs(inputs, cutoff))
        )
        expected = [
            schedule_record(row)
            for row in schedules
            if row.observation_date < cutoff_day
        ]
        prefixes[cutoff] = (
            [schedule_record(row) for row in rebuilt] == expected
        )
    appended = build_schedules(_append_synthetic(common))
    appended_prior = [
        schedule_record(row)
        for row in appended
        if row.observation_date < SOURCE_END
    ]
    synthetic_identical = appended_prior == full_records
    return {
        "prefixes": prefixes,
        "synthetic_append_prior_identical": synthetic_identical,
        "passed": all(prefixes.values()) and synthetic_identical,
    }


def schedule_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    years = metrics["by_entry_year"]
    quarters = metrics["by_entry_quarter"]
    expected_quarters = [
        f"{year}Q{quarter}"
        for year in range(2020, 2024)
        for quarter in range(1, 5)
    ]
    return {
        "total_at_least_920": metrics["complete_primary"] >= 920,
        "each_year_at_least_230": all(
            years.get(str(year), 0) >= 230 for year in range(2020, 2024)
        ),
        "each_quarter_at_least_50": all(
            quarters.get(quarter, 0) >= 50 for quarter in expected_quarters
        ),
        "accepted_nonoverlap": metrics["accepted_overlap_count"] == 0,
        "exact_288_bar_hold": metrics["wrong_hold_count"] == 0,
        "complete_primary_role_crossing_zero": (
            metrics["complete_primary_role_crossing"] == 0
        ),
        "future_row_independent": metrics["schedule_replay"]["passed"],
    }


def _role_rows(
    schedules: Sequence[ScheduleRow],
) -> dict[str, list[ScheduleRow]]:
    return {
        role: [
            row for row in schedules if row.model_eligible and row.role == role
        ]
        for role in ROLE_WINDOWS
    }


def edge_metrics(schedules: Sequence[ScheduleRow]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role, rows in _role_rows(schedules).items():
        fields: dict[str, Any] = {}
        for index, edge in enumerate(prereg.EDGE_NAMES):
            counts = Counter(row.states[-1][index] for row in rows)
            total = len(rows)
            shares = {
                level: counts.get(level, 0) / total if total else 0.0
                for level in prereg.EDGE_LEVELS
            }
            fields[edge] = {
                "counts": {level: counts.get(level, 0) for level in prereg.EDGE_LEVELS},
                "shares": shares,
            }
        result[role] = {"rows": len(rows), "edges": fields}
    return result


def edge_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for role in ROLE_WINDOWS:
        edges = metrics[role]["edges"]
        for index, edge in enumerate(prereg.EDGE_NAMES):
            shares = edges[edge]["shares"]
            prefix = f"{role.lower()}_{edge.lower()}"
            if index < 2:
                checks[f"{prefix}_two_levels"] = sum(
                    share > 0 for share in shares.values()
                ) >= 2
                checks[f"{prefix}_max_share"] = max(shares.values()) <= 0.98
            else:
                checks[f"{prefix}_lower_share"] = shares["LOWER"] >= 0.10
                checks[f"{prefix}_higher_share"] = shares["HIGHER"] >= 0.10
                checks[f"{prefix}_max_share"] = max(shares.values()) <= 0.88
    return checks


def diversity_metrics(schedules: Sequence[ScheduleRow]) -> dict[str, Any]:
    eligible = [row for row in schedules if row.model_eligible]
    annual: dict[str, Any] = {}
    for year in range(2020, 2024):
        rows = [row for row in eligible if row.entry_utc.year == year]
        current = Counter(row.current_signature for row in rows)
        sequences = Counter(row.sequence_signature for row in rows)
        total = len(rows)
        annual[str(year)] = {
            "rows": total,
            "distinct_current": len(current),
            "largest_current_share": (
                max(current.values()) / total if total and current else 0.0
            ),
            "unique_sequence_share": (
                sum(count == 1 for count in sequences.values()) / total
                if total
                else 0.0
            ),
            "largest_sequence_share": (
                max(sequences.values()) / total if total and sequences else 0.0
            ),
        }
    role_edge = edge_metrics(schedules)
    drift: dict[str, Any] = {}
    for edge in prereg.EDGE_NAMES:
        edge_drift: dict[str, float] = {}
        for comparison_role in ("TEST", "EVAL"):
            maximum = max(
                abs(
                    role_edge["TRAIN"]["edges"][edge]["shares"][level]
                    - role_edge[comparison_role]["edges"][edge]["shares"][level]
                )
                for level in prereg.EDGE_LEVELS
            )
            edge_drift[comparison_role] = maximum
        drift[edge] = edge_drift
    return {"annual": annual, "role_level_share_drift": drift}


def diversity_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for year, values in metrics["annual"].items():
        checks[f"{year}_distinct_current"] = values["distinct_current"] >= 40
        checks[f"{year}_largest_current"] = (
            values["largest_current_share"] <= 0.15
        )
        checks[f"{year}_unique_sequence"] = (
            values["unique_sequence_share"] >= 0.80
        )
        checks[f"{year}_largest_sequence"] = (
            values["largest_sequence_share"] <= 0.02
        )
    for edge, roles in metrics["role_level_share_drift"].items():
        for role, value in roles.items():
            checks[f"{edge.lower()}_{role.lower()}_drift"] = value <= 0.25
    return checks


def control_metrics(
    schedules: Sequence[ScheduleRow],
    controls: Sequence[ControlRow],
) -> dict[str, Any]:
    primary = {
        (row.observation_date, position): row.prompts[index]
        for row in schedules
        if row.model_eligible
        for index, position in enumerate(prereg.POSITION_CONTEXTS)
    }
    metrics: dict[str, Any] = {
        "eligible_schedules": sum(row.model_eligible for row in schedules),
        "control_rows": len(controls),
        "by_control": {},
    }
    for control_id in prereg.CONTROL_IDS:
        subset = [row for row in controls if row.control_id == control_id]
        byte_different = sum(
            row.prompt
            != primary[(row.observation_date, row.position_context)]
            for row in subset
        )
        semantic = sum(row.semantic_difference for row in subset)
        total = len(subset)
        metrics["by_control"][control_id] = {
            "rows": total,
            "byte_difference_share": byte_different / total if total else 0.0,
            "semantic_difference_share": semantic / total if total else 0.0,
        }
    return metrics


def control_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    expected = metrics["eligible_schedules"] * len(prereg.POSITION_CONTEXTS)
    by_control = metrics["by_control"]
    checks = {
        "exact_total_control_rows": (
            metrics["control_rows"] == expected * len(prereg.CONTROL_IDS)
        ),
        "each_control_complete": all(
            by_control[control]["rows"] == expected
            for control in prereg.CONTROL_IDS
        ),
        "group_order_byte_difference_all": (
            by_control["group_order_rotation"]["byte_difference_share"] == 1.0
        ),
        "reverse_byte_difference_min": (
            by_control["reverse_sequence"]["byte_difference_share"] >= 0.95
        ),
        "stale_semantic_difference_min": (
            by_control["stale_current"]["semantic_difference_share"] >= 0.35
        ),
        "rotation_semantic_difference_min": (
            by_control["within_group_value_rotation"][
                "semantic_difference_share"
            ]
            >= 0.50
        ),
    }
    for control in ("term_only", "tail_only", "flow_only", "current_only"):
        checks[f"{control}_byte_difference_all"] = (
            by_control[control]["byte_difference_share"] == 1.0
        )
    return checks


def _append_synthetic(rows: Sequence[CommonRow]) -> list[CommonRow]:
    if not rows:
        raise ValueError("CEFS synthetic append requires source rows")
    synthetic = CommonRow(
        observation_date=date(2024, 1, 2),
        vix9d=Decimal("12"),
        vix=Decimal("13"),
        vix3m=Decimal("14"),
        skew=Decimal("120"),
        vvix=Decimal("90"),
        total_pcr=Decimal("1.0"),
        index_pcr=Decimal("1.1"),
        equity_pcr=Decimal("0.9"),
        vix_pcr=Decimal("1.2"),
        spx_pcr=Decimal("1.0"),
        index_volume=1000,
        vix_volume=100,
    )
    return [*rows, synthetic]


def append_replay_metrics(
    inputs: SourceInputs,
    full_common: Sequence[CommonRow],
    full_schedules: Sequence[ScheduleRow],
    full_controls: Sequence[ControlRow],
) -> dict[str, Any]:
    full_schedule_records = [
        schedule_record(row) for row in full_schedules
    ]
    full_control_records = [control_record(row) for row in full_controls]
    prefixes: dict[str, Any] = {}
    all_passed = True
    for cutoff_day in PREFIX_CUTOFFS:
        cutoff = cutoff_day.isoformat()
        prefix_common = join_common_rows(_prefix_inputs(inputs, cutoff))
        prefix_schedules = build_schedules(prefix_common)
        prefix_controls = build_controls(prefix_schedules)
        expected_schedules = [
            schedule_record(row)
            for row in full_schedules
            if row.observation_date < cutoff_day
        ]
        expected_controls = [
            control_record(row)
            for row in full_controls
            if row.observation_date < cutoff_day
        ]
        schedule_match = (
            [schedule_record(row) for row in prefix_schedules]
            == expected_schedules
        )
        control_match = (
            [control_record(row) for row in prefix_controls]
            == expected_controls
        )
        passed = schedule_match and control_match
        prefixes[cutoff] = {
            "common_rows": len(prefix_common),
            "schedule_rows": len(prefix_schedules),
            "control_rows": len(prefix_controls),
            "schedule_match": schedule_match,
            "control_match": control_match,
            "passed": passed,
        }
        all_passed = all_passed and passed
    second_schedules = build_schedules(full_common)
    second_controls = build_controls(second_schedules)
    deterministic = (
        _canonical_records([schedule_record(row) for row in second_schedules])
        == _canonical_records(full_schedule_records)
        and _canonical_records([control_record(row) for row in second_controls])
        == _canonical_records(full_control_records)
    )
    appended_schedules = build_schedules(_append_synthetic(full_common))
    appended_controls = build_controls(appended_schedules)
    appended_prior_schedules = [
        schedule_record(row)
        for row in appended_schedules
        if row.observation_date < SOURCE_END
    ]
    appended_prior_controls = [
        control_record(row)
        for row in appended_controls
        if row.observation_date < SOURCE_END
    ]
    synthetic_schedule_match = (
        appended_prior_schedules == full_schedule_records
    )
    synthetic_control_match = (
        appended_prior_controls == full_control_records
    )
    synthetic_append = synthetic_schedule_match and synthetic_control_match
    return {
        "prefixes": prefixes,
        "two_builds_byte_identical": deterministic,
        "synthetic_append_prior_identical": synthetic_append,
        "synthetic_append_prior_schedule_identical": (
            synthetic_schedule_match
        ),
        "synthetic_append_prior_control_identical": synthetic_control_match,
        "passed": all_passed and deterministic and synthetic_append,
    }


def forbidden_counters() -> dict[str, int]:
    return {name: 0 for name in FORBIDDEN_COUNTER_NAMES}


def deterministic_csv_gzip(
    records: Sequence[Mapping[str, str]],
    columns: Sequence[str],
) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(
        text,
        fieldnames=columns,
        lineterminator="\n",
        extrasaction="raise",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    for record in records:
        if tuple(record) != tuple(columns):
            raise ValueError("CEFS output column order changed")
        writer.writerow(record)
    raw = text.getvalue().encode("utf-8")
    compressed = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=compressed,
        mtime=0,
    ) as handle:
        handle.write(raw)
    return compressed.getvalue()


def _gate_record(
    index: int,
    name: str,
    checks: Mapping[str, bool],
) -> dict[str, Any]:
    return {
        "index": index,
        "name": name,
        "checks": dict(checks),
        "passed": bool(checks) and all(checks.values()),
    }


def _authority_report(
    seal: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **dict(authority),
        "execution_seal": {
            "path": EXECUTION_SEAL_PATH,
            "manifest_hash": seal["manifest_hash"],
            "runner": seal["runner"],
            "tests": seal["tests"],
        },
        "sources": prereg._source_contracts(),
    }


def _result_report(
    *,
    decision: str,
    failure_action: str | None,
    pass_action: str | None,
    authority: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
    details: Mapping[str, Any],
    counters: Mapping[str, int],
    source_hash: str | None,
    control_hash: str | None,
    source_output: Mapping[str, Any] | None,
    control_output: Mapping[str, Any] | None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    core = {
        "protocol_version": RESULT_PROTOCOL,
        "policy_id": prereg.POLICY_ID,
        "decision": decision,
        "failure_action": failure_action,
        "pass_action": pass_action,
        "authority": dict(authority),
        "gates": list(gates),
        "details": dict(details),
        "forbidden_counters": dict(counters),
        "source_row_hash": source_hash,
        "control_row_hash": control_hash,
        "source_output": source_output,
        "control_output": control_output,
        "error": (
            None
            if error is None
            else {"type": type(error).__name__, "message": str(error)}
        ),
    }
    return {**core, "result_hash": canonical_hash(core)}


def _git_blob_sha256(commit: str, path: str) -> str:
    completed = subprocess.run(
        ("git", "show", f"{commit}:{path}"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _validate_terminal_execution_seal() -> dict[str, Any]:
    path = repository_path(EXECUTION_SEAL_PATH)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("CEFS terminal execution seal is not regular")
    _assert_tracked_clean(EXECUTION_SEAL_PATH)
    payload = json.loads(path.read_text())
    expected_keys = {
        "protocol_version",
        "policy_id",
        "contract",
        "preregistration",
        "preregistration_producer",
        "runner",
        "tests",
        "source_values_opened",
        "outcomes_opened",
        "manifest_hash",
    }
    if set(payload) != expected_keys:
        raise RuntimeError("CEFS terminal execution seal fields mismatch")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload["manifest_hash"] != canonical_hash(core):
        raise RuntimeError("CEFS terminal execution seal hash mismatch")
    if (
        payload["protocol_version"] != SEAL_PROTOCOL
        or payload["policy_id"] != prereg.POLICY_ID
        or payload["contract"]
        != {
            "path": CONTRACT_PATH,
            "commit": CONTRACT_COMMIT,
            "sha256": CONTRACT_SHA256,
        }
        or payload["preregistration"]
        != {
            "path": PREREGISTRATION_PATH,
            "commit": PREREGISTRATION_COMMIT,
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        }
        or payload["preregistration_producer"]
        != {
            "path": prereg.PRODUCER_SCRIPT,
            "commit": PREREGISTRATION_PRODUCER_COMMIT,
            "sha256": PREREGISTRATION_PRODUCER_SHA256,
        }
        or payload["source_values_opened"] is not False
        or payload["outcomes_opened"] is not False
    ):
        raise RuntimeError("CEFS terminal execution seal authority mismatch")
    frozen_files = (
        (
            CONTRACT_PATH,
            CONTRACT_COMMIT,
            CONTRACT_SHA256,
        ),
        (
            prereg.BOUNDARY_DOCUMENT,
            prereg.BOUNDARY_COMMIT,
            prereg.BOUNDARY_DOCUMENT_SHA256,
        ),
        (
            PREREGISTRATION_PATH,
            PREREGISTRATION_COMMIT,
            PREREGISTRATION_SHA256,
        ),
        (
            prereg.PRODUCER_SCRIPT,
            PREREGISTRATION_PRODUCER_COMMIT,
            PREREGISTRATION_PRODUCER_SHA256,
        ),
    )
    for frozen_path, expected_commit, expected_sha in frozen_files:
        if (
            _assert_tracked_clean(frozen_path) != expected_commit
            or sha256_file(frozen_path) != expected_sha
        ):
            raise RuntimeError("CEFS terminal frozen authority file mismatch")
    runner = payload["runner"]
    tests = payload["tests"]
    for binding, expected_path in (
        (runner, RUNNER_PATH),
        (tests, TEST_PATH),
    ):
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "commit", "sha256"}
            or binding.get("path") != expected_path
            or not isinstance(binding.get("commit"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", binding["commit"], re.ASCII)
            or not isinstance(binding.get("sha256"), str)
            or not HEX64.fullmatch(binding["sha256"])
            or _git_blob_sha256(binding["commit"], expected_path)
            != binding["sha256"]
            or _assert_tracked_clean(expected_path) != binding["commit"]
            or sha256_file(expected_path) != binding["sha256"]
        ):
            raise RuntimeError("CEFS terminal runner/test seal mismatch")
    if runner["commit"] != tests["commit"]:
        raise RuntimeError("CEFS terminal runner/tests commit mismatch")
    return payload


def _expected_terminal_authority(
    seal: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "boundary": {
            "path": prereg.BOUNDARY_DOCUMENT,
            "commit": prereg.BOUNDARY_COMMIT,
            "sha256": prereg.BOUNDARY_DOCUMENT_SHA256,
        },
        "contract": {
            "path": CONTRACT_PATH,
            "commit": CONTRACT_COMMIT,
            "sha256": CONTRACT_SHA256,
        },
        "preregistration": {
            "path": PREREGISTRATION_PATH,
            "commit": PREREGISTRATION_COMMIT,
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "preregistration_producer": {
            "path": prereg.PRODUCER_SCRIPT,
            "commit": PREREGISTRATION_PRODUCER_COMMIT,
            "sha256": PREREGISTRATION_PRODUCER_SHA256,
        },
        "execution_seal": {
            "path": EXECUTION_SEAL_PATH,
            "manifest_hash": seal["manifest_hash"],
            "runner": seal["runner"],
            "tests": seal["tests"],
        },
        "sources": prereg._source_contracts(),
    }


def _validate_terminal_gates(
    gates: Any,
    *,
    decision: str,
) -> int:
    if (
        not isinstance(gates, list)
        or not gates
        or len(gates) > len(GATE_NAMES)
        or (decision == "pass" and len(gates) != len(GATE_NAMES))
    ):
        raise RuntimeError("CEFS terminal gate count mismatch")
    for expected_index, (gate, expected_name) in enumerate(
        zip(gates, GATE_NAMES),
        start=1,
    ):
        if (
            not isinstance(gate, dict)
            or set(gate) != {"index", "name", "checks", "passed"}
            or type(gate.get("index")) is not int
            or gate["index"] != expected_index
            or gate.get("name") != expected_name
            or not isinstance(gate.get("checks"), dict)
            or not gate["checks"]
            or any(
                type(value) is not bool
                for value in gate["checks"].values()
            )
            or type(gate.get("passed")) is not bool
            or gate["passed"] != all(gate["checks"].values())
        ):
            raise RuntimeError("CEFS terminal gate structure mismatch")
    last_index = len(gates)
    if decision == "pass":
        if not all(gate["passed"] for gate in gates):
            raise RuntimeError("CEFS terminal pass has a failed gate")
    elif (
        any(not gate["passed"] for gate in gates[:-1])
        or gates[-1]["passed"]
    ):
        raise RuntimeError("CEFS terminal rejection violates first-stop gates")
    gate_one = gates[0]
    if last_index == 1 and decision == "fail":
        if gate_one["checks"] != {"authority_valid": False}:
            raise RuntimeError("CEFS terminal Gate 1 rejection mismatch")
    else:
        expected_gate_one_keys = {
            "authority_valid",
            "worktree_clean",
            *FORBIDDEN_COUNTER_NAMES,
        }
        if (
            set(gate_one["checks"]) != expected_gate_one_keys
            or not all(gate_one["checks"].values())
        ):
            raise RuntimeError("CEFS terminal Gate 1 pass binding mismatch")
    return last_index


def _validate_terminal_details(
    details: Any,
    *,
    decision: str,
    last_gate: int,
) -> None:
    if not isinstance(details, dict):
        raise RuntimeError("CEFS terminal details are not a mapping")
    known = set(DETAIL_KEYS_BY_GATE.values())
    if not set(details) <= known:
        raise RuntimeError("CEFS terminal details contain unknown stages")
    required_gates = (
        range(2, len(GATE_NAMES) + 1)
        if decision == "pass"
        else range(2, last_gate)
    )
    required = {DETAIL_KEYS_BY_GATE[index] for index in required_gates}
    allowed = set(required)
    if decision == "fail" and last_gate in DETAIL_KEYS_BY_GATE:
        allowed.add(DETAIL_KEYS_BY_GATE[last_gate])
    if not required <= set(details) or not set(details) <= allowed:
        raise RuntimeError("CEFS terminal details violate first-stop stages")
    if any(not isinstance(value, dict) for value in details.values()):
        raise RuntimeError("CEFS terminal stage detail is not a mapping")
    _validate_terminal_detail_schema(details)


def _require_exact_keys(
    value: Any,
    keys: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise RuntimeError(f"CEFS terminal {label} fields mismatch")
    return value


def _require_nonnegative_int(value: Any, *, label: str) -> None:
    if type(value) is not int or value < 0:
        raise RuntimeError(f"CEFS terminal {label} integer mismatch")


def _require_bool(value: Any, *, label: str) -> None:
    if type(value) is not bool:
        raise RuntimeError(f"CEFS terminal {label} boolean mismatch")


def _require_fraction(value: Any, *, label: str) -> None:
    if type(value) is not float or not 0.0 <= value <= 1.0:
        raise RuntimeError(f"CEFS terminal {label} fraction mismatch")


def _require_date_text_or_none(value: Any, *, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise RuntimeError(f"CEFS terminal {label} date mismatch")
    try:
        _parse_date(value)
    except ValueError as error:
        raise RuntimeError(f"CEFS terminal {label} date mismatch") from error


def _validate_parser_detail(detail: Mapping[str, Any]) -> None:
    parser = _require_exact_keys(
        detail,
        {
            "panels",
            "common_dates",
            "common_first",
            "common_last",
            "maximum_common_gap_days",
            "vix_identity_rows",
            "post_2023_non_date_rows",
        },
        label="parser",
    )
    panels = _require_exact_keys(
        parser["panels"],
        {"term", "tail", "flow"},
        label="parser panels",
    )
    for panel_name, value in panels.items():
        panel = _require_exact_keys(
            value,
            {
                "physical_rows",
                "date_only_rows",
                "value_rows",
                "first_date",
                "last_date",
                "prefix_rows",
                "post_2023_non_date_rows",
            },
            label=f"parser {panel_name}",
        )
        for key in (
            "physical_rows",
            "date_only_rows",
            "value_rows",
            "post_2023_non_date_rows",
        ):
            _require_nonnegative_int(
                panel[key],
                label=f"parser {panel_name} {key}",
            )
        _require_date_text_or_none(
            panel["first_date"],
            label=f"parser {panel_name} first",
        )
        _require_date_text_or_none(
            panel["last_date"],
            label=f"parser {panel_name} last",
        )
        prefix = _require_exact_keys(
            panel["prefix_rows"],
            {cutoff.isoformat() for cutoff in PREFIX_CUTOFFS},
            label=f"parser {panel_name} prefixes",
        )
        for count in prefix.values():
            _require_nonnegative_int(
                count,
                label=f"parser {panel_name} prefix count",
            )
    for key in (
        "common_dates",
        "vix_identity_rows",
        "post_2023_non_date_rows",
    ):
        _require_nonnegative_int(parser[key], label=f"parser {key}")
    _require_date_text_or_none(parser["common_first"], label="common first")
    _require_date_text_or_none(parser["common_last"], label="common last")
    maximum_gap = parser["maximum_common_gap_days"]
    if maximum_gap is not None:
        _require_nonnegative_int(maximum_gap, label="maximum common gap")


def _validate_schedule_detail(detail: Mapping[str, Any]) -> None:
    schedule = _require_exact_keys(
        detail,
        {
            "sequence_ready",
            "accepted",
            "suppressed_overlap",
            "role_crossing_audit_rows",
            "complete_primary",
            "by_entry_year",
            "by_entry_quarter",
            "accepted_overlap_count",
            "wrong_hold_count",
            "complete_primary_role_crossing",
            "schedule_replay",
        },
        label="schedule",
    )
    for key in (
        "sequence_ready",
        "accepted",
        "suppressed_overlap",
        "role_crossing_audit_rows",
        "complete_primary",
        "accepted_overlap_count",
        "wrong_hold_count",
        "complete_primary_role_crossing",
    ):
        _require_nonnegative_int(schedule[key], label=f"schedule {key}")
    years = schedule["by_entry_year"]
    if (
        not isinstance(years, dict)
        or not set(years) <= {str(year) for year in range(2020, 2024)}
    ):
        raise RuntimeError("CEFS terminal schedule year fields mismatch")
    quarters = schedule["by_entry_quarter"]
    expected_quarters = {
        f"{year}Q{quarter}"
        for year in range(2020, 2024)
        for quarter in range(1, 5)
    }
    if not isinstance(quarters, dict) or not set(quarters) <= expected_quarters:
        raise RuntimeError("CEFS terminal schedule quarter fields mismatch")
    for count in (*years.values(), *quarters.values()):
        _require_nonnegative_int(count, label="schedule role count")
    replay = _require_exact_keys(
        schedule["schedule_replay"],
        {
            "prefixes",
            "synthetic_append_prior_identical",
            "passed",
        },
        label="schedule replay",
    )
    prefix = _require_exact_keys(
        replay["prefixes"],
        {cutoff.isoformat() for cutoff in PREFIX_CUTOFFS},
        label="schedule replay prefixes",
    )
    for value in (
        *prefix.values(),
        replay["synthetic_append_prior_identical"],
        replay["passed"],
    ):
        _require_bool(value, label="schedule replay")


def _validate_edge_detail(detail: Mapping[str, Any]) -> None:
    roles = _require_exact_keys(
        detail,
        set(ROLE_WINDOWS),
        label="edge roles",
    )
    for role, value in roles.items():
        role_detail = _require_exact_keys(
            value,
            {"rows", "edges"},
            label=f"edge {role}",
        )
        _require_nonnegative_int(
            role_detail["rows"],
            label=f"edge {role} rows",
        )
        edges = _require_exact_keys(
            role_detail["edges"],
            set(prereg.EDGE_NAMES),
            label=f"edge {role} names",
        )
        for edge, edge_value in edges.items():
            metrics = _require_exact_keys(
                edge_value,
                {"counts", "shares"},
                label=f"edge {role} {edge}",
            )
            counts = _require_exact_keys(
                metrics["counts"],
                set(prereg.EDGE_LEVELS),
                label=f"edge {role} {edge} counts",
            )
            shares = _require_exact_keys(
                metrics["shares"],
                set(prereg.EDGE_LEVELS),
                label=f"edge {role} {edge} shares",
            )
            for count in counts.values():
                _require_nonnegative_int(count, label="edge count")
            for share in shares.values():
                _require_fraction(share, label="edge share")


def _validate_diversity_detail(detail: Mapping[str, Any]) -> None:
    diversity = _require_exact_keys(
        detail,
        {"annual", "role_level_share_drift"},
        label="diversity",
    )
    annual = _require_exact_keys(
        diversity["annual"],
        {str(year) for year in range(2020, 2024)},
        label="diversity annual",
    )
    for year, value in annual.items():
        metrics = _require_exact_keys(
            value,
            {
                "rows",
                "distinct_current",
                "largest_current_share",
                "unique_sequence_share",
                "largest_sequence_share",
            },
            label=f"diversity {year}",
        )
        _require_nonnegative_int(metrics["rows"], label="diversity rows")
        _require_nonnegative_int(
            metrics["distinct_current"],
            label="diversity distinct",
        )
        for key in (
            "largest_current_share",
            "unique_sequence_share",
            "largest_sequence_share",
        ):
            _require_fraction(metrics[key], label=f"diversity {key}")
    drift = _require_exact_keys(
        diversity["role_level_share_drift"],
        set(prereg.EDGE_NAMES),
        label="diversity drift edges",
    )
    for edge, value in drift.items():
        roles = _require_exact_keys(
            value,
            {"TEST", "EVAL"},
            label=f"diversity drift {edge}",
        )
        for amount in roles.values():
            _require_fraction(amount, label="diversity drift")


def _validate_control_detail(detail: Mapping[str, Any]) -> None:
    controls = _require_exact_keys(
        detail,
        {"eligible_schedules", "control_rows", "by_control"},
        label="controls",
    )
    _require_nonnegative_int(
        controls["eligible_schedules"],
        label="eligible schedules",
    )
    _require_nonnegative_int(controls["control_rows"], label="control rows")
    by_control = _require_exact_keys(
        controls["by_control"],
        set(prereg.CONTROL_IDS),
        label="control identities",
    )
    for control_id, value in by_control.items():
        metrics = _require_exact_keys(
            value,
            {
                "rows",
                "byte_difference_share",
                "semantic_difference_share",
            },
            label=f"control {control_id}",
        )
        _require_nonnegative_int(metrics["rows"], label="control count")
        _require_fraction(
            metrics["byte_difference_share"],
            label="control byte difference",
        )
        _require_fraction(
            metrics["semantic_difference_share"],
            label="control semantic difference",
        )


def _validate_append_replay_detail(detail: Mapping[str, Any]) -> None:
    replay = _require_exact_keys(
        detail,
        {
            "prefixes",
            "two_builds_byte_identical",
            "synthetic_append_prior_identical",
            "synthetic_append_prior_schedule_identical",
            "synthetic_append_prior_control_identical",
            "passed",
        },
        label="append replay",
    )
    prefixes = _require_exact_keys(
        replay["prefixes"],
        {cutoff.isoformat() for cutoff in PREFIX_CUTOFFS},
        label="append replay prefixes",
    )
    for cutoff, value in prefixes.items():
        metrics = _require_exact_keys(
            value,
            {
                "common_rows",
                "schedule_rows",
                "control_rows",
                "schedule_match",
                "control_match",
                "passed",
            },
            label=f"append replay {cutoff}",
        )
        for key in ("common_rows", "schedule_rows", "control_rows"):
            _require_nonnegative_int(
                metrics[key],
                label=f"append replay {key}",
            )
        for key in ("schedule_match", "control_match", "passed"):
            _require_bool(metrics[key], label=f"append replay {key}")
    for key in (
        "two_builds_byte_identical",
        "synthetic_append_prior_identical",
        "synthetic_append_prior_schedule_identical",
        "synthetic_append_prior_control_identical",
        "passed",
    ):
        _require_bool(replay[key], label=f"append replay {key}")


def _validate_terminal_detail_schema(details: Mapping[str, Any]) -> None:
    validators = {
        "parser": _validate_parser_detail,
        "schedule": _validate_schedule_detail,
        "edge_support": _validate_edge_detail,
        "diversity_stability": _validate_diversity_detail,
        "controls": _validate_control_detail,
        "determinism_append_replay": _validate_append_replay_detail,
    }
    for key, value in details.items():
        validators[key](value)


def _terminal_expected_gate_checks(
    index: int,
    details: Mapping[str, Any],
) -> dict[str, bool]:
    try:
        if index == 2:
            if "parser" not in details:
                return {"source_parse_completed": False}
            return parser_checks(details["parser"])
        if index == 3:
            return schedule_checks(details["schedule"])
        if index == 4:
            return edge_checks(details["edge_support"])
        if index == 5:
            return diversity_checks(details["diversity_stability"])
        if index == 6:
            return control_checks(details["controls"])
        if index == 7:
            replay = details["determinism_append_replay"]
            return {
                "prefixes_match": all(
                    value["passed"]
                    for value in replay["prefixes"].values()
                ),
                "two_builds_byte_identical": replay[
                    "two_builds_byte_identical"
                ],
                "synthetic_append_prior_identical": replay[
                    "synthetic_append_prior_identical"
                ],
                "synthetic_append_prior_schedule_identical": replay[
                    "synthetic_append_prior_schedule_identical"
                ],
                "synthetic_append_prior_control_identical": replay[
                    "synthetic_append_prior_control_identical"
                ],
            }
    except Exception as error:
        raise RuntimeError(
            "CEFS terminal stage details cannot reproduce gate checks"
        ) from error
    raise RuntimeError("CEFS terminal gate index has no check reproducer")


def _terminal_output_records(
    path: str,
    columns: Sequence[str],
) -> list[dict[str, str]]:
    compressed = repository_path(path).read_bytes()
    try:
        decoded = gzip.decompress(compressed).decode("utf-8")
        reader = iter(csv.reader(io.StringIO(decoded, newline=""), strict=True))
        header = next(reader)
    except (EOFError, OSError, UnicodeDecodeError, csv.Error, StopIteration) as error:
        raise RuntimeError("CEFS terminal output decoding failed") from error
    if tuple(header) != tuple(columns):
        raise RuntimeError("CEFS terminal output header mismatch")
    records: list[dict[str, str]] = []
    try:
        for row in reader:
            if len(row) != len(columns):
                raise RuntimeError("CEFS terminal output row width mismatch")
            records.append(dict(zip(columns, row)))
    except csv.Error as error:
        raise RuntimeError("CEFS terminal output CSV grammar failed") from error
    if deterministic_csv_gzip(records, columns) != compressed:
        raise RuntimeError("CEFS terminal output is not deterministic gzip")
    return records


def _parse_terminal_utc(token: str, *, label: str) -> datetime:
    if not UTC_TEXT.fullmatch(token):
        raise RuntimeError(f"CEFS terminal {label} timestamp grammar failed")
    try:
        parsed = datetime.strptime(token, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise RuntimeError(
            f"CEFS terminal {label} timestamp parse failed"
        ) from error
    if format_utc(parsed) != token:
        raise RuntimeError(
            f"CEFS terminal {label} timestamp round-trip failed"
        )
    return parsed


def _parse_terminal_states(
    sequence_signature: str,
) -> tuple[tuple[str, ...], ...]:
    states = tuple(
        tuple(state.split("|")) for state in sequence_signature.split("/")
    )
    if (
        len(states) != len(prereg.STATE_LABELS)
        or any(len(state) != len(prereg.EDGE_NAMES) for state in states)
        or any(
            level not in prereg.EDGE_LEVELS
            for state in states
            for level in state
        )
    ):
        raise RuntimeError("CEFS terminal sequence signature grammar failed")
    return states


def _terminal_schedule_rows(
    records: Sequence[Mapping[str, str]],
) -> list[ScheduleRow]:
    schedules: list[ScheduleRow] = []
    previous_day: date | None = None
    prior_accepted_exit: datetime | None = None
    for record in records:
        day = _parse_date(record["observation_date"])
        if (
            day < SOURCE_START
            or day >= SOURCE_END
            or (previous_day is not None and day <= previous_day)
        ):
            raise RuntimeError("CEFS terminal source date order failed")
        previous_day = day
        clock = prereg.fixed_clock(day)
        available = _parse_terminal_utc(
            record["available_utc"],
            label="available",
        )
        entry = _parse_terminal_utc(record["entry_utc"], label="entry")
        exit_at = _parse_terminal_utc(record["exit_utc"], label="exit")
        if (
            available != clock["available_utc"]
            or entry != clock["entry_utc"]
            or exit_at != clock["exit_utc"]
        ):
            raise RuntimeError("CEFS terminal fixed clock mismatch")
        accepted = prior_accepted_exit is None or entry >= prior_accepted_exit
        expected_reservation = (
            "ACCEPTED" if accepted else "SUPPRESSED_OVERLAP"
        )
        if record["reservation_state"] != expected_reservation:
            raise RuntimeError("CEFS terminal reservation mismatch")
        if accepted:
            prior_accepted_exit = exit_at
        expected_role = _role(entry, exit_at) if accepted else "SUPPRESSED"
        if record["role"] != expected_role:
            raise RuntimeError("CEFS terminal role mismatch")
        expected_eligible = accepted and expected_role in ROLE_WINDOWS
        eligible_text = str(expected_eligible).lower()
        if record["model_eligible"] != eligible_text:
            raise RuntimeError("CEFS terminal model eligibility mismatch")
        states = _parse_terminal_states(record["sequence_signature"])
        current_signature = "|".join(states[-1])
        if record["current_signature"] != current_signature:
            raise RuntimeError("CEFS terminal current signature mismatch")
        mapping = _state_mapping(states)
        prompts = tuple(
            prereg.serialize_prompt(mapping, position)
            for position in prereg.POSITION_CONTEXTS
        )
        recorded_prompts = tuple(
            record[column]
            for column in (
                "prompt_target_flat",
                "prompt_target_long",
                "prompt_target_short",
            )
        )
        if recorded_prompts != prompts:
            raise RuntimeError("CEFS terminal primary prompt mismatch")
        schedule = ScheduleRow(
            observation_date=day,
            available_utc=available,
            entry_utc=entry,
            exit_utc=exit_at,
            reservation_state=expected_reservation,
            role=expected_role,
            model_eligible=expected_eligible,
            states=states,
            current_signature=current_signature,
            sequence_signature=record["sequence_signature"],
            prompts=prompts,
        )
        if schedule_record(schedule) != record:
            raise RuntimeError("CEFS terminal source row canonical mismatch")
        schedules.append(schedule)
    return schedules


def _validate_terminal_output_semantics(
    source_records: Sequence[Mapping[str, str]],
    control_records: Sequence[Mapping[str, str]],
    details: Mapping[str, Any],
) -> None:
    schedules = _terminal_schedule_rows(source_records)
    controls = build_controls(schedules)
    expected_controls = [control_record(row) for row in controls]
    if list(control_records) != expected_controls:
        raise RuntimeError("CEFS terminal control rows do not replay")
    reported_schedule = dict(details["schedule"])
    reported_schedule.pop("schedule_replay")
    if schedule_metrics(schedules) != reported_schedule:
        raise RuntimeError("CEFS terminal schedule metrics do not replay")
    if edge_metrics(schedules) != details["edge_support"]:
        raise RuntimeError("CEFS terminal edge metrics do not replay")
    if diversity_metrics(schedules) != details["diversity_stability"]:
        raise RuntimeError("CEFS terminal diversity metrics do not replay")
    if control_metrics(schedules, controls) != details["controls"]:
        raise RuntimeError("CEFS terminal control metrics do not replay")


def _validate_terminal_report(path: Path, *, decision: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("CEFS terminal report is not a regular file")
    payload = json.loads(path.read_text())
    expected_fields = {
        "protocol_version",
        "policy_id",
        "decision",
        "failure_action",
        "pass_action",
        "authority",
        "gates",
        "details",
        "forbidden_counters",
        "source_row_hash",
        "control_row_hash",
        "source_output",
        "control_output",
        "error",
        "result_hash",
    }
    if set(payload) != expected_fields:
        raise RuntimeError("CEFS terminal report fields mismatch")
    if payload.get("protocol_version") != RESULT_PROTOCOL:
        raise RuntimeError("CEFS terminal protocol mismatch")
    if payload.get("policy_id") != prereg.POLICY_ID:
        raise RuntimeError("CEFS terminal policy mismatch")
    if payload.get("decision") != decision:
        raise RuntimeError("CEFS terminal decision mismatch")
    core = {key: value for key, value in payload.items() if key != "result_hash"}
    if payload.get("result_hash") != canonical_hash(core):
        raise RuntimeError("CEFS terminal result hash mismatch")
    last_gate = _validate_terminal_gates(payload["gates"], decision=decision)
    _validate_terminal_details(
        payload["details"],
        decision=decision,
        last_gate=last_gate,
    )
    for gate in payload["gates"][1:]:
        expected_checks = _terminal_expected_gate_checks(
            gate["index"],
            payload["details"],
        )
        if gate["checks"] != expected_checks:
            raise RuntimeError("CEFS terminal gate checks are not reproducible")
    counters = payload.get("forbidden_counters")
    if (
        not isinstance(counters, dict)
        or set(counters) != set(FORBIDDEN_COUNTER_NAMES)
        or any(
            type(counters[name]) is not int or counters[name] != 0
            for name in FORBIDDEN_COUNTER_NAMES
        )
    ):
        raise RuntimeError("CEFS terminal forbidden counters mismatch")
    for key in ("source_row_hash", "control_row_hash"):
        value = payload.get(key)
        if value is not None and (
            not isinstance(value, str) or not HEX64.fullmatch(value)
        ):
            raise RuntimeError("CEFS terminal canonical row hash mismatch")
    if decision == "pass":
        if (
            payload.get("failure_action") is not None
            or payload.get("pass_action")
            != "authorize_economic_rllm_evaluator_freeze_only"
            or payload.get("error") is not None
            or payload.get("source_row_hash") is None
            or payload.get("control_row_hash") is None
        ):
            raise RuntimeError("CEFS terminal pass action mismatch")
        expected_outputs = {
            "source_output": (SOURCE_OUTPUT, SOURCE_OUTPUT_COLUMNS),
            "control_output": (CONTROL_OUTPUT, CONTROL_OUTPUT_COLUMNS),
        }
        decoded_outputs: dict[str, list[dict[str, str]]] = {}
        for key, (expected_path, columns) in expected_outputs.items():
            output = payload.get(key)
            if (
                not isinstance(output, dict)
                or set(output) != {"path", "sha256", "rows"}
                or output.get("path") != expected_path
                or not isinstance(output.get("sha256"), str)
                or not HEX64.fullmatch(output["sha256"])
                or type(output.get("rows")) is not int
                or output["rows"] <= 0
            ):
                raise RuntimeError("CEFS terminal pass output binding missing")
            output_path = repository_path(expected_path)
            if output_path.is_symlink() or not output_path.is_file():
                raise RuntimeError("CEFS terminal pass output is not regular")
            if sha256_file(expected_path) != output["sha256"]:
                raise RuntimeError("CEFS terminal pass output hash mismatch")
            decoded_outputs[key] = _terminal_output_records(
                expected_path,
                columns,
            )
            if len(decoded_outputs[key]) != output["rows"]:
                raise RuntimeError("CEFS terminal pass decoded row count mismatch")
        if (
            payload["source_output"]["rows"]
            != payload["details"]["schedule"]["sequence_ready"]
            or payload["control_output"]["rows"]
            != payload["details"]["controls"]["control_rows"]
        ):
            raise RuntimeError("CEFS terminal pass output row count mismatch")
        if (
            hashlib.sha256(
                _canonical_records(decoded_outputs["source_output"])
            ).hexdigest()
            != payload["source_row_hash"]
            or hashlib.sha256(
                _canonical_records(decoded_outputs["control_output"])
            ).hexdigest()
            != payload["control_row_hash"]
        ):
            raise RuntimeError("CEFS terminal canonical row hash mismatch")
        _validate_terminal_output_semantics(
            decoded_outputs["source_output"],
            decoded_outputs["control_output"],
            payload["details"],
        )
        seal = _validate_terminal_execution_seal()
        if payload.get("authority") != _expected_terminal_authority(seal):
            raise RuntimeError("CEFS terminal pass authority mismatch")
    else:
        if (
            payload.get("failure_action")
            != "retire_cefs_d1_unchanged_before_outcomes"
            or payload.get("pass_action") is not None
            or payload.get("source_output") is not None
            or payload.get("control_output") is not None
        ):
            raise RuntimeError("CEFS terminal rejection binding mismatch")
        error = payload.get("error")
        if error is not None and (
            not isinstance(error, dict)
            or set(error) != {"type", "message"}
            or not all(isinstance(value, str) for value in error.values())
        ):
            raise RuntimeError("CEFS terminal rejection error mismatch")
        if last_gate == 1:
            if payload.get("authority") != {} or error is None:
                raise RuntimeError("CEFS Gate 1 rejection authority mismatch")
        else:
            if (
                (last_gate >= 3 and payload.get("source_row_hash") is None)
                or (last_gate < 3 and payload.get("source_row_hash") is not None)
                or (last_gate >= 6 and payload.get("control_row_hash") is None)
                or (last_gate < 6 and payload.get("control_row_hash") is not None)
            ):
                raise RuntimeError("CEFS terminal rejection row hash stage mismatch")
            seal = _validate_terminal_execution_seal()
            if payload.get("authority") != _expected_terminal_authority(seal):
                raise RuntimeError("CEFS terminal rejection authority mismatch")
    return payload


def _path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def pre_run_terminal_state() -> dict[str, Any] | None:
    pass_path = repository_path(PASS_REPORT)
    rejection_path = repository_path(REJECTION_REPORT)
    source_path = repository_path(SOURCE_OUTPUT)
    control_path = repository_path(CONTROL_OUTPUT)
    pass_present = _path_present(pass_path)
    rejection_present = _path_present(rejection_path)
    source_present = _path_present(source_path)
    control_present = _path_present(control_path)
    if pass_present and rejection_present:
        raise RuntimeError("CEFS conflicting terminal reports")
    if pass_present:
        return _validate_terminal_report(pass_path, decision="pass")
    if rejection_present:
        if source_present or control_present:
            raise RuntimeError("CEFS rejection conflicts with pass outputs")
        return _validate_terminal_report(rejection_path, decision="fail")
    if source_present or control_present:
        raise RuntimeError("CEFS partial terminal pass state")
    return None


def run_official() -> dict[str, Any]:
    terminal = pre_run_terminal_state()
    if terminal is not None:
        return terminal
    gates: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    authority_report: dict[str, Any] = {}
    counters = forbidden_counters()
    schedules: list[ScheduleRow] = []
    controls: list[ControlRow] = []

    def fail(error: BaseException | None = None) -> dict[str, Any]:
        source_records = [schedule_record(row) for row in schedules]
        control_records = [control_record(row) for row in controls]
        report = _result_report(
            decision="fail",
            failure_action="retire_cefs_d1_unchanged_before_outcomes",
            pass_action=None,
            authority=authority_report,
            gates=gates,
            details=details,
            counters=counters,
            source_hash=(
                hashlib.sha256(_canonical_records(source_records)).hexdigest()
                if source_records
                else None
            ),
            control_hash=(
                hashlib.sha256(_canonical_records(control_records)).hexdigest()
                if control_records
                else None
            ),
            source_output=None,
            control_output=None,
            error=error,
        )
        write_once_json(REJECTION_REPORT, report)
        return report

    try:
        seal = validate_execution_seal()
        authority = validate_frozen_authority()
        if not _worktree_clean():
            raise RuntimeError("CEFS worktree is not clean")
        authority_report = _authority_report(seal, authority)
        gate = _gate_record(
            1,
            "authority_forbidden_access",
            {
                "authority_valid": True,
                "worktree_clean": True,
                **{name: value == 0 for name, value in counters.items()},
            },
        )
        gates.append(gate)
    except Exception as error:
        gates.append(
            _gate_record(
                1,
                "authority_forbidden_access",
                {"authority_valid": False},
            )
        )
        return fail(error)

    try:
        inputs = load_source_inputs()
        common = join_common_rows(inputs)
        parser = parser_metrics(inputs, common)
        details["parser"] = parser
        gate = _gate_record(2, "schema_chronology", parser_checks(parser))
        gates.append(gate)
        if not gate["passed"]:
            return fail()
    except Exception as error:
        gates.append(
            _gate_record(
                2,
                "schema_chronology",
                {"source_parse_completed": False},
            )
        )
        return fail(error)

    schedules = build_schedules(common)
    schedule = schedule_metrics(schedules)
    schedule["schedule_replay"] = schedule_replay_metrics(
        inputs,
        common,
        schedules,
    )
    details["schedule"] = schedule
    gate = _gate_record(3, "schedule_support", schedule_checks(schedule))
    gates.append(gate)
    if not gate["passed"]:
        return fail()

    edges = edge_metrics(schedules)
    details["edge_support"] = edges
    gate = _gate_record(4, "primitive_edge_support", edge_checks(edges))
    gates.append(gate)
    if not gate["passed"]:
        return fail()

    diversity = diversity_metrics(schedules)
    details["diversity_stability"] = diversity
    gate = _gate_record(
        5,
        "state_diversity_stability",
        diversity_checks(diversity),
    )
    gates.append(gate)
    if not gate["passed"]:
        return fail()

    controls = build_controls(schedules)
    control_detail = control_metrics(schedules, controls)
    details["controls"] = control_detail
    gate = _gate_record(
        6,
        "source_only_controls",
        control_checks(control_detail),
    )
    gates.append(gate)
    if not gate["passed"]:
        return fail()

    replay = append_replay_metrics(inputs, common, schedules, controls)
    details["determinism_append_replay"] = replay
    gate = _gate_record(
        7,
        "determinism_append_replay",
        {
            "prefixes_match": all(
                value["passed"] for value in replay["prefixes"].values()
            ),
            "two_builds_byte_identical": replay[
                "two_builds_byte_identical"
            ],
            "synthetic_append_prior_identical": replay[
                "synthetic_append_prior_identical"
            ],
            "synthetic_append_prior_schedule_identical": replay[
                "synthetic_append_prior_schedule_identical"
            ],
            "synthetic_append_prior_control_identical": replay[
                "synthetic_append_prior_control_identical"
            ],
        },
    )
    gates.append(gate)
    if not gate["passed"]:
        return fail()

    source_records = [schedule_record(row) for row in schedules]
    control_records = [control_record(row) for row in controls]
    source_row_bytes = _canonical_records(source_records)
    control_row_bytes = _canonical_records(control_records)
    source_gzip = deterministic_csv_gzip(source_records, SOURCE_OUTPUT_COLUMNS)
    control_gzip = deterministic_csv_gzip(control_records, CONTROL_OUTPUT_COLUMNS)
    source_sha = hashlib.sha256(source_gzip).hexdigest()
    control_sha = hashlib.sha256(control_gzip).hexdigest()
    report = _result_report(
        decision="pass",
        failure_action=None,
        pass_action="authorize_economic_rllm_evaluator_freeze_only",
        authority=authority_report,
        gates=gates,
        details=details,
        counters=counters,
        source_hash=hashlib.sha256(source_row_bytes).hexdigest(),
        control_hash=hashlib.sha256(control_row_bytes).hexdigest(),
        source_output={
            "path": SOURCE_OUTPUT,
            "sha256": source_sha,
            "rows": len(source_records),
        },
        control_output={
            "path": CONTROL_OUTPUT,
            "sha256": control_sha,
            "rows": len(control_records),
        },
    )
    publish_write_once_transaction(
        (
            (SOURCE_OUTPUT, source_gzip),
            (CONTROL_OUTPUT, control_gzip),
            (PASS_REPORT, _json_bytes(report)),
        )
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("create-seal")
    subparsers.add_parser("validate-seal")
    subparsers.add_parser("run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "create-seal":
        payload = create_execution_seal()
        print(
            json.dumps(
                {
                    "path": EXECUTION_SEAL_PATH,
                    "manifest_hash": payload["manifest_hash"],
                    "source_values_opened": False,
                    "outcomes_opened": False,
                },
                indent=2,
            )
        )
        return
    if args.command == "validate-seal":
        payload = validate_execution_seal()
        print(
            json.dumps(
                {
                    "path": EXECUTION_SEAL_PATH,
                    "manifest_hash": payload["manifest_hash"],
                    "valid": True,
                },
                indent=2,
            )
        )
        return
    result = run_official()
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "failure_action": result["failure_action"],
                "pass_action": result["pass_action"],
                "gates": result["gates"],
                "result_hash": result["result_hash"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result["decision"] == "pass" else 2)


if __name__ == "__main__":
    main()
