"""Build outcome-blind URCD-72 source-support clocks and novelty evidence."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import io
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from training import preregister_usdc_recipient_concentration_dislocation as prereg


PROTOCOL_VERSION = "usdc_recipient_concentration_dislocation_support_v1"
CANDIDATE = prereg.POLICY_ID
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/build_usdc_recipient_concentration_dislocation_support.py")
TEST_PATH = Path("tests/test_build_usdc_recipient_concentration_dislocation_support.py")
IMPLEMENTATION_CONTRACT = Path(
    "docs/urcd-source-support-implementation-contract-2026-07-23.md"
)
IMPLEMENTATION_CONTRACT_SHA256 = (
    "2b6f6ff4f71ccb05cc0963cb7357eaef462fe1f26b71cf4183c86f27fb0e5317"
)
PREREGISTRATION_COMMIT = "25fecf9e33737e057b562c4816a3ecef84860b2e"
PREREGISTRATION_ARTIFACT = prereg.DEFAULT_OUTPUT
PREREGISTRATION_ARTIFACT_SHA256 = (
    "078a40ee5e8604ec1dae1087541237617a287c6e2e953609bb71a392808e705e"
)
PREREGISTRATION_MANIFEST_HASH = (
    "5be1af6621e59086ae5bf9a487e425e97c3c3403eebd227ae055b789c85ccb2e"
)
PREREGISTRATION_POLICY_HASH = (
    "c9e7a04d5d41ae572f77bcb9f02b30ac2bc9b500c5dedcc7697bb9e4b5814bbd"
)

SOURCE_CSV = prereg.SOURCE_CSV
SOURCE_CSV_SHA256 = prereg.SOURCE_CSV_SHA256
SOURCE_MANIFEST = prereg.SOURCE_MANIFEST
SOURCE_MANIFEST_SHA256 = prereg.SOURCE_MANIFEST_SHA256
SOURCE_START = datetime(2020, 1, 1, tzinfo=timezone.utc)
TRAIN_START = datetime(2021, 1, 1, tzinfo=timezone.utc)
SELECTION_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
SEALED_FROM = datetime(2024, 1, 1, tzinfo=timezone.utc)
SIX_HOURS = timedelta(hours=6)
ONE_DAY = timedelta(days=1)
CURRENT_WINDOW = timedelta(hours=24)
HOLD = timedelta(hours=72)
ENTRY_DELAY = timedelta(minutes=10)

DEFAULT_CLOCK_OUTPUT = Path(
    "data/usdc_recipient_concentration_dislocation_2021_2023/"
    "urcd72_support_clocks_2021_2023.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/usdc_recipient_concentration_dislocation_support_2026-07-23.json"
)

WINDOWS: Mapping[str, tuple[datetime, datetime]] = {
    "train": (TRAIN_START, SELECTION_START),
    "selection": (SELECTION_START, SEALED_FROM),
}
CONTROL_ORDER = prereg.SOURCE_CONTROLS
PERMUTATION_CONTROLS = (
    "recipient_year_permutation",
    "amount_year_permutation",
)

CLOCK_COLUMNS = (
    "candidate",
    "control",
    "signal_id",
    "split",
    "side",
    "decision_time",
    "source_window_start_exclusive",
    "source_window_end_inclusive",
    "entry_time",
    "exit_time",
    "statistic",
    "current_stat_num",
    "current_stat_den",
    "prior_q20_num",
    "prior_q20_den",
    "prior_q80_num",
    "prior_q80_den",
    "prior_q50_amount_raw",
    "valid_prior_windows",
    "materiality_applied",
    "current_event_count",
    "current_recipient_count",
    "current_total_amount_raw",
    "previous_state",
)

UTC = timezone.utc
ADDRESS = re.compile(r"^0x[0-9a-f]{40}$")
HASH32 = re.compile(r"^0x[0-9a-f]{64}$")
ZERO_ADDRESS = "0x" + "0" * 40
FORBIDDEN_COMPARATOR_TOKENS = prereg.FORBIDDEN_COMPARATOR_HEADER_TOKENS
KNOWN_COMPARATOR_CONTROLS: Mapping[str, frozenset[str]] = {
    "AMTR-48": frozenset(
        {
            "primary",
            "cross_minter",
            "no_amount_ratio",
            "no_minimum_gap",
            "stale_6h",
        }
    ),
    "UGCI-288": frozenset(
        {"primary", "no_gross_tail", "no_imbalance_floor", "stale_6h"}
    ),
    "WCDR-2016": frozenset(
        {
            "primary",
            "direction_flip",
            "wbtc_only_contrarian",
            "usdc_only_direct",
            "same_sign_direct",
            "stale_7d",
            "count_sign_consensus",
            "year_amount_permutation",
            "deterministic_random_side",
        }
    ),
    "WTSL-168-SOURCE-SEEN": frozenset(
        {
            "primary",
            "direction_flip",
            "stablecoin_only_direct",
            "wbtc_signed_placebo",
            "stale_24h",
            "stale_48h",
            "actor_cap_60",
            "no_black_funds_veto",
            "usdc_only_direct",
            "usdt_only_direct",
            "year_amount_permutation",
            "deterministic_random_side",
        }
    ),
    "WSCF-72-SOURCE-FAMILY-SEEN": frozenset(
        {
            "primary",
            "direction_flip",
            "deterministic_random_side",
            "wbtc_only_direct",
            "stablecoin_only_12h_grid",
            "anchored_first_nonzero",
            "opposite_confirmation",
            "lead_lag_reverse",
            "stale_wbtc_24h",
            "stale_wbtc_72h",
            "stablecoin_year_amount_permutation",
            "black_funds_veto",
            "usdc_only_confirmation",
            "usdt_only_confirmation",
        }
    ),
    "FCCM-72": frozenset(
        {
            "primary",
            "bitfinex_consensus_only",
            "utilization_only",
            "draw_only",
            "tenor_only",
            "majority_without_score",
            "wbtc_stale_7d",
            "bitfinex_stale_24h",
            "exact_direction_flip",
            "deterministic_random_side",
            "one_bar_delay",
        }
    ),
    "SQFD-6": frozenset(
        {
            "primary",
            "no_alt_breadth",
            "no_usdt_lag",
            "no_participation",
            "usdt_only",
            "direction_flip",
            "deterministic_random_side",
            "extra_latency_1h",
        }
    ),
    "SDDR-12": frozenset(
        {"primary", "no_disagreement", "usdc_only", "fdusd_only", "stale_1h"}
    ),
    "UCBR-12": frozenset(
        {
            "primary",
            "all_four",
            "leave_out_usdc",
            "leave_out_tusd",
            "leave_out_usdp",
            "leave_out_fdusd",
            "median_only",
            "stale_1h",
        }
    ),
}


@dataclass(frozen=True)
class Event:
    amount_raw: int
    recipient: str
    available_at: datetime
    block_timestamp: datetime
    block_number: int
    transaction_index: int
    log_index: int
    block_hash: str
    transaction_hash: str

    @property
    def identity(self) -> tuple[str, str, int]:
        return self.block_hash, self.transaction_hash, self.log_index

    @property
    def identity_text(self) -> str:
        return f"{self.block_hash}:{self.transaction_hash}:{self.log_index}"

    @property
    def order_key(self) -> tuple[Any, ...]:
        return (
            self.available_at,
            self.block_number,
            self.transaction_index,
            self.log_index,
            self.block_hash,
            self.transaction_hash,
        )


@dataclass(frozen=True)
class WindowMetric:
    endpoint: datetime
    valid: bool
    event_count: int
    recipient_count: int
    total_amount_raw: int
    amount_hhi: Fraction
    event_count_hhi: Fraction
    row_identities: tuple[tuple[str, str, int], ...]


@dataclass(frozen=True)
class Snapshot:
    metric: WindowMetric
    statistic: str
    current_stat: Fraction
    q20: Fraction
    q80: Fraction
    q50_amount_raw: int
    valid_prior_windows: int
    materiality_applied: bool
    state: int


@dataclass(frozen=True)
class RawCandidate:
    control: str
    decision_time: datetime
    side: int
    previous_state: int
    snapshot: Snapshot
    signal_id: str

    @property
    def entry_time(self) -> datetime:
        return self.decision_time + ENTRY_DELAY

    @property
    def exit_time(self) -> datetime:
        return self.entry_time + HOLD


@dataclass(frozen=True)
class Candidate:
    control: str
    split: str
    decision_time: datetime
    entry_time: datetime
    exit_time: datetime
    side: int
    previous_state: int
    snapshot: Snapshot
    signal_id: str


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError("URCD support path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("URCD support path escaped repository") from exc
    return resolved


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def parse_time(value: str) -> datetime:
    if not value.endswith("Z"):
        raise RuntimeError("URCD timestamp must use canonical UTC Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RuntimeError("URCD timestamp is malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise RuntimeError("URCD timestamp is not UTC")
    if parsed.microsecond:
        raise RuntimeError("URCD timestamp must have whole-second precision")
    return parsed.astimezone(UTC)


def format_time(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise RuntimeError("URCD datetime must be UTC-aware")
    if value.microsecond:
        raise RuntimeError("URCD datetime must have whole-second precision")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _nonnegative_int(value: str, field: str) -> int:
    if not value or not value.isascii() or not value.isdigit():
        raise RuntimeError(f"URCD {field} is not a canonical integer")
    if len(value) > 1 and value.startswith("0"):
        raise RuntimeError(f"URCD {field} has a leading zero")
    parsed = int(value)
    if parsed < 0:
        raise RuntimeError(f"URCD {field} is negative")
    return parsed


def _positive_int(value: str, field: str) -> int:
    parsed = _nonnegative_int(value, field)
    if parsed <= 0:
        raise RuntimeError(f"URCD {field} must be positive")
    return parsed


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _side_text(side: int) -> str:
    if side == 1:
        return "LONG"
    if side == -1:
        return "SHORT"
    raise RuntimeError("URCD candidate side must be nonzero")


def _git_check(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_protocol_committed() -> None:
    paths = (str(SCRIPT_PATH), str(TEST_PATH), str(IMPLEMENTATION_CONTRACT))
    tracked = _git_check("ls-files", "--error-unmatch", "--", *paths)
    if tracked.returncode:
        raise RuntimeError("URCD support protocol is not committed")
    clean = _git_check("diff", "--quiet", "HEAD", "--", *paths)
    if clean.returncode:
        raise RuntimeError("URCD support protocol differs from HEAD")


def validate_preregistration() -> Mapping[str, Any]:
    if sha256_file(PREREGISTRATION_ARTIFACT) != PREREGISTRATION_ARTIFACT_SHA256:
        raise RuntimeError("URCD preregistration artifact hash drift")
    payload = json.loads(_path(PREREGISTRATION_ARTIFACT).read_text("utf-8"))
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("URCD preregistration canonical hash drift")
    if (
        payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH
        or payload.get("policy_hash") != PREREGISTRATION_POLICY_HASH
        or payload.get("candidate") != CANDIDATE
        or payload.get("artifact_eligible") is not True
        or payload.get("outcomes_opened") is not False
        or payload.get("source_values_or_incidence_opened") is not False
        or payload.get("comparator_rows_opened_during_preregistration") is not False
    ):
        raise RuntimeError("URCD preregistration contract drift")
    if payload.get("outcome_boundary") != dict(prereg.EXPECTED_BOUNDARY):
        raise RuntimeError("URCD preregistration outcome boundary drift")
    if (
        canonical_hash(payload.get("policy")) != PREREGISTRATION_POLICY_HASH
        or canonical_hash(prereg.policy_payload()) != PREREGISTRATION_POLICY_HASH
    ):
        raise RuntimeError("URCD preregistration policy drift")
    return payload


def verify_pre_source_bindings() -> Mapping[str, Any]:
    bindings = (
        (
            prereg.BOUNDARY_DOCUMENT,
            prereg.BOUNDARY_DOCUMENT_SHA256,
            "boundary document",
        ),
        (
            prereg.MECHANISM_DOCUMENT,
            prereg.MECHANISM_DOCUMENT_SHA256,
            "mechanism document",
        ),
        (SOURCE_CSV, SOURCE_CSV_SHA256, "source CSV"),
        (SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256, "source manifest"),
        (
            IMPLEMENTATION_CONTRACT,
            IMPLEMENTATION_CONTRACT_SHA256,
            "implementation contract",
        ),
    )
    for path, expected, label in bindings:
        if sha256_file(path) != expected:
            raise RuntimeError(f"URCD {label} hash drift")
    with gzip.open(_path(SOURCE_CSV), "rt", encoding="utf-8", newline="") as handle:
        header = tuple(next(csv.reader(handle)))
    if header != prereg.SOURCE_HEADER:
        raise RuntimeError("URCD source header drift before value access")
    manifest = json.loads(_path(SOURCE_MANIFEST).read_text("utf-8"))
    output = manifest.get("output", {})
    boundary = manifest.get("outcome_boundary", {})
    replay = manifest.get("dual_replay", {})
    headers = manifest.get("header_materialization", {})
    source_audit = manifest.get("source_audit", {})
    source_contract = manifest.get("source_contract", {})
    finalized = source_audit.get("finalized_coverage", {})
    if (
        manifest.get("manifest_hash") != prereg.SOURCE_MANIFEST_HASH
        or output.get("path") != str(SOURCE_CSV)
        or output.get("sha256") != SOURCE_CSV_SHA256
        or output.get("rows") != 266_362
        or tuple(output.get("columns", ())) != prereg.SOURCE_HEADER
        or replay.get("canonical_replay_equal") is not True
        or replay.get("independent_transport_count") != 2
        or headers.get("event_block_hash_cross_checked") is not True
        or headers.get("transport_independent_from_primary_logs") is not True
        or finalized.get("observed_finalized_block_at_least_required") is not True
        or source_contract.get("chain_id") != 1
        or source_contract.get("confirmation_blocks") != 64
        or boundary.get("source_only") is not True
        or boundary.get("pnl_cagr_mdd_opened") is not False
    ):
        raise RuntimeError("URCD source manifest integrity contract drift")
    for field in (
        "btc_market_rows_read",
        "funding_rows_read",
        "future_return_rows_read",
        "post_2023_contract_event_rows_read",
    ):
        if boundary.get(field) != 0:
            raise RuntimeError(f"URCD source manifest opened {field}")
    return {
        "source_header": list(header),
        "manifest_hash": manifest["manifest_hash"],
        "dual_replay_equal": True,
        "independent_transport_count": 2,
        "event_block_hash_cross_checked": True,
        "confirmation_blocks": 64,
    }


def load_source_events(
    path: str | Path = SOURCE_CSV,
) -> tuple[list[Event], dict[str, Any]]:
    if sha256_file(path) != SOURCE_CSV_SHA256:
        raise RuntimeError("URCD source CSV hash drift")
    if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("URCD source manifest hash drift")
    events: list[Event] = []
    identities: set[tuple[str, str, int]] = set()
    physical_rows = 0
    preseal_rows = 0
    sealed_timestamp_rows = 0
    previous: tuple[Any, ...] | None = None
    with gzip.open(_path(path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise RuntimeError("URCD source is empty") from exc
        if header != prereg.SOURCE_HEADER:
            raise RuntimeError("URCD source header drift")
        index = {name: header.index(name) for name in header}
        for row in reader:
            physical_rows += 1
            if len(row) != len(header):
                raise RuntimeError("URCD source row width drift")
            available_at = parse_time(row[index["available_at"]])
            if available_at >= SEALED_FROM:
                sealed_timestamp_rows += 1
                continue
            preseal_rows += 1
            if row[index["asset"]] != "usdc_eth" or row[index["event"]] != "mint":
                continue
            if row[index["event_sign"]] != "1" or row[index["decimals"]] != "6":
                raise RuntimeError("URCD eligible event semantics drift")
            recipient = row[index["indexed_address_2"]].lower()
            if not ADDRESS.fullmatch(recipient) or recipient == ZERO_ADDRESS:
                raise RuntimeError("URCD mint recipient is malformed")
            block_hash = row[index["block_hash"]].lower()
            transaction_hash = row[index["transaction_hash"]].lower()
            if not HASH32.fullmatch(block_hash) or not HASH32.fullmatch(transaction_hash):
                raise RuntimeError("URCD source identity hash is malformed")
            log_index = _nonnegative_int(row[index["log_index"]], "log_index")
            identity = (block_hash, transaction_hash, log_index)
            if identity in identities:
                raise RuntimeError("URCD eligible source identity duplicated")
            identities.add(identity)
            event = Event(
                amount_raw=_positive_int(row[index["amount_raw"]], "amount_raw"),
                recipient=recipient,
                available_at=available_at,
                block_timestamp=parse_time(row[index["block_timestamp"]]),
                block_number=_nonnegative_int(
                    row[index["block_number"]], "block_number"
                ),
                transaction_index=_nonnegative_int(
                    row[index["transaction_index"]], "transaction_index"
                ),
                log_index=log_index,
                block_hash=block_hash,
                transaction_hash=transaction_hash,
            )
            if event.block_timestamp > event.available_at:
                raise RuntimeError("URCD event is available before occurrence")
            if previous is not None and event.order_key < previous:
                raise RuntimeError("URCD eligible source is not causally sorted")
            previous = event.order_key
            events.append(event)
    if physical_rows != 266_362:
        raise RuntimeError("URCD source physical row count drift")
    if not events:
        raise RuntimeError("URCD source contains no eligible mint events")
    return events, {
        "physical_rows_scanned": physical_rows,
        "preseal_rows_scanned": preseal_rows,
        "sealed_timestamp_rows_scanned": sealed_timestamp_rows,
        "eligible_mint_value_rows_decoded": len(events),
        "unique_eligible_identities": len(identities),
        "first_available_at": format_time(events[0].available_at),
        "last_available_at": format_time(events[-1].available_at),
        "post_2023_source_value_rows_decoded": 0,
    }


def _hash_key(control: str, role: str, year: int, event: Event) -> tuple[bytes, str]:
    preimage = (
        f"URCD-72|{control}|{role}|{year}|{event.identity_text}".encode("ascii")
    )
    return hashlib.sha256(preimage).digest(), event.identity_text


def permute_field(events: Sequence[Event], control: str) -> list[Event]:
    if control not in PERMUTATION_CONTROLS:
        raise RuntimeError("URCD unknown permutation control")
    groups: dict[int, list[Event]] = defaultdict(list)
    for event in events:
        groups[event.available_at.year].append(event)
    output: list[Event] = []
    for year in sorted(groups):
        group = groups[year]
        if len({event.identity for event in group}) != len(group):
            raise RuntimeError("URCD permutation population identity duplicated")
        donors = sorted(group, key=lambda event: _hash_key(control, "source", year, event))
        destinations = sorted(
            group, key=lambda event: _hash_key(control, "destination", year, event)
        )
        if control == "recipient_year_permutation":
            output.extend(
                replace(destination, recipient=donor.recipient)
                for donor, destination in zip(donors, destinations)
            )
        else:
            output.extend(
                replace(destination, amount_raw=donor.amount_raw)
                for donor, destination in zip(donors, destinations)
            )
    return sorted(output, key=lambda event: event.order_key)


def iter_anchors(start: datetime, end_exclusive: datetime) -> Iterable[datetime]:
    if start.minute or start.second or start.microsecond or start.hour % 6:
        raise RuntimeError("URCD anchor start is not on six-hour UTC grid")
    current = start
    while current < end_exclusive:
        yield current
        current += SIX_HOURS


def build_window_metrics(
    events: Sequence[Event],
    *,
    anchor_start: datetime = SOURCE_START,
    end_exclusive: datetime = SEALED_FROM,
    coverage_start: datetime = SOURCE_START,
) -> dict[datetime, WindowMetric]:
    ordered = sorted(events, key=lambda event: event.order_key)
    if list(events) != ordered:
        raise RuntimeError("URCD metric input must be causally sorted")
    times = [event.available_at for event in ordered]
    metrics: dict[datetime, WindowMetric] = {}
    for endpoint in iter_anchors(anchor_start, end_exclusive):
        lower = endpoint - CURRENT_WINDOW
        if lower < coverage_start:
            metrics[endpoint] = WindowMetric(
                endpoint, False, 0, 0, 0, Fraction(0), Fraction(0), ()
            )
            continue
        left = bisect.bisect_right(times, lower)
        right = bisect.bisect_right(times, endpoint)
        rows = ordered[left:right]
        recipient_amounts: Counter[str] = Counter()
        recipient_events: Counter[str] = Counter()
        for event in rows:
            recipient_amounts[event.recipient] += event.amount_raw
            recipient_events[event.recipient] += 1
        total_amount = sum(recipient_amounts.values())
        event_count = len(rows)
        recipient_count = len(recipient_amounts)
        valid = event_count >= 4 and recipient_count >= 3 and total_amount > 0
        amount_hhi = (
            Fraction(
                sum(value * value for value in recipient_amounts.values()),
                total_amount * total_amount,
            )
            if total_amount
            else Fraction(0)
        )
        event_hhi = (
            Fraction(
                sum(value * value for value in recipient_events.values()),
                event_count * event_count,
            )
            if event_count
            else Fraction(0)
        )
        metrics[endpoint] = WindowMetric(
            endpoint=endpoint,
            valid=valid,
            event_count=event_count,
            recipient_count=recipient_count,
            total_amount_raw=total_amount,
            amount_hhi=amount_hhi,
            event_count_hhi=event_hhi,
            row_identities=tuple(sorted(event.identity for event in rows)),
        )
    return metrics


def _nearest_rank_index(length: int, numerator: int, denominator: int) -> int:
    if length <= 0 or numerator <= 0 or numerator > denominator:
        raise RuntimeError("URCD nearest-rank request is invalid")
    return (numerator * length + denominator - 1) // denominator - 1


def _metric_stat(metric: WindowMetric, statistic: str) -> Fraction:
    if statistic == "amount_hhi":
        return metric.amount_hhi
    if statistic == "event_count_hhi":
        return metric.event_count_hhi
    if statistic == "recipient_breadth":
        return Fraction(metric.recipient_count, 1)
    raise RuntimeError("URCD unknown routing statistic")


def snapshot_at(
    metrics: Mapping[datetime, WindowMetric],
    endpoint: datetime,
    *,
    statistic: str,
    apply_materiality: bool,
) -> Snapshot:
    current = metrics[endpoint]
    references = [metrics.get(endpoint - day * ONE_DAY) for day in range(1, 181)]
    valid = [metric for metric in references if metric is not None and metric.valid]
    neutral = Snapshot(
        metric=current,
        statistic=statistic,
        current_stat=_metric_stat(current, statistic),
        q20=Fraction(0),
        q80=Fraction(0),
        q50_amount_raw=0,
        valid_prior_windows=len(valid),
        materiality_applied=apply_materiality,
        state=0,
    )
    if not current.valid or len(valid) < 120:
        return neutral
    stat_values = sorted(
        ((_metric_stat(metric, statistic), metric.endpoint) for metric in valid),
        key=lambda item: (item[0], item[1]),
    )
    amounts = sorted(
        ((metric.total_amount_raw, metric.endpoint) for metric in valid),
        key=lambda item: (item[0], item[1]),
    )
    q20 = stat_values[_nearest_rank_index(len(valid), 1, 5)][0]
    q80 = stat_values[_nearest_rank_index(len(valid), 4, 5)][0]
    q50 = amounts[_nearest_rank_index(len(valid), 1, 2)][0]
    current_stat = _metric_stat(current, statistic)
    material = (not apply_materiality) or current.total_amount_raw >= q50
    state = 0
    if material and q20 != q80:
        if statistic == "recipient_breadth":
            if current_stat >= q80:
                state = 1
            elif current_stat <= q20:
                state = -1
        else:
            if current_stat <= q20:
                state = 1
            elif current_stat >= q80:
                state = -1
    return Snapshot(
        metric=current,
        statistic=statistic,
        current_stat=current_stat,
        q20=q20,
        q80=q80,
        q50_amount_raw=q50,
        valid_prior_windows=len(valid),
        materiality_applied=apply_materiality,
        state=state,
    )


def signal_id(
    control: str,
    decision_time: datetime,
    side: int,
    row_identities: Sequence[tuple[str, str, int]],
) -> str:
    payload = {
        "candidate": CANDIDATE,
        "control": control,
        "decision_time": format_time(decision_time),
        "row_identities": [list(identity) for identity in sorted(row_identities)],
        "side": _side_text(side),
    }
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def build_raw_candidates(
    control: str,
    metrics: Mapping[datetime, WindowMetric],
    *,
    statistic: str = "amount_hhi",
    apply_materiality: bool = True,
) -> list[RawCandidate]:
    raw: list[RawCandidate] = []
    previous_state = 0
    seen: set[str] = set()
    for endpoint in iter_anchors(TRAIN_START - SIX_HOURS, SEALED_FROM):
        snapshot = snapshot_at(
            metrics,
            endpoint,
            statistic=statistic,
            apply_materiality=apply_materiality,
        )
        if (
            endpoint >= TRAIN_START
            and snapshot.state != 0
            and snapshot.state != previous_state
        ):
            identifier = signal_id(
                control,
                endpoint,
                snapshot.state,
                snapshot.metric.row_identities,
            )
            if identifier in seen:
                raise RuntimeError(f"URCD duplicate raw signal id in {control}")
            seen.add(identifier)
            raw.append(
                RawCandidate(
                    control=control,
                    decision_time=endpoint,
                    side=snapshot.state,
                    previous_state=previous_state,
                    snapshot=snapshot,
                    signal_id=identifier,
                )
            )
        previous_state = snapshot.state
    return raw


def stale_primary_candidates(primary: Sequence[RawCandidate]) -> list[RawCandidate]:
    output: list[RawCandidate] = []
    for candidate in primary:
        decision = candidate.decision_time + ONE_DAY
        identifier = signal_id(
            "stale_24h",
            decision,
            candidate.side,
            candidate.snapshot.metric.row_identities,
        )
        output.append(
            replace(
                candidate,
                control="stale_24h",
                decision_time=decision,
                signal_id=identifier,
            )
        )
    return output


def schedule_candidates(
    raw: Sequence[RawCandidate], control: str
) -> list[Candidate]:
    output: list[Candidate] = []
    for split, (start, end) in WINDOWS.items():
        contained = [
            candidate
            for candidate in raw
            if start <= candidate.entry_time and candidate.exit_time <= end
        ]
        prior_exit: datetime | None = None
        seen: set[str] = set()
        for candidate in sorted(
            contained, key=lambda row: (row.entry_time, row.signal_id)
        ):
            if candidate.control != control:
                raise RuntimeError("URCD scheduler control drift")
            if candidate.signal_id in seen:
                raise RuntimeError("URCD duplicate scheduled signal id")
            seen.add(candidate.signal_id)
            if prior_exit is not None and candidate.entry_time < prior_exit:
                continue
            output.append(
                Candidate(
                    control=control,
                    split=split,
                    decision_time=candidate.decision_time,
                    entry_time=candidate.entry_time,
                    exit_time=candidate.exit_time,
                    side=candidate.side,
                    previous_state=candidate.previous_state,
                    snapshot=candidate.snapshot,
                    signal_id=candidate.signal_id,
                )
            )
            prior_exit = candidate.exit_time
    return sorted(output, key=lambda row: (row.entry_time, row.signal_id))


def direction_flip(primary: Sequence[Candidate]) -> list[Candidate]:
    output: list[Candidate] = []
    for candidate in primary:
        side = -candidate.side
        identifier = signal_id(
            "direction_flip",
            candidate.decision_time,
            side,
            candidate.snapshot.metric.row_identities,
        )
        output.append(
            replace(
                candidate,
                control="direction_flip",
                side=side,
                signal_id=identifier,
            )
        )
    return output


def build_controls(events: Sequence[Event]) -> dict[str, list[Candidate]]:
    primary_metrics = build_window_metrics(events)
    recipient_metrics = build_window_metrics(
        permute_field(events, "recipient_year_permutation")
    )
    amount_metrics = build_window_metrics(
        permute_field(events, "amount_year_permutation")
    )
    primary_raw = build_raw_candidates("primary", primary_metrics)
    raw_controls: dict[str, list[RawCandidate]] = {
        "primary": primary_raw,
        "event_count_hhi": build_raw_candidates(
            "event_count_hhi", primary_metrics, statistic="event_count_hhi"
        ),
        "equal_recipient_breadth": build_raw_candidates(
            "equal_recipient_breadth",
            primary_metrics,
            statistic="recipient_breadth",
        ),
        "no_materiality": build_raw_candidates(
            "no_materiality", primary_metrics, apply_materiality=False
        ),
        "stale_24h": stale_primary_candidates(primary_raw),
        "recipient_year_permutation": build_raw_candidates(
            "recipient_year_permutation", recipient_metrics
        ),
        "amount_year_permutation": build_raw_candidates(
            "amount_year_permutation", amount_metrics
        ),
    }
    controls = {
        name: schedule_candidates(raw_controls[name], name) for name in raw_controls
    }
    controls["direction_flip"] = direction_flip(controls["primary"])
    if tuple(controls) != CONTROL_ORDER:
        raise RuntimeError("URCD control order drift")
    return controls


def split_statistics(rows: Sequence[Candidate], split: str) -> dict[str, Any]:
    selected = sorted(
        (row for row in rows if row.split == split), key=lambda row: row.entry_time
    )
    total = len(selected)
    years = Counter(str(row.entry_time.year) for row in selected)
    halves = Counter(
        f"{row.entry_time.year}H{1 if row.entry_time.month <= 6 else 2}"
        for row in selected
    )
    months = Counter(row.entry_time.strftime("%Y-%m") for row in selected)
    quarters = Counter(
        f"{row.entry_time.year}Q{(row.entry_time.month - 1) // 3 + 1}"
        for row in selected
    )
    side_counts = Counter(_side_text(row.side) for row in selected)
    maximum_month = Fraction(max(months.values()), total) if total else Fraction(1)
    maximum_quarter = (
        Fraction(max(quarters.values()), total) if total else Fraction(1)
    )
    gaps = [
        int((right.entry_time - left.entry_time).total_seconds())
        for left, right in zip(selected, selected[1:])
    ]
    longest = 0
    running = 0
    previous: int | None = None
    for row in selected:
        if row.side == previous:
            running += 1
        else:
            running = 1
            previous = row.side
        longest = max(longest, running)
    return {
        "trades": total,
        "side_counts": dict(sorted(side_counts.items())),
        "year_counts": dict(sorted(years.items())),
        "half_year_counts": dict(sorted(halves.items())),
        "month_counts": dict(sorted(months.items())),
        "quarter_counts": dict(sorted(quarters.items())),
        "maximum_month_share": _fraction_text(maximum_month),
        "maximum_quarter_share": _fraction_text(maximum_quarter),
        "maximum_gap_seconds": max(gaps) if gaps else None,
        "maximum_same_side_run": longest,
    }


def exact_entry_jaccard(
    left: Sequence[Candidate], right: Sequence[Candidate], split: str
) -> Fraction:
    left_set = {row.entry_time for row in left if row.split == split}
    right_set = {row.entry_time for row in right if row.split == split}
    union = left_set | right_set
    return Fraction(len(left_set & right_set), len(union)) if union else Fraction(1)


def same_side_reproduction(
    primary: Sequence[Candidate], control: Sequence[Candidate], split: str
) -> Fraction:
    left = {(row.entry_time, row.side) for row in primary if row.split == split}
    right = {(row.entry_time, row.side) for row in control if row.split == split}
    return Fraction(len(left & right), len(left)) if left else Fraction(1)


def source_support_checks(
    controls: Mapping[str, Sequence[Candidate]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    primary = controls["primary"]
    train = split_statistics(primary, "train")
    selection = split_statistics(primary, "selection")
    statistics = {"train": train, "selection": selection}
    train_total = train["trades"]
    selection_total = selection["trades"]
    checks: dict[str, bool] = {
        "train_total_minimum": train_total >= 80,
        "selection_total_minimum": selection_total >= 30,
        "each_train_year_minimum": all(
            train["year_counts"].get(str(year), 0) >= 30 for year in (2021, 2022)
        ),
        "each_train_half_minimum": all(
            train["half_year_counts"].get(f"{year}H{half}", 0) >= 12
            for year in (2021, 2022)
            for half in (1, 2)
        ),
        "each_selection_half_minimum": all(
            selection["half_year_counts"].get(f"2023H{half}", 0) >= 10
            for half in (1, 2)
        ),
        "train_each_side_minimum": all(
            train["side_counts"].get(side, 0) >= 16 for side in ("LONG", "SHORT")
        ),
        "selection_each_side_minimum": all(
            selection["side_counts"].get(side, 0) >= 6
            for side in ("LONG", "SHORT")
        ),
        "train_each_side_share": all(
            train_total > 0
            and Fraction(train["side_counts"].get(side, 0), train_total)
            >= Fraction(1, 5)
            for side in ("LONG", "SHORT")
        ),
        "selection_each_side_share": all(
            selection_total > 0
            and Fraction(selection["side_counts"].get(side, 0), selection_total)
            >= Fraction(1, 5)
            for side in ("LONG", "SHORT")
        ),
        "maximum_month_share": all(
            Fraction(statistics[split]["maximum_month_share"]) <= Fraction(1, 5)
            for split in WINDOWS
        ),
        "maximum_quarter_share": all(
            Fraction(statistics[split]["maximum_quarter_share"]) <= Fraction(2, 5)
            for split in WINDOWS
        ),
        "maximum_gap_days": all(
            statistics[split]["maximum_gap_seconds"] is not None
            and statistics[split]["maximum_gap_seconds"] <= 60 * 86_400
            for split in WINDOWS
        ),
        "maximum_same_side_run": all(
            statistics[split]["maximum_same_side_run"] <= 12 for split in WINDOWS
        ),
    }
    selectivity: dict[str, Any] = {}
    for control in PERMUTATION_CONTROLS:
        selectivity[control] = {}
        for split in WINDOWS:
            jaccard = exact_entry_jaccard(primary, controls[control], split)
            reproduction = same_side_reproduction(primary, controls[control], split)
            selectivity[control][split] = {
                "exact_entry_jaccard": _fraction_text(jaccard),
                "exact_same_side_reproduction": _fraction_text(reproduction),
            }
            checks[f"{control}_{split}_exact_entry_jaccard"] = (
                jaccard <= Fraction(7, 20)
            )
            checks[f"{control}_{split}_same_side_reproduction"] = (
                reproduction <= Fraction(3, 5)
            )
    statistics["permutation_selectivity"] = selectivity
    return statistics, checks


def _comparator_views() -> tuple[Mapping[str, Any], ...]:
    return prereg.COMPARATOR_SPECS


def load_comparator_entries(
    specs: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, tuple[datetime, ...]], dict[str, Any]]:
    selected_specs = tuple(specs) if specs is not None else _comparator_views()
    required = ("candidate", "control", "entry_time", "side")
    verified: list[dict[str, Any]] = []

    # Phase one is metadata-only.  No comparator data row may be read until
    # every member of the frozen cohort has passed its whole-file and header
    # contract.
    for spec in selected_specs:
        candidate_name = str(spec["candidate"])
        known_controls = KNOWN_COMPARATOR_CONTROLS.get(candidate_name)
        if known_controls is None and spec.get("known_controls") is not None:
            known_controls = frozenset(str(value) for value in spec["known_controls"])
        if known_controls is None:
            raise RuntimeError(f"URCD comparator candidate is unknown: {candidate_name}")
        selected_controls = tuple(str(value) for value in spec["controls"])
        if not set(selected_controls).issubset(known_controls):
            raise RuntimeError(f"URCD selected comparator control is unknown: {candidate_name}")
        if sha256_file(spec["path"]) != spec["sha256"]:
            raise RuntimeError(f"URCD comparator hash drift: {candidate_name}")
        with gzip.open(_path(spec["path"]), "rb") as binary:
            header_line = binary.readline()
        if hashlib.sha256(header_line).hexdigest() != spec["header_line_sha256"]:
            raise RuntimeError(f"URCD comparator header hash drift: {candidate_name}")
        header = tuple(next(csv.reader([header_line.decode("utf-8")])))
        if len(header) != len(set(header)):
            raise RuntimeError("URCD comparator duplicate header")
        if not set(required).issubset(header):
            raise RuntimeError("URCD comparator required header missing")
        lowered = tuple(field.lower() for field in header)
        if any(
            token in field
            for field in lowered
            for token in FORBIDDEN_COMPARATOR_TOKENS
        ):
            raise RuntimeError("URCD comparator outcome field forbidden")
        start_text = str(spec["comparison_start"])
        end_text = str(spec["comparison_end_exclusive"])
        start = parse_time(start_text)
        end = parse_time(end_text)
        if start >= end:
            raise RuntimeError("URCD comparator overlap interval is empty")
        verified.append(
            {
                "spec": spec,
                "candidate": candidate_name,
                "selected_controls": selected_controls,
                "known_controls": known_controls,
                "header": header,
                "index": {field: header.index(field) for field in required},
                "start_text": start_text,
                "end_text": end_text,
                "start": start,
                "end": end,
            }
        )

    views: dict[str, tuple[datetime, ...]] = {}
    total_rows_scanned = 0
    relevant_rows_decoded = 0
    out_of_overlap_timestamp_sentinels = 0
    file_audit: dict[str, Any] = {}
    allowed_sides = {"1", "-1", "LONG", "SHORT"}
    for item in verified:
        spec = item["spec"]
        candidate_name = item["candidate"]
        selected_controls = item["selected_controls"]
        known_controls = item["known_controls"]
        header = item["header"]
        index = item["index"]
        retained: dict[str, list[datetime]] = {
            f"{candidate_name}:{control}": [] for control in selected_controls
        }
        seen: set[tuple[str, str, datetime]] = set()
        start_text = item["start_text"]
        end_text = item["end_text"]
        start = item["start"]
        end = item["end"]
        scanned = 0
        relevant = 0
        out_of_overlap = 0
        with gzip.open(_path(spec["path"]), "rt", encoding="utf-8", newline="") as text:
            reader = csv.reader(text)
            next(reader)
            for row in reader:
                scanned += 1
                total_rows_scanned += 1
                if len(row) != len(header):
                    raise RuntimeError("URCD comparator row width drift")
                entry_text = row[index["entry_time"]]
                if not start_text <= entry_text < end_text:
                    out_of_overlap += 1
                    out_of_overlap_timestamp_sentinels += 1
                    continue
                candidate = row[index["candidate"]]
                control = row[index["control"]]
                if candidate != candidate_name:
                    continue
                if control not in known_controls:
                    raise RuntimeError(
                        f"URCD comparator row control is unknown: {candidate_name}:{control}"
                    )
                if control not in selected_controls:
                    continue
                relevant += 1
                relevant_rows_decoded += 1
                entry = parse_time(entry_text)
                side = row[index["side"]]
                if side not in allowed_sides:
                    raise RuntimeError("URCD comparator side drift")
                identity = (candidate, control, entry)
                if identity in seen:
                    raise RuntimeError("URCD comparator entry duplicated")
                seen.add(identity)
                if not start <= entry < end:
                    raise RuntimeError("URCD comparator lexical/time boundary mismatch")
                retained[f"{candidate}:{control}"].append(entry)
        for key, entries in retained.items():
            views[key] = tuple(sorted(entries))
        file_audit[candidate_name] = {
            "rows_scanned": scanned,
            "relevant_four_field_rows_decoded": relevant,
            "out_of_overlap_timestamp_sentinels_scanned": out_of_overlap,
            "header_line_sha256": spec["header_line_sha256"],
        }
    return views, {
        "files": file_audit,
        "physical_rows_scanned": total_rows_scanned,
        "relevant_four_field_rows_decoded": relevant_rows_decoded,
        "out_of_overlap_timestamp_sentinels_scanned": (
            out_of_overlap_timestamp_sentinels
        ),
        "allowed_columns": ["candidate", "control", "entry_time", "side"],
    }


def novelty_metrics(
    left: Sequence[datetime], right: Sequence[datetime], tolerance: timedelta
) -> dict[str, Any]:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    exact = Fraction(len(left_set & right_set), len(union)) if union else Fraction(1)

    def near_share(source: set[datetime], other: set[datetime]) -> Fraction:
        if not source:
            return Fraction(1)
        ordered = sorted(other)
        matched = 0
        for point in source:
            position = bisect.bisect_left(ordered, point)
            neighbors = ordered[max(0, position - 1) : position + 1]
            if any(abs(point - neighbor) <= tolerance for neighbor in neighbors):
                matched += 1
        return Fraction(matched, len(source))

    left_near = near_share(left_set, right_set)
    right_near = near_share(right_set, left_set)
    return {
        "left_entries": len(left_set),
        "right_entries": len(right_set),
        "exact_entry_jaccard": _fraction_text(exact),
        "left_near_share": _fraction_text(left_near),
        "right_near_share": _fraction_text(right_near),
        "maximum_bidirectional_containment": _fraction_text(max(left_near, right_near)),
    }


def evaluate_novelty(
    primary: Sequence[Candidate],
    views: Mapping[str, Sequence[datetime]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    report: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for spec in _comparator_views():
        start = parse_time(spec["comparison_start"])
        end = parse_time(spec["comparison_end_exclusive"])
        left = [
            row.entry_time for row in primary if start <= row.entry_time < end
        ]
        for control in spec["controls"]:
            key = f"{spec['candidate']}:{control}"
            metrics = novelty_metrics(left, views.get(key, ()), timedelta(hours=6))
            report[key] = metrics
            enough = metrics["left_entries"] >= 10 and metrics["right_entries"] >= 5
            checks[f"{key}:minimum_support"] = enough
            checks[f"{key}:exact_entry_jaccard"] = (
                Fraction(metrics["exact_entry_jaccard"]) <= Fraction(1, 10)
            )
            checks[f"{key}:bidirectional_containment"] = (
                Fraction(metrics["maximum_bidirectional_containment"])
                <= Fraction(2, 5)
            )
    return report, checks


def candidate_row(row: Candidate) -> dict[str, Any]:
    snapshot = row.snapshot
    metric = snapshot.metric
    return {
        "candidate": CANDIDATE,
        "control": row.control,
        "signal_id": row.signal_id,
        "split": row.split,
        "side": _side_text(row.side),
        "decision_time": format_time(row.decision_time),
        "source_window_start_exclusive": format_time(
            metric.endpoint - CURRENT_WINDOW
        ),
        "source_window_end_inclusive": format_time(metric.endpoint),
        "entry_time": format_time(row.entry_time),
        "exit_time": format_time(row.exit_time),
        "statistic": snapshot.statistic,
        "current_stat_num": snapshot.current_stat.numerator,
        "current_stat_den": snapshot.current_stat.denominator,
        "prior_q20_num": snapshot.q20.numerator,
        "prior_q20_den": snapshot.q20.denominator,
        "prior_q80_num": snapshot.q80.numerator,
        "prior_q80_den": snapshot.q80.denominator,
        "prior_q50_amount_raw": snapshot.q50_amount_raw,
        "valid_prior_windows": snapshot.valid_prior_windows,
        "materiality_applied": "1" if snapshot.materiality_applied else "0",
        "current_event_count": metric.event_count,
        "current_recipient_count": metric.recipient_count,
        "current_total_amount_raw": metric.total_amount_raw,
        "previous_state": (
            "NEUTRAL" if row.previous_state == 0 else _side_text(row.previous_state)
        ),
    }


def deterministic_gzip_csv(rows: Iterable[Mapping[str, Any]]) -> bytes:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["entry_time"]),
            str(row["signal_id"]),
            str(row["control"]),
        ),
    )
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
            writer = csv.DictWriter(text, fieldnames=CLOCK_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row in ordered:
                writer.writerow({field: row[field] for field in CLOCK_COLUMNS})
    return buffer.getvalue()


def _control_report(
    controls: Mapping[str, Sequence[Candidate]],
) -> dict[str, Any]:
    primary = controls["primary"]
    report: dict[str, Any] = {}
    for control in CONTROL_ORDER:
        rows = controls[control]
        report[control] = {
            "clock_rows": len(rows),
            "train": split_statistics(rows, "train"),
            "selection": split_statistics(rows, "selection"),
            "exact_entry_jaccard_to_primary": {
                split: _fraction_text(exact_entry_jaccard(primary, rows, split))
                for split in WINDOWS
            },
        }
    return report


def _build_core(
    events: Sequence[Event],
    source_audit: Mapping[str, Any],
    *,
    artifact_eligible: bool,
    comparator_loader: Callable[
        [], tuple[dict[str, tuple[datetime, ...]], dict[str, Any]]
    ]
    | None,
    clock_output: str | Path,
) -> tuple[dict[str, Any], bytes]:
    controls = build_controls(events)
    support_stats, support_checks = source_support_checks(controls)
    support_passed = all(support_checks.values())
    comparator_audit: Mapping[str, Any] = {
        "physical_rows_scanned": 0,
        "relevant_four_field_rows_decoded": 0,
        "out_of_overlap_timestamp_sentinels_scanned": 0,
        "allowed_columns": [],
        "files": {},
    }
    novelty_report: Mapping[str, Any] = {}
    novelty_checks: Mapping[str, bool] = {}
    novelty_status = "not_opened_source_support_failed"
    novelty_passed = False
    if support_passed and artifact_eligible:
        if comparator_loader is None:
            raise RuntimeError("URCD eligible support pass lacks comparator loader")
        views, comparator_audit = comparator_loader()
        novelty_report, novelty_checks = evaluate_novelty(controls["primary"], views)
        novelty_passed = all(novelty_checks.values())
        novelty_status = "passed" if novelty_passed else "failed"
    elif support_passed:
        novelty_status = "forbidden_for_injected_or_synthetic_build"
    rows = [
        candidate_row(row)
        for control in CONTROL_ORDER
        for row in controls[control]
    ]
    clock_bytes = deterministic_gzip_csv(rows)
    if not support_passed:
        decision = "retire_URCD_72_unchanged_before_comparators_and_outcomes"
    elif not artifact_eligible:
        decision = "synthetic_build_cannot_authorize_novelty_or_outcomes"
    elif not novelty_passed:
        decision = "retire_URCD_72_unchanged_before_outcomes"
    else:
        decision = "advance_to_strict_outcome_evaluator_freeze"
    outcome_boundary = {
        "source_value_rows_decoded": source_audit.get(
            "preseal_rows_scanned",
            source_audit.get("eligible_mint_value_rows_decoded", len(events)),
        ),
        "comparator_four_field_rows_decoded": comparator_audit.get(
            "relevant_four_field_rows_decoded", 0
        ),
        "btc_market_rows_decoded": 0,
        "funding_rows_decoded": 0,
        "future_return_rows_decoded": 0,
        "return_or_pnl_fields_decoded": 0,
        "pnl_cagr_mdd_values_decoded": 0,
        "post_2023_source_value_rows_decoded": 0,
        "network_calls": 0,
    }
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": CANDIDATE,
        "artifact_eligible": artifact_eligible,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_artifact": str(PREREGISTRATION_ARTIFACT),
        "preregistration_artifact_sha256": PREREGISTRATION_ARTIFACT_SHA256,
        "preregistration_manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        "preregistration_policy_hash": PREREGISTRATION_POLICY_HASH,
        "implementation": {
            "source": str(SCRIPT_PATH),
            "source_sha256": sha256_file(SCRIPT_PATH),
            "tests": str(TEST_PATH),
            "tests_sha256": sha256_file(TEST_PATH),
            "contract": str(IMPLEMENTATION_CONTRACT),
            "contract_sha256": IMPLEMENTATION_CONTRACT_SHA256,
        },
        "source_binding": {
            "csv": str(SOURCE_CSV),
            "csv_sha256": SOURCE_CSV_SHA256,
            "manifest": str(SOURCE_MANIFEST),
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
        },
        "source_audit": dict(source_audit),
        "clock": {
            "path": str(clock_output),
            "sha256": hashlib.sha256(clock_bytes).hexdigest(),
            "rows": len(rows),
            "columns": list(CLOCK_COLUMNS),
            "control_counts": {
                control: len(controls[control]) for control in CONTROL_ORDER
            },
        },
        "primary_support": support_stats,
        "support_checks": support_checks,
        "source_support_passed": support_passed,
        "control_report": _control_report(controls),
        "novelty_status": novelty_status,
        "novelty_report": dict(novelty_report),
        "novelty_checks": dict(novelty_checks),
        "novelty_passed": novelty_passed,
        "comparator_audit": dict(comparator_audit),
        "decision": decision,
        "advance_to_strict_outcome_evaluator_freeze": (
            artifact_eligible and support_passed and novelty_passed
        ),
        "outcome_boundary": outcome_boundary,
    }
    return {**core, "manifest_hash": canonical_hash(core)}, clock_bytes


def build_support_from_events(
    events: Sequence[Event],
    *,
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
) -> tuple[dict[str, Any], bytes]:
    """Synthetic/injected entry point; it can never open comparators or advance."""
    audit = {
        "physical_rows_scanned": 0,
        "preseal_rows_scanned": 0,
        "sealed_timestamp_rows_scanned": 0,
        "eligible_mint_value_rows_decoded": len(events),
        "unique_eligible_identities": len({event.identity for event in events}),
        "post_2023_source_value_rows_decoded": 0,
        "synthetic_or_injected": True,
    }
    return _build_core(
        events,
        audit,
        artifact_eligible=False,
        comparator_loader=None,
        clock_output=clock_output,
    )


def build_real_support_payload(
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
) -> tuple[dict[str, Any], bytes]:
    _assert_protocol_committed()
    validate_preregistration()
    binding_audit = verify_pre_source_bindings()
    events, audit = load_source_events()
    audit = {**audit, "pre_source_bindings": binding_audit}
    return _build_core(
        events,
        audit,
        artifact_eligible=True,
        comparator_loader=load_comparator_entries,
        clock_output=clock_output,
    )


def _write_once(path: str | Path, payload: bytes) -> None:
    destination = _path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise FileExistsError(f"URCD support artifact is write-once: {path}") from exc


def write_support(
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
) -> dict[str, Any]:
    if Path(clock_output) != DEFAULT_CLOCK_OUTPUT or Path(report_output) != DEFAULT_REPORT_OUTPUT:
        raise RuntimeError("URCD eligible support outputs must use frozen default paths")
    if _path(clock_output).exists() or _path(report_output).exists():
        raise FileExistsError("URCD support outputs are write-once")
    payload, clock_bytes = build_real_support_payload(clock_output)
    report_bytes = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    _write_once(clock_output, clock_bytes)
    try:
        _write_once(report_output, report_bytes)
    except Exception:
        _path(clock_output).unlink(missing_ok=True)
        raise
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK_OUTPUT))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT_OUTPUT))
    args = parser.parse_args()
    payload = write_support(args.clock_output, args.report_output)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
