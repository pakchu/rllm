"""Build source-only WSCF-72 clocks, controls, and novelty diagnostics."""

from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from training import preregister_wbtc_stablecoin_finalized_confirmation_relay as prereg


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "wbtc_stablecoin_finalized_confirmation_relay_support_v1"
CANDIDATE = prereg.POLICY_ID
PREREGISTRATION_COMMIT = "8729160"
PREREGISTRATION_ARTIFACT = prereg.DEFAULT_OUTPUT
PREREGISTRATION_ARTIFACT_SHA256 = (
    "b105051e2b3bdf806c3abff30312889656534f49914ca4e4f584cb9723fb2fe0"
)
PREREGISTRATION_MANIFEST_HASH = (
    "1466ec5118df70985dda8692df1496d2d03285449ecadd5f8fcdec216b3f978f"
)
SCRIPT_PATH = Path(
    "training/build_wbtc_stablecoin_finalized_confirmation_relay_support.py"
)
DEFAULT_CLOCK_OUTPUT = Path(
    "data/wbtc_stablecoin_finalized_confirmation_relay_2021_2023/"
    "wscf72_support_clocks_2021_2023.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/wbtc_stablecoin_finalized_confirmation_relay_"
    "support_2026-07-23.json"
)

UTC = timezone.utc
ZERO_ADDRESS = "0x" + "0" * 40
ADDRESS = re.compile(r"^0x[0-9a-f]{40}$")
SOURCE_SEALED_FROM = datetime(2024, 1, 1, tzinfo=UTC)
CONFIRMATION_HOURS = 12
HOLD_HOURS = 72
EXPECTED_WBTC_ROWS = 993
EXPECTED_STABLECOIN_ROWS_BEFORE_SEAL = 266_360
EXPECTED_STABLECOIN_DIRECTIONAL_ROWS = 265_718
EXPECTED_STABLECOIN_VETO_ROWS = 642

WINDOWS = {
    "train": ("2021-01-01T00:00:00Z", "2023-01-01T00:00:00Z"),
    "selection": ("2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
}

CONTROL_ORDER = (
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
)

CLOCK_COLUMNS = (
    "candidate",
    "control",
    "signal_id",
    "window",
    "wbtc_available_at",
    "anchor_time",
    "signal_time",
    "entry_time",
    "exit_time",
    "side",
    "wbtc_batch_identity",
    "wbtc_net_raw",
    "wbtc_gross_raw",
    "wbtc_rows",
    "wbtc_distinct_actors",
    "wbtc_top_actor_share",
    "confirmation_batch_identity",
    "stablecoin_scope",
    "cumulative_stablecoin_net_raw",
    "cumulative_stablecoin_gross_raw",
    "stablecoin_batches",
    "confirmation_delay_seconds",
    "usdc_net_raw",
    "usdc_gross_raw",
    "usdt_net_raw",
    "usdt_gross_raw",
)


@dataclass(frozen=True)
class SourceEvent:
    source: str
    asset: str
    event: str
    sign: int
    amount_raw: int
    available_at: datetime
    identity: tuple[int, int, int]
    actor: str = ""


@dataclass(frozen=True)
class FlowBatch:
    source: str
    scope: str
    available_at: datetime
    identity_hash: str
    identities: tuple[tuple[int, int, int], ...]
    net_raw: int
    gross_raw: int
    rows: int
    actors: tuple[str, ...] = ()
    top_actor_share: float = 0.0
    usdc_net_raw: int = 0
    usdc_gross_raw: int = 0
    usdt_net_raw: int = 0
    usdt_gross_raw: int = 0


@dataclass(frozen=True)
class RawCandidate:
    control: str
    anchor_time: datetime
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    side: int
    wbtc: FlowBatch | None
    confirmation_identity: str
    stablecoin_scope: str
    cumulative_net_raw: int
    cumulative_gross_raw: int
    stablecoin_batches: int
    usdc_net_raw: int
    usdc_gross_raw: int
    usdt_net_raw: int
    usdt_gross_raw: int


@dataclass(frozen=True)
class Candidate(RawCandidate):
    window: str = ""

    @property
    def signal_id(self) -> str:
        payload = {
            "candidate": CANDIDATE,
            "control": self.control,
            "anchor_time": format_time(self.anchor_time),
            "signal_time": format_time(self.signal_time),
            "entry_time": format_time(self.entry_time),
            "side": self.side,
            "wbtc_batch_identity": self.wbtc.identity_hash if self.wbtc else "",
            "confirmation_batch_identity": self.confirmation_identity,
        }
        return canonical_hash(payload)


@dataclass(frozen=True)
class ComparatorEntry:
    entry_time: datetime
    side: int | None


@dataclass(frozen=True)
class ComparatorView:
    name: str
    start: datetime
    end: datetime
    entries: tuple[ComparatorEntry, ...]


class BatchIndex:
    def __init__(self, batches: Sequence[FlowBatch]) -> None:
        self.batches = tuple(
            sorted(batches, key=lambda row: (row.available_at, row.identity_hash))
        )
        self.times = tuple(row.available_at for row in self.batches)

    def after_through(self, start: datetime, end: datetime) -> tuple[FlowBatch, ...]:
        left = bisect.bisect_right(self.times, start)
        right = bisect.bisect_right(self.times, end)
        return self.batches[left:right]

    def after_before_or_at(
        self, start_exclusive: datetime, end_inclusive: datetime
    ) -> tuple[FlowBatch, ...]:
        return self.after_through(start_exclusive, end_inclusive)


class TimeIndex:
    def __init__(self, values: Sequence[datetime]) -> None:
        self.values = tuple(sorted(values))

    def any_after_through(self, start: datetime, end: datetime) -> bool:
        left = bisect.bisect_right(self.values, start)
        right = bisect.bisect_right(self.values, end)
        return right > left


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("WSCF timestamp must be UTC")
    return parsed.astimezone(UTC)


def format_time(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_positive_int(value: str, field: str) -> int:
    if not value or not value.isdigit() or int(value) <= 0:
        raise ValueError(f"WSCF {field} must be a canonical positive integer")
    return int(value)


def _canonical_nonnegative_int(value: str, field: str) -> int:
    if not value or not value.isdigit():
        raise ValueError(f"WSCF {field} must be a canonical nonnegative integer")
    return int(value)


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def ceil_5m_plus_one_bar(value: datetime) -> datetime:
    epoch_seconds = int(value.timestamp())
    ceiling = ((epoch_seconds + 299) // 300) * 300
    return datetime.fromtimestamp(ceiling + 300, tz=UTC)


def validate_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION_ARTIFACT) != PREREGISTRATION_ARTIFACT_SHA256:
        raise RuntimeError("WSCF preregistration artifact SHA-256 drift")
    payload = json.loads(_path(PREREGISTRATION_ARTIFACT).read_text(encoding="utf-8"))
    prereg.validate_preregistration(payload)
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("WSCF preregistration manifest hash drift")
    if payload.get("candidate") != CANDIDATE:
        raise RuntimeError("WSCF preregistration candidate drift")
    if payload.get("exact_source_incidence_opened") is not False:
        raise RuntimeError("WSCF preregistration opened exact incidence")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("WSCF preregistration opened outcomes")
    return payload


def load_wbtc_events(
    path: str | Path = prereg.WBTC_SOURCE,
    *,
    expected_rows: int = EXPECTED_WBTC_ROWS,
) -> tuple[list[SourceEvent], dict[str, Any]]:
    events: list[SourceEvent] = []
    identities: set[tuple[int, int, int]] = set()
    previous: tuple[datetime, tuple[int, int, int]] | None = None
    with gzip.open(_path(path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != prereg.WBTC_HEADER:
            raise RuntimeError("WSCF WBTC source header drift")
        for row in reader:
            available_at = parse_time(row["available_at"])
            if available_at >= SOURCE_SEALED_FROM:
                raise RuntimeError("WSCF WBTC source crossed sealed boundary")
            if row["asset"] != "wbtc_eth" or row["decimals"] != "8":
                raise RuntimeError("WSCF WBTC asset or decimals drift")
            event = row["event"]
            expected_sign = {"mint": 1, "burn": -1}.get(event)
            if expected_sign is None or row["event_sign"] != str(expected_sign):
                raise RuntimeError("WSCF WBTC event mapping drift")
            block = _canonical_nonnegative_int(row["block_number"], "block_number")
            confirmation = _canonical_nonnegative_int(
                row["confirmation_block_number"], "confirmation_block_number"
            )
            if confirmation != block + 64:
                raise RuntimeError("WSCF WBTC N+64 confirmation drift")
            identity = (
                block,
                _canonical_nonnegative_int(
                    row["transaction_index"], "transaction_index"
                ),
                _canonical_nonnegative_int(
                    row["semantic_log_index"], "semantic_log_index"
                ),
            )
            if identity in identities:
                raise RuntimeError("WSCF WBTC identity duplicated")
            identities.add(identity)
            actor = row["actor_address"].lower()
            if not ADDRESS.fullmatch(actor) or actor == ZERO_ADDRESS:
                raise RuntimeError("WSCF WBTC actor drift")
            parsed = SourceEvent(
                source="wbtc",
                asset="wbtc_eth",
                event=event,
                sign=expected_sign,
                amount_raw=_canonical_positive_int(row["amount_raw"], "amount_raw"),
                available_at=available_at,
                identity=identity,
                actor=actor,
            )
            order_key = (parsed.available_at, parsed.identity)
            if previous is not None and order_key < previous:
                raise RuntimeError("WSCF WBTC source is not sorted")
            previous = order_key
            events.append(parsed)
    if len(events) != expected_rows:
        raise RuntimeError("WSCF WBTC row count drift")
    return events, {
        "physical_rows_read": len(events),
        "eligible_rows": len(events),
        "unique_identities": len(identities),
        "first_available_at": format_time(events[0].available_at) if events else None,
        "last_available_at": format_time(events[-1].available_at) if events else None,
        "post_2023_contract_event_value_rows_loaded": 0,
    }


def _raw_available_at(line: str) -> datetime:
    stripped = line.rstrip("\r\n")
    if not stripped or "," not in stripped:
        raise RuntimeError("WSCF stablecoin physical row malformed")
    return parse_time(stripped.rsplit(",", 1)[1])


def load_stablecoin_events(
    path: str | Path = prereg.STABLECOIN_SOURCE,
    *,
    expected_rows_before_seal: int = EXPECTED_STABLECOIN_ROWS_BEFORE_SEAL,
    expected_directional_rows: int = EXPECTED_STABLECOIN_DIRECTIONAL_ROWS,
    expected_veto_rows: int = EXPECTED_STABLECOIN_VETO_ROWS,
) -> tuple[list[SourceEvent], list[SourceEvent], dict[str, Any]]:
    directional: list[SourceEvent] = []
    veto: list[SourceEvent] = []
    identities: set[tuple[int, int, int]] = set()
    previous: tuple[datetime, tuple[int, int, int]] | None = None
    rows_before_seal = 0
    boundary_sentinels = 0
    allowed = {
        ("usdc_eth", "mint"): 1,
        ("usdc_eth", "burn"): -1,
        ("usdt_eth", "issue"): 1,
        ("usdt_eth", "redeem"): -1,
        ("usdt_eth", "destroyed_black_funds"): -1,
    }
    with gzip.open(_path(path), "rt", encoding="utf-8", newline="") as handle:
        header_line = handle.readline()
        header = tuple(next(csv.reader([header_line])))
        if header != prereg.STABLECOIN_HEADER:
            raise RuntimeError("WSCF stablecoin source header drift")
        for line in handle:
            available_at = _raw_available_at(line)
            if available_at >= SOURCE_SEALED_FROM:
                boundary_sentinels += 1
                break
            values = next(csv.reader([line]))
            if len(values) != len(header):
                raise RuntimeError("WSCF stablecoin source column drift")
            row = dict(zip(header, values))
            rows_before_seal += 1
            key = (row["asset"], row["event"])
            expected_sign = allowed.get(key)
            if expected_sign is None or row["event_sign"] != str(expected_sign):
                raise RuntimeError("WSCF stablecoin event mapping drift")
            if row["decimals"] != "6":
                raise RuntimeError("WSCF stablecoin decimals drift")
            block = _canonical_nonnegative_int(row["block_number"], "block_number")
            confirmation = _canonical_nonnegative_int(
                row["confirmation_block_number"], "confirmation_block_number"
            )
            if confirmation != block + 64:
                raise RuntimeError("WSCF stablecoin N+64 confirmation drift")
            identity = (
                block,
                _canonical_nonnegative_int(
                    row["transaction_index"], "transaction_index"
                ),
                _canonical_nonnegative_int(row["log_index"], "log_index"),
            )
            if identity in identities:
                raise RuntimeError("WSCF stablecoin identity duplicated")
            identities.add(identity)
            parsed = SourceEvent(
                source="stablecoin",
                asset=row["asset"],
                event=row["event"],
                sign=expected_sign,
                amount_raw=_canonical_positive_int(row["amount_raw"], "amount_raw"),
                available_at=available_at,
                identity=identity,
            )
            order_key = (parsed.available_at, parsed.identity)
            if previous is not None and order_key < previous:
                raise RuntimeError("WSCF stablecoin source is not sorted")
            previous = order_key
            if parsed.event == "destroyed_black_funds":
                veto.append(parsed)
            else:
                directional.append(parsed)
    if rows_before_seal != expected_rows_before_seal:
        raise RuntimeError("WSCF stablecoin pre-seal row count drift")
    if len(directional) != expected_directional_rows:
        raise RuntimeError("WSCF stablecoin directional row count drift")
    if len(veto) != expected_veto_rows:
        raise RuntimeError("WSCF stablecoin veto row count drift")
    if boundary_sentinels != 1:
        raise RuntimeError("WSCF stablecoin boundary sentinel drift")
    all_events = sorted(
        directional + veto,
        key=lambda event: (event.available_at, event.identity),
    )
    return directional, veto, {
        "physical_rows_read_before_seal": rows_before_seal,
        "directional_rows": len(directional),
        "veto_rows": len(veto),
        "unique_identities": len(identities),
        "first_available_at": format_time(all_events[0].available_at)
        if all_events
        else None,
        "last_available_at_before_seal": format_time(all_events[-1].available_at)
        if all_events
        else None,
        "sealed_from": format_time(SOURCE_SEALED_FROM),
        "boundary_sentinel_timestamp_rows_scanned": boundary_sentinels,
        "sealed_non_timestamp_fields_decoded": 0,
        "post_2023_contract_event_value_rows_loaded": 0,
    }


def _batch_identity(
    source: str,
    scope: str,
    available_at: datetime,
    identities: Sequence[tuple[int, int, int]],
) -> str:
    return canonical_hash(
        {
            "source": source,
            "scope": scope,
            "available_at": format_time(available_at),
            "identities": [list(identity) for identity in identities],
        }
    )


def batch_events(
    events: Sequence[SourceEvent], *, source: str, scope: str
) -> list[FlowBatch]:
    ordered = sorted(events, key=lambda row: (row.available_at, row.identity))
    batches: list[FlowBatch] = []
    cursor = 0
    while cursor < len(ordered):
        available_at = ordered[cursor].available_at
        end = cursor + 1
        while end < len(ordered) and ordered[end].available_at == available_at:
            end += 1
        rows = ordered[cursor:end]
        identities = tuple(row.identity for row in rows)
        actor_gross: Counter[str] = Counter()
        for row in rows:
            if row.actor:
                actor_gross[row.actor] += row.amount_raw
        gross = sum(row.amount_raw for row in rows)
        usdc = [row for row in rows if row.asset == "usdc_eth"]
        usdt = [row for row in rows if row.asset == "usdt_eth"]
        batches.append(
            FlowBatch(
                source=source,
                scope=scope,
                available_at=available_at,
                identity_hash=_batch_identity(
                    source, scope, available_at, identities
                ),
                identities=identities,
                net_raw=sum(row.sign * row.amount_raw for row in rows),
                gross_raw=gross,
                rows=len(rows),
                actors=tuple(sorted(actor_gross)),
                top_actor_share=(
                    max(actor_gross.values()) / gross
                    if actor_gross and gross > 0
                    else 0.0
                ),
                usdc_net_raw=sum(row.sign * row.amount_raw for row in usdc),
                usdc_gross_raw=sum(row.amount_raw for row in usdc),
                usdt_net_raw=sum(row.sign * row.amount_raw for row in usdt),
                usdt_gross_raw=sum(row.amount_raw for row in usdt),
            )
        )
        cursor = end
    return batches


def stablecoin_batches(
    events: Sequence[SourceEvent], scope: str = "combined"
) -> list[FlowBatch]:
    if scope == "combined":
        selected = list(events)
    elif scope == "usdc":
        selected = [row for row in events if row.asset == "usdc_eth"]
    elif scope == "usdt":
        selected = [row for row in events if row.asset == "usdt_eth"]
    else:
        raise ValueError("WSCF stablecoin scope is unsupported")
    return batch_events(selected, source="stablecoin", scope=scope)


def permute_stablecoin_amounts(events: Sequence[SourceEvent]) -> list[SourceEvent]:
    groups: dict[tuple[str, str, int], list[SourceEvent]] = defaultdict(list)
    for event in events:
        groups[(event.asset, event.event, event.available_at.year)].append(event)
    output: list[SourceEvent] = []
    for key in sorted(groups):
        canonical = sorted(groups[key], key=lambda event: event.identity)
        donors = sorted(
            canonical,
            key=lambda event: hashlib.sha256(
                f"WSCF-AMOUNT-PERMUTE|{key}|{event.identity}".encode("ascii")
            ).digest(),
        )
        output.extend(
            replace(target, amount_raw=donor.amount_raw)
            for target, donor in zip(canonical, donors)
        )
    return sorted(output, key=lambda event: (event.available_at, event.identity))


def _raw_candidate(
    *,
    control: str,
    anchor_time: datetime,
    signal_time: datetime,
    side: int,
    wbtc: FlowBatch | None,
    confirmation_identity: str,
    stablecoin_scope: str,
    cumulative_net_raw: int,
    cumulative_gross_raw: int,
    stablecoin_batches_seen: int,
    usdc_net_raw: int,
    usdc_gross_raw: int,
    usdt_net_raw: int,
    usdt_gross_raw: int,
) -> RawCandidate:
    if side not in {-1, 1}:
        raise ValueError("WSCF candidate side must be nonzero")
    entry = ceil_5m_plus_one_bar(signal_time)
    return RawCandidate(
        control=control,
        anchor_time=anchor_time,
        signal_time=signal_time,
        entry_time=entry,
        exit_time=entry + timedelta(hours=HOLD_HOURS),
        side=side,
        wbtc=wbtc,
        confirmation_identity=confirmation_identity,
        stablecoin_scope=stablecoin_scope,
        cumulative_net_raw=cumulative_net_raw,
        cumulative_gross_raw=cumulative_gross_raw,
        stablecoin_batches=stablecoin_batches_seen,
        usdc_net_raw=usdc_net_raw,
        usdc_gross_raw=usdc_gross_raw,
        usdt_net_raw=usdt_net_raw,
        usdt_gross_raw=usdt_gross_raw,
    )


def confirmation_candidates(
    wbtc_batches: Sequence[FlowBatch],
    stable_batches: Sequence[FlowBatch],
    *,
    control: str,
    relation: str,
    shift_hours: int = 0,
    veto_times: Sequence[datetime] = (),
) -> list[RawCandidate]:
    stable_index = BatchIndex(stable_batches)
    veto_index = TimeIndex(veto_times)
    output: list[RawCandidate] = []
    for wbtc in wbtc_batches:
        wbtc_sign = _sign(wbtc.net_raw)
        if wbtc_sign == 0:
            continue
        anchor_time = wbtc.available_at + timedelta(hours=shift_hours)
        end = anchor_time + timedelta(hours=CONFIRMATION_HOURS)
        cumulative_net = cumulative_gross = 0
        usdc_net = usdc_gross = usdt_net = usdt_gross = 0
        batches_seen = 0
        for stable in stable_index.after_through(anchor_time, end):
            if veto_times and veto_index.any_after_through(
                anchor_time, stable.available_at
            ):
                break
            batches_seen += 1
            cumulative_net += stable.net_raw
            cumulative_gross += stable.gross_raw
            usdc_net += stable.usdc_net_raw
            usdc_gross += stable.usdc_gross_raw
            usdt_net += stable.usdt_net_raw
            usdt_gross += stable.usdt_gross_raw
            cumulative_sign = _sign(cumulative_net)
            matched = {
                "same": cumulative_sign == wbtc_sign,
                "opposite": cumulative_sign == -wbtc_sign,
                "first_nonzero": cumulative_sign != 0,
            }.get(relation)
            if matched is None:
                raise ValueError("WSCF confirmation relation is unsupported")
            if not matched:
                continue
            side = cumulative_sign if relation == "first_nonzero" else wbtc_sign
            output.append(
                _raw_candidate(
                    control=control,
                    anchor_time=anchor_time,
                    signal_time=stable.available_at,
                    side=side,
                    wbtc=wbtc,
                    confirmation_identity=stable.identity_hash,
                    stablecoin_scope=stable.scope,
                    cumulative_net_raw=cumulative_net,
                    cumulative_gross_raw=cumulative_gross,
                    stablecoin_batches_seen=batches_seen,
                    usdc_net_raw=usdc_net,
                    usdc_gross_raw=usdc_gross,
                    usdt_net_raw=usdt_net,
                    usdt_gross_raw=usdt_gross,
                )
            )
            break
    return output


def wbtc_only_candidates(wbtc_batches: Sequence[FlowBatch]) -> list[RawCandidate]:
    return [
        _raw_candidate(
            control="wbtc_only_direct",
            anchor_time=batch.available_at,
            signal_time=batch.available_at,
            side=_sign(batch.net_raw),
            wbtc=batch,
            confirmation_identity="",
            stablecoin_scope="none",
            cumulative_net_raw=0,
            cumulative_gross_raw=0,
            stablecoin_batches_seen=0,
            usdc_net_raw=0,
            usdc_gross_raw=0,
            usdt_net_raw=0,
            usdt_gross_raw=0,
        )
        for batch in wbtc_batches
        if batch.net_raw != 0
    ]


def _aggregate_batches(
    batches: Sequence[FlowBatch], *, scope: str, identity_prefix: str
) -> tuple[str, int, int, int, int, int, int, int, int]:
    identity = canonical_hash(
        {
            "prefix": identity_prefix,
            "scope": scope,
            "batch_identities": [batch.identity_hash for batch in batches],
        }
    )
    return (
        identity,
        sum(batch.net_raw for batch in batches),
        sum(batch.gross_raw for batch in batches),
        len(batches),
        sum(batch.usdc_net_raw for batch in batches),
        sum(batch.usdc_gross_raw for batch in batches),
        sum(batch.usdt_net_raw for batch in batches),
        sum(batch.usdt_gross_raw for batch in batches),
        _sign(sum(batch.net_raw for batch in batches)),
    )


def stablecoin_grid_candidates(
    stable_batches: Sequence[FlowBatch],
) -> list[RawCandidate]:
    index = BatchIndex(stable_batches)
    current = parse_time(WINDOWS["train"][0])
    end = parse_time(WINDOWS["selection"][1])
    output: list[RawCandidate] = []
    while current < end:
        selected = index.after_before_or_at(current - timedelta(hours=12), current)
        (
            identity,
            net,
            gross,
            rows,
            usdc_net,
            usdc_gross,
            usdt_net,
            usdt_gross,
            side,
        ) = _aggregate_batches(
            selected,
            scope="combined",
            identity_prefix=f"stablecoin-grid|{format_time(current)}",
        )
        if side:
            output.append(
                _raw_candidate(
                    control="stablecoin_only_12h_grid",
                    anchor_time=current,
                    signal_time=current,
                    side=side,
                    wbtc=None,
                    confirmation_identity=identity,
                    stablecoin_scope="combined",
                    cumulative_net_raw=net,
                    cumulative_gross_raw=gross,
                    stablecoin_batches_seen=rows,
                    usdc_net_raw=usdc_net,
                    usdc_gross_raw=usdc_gross,
                    usdt_net_raw=usdt_net,
                    usdt_gross_raw=usdt_gross,
                )
            )
        current += timedelta(hours=12)
    return output


def lead_lag_reverse_candidates(
    wbtc_batches: Sequence[FlowBatch], stable_batches: Sequence[FlowBatch]
) -> list[RawCandidate]:
    index = BatchIndex(stable_batches)
    output: list[RawCandidate] = []
    for wbtc in wbtc_batches:
        side = _sign(wbtc.net_raw)
        if side == 0:
            continue
        selected = index.after_before_or_at(
            wbtc.available_at - timedelta(hours=12), wbtc.available_at
        )
        (
            identity,
            net,
            gross,
            rows,
            usdc_net,
            usdc_gross,
            usdt_net,
            usdt_gross,
            stable_side,
        ) = _aggregate_batches(
            selected,
            scope="combined",
            identity_prefix=f"lead-lag-reverse|{wbtc.identity_hash}",
        )
        if stable_side != side:
            continue
        output.append(
            _raw_candidate(
                control="lead_lag_reverse",
                anchor_time=wbtc.available_at,
                signal_time=wbtc.available_at,
                side=side,
                wbtc=wbtc,
                confirmation_identity=identity,
                stablecoin_scope="combined",
                cumulative_net_raw=net,
                cumulative_gross_raw=gross,
                stablecoin_batches_seen=rows,
                usdc_net_raw=usdc_net,
                usdc_gross_raw=usdc_gross,
                usdt_net_raw=usdt_net,
                usdt_gross_raw=usdt_gross,
            )
        )
    return output


def _window_for(entry_time: datetime, exit_time: datetime) -> str | None:
    for name, raw in WINDOWS.items():
        start, end = map(parse_time, raw)
        if start <= entry_time and exit_time <= end:
            return name
    return None


def schedule(raw: Sequence[RawCandidate], control: str) -> list[Candidate]:
    ordered = sorted(
        raw,
        key=lambda row: (
            row.entry_time,
            row.signal_time,
            row.anchor_time,
            row.wbtc.identity_hash if row.wbtc else "",
            row.confirmation_identity,
            row.side,
        ),
    )
    output: list[Candidate] = []
    last_exit: datetime | None = None
    used_confirmations: set[str] = set()
    for row in ordered:
        window = _window_for(row.entry_time, row.exit_time)
        if window is None:
            continue
        if last_exit is not None and row.entry_time < last_exit:
            continue
        if row.confirmation_identity and row.confirmation_identity in used_confirmations:
            continue
        candidate = Candidate(**replace(row, control=control).__dict__, window=window)
        output.append(candidate)
        last_exit = candidate.exit_time
        if candidate.confirmation_identity:
            used_confirmations.add(candidate.confirmation_identity)
    return output


def exact_clock_control(
    primary: Sequence[Candidate], control: str, sides: Sequence[int]
) -> list[Candidate]:
    if len(primary) != len(sides):
        raise ValueError("WSCF exact-clock side length mismatch")
    return [replace(row, control=control, side=side) for row, side in zip(primary, sides)]


def deterministic_random_sides(primary: Sequence[Candidate]) -> list[int]:
    sides = [row.side for row in primary]
    ranked = sorted(
        range(len(primary)),
        key=lambda index: hashlib.sha256(
            f"WSCF-RANDOM-SIDE|{primary[index].signal_id}".encode("ascii")
        ).digest(),
    )
    shuffled = sides.copy()
    for target, source in enumerate(ranked):
        shuffled[source] = sides[target]
    return shuffled


def build_controls(
    wbtc_events: Sequence[SourceEvent],
    stablecoin_events: Sequence[SourceEvent],
    veto_events: Sequence[SourceEvent],
) -> dict[str, list[Candidate]]:
    wbtc = batch_events(wbtc_events, source="wbtc", scope="wbtc")
    combined = stablecoin_batches(stablecoin_events, "combined")
    usdc = stablecoin_batches(stablecoin_events, "usdc")
    usdt = stablecoin_batches(stablecoin_events, "usdt")
    veto_times = [event.available_at for event in veto_events]

    controls: dict[str, list[Candidate]] = {}
    primary = schedule(
        confirmation_candidates(
            wbtc,
            combined,
            control="primary",
            relation="same",
        ),
        "primary",
    )
    controls["primary"] = primary
    controls["direction_flip"] = exact_clock_control(
        primary, "direction_flip", [-row.side for row in primary]
    )
    controls["deterministic_random_side"] = exact_clock_control(
        primary,
        "deterministic_random_side",
        deterministic_random_sides(primary),
    )
    controls["wbtc_only_direct"] = schedule(
        wbtc_only_candidates(wbtc), "wbtc_only_direct"
    )
    controls["stablecoin_only_12h_grid"] = schedule(
        stablecoin_grid_candidates(combined), "stablecoin_only_12h_grid"
    )
    controls["anchored_first_nonzero"] = schedule(
        confirmation_candidates(
            wbtc,
            combined,
            control="anchored_first_nonzero",
            relation="first_nonzero",
        ),
        "anchored_first_nonzero",
    )
    controls["opposite_confirmation"] = schedule(
        confirmation_candidates(
            wbtc,
            combined,
            control="opposite_confirmation",
            relation="opposite",
        ),
        "opposite_confirmation",
    )
    controls["lead_lag_reverse"] = schedule(
        lead_lag_reverse_candidates(wbtc, combined), "lead_lag_reverse"
    )
    controls["stale_wbtc_24h"] = schedule(
        confirmation_candidates(
            wbtc,
            combined,
            control="stale_wbtc_24h",
            relation="same",
            shift_hours=24,
        ),
        "stale_wbtc_24h",
    )
    controls["stale_wbtc_72h"] = schedule(
        confirmation_candidates(
            wbtc,
            combined,
            control="stale_wbtc_72h",
            relation="same",
            shift_hours=72,
        ),
        "stale_wbtc_72h",
    )
    permuted = stablecoin_batches(
        permute_stablecoin_amounts(stablecoin_events), "combined"
    )
    controls["stablecoin_year_amount_permutation"] = schedule(
        confirmation_candidates(
            wbtc,
            permuted,
            control="stablecoin_year_amount_permutation",
            relation="same",
        ),
        "stablecoin_year_amount_permutation",
    )
    controls["black_funds_veto"] = schedule(
        confirmation_candidates(
            wbtc,
            combined,
            control="black_funds_veto",
            relation="same",
            veto_times=veto_times,
        ),
        "black_funds_veto",
    )
    controls["usdc_only_confirmation"] = schedule(
        confirmation_candidates(
            wbtc,
            usdc,
            control="usdc_only_confirmation",
            relation="same",
        ),
        "usdc_only_confirmation",
    )
    controls["usdt_only_confirmation"] = schedule(
        confirmation_candidates(
            wbtc,
            usdt,
            control="usdt_only_confirmation",
            relation="same",
        ),
        "usdt_only_confirmation",
    )
    if tuple(controls) != CONTROL_ORDER:
        raise RuntimeError("WSCF control order drift")
    return controls


def _maximum_run(rows: Sequence[Candidate]) -> int:
    best = current = 0
    previous: int | None = None
    for row in sorted(rows, key=lambda candidate: candidate.entry_time):
        current = current + 1 if row.side == previous else 1
        previous = row.side
        best = max(best, current)
    return best


def _maximum_gap_days(rows: Sequence[Candidate]) -> float | None:
    times = sorted(row.entry_time for row in rows)
    if len(times) < 2:
        return None
    return max((right - left).total_seconds() / 86400 for left, right in zip(times, times[1:]))


def _window_stats(rows: Sequence[Candidate], window: str) -> dict[str, Any]:
    selected = sorted(
        (row for row in rows if row.window == window),
        key=lambda row: row.entry_time,
    )
    years = Counter(str(row.entry_time.year) for row in selected)
    halves = Counter(
        f"{row.entry_time.year}H{1 if row.entry_time.month <= 6 else 2}"
        for row in selected
    )
    quarters = Counter(
        f"{row.entry_time.year}Q{(row.entry_time.month - 1) // 3 + 1}"
        for row in selected
    )
    months = Counter(row.entry_time.strftime("%Y-%m") for row in selected)
    sides = Counter("long" if row.side > 0 else "short" for row in selected)
    actors = sorted(
        {actor for row in selected if row.wbtc for actor in row.wbtc.actors}
    )
    wbtc_ids = [row.wbtc.identity_hash for row in selected if row.wbtc]
    confirmation_ids = [
        row.confirmation_identity for row in selected if row.confirmation_identity
    ]
    return {
        "trades": len(selected),
        "year_counts": dict(sorted(years.items())),
        "half_year_counts": dict(sorted(halves.items())),
        "quarter_counts": dict(sorted(quarters.items())),
        "month_counts": dict(sorted(months.items())),
        "active_months": len(months),
        "side_counts": {side: sides.get(side, 0) for side in ("long", "short")},
        "maximum_month_share": (
            max(months.values(), default=0) / len(selected) if selected else 0.0
        ),
        "maximum_quarter_share": (
            max(quarters.values(), default=0) / len(selected) if selected else 0.0
        ),
        "maximum_consecutive_same_side": _maximum_run(selected),
        "maximum_calendar_gap_days": _maximum_gap_days(selected),
        "distinct_wbtc_actors": len(actors),
        "wbtc_actor_set_hash": hashlib.sha256(
            "\n".join(actors).encode("ascii")
        ).hexdigest(),
        "duplicate_wbtc_batch_identities": len(wbtc_ids) - len(set(wbtc_ids)),
        "duplicate_confirmation_batch_identities": (
            len(confirmation_ids) - len(set(confirmation_ids))
        ),
    }


def support_statistics(primary: Sequence[Candidate]) -> dict[str, Any]:
    return {
        "total_trades": len(primary),
        "all_year_counts": dict(
            sorted(Counter(str(row.entry_time.year) for row in primary).items())
        ),
        "all_side_counts": {
            side: sum(
                1 for row in primary if (row.side > 0) == (side == "long")
            )
            for side in ("long", "short")
        },
        "train": _window_stats(primary, "train"),
        "selection": _window_stats(primary, "selection"),
    }


def support_checks(stats: Mapping[str, Any]) -> dict[str, bool]:
    train = stats["train"]
    selection = stats["selection"]
    gates = prereg.policy_payload()["source_support_gates"]
    return {
        "train_total_minimum": train["trades"] >= gates["train_total_minimum"],
        "selection_total_minimum": (
            selection["trades"] >= gates["selection_total_minimum"]
        ),
        "each_train_year_minimum": all(
            train["year_counts"].get(str(year), 0)
            >= gates["each_train_year_minimum"]
            for year in (2021, 2022)
        ),
        "each_train_half_year_minimum": all(
            train["half_year_counts"].get(f"{year}H{half}", 0)
            >= gates["each_train_half_year_minimum"]
            for year in (2021, 2022)
            for half in (1, 2)
        ),
        "each_selection_half_year_minimum": all(
            selection["half_year_counts"].get(f"2023H{half}", 0)
            >= gates["each_selection_half_year_minimum"]
            for half in (1, 2)
        ),
        "train_each_side_minimum": all(
            train["side_counts"].get(side, 0)
            >= gates["train_each_side_minimum"]
            for side in ("long", "short")
        ),
        "selection_each_side_minimum": all(
            selection["side_counts"].get(side, 0)
            >= gates["selection_each_side_minimum"]
            for side in ("long", "short")
        ),
        "maximum_month_share": all(
            split["maximum_month_share"] <= gates["maximum_month_share"]
            for split in (train, selection)
        ),
        "maximum_quarter_share": all(
            split["maximum_quarter_share"] <= gates["maximum_quarter_share"]
            for split in (train, selection)
        ),
        "maximum_consecutive_same_side": all(
            split["maximum_consecutive_same_side"]
            <= gates["maximum_consecutive_same_side"]
            for split in (train, selection)
        ),
        "train_distinct_wbtc_actors_minimum": (
            train["distinct_wbtc_actors"]
            >= gates["train_distinct_wbtc_actors_minimum"]
        ),
        "selection_distinct_wbtc_actors_minimum": (
            selection["distinct_wbtc_actors"]
            >= gates["selection_distinct_wbtc_actors_minimum"]
        ),
        "maximum_calendar_gap_days": all(
            split["maximum_calendar_gap_days"] is not None
            and split["maximum_calendar_gap_days"]
            <= gates["maximum_calendar_gap_days"]
            for split in (train, selection)
        ),
        "duplicate_accepted_wbtc_batch_forbidden": all(
            split["duplicate_wbtc_batch_identities"] == 0
            for split in (train, selection)
        ),
        "duplicate_accepted_confirmation_batch_forbidden": all(
            split["duplicate_confirmation_batch_identities"] == 0
            for split in (train, selection)
        ),
    }


def _read_comparator_rows(spec: Mapping[str, Any]) -> list[dict[str, str]]:
    with gzip.open(_path(spec["clock"]), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(spec["header"]):
            raise RuntimeError(f"WSCF comparator header drift: {spec['name']}")
        return list(reader)


def load_comparator_views() -> tuple[dict[str, ComparatorView], dict[str, Any]]:
    views: dict[str, ComparatorView] = {}
    audit: dict[str, Any] = {}
    for spec in prereg.COMPARATOR_SPECS:
        if sha256_file(spec["clock"]) != spec["clock_sha256"]:
            raise RuntimeError(f"WSCF comparator clock hash drift: {spec['name']}")
        rows = _read_comparator_rows(spec)
        audit[spec["name"]] = {"physical_rows_read": len(rows), "views": {}}
        outer_start, outer_end = map(parse_time, spec["comparison"])
        filtered = [
            row
            for row in rows
            if all(row.get(field) == value for field, value in spec["filters"].items())
        ]
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        if spec["name"] == "sealed_prior_stablecoin_bundle":
            for row in filtered:
                grouped[f"{row['candidate']}:{row['control']}"] .append(row)
        elif spec.get("group_field"):
            field = spec["group_field"]
            for row in filtered:
                grouped[row[field]].append(row)
        else:
            grouped[spec["name"]] = filtered
        for suffix, group in sorted(grouped.items()):
            name = (
                spec["name"]
                if suffix == spec["name"]
                else f"{spec['name']}:{suffix}"
            )
            if spec["name"] == "sealed_prior_stablecoin_bundle":
                starts = {parse_time(row["comparison_start"]) for row in group}
                ends = {parse_time(row["comparison_end_exclusive"]) for row in group}
                if len(starts) != 1 or len(ends) != 1:
                    raise RuntimeError("WSCF sealed comparator interval drift")
                start, end = next(iter(starts)), next(iter(ends))
            else:
                start, end = outer_start, outer_end
            entries: list[ComparatorEntry] = []
            seen: set[tuple[datetime, int | None]] = set()
            for row in group:
                entry_time = parse_time(row[spec["entry_field"]])
                if not start <= entry_time < end:
                    raise RuntimeError(f"WSCF comparator row outside interval: {name}")
                side = (
                    int(row[spec["side_field"]])
                    if spec["side_field"] is not None
                    else None
                )
                if side not in {-1, 1, None}:
                    raise RuntimeError(f"WSCF comparator side drift: {name}")
                identity = (entry_time, side)
                if identity in seen:
                    raise RuntimeError(f"WSCF comparator duplicate entry: {name}")
                seen.add(identity)
                entries.append(ComparatorEntry(entry_time=entry_time, side=side))
            view = ComparatorView(
                name=name,
                start=start,
                end=end,
                entries=tuple(sorted(entries, key=lambda row: (row.entry_time, row.side or 0))),
            )
            views[name] = view
            audit[spec["name"]]["views"][name] = len(entries)
    return views, audit


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _near_share(
    source: Sequence[tuple[datetime, int | None]],
    target: Sequence[tuple[datetime, int | None]],
    *,
    hours: int,
    require_same_side: bool,
) -> float:
    if not source:
        return 0.0
    by_side: dict[int | None, list[datetime]] = defaultdict(list)
    for time, side in target:
        by_side[side if require_same_side else None].append(time)
    for times in by_side.values():
        times.sort()
    radius = timedelta(hours=hours)
    matched = 0
    for time, side in source:
        times = by_side.get(side if require_same_side else None, [])
        index = bisect.bisect_left(times, time - radius)
        if index < len(times) and times[index] <= time + radius:
            matched += 1
    return matched / len(source)


def novelty_report(
    primary: Sequence[Candidate], views: Mapping[str, ComparatorView]
) -> tuple[dict[str, Any], dict[str, bool]]:
    config = prereg.policy_payload()["novelty"]
    report: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for name, view in sorted(views.items()):
        wscf = [
            (row.entry_time, row.side)
            for row in primary
            if view.start <= row.entry_time < view.end
        ]
        comparator = [(row.entry_time, row.side) for row in view.entries]
        direction_aware = all(side is not None for _, side in comparator)
        signless_left = {(time, None) for time, _ in wscf}
        signless_right = {(time, None) for time, _ in comparator}
        metrics: dict[str, Any] = {
            "comparison": [format_time(view.start), format_time(view.end)],
            "wscf_entries": len(wscf),
            "comparator_entries": len(comparator),
            "direction_aware": direction_aware,
            "signless": {
                "exact_entry_jaccard": _jaccard(signless_left, signless_right),
                "wscf_to_comparator_near_share": _near_share(
                    wscf,
                    comparator,
                    hours=config["near_window_elapsed_hours"],
                    require_same_side=False,
                ),
                "comparator_to_wscf_near_share": _near_share(
                    comparator,
                    wscf,
                    hours=config["near_window_elapsed_hours"],
                    require_same_side=False,
                ),
            },
        }
        if direction_aware:
            same_left = set(wscf)
            same_right = set(comparator)
            metrics["same_side"] = {
                "exact_entry_jaccard": _jaccard(same_left, same_right),
                "wscf_to_comparator_near_share": _near_share(
                    wscf,
                    comparator,
                    hours=config["near_window_elapsed_hours"],
                    require_same_side=True,
                ),
                "comparator_to_wscf_near_share": _near_share(
                    comparator,
                    wscf,
                    hours=config["near_window_elapsed_hours"],
                    require_same_side=True,
                ),
            }
        eligible = len(comparator) >= config["minimum_comparator_entries"]
        applicable = [metrics["signless"]]
        if direction_aware:
            applicable.append(metrics["same_side"])
        metrics["gate_eligible"] = eligible
        metrics["gate_pass"] = bool(
            eligible
            and all(
                item["exact_entry_jaccard"]
                <= config["maximum_exact_entry_jaccard"]
                and item["wscf_to_comparator_near_share"]
                <= config["maximum_wscf_to_comparator_near_containment"]
                for item in applicable
            )
        )
        report[name] = metrics
        if eligible:
            checks[f"novelty:{name}"] = metrics["gate_pass"]
    return report, checks


def candidate_row(row: Candidate) -> dict[str, Any]:
    wbtc = row.wbtc
    return {
        "candidate": CANDIDATE,
        "control": row.control,
        "signal_id": row.signal_id,
        "window": row.window,
        "wbtc_available_at": format_time(wbtc.available_at) if wbtc else "",
        "anchor_time": format_time(row.anchor_time),
        "signal_time": format_time(row.signal_time),
        "entry_time": format_time(row.entry_time),
        "exit_time": format_time(row.exit_time),
        "side": row.side,
        "wbtc_batch_identity": wbtc.identity_hash if wbtc else "",
        "wbtc_net_raw": wbtc.net_raw if wbtc else "",
        "wbtc_gross_raw": wbtc.gross_raw if wbtc else "",
        "wbtc_rows": wbtc.rows if wbtc else "",
        "wbtc_distinct_actors": len(wbtc.actors) if wbtc else "",
        "wbtc_top_actor_share": f"{wbtc.top_actor_share:.12f}" if wbtc else "",
        "confirmation_batch_identity": row.confirmation_identity,
        "stablecoin_scope": row.stablecoin_scope,
        "cumulative_stablecoin_net_raw": row.cumulative_net_raw,
        "cumulative_stablecoin_gross_raw": row.cumulative_gross_raw,
        "stablecoin_batches": row.stablecoin_batches,
        "confirmation_delay_seconds": int(
            (row.signal_time - row.anchor_time).total_seconds()
        ),
        "usdc_net_raw": row.usdc_net_raw,
        "usdc_gross_raw": row.usdc_gross_raw,
        "usdt_net_raw": row.usdt_net_raw,
        "usdt_gross_raw": row.usdt_gross_raw,
    }


def deterministic_gzip_csv(rows: Iterable[Mapping[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
            writer = csv.DictWriter(text, fieldnames=CLOCK_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row[column] for column in CLOCK_COLUMNS})
    return buffer.getvalue()


def build_support_payload(
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
) -> tuple[dict[str, Any], bytes]:
    registration = validate_preregistration()
    wbtc_events, wbtc_audit = load_wbtc_events()
    stablecoin_events, veto_events, stablecoin_audit = load_stablecoin_events()
    controls = build_controls(wbtc_events, stablecoin_events, veto_events)
    primary = controls["primary"]
    stats = support_statistics(primary)
    structural_checks = support_checks(stats)
    comparator_views, comparator_audit = load_comparator_views()
    novelty, novelty_checks = novelty_report(primary, comparator_views)
    checks = {**structural_checks, **novelty_checks}
    passed = all(checks.values())
    rows = [candidate_row(row) for name in CONTROL_ORDER for row in controls[name]]
    clock_bytes = deterministic_gzip_csv(rows)
    control_report = {
        name: {
            "trades": len(candidates),
            "train": _window_stats(candidates, "train"),
            "selection": _window_stats(candidates, "selection"),
        }
        for name, candidates in controls.items()
    }
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": CANDIDATE,
        "research_status": "source-family-seen_candidate-outcome-blind",
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_artifact": str(PREREGISTRATION_ARTIFACT),
        "preregistration_artifact_sha256": PREREGISTRATION_ARTIFACT_SHA256,
        "preregistration_manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        "preregistration_policy_hash": registration["policy_hash"],
        "implementation": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "source_bindings": registration["source_bindings"],
        "comparator_bindings": registration["comparator_bindings"],
        "source_audit": {
            "wbtc": wbtc_audit,
            "stablecoin": stablecoin_audit,
            "wbtc_atomic_batches": len(
                batch_events(wbtc_events, source="wbtc", scope="wbtc")
            ),
            "stablecoin_atomic_batches": len(
                stablecoin_batches(stablecoin_events, "combined")
            ),
        },
        "comparator_audit": comparator_audit,
        "primary_support": stats,
        "checks": checks,
        "failed_checks": sorted(name for name, value in checks.items() if not value),
        "controls": control_report,
        "novelty": novelty,
        "decision": "PASS_SOURCE" if passed else "REJECT_SOURCE",
        "source_incidence_opened": True,
        "outcomes_opened": False,
        "performance_values_opened": False,
        "outcome_boundary": {
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "pnl_cagr_mdd_opened": False,
            "post_2023_contract_event_value_rows_loaded": 0,
            "sealed_non_timestamp_fields_decoded": 0,
            "network_calls": 0,
            "subprocess_calls": 0,
        },
        "black_funds_control_causal_interpretation": (
            "veto only when confiscation is available after the WBTC anchor and "
            "no later than the would-be confirmation batch; same-time veto wins"
        ),
        "clock_output": {
            "path": str(clock_output),
            "sha256": hashlib.sha256(clock_bytes).hexdigest(),
            "rows": len(rows),
            "columns": list(CLOCK_COLUMNS),
            "gzip_mtime": 0,
        },
        "next_action": (
            "freeze strict evaluator before BTC outcomes"
            if passed
            else "retire candidate without BTC outcomes or repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}, clock_bytes


def _write_once(path: str | Path, payload: bytes) -> str:
    target = _path(path)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file():
            raise FileExistsError("WSCF output is not a regular file")
        if target.read_bytes() != payload:
            raise RuntimeError("refusing to overwrite WSCF output with drift")
        return "verified_existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return "created"


def write_support(
    *,
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
) -> tuple[dict[str, Any], dict[str, str]]:
    if _path(clock_output) == _path(report_output):
        raise ValueError("WSCF clock and report outputs must differ")
    payload, clock_bytes = build_support_payload(clock_output)
    report_bytes = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    status = {
        "clock": _write_once(clock_output, clock_bytes),
        "report": _write_once(report_output, report_bytes),
    }
    return payload, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK_OUTPUT))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, status = write_support(
        clock_output=args.clock_output,
        report_output=args.report_output,
    )
    print(
        json.dumps(
            {
                "candidate": payload["candidate"],
                "decision": payload["decision"],
                "failed_checks": payload["failed_checks"],
                "manifest_hash": payload["manifest_hash"],
                "outcomes_opened": payload["outcomes_opened"],
                "source_incidence_opened": payload["source_incidence_opened"],
                "status": status,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
