"""Build FCCM-72 source-only clocks before comparators or market outcomes."""

from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict, deque
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

from training import (
    preregister_funding_currency_custody_mobility_consensus as prereg,
)


PROTOCOL_VERSION = "funding_currency_custody_mobility_consensus_support_v1"
CANDIDATE = prereg.POLICY_ID
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/build_funding_currency_custody_mobility_consensus_support.py"
)
TEST_PATH = Path(
    "tests/test_build_funding_currency_custody_mobility_consensus_support.py"
)
IMPLEMENTATION_CONTRACT = Path(
    "docs/fccm-source-support-implementation-contract-2026-07-23.md"
)
IMPLEMENTATION_CONTRACT_SHA256 = (
    "242cc6ad6f46f4701e5085a5294322ab7c232ee1473fef9239e2031f0608bc68"
)
PREREGISTRATION_COMMIT = "b49788e6b155021ce0300d64e7f9254ded793e44"
PREREGISTRATION_ARTIFACT = prereg.DEFAULT_OUTPUT
PREREGISTRATION_ARTIFACT_SHA256 = (
    "90c7cb4e110ddb15466702414a0cfbbac9fed681cba0922095817524560ac204"
)
PREREGISTRATION_MANIFEST_HASH = (
    "b33a786aaf9e9c2457e07eaebd0771c9c82971ccdea25676e2a6a9f8bfe2ddf1"
)
PREREGISTRATION_POLICY_HASH = (
    "7e22d0b2559ae6c509ee45e6a8c5bb81a71501a305406c5951c295c4e7376ea3"
)

BITFINEX_SOURCE = prereg.BITFINEX_SOURCE
BITFINEX_SOURCE_SHA256 = prereg.BITFINEX_SOURCE_SHA256
WBTC_SOURCE = prereg.WBTC_SOURCE
WBTC_SOURCE_SHA256 = prereg.WBTC_SOURCE_SHA256
DEFAULT_CLOCK_OUTPUT = Path(
    "data/funding_currency_custody_mobility_consensus_2021_2023/"
    "fccm72_support_clocks_2021_2023.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/funding_currency_custody_mobility_consensus_support_2026-07-23.json"
)

UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
SOURCE_START = datetime(2020, 1, 1, tzinfo=UTC)
TRAIN_START = datetime(2021, 1, 1, tzinfo=UTC)
SELECTION_START = datetime(2023, 1, 1, tzinfo=UTC)
SEALED_FROM = datetime(2024, 1, 1, tzinfo=UTC)
FIVE_MINUTES = timedelta(minutes=5)
HOLD = timedelta(hours=72)
BITFINEX_HISTORY = 720
WBTC_HISTORY = 180
WBTC_WINDOW = timedelta(days=14)
ZERO_ADDRESS = "0x" + "0" * 40
ADDRESS = re.compile(r"^0x[0-9a-f]{40}$")
HASH32 = re.compile(r"^0x[0-9a-f]{64}$")
WBTC_FEATURE_COLUMNS = (
    "amount_raw",
    "actor_address",
    "block_hash",
    "transaction_hash",
    "semantic_log_index",
    "available_at",
)

WINDOWS: Mapping[str, tuple[datetime, datetime]] = {
    "train": (TRAIN_START, SELECTION_START),
    "selection": (SELECTION_START, SEALED_FROM),
}

CONTROL_ORDER = (
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
)

CLOCK_COLUMNS = (
    "candidate",
    "control",
    "row_identity",
    "primary_identity",
    "split",
    "side",
    "bitfinex_hour",
    "directional_source_hour",
    "signal_available_at",
    "entry_time",
    "exit_time",
    "paired_hour_identity",
    "directional_pair_identity",
    "fusd_timestamp_ms",
    "fbtc_timestamp_ms",
    "util_rotation",
    "draw_rotation",
    "tenor_rotation",
    "util_unit",
    "draw_unit",
    "tenor_unit",
    "util_vote",
    "draw_vote",
    "tenor_vote",
    "score",
    "wbtc_anchor",
    "wbtc_window_identity",
    "wbtc_gross_raw",
    "wbtc_gross_unit",
    "wbtc_actor_count",
    "wbtc_top_share",
    "wbtc_active",
)


@dataclass(frozen=True)
class BitfinexRow:
    symbol: str
    observation_time: datetime
    hour: datetime
    available_at: datetime
    timestamp_ms: int
    average_period_days: Fraction
    total: Fraction
    used: Fraction

    @property
    def identity(self) -> str:
        return f"{self.symbol}|{self.timestamp_ms}"

    @property
    def unused(self) -> Fraction:
        return self.total - self.used

    @property
    def utilization(self) -> Fraction:
        return self.used / self.total


@dataclass(frozen=True)
class PairAnchor:
    hour: datetime
    available_at: datetime
    valid: bool
    reason: str
    usd: BitfinexRow | None = None
    btc: BitfinexRow | None = None
    identity: str = ""


@dataclass(frozen=True)
class FeatureAnchor:
    pair: PairAnchor
    available_at: datetime
    valid: bool
    reason: str
    rotations: tuple[Fraction, Fraction, Fraction] | None = None


@dataclass(frozen=True)
class RankedFeature:
    feature: FeatureAnchor
    units: tuple[Fraction, Fraction, Fraction] | None
    votes: tuple[int, int, int] | None
    score: Fraction | None
    consensus_state: int | None
    majority_state: int | None


@dataclass(frozen=True)
class CausalBatch:
    available_at: datetime
    rows: tuple[RankedFeature, ...]
    invalid: bool
    eligible: RankedFeature | None


@dataclass(frozen=True)
class WBTCEvent:
    available_at: datetime
    amount_raw: int
    actor: str
    block_hash: str
    transaction_hash: str
    semantic_log_index: int

    @property
    def identity(self) -> str:
        return (
            f"{self.block_hash}|{self.transaction_hash}|"
            f"{self.semantic_log_index}"
        )


@dataclass(frozen=True)
class WBTCState:
    anchor: datetime
    valid: bool
    gross_raw: int
    gross_unit: Fraction | None
    actors: tuple[str, ...]
    top_share: Fraction | None
    active: bool
    window_identity: str


@dataclass(frozen=True)
class RawTransition:
    directional: RankedFeature
    clock: RankedFeature
    side: int


@dataclass(frozen=True)
class Opportunity:
    transition: RawTransition
    wbtc: WBTCState
    sponsored: bool
    entry_time: datetime
    exit_time: datetime
    entry_split: str | None
    contained_split: str | None


@dataclass(frozen=True)
class CandidateClock:
    control: str
    transition: RawTransition
    wbtc: WBTCState
    side: int
    entry_time: datetime
    exit_time: datetime
    split: str
    row_identity: str
    primary_identity: str = ""


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError("FCCM support path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("FCCM support path must remain in repository") from exc
    return resolved


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
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


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_time(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("FCCM timestamp must include timezone")
    return parsed.astimezone(UTC)


def format_time(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("FCCM timestamp must include timezone")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def floor_hour(value: datetime) -> datetime:
    return value.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def floor_day(value: datetime) -> datetime:
    return value.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def fraction_text(value: Fraction | None) -> str:
    return "" if value is None else str(value)


def parse_fraction(value: str, field: str) -> Fraction:
    text = value.strip()
    if not text:
        raise ValueError(f"FCCM {field} is empty")
    try:
        parsed = Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"FCCM {field} is not an exact rational") from exc
    return parsed


def parse_uint(value: str, field: str, *, positive: bool = False) -> int:
    text = value.strip()
    if not text or not text.isascii() or not text.isdigit():
        raise ValueError(f"FCCM {field} is not a canonical unsigned integer")
    if len(text) > 1 and text.startswith("0"):
        raise ValueError(f"FCCM {field} has leading zeros")
    parsed = int(text)
    if positive and parsed <= 0:
        raise ValueError(f"FCCM {field} must be positive")
    return parsed


def parse_bitfinex_records(records: Iterable[Mapping[str, str]]) -> list[BitfinexRow]:
    rows: list[BitfinexRow] = []
    identities: set[str] = set()
    for record in records:
        symbol = record["symbol"].strip()
        if symbol not in {"fUSD", "fBTC"}:
            raise ValueError("FCCM Bitfinex symbol is outside frozen pair")
        observation = parse_time(record["observation_time"])
        if observation.microsecond:
            raise ValueError("FCCM Bitfinex observation has subsecond precision")
        hour = floor_hour(observation)
        if not SOURCE_START <= hour < SEALED_FROM:
            raise ValueError("FCCM Bitfinex observation is outside source interval")
        available = parse_time(record["available_at"])
        if available.microsecond:
            raise ValueError("FCCM Bitfinex availability has subsecond precision")
        if available < observation:
            raise ValueError("FCCM Bitfinex row is available before observation")
        timestamp_ms = parse_uint(record["timestamp_ms"], "timestamp_ms")
        row = BitfinexRow(
            symbol=symbol,
            observation_time=observation,
            hour=hour,
            available_at=available,
            timestamp_ms=timestamp_ms,
            average_period_days=parse_fraction(
                record["average_period_days"], "average_period_days"
            ),
            total=parse_fraction(record["funding_amount"], "funding_amount"),
            used=parse_fraction(
                record["funding_amount_used"], "funding_amount_used"
            ),
        )
        if row.identity in identities:
            raise RuntimeError("FCCM duplicate Bitfinex source identity")
        identities.add(row.identity)
        rows.append(row)
    return sorted(rows, key=lambda row: (row.hour, row.symbol, row.timestamp_ms))


def parse_wbtc_records(records: Iterable[Mapping[str, str]]) -> list[WBTCEvent]:
    events: list[WBTCEvent] = []
    identities: set[str] = set()
    for record in records:
        available = parse_time(record["available_at"])
        if available.microsecond:
            raise ValueError("FCCM WBTC availability has subsecond precision")
        if not SOURCE_START <= available < SEALED_FROM:
            raise ValueError("FCCM WBTC feature value is outside source interval")
        actor = record["actor_address"].strip().lower()
        if not ADDRESS.fullmatch(actor) or actor == ZERO_ADDRESS:
            raise ValueError("FCCM WBTC actor is malformed or zero")
        block_hash = record["block_hash"].strip().lower()
        transaction_hash = record["transaction_hash"].strip().lower()
        if not HASH32.fullmatch(block_hash) or not HASH32.fullmatch(transaction_hash):
            raise ValueError("FCCM WBTC hash identity is malformed")
        event = WBTCEvent(
            available_at=available,
            amount_raw=parse_uint(record["amount_raw"], "amount_raw", positive=True),
            actor=actor,
            block_hash=block_hash,
            transaction_hash=transaction_hash,
            semantic_log_index=parse_uint(
                record["semantic_log_index"], "semantic_log_index"
            ),
        )
        if event.identity in identities:
            raise RuntimeError("FCCM duplicate WBTC source identity")
        identities.add(event.identity)
        events.append(event)
    return sorted(events, key=lambda event: (event.available_at, event.identity))


def paired_hour_identity(hour: datetime, usd: BitfinexRow, btc: BitfinexRow) -> str:
    return hash_text(
        f"{CANDIDATE}|bitfinex-pair|{format_time(hour)}|"
        f"fUSD|{usd.timestamp_ms}|fBTC|{btc.timestamp_ms}"
    )


def build_pair_anchors(
    rows: Sequence[BitfinexRow],
    *,
    start: datetime = SOURCE_START,
    end: datetime = SEALED_FROM,
) -> list[PairAnchor]:
    grouped: dict[datetime, dict[str, list[BitfinexRow]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if start <= row.hour < end:
            grouped[row.hour][row.symbol].append(row)

    anchors: list[PairAnchor] = []
    hour = start
    while hour < end:
        bucket = grouped.get(hour, {})
        usd_rows = list(bucket.get("fUSD", ()))
        btc_rows = list(bucket.get("fBTC", ()))
        all_rows = usd_rows + btc_rows
        deadline = hour + timedelta(hours=1, minutes=15)
        if not all_rows:
            anchors.append(PairAnchor(hour, deadline, False, "missing_pair"))
        elif len(usd_rows) != 1 or len(btc_rows) != 1:
            if not usd_rows or not btc_rows:
                known = max([deadline, *(row.available_at for row in all_rows)])
                reason = "partial_pair"
            else:
                known = max(row.available_at for row in all_rows)
                reason = "duplicate_pair"
            anchors.append(PairAnchor(hour, known, False, reason))
        else:
            usd, btc = usd_rows[0], btc_rows[0]
            available = max(usd.available_at, btc.available_at)
            valid = all(
                row.total > 0 and 0 <= row.used <= row.total for row in (usd, btc)
            )
            anchors.append(
                PairAnchor(
                    hour=hour,
                    available_at=available,
                    valid=valid,
                    reason="valid" if valid else "invalid_balance",
                    usd=usd,
                    btc=btc,
                    identity=(paired_hour_identity(hour, usd, btc) if valid else ""),
                )
            )
        hour += timedelta(hours=1)
    return anchors


def build_feature_anchors(pairs: Sequence[PairAnchor]) -> list[FeatureAnchor]:
    by_hour = {pair.hour: pair for pair in pairs}
    features: list[FeatureAnchor] = []
    for pair in pairs:
        lag = by_hour.get(pair.hour - timedelta(hours=24))
        if not pair.valid or lag is None or not lag.valid:
            available = max(
                pair.available_at,
                lag.available_at if lag is not None else pair.available_at,
            )
            reason = pair.reason if not pair.valid else "invalid_exact_24h_lag"
            features.append(FeatureAnchor(pair, available, False, reason))
            continue
        assert pair.usd is not None and pair.btc is not None
        assert lag.usd is not None and lag.btc is not None
        util = pair.usd.utilization - pair.btc.utilization
        usd_draw = (lag.usd.unused - pair.usd.unused) / lag.usd.total
        btc_draw = (lag.btc.unused - pair.btc.unused) / lag.btc.total
        draw = usd_draw - btc_draw
        tenor = pair.usd.average_period_days - pair.btc.average_period_days
        features.append(
            FeatureAnchor(
                pair=pair,
                available_at=max(pair.available_at, lag.available_at),
                valid=True,
                reason="valid",
                rotations=(util, draw, tenor),
            )
        )
    return sorted(features, key=lambda row: (row.available_at, row.pair.hour))


def midrank_unit(
    prior: Sequence[Fraction], current: Fraction, required: int
) -> Fraction:
    if len(prior) != required:
        raise ValueError("FCCM exact rank history length mismatch")
    lower = sum(value < current for value in prior)
    equal = sum(value == current for value in prior)
    return Fraction(2 * lower + equal - required, required)


def vote(unit: Fraction) -> int:
    if unit >= Fraction(1, 4):
        return 1
    if unit <= Fraction(-1, 4):
        return -1
    return 0


def consensus_state(
    units: tuple[Fraction, Fraction, Fraction], votes: tuple[int, int, int]
) -> int:
    score = sum(units, Fraction()) / 3
    if votes.count(1) >= 2 and score >= Fraction(1, 3):
        return 1
    if votes.count(-1) >= 2 and score <= Fraction(-1, 3):
        return -1
    return 0


class _RankHistory:
    def __init__(self, size: int) -> None:
        self.size = size
        self.ordered: deque[tuple[Fraction, Fraction, Fraction]] = deque()
        self.sorted: list[list[Fraction]] = [[], [], []]

    def ready(self) -> bool:
        return len(self.ordered) == self.size

    def units(
        self, values: tuple[Fraction, Fraction, Fraction]
    ) -> tuple[Fraction, Fraction, Fraction]:
        if not self.ready():
            raise ValueError("FCCM rank history is not ready")
        output: list[Fraction] = []
        for index, current in enumerate(values):
            ordered = self.sorted[index]
            lower = bisect.bisect_left(ordered, current)
            upper = bisect.bisect_right(ordered, current)
            output.append(Fraction(2 * lower + upper - lower - self.size, self.size))
        return output[0], output[1], output[2]

    def append(self, values: tuple[Fraction, Fraction, Fraction]) -> None:
        if len(self.ordered) == self.size:
            removed = self.ordered.popleft()
            for index, value in enumerate(removed):
                position = bisect.bisect_left(self.sorted[index], value)
                if position == len(self.sorted[index]) or self.sorted[index][position] != value:
                    raise RuntimeError("FCCM rank history removal drift")
                self.sorted[index].pop(position)
        self.ordered.append(values)
        for index, value in enumerate(values):
            bisect.insort(self.sorted[index], value)


def rank_causal_batches(
    features: Sequence[FeatureAnchor], *, history_size: int = BITFINEX_HISTORY
) -> list[CausalBatch]:
    grouped: dict[datetime, list[FeatureAnchor]] = defaultdict(list)
    for feature in features:
        grouped[feature.available_at].append(feature)
    history = _RankHistory(history_size)
    batches: list[CausalBatch] = []
    for available in sorted(grouped):
        feature_rows = sorted(grouped[available], key=lambda row: row.pair.hour)
        ranked: list[RankedFeature] = []
        for feature in feature_rows:
            if not feature.valid or feature.rotations is None or not history.ready():
                ranked.append(RankedFeature(feature, None, None, None, None, None))
                continue
            units = history.units(feature.rotations)
            votes = tuple(vote(unit) for unit in units)
            score = sum(units, Fraction()) / 3
            majority = 1 if votes.count(1) >= 2 else -1 if votes.count(-1) >= 2 else 0
            ranked.append(
                RankedFeature(
                    feature=feature,
                    units=units,
                    votes=votes,
                    score=score,
                    consensus_state=consensus_state(units, votes),
                    majority_state=majority,
                )
            )
        invalid = any(not feature.valid for feature in feature_rows)
        eligible = None if invalid else max(ranked, key=lambda row: row.feature.pair.hour)
        batches.append(
            CausalBatch(
                available_at=available,
                rows=tuple(ranked),
                invalid=invalid,
                eligible=eligible,
            )
        )
        for feature in feature_rows:
            if feature.valid and feature.rotations is not None:
                history.append(feature.rotations)
    return batches


def wbtc_window_identity(anchor: datetime, events: Sequence[WBTCEvent]) -> str:
    identities = sorted(event.identity for event in events)
    payload = f"{CANDIDATE}|wbtc-window|{format_time(anchor)}\n" + "\n".join(
        identities
    )
    return hash_text(payload)


def build_wbtc_states(
    events: Sequence[WBTCEvent],
    *,
    coverage_start: datetime = SOURCE_START,
    end: datetime = SEALED_FROM,
    history_size: int = WBTC_HISTORY,
) -> dict[datetime, WBTCState]:
    ordered_events = sorted(events, key=lambda event: (event.available_at, event.identity))
    times = [event.available_at for event in ordered_events]
    raw: dict[datetime, tuple[int, tuple[str, ...], Fraction | None, str, bool]] = {}
    anchor = coverage_start
    while anchor < end:
        lower = anchor - WBTC_WINDOW
        left = bisect.bisect_right(times, lower)
        right = bisect.bisect_right(times, anchor)
        window = ordered_events[left:right]
        gross = sum(event.amount_raw for event in window)
        actor_gross: Counter[str] = Counter()
        for event in window:
            actor_gross[event.actor] += event.amount_raw
        actors = tuple(sorted(actor_gross))
        top = (
            Fraction(max(actor_gross.values()), gross)
            if gross > 0 and len(actor_gross) >= 2
            else None
        )
        complete = lower >= coverage_start
        raw[anchor] = (
            gross,
            actors,
            top,
            wbtc_window_identity(anchor, window),
            complete,
        )
        anchor += timedelta(days=1)

    states: dict[datetime, WBTCState] = {}
    for anchor in sorted(raw):
        gross, actors, top, identity, complete = raw[anchor]
        prior_anchors = [
            anchor - timedelta(days=offset)
            for offset in range(history_size, 0, -1)
        ]
        prior_complete = all(
            prior in raw and raw[prior][4] for prior in prior_anchors
        )
        if not complete or not prior_complete:
            states[anchor] = WBTCState(
                anchor, False, gross, None, actors, top, False, identity
            )
            continue
        prior_gross = [raw[prior][0] for prior in prior_anchors]
        unit = midrank_unit(
            [Fraction(value) for value in prior_gross], Fraction(gross), history_size
        )
        active = bool(
            gross > 0
            and len(actors) >= 2
            and top is not None
            and unit >= Fraction(1, 5)
            and top <= Fraction(4, 5)
        )
        states[anchor] = WBTCState(
            anchor, True, gross, unit, actors, top, active, identity
        )
    return states


def _state_for(row: RankedFeature, rule: str) -> int | None:
    if row.units is None or row.votes is None:
        return None
    if rule == "consensus":
        return row.consensus_state
    if rule == "majority":
        return row.majority_state
    component = {"utilization": 0, "draw": 1, "tenor": 2}.get(rule)
    if component is None:
        raise ValueError(f"unknown FCCM state rule: {rule}")
    return row.votes[component]


def derive_transitions(
    batches: Sequence[CausalBatch], *, rule: str = "consensus"
) -> list[RawTransition]:
    established: int | None = None
    transitions: list[RawTransition] = []
    for batch in sorted(batches, key=lambda row: row.available_at):
        if batch.invalid:
            established = None
            continue
        eligible = batch.eligible
        if eligible is None:
            continue
        state = _state_for(eligible, rule)
        if state is None:
            continue
        if established is None:
            established = state
            continue
        if state in {-1, 1} and state != established:
            transitions.append(
                RawTransition(directional=eligible, clock=eligible, side=state)
            )
        established = state
    return transitions


def shift_bitfinex_transitions_24h(
    transitions: Sequence[RawTransition], batches: Sequence[CausalBatch]
) -> list[RawTransition]:
    current_by_hour = {
        batch.eligible.feature.pair.hour: batch.eligible
        for batch in batches
        if not batch.invalid and batch.eligible is not None
    }
    shifted: list[RawTransition] = []
    for transition in transitions:
        source_hour = transition.directional.feature.pair.hour
        current = current_by_hour.get(source_hour + timedelta(hours=24))
        if current is None or current.units is None:
            continue
        shifted.append(
            RawTransition(
                directional=transition.directional,
                clock=current,
                side=transition.side,
            )
        )
    return shifted


def execution_entry(signal_available_at: datetime) -> datetime:
    value = signal_available_at.astimezone(UTC)
    delta = value - EPOCH
    epoch_seconds = delta.days * 86_400 + delta.seconds
    if delta.microseconds:
        epoch_seconds += 1
    grid = 5 * 60
    ceiling = ((epoch_seconds + grid - 1) // grid) * grid
    return datetime.fromtimestamp(ceiling, tz=UTC) + FIVE_MINUTES


def entry_split(entry: datetime) -> str | None:
    for name, (start, end) in WINDOWS.items():
        if start <= entry < end:
            return name
    return None


def contained_split(entry: datetime, exit_time: datetime) -> str | None:
    for name, (start, end) in WINDOWS.items():
        if start <= entry and exit_time <= end:
            return name
    return None


def lookup_wbtc_state(
    states: Mapping[datetime, WBTCState],
    signal_available_at: datetime,
    *,
    stale_days: int = 0,
) -> WBTCState:
    anchor = floor_day(signal_available_at) - timedelta(days=stale_days)
    state = states.get(anchor)
    if state is None:
        return WBTCState(
            anchor=anchor,
            valid=False,
            gross_raw=0,
            gross_unit=None,
            actors=(),
            top_share=None,
            active=False,
            window_identity=wbtc_window_identity(anchor, ()),
        )
    if stale_days == 0 and not timedelta(0) <= signal_available_at - anchor < timedelta(
        days=1
    ):
        raise RuntimeError("FCCM latest WBTC state is not less than 24h old")
    return state


def materialize_opportunities(
    transitions: Sequence[RawTransition],
    states: Mapping[datetime, WBTCState],
    *,
    stale_wbtc_days: int = 0,
) -> list[Opportunity]:
    output: list[Opportunity] = []
    for transition in transitions:
        signal = transition.clock.feature.available_at
        wbtc = lookup_wbtc_state(states, signal, stale_days=stale_wbtc_days)
        entry = execution_entry(signal)
        exit_time = entry + HOLD
        output.append(
            Opportunity(
                transition=transition,
                wbtc=wbtc,
                sponsored=wbtc.valid and wbtc.active,
                entry_time=entry,
                exit_time=exit_time,
                entry_split=entry_split(entry),
                contained_split=contained_split(entry, exit_time),
            )
        )
    return output


def primary_identity(opportunity: Opportunity) -> str:
    pair_identity = opportunity.transition.clock.feature.pair.identity
    if not pair_identity:
        raise RuntimeError("FCCM primary candidate lacks paired-hour identity")
    return hash_text(
        f"{CANDIDATE}|candidate|{pair_identity}|"
        f"{opportunity.wbtc.window_identity}|{opportunity.transition.side}|"
        f"{format_time(opportunity.entry_time)}|{format_time(opportunity.exit_time)}"
    )


def control_identity(
    control: str,
    opportunity: Opportunity,
    side: int,
    entry: datetime,
    exit_time: datetime,
) -> str:
    pair_identity = opportunity.transition.clock.feature.pair.identity
    if not pair_identity:
        raise RuntimeError("FCCM control candidate lacks paired-hour identity")
    return hash_text(
        f"{CANDIDATE}|control-row|{control}|{pair_identity}|"
        f"{opportunity.wbtc.window_identity}|{side}|{format_time(entry)}|"
        f"{format_time(exit_time)}"
    )


def schedule_opportunities(
    control: str,
    opportunities: Sequence[Opportunity],
    *,
    require_sponsorship: bool,
) -> list[CandidateClock]:
    accepted: list[CandidateClock] = []
    prior_exit: datetime | None = None
    seen_rows: set[str] = set()
    for opportunity in sorted(
        opportunities,
        key=lambda row: (
            row.entry_time,
            row.transition.clock.feature.pair.identity,
            row.transition.side,
        ),
    ):
        if require_sponsorship and not opportunity.sponsored:
            continue
        if opportunity.contained_split is None:
            continue
        if prior_exit is not None and opportunity.entry_time < prior_exit:
            continue
        if control == "primary":
            identity = primary_identity(opportunity)
            reference = identity
        else:
            identity = control_identity(
                control,
                opportunity,
                opportunity.transition.side,
                opportunity.entry_time,
                opportunity.exit_time,
            )
            reference = ""
        if identity in seen_rows:
            raise RuntimeError(f"FCCM duplicate row identity in control {control}")
        seen_rows.add(identity)
        accepted.append(
            CandidateClock(
                control=control,
                transition=opportunity.transition,
                wbtc=opportunity.wbtc,
                side=opportunity.transition.side,
                entry_time=opportunity.entry_time,
                exit_time=opportunity.exit_time,
                split=opportunity.contained_split,
                row_identity=identity,
                primary_identity=reference,
            )
        )
        prior_exit = opportunity.exit_time
    return accepted


def _transformed_primary_control(
    control: str,
    primary: Sequence[CandidateClock],
    *,
    side_for: Callable[[CandidateClock], int],
    delay: timedelta = timedelta(0),
) -> list[CandidateClock]:
    output: list[CandidateClock] = []
    for row in primary:
        entry = row.entry_time + delay
        exit_time = row.exit_time + delay
        if contained_split(entry, exit_time) != row.split:
            continue
        side = side_for(row)
        opportunity = Opportunity(
            transition=row.transition,
            wbtc=row.wbtc,
            sponsored=True,
            entry_time=entry,
            exit_time=exit_time,
            entry_split=row.split,
            contained_split=row.split,
        )
        identity = control_identity(control, opportunity, side, entry, exit_time)
        output.append(
            CandidateClock(
                control=control,
                transition=row.transition,
                wbtc=row.wbtc,
                side=side,
                entry_time=entry,
                exit_time=exit_time,
                split=row.split,
                row_identity=identity,
                primary_identity=row.primary_identity,
            )
        )
    return output


def deterministic_random_side(entry: datetime) -> int:
    digest = hashlib.sha256(
        f"{CANDIDATE}|random-side|{format_time(entry)}".encode("utf-8")
    ).digest()
    return 1 if digest[0] < 128 else -1


def build_controls(
    batches: Sequence[CausalBatch], states: Mapping[datetime, WBTCState]
) -> tuple[dict[str, list[CandidateClock]], list[Opportunity]]:
    consensus = derive_transitions(batches, rule="consensus")
    primary_opportunities = materialize_opportunities(consensus, states)
    controls: dict[str, list[CandidateClock]] = {}
    controls["primary"] = schedule_opportunities(
        "primary", primary_opportunities, require_sponsorship=True
    )
    controls["bitfinex_consensus_only"] = schedule_opportunities(
        "bitfinex_consensus_only",
        primary_opportunities,
        require_sponsorship=False,
    )
    for control, rule in (
        ("utilization_only", "utilization"),
        ("draw_only", "draw"),
        ("tenor_only", "tenor"),
        ("majority_without_score", "majority"),
    ):
        transitions = derive_transitions(batches, rule=rule)
        controls[control] = schedule_opportunities(
            control,
            materialize_opportunities(transitions, states),
            require_sponsorship=True,
        )
    controls["wbtc_stale_7d"] = schedule_opportunities(
        "wbtc_stale_7d",
        materialize_opportunities(consensus, states, stale_wbtc_days=7),
        require_sponsorship=True,
    )
    stale_bitfinex = shift_bitfinex_transitions_24h(consensus, batches)
    controls["bitfinex_stale_24h"] = schedule_opportunities(
        "bitfinex_stale_24h",
        materialize_opportunities(stale_bitfinex, states),
        require_sponsorship=True,
    )
    primary = controls["primary"]
    controls["exact_direction_flip"] = _transformed_primary_control(
        "exact_direction_flip", primary, side_for=lambda row: -row.side
    )
    controls["deterministic_random_side"] = _transformed_primary_control(
        "deterministic_random_side",
        primary,
        side_for=lambda row: deterministic_random_side(row.entry_time),
    )
    controls["one_bar_delay"] = _transformed_primary_control(
        "one_bar_delay",
        primary,
        side_for=lambda row: row.side,
        delay=FIVE_MINUTES,
    )
    if tuple(controls) != CONTROL_ORDER:
        raise RuntimeError("FCCM control order drift")
    return controls, primary_opportunities


def maximum_same_side_run(rows: Sequence[CandidateClock]) -> int:
    maximum = 0
    current = 0
    previous: int | None = None
    for row in sorted(rows, key=lambda candidate: candidate.entry_time):
        current = current + 1 if row.side == previous else 1
        maximum = max(maximum, current)
        previous = row.side
    return maximum


def split_statistics(rows: Sequence[CandidateClock], split: str) -> dict[str, Any]:
    selected = sorted(
        (row for row in rows if row.split == split), key=lambda row: row.entry_time
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
    sides = Counter("long" if row.side == 1 else "short" for row in selected)
    actors = sorted({actor for row in selected for actor in row.wbtc.actors})
    month_share = (
        Fraction(max(months.values()), len(selected)) if selected else Fraction()
    )
    gaps: list[Fraction] = []
    for left, right in zip(selected, selected[1:]):
        delta = right.entry_time - left.entry_time
        if delta.microseconds:
            raise RuntimeError("FCCM accepted entry gap has subsecond precision")
        gaps.append(Fraction(delta.days * 86_400 + delta.seconds, 86_400))
    component_support: dict[str, str] = {}
    for name, index in (("utilization", 0), ("draw", 1), ("tenor", 2)):
        agreeing = sum(
            row.transition.directional.votes is not None
            and row.transition.directional.votes[index] == row.side
            for row in selected
        )
        component_support[name] = str(
            Fraction(agreeing, len(selected)) if selected else Fraction()
        )
    vote_patterns = sorted(
        {
            ",".join(str(value) for value in row.transition.directional.votes)
            for row in selected
            if row.transition.directional.votes is not None
        }
    )
    return {
        "accepted_entries": len(selected),
        "year_counts": dict(sorted(years.items())),
        "half_year_counts": dict(sorted(halves.items())),
        "quarter_counts": dict(sorted(quarters.items())),
        "month_counts": dict(sorted(months.items())),
        "side_counts": {side: sides.get(side, 0) for side in ("long", "short")},
        "maximum_month_share": str(month_share),
        "maximum_entry_gap_days": str(max(gaps, default=Fraction())),
        "maximum_consecutive_same_side": maximum_same_side_run(selected),
        "distinct_wbtc_actors": len(actors),
        "wbtc_actor_set_hash": hash_text("\n".join(actors)),
        "component_vote_with_side_share": component_support,
        "distinct_vote_patterns": vote_patterns,
    }


def raw_sponsorship_statistics(
    opportunities: Sequence[Opportunity], split: str
) -> dict[str, Any]:
    selected = [row for row in opportunities if row.entry_split == split]
    sponsored = sum(row.sponsored for row in selected)
    return {
        "raw_directional_transitions_before_split_and_nonoverlap": len(selected),
        "wbtc_active": sponsored,
        "wbtc_active_share": str(
            Fraction(sponsored, len(selected)) if selected else Fraction()
        ),
        "split_crossing_transitions": sum(
            row.contained_split is None for row in selected
        ),
    }


def primary_uniqueness_checks(rows: Sequence[CandidateClock]) -> dict[str, bool]:
    ordered = sorted(rows, key=lambda row: row.entry_time)
    fields: Mapping[str, list[Any]] = {
        "bitfinex_anchor": [row.transition.clock.feature.pair.identity for row in rows],
        "wbtc_anchor": [row.wbtc.anchor for row in rows],
        "candidate_identity": [row.row_identity for row in rows],
        "entry": [row.entry_time for row in rows],
        "occupied_interval": [(row.entry_time, row.exit_time) for row in rows],
    }
    checks = {
        f"unique_{name}": len(values) == len(set(values))
        for name, values in fields.items()
    }
    checks["chronological_nonoverlap"] = all(
        left.exit_time <= right.entry_time for left, right in zip(ordered, ordered[1:])
    )
    return checks


def support_report(
    primary: Sequence[CandidateClock], raw: Sequence[Opportunity]
) -> tuple[dict[str, Any], dict[str, bool]]:
    train = split_statistics(primary, "train")
    selection = split_statistics(primary, "selection")
    raw_train = raw_sponsorship_statistics(raw, "train")
    raw_selection = raw_sponsorship_statistics(raw, "selection")
    uniqueness = primary_uniqueness_checks(primary)
    stats = {
        "train": train,
        "selection": selection,
        "raw_sponsorship": {"train": raw_train, "selection": raw_selection},
        "uniqueness": uniqueness,
    }
    train_month = Fraction(train["maximum_month_share"])
    selection_month = Fraction(selection["maximum_month_share"])
    train_gap = Fraction(train["maximum_entry_gap_days"])
    selection_gap = Fraction(selection["maximum_entry_gap_days"])
    train_active = Fraction(raw_train["wbtc_active_share"])
    selection_active = Fraction(raw_selection["wbtc_active_share"])
    checks = {
        "train_total_minimum": train["accepted_entries"] >= 60,
        "each_train_year_minimum": all(
            train["year_counts"].get(str(year), 0) >= 24 for year in (2021, 2022)
        ),
        "each_train_half_minimum": all(
            train["half_year_counts"].get(f"{year}H{half}", 0) >= 10
            for year in (2021, 2022)
            for half in (1, 2)
        ),
        "train_each_side_minimum": all(
            train["side_counts"].get(side, 0) >= 15 for side in ("long", "short")
        ),
        "train_every_quarter_active": all(
            train["quarter_counts"].get(f"{year}Q{quarter}", 0) > 0
            for year in (2021, 2022)
            for quarter in range(1, 5)
        ),
        "train_maximum_month_share": train_month <= Fraction(3, 20),
        "train_maximum_entry_gap": train_gap <= 45,
        "selection_total_minimum": selection["accepted_entries"] >= 24,
        "each_selection_half_minimum": all(
            selection["half_year_counts"].get(f"2023H{half}", 0) >= 10
            for half in (1, 2)
        ),
        "selection_each_side_minimum": all(
            selection["side_counts"].get(side, 0) >= 6
            for side in ("long", "short")
        ),
        "selection_every_quarter_active": all(
            selection["quarter_counts"].get(f"2023Q{quarter}", 0) > 0
            for quarter in range(1, 5)
        ),
        "selection_maximum_month_share": selection_month <= Fraction(1, 5),
        "selection_maximum_entry_gap": selection_gap <= 60,
        "maximum_consecutive_same_side": all(
            split["maximum_consecutive_same_side"] <= 8
            for split in (train, selection)
        ),
        "train_distinct_wbtc_actors": train["distinct_wbtc_actors"] >= 10,
        "selection_distinct_wbtc_actors": selection["distinct_wbtc_actors"] >= 5,
        "train_wbtc_raw_transition_active_share": (
            raw_train["raw_directional_transitions_before_split_and_nonoverlap"] > 0
            and Fraction(1, 5) <= train_active <= Fraction(7, 10)
        ),
        "selection_wbtc_raw_transition_active_share": (
            raw_selection["raw_directional_transitions_before_split_and_nonoverlap"]
            > 0
            and Fraction(1, 5) <= selection_active <= Fraction(7, 10)
        ),
        "train_each_component_vote_share": all(
            Fraction(value) >= Fraction(7, 20)
            for value in train["component_vote_with_side_share"].values()
        ),
        "selection_each_component_vote_share": all(
            Fraction(value) >= Fraction(7, 20)
            for value in selection["component_vote_with_side_share"].values()
        ),
        "train_distinct_vote_patterns": len(train["distinct_vote_patterns"]) >= 3,
        "selection_distinct_vote_patterns": (
            len(selection["distinct_vote_patterns"]) >= 2
        ),
        **uniqueness,
    }
    return stats, checks


def permute_wbtc_field(
    events: Sequence[WBTCEvent], *, field: str
) -> list[WBTCEvent]:
    attribute = {"amount_raw": "amount_raw", "actor_address": "actor"}.get(field)
    if attribute is None:
        raise ValueError("FCCM placebo field is not frozen")
    by_year: dict[int, list[WBTCEvent]] = defaultdict(list)
    for event in events:
        by_year[event.available_at.year].append(event)
    output: list[WBTCEvent] = []
    for year in sorted(by_year):
        destination = sorted(by_year[year], key=lambda event: event.identity)
        source = sorted(
            by_year[year],
            key=lambda event: (
                hash_text(
                    f"{CANDIDATE}|placebo|{field}|{year}|{event.identity}"
                ),
                event.identity,
            ),
        )
        values = [getattr(event, attribute) for event in source]
        output.extend(
            replace(event, **{attribute: value})
            for event, value in zip(destination, values)
        )
    return sorted(output, key=lambda event: (event.available_at, event.identity))


def placebo_incidence_report(
    batches: Sequence[CausalBatch], events: Sequence[WBTCEvent]
) -> dict[str, Any]:
    consensus = derive_transitions(batches, rule="consensus")
    output: dict[str, Any] = {}
    for name, field in (
        ("within_year_wbtc_amount_hash_permutation", "amount_raw"),
        ("within_year_wbtc_actor_hash_permutation", "actor_address"),
    ):
        permuted = permute_wbtc_field(events, field=field)
        attribute = "actor" if field == "actor_address" else field
        source_multisets = {
            year: sorted(
                str(getattr(event, attribute))
                for event in events
                if event.available_at.year == year
            )
            for year in sorted({event.available_at.year for event in events})
        }
        permuted_multisets = {
            year: sorted(
                str(getattr(event, attribute))
                for event in permuted
                if event.available_at.year == year
            )
            for year in sorted({event.available_at.year for event in permuted})
        }
        states = build_wbtc_states(permuted)
        opportunities = materialize_opportunities(consensus, states)
        output[name] = {
            "train": raw_sponsorship_statistics(opportunities, "train"),
            "selection": raw_sponsorship_statistics(opportunities, "selection"),
            "execution_clock_rows_emitted": 0,
            "scheduler_invoked": False,
            "economics_opened": False,
            "within_year_multiset_preserved": source_multisets
            == permuted_multisets,
            "within_year_multiset_hash": canonical_hash(permuted_multisets),
        }
    return output


def candidate_row(row: CandidateClock) -> dict[str, Any]:
    clock_pair = row.transition.clock.feature.pair
    directional = row.transition.directional
    directional_pair = directional.feature.pair
    if clock_pair.usd is None or clock_pair.btc is None:
        raise RuntimeError("FCCM candidate clock lacks exact Bitfinex pair")
    rotations = directional.feature.rotations
    if rotations is None or directional.units is None or directional.votes is None:
        raise RuntimeError("FCCM candidate clock lacks directional features")
    return {
        "candidate": CANDIDATE,
        "control": row.control,
        "row_identity": row.row_identity,
        "primary_identity": row.primary_identity,
        "split": row.split,
        "side": str(row.side),
        "bitfinex_hour": format_time(clock_pair.hour),
        "directional_source_hour": format_time(directional_pair.hour),
        "signal_available_at": format_time(row.transition.clock.feature.available_at),
        "entry_time": format_time(row.entry_time),
        "exit_time": format_time(row.exit_time),
        "paired_hour_identity": clock_pair.identity,
        "directional_pair_identity": directional_pair.identity,
        "fusd_timestamp_ms": str(clock_pair.usd.timestamp_ms),
        "fbtc_timestamp_ms": str(clock_pair.btc.timestamp_ms),
        "util_rotation": str(rotations[0]),
        "draw_rotation": str(rotations[1]),
        "tenor_rotation": str(rotations[2]),
        "util_unit": str(directional.units[0]),
        "draw_unit": str(directional.units[1]),
        "tenor_unit": str(directional.units[2]),
        "util_vote": str(directional.votes[0]),
        "draw_vote": str(directional.votes[1]),
        "tenor_vote": str(directional.votes[2]),
        "score": fraction_text(directional.score),
        "wbtc_anchor": format_time(row.wbtc.anchor),
        "wbtc_window_identity": row.wbtc.window_identity,
        "wbtc_gross_raw": str(row.wbtc.gross_raw),
        "wbtc_gross_unit": fraction_text(row.wbtc.gross_unit),
        "wbtc_actor_count": str(len(row.wbtc.actors)),
        "wbtc_top_share": fraction_text(row.wbtc.top_share),
        "wbtc_active": "1" if row.wbtc.active else "0",
    }


def deterministic_gzip_csv(rows: Iterable[Mapping[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
            writer = csv.DictWriter(
                text,
                fieldnames=CLOCK_COLUMNS,
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row[column] for column in CLOCK_COLUMNS})
    return buffer.getvalue()


def _validate_preregistration_artifact() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION_ARTIFACT) != PREREGISTRATION_ARTIFACT_SHA256:
        raise RuntimeError("FCCM preregistration artifact hash drift")
    payload = json.loads(
        _repository_path(PREREGISTRATION_ARTIFACT).read_text(encoding="utf-8")
    )
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("FCCM preregistration manifest drift")
    if canonical_hash(core) != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("FCCM preregistration canonical hash drift")
    if payload.get("policy_hash") != PREREGISTRATION_POLICY_HASH:
        raise RuntimeError("FCCM preregistration policy drift")
    if payload.get("candidate") != CANDIDATE or payload.get("artifact_eligible") is not True:
        raise RuntimeError("FCCM preregistration is not eligible")
    for field in (
        "fccm_source_values_or_incidence_opened",
        "comparator_rows_opened_during_preregistration",
        "outcomes_opened",
        "performance_values_opened",
    ):
        if payload.get(field) is not False:
            raise RuntimeError(f"FCCM preregistration boundary opened: {field}")
    return payload


def _screened_gzip_records(
    path: Path,
    *,
    expected_hash: str,
    expected_header: Sequence[str],
    allowed_columns: Sequence[str],
    sentinel_column: str,
    causal_available_column: str | None = None,
    expected_physical_rows: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if sha256_file(path) != expected_hash:
        raise RuntimeError(f"FCCM source hash drift: {path}")
    records: list[dict[str, str]] = []
    physical_rows = 0
    post_seal_timestamp_sentinels = 0
    with gzip.open(_repository_path(path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise RuntimeError("FCCM source is empty") from exc
        if header != tuple(expected_header):
            raise RuntimeError(f"FCCM source header drift: {path}")
        indices = {name: header.index(name) for name in allowed_columns}
        sentinel_index = header.index(sentinel_column)
        causal_index = (
            header.index(causal_available_column)
            if causal_available_column is not None
            else None
        )
        for raw in reader:
            physical_rows += 1
            if len(raw) != len(header):
                raise RuntimeError("FCCM source row width drift")
            sentinel = parse_time(raw[sentinel_index])
            compare = floor_hour(sentinel) if sentinel_column == "observation_time" else sentinel
            causal_available = (
                parse_time(raw[causal_index]) if causal_index is not None else compare
            )
            if compare >= SEALED_FROM or causal_available >= SEALED_FROM:
                post_seal_timestamp_sentinels += 1
                continue
            records.append({name: raw[index] for name, index in indices.items()})
    if physical_rows != expected_physical_rows:
        raise RuntimeError(f"FCCM source physical row count drift: {path}")
    return records, {
        "path": str(path),
        "sha256": expected_hash,
        "physical_rows": physical_rows,
        "pre_seal_value_rows_loaded": len(records),
        "post_seal_timestamp_sentinels_scanned": post_seal_timestamp_sentinels,
        "post_seal_value_rows_loaded": 0,
        "allowed_columns_loaded": list(allowed_columns),
        "forbidden_columns_loaded": [],
    }


def _load_sources() -> tuple[list[BitfinexRow], list[WBTCEvent], dict[str, Any]]:
    bitfinex_records, bitfinex_audit = _screened_gzip_records(
        BITFINEX_SOURCE,
        expected_hash=BITFINEX_SOURCE_SHA256,
        expected_header=prereg.BITFINEX_HEADER,
        allowed_columns=prereg.BITFINEX_ALLOWED_COLUMNS,
        sentinel_column="observation_time",
        causal_available_column="available_at",
        expected_physical_rows=70_116,
    )
    wbtc_records, wbtc_audit = _screened_gzip_records(
        WBTC_SOURCE,
        expected_hash=WBTC_SOURCE_SHA256,
        expected_header=prereg.WBTC_HEADER,
        allowed_columns=WBTC_FEATURE_COLUMNS,
        sentinel_column="available_at",
        expected_physical_rows=993,
    )
    bitfinex = parse_bitfinex_records(bitfinex_records)
    wbtc = parse_wbtc_records(wbtc_records)
    return bitfinex, wbtc, {"bitfinex": bitfinex_audit, "wbtc": wbtc_audit}


def build_support_payload(
    bitfinex_rows: Sequence[BitfinexRow],
    wbtc_events: Sequence[WBTCEvent],
    source_audit: Mapping[str, Any],
    *,
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
) -> tuple[dict[str, Any], bytes]:
    pairs = build_pair_anchors(bitfinex_rows)
    features = build_feature_anchors(pairs)
    batches = rank_causal_batches(features)
    states = build_wbtc_states(wbtc_events)
    controls, raw_primary = build_controls(batches, states)
    primary = controls["primary"]
    stats, checks = support_report(primary, raw_primary)
    passed = all(checks.values())
    all_clocks = [row for control in CONTROL_ORDER for row in controls[control]]
    rows = [
        candidate_row(row)
        for row in sorted(
            all_clocks,
            key=lambda candidate: (
                candidate.entry_time,
                candidate.row_identity,
                candidate.control,
            ),
        )
    ]
    clock_bytes = deterministic_gzip_csv(rows)
    control_stats = {
        control: {
            "clock_rows": len(controls[control]),
            "train": split_statistics(controls[control], "train"),
            "selection": split_statistics(controls[control], "selection"),
        }
        for control in CONTROL_ORDER
    }
    pair_reasons = Counter(pair.reason for pair in pairs)
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": CANDIDATE,
        "research_status": "source_seen_candidate_incidence_seen_outcome_blind",
        "verification_mode": "uncommitted_or_injected_sources",
        "artifact_eligible": False,
        "preregistration": {
            "commit": PREREGISTRATION_COMMIT,
            "path": str(PREREGISTRATION_ARTIFACT),
            "sha256": PREREGISTRATION_ARTIFACT_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
            "policy_hash": PREREGISTRATION_POLICY_HASH,
        },
        "implementation_contract": {
            "path": str(IMPLEMENTATION_CONTRACT),
            "sha256": IMPLEMENTATION_CONTRACT_SHA256,
        },
        "evaluator_source": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "source_audit": dict(source_audit),
        "feature_audit": {
            "pair_hours": len(pairs),
            "pair_reason_counts": dict(sorted(pair_reasons.items())),
            "valid_feature_hours": sum(feature.valid for feature in features),
            "causal_batches": len(batches),
            "invalid_causal_batches": sum(batch.invalid for batch in batches),
            "raw_consensus_directional_transitions": len(raw_primary),
            "wbtc_daily_states": len(states),
            "wbtc_valid_daily_states": sum(state.valid for state in states.values()),
            "wbtc_active_daily_states": sum(state.active for state in states.values()),
            "fraction_arithmetic_only": True,
        },
        "clock": {
            "path": str(clock_output),
            "sha256": hashlib.sha256(clock_bytes).hexdigest(),
            "rows": len(rows),
            "columns": list(CLOCK_COLUMNS),
            "control_counts": {
                control: len(controls[control]) for control in CONTROL_ORDER
            },
        },
        "primary_support": stats,
        "support_checks": checks,
        "source_support_passed": passed,
        "control_source_statistics": control_stats,
        "noncausal_placebo_incidence": placebo_incidence_report(batches, wbtc_events),
        "decision": "ineligible_uncommitted_or_injected_source_support_payload",
        "advance_to_novelty_evaluator": False,
        "outcome_boundary": {
            "bitfinex_source_value_rows_read": source_audit["bitfinex"][
                "pre_seal_value_rows_loaded"
            ],
            "wbtc_source_value_rows_read": source_audit["wbtc"][
                "pre_seal_value_rows_loaded"
            ],
            "post_2023_source_value_rows_read": 0,
            "comparator_value_rows_read": 0,
            "btc_market_rows_read": 0,
            "realized_funding_rows_read": 0,
            "future_return_rows_read": 0,
            "pnl_cagr_mdd_values_opened": 0,
            "network_calls": 0,
            "git_protocol_subprocess_calls_before_source_read": 0,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}, clock_bytes


def serialized_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _git_check(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_protocol_committed() -> None:
    paths = (SCRIPT_PATH, TEST_PATH, IMPLEMENTATION_CONTRACT)
    labels = tuple(str(path) for path in paths)
    if not all(_repository_path(path).is_file() for path in paths):
        raise RuntimeError("FCCM source-support protocol file is missing")
    tracked = _git_check("ls-files", "--error-unmatch", "--", *labels)
    if tracked.returncode != 0:
        raise RuntimeError("FCCM source-support protocol is not committed")
    clean = _git_check("diff", "--quiet", "HEAD", "--", *labels)
    if clean.returncode != 0:
        raise RuntimeError("FCCM source-support protocol differs from HEAD")


def _atomic_link(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_support(
    *,
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
) -> tuple[dict[str, Any], str]:
    _assert_protocol_committed()
    _validate_preregistration_artifact()
    if sha256_file(IMPLEMENTATION_CONTRACT) != IMPLEMENTATION_CONTRACT_SHA256:
        raise RuntimeError("FCCM source-support implementation contract drift")
    bitfinex, wbtc, source_audit = _load_sources()
    payload, clock_bytes = build_support_payload(
        bitfinex,
        wbtc,
        source_audit,
        clock_output=clock_output,
    )
    if (
        payload.get("artifact_eligible") is not False
        or payload.get("advance_to_novelty_evaluator") is not False
        or payload.get("decision")
        != "ineligible_uncommitted_or_injected_source_support_payload"
    ):
        raise RuntimeError("FCCM direct source-support envelope was pre-authorized")
    payload["verification_mode"] = "committed_protocol_and_verified_source_hashes"
    payload["artifact_eligible"] = True
    payload["outcome_boundary"][
        "git_protocol_subprocess_calls_before_source_read"
    ] = 2
    payload["decision"] = (
        "advance_to_committed_novelty_evaluator"
        if payload["source_support_passed"]
        else "retire_FCCM_72_unchanged_before_comparators_and_outcomes"
    )
    payload["advance_to_novelty_evaluator"] = payload["source_support_passed"]
    payload_core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    payload["manifest_hash"] = canonical_hash(payload_core)
    report_bytes = serialized_json(payload)
    clock_path = _repository_path(clock_output)
    report_path = _repository_path(report_output)
    if clock_path.exists() or report_path.exists():
        if (
            clock_path.is_file()
            and report_path.is_file()
            and clock_path.read_bytes() == clock_bytes
            and report_path.read_bytes() == report_bytes
        ):
            return payload, "verified_existing"
        raise RuntimeError("existing FCCM source-support output differs")
    _atomic_link(clock_path, clock_bytes)
    try:
        _atomic_link(report_path, report_bytes)
    except Exception:
        clock_path.unlink(missing_ok=True)
        raise
    return payload, "created"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK_OUTPUT))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT_OUTPUT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload, status = write_support(
        clock_output=args.clock_output,
        report_output=args.report_output,
    )
    print(
        json.dumps(
            {
                "candidate": payload["candidate"],
                "status": status,
                "source_support_passed": payload["source_support_passed"],
                "decision": payload["decision"],
                "clock_rows": payload["clock"]["rows"],
                "manifest_hash": payload["manifest_hash"],
                "comparator_value_rows_read": payload["outcome_boundary"][
                    "comparator_value_rows_read"
                ],
                "outcomes_opened": any(
                    payload["outcome_boundary"][field]
                    for field in (
                        "btc_market_rows_read",
                        "realized_funding_rows_read",
                        "future_return_rows_read",
                        "pnl_cagr_mdd_values_opened",
                    )
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
