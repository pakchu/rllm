"""Reproduce source-only WTSL-168-SOURCE-SEEN clocks and controls."""

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
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from training import preregister_wbtc_turnover_stablecoin_liquidity as prereg


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "wbtc_turnover_stablecoin_liquidity_support_v1"
CANDIDATE = prereg.POLICY_ID
PREREGISTRATION_COMMIT = "48ef81f"
PREREGISTRATION_ARTIFACT = prereg.DEFAULT_OUTPUT
PREREGISTRATION_ARTIFACT_SHA256 = (
    "23a1c884306fbde2ef90d02f20de985229c334e1a21992796206d3db6413f92c"
)
PREREGISTRATION_MANIFEST_HASH = (
    "81f41c68b526a2e22a4da769e973026255d44251f7df996e7cdbc5eb8a66ac4a"
)
SCRIPT_PATH = Path("training/build_wbtc_turnover_stablecoin_liquidity_support.py")
DEFAULT_CLOCK_OUTPUT = Path(
    "data/wbtc_turnover_stablecoin_liquidity_2021_2023/"
    "wtsl168_support_clocks_2021_2023.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/wbtc_turnover_stablecoin_liquidity_support_2026-07-23.json"
)

UTC = timezone.utc
ZERO_ADDRESS = "0x" + "0" * 40
ADDRESS = re.compile(r"^0x[0-9a-f]{40}$")
SOURCE_COVERAGE_START = datetime(2020, 1, 1, tzinfo=UTC)
SOURCE_SEALED_FROM = datetime(2024, 1, 1, tzinfo=UTC)
CURRENT_WINDOW_HOURS = 168
PRIOR_ENDPOINTS = 1460
PRIOR_STEP_HOURS = 6
COMPLETE_HISTORY_HOURS = 8928
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
)

CLOCK_COLUMNS = (
    "candidate",
    "control",
    "signal_id",
    "window",
    "decision_time",
    "source_cutoff",
    "entry_time",
    "exit_time",
    "side",
    "wbtc_net_raw",
    "wbtc_gross_raw",
    "wbtc_rows",
    "wbtc_distinct_actors",
    "wbtc_top_actor_share",
    "wbtc_prior_median_twice_raw",
    "stablecoin_scope",
    "stablecoin_net_raw",
    "stablecoin_gross_raw",
    "stablecoin_rows",
    "stablecoin_veto_rows",
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
class FlowState:
    source: str
    cutoff: datetime
    net_raw: int
    gross_raw: int
    rows: int
    actors: tuple[str, ...] = ()
    top_actor_share: float = 0.0
    prior_median_twice_raw: int = 0
    veto_rows: int = 0
    valid: bool = False


@dataclass(frozen=True)
class AnchorState:
    decision_time: datetime
    source_cutoff: datetime
    stablecoin_scope: str
    wbtc: FlowState
    stablecoin: FlowState
    usdc: FlowState
    usdt: FlowState


@dataclass(frozen=True)
class Candidate:
    control: str
    decision_time: datetime
    source_cutoff: datetime
    entry_time: datetime
    exit_time: datetime
    side: int
    window: str
    stablecoin_scope: str
    wbtc: FlowState
    stablecoin: FlowState
    usdc: FlowState
    usdt: FlowState

    @property
    def signal_id(self) -> str:
        raw = (
            f"{CANDIDATE}|{self.control}|{format_time(self.decision_time)}|"
            f"{format_time(self.source_cutoff)}|{self.side}"
        ).encode("ascii")
        return hashlib.sha256(raw).hexdigest()


class FlowIndex:
    def __init__(self, events: Sequence[SourceEvent]) -> None:
        self.events = tuple(
            sorted(events, key=lambda event: (event.available_at, event.identity))
        )
        self.times = tuple(event.available_at for event in self.events)
        self.prefix_net = [0]
        self.prefix_gross = [0]
        self._gross_cache: dict[datetime, int] = {}
        for event in self.events:
            self.prefix_net.append(self.prefix_net[-1] + event.sign * event.amount_raw)
            self.prefix_gross.append(self.prefix_gross[-1] + event.amount_raw)

    def bounds(self, cutoff: datetime, hours: int = CURRENT_WINDOW_HOURS) -> tuple[int, int]:
        lower = cutoff - timedelta(hours=hours)
        return bisect.bisect_right(self.times, lower), bisect.bisect_right(
            self.times, cutoff
        )

    def basic(self, cutoff: datetime, *, source: str) -> FlowState:
        left, right = self.bounds(cutoff)
        return FlowState(
            source=source,
            cutoff=cutoff,
            net_raw=self.prefix_net[right] - self.prefix_net[left],
            gross_raw=self.prefix_gross[right] - self.prefix_gross[left],
            rows=right - left,
        )

    def gross(self, cutoff: datetime) -> int:
        cached = self._gross_cache.get(cutoff)
        if cached is not None:
            return cached
        left, right = self.bounds(cutoff)
        value = self.prefix_gross[right] - self.prefix_gross[left]
        self._gross_cache[cutoff] = value
        return value

    def wbtc(self, cutoff: datetime, actor_cap: float) -> FlowState:
        left, right = self.bounds(cutoff)
        basic = self.basic(cutoff, source="wbtc")
        actor_gross: Counter[str] = Counter()
        for event in self.events[left:right]:
            actor_gross[event.actor] += event.amount_raw
        actors = tuple(sorted(actor_gross))
        top_share = (
            max(actor_gross.values()) / basic.gross_raw
            if actor_gross and basic.gross_raw > 0
            else 0.0
        )
        prior = [
            self.gross(cutoff - timedelta(hours=PRIOR_STEP_HOURS * offset))
            for offset in range(1, PRIOR_ENDPOINTS + 1)
        ]
        ordered = sorted(prior)
        middle_sum = ordered[PRIOR_ENDPOINTS // 2 - 1] + ordered[PRIOR_ENDPOINTS // 2]
        complete = cutoff - timedelta(hours=COMPLETE_HISTORY_HOURS) >= SOURCE_COVERAGE_START
        valid = bool(
            complete
            and basic.gross_raw > 0
            and basic.rows >= 2
            and len(actors) >= 2
            and top_share <= actor_cap
            and 2 * basic.gross_raw >= middle_sum
        )
        return replace(
            basic,
            actors=actors,
            top_actor_share=top_share,
            prior_median_twice_raw=middle_sum,
            valid=valid,
        )


class VetoIndex:
    def __init__(self, events: Sequence[SourceEvent]) -> None:
        self.times = tuple(sorted(event.available_at for event in events))

    def rows(self, cutoff: datetime) -> int:
        lower = cutoff - timedelta(hours=CURRENT_WINDOW_HOURS)
        left = bisect.bisect_right(self.times, lower)
        right = bisect.bisect_right(self.times, cutoff)
        return right - left


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
        raise ValueError("WTSL timestamp must be UTC")
    return parsed.astimezone(UTC)


def format_time(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_positive_int(value: str, field: str) -> int:
    if not value or not value.isdigit() or int(value) <= 0:
        raise ValueError(f"WTSL {field} must be a canonical positive integer")
    return int(value)


def _canonical_nonnegative_int(value: str, field: str) -> int:
    if not value or not value.isdigit():
        raise ValueError(f"WTSL {field} must be a canonical nonnegative integer")
    return int(value)


def validate_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION_ARTIFACT) != PREREGISTRATION_ARTIFACT_SHA256:
        raise RuntimeError("WTSL preregistration artifact SHA-256 drift")
    payload = json.loads(_path(PREREGISTRATION_ARTIFACT).read_text(encoding="utf-8"))
    prereg.validate_preregistration(payload)
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("WTSL preregistration manifest hash drift")
    if payload.get("candidate") != CANDIDATE:
        raise RuntimeError("WTSL preregistration candidate drift")
    if payload.get("source_incidence_disclosure") != prereg.PRIOR_SOURCE_DISCLOSURE:
        raise RuntimeError("WTSL source-incidence disclosure drift")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("WTSL preregistration opened outcomes")
    return payload


def load_wbtc_events(
    path: str | Path = prereg.WBTC_SOURCE,
) -> tuple[list[SourceEvent], dict[str, Any]]:
    events: list[SourceEvent] = []
    identities: set[tuple[int, int, int]] = set()
    previous: tuple[datetime, tuple[int, int, int]] | None = None
    with gzip.open(_path(path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != prereg.WBTC_HEADER:
            raise RuntimeError("WTSL WBTC source header drift")
        for row in reader:
            available_at = parse_time(row["available_at"])
            if available_at >= SOURCE_SEALED_FROM:
                raise RuntimeError("WTSL WBTC source crossed sealed boundary")
            event = row["event"]
            if event not in {"mint", "burn"}:
                raise RuntimeError("WTSL WBTC source event drift")
            if row["event_sign"] not in {"1", "-1"}:
                raise RuntimeError("WTSL WBTC event sign is not canonical")
            sign = int(row["event_sign"])
            if sign != (1 if event == "mint" else -1):
                raise RuntimeError("WTSL WBTC event sign drift")
            actor = row["actor_address"].lower()
            if not ADDRESS.fullmatch(actor) or actor == ZERO_ADDRESS:
                raise RuntimeError("WTSL WBTC actor drift")
            identity = (
                _canonical_nonnegative_int(row["block_number"], "block_number"),
                _canonical_nonnegative_int(row["transaction_index"], "transaction_index"),
                _canonical_nonnegative_int(row["semantic_log_index"], "semantic_log_index"),
            )
            if identity in identities:
                raise RuntimeError("WTSL WBTC identity duplicated")
            identities.add(identity)
            parsed = SourceEvent(
                source="wbtc",
                asset="wbtc_eth",
                event=event,
                sign=sign,
                amount_raw=_canonical_positive_int(row["amount_raw"], "amount_raw"),
                available_at=available_at,
                identity=identity,
                actor=actor,
            )
            key = (parsed.available_at, parsed.identity)
            if previous is not None and key < previous:
                raise RuntimeError("WTSL WBTC source is not sorted")
            previous = key
            events.append(parsed)
    if len(events) != 993:
        raise RuntimeError("WTSL WBTC row count drift")
    return events, {
        "physical_rows_read": len(events),
        "eligible_rows": len(events),
        "unique_identities": len(identities),
        "first_available_at": format_time(events[0].available_at),
        "last_available_at": format_time(events[-1].available_at),
        "post_2023_contract_event_value_rows_loaded": 0,
    }


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
    rows_before_seal = 0
    boundary_sentinels = 0
    previous: tuple[datetime, tuple[int, int, int]] | None = None
    allowed = {
        ("usdc_eth", "mint"): 1,
        ("usdc_eth", "burn"): -1,
        ("usdt_eth", "issue"): 1,
        ("usdt_eth", "redeem"): -1,
        ("usdt_eth", "destroyed_black_funds"): -1,
    }
    with gzip.open(_path(path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != prereg.STABLECOIN_HEADER:
            raise RuntimeError("WTSL stablecoin source header drift")
        for row in reader:
            available_at = parse_time(row["available_at"])
            if available_at >= SOURCE_SEALED_FROM:
                boundary_sentinels += 1
                break
            rows_before_seal += 1
            key = (row["asset"], row["event"])
            if key not in allowed or int(row["event_sign"]) != allowed[key]:
                raise RuntimeError("WTSL stablecoin event mapping drift")
            if row["decimals"] != "6":
                raise RuntimeError("WTSL stablecoin decimals drift")
            identity = (
                _canonical_nonnegative_int(row["block_number"], "block_number"),
                _canonical_nonnegative_int(row["transaction_index"], "transaction_index"),
                _canonical_nonnegative_int(row["log_index"], "log_index"),
            )
            if identity in identities:
                raise RuntimeError("WTSL stablecoin identity duplicated")
            identities.add(identity)
            parsed = SourceEvent(
                source="stablecoin",
                asset=row["asset"],
                event=row["event"],
                sign=allowed[key],
                amount_raw=_canonical_positive_int(row["amount_raw"], "amount_raw"),
                available_at=available_at,
                identity=identity,
            )
            order_key = (parsed.available_at, parsed.identity)
            if previous is not None and order_key < previous:
                raise RuntimeError("WTSL stablecoin source is not sorted")
            previous = order_key
            if parsed.event == "destroyed_black_funds":
                veto.append(parsed)
            else:
                directional.append(parsed)
    if rows_before_seal != expected_rows_before_seal:
        raise RuntimeError("WTSL stablecoin pre-seal row count drift")
    if len(directional) != expected_directional_rows:
        raise RuntimeError("WTSL stablecoin directional row count drift")
    if len(veto) != expected_veto_rows:
        raise RuntimeError("WTSL stablecoin veto row count drift")
    if boundary_sentinels != 1:
        raise RuntimeError("WTSL stablecoin boundary sentinel drift")
    all_events = sorted(
        directional + veto,
        key=lambda event: (event.available_at, event.identity),
    )
    return directional, veto, {
        "physical_rows_read": rows_before_seal,
        "directional_rows": len(directional),
        "veto_rows": len(veto),
        "unique_identities": len(identities),
        "first_available_at": format_time(all_events[0].available_at),
        "last_available_at": format_time(all_events[-1].available_at),
        "sealed_from": format_time(SOURCE_SEALED_FROM),
        "boundary_sentinel_timestamp_rows_scanned": boundary_sentinels,
        "post_2023_contract_event_value_rows_loaded": 0,
    }


def permute_amounts(events: Sequence[SourceEvent]) -> list[SourceEvent]:
    groups: dict[tuple[str, str, str, int], list[SourceEvent]] = defaultdict(list)
    for event in events:
        groups[(event.source, event.asset, event.event, event.available_at.year)].append(event)
    output: list[SourceEvent] = []
    for key in sorted(groups):
        canonical = sorted(groups[key], key=lambda event: event.identity)
        donors = sorted(
            canonical,
            key=lambda event: hashlib.sha256(
                f"WTSL-AMOUNT-PERMUTE|{key}|{event.identity}".encode("ascii")
            ).digest(),
        )
        amounts = [event.amount_raw for event in donors]
        output.extend(
            replace(event, amount_raw=amount)
            for event, amount in zip(canonical, amounts)
        )
    return sorted(output, key=lambda event: (event.available_at, event.identity))


def _stable_state(
    index: FlowIndex,
    veto_index: VetoIndex,
    cutoff: datetime,
    *,
    scope: str,
    apply_veto: bool,
) -> FlowState:
    state = index.basic(cutoff, source=scope)
    veto_rows = veto_index.rows(cutoff) if apply_veto else 0
    valid = bool(
        state.gross_raw > 0
        and abs(state.net_raw) * 20 >= state.gross_raw
        and state.net_raw != 0
        and veto_rows == 0
    )
    return replace(state, veto_rows=veto_rows, valid=valid)


def _indices(
    wbtc_events: Sequence[SourceEvent],
    stablecoin_events: Sequence[SourceEvent],
    veto_events: Sequence[SourceEvent],
) -> dict[str, Any]:
    return {
        "wbtc": FlowIndex(wbtc_events),
        "combined": FlowIndex(stablecoin_events),
        "usdc": FlowIndex([event for event in stablecoin_events if event.asset == "usdc_eth"]),
        "usdt": FlowIndex([event for event in stablecoin_events if event.asset == "usdt_eth"]),
        "veto": VetoIndex(veto_events),
    }


def anchor_state(
    decision_time: datetime,
    indexes: Mapping[str, Any],
    *,
    stale_hours: int = 0,
    actor_cap: float = 0.85,
    stablecoin_scope: str = "combined",
    apply_veto: bool = True,
) -> AnchorState:
    cutoff = decision_time - timedelta(hours=6 + stale_hours)
    wbtc = indexes["wbtc"].wbtc(cutoff, actor_cap)
    usdc = _stable_state(
        indexes["usdc"], indexes["veto"], cutoff, scope="usdc", apply_veto=False
    )
    usdt = _stable_state(
        indexes["usdt"], indexes["veto"], cutoff, scope="usdt", apply_veto=True
    )
    stablecoin = _stable_state(
        indexes[stablecoin_scope],
        indexes["veto"],
        cutoff,
        scope=stablecoin_scope,
        apply_veto=apply_veto,
    )
    return AnchorState(decision_time, cutoff, stablecoin_scope, wbtc, stablecoin, usdc, usdt)


def state_side(state: AnchorState, control: str) -> int:
    if control == "stablecoin_only_direct":
        return (1 if state.stablecoin.net_raw > 0 else -1) if state.stablecoin.valid else 0
    if control == "wbtc_signed_placebo":
        if not state.wbtc.valid or state.wbtc.net_raw == 0:
            return 0
        return 1 if state.wbtc.net_raw > 0 else -1
    if not state.wbtc.valid or not state.stablecoin.valid:
        return 0
    return 1 if state.stablecoin.net_raw > 0 else -1


def _window_for(entry_time: datetime, exit_time: datetime) -> str | None:
    for name, raw in WINDOWS.items():
        start, end = map(parse_time, raw)
        if start <= entry_time and exit_time <= end:
            return name
    return None


def six_hour_anchors() -> Iterable[datetime]:
    current = parse_time(WINDOWS["train"][0])
    end = parse_time(WINDOWS["selection"][1])
    while current < end:
        yield current
        current += timedelta(hours=6)


def schedule_states(states: Sequence[AnchorState], control: str) -> list[Candidate]:
    output: list[Candidate] = []
    last_exit: datetime | None = None
    for state in states:
        side = state_side(state, control)
        if side == 0:
            continue
        entry = state.decision_time + timedelta(minutes=10)
        exit_time = entry + timedelta(hours=24)
        window = _window_for(entry, exit_time)
        if window is None:
            continue
        if last_exit is not None and entry < last_exit:
            continue
        candidate = Candidate(
            control=control,
            decision_time=state.decision_time,
            source_cutoff=state.source_cutoff,
            entry_time=entry,
            exit_time=exit_time,
            side=side,
            window=window,
            stablecoin_scope=state.stablecoin_scope,
            wbtc=state.wbtc,
            stablecoin=state.stablecoin,
            usdc=state.usdc,
            usdt=state.usdt,
        )
        output.append(candidate)
        last_exit = exit_time
    return output


def exact_clock_control(
    primary: Sequence[Candidate], control: str, sides: Sequence[int]
) -> list[Candidate]:
    if len(primary) != len(sides):
        raise ValueError("WTSL exact-clock side length mismatch")
    return [replace(row, control=control, side=side) for row, side in zip(primary, sides)]


def deterministic_random_sides(primary: Sequence[Candidate]) -> list[int]:
    sides = [row.side for row in primary]
    ranked = sorted(
        range(len(primary)),
        key=lambda index: hashlib.sha256(
            f"WTSL-RANDOM-SIDE|{primary[index].signal_id}".encode("ascii")
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
    indexes = _indices(wbtc_events, stablecoin_events, veto_events)
    anchors = list(six_hour_anchors())

    def states(**kwargs: Any) -> list[AnchorState]:
        return [anchor_state(anchor, indexes, **kwargs) for anchor in anchors]

    controls: dict[str, list[Candidate]] = {}
    base_states = states()
    primary = schedule_states(base_states, "primary")
    controls["primary"] = primary
    controls["direction_flip"] = exact_clock_control(
        primary, "direction_flip", [-row.side for row in primary]
    )
    controls["stablecoin_only_direct"] = schedule_states(
        base_states, "stablecoin_only_direct"
    )
    controls["wbtc_signed_placebo"] = schedule_states(
        base_states, "wbtc_signed_placebo"
    )
    controls["stale_24h"] = schedule_states(states(stale_hours=24), "stale_24h")
    controls["stale_48h"] = schedule_states(states(stale_hours=48), "stale_48h")
    actor_cap_states = [
        replace(
            state,
            wbtc=replace(
                state.wbtc,
                valid=state.wbtc.valid and state.wbtc.top_actor_share <= 0.60,
            ),
        )
        for state in base_states
    ]
    controls["actor_cap_60"] = schedule_states(actor_cap_states, "actor_cap_60")
    no_veto_states = [
        replace(
            state,
            stablecoin=replace(
                state.stablecoin,
                veto_rows=0,
                valid=(
                    state.stablecoin.gross_raw > 0
                    and abs(state.stablecoin.net_raw) * 20
                    >= state.stablecoin.gross_raw
                    and state.stablecoin.net_raw != 0
                ),
            ),
        )
        for state in base_states
    ]
    controls["no_black_funds_veto"] = schedule_states(
        no_veto_states, "no_black_funds_veto"
    )
    usdc_states = [
        replace(
            state,
            stablecoin_scope="usdc",
            stablecoin=state.usdc,
        )
        for state in base_states
    ]
    controls["usdc_only_direct"] = schedule_states(
        usdc_states, "usdc_only_direct"
    )
    usdt_states = [
        replace(
            state,
            stablecoin_scope="usdt",
            stablecoin=state.usdt,
        )
        for state in base_states
    ]
    controls["usdt_only_direct"] = schedule_states(
        usdt_states, "usdt_only_direct"
    )
    permuted_indexes = _indices(
        permute_amounts(wbtc_events),
        permute_amounts(stablecoin_events),
        veto_events,
    )
    controls["year_amount_permutation"] = schedule_states(
        [anchor_state(anchor, permuted_indexes) for anchor in anchors],
        "year_amount_permutation",
    )
    controls["deterministic_random_side"] = exact_clock_control(
        primary,
        "deterministic_random_side",
        deterministic_random_sides(primary),
    )
    if tuple(controls) != CONTROL_ORDER:
        raise RuntimeError("WTSL control order drift")
    return controls


def _maximum_run(rows: Sequence[Candidate]) -> int:
    best = current = 0
    prior: int | None = None
    for row in rows:
        current = current + 1 if row.side == prior else 1
        prior = row.side
        best = max(best, current)
    return best


def _window_stats(rows: Sequence[Candidate], window: str) -> dict[str, Any]:
    selected = [row for row in rows if row.window == window]
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
    actors = sorted({actor for row in selected for actor in row.wbtc.actors})
    return {
        "trades": len(selected),
        "year_counts": dict(sorted(years.items())),
        "half_year_counts": dict(sorted(halves.items())),
        "quarter_counts": dict(sorted(quarters.items())),
        "side_counts": {side: sides.get(side, 0) for side in ("long", "short")},
        "maximum_month_share": max(months.values(), default=0) / len(selected)
        if selected
        else 0.0,
        "maximum_quarter_share": max(quarters.values(), default=0) / len(selected)
        if selected
        else 0.0,
        "maximum_consecutive_same_side": _maximum_run(selected),
        "distinct_wbtc_actors": len(actors),
        "wbtc_actor_set_hash": hashlib.sha256("\n".join(actors).encode("ascii")).hexdigest(),
    }


def support_statistics(primary: Sequence[Candidate]) -> dict[str, Any]:
    return {
        "total_trades": len(primary),
        "all_side_counts": {
            side: sum(1 for row in primary if (row.side > 0) == (side == "long"))
            for side in ("long", "short")
        },
        "all_year_counts": dict(
            sorted(Counter(str(row.entry_time.year) for row in primary).items())
        ),
        "train": _window_stats(primary, "train"),
        "selection": _window_stats(primary, "selection"),
    }


def support_checks(stats: Mapping[str, Any]) -> dict[str, bool]:
    train = stats["train"]
    selection = stats["selection"]
    disclosed = prereg.PRIOR_SOURCE_DISCLOSURE
    return {
        "disclosed_total_reproduced": stats["total_trades"]
        == disclosed["pre_2024_primary_candidates"],
        "disclosed_sides_reproduced": stats["all_side_counts"]
        == disclosed["side_counts"],
        "disclosed_years_reproduced": stats["all_year_counts"]
        == disclosed["year_counts"],
        "disclosed_selection_sides_reproduced": selection["side_counts"]
        == disclosed["selection_2023_side_counts"],
        "train_total_minimum": train["trades"] >= 120,
        "selection_total_minimum": selection["trades"] >= 20,
        "each_train_year_minimum": all(
            train["year_counts"].get(str(year), 0) >= 20 for year in (2021, 2022)
        ),
        "each_train_half_year_minimum": all(
            train["half_year_counts"].get(f"{year}H{half}", 0) >= 8
            for year in (2021, 2022)
            for half in (1, 2)
        ),
        "each_selection_half_year_minimum": all(
            selection["half_year_counts"].get(f"2023H{half}", 0) >= 6
            for half in (1, 2)
        ),
        "train_each_side_minimum": all(
            train["side_counts"].get(side, 0) >= 24 for side in ("long", "short")
        ),
        "selection_each_side_minimum": all(
            selection["side_counts"].get(side, 0) >= 8 for side in ("long", "short")
        ),
        "maximum_month_share": all(
            split["maximum_month_share"] <= 0.20 for split in (train, selection)
        ),
        "maximum_quarter_share": all(
            split["maximum_quarter_share"] <= 0.40 for split in (train, selection)
        ),
        "maximum_consecutive_same_side": all(
            split["maximum_consecutive_same_side"] <= 20
            for split in (train, selection)
        ),
        "train_distinct_wbtc_actors_minimum": train["distinct_wbtc_actors"] >= 10,
        "selection_distinct_wbtc_actors_minimum": (
            selection["distinct_wbtc_actors"] >= 5
        ),
    }


def exact_entry_jaccard(left: Sequence[Candidate], right: Sequence[Candidate]) -> float:
    left_set = {row.entry_time for row in left}
    right_set = {row.entry_time for row in right}
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def candidate_row(row: Candidate) -> dict[str, Any]:
    return {
        "candidate": CANDIDATE,
        "control": row.control,
        "signal_id": row.signal_id,
        "window": row.window,
        "decision_time": format_time(row.decision_time),
        "source_cutoff": format_time(row.source_cutoff),
        "entry_time": format_time(row.entry_time),
        "exit_time": format_time(row.exit_time),
        "side": row.side,
        "wbtc_net_raw": row.wbtc.net_raw,
        "wbtc_gross_raw": row.wbtc.gross_raw,
        "wbtc_rows": row.wbtc.rows,
        "wbtc_distinct_actors": len(row.wbtc.actors),
        "wbtc_top_actor_share": f"{row.wbtc.top_actor_share:.12f}",
        "wbtc_prior_median_twice_raw": row.wbtc.prior_median_twice_raw,
        "stablecoin_scope": row.stablecoin_scope,
        "stablecoin_net_raw": row.stablecoin.net_raw,
        "stablecoin_gross_raw": row.stablecoin.gross_raw,
        "stablecoin_rows": row.stablecoin.rows,
        "stablecoin_veto_rows": row.stablecoin.veto_rows,
        "usdc_net_raw": row.usdc.net_raw,
        "usdc_gross_raw": row.usdc.gross_raw,
        "usdt_net_raw": row.usdt.net_raw,
        "usdt_gross_raw": row.usdt.gross_raw,
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
    checks = support_checks(stats)
    passed = all(checks.values())
    rows = [candidate_row(row) for control in CONTROL_ORDER for row in controls[control]]
    clock_bytes = deterministic_gzip_csv(rows)
    control_report = {
        control: {
            "trades": len(candidates),
            "train": _window_stats(candidates, "train"),
            "selection": _window_stats(candidates, "selection"),
            "exact_entry_jaccard_to_primary": exact_entry_jaccard(primary, candidates),
        }
        for control, candidates in controls.items()
    }
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": CANDIDATE,
        "research_status": "source-seen_outcome-blind",
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_artifact": str(PREREGISTRATION_ARTIFACT),
        "preregistration_artifact_sha256": PREREGISTRATION_ARTIFACT_SHA256,
        "preregistration_manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        "preregistration_reproduced": registration["manifest_hash"]
        == PREREGISTRATION_MANIFEST_HASH,
        "evaluator_source": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        "source_audit": {
            "wbtc": wbtc_audit,
            "stablecoin": stablecoin_audit,
            "source_value_rows_read": wbtc_audit["physical_rows_read"]
            + stablecoin_audit["physical_rows_read"],
            "post_2023_contract_event_value_rows_loaded": 0,
            "post_2023_timestamp_sentinels_scanned": stablecoin_audit[
                "boundary_sentinel_timestamp_rows_scanned"
            ],
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
        "control_report": control_report,
        "decision": (
            "advance_to_strict_evaluator_freeze"
            if passed
            else "retire_WTSL_168_SOURCE_SEEN_without_outcomes"
        ),
        "advance_to_strict_evaluator_freeze": passed,
        "outcome_boundary": {
            "outcomes_opened": False,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_contract_event_value_rows_loaded": 0,
            "network_calls": 0,
            "subprocess_calls": 0,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}, clock_bytes


def _write_once(path: str | Path, payload: bytes) -> None:
    destination = _path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise FileExistsError(f"WTSL support artifact is write-once: {path}") from error


def write_support(
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
) -> dict[str, Any]:
    if _path(clock_output).exists() or _path(report_output).exists():
        raise FileExistsError("WTSL support outputs are write-once")
    payload, clock_bytes = build_support_payload(clock_output)
    report_bytes = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
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
