"""Build outcome-blind WCDR-2016 source states, controls, and support verdict."""

from __future__ import annotations

import argparse
import bisect
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from training import preregister_wrapped_collateral_dollar_liquidity_rotation as prereg


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "wrapped_collateral_dollar_liquidity_rotation_support_v2"
CANDIDATE = prereg.POLICY_ID
PREREGISTRATION_COMMIT = "501a767"
PREREGISTRATION_ARTIFACT = prereg.DEFAULT_OUTPUT
PREREGISTRATION_ARTIFACT_SHA256 = (
    "aa4dba88103dc6be1204d4cfb6185a9396c5fe919a4b18fb96ad54188c812dfc"
)
PREREGISTRATION_MANIFEST_HASH = (
    "267fae61f29caa3117846349c6346e14cbd041e0ed121249c2f9fcdf8f37bf4f"
)
SCRIPT_PATH = Path(
    "training/build_wrapped_collateral_dollar_liquidity_rotation_support.py"
)
DEFAULT_CLOCK_OUTPUT = Path(
    "data/wrapped_collateral_dollar_liquidity_rotation_2021_2023/"
    "wcdr2016_support_clocks_2021_2023.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/wrapped_collateral_dollar_liquidity_rotation_"
    "support_2026-07-23.json"
)

UTC = timezone.utc
ZERO_ADDRESS = "0x" + "0" * 40
ADDRESS = re.compile(r"^0x[0-9a-f]{40}$")

CONTROL_ORDER = (
    "primary",
    "direction_flip",
    "wbtc_only_contrarian",
    "usdc_only_direct",
    "same_sign_direct",
    "stale_7d",
    "count_sign_consensus",
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
    "wbtc_count_net",
    "wbtc_rows",
    "wbtc_distinct_actors",
    "wbtc_top_actor_share",
    "usdc_net_raw",
    "usdc_gross_raw",
    "usdc_count_net",
    "usdc_rows",
)

WINDOWS = {
    "train": (
        "2021-01-01T00:00:00Z",
        "2023-01-01T00:00:00Z",
    ),
    "selection": (
        "2023-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
    ),
}
SOURCE_SEALED_FROM = "2024-01-01T00:00:00Z"
EXPECTED_USDC_ROWS_BEFORE_SEAL = 266_360
EXPECTED_USDC_ELIGIBLE_ROWS_BEFORE_SEAL = 265_583


@dataclass(frozen=True)
class SourceEvent:
    source: str
    event: str
    sign: int
    amount_raw: int
    available_at: datetime
    identity: tuple[int, int, int]
    actor: str = ""


@dataclass(frozen=True)
class WindowState:
    source: str
    cutoff: datetime
    lookback_days: int
    net_raw: int
    gross_raw: int
    count_net: int
    rows: int
    actors: tuple[str, ...] = ()
    top_actor_share: float = 0.0
    valid: bool = False


@dataclass(frozen=True)
class DailyState:
    decision_time: datetime
    source_cutoff: datetime
    wbtc: WindowState
    usdc: WindowState


@dataclass(frozen=True)
class Candidate:
    control: str
    decision_time: datetime
    source_cutoff: datetime
    entry_time: datetime
    exit_time: datetime
    side: int
    window: str
    wbtc: WindowState
    usdc: WindowState

    @property
    def signal_id(self) -> str:
        encoded = (
            f"{CANDIDATE}|{self.control}|{format_time(self.decision_time)}|"
            f"{format_time(self.source_cutoff)}|{self.side}"
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


class EventIndex:
    def __init__(self, events: Sequence[SourceEvent]) -> None:
        self.events = tuple(
            sorted(events, key=lambda event: (event.available_at, event.identity))
        )
        self.times = tuple(event.available_at for event in self.events)
        self.prefix_net = [0]
        self.prefix_gross = [0]
        self.prefix_count = [0]
        for event in self.events:
            self.prefix_net.append(
                self.prefix_net[-1] + event.sign * event.amount_raw
            )
            self.prefix_gross.append(self.prefix_gross[-1] + event.amount_raw)
            self.prefix_count.append(self.prefix_count[-1] + event.sign)

    def window(self, cutoff: datetime, lookback_days: int) -> tuple[int, int]:
        lower = cutoff - timedelta(days=lookback_days)
        left = bisect.bisect_right(self.times, lower)
        right = bisect.bisect_right(self.times, cutoff)
        return left, right

    def aggregate(
        self,
        cutoff: datetime,
        lookback_days: int,
        *,
        source: str,
    ) -> WindowState:
        left, right = self.window(cutoff, lookback_days)
        net_raw = self.prefix_net[right] - self.prefix_net[left]
        gross_raw = self.prefix_gross[right] - self.prefix_gross[left]
        count_net = self.prefix_count[right] - self.prefix_count[left]
        rows = right - left
        if source == "wbtc":
            actor_gross: Counter[str] = Counter()
            for event in self.events[left:right]:
                actor_gross[event.actor] += event.amount_raw
            actors = tuple(sorted(actor_gross))
            top_actor_share = (
                max(actor_gross.values()) / gross_raw
                if actor_gross and gross_raw > 0
                else 0.0
            )
            valid = bool(
                gross_raw > 0
                and rows >= 3
                and len(actors) >= 2
                and top_actor_share <= 0.90
            )
        elif source == "usdc":
            actors = ()
            top_actor_share = 0.0
            valid = bool(gross_raw > 0 and rows >= 30)
        else:
            raise KeyError(source)
        return WindowState(
            source=source,
            cutoff=cutoff,
            lookback_days=lookback_days,
            net_raw=net_raw,
            gross_raw=gross_raw,
            count_net=count_net,
            rows=rows,
            actors=actors,
            top_actor_share=top_actor_share,
            valid=valid,
        )


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
        raise ValueError("WCDR timestamp must be UTC")
    return parsed.astimezone(UTC)


def format_time(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_positive_int(value: str, field: str) -> int:
    if not value or not value.isdigit():
        raise ValueError(f"WCDR {field} is not a canonical integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"WCDR {field} must be positive")
    return parsed


def _canonical_nonnegative_int(value: str, field: str) -> int:
    if not value or not value.isdigit():
        raise ValueError(f"WCDR {field} is not a canonical integer")
    return int(value)


def validate_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION_ARTIFACT) != PREREGISTRATION_ARTIFACT_SHA256:
        raise RuntimeError("WCDR preregistration artifact SHA-256 drift")
    payload = json.loads(
        _path(PREREGISTRATION_ARTIFACT).read_text(encoding="utf-8")
    )
    prereg.validate_preregistration(payload)
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("WCDR preregistration manifest hash drift")
    if payload.get("candidate") != CANDIDATE:
        raise RuntimeError("WCDR preregistration candidate drift")
    if payload.get("source_incidence_opened") is not False:
        raise RuntimeError("WCDR preregistration already opened source incidence")
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
            raise RuntimeError("WCDR WBTC source header drift")
        for row in reader:
            event_name = row["event"]
            if event_name not in {"mint", "burn"}:
                raise RuntimeError("WCDR WBTC source contains an ineligible event")
            sign = int(row["event_sign"])
            if sign != (1 if event_name == "mint" else -1):
                raise RuntimeError("WCDR WBTC event sign drift")
            actor = row["actor_address"].lower()
            if not ADDRESS.fullmatch(actor) or actor == ZERO_ADDRESS:
                raise RuntimeError("WCDR WBTC actor is malformed or zero")
            identity = (
                _canonical_nonnegative_int(row["block_number"], "block_number"),
                _canonical_nonnegative_int(
                    row["transaction_index"], "transaction_index"
                ),
                _canonical_nonnegative_int(
                    row["semantic_log_index"], "semantic_log_index"
                ),
            )
            if identity in identities:
                raise RuntimeError("WCDR WBTC source identity duplicated")
            identities.add(identity)
            event = SourceEvent(
                source="wbtc",
                event=event_name,
                sign=sign,
                amount_raw=_canonical_positive_int(row["amount_raw"], "amount_raw"),
                available_at=parse_time(row["available_at"]),
                identity=identity,
                actor=actor,
            )
            order_key = (event.available_at, event.identity)
            if previous is not None and order_key < previous:
                raise RuntimeError("WCDR WBTC source is not causally sorted")
            previous = order_key
            events.append(event)
    if len(events) != 993:
        raise RuntimeError("WCDR WBTC source row count drift")
    return events, {
        "physical_rows_read": len(events),
        "eligible_rows": len(events),
        "unique_identities": len(identities),
        "first_available_at": format_time(events[0].available_at),
        "last_available_at": format_time(events[-1].available_at),
    }


def load_usdc_events(
    path: str | Path = prereg.STABLECOIN_SOURCE,
    *,
    expected_rows_before_seal: int = EXPECTED_USDC_ROWS_BEFORE_SEAL,
    expected_eligible_rows: int = EXPECTED_USDC_ELIGIBLE_ROWS_BEFORE_SEAL,
) -> tuple[list[SourceEvent], dict[str, Any]]:
    events: list[SourceEvent] = []
    identities: set[tuple[int, int, int]] = set()
    rows_before_seal = 0
    boundary_sentinel_rows_scanned = 0
    previous: tuple[datetime, tuple[int, int, int]] | None = None
    sealed_from = parse_time(SOURCE_SEALED_FROM)
    with gzip.open(_path(path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != prereg.STABLECOIN_HEADER:
            raise RuntimeError("WCDR stablecoin source header drift")
        for row in reader:
            available_at = parse_time(row["available_at"])
            if available_at >= sealed_from:
                # The source is causally sorted. Inspect only the timestamp of the
                # first boundary sentinel, then stop before parsing any event value.
                boundary_sentinel_rows_scanned += 1
                break
            rows_before_seal += 1
            if row["asset"] != "usdc_eth" or row["event"] not in {"mint", "burn"}:
                continue
            event_name = row["event"]
            sign = int(row["event_sign"])
            if sign != (1 if event_name == "mint" else -1):
                raise RuntimeError("WCDR USDC event sign drift")
            identity = (
                _canonical_nonnegative_int(row["block_number"], "block_number"),
                _canonical_nonnegative_int(
                    row["transaction_index"], "transaction_index"
                ),
                _canonical_nonnegative_int(row["log_index"], "log_index"),
            )
            if identity in identities:
                raise RuntimeError("WCDR USDC source identity duplicated")
            identities.add(identity)
            event = SourceEvent(
                source="usdc",
                event=event_name,
                sign=sign,
                amount_raw=_canonical_positive_int(row["amount_raw"], "amount_raw"),
                available_at=available_at,
                identity=identity,
                actor=row["indexed_address_1"].lower(),
            )
            order_key = (event.available_at, event.identity)
            if previous is not None and order_key < previous:
                raise RuntimeError("WCDR USDC source is not causally sorted")
            previous = order_key
            events.append(event)
    if rows_before_seal != expected_rows_before_seal:
        raise RuntimeError("WCDR stablecoin source row count drift")
    if len(events) != expected_eligible_rows:
        raise RuntimeError("WCDR stablecoin eligible row count drift")
    if boundary_sentinel_rows_scanned != 1:
        raise RuntimeError("WCDR stablecoin sealed boundary sentinel drift")
    return events, {
        "physical_rows_read": rows_before_seal,
        "eligible_rows": len(events),
        "unique_identities": len(identities),
        "first_available_at": format_time(events[0].available_at),
        "last_available_at": format_time(events[-1].available_at),
        "sealed_from": SOURCE_SEALED_FROM,
        "boundary_sentinel_timestamp_rows_scanned": boundary_sentinel_rows_scanned,
        "post_2023_contract_event_value_rows_loaded": 0,
    }


def permute_amounts(events: Sequence[SourceEvent]) -> list[SourceEvent]:
    groups: dict[tuple[str, str, int], list[SourceEvent]] = defaultdict(list)
    for event in events:
        groups[(event.source, event.event, event.available_at.year)].append(event)
    output: list[SourceEvent] = []
    for key in sorted(groups):
        canonical = sorted(groups[key], key=lambda event: event.identity)
        donors = sorted(
            canonical,
            key=lambda event: hashlib.sha256(
                (
                    f"{CANDIDATE}|year_amount_permutation|{key}|{event.identity}"
                ).encode("ascii")
            ).digest(),
        )
        output.extend(
            replace(recipient, amount_raw=donor.amount_raw)
            for recipient, donor in zip(canonical, donors, strict=True)
        )
    return sorted(output, key=lambda event: (event.available_at, event.identity))


def daily_states(
    wbtc_index: EventIndex,
    usdc_index: EventIndex,
    *,
    extra_stale_days: int = 0,
) -> list[DailyState]:
    start = parse_time(WINDOWS["train"][0])
    end = parse_time(WINDOWS["selection"][1])
    output: list[DailyState] = []
    decision = start
    while decision < end:
        cutoff = decision - timedelta(hours=6, days=extra_stale_days)
        output.append(
            DailyState(
                decision_time=decision,
                source_cutoff=cutoff,
                wbtc=wbtc_index.aggregate(cutoff, 30, source="wbtc"),
                usdc=usdc_index.aggregate(cutoff, 7, source="usdc"),
            )
        )
        decision += timedelta(days=1)
    return output


def _sign(value: int) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def state_side(state: DailyState, control: str) -> int:
    wbtc_sign = _sign(state.wbtc.net_raw)
    usdc_sign = _sign(state.usdc.net_raw)
    if control in {"primary", "stale_7d", "year_amount_permutation"}:
        if not state.wbtc.valid or not state.usdc.valid:
            return 0
        if wbtc_sign == -1 and usdc_sign == 1:
            return 1
        if wbtc_sign == 1 and usdc_sign == -1:
            return -1
        return 0
    if control == "wbtc_only_contrarian":
        return -wbtc_sign if state.wbtc.valid else 0
    if control == "usdc_only_direct":
        return usdc_sign if state.usdc.valid else 0
    if control == "same_sign_direct":
        if state.wbtc.valid and state.usdc.valid and wbtc_sign == usdc_sign:
            return wbtc_sign
        return 0
    if control == "count_sign_consensus":
        primary = state_side(state, "primary")
        if primary == 0:
            return 0
        if _sign(state.wbtc.count_net) != wbtc_sign:
            return 0
        if _sign(state.usdc.count_net) != usdc_sign:
            return 0
        return primary
    raise KeyError(control)


def _contained_window(entry: datetime, exit_time: datetime) -> str | None:
    for name, (start_raw, end_raw) in WINDOWS.items():
        start, end = parse_time(start_raw), parse_time(end_raw)
        if start <= entry and exit_time <= end:
            return name
    return None


def schedule(states: Sequence[DailyState], control: str) -> list[Candidate]:
    output: list[Candidate] = []
    last_exit: datetime | None = None
    for state in states:
        side = state_side(state, control)
        if side == 0:
            continue
        entry = state.decision_time + timedelta(minutes=5)
        exit_time = entry + timedelta(minutes=5 * 2016)
        window = _contained_window(entry, exit_time)
        if window is None:
            continue
        if last_exit is not None and entry < last_exit:
            continue
        output.append(
            Candidate(
                control=control,
                decision_time=state.decision_time,
                source_cutoff=state.source_cutoff,
                entry_time=entry,
                exit_time=exit_time,
                side=side,
                window=window,
                wbtc=state.wbtc,
                usdc=state.usdc,
            )
        )
        last_exit = exit_time
    return output


def exact_clock_control(
    primary: Sequence[Candidate], control: str, sides: Sequence[int]
) -> list[Candidate]:
    if len(primary) != len(sides) or any(side not in {-1, 1} for side in sides):
        raise ValueError("WCDR exact-clock control sides are invalid")
    return [
        replace(candidate, control=control, side=side)
        for candidate, side in zip(primary, sides, strict=True)
    ]


def deterministic_random_sides(primary: Sequence[Candidate]) -> list[int]:
    original = sorted(candidate.side for candidate in primary)
    ranked_positions = sorted(
        range(len(primary)),
        key=lambda index: hashlib.sha256(
            (
                f"{CANDIDATE}|deterministic_random_side|"
                f"{format_time(primary[index].entry_time)}"
            ).encode("ascii")
        ).digest(),
    )
    assigned = [0] * len(primary)
    for side, position in zip(original, ranked_positions, strict=True):
        assigned[position] = side
    return assigned


def build_controls(
    wbtc_events: Sequence[SourceEvent], usdc_events: Sequence[SourceEvent]
) -> dict[str, list[Candidate]]:
    wbtc_index, usdc_index = EventIndex(wbtc_events), EventIndex(usdc_events)
    base_states = daily_states(wbtc_index, usdc_index)
    primary = schedule(base_states, "primary")
    controls: dict[str, list[Candidate]] = {
        "primary": primary,
        "direction_flip": exact_clock_control(
            primary, "direction_flip", [-candidate.side for candidate in primary]
        ),
        "wbtc_only_contrarian": schedule(base_states, "wbtc_only_contrarian"),
        "usdc_only_direct": schedule(base_states, "usdc_only_direct"),
        "same_sign_direct": schedule(base_states, "same_sign_direct"),
        "stale_7d": schedule(
            daily_states(wbtc_index, usdc_index, extra_stale_days=7), "stale_7d"
        ),
        "count_sign_consensus": schedule(base_states, "count_sign_consensus"),
    }
    permuted_states = daily_states(
        EventIndex(permute_amounts(wbtc_events)),
        EventIndex(permute_amounts(usdc_events)),
    )
    controls["year_amount_permutation"] = schedule(
        permuted_states, "year_amount_permutation"
    )
    controls["deterministic_random_side"] = exact_clock_control(
        primary,
        "deterministic_random_side",
        deterministic_random_sides(primary),
    )
    if tuple(controls) != CONTROL_ORDER:
        raise RuntimeError("WCDR control order drift")
    return controls


def _maximum_same_side_run(candidates: Sequence[Candidate]) -> int:
    maximum = current = 0
    previous = 0
    for candidate in sorted(candidates, key=lambda row: row.entry_time):
        if candidate.side == previous:
            current += 1
        else:
            previous = candidate.side
            current = 1
        maximum = max(maximum, current)
    return maximum


def _window_stats(candidates: Sequence[Candidate], window: str) -> dict[str, Any]:
    rows = [candidate for candidate in candidates if candidate.window == window]
    years = Counter(str(candidate.entry_time.year) for candidate in rows)
    halves = Counter(
        f"{candidate.entry_time.year}H{1 if candidate.entry_time.month <= 6 else 2}"
        for candidate in rows
    )
    sides = Counter("long" if candidate.side == 1 else "short" for candidate in rows)
    months = Counter(candidate.entry_time.strftime("%Y-%m") for candidate in rows)
    actors = sorted({actor for candidate in rows for actor in candidate.wbtc.actors})
    return {
        "trades": len(rows),
        "side_counts": {side: sides.get(side, 0) for side in ("long", "short")},
        "year_counts": dict(sorted(years.items())),
        "half_year_counts": dict(sorted(halves.items())),
        "month_counts": dict(sorted(months.items())),
        "maximum_month_share": max(months.values()) / len(rows) if rows else None,
        "maximum_consecutive_same_side": _maximum_same_side_run(rows),
        "distinct_wbtc_actors": len(actors),
        "wbtc_actor_set_hash": canonical_hash(actors),
    }


def support_statistics(primary: Sequence[Candidate]) -> dict[str, Any]:
    return {
        window: _window_stats(primary, window) for window in ("train", "selection")
    }


def support_checks(stats: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    train, selection = stats["train"], stats["selection"]
    train_years = train["year_counts"]
    train_halves = train["half_year_counts"]
    selection_halves = selection["half_year_counts"]
    return {
        "train_total_minimum": train["trades"] >= 50,
        "selection_total_minimum": selection["trades"] >= 20,
        "each_train_year_minimum": all(
            train_years.get(year, 0) >= 20 for year in ("2021", "2022")
        ),
        "each_train_half_year_minimum": all(
            train_halves.get(half, 0) >= 8
            for half in ("2021H1", "2021H2", "2022H1", "2022H2")
        ),
        "each_selection_half_year_minimum": all(
            selection_halves.get(half, 0) >= 8
            for half in ("2023H1", "2023H2")
        ),
        "train_each_side_minimum": all(
            train["side_counts"].get(side, 0) >= 12 for side in ("long", "short")
        ),
        "selection_each_side_minimum": all(
            selection["side_counts"].get(side, 0) >= 4
            for side in ("long", "short")
        ),
        "maximum_month_share": all(
            split["maximum_month_share"] is not None
            and split["maximum_month_share"] <= 0.20
            for split in (train, selection)
        ),
        "maximum_consecutive_same_side": all(
            split["maximum_consecutive_same_side"] <= 10
            for split in (train, selection)
        ),
        "train_distinct_wbtc_actors_minimum": train["distinct_wbtc_actors"] >= 10,
        "selection_distinct_wbtc_actors_minimum": (
            selection["distinct_wbtc_actors"] >= 5
        ),
    }


def exact_entry_jaccard(
    left: Sequence[Candidate], right: Sequence[Candidate]
) -> float:
    left_set = {candidate.entry_time for candidate in left}
    right_set = {candidate.entry_time for candidate in right}
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def candidate_row(candidate: Candidate) -> dict[str, Any]:
    return {
        "candidate": CANDIDATE,
        "control": candidate.control,
        "signal_id": candidate.signal_id,
        "window": candidate.window,
        "decision_time": format_time(candidate.decision_time),
        "source_cutoff": format_time(candidate.source_cutoff),
        "entry_time": format_time(candidate.entry_time),
        "exit_time": format_time(candidate.exit_time),
        "side": candidate.side,
        "wbtc_net_raw": candidate.wbtc.net_raw,
        "wbtc_gross_raw": candidate.wbtc.gross_raw,
        "wbtc_count_net": candidate.wbtc.count_net,
        "wbtc_rows": candidate.wbtc.rows,
        "wbtc_distinct_actors": len(candidate.wbtc.actors),
        "wbtc_top_actor_share": f"{candidate.wbtc.top_actor_share:.12f}",
        "usdc_net_raw": candidate.usdc.net_raw,
        "usdc_gross_raw": candidate.usdc.gross_raw,
        "usdc_count_net": candidate.usdc.count_net,
        "usdc_rows": candidate.usdc.rows,
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
    usdc_events, usdc_audit = load_usdc_events()
    controls = build_controls(wbtc_events, usdc_events)
    primary = controls["primary"]
    stats = support_statistics(primary)
    checks = support_checks(stats)
    passed = all(checks.values())
    rows = [
        candidate_row(candidate)
        for control in CONTROL_ORDER
        for candidate in controls[control]
    ]
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
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_artifact": str(PREREGISTRATION_ARTIFACT),
        "preregistration_artifact_sha256": PREREGISTRATION_ARTIFACT_SHA256,
        "preregistration_manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        "preregistration_reproduced": (
            registration["manifest_hash"] == PREREGISTRATION_MANIFEST_HASH
        ),
        "evaluator_source": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "source_audit": {
            "wbtc": wbtc_audit,
            "usdc": usdc_audit,
            "source_value_rows_read": (
                wbtc_audit["physical_rows_read"] + usdc_audit["physical_rows_read"]
            ),
            "post_2023_contract_event_rows_read": 0,
            "post_2023_contract_event_timestamp_sentinels_scanned": (
                usdc_audit["boundary_sentinel_timestamp_rows_scanned"]
            ),
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
            else "retire_WCDR_2016_without_repair"
        ),
        "advance_to_strict_evaluator_freeze": passed,
        "outcome_boundary": {
            "outcomes_opened": False,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_contract_event_rows_read": 0,
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
        raise FileExistsError(f"WCDR support artifact is write-once: {path}") from error


def write_support(
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
) -> dict[str, Any]:
    if _path(clock_output).exists() or _path(report_output).exists():
        raise FileExistsError("WCDR support outputs are write-once")
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
