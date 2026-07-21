"""Build the outcome-blind AMTR-48 source-support clock and verdict."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
from collections import Counter, deque
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_VERSION = "authorized_minter_turnaround_relay_support_v1"
POLICY_ID = "AMTR-48"
AS_OF_DATE = "2026-07-21"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MECHANISM_FREEZE = Path(
    "docs/authorized-minter-turnaround-relay-mechanism-freeze-2026-07-21.md"
)
MECHANISM_FREEZE_SHA256 = (
    "65be06b30bafb41ce81cf1f13664ec656905181ca8d4b623543964a931e94baa"
)
SOURCE_MANIFEST = Path(
    "results/ethereum_stablecoin_issuance_redemption_source_manifest_2026-07-21.json"
)
SOURCE_MANIFEST_SHA256 = (
    "8ec9ab08c413bf6f5f8170fb800b05105522d4cf1a7932943c214288701e31fe"
)
SOURCE_CSV = Path(
    "data/ethereum_stablecoin_issuance_redemption_2020_2023/"
    "ethereum_usdt_usdc_issuance_redemption_2020_2023.csv.gz"
)
SOURCE_CSV_SHA256 = "70ba3799ba84dc671051623a8d167b1731f043cf84a686b9878a67fcd52e5901"
EVALUATOR_SOURCE = Path(
    "training/evaluate_authorized_minter_turnaround_relay_support.py"
)
DEFAULT_CLOCK_OUTPUT = Path(
    "data/authorized_minter_turnaround_relay_clocks_2020_2023.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/authorized_minter_turnaround_relay_source_support_2026-07-21.json"
)

LOOKBACK = timedelta(days=365)
MINIMUM_HISTORY = 256
TAIL_QUANTILE = 0.95
MINIMUM_GAP = timedelta(minutes=30)
MAXIMUM_GAP = timedelta(hours=24)
STALE_DELAY = timedelta(hours=6)
HOLD = timedelta(hours=48)
CONTROL_NAMES = (
    "primary",
    "cross_minter",
    "no_amount_ratio",
    "no_minimum_gap",
    "stale_6h",
)
ADDRESS = re.compile(r"^0x[0-9a-f]{40}$")

FROZEN_CONFIG = {
    "lookback_days": 365,
    "minimum_history": MINIMUM_HISTORY,
    "tail_quantile": TAIL_QUANTILE,
    "tail_interpolation": "nearest_rank",
    "minimum_gap_minutes": 30,
    "maximum_gap_hours": 24,
    "minimum_amount_ratio": 0.50,
    "hold_hours": 48,
    "stale_control_hours": 6,
    "controls": list(CONTROL_NAMES),
}

OUTCOME_BOUNDARY = {
    "source_csv_rows_read": 266_362,
    "eligible_usdc_rows_read": 265_585,
    "post_2023_event_rows_read": 0,
    "comparator_clock_rows_read": 0,
    "btc_market_rows_read": 0,
    "funding_rows_read": 0,
    "future_return_rows_read": 0,
    "return_or_pnl_fields_read": 0,
    "network_calls": 0,
    "subprocess_calls": 0,
}

CLOCK_COLUMNS = (
    "candidate",
    "control",
    "pair_id",
    "side",
    "minter",
    "mint_to",
    "prior_event",
    "current_event",
    "prior_identity",
    "current_identity",
    "prior_amount_raw",
    "current_amount_raw",
    "availability_gap_seconds",
    "occurrence_gap_seconds",
    "pair_completion",
    "entry_time",
    "scheduled_exit",
)


@dataclass(frozen=True)
class Event:
    event: str
    amount_raw: int
    minter: str
    mint_to: str
    available_at: datetime
    block_timestamp: datetime
    block_number: int
    transaction_index: int
    log_index: int
    block_hash: str
    transaction_hash: str
    warmup_ready: bool = False
    large: bool = False

    @property
    def identity(self) -> str:
        return f"{self.block_hash}:{self.transaction_hash}:{self.log_index}"

    @property
    def identity_key(self) -> tuple[str, str, int]:
        return (self.block_hash, self.transaction_hash, self.log_index)

    @property
    def order_key(
        self,
    ) -> tuple[datetime, int, int, int, tuple[str, str, int]]:
        return (
            self.available_at,
            self.block_number,
            self.transaction_index,
            self.log_index,
            self.identity_key,
        )


@dataclass(frozen=True)
class Pair:
    prior: Event
    current: Event
    control: str

    @property
    def side(self) -> int:
        return 1 if (self.prior.event, self.current.event) == ("burn", "mint") else -1

    @property
    def completion(self) -> datetime:
        return self.current.available_at

    @property
    def minter(self) -> str:
        return (
            self.current.minter if self.control == "cross_minter" else self.prior.minter
        )

    @property
    def mint_to(self) -> str:
        mint = self.prior if self.prior.event == "mint" else self.current
        return mint.mint_to

    @property
    def pair_id(self) -> str:
        encoded = (
            f"{self.control}|{self.prior.identity}|{self.current.identity}".encode()
        )
        return hashlib.sha256(encoded).hexdigest()


class FenwickCounts:
    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("Fenwick size must be positive")
        self._tree = [0] * (size + 1)

    def add(self, index: int, delta: int) -> None:
        if not 1 <= index < len(self._tree):
            raise IndexError("Fenwick index out of range")
        while index < len(self._tree):
            self._tree[index] += delta
            if self._tree[index] < 0:
                raise RuntimeError("Fenwick count became negative")
            index += index & -index

    def kth(self, rank: int) -> int:
        total = self.prefix(len(self._tree) - 1)
        if not 1 <= rank <= total:
            raise ValueError("Fenwick rank out of range")
        index = 0
        bit = 1 << ((len(self._tree) - 1).bit_length() - 1)
        while bit:
            candidate = index + bit
            if candidate < len(self._tree) and self._tree[candidate] < rank:
                index = candidate
                rank -= self._tree[candidate]
            bit >>= 1
        return index + 1

    def prefix(self, index: int) -> int:
        total = 0
        while index > 0:
            total += self._tree[index]
            index -= index & -index
        return total


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("AMTR timestamp must be UTC")
    return parsed.astimezone(timezone.utc)


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def nearest_rank(values: Sequence[int], quantile: float) -> int:
    if not values:
        raise ValueError("nearest-rank sample must be non-empty")
    if not 0 < quantile <= 1:
        raise ValueError("nearest-rank quantile must be in (0, 1]")
    index = math.ceil(quantile * len(values)) - 1
    return values[index]


def _validate_address(value: str, field: str) -> str:
    normalized = value.lower()
    if not ADDRESS.fullmatch(normalized):
        raise ValueError(f"AMTR {field} is not a canonical address")
    return normalized


def load_source(path: str | Path = SOURCE_CSV) -> tuple[list[Event], int]:
    events: list[Event] = []
    total_rows = 0
    identities: set[str] = set()
    with gzip.open(_path(path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "asset",
            "event",
            "amount_raw",
            "indexed_address_1",
            "indexed_address_2",
            "block_number",
            "block_hash",
            "block_timestamp",
            "transaction_hash",
            "transaction_index",
            "log_index",
            "confirmation_block_number",
            "available_at",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError("AMTR source schema changed")
        for row in reader:
            total_rows += 1
            if row["asset"] != "usdc_eth" or row["event"] not in {"mint", "burn"}:
                continue
            event = row["event"]
            mint_to = (
                _validate_address(row["indexed_address_2"], "mint recipient")
                if event == "mint"
                else ""
            )
            parsed = Event(
                event=event,
                amount_raw=int(row["amount_raw"]),
                minter=_validate_address(row["indexed_address_1"], "minter"),
                mint_to=mint_to,
                available_at=_parse_time(row["available_at"]),
                block_timestamp=_parse_time(row["block_timestamp"]),
                block_number=int(row["block_number"]),
                transaction_index=int(row["transaction_index"]),
                log_index=int(row["log_index"]),
                block_hash=row["block_hash"],
                transaction_hash=row["transaction_hash"],
            )
            if parsed.amount_raw <= 0:
                raise RuntimeError("AMTR source contains a non-positive amount")
            if int(row["confirmation_block_number"]) != parsed.block_number + 64:
                raise RuntimeError("AMTR source confirmation delay changed")
            if parsed.available_at <= parsed.block_timestamp:
                raise RuntimeError("AMTR source violates causal availability")
            if parsed.identity in identities:
                raise RuntimeError("AMTR source contains a duplicate identity")
            identities.add(parsed.identity)
            events.append(parsed)
    events.sort(key=lambda event: event.order_key)
    return events, total_rows


def annotate_tail(events: Sequence[Event]) -> list[Event]:
    ordered = sorted(events, key=lambda event: event.order_key)
    coordinates = {
        event_type: sorted(
            {event.amount_raw for event in ordered if event.event == event_type}
        )
        for event_type in ("mint", "burn")
    }
    coordinate_index = {
        event_type: {
            amount: index + 1 for index, amount in enumerate(coordinates[event_type])
        }
        for event_type in ("mint", "burn")
    }
    queues: dict[str, deque[tuple[datetime, int]]] = {
        "mint": deque(),
        "burn": deque(),
    }
    counts = {
        event_type: FenwickCounts(max(1, len(coordinates[event_type])))
        for event_type in ("mint", "burn")
    }
    history_sizes = {"mint": 0, "burn": 0}
    annotated: list[Event] = []
    index = 0
    while index < len(ordered):
        timestamp = ordered[index].available_at
        end = index + 1
        while end < len(ordered) and ordered[end].available_at == timestamp:
            end += 1
        cutoff = timestamp - LOOKBACK
        for event_type in ("mint", "burn"):
            queue = queues[event_type]
            while queue and queue[0][0] < cutoff:
                _, amount = queue.popleft()
                counts[event_type].add(coordinate_index[event_type][amount], -1)
                history_sizes[event_type] -= 1
        group: list[Event] = []
        for event in ordered[index:end]:
            history_size = history_sizes[event.event]
            ready = history_size >= MINIMUM_HISTORY
            if ready:
                rank = math.ceil(TAIL_QUANTILE * history_size)
                threshold_index = counts[event.event].kth(rank)
                threshold = coordinates[event.event][threshold_index - 1]
            else:
                threshold = None
            group.append(
                replace(
                    event,
                    warmup_ready=ready,
                    large=ready
                    and threshold is not None
                    and event.amount_raw >= threshold,
                )
            )
        annotated.extend(group)
        for event in ordered[index:end]:
            queues[event.event].append((event.available_at, event.amount_raw))
            counts[event.event].add(coordinate_index[event.event][event.amount_raw], 1)
            history_sizes[event.event] += 1
        index = end
    return annotated


def _event_eligible(event: Event, *, use_tail: bool) -> bool:
    return event.large if use_tail else event.warmup_ready


def build_pairs(
    events: Sequence[Event],
    *,
    control: str,
    use_tail: bool = True,
    same_minter: bool = True,
    require_ratio: bool = True,
    minimum_gap: timedelta = MINIMUM_GAP,
) -> list[Pair]:
    if control not in CONTROL_NAMES:
        raise ValueError("unknown AMTR control")
    unmatched: dict[str, list[Event]] = {"mint": [], "burn": []}
    consumed: set[str] = set()
    pairs: list[Pair] = []
    for current in sorted(events, key=lambda event: event.order_key):
        if not _event_eligible(current, use_tail=use_tail):
            continue
        opposite = "burn" if current.event == "mint" else "mint"
        candidates = unmatched[opposite]
        candidates[:] = [
            event
            for event in candidates
            if event.identity not in consumed
            and current.available_at - event.available_at <= MAXIMUM_GAP
        ]
        qualifying: list[Event] = []
        latest_time: datetime | None = None
        for prior in reversed(candidates):
            available_gap = current.available_at - prior.available_at
            if available_gap > MAXIMUM_GAP:
                break
            if available_gap < minimum_gap:
                continue
            occurrence_gap = current.block_timestamp - prior.block_timestamp
            if not minimum_gap <= occurrence_gap <= MAXIMUM_GAP:
                continue
            if same_minter != (prior.minter == current.minter):
                continue
            if require_ratio and min(prior.amount_raw, current.amount_raw) * 2 < max(
                prior.amount_raw, current.amount_raw
            ):
                continue
            if latest_time is None:
                latest_time = prior.available_at
            if prior.available_at != latest_time:
                break
            qualifying.append(prior)
        if qualifying:
            prior = min(qualifying, key=lambda event: event.identity_key)
            pair = Pair(prior=prior, current=current, control=control)
            consumed.update((prior.identity, current.identity))
            pairs.append(pair)
        else:
            unmatched[current.event].append(current)
    return pairs


def _entry_time(completion: datetime) -> datetime:
    epoch = math.ceil(completion.timestamp())
    boundary = ((epoch + 299) // 300) * 300
    return datetime.fromtimestamp(boundary + 300, tz=timezone.utc)


def schedule_pairs(
    pairs: Sequence[Pair], *, delay: timedelta = timedelta(0)
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_exit: datetime | None = None
    prepared = [(_entry_time(pair.completion + delay), pair) for pair in pairs]
    prepared.sort(
        key=lambda item: (
            item[0],
            item[1].prior.identity_key,
            item[1].current.identity_key,
        )
    )
    for entry, pair in prepared:
        completion = pair.completion + delay
        scheduled_exit = entry + HOLD
        if previous_exit is not None and entry < previous_exit:
            continue
        available_gap = int(
            (pair.current.available_at - pair.prior.available_at).total_seconds()
        )
        occurrence_gap = int(
            (pair.current.block_timestamp - pair.prior.block_timestamp).total_seconds()
        )
        rows.append(
            {
                "candidate": POLICY_ID,
                "control": pair.control,
                "pair_id": pair.pair_id,
                "side": pair.side,
                "minter": pair.minter,
                "mint_to": pair.mint_to,
                "prior_event": pair.prior.event,
                "current_event": pair.current.event,
                "prior_identity": pair.prior.identity,
                "current_identity": pair.current.identity,
                "prior_amount_raw": pair.prior.amount_raw,
                "current_amount_raw": pair.current.amount_raw,
                "availability_gap_seconds": available_gap,
                "occurrence_gap_seconds": occurrence_gap,
                "pair_completion": _format_time(completion),
                "entry_time": _format_time(entry),
                "scheduled_exit": _format_time(scheduled_exit),
            }
        )
        previous_exit = scheduled_exit
    return rows


def _maximum_share(values: Iterable[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    return max(counts.values(), default=0) / total if total else 0.0


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    side_counts = Counter(int(row["side"]) for row in rows)
    year_counts = Counter(str(row["entry_time"])[:4] for row in rows)
    month_counts = Counter(str(row["entry_time"])[:7] for row in rows)
    minters = [str(row["minter"]) for row in rows]
    side_minter_share = {
        side: _maximum_share(
            str(row["minter"]) for row in rows if int(row["side"]) == sign
        )
        for side, sign in (("long", 1), ("short", -1))
    }
    summary = {
        "events": count,
        "longs": side_counts[1],
        "shorts": side_counts[-1],
        "long_share": side_counts[1] / count if count else 0.0,
        "short_share": side_counts[-1] / count if count else 0.0,
        "year_counts": dict(sorted(year_counts.items())),
        "month_counts": dict(sorted(month_counts.items())),
        "maximum_month_share": max(month_counts.values(), default=0) / count
        if count
        else 0.0,
        "distinct_minters": len(set(minters)),
        "maximum_minter_share": _maximum_share(minters),
        "maximum_minter_share_by_side": side_minter_share,
        "maximum_mint_recipient_share": _maximum_share(
            str(row["mint_to"]) for row in rows
        ),
    }
    summary["checks"] = {
        "minimum_events": count >= 60,
        "year_2021": year_counts["2021"] >= 12,
        "year_2022": year_counts["2022"] >= 12,
        "year_2023": year_counts["2023"] >= 12,
        "side_balance": (
            summary["long_share"] >= 0.30 and summary["short_share"] >= 0.30
        ),
        "actor_breadth": summary["distinct_minters"] >= 5,
        "actor_concentration": summary["maximum_minter_share"] <= 0.40,
        "side_actor_concentration": all(
            share <= 0.60 for share in side_minter_share.values()
        ),
        "mint_recipient_concentration": (
            summary["maximum_mint_recipient_share"] <= 0.50
        ),
        "month_concentration": summary["maximum_month_share"] <= 0.20,
    }
    return summary


def _write_clock(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows, key=lambda row: (str(row["control"]), str(row["entry_time"]))
    )
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(CLOCK_COLUMNS)
                writer.writerows(
                    [[row[column] for column in CLOCK_COLUMNS] for row in ordered]
                )


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("AMTR expected a JSON object")
    return payload


def _validate_inputs() -> dict[str, Any]:
    if sha256_file(MECHANISM_FREEZE) != MECHANISM_FREEZE_SHA256:
        raise RuntimeError("AMTR mechanism freeze hash mismatch")
    if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("AMTR source manifest file hash mismatch")
    if sha256_file(SOURCE_CSV) != SOURCE_CSV_SHA256:
        raise RuntimeError("AMTR source CSV hash mismatch")
    manifest = _read_json(SOURCE_MANIFEST)
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if manifest.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("AMTR source manifest hash mismatch")
    if manifest.get("output", {}).get("sha256") != SOURCE_CSV_SHA256:
        raise RuntimeError("AMTR source manifest output binding drift")
    _validate_source_contract(manifest)
    return manifest


def _validate_source_contract(manifest: Mapping[str, Any]) -> None:
    if manifest.get("dual_replay", {}).get("canonical_replay_equal") is not True:
        raise RuntimeError("AMTR source lacks dual replay")
    if (
        manifest.get("header_materialization", {}).get("event_block_hash_cross_checked")
        is not True
    ):
        raise RuntimeError("AMTR source lacks header cross-check")
    boundary = manifest.get("outcome_boundary")
    if not isinstance(boundary, dict) or boundary.get("source_only") is not True:
        raise RuntimeError("AMTR source is not outcome-blind")
    source_contract = manifest.get("source_contract")
    if (
        not isinstance(source_contract, dict)
        or source_contract.get("confirmation_blocks") != 64
    ):
        raise RuntimeError("AMTR source N+64 contract drift")
    source_audit = manifest.get("source_audit")
    if not isinstance(source_audit, dict):
        raise RuntimeError("AMTR source audit missing")
    coverage = source_audit.get("finalized_coverage")
    if (
        not isinstance(coverage, dict)
        or coverage.get("observed_finalized_block_at_least_required") is not True
        or not isinstance(coverage.get("required_through_block"), int)
        or int(coverage["required_through_block"]) <= 0
    ):
        raise RuntimeError("AMTR source finalized coverage drift")


def build_support() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_manifest = _validate_inputs()
    events, total_rows = load_source()
    if total_rows != OUTCOME_BOUNDARY["source_csv_rows_read"]:
        raise RuntimeError("AMTR source row count drift")
    if len(events) != OUTCOME_BOUNDARY["eligible_usdc_rows_read"]:
        raise RuntimeError("AMTR eligible USDC row count drift")
    annotated = annotate_tail(events)

    primary_pairs = build_pairs(annotated, control="primary")
    clocks: dict[str, list[dict[str, Any]]] = {
        "primary": schedule_pairs(primary_pairs),
        "cross_minter": schedule_pairs(
            build_pairs(
                annotated,
                control="cross_minter",
                same_minter=False,
            )
        ),
        "no_amount_ratio": schedule_pairs(
            build_pairs(
                annotated,
                control="no_amount_ratio",
                require_ratio=False,
            )
        ),
        "no_minimum_gap": schedule_pairs(
            build_pairs(
                annotated,
                control="no_minimum_gap",
                minimum_gap=timedelta(0),
            )
        ),
        "stale_6h": schedule_pairs(
            [replace(pair, control="stale_6h") for pair in primary_pairs],
            delay=STALE_DELAY,
        ),
    }
    if tuple(clocks) != CONTROL_NAMES:
        raise RuntimeError("AMTR control set drift")
    primary_summary = summarize(clocks["primary"])
    support_pass = all(primary_summary["checks"].values())
    failed_gates = sorted(
        key for key, passed in primary_summary["checks"].items() if not passed
    )
    all_rows = [row for control in CONTROL_NAMES for row in clocks[control]]
    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "mechanism_freeze": {
            "path": str(MECHANISM_FREEZE),
            "sha256": MECHANISM_FREEZE_SHA256,
        },
        "source": {
            "manifest_path": str(SOURCE_MANIFEST),
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "manifest_hash": source_manifest["manifest_hash"],
            "csv_path": str(SOURCE_CSV),
            "csv_sha256": SOURCE_CSV_SHA256,
            "rows": total_rows,
            "eligible_usdc_rows": len(events),
        },
        "evaluator": {
            "path": str(EVALUATOR_SOURCE),
            "sha256": sha256_file(EVALUATOR_SOURCE),
        },
        "frozen_config": dict(FROZEN_CONFIG),
        "primary_support": primary_summary,
        "control_support": {
            control: summarize(clocks[control])
            for control in CONTROL_NAMES
            if control != "primary"
        },
        "decision": {
            "status": (
                "source_support_pass_novelty_pending"
                if support_pass
                else "retired_before_novelty"
            ),
            "source_support_pass": support_pass,
            "failed_gates": failed_gates,
            "novelty_opened": False,
            "economic_outcomes_opened": False,
            "repair_authorized": False,
        },
        "authorization": {
            "novelty_evaluator": support_pass,
            "outcome_evaluator": False,
            "post_2023_event_access": False,
            "next_action": (
                "open only checksum-bound comparator timestamps"
                if support_pass
                else "new independently frozen alpha only"
            ),
        },
        "outcomes_opened": False,
        "outcome_boundary": dict(OUTCOME_BOUNDARY),
    }
    return all_rows, report


def validate_report(payload: Mapping[str, Any], *, verify_files: bool = True) -> None:
    required = {
        "protocol_version",
        "policy_id",
        "as_of_date",
        "mechanism_freeze",
        "source",
        "evaluator",
        "frozen_config",
        "primary_support",
        "control_support",
        "decision",
        "authorization",
        "outcomes_opened",
        "outcome_boundary",
        "clock_output",
        "manifest_hash",
    }
    if set(payload) != required:
        raise RuntimeError("AMTR support report schema drift")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("AMTR support protocol drift")
    if payload.get("policy_id") != POLICY_ID or payload.get("as_of_date") != AS_OF_DATE:
        raise RuntimeError("AMTR support identity drift")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("AMTR support manifest hash mismatch")
    if payload.get("frozen_config") != FROZEN_CONFIG:
        raise RuntimeError("AMTR support config drift")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("AMTR support opened outcomes")
    if payload.get("outcome_boundary") != OUTCOME_BOUNDARY:
        raise RuntimeError("AMTR support outcome boundary drift")
    primary = payload.get("primary_support")
    if not isinstance(primary, dict) or not isinstance(primary.get("checks"), dict):
        raise RuntimeError("AMTR primary support missing")
    support_pass = all(primary["checks"].values())
    failed = sorted(key for key, passed in primary["checks"].items() if not passed)
    decision = payload.get("decision")
    if not isinstance(decision, dict) or decision != {
        "status": (
            "source_support_pass_novelty_pending"
            if support_pass
            else "retired_before_novelty"
        ),
        "source_support_pass": support_pass,
        "failed_gates": failed,
        "novelty_opened": False,
        "economic_outcomes_opened": False,
        "repair_authorized": False,
    }:
        raise RuntimeError("AMTR support decision drift")
    authorization = payload.get("authorization")
    if not isinstance(authorization, dict) or authorization != {
        "novelty_evaluator": support_pass,
        "outcome_evaluator": False,
        "post_2023_event_access": False,
        "next_action": (
            "open only checksum-bound comparator timestamps"
            if support_pass
            else "new independently frozen alpha only"
        ),
    }:
        raise RuntimeError("AMTR support authorization drift")
    controls = payload.get("control_support")
    if not isinstance(controls, dict) or set(controls) != set(CONTROL_NAMES) - {
        "primary"
    }:
        raise RuntimeError("AMTR support control set drift")
    clock = payload.get("clock_output")
    if not isinstance(clock, dict) or clock.get("columns") != list(CLOCK_COLUMNS):
        raise RuntimeError("AMTR support clock schema drift")
    expected_rows = int(primary["events"]) + sum(
        int(summary["events"])
        for summary in controls.values()
        if isinstance(summary, dict)
    )
    if clock.get("rows") != expected_rows:
        raise RuntimeError("AMTR support clock row count drift")
    if verify_files:
        if sha256_file(MECHANISM_FREEZE) != MECHANISM_FREEZE_SHA256:
            raise RuntimeError("AMTR mechanism freeze changed")
        if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
            raise RuntimeError("AMTR source manifest changed")
        if sha256_file(SOURCE_CSV) != SOURCE_CSV_SHA256:
            raise RuntimeError("AMTR source CSV changed")
        evaluator = payload.get("evaluator")
        if not isinstance(evaluator, dict) or sha256_file(
            evaluator["path"]
        ) != evaluator.get("sha256"):
            raise RuntimeError("AMTR evaluator changed")
        if sha256_file(clock["path"]) != clock.get("sha256"):
            raise RuntimeError("AMTR clock changed")


def run(
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
) -> dict[str, Any]:
    rows, report = build_support()
    clock_path = _path(clock_output)
    _write_clock(clock_path, rows)
    report["clock_output"] = {
        "path": str(Path(clock_output)),
        "rows": len(rows),
        "columns": list(CLOCK_COLUMNS),
        "bytes": clock_path.stat().st_size,
        "sha256": sha256_file(clock_path),
    }
    report["manifest_hash"] = canonical_hash(report)
    report_path = _path(report_output)
    encoded = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if report_path.exists() and report_path.read_text(encoding="utf-8") != encoded:
        raise FileExistsError("existing AMTR support report differs")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(encoded, encoding="utf-8")
    validate_report(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK_OUTPUT))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(run(args.clock_output, args.report_output), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
