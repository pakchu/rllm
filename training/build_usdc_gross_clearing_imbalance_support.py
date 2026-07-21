"""Build the frozen UGCI-288 source clock and outcome-blind support verdict.

The evaluator reads the promoted Ethereum event ledger.  Comparator timestamps
are opened only if every primary incidence gate passes.  It never loads BTC
market data, funding, returns, labels, PnL, or any post-2023 contract event.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import io
import json
import math
import re
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from training import preregister_usdc_gross_clearing_imbalance as prereg


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "usdc_gross_clearing_imbalance_support_v1"
CANDIDATE = prereg.CANDIDATE
AS_OF_DATE = prereg.AS_OF_DATE
PREREGISTRATION_COMMIT = "768342ea54e5ce8c7e94f8485955c0fedd1aada9"
PREREGISTRATION_ARTIFACT = Path(
    "results/usdc_gross_clearing_imbalance_preregistration_2026-07-22.json"
)
PREREGISTRATION_ARTIFACT_SHA256 = (
    "7056eadfd5b347b8b9afbe06cbc2a33f832a2913dc3227891a2a8d211aaa454a"
)
PREREGISTRATION_MANIFEST_HASH = (
    "61b6d60f8c2ef21b94b3343bc3cf2a5fd82366679ae9d768d846831b12829722"
)
EVALUATOR_SOURCE = Path("training/build_usdc_gross_clearing_imbalance_support.py")
DEFAULT_CLOCK_OUTPUT = Path(
    "data/usdc_gross_clearing_imbalance_clocks_2021_2023.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/usdc_gross_clearing_imbalance_support_2026-07-22.json"
)
SEALED_COMPARATOR_CLOCK = Path(
    "results/ugci_prior_comparator_views_pre2024_2026-07-22.csv.gz"
)
SEALED_COMPARATOR_CLOCK_SHA256 = (
    "dfbf4808813c1b0db4c5a4f05af324473d3a92dfa5cdfc6581e1b07bc17271bd"
)
SEALED_COMPARATOR_MANIFEST = Path(
    "results/ugci_prior_comparator_views_pre2024_manifest_2026-07-22.json"
)
SEALED_COMPARATOR_MANIFEST_SHA256 = (
    "38abf60a8c9aa44c7fb53a5435f22cb650151b58e33a6fca1ffae1aeb36ed5c2"
)
SEALED_COMPARATOR_MANIFEST_HASH = (
    "a00301a229bc1c620f355cb42adc05b760d8734d7a903490ab2c1d3a0fd92d33"
)
SEALED_COMPARATOR_COLUMNS = (
    "candidate",
    "control",
    "entry_time",
    "comparison_start",
    "comparison_end_exclusive",
)

EXPECTED_SOURCE_COLUMNS = (
    "asset",
    "contract_address",
    "event",
    "event_sign",
    "amount_raw",
    "decimals",
    "indexed_address_1",
    "indexed_address_2",
    "data_address",
    "block_number",
    "block_hash",
    "block_timestamp",
    "transaction_hash",
    "transaction_index",
    "log_index",
    "confirmation_block_number",
    "confirmation_block_hash",
    "available_at",
)
EXPECTED_SOURCE_ROWS = 266_362
EXPECTED_ELIGIBLE_ROWS = 265_585
UTC = timezone.utc
HEX_32 = re.compile(r"^0x[0-9a-f]{64}$")

CLOCK_COLUMNS = (
    "candidate",
    "control",
    "signal_id",
    "source_packet_start",
    "source_packet_end",
    "feature_available_time",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
    "mint_raw",
    "burn_raw",
    "gross_raw",
    "net_raw",
    "imbalance_ratio",
    "prior_gross_q95",
    "prior_history_packets",
)


@dataclass(frozen=True)
class SourceEvent:
    event: str
    amount_raw: int
    block_timestamp: datetime
    available_at: datetime
    block_number: int
    transaction_index: int
    log_index: int
    block_hash: str
    transaction_hash: str

    @property
    def identity(self) -> tuple[str, str, int]:
        return self.block_hash, self.transaction_hash, self.log_index


@dataclass(frozen=True)
class Packet:
    start: datetime
    end: datetime
    mint_raw: int
    burn_raw: int
    prior_gross_q95: int | None = None
    prior_history_packets: int = 0

    @property
    def gross_raw(self) -> int:
        return self.mint_raw + self.burn_raw

    @property
    def net_raw(self) -> int:
        return self.mint_raw - self.burn_raw

    @property
    def imbalance_ratio(self) -> float:
        if self.gross_raw == 0:
            return 0.0
        return abs(self.net_raw) / self.gross_raw

    @property
    def side(self) -> int:
        return 1 if self.net_raw > 0 else -1 if self.net_raw < 0 else 0


@dataclass(frozen=True)
class Signal:
    control: str
    packet: Packet
    decision_time: datetime
    entry_time: datetime
    exit_time: datetime

    @property
    def signal_id(self) -> str:
        key = (
            f"{CANDIDATE}|{self.control}|{format_time(self.packet.start)}|"
            f"{self.packet.side}"
        ).encode("ascii")
        return hashlib.sha256(key).hexdigest()


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
        raise ValueError("UGCI-288 timestamp must be UTC")
    return parsed.astimezone(UTC)


def format_time(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def floor_packet(value: datetime, packet_hours: int = 6) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("UGCI-288 packet timestamp must be UTC")
    hour = value.hour - value.hour % packet_hours
    return value.replace(hour=hour, minute=0, second=0, microsecond=0)


def nearest_rank(values: Sequence[int], quantile: float) -> int:
    if not values:
        raise ValueError("UGCI-288 nearest-rank sample is empty")
    if not 0 < quantile <= 1:
        raise ValueError("UGCI-288 nearest-rank quantile is invalid")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def validate_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION_ARTIFACT) != PREREGISTRATION_ARTIFACT_SHA256:
        raise ValueError("UGCI-288 preregistration artifact hash mismatch")
    payload = json.loads(_path(PREREGISTRATION_ARTIFACT).read_text(encoding="utf-8"))
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise ValueError("UGCI-288 preregistration manifest mismatch")
    if payload.get("candidate") != CANDIDATE:
        raise ValueError("UGCI-288 preregistration identity changed")
    frozen_config = json.loads(
        json.dumps(asdict(prereg.FROZEN_CONFIG), sort_keys=True, allow_nan=False)
    )
    if payload.get("policy", {}).get("config") != frozen_config:
        raise ValueError("UGCI-288 preregistration policy changed")
    if payload.get("support_gate") != prereg.SUPPORT_GATES:
        raise ValueError("UGCI-288 support gates changed")
    boundary = payload.get("outcome_boundary", {})
    if any(boundary.values()):
        raise ValueError("UGCI-288 preregistration opened an outcome")
    return payload


def validate_source_inputs() -> dict[str, Any]:
    if sha256_file(prereg.SOURCE_CSV) != prereg.SOURCE_CSV_SHA256:
        raise ValueError("UGCI-288 source CSV hash mismatch")
    if sha256_file(prereg.SOURCE_MANIFEST) != prereg.SOURCE_MANIFEST_SHA256:
        raise ValueError("UGCI-288 source manifest hash mismatch")
    manifest = json.loads(_path(prereg.SOURCE_MANIFEST).read_text(encoding="utf-8"))
    output = manifest.get("output", {})
    boundary = manifest.get("outcome_boundary", {})
    if output.get("sha256") != prereg.SOURCE_CSV_SHA256:
        raise ValueError("UGCI-288 manifest source binding changed")
    if output.get("rows") != EXPECTED_SOURCE_ROWS:
        raise ValueError("UGCI-288 source row count changed")
    if boundary.get("source_only") is not True:
        raise ValueError("UGCI-288 source is not source-only")
    if boundary.get("pnl_cagr_mdd_opened") is not False:
        raise ValueError("UGCI-288 source opened economic outcomes")
    for field in (
        "btc_market_rows_read",
        "funding_rows_read",
        "future_return_rows_read",
        "post_2023_contract_event_rows_read",
    ):
        if boundary.get(field) != 0:
            raise ValueError(f"UGCI-288 source manifest violates {field}")
    return manifest


def _parse_nonnegative_int(value: str, field: str) -> int:
    if not value or not value.isdigit():
        raise ValueError(f"UGCI-288 {field} is not a canonical nonnegative integer")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"UGCI-288 {field} is negative")
    return parsed


def load_source_events(
    path: str | Path = prereg.SOURCE_CSV,
) -> tuple[list[SourceEvent], dict[str, Any]]:
    events: list[SourceEvent] = []
    total_rows = 0
    identities: set[tuple[str, str, int]] = set()
    first_block_time: datetime | None = None
    last_block_time: datetime | None = None
    first_available: datetime | None = None
    last_available: datetime | None = None
    with gzip.open(_path(path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_SOURCE_COLUMNS:
            raise ValueError("UGCI-288 source schema changed")
        for row in reader:
            total_rows += 1
            block_time = parse_time(row["block_timestamp"])
            available = parse_time(row["available_at"])
            if first_block_time is None:
                first_block_time = block_time
                first_available = available
            last_block_time = block_time
            last_available = available
            if block_time >= parse_time(prereg.FROZEN_CONFIG.selection_end_exclusive):
                raise ValueError("UGCI-288 source contains a post-2023 contract event")
            if available < block_time:
                raise ValueError("UGCI-288 event became available before occurrence")
            if row["asset"] != prereg.FROZEN_CONFIG.asset or row["event"] not in {
                "mint",
                "burn",
            }:
                continue
            if row["decimals"] != "6":
                raise ValueError("UGCI-288 USDC decimals changed")
            block_hash = row["block_hash"].lower()
            transaction_hash = row["transaction_hash"].lower()
            if not HEX_32.fullmatch(block_hash) or not HEX_32.fullmatch(
                transaction_hash
            ):
                raise ValueError("UGCI-288 source hash is malformed")
            event = SourceEvent(
                event=row["event"],
                amount_raw=_parse_nonnegative_int(row["amount_raw"], "amount_raw"),
                block_timestamp=block_time,
                available_at=available,
                block_number=_parse_nonnegative_int(
                    row["block_number"], "block_number"
                ),
                transaction_index=_parse_nonnegative_int(
                    row["transaction_index"], "transaction_index"
                ),
                log_index=_parse_nonnegative_int(row["log_index"], "log_index"),
                block_hash=block_hash,
                transaction_hash=transaction_hash,
            )
            if event.identity in identities:
                raise ValueError("UGCI-288 duplicate source identity")
            identities.add(event.identity)
            events.append(event)
    if total_rows != EXPECTED_SOURCE_ROWS:
        raise ValueError("UGCI-288 source physical row count changed")
    if len(events) != EXPECTED_ELIGIBLE_ROWS:
        raise ValueError("UGCI-288 eligible USDC row count changed")
    if events != sorted(
        events,
        key=lambda event: (
            event.available_at,
            event.block_number,
            event.transaction_index,
            event.log_index,
            event.identity,
        ),
    ):
        raise ValueError("UGCI-288 eligible events are not causally ordered")
    return events, {
        "source_rows_read": total_rows,
        "eligible_usdc_rows": len(events),
        "unique_eligible_identities": len(identities),
        "first_block_timestamp": format_time(first_block_time)
        if first_block_time
        else None,
        "last_block_timestamp": format_time(last_block_time)
        if last_block_time
        else None,
        "first_available_at": format_time(first_available) if first_available else None,
        "last_available_at": format_time(last_available) if last_available else None,
    }


def build_packets(
    events: Iterable[SourceEvent],
    cfg: prereg.PolicyConfig = prereg.FROZEN_CONFIG,
) -> list[Packet]:
    prereg._validate_config(cfg)
    start = parse_time(cfg.warmup_start)
    end = parse_time(cfg.selection_end_exclusive)
    step = timedelta(hours=cfg.packet_hours)
    packet_sums: dict[datetime, list[int]] = {}
    for event in events:
        if not start <= event.available_at < end:
            continue
        packet_start = floor_packet(event.available_at, cfg.packet_hours)
        totals = packet_sums.setdefault(packet_start, [0, 0])
        totals[0 if event.event == "mint" else 1] += event.amount_raw
    packets: list[Packet] = []
    cursor = start
    while cursor < end:
        mint_raw, burn_raw = packet_sums.get(cursor, [0, 0])
        packets.append(
            Packet(
                start=cursor,
                end=cursor + step,
                mint_raw=mint_raw,
                burn_raw=burn_raw,
            )
        )
        cursor += step
    expected = int((end - start) / step)
    if len(packets) != expected or any(
        packet.start != start + index * step for index, packet in enumerate(packets)
    ):
        raise ValueError("UGCI-288 packet grid is incomplete")
    return packets


def attach_prior_thresholds(
    packets: Sequence[Packet],
    cfg: prereg.PolicyConfig = prereg.FROZEN_CONFIG,
) -> list[Packet]:
    prereg._validate_config(cfg)
    history: deque[Packet] = deque()
    lookback = timedelta(days=cfg.lookback_days)
    enriched: list[Packet] = []
    for packet in packets:
        while history and history[0].start < packet.start - lookback:
            history.popleft()
        threshold = None
        if len(history) >= cfg.minimum_history_packets:
            threshold = nearest_rank(
                [prior.gross_raw for prior in history], cfg.gross_tail_quantile
            )
        enriched.append(
            Packet(
                start=packet.start,
                end=packet.end,
                mint_raw=packet.mint_raw,
                burn_raw=packet.burn_raw,
                prior_gross_q95=threshold,
                prior_history_packets=len(history),
            )
        )
        history.append(packet)
    return enriched


def primary_active(packet: Packet, cfg: prereg.PolicyConfig) -> bool:
    return bool(
        packet.prior_gross_q95 is not None
        and packet.gross_raw > 0
        and packet.gross_raw >= packet.prior_gross_q95
        and packet.net_raw != 0
        and packet.imbalance_ratio >= cfg.minimum_imbalance_ratio
    )


def control_active(packet: Packet, control: str, cfg: prereg.PolicyConfig) -> bool:
    history_ready = packet.prior_gross_q95 is not None
    if control == "primary":
        return primary_active(packet, cfg)
    if control == "no_gross_tail":
        return bool(
            history_ready
            and packet.gross_raw > 0
            and packet.net_raw != 0
            and packet.imbalance_ratio >= cfg.minimum_imbalance_ratio
        )
    if control == "no_imbalance_floor":
        return bool(
            history_ready
            and packet.gross_raw > 0
            and packet.gross_raw >= int(packet.prior_gross_q95 or 0)
            and packet.net_raw != 0
        )
    raise KeyError(control)


def schedule_signals(
    packets: Sequence[Packet],
    control: str,
    cfg: prereg.PolicyConfig = prereg.FROZEN_CONFIG,
) -> list[Signal]:
    prereg._validate_config(cfg)
    if control not in {"primary", *prereg.SOURCE_ONLY_CONTROLS}:
        raise KeyError(control)
    start = parse_time(cfg.train_start)
    end = parse_time(cfg.selection_end_exclusive)
    if control == "stale_6h":
        delay = timedelta(hours=cfg.packet_hours)
        return [
            Signal(
                control=control,
                packet=signal.packet,
                decision_time=signal.decision_time + delay,
                entry_time=signal.entry_time + delay,
                exit_time=signal.exit_time + delay,
            )
            for signal in schedule_signals(packets, "primary", cfg)
            if signal.exit_time + delay <= end
        ]
    entry_delay = timedelta(minutes=cfg.entry_delay_minutes)
    hold = timedelta(minutes=cfg.hold_bars * cfg.bar_minutes)
    last_exit: datetime | None = None
    signals: list[Signal] = []
    for packet in packets:
        if not control_active(packet, control, cfg):
            continue
        decision = packet.end
        entry = packet.end + entry_delay
        exit_time = entry + hold
        if entry < start or entry >= end or exit_time > end:
            continue
        if last_exit is not None and entry < last_exit:
            continue
        signal = Signal(
            control=control,
            packet=packet,
            decision_time=decision,
            entry_time=entry,
            exit_time=exit_time,
        )
        signals.append(signal)
        last_exit = exit_time
    return signals


def signal_row(signal: Signal) -> dict[str, Any]:
    packet = signal.packet
    return {
        "candidate": CANDIDATE,
        "control": signal.control,
        "signal_id": signal.signal_id,
        "source_packet_start": format_time(packet.start),
        "source_packet_end": format_time(packet.end),
        "feature_available_time": format_time(packet.end),
        "decision_time": format_time(signal.decision_time),
        "entry_time": format_time(signal.entry_time),
        "exit_time": format_time(signal.exit_time),
        "side": packet.side,
        "mint_raw": packet.mint_raw,
        "burn_raw": packet.burn_raw,
        "gross_raw": packet.gross_raw,
        "net_raw": packet.net_raw,
        "imbalance_ratio": f"{packet.imbalance_ratio:.12f}",
        "prior_gross_q95": packet.prior_gross_q95,
        "prior_history_packets": packet.prior_history_packets,
    }


def _signals_in_window(
    signals: Iterable[Signal], start: datetime, end: datetime
) -> list[Signal]:
    return [
        signal
        for signal in signals
        if start <= signal.entry_time and signal.exit_time <= end
    ]


def _window_stats(signals: Sequence[Signal]) -> dict[str, Any]:
    longs = sum(signal.packet.side == 1 for signal in signals)
    shorts = sum(signal.packet.side == -1 for signal in signals)
    months = Counter(signal.entry_time.strftime("%Y-%m") for signal in signals)
    count = len(signals)
    return {
        "events": count,
        "longs": longs,
        "shorts": shorts,
        "long_share": longs / count if count else None,
        "short_share": shorts / count if count else None,
        "maximum_month_share": max(months.values()) / count if count else None,
        "month_counts": dict(sorted(months.items())),
    }


def support_statistics(primary: Sequence[Signal]) -> dict[str, Any]:
    windows = {
        "train_2021_2022": ("2021-01-01T00:00:00Z", "2023-01-01T00:00:00Z"),
        "train_2021": ("2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z"),
        "train_2022": ("2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z"),
        "selection_2023": ("2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
        "selection_2023_h1": ("2023-01-01T00:00:00Z", "2023-07-01T00:00:00Z"),
        "selection_2023_h2": ("2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    }
    return {
        name: _window_stats(
            _signals_in_window(primary, parse_time(start), parse_time(end))
        )
        for name, (start, end) in windows.items()
    }


def evaluate_support_gates(stats: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    gates = prereg.SUPPORT_GATES
    train = stats["train_2021_2022"]
    selection = stats["selection_2023"]
    side_ok_train = (
        train["long_share"] is not None
        and gates["minimum_side_share_train"]
        <= train["long_share"]
        <= gates["maximum_side_share_train"]
    )
    side_ok_selection = (
        selection["long_share"] is not None
        and gates["minimum_side_share_selection"]
        <= selection["long_share"]
        <= gates["maximum_side_share_selection"]
    )
    month_shares = [
        stats[name]["maximum_month_share"]
        for name in ("train_2021_2022", "selection_2023")
    ]
    return {
        "minimum_train_events": train["events"] >= gates["minimum_train_events"],
        "minimum_selection_events": selection["events"]
        >= gates["minimum_selection_events"],
        "minimum_events_each_train_year": min(
            stats["train_2021"]["events"], stats["train_2022"]["events"]
        )
        >= gates["minimum_events_each_train_year"],
        "minimum_events_each_selection_half": min(
            stats["selection_2023_h1"]["events"],
            stats["selection_2023_h2"]["events"],
        )
        >= gates["minimum_events_each_selection_half"],
        "side_balance_train": bool(side_ok_train),
        "side_balance_selection": bool(side_ok_selection),
        "maximum_entry_month_share": all(
            share is not None and share <= gates["maximum_entry_month_share"]
            for share in month_shares
        ),
    }


def load_sealed_comparator_entries() -> tuple[
    dict[tuple[str, str], list[datetime]], dict[str, int]
]:
    if sha256_file(SEALED_COMPARATOR_CLOCK) != SEALED_COMPARATOR_CLOCK_SHA256:
        raise ValueError("UGCI-288 sealed comparator clock hash mismatch")
    if sha256_file(SEALED_COMPARATOR_MANIFEST) != SEALED_COMPARATOR_MANIFEST_SHA256:
        raise ValueError("UGCI-288 sealed comparator manifest hash mismatch")
    manifest = json.loads(_path(SEALED_COMPARATOR_MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("manifest_hash") != SEALED_COMPARATOR_MANIFEST_HASH:
        raise ValueError("UGCI-288 sealed comparator manifest changed")
    if manifest.get("output", {}).get("sha256") != SEALED_COMPARATOR_CLOCK_SHA256:
        raise ValueError("UGCI-288 sealed comparator output binding changed")
    contracts = {
        comparator["candidate"]: comparator for comparator in prereg.COMPARATORS
    }
    retained = {
        (comparator["candidate"], control): []
        for comparator in prereg.COMPARATORS
        for control in comparator["controls"]
    }
    physical_rows = 0
    with gzip.open(
        _path(SEALED_COMPARATOR_CLOCK), "rt", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SEALED_COMPARATOR_COLUMNS:
            raise ValueError("UGCI-288 sealed comparator schema changed")
        for row in reader:
            physical_rows += 1
            candidate = row["candidate"]
            control = row["control"]
            if candidate not in contracts or (candidate, control) not in retained:
                raise ValueError("UGCI-288 sealed comparator identity changed")
            contract = contracts[candidate]
            if (
                row["comparison_start"] != contract["comparison_start"]
                or row["comparison_end_exclusive"]
                != contract["comparison_end_exclusive"]
            ):
                raise ValueError("UGCI-288 sealed comparator interval changed")
            entry = parse_time(row["entry_time"])
            start = parse_time(row["comparison_start"])
            end = parse_time(row["comparison_end_exclusive"])
            if not start <= entry < end or end > parse_time("2024-01-01T00:00:00Z"):
                raise ValueError("UGCI-288 sealed comparator escaped pre-2024 interval")
            retained[(candidate, control)].append(entry)
    if physical_rows != manifest.get("output", {}).get("rows"):
        raise ValueError("UGCI-288 sealed comparator row count changed")
    for identity, entries in retained.items():
        if entries != sorted(entries) or len(entries) != len(set(entries)):
            raise ValueError(
                f"UGCI-288 sealed comparator is not unique and sorted: {identity}"
            )
    return retained, {
        "sealed_comparator_bundle_files_opened": 1,
        "sealed_comparator_rows_parsed": physical_rows,
        "post_2023_comparator_rows_parsed": 0,
        "original_comparator_files_opened": 0,
    }


def exact_jaccard(left: Sequence[datetime], right: Sequence[datetime]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def near_share(
    source: Sequence[datetime], target: Sequence[datetime], delta: timedelta
) -> float:
    if not source:
        return 1.0
    if not target:
        return 0.0
    ordered = sorted(target)
    near = 0
    for value in source:
        index = bisect.bisect_left(ordered, value - delta)
        if index < len(ordered) and ordered[index] <= value + delta:
            near += 1
    return near / len(source)


def novelty_report(primary: Sequence[Signal]) -> tuple[dict[str, Any], dict[str, int]]:
    reports: dict[str, Any] = {}
    entries_by_identity, totals = load_sealed_comparator_entries()
    gates = prereg.SUPPORT_GATES
    delta = timedelta(hours=gates["novelty_containment_hours"])
    for comparator in prereg.COMPARATORS:
        start = parse_time(comparator["comparison_start"])
        end = parse_time(comparator["comparison_end_exclusive"])
        candidate_entries = [
            signal.entry_time for signal in primary if start <= signal.entry_time < end
        ]
        for control in comparator["controls"]:
            entries = entries_by_identity[(comparator["candidate"], control)]
            exact = exact_jaccard(candidate_entries, entries)
            candidate_near = near_share(candidate_entries, entries, delta)
            comparator_near = near_share(entries, candidate_entries, delta)
            maximum_near = max(candidate_near, comparator_near)
            support_defined = (
                len(candidate_entries) >= gates["minimum_common_candidate_events"]
                and len(entries) >= gates["minimum_common_comparator_events"]
            )
            passed = bool(
                support_defined
                and exact <= gates["maximum_exact_entry_jaccard"]
                and maximum_near <= gates["maximum_bidirectional_novelty_containment"]
            )
            name = f"{comparator['candidate']}:{control}"
            reports[name] = {
                "comparison_start": comparator["comparison_start"],
                "comparison_end_exclusive": comparator["comparison_end_exclusive"],
                "candidate_events": len(candidate_entries),
                "comparator_events": len(entries),
                "exact_entry_jaccard": exact,
                "candidate_within_6h_share": candidate_near,
                "comparator_within_6h_share": comparator_near,
                "maximum_bidirectional_containment": maximum_near,
                "support_defined": support_defined,
                "passed": passed,
                "source": str(SEALED_COMPARATOR_CLOCK),
            }
    return reports, totals


def deterministic_gzip_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as gz:
        with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
            writer = csv.DictWriter(text, fieldnames=CLOCK_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row[column] for column in CLOCK_COLUMNS})
    return buffer.getvalue()


def write_once(path: str | Path, content: bytes) -> None:
    destination = _path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(content)
    except FileExistsError as error:
        raise FileExistsError(
            f"UGCI-288 artifact is write-once: {destination}"
        ) from error


def build_support_payload(
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
) -> tuple[dict[str, Any], bytes]:
    preregistration = validate_preregistration()
    source_manifest = validate_source_inputs()
    events, source_audit = load_source_events()
    packets = attach_prior_thresholds(build_packets(events))
    controls = ("primary", *prereg.SOURCE_ONLY_CONTROLS)
    signals = {control: schedule_signals(packets, control) for control in controls}
    primary = signals["primary"]
    stats = support_statistics(primary)
    support_checks = evaluate_support_gates(stats)
    support_passed_before_novelty = all(support_checks.values())
    if support_passed_before_novelty:
        novelty, comparator_audit = novelty_report(primary)
        novelty_checks = {name: row["passed"] for name, row in novelty.items()}
    else:
        novelty = {}
        novelty_checks = {}
        comparator_audit = {
            "sealed_comparator_bundle_files_opened": 0,
            "sealed_comparator_rows_parsed": 0,
            "post_2023_comparator_rows_parsed": 0,
            "original_comparator_files_opened": 0,
        }
    novelty_passed = bool(novelty_checks) and all(novelty_checks.values())
    source_support_passed = support_passed_before_novelty and novelty_passed
    all_rows = [
        signal_row(signal) for control in controls for signal in signals[control]
    ]
    clock_bytes = deterministic_gzip_csv(all_rows)
    clock_sha = hashlib.sha256(clock_bytes).hexdigest()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": CANDIDATE,
        "as_of_date": AS_OF_DATE,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_artifact": str(PREREGISTRATION_ARTIFACT),
        "preregistration_artifact_sha256": PREREGISTRATION_ARTIFACT_SHA256,
        "preregistration_manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        "evaluator_source": str(EVALUATOR_SOURCE),
        "evaluator_source_sha256": sha256_file(EVALUATOR_SOURCE),
        "frozen_config": asdict(prereg.FROZEN_CONFIG),
        "source_manifest_hash": source_manifest["manifest_hash"],
        "source_audit": source_audit,
        "packet_audit": {
            "packets": len(packets),
            "first_packet_start": format_time(packets[0].start),
            "last_packet_start": format_time(packets[-1].start),
            "zero_event_packets": sum(packet.gross_raw == 0 for packet in packets),
            "threshold_ready_packets": sum(
                packet.prior_gross_q95 is not None for packet in packets
            ),
        },
        "clock": {
            "path": str(clock_output),
            "sha256": clock_sha,
            "rows": len(all_rows),
            "columns": list(CLOCK_COLUMNS),
            "control_counts": {
                control: len(control_signals)
                for control, control_signals in signals.items()
            },
        },
        "primary_support": stats,
        "support_checks": support_checks,
        "support_passed_before_novelty": support_passed_before_novelty,
        "novelty": novelty,
        "novelty_checks": novelty_checks,
        "novelty_passed": novelty_passed,
        "source_support_passed": source_support_passed,
        "advance_to_strict_evaluator_freeze": source_support_passed,
        "decision": (
            "advance_to_strict_evaluator_freeze"
            if source_support_passed
            else "retire_UGCI_288_without_repair"
        ),
        "outcome_boundary": {
            "outcomes_opened": False,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_contract_event_rows_read": 0,
            **comparator_audit,
            "network_calls": 0,
            "subprocess_calls": 0,
        },
        "preregistration_reproduced": preregistration["manifest_hash"]
        == PREREGISTRATION_MANIFEST_HASH,
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload, clock_bytes


def write_support(
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
) -> dict[str, Any]:
    if _path(clock_output).exists() or _path(report_output).exists():
        raise FileExistsError("UGCI-288 support outputs are write-once")
    payload, clock_bytes = build_support_payload(clock_output)
    report_bytes = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    write_once(clock_output, clock_bytes)
    try:
        write_once(report_output, report_bytes)
    except Exception:
        _path(clock_output).unlink(missing_ok=True)
        raise
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-output", type=Path, default=DEFAULT_CLOCK_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    args = parser.parse_args()
    payload = write_support(args.clock_output, args.report_output)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
