"""Evaluate the frozen GNRC family on pre-2024 BTC market outcomes.

This module has no command-line entry point.  A separately committed launcher
must bind this file and its protocol document before any market value is read.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROTOCOL_VERSION = "gdelt_narrative_rotation_clearing_economic_selection_v1"
AS_OF_DATE = "2026-07-20"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = Path(
    "results/gdelt_narrative_rotation_clearing_preregistration_2026-07-20.json"
)
PREREGISTRATION_SHA256 = (
    "ae175a242db1fa850164789e4a3e6f3f39b4ac8eae0fb877ce79e915ae3d67f3"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_gdelt_narrative_rotation_clearing.py"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "68c8402c4b04f9d301a76bf4ed202d2488154de03365467a13abf435e5ffe587"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/gdelt-narrative-rotation-clearing-preregistration-2026-07-20.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "50b1e2550b8ec3e36b4db39873ed404b734d689f3094ff5da1d9d0bb10e2a388"
)
SOURCE_SUPPORT_EVALUATOR = Path(
    "training/evaluate_gdelt_narrative_source_support.py"
)
SOURCE_SUPPORT_EVALUATOR_SHA256 = (
    "b09ae64c831376bce686e55de4bcbe630924faad7acc8cf81bc6cd31ff2b735a"
)
SOURCE_SUPPORT_REPORT = Path(
    "results/gdelt_narrative_rotation_clearing_source_support_2026-07-20.json"
)
SOURCE_SUPPORT_REPORT_SHA256 = (
    "1b35c6fef694f1b352129cd3b40ae85832834561f61b731bccaf4d8b24c2a5e4"
)
SOURCE_SUPPORT_MANIFEST_HASH = (
    "fa4465fa3a8f6b001d4179c692e2d0a7b11e6ce7439a474bb995541b9aa32780"
)
MARKET_DATA = Path(
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_DATA_SHA256 = (
    "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
)
MARKET_MANIFEST = Path(
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
FUNDING_DATA = Path("data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz")
FUNDING_DATA_SHA256 = (
    "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
)
FUNDING_MANIFEST = Path(
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)
FUNDING_MANIFEST_SHA256 = (
    "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
)
EVALUATOR_SOURCE = Path(
    "training/evaluate_gdelt_narrative_economic_selection.py"
)
PROTOCOL_DOCUMENT = Path(
    "docs/gdelt-narrative-rotation-clearing-economic-selection-protocol-"
    "2026-07-20.md"
)
TEST_SOURCE = Path(
    "tests/test_evaluate_gdelt_narrative_economic_selection.py"
)
PREMARKET_ACCESS_SEAL = Path(
    "results/gdelt_gnrc_premarket_access_seal_2026-07-22.json"
)
DEFAULT_OUTPUT = Path(
    "results/gdelt_narrative_rotation_clearing_economic_selection_2026-07-20.json"
)

MARKET_START = "2021-01-01T00:00:00Z"
MARKET_END_EXCLUSIVE = "2024-01-01T00:00:00Z"
EXPECTED_MARKET_ROWS = 1_095 * 24 * 12
EXPECTED_FUNDING_ROWS = 1_095 * 3
EXPECTED_MARKET_COLUMNS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base",
    "taker_buy_quote",
)
EXPECTED_FUNDING_COLUMNS = (
    "funding_time_ms",
    "funding_time_utc",
    "symbol",
    "funding_rate",
    "settlement_mark_price",
    "mark_open_time_ms",
    "mark_open_time_utc",
    "funding_time_offset_ms",
    "mark_source",
)
BOOTSTRAP_BATCH_DRAWS = 1_000
PREMARKET_SEAL_FIELDS = (
    "protocol_version",
    "source_support_report_path",
    "source_support_report_sha256",
    "evaluator_source_path",
    "evaluator_source_sha256",
    "protocol_document_path",
    "protocol_document_sha256",
    "test_source_path",
    "test_source_sha256",
    "market_data_path",
    "market_data_sha256",
    "market_manifest_path",
    "market_manifest_sha256",
    "funding_data_path",
    "funding_data_sha256",
    "funding_manifest_path",
    "funding_manifest_sha256",
    "market_values_inspected_before_seal",
    "funding_values_inspected_before_seal",
    "post_2023_outcomes_inspected_before_seal",
    "sealed_at",
)


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
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


def _load_json(path: str | Path) -> dict[str, Any]:
    with repository_path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"GNRC economic input is not a JSON object: {path}")
    return payload


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _bootstrap_frozen_modules() -> tuple[Any, Any]:
    expected_hashes = {
        PREREGISTRATION: PREREGISTRATION_SHA256,
        PREREGISTRATION_SOURCE: PREREGISTRATION_SOURCE_SHA256,
        PREREGISTRATION_DOCUMENT: PREREGISTRATION_DOCUMENT_SHA256,
        SOURCE_SUPPORT_EVALUATOR: SOURCE_SUPPORT_EVALUATOR_SHA256,
        SOURCE_SUPPORT_REPORT: SOURCE_SUPPORT_REPORT_SHA256,
    }
    for path, expected_hash in expected_hashes.items():
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"GNRC economic frozen input changed before import: {path}")
    prereg_module = importlib.import_module(
        "training.preregister_gdelt_narrative_rotation_clearing"
    )
    source_module = importlib.import_module(
        "training.evaluate_gdelt_narrative_source_support"
    )
    return prereg_module, source_module


prereg, source_support = _bootstrap_frozen_modules()


def validate_premarket_access_seal() -> dict[str, Any]:
    seal = _load_json(PREMARKET_ACCESS_SEAL)
    if set(seal) != set(PREMARKET_SEAL_FIELDS):
        raise ValueError("GNRC premarket seal fields changed")
    if seal.get("protocol_version") != "gdelt_gnrc_premarket_access_seal_v1":
        raise ValueError("GNRC premarket seal protocol changed")
    paths = {
        "source_support_report_path": SOURCE_SUPPORT_REPORT,
        "evaluator_source_path": EVALUATOR_SOURCE,
        "protocol_document_path": PROTOCOL_DOCUMENT,
        "test_source_path": TEST_SOURCE,
        "market_data_path": MARKET_DATA,
        "market_manifest_path": MARKET_MANIFEST,
        "funding_data_path": FUNDING_DATA,
        "funding_manifest_path": FUNDING_MANIFEST,
    }
    hashes = {
        "source_support_report_sha256": SOURCE_SUPPORT_REPORT,
        "evaluator_source_sha256": EVALUATOR_SOURCE,
        "protocol_document_sha256": PROTOCOL_DOCUMENT,
        "test_source_sha256": TEST_SOURCE,
        "market_data_sha256": MARKET_DATA,
        "market_manifest_sha256": MARKET_MANIFEST,
        "funding_data_sha256": FUNDING_DATA,
        "funding_manifest_sha256": FUNDING_MANIFEST,
    }
    for field, expected_path in paths.items():
        if seal[field] != str(expected_path):
            raise ValueError(f"GNRC premarket seal path changed: {field}")
    for field, path in hashes.items():
        if not _is_sha256(seal[field]) or seal[field] != sha256_file(path):
            raise ValueError(f"GNRC premarket seal hash changed: {field}")
    if (
        seal["source_support_report_sha256"] != SOURCE_SUPPORT_REPORT_SHA256
        or seal["market_data_sha256"] != MARKET_DATA_SHA256
        or seal["market_manifest_sha256"] != MARKET_MANIFEST_SHA256
        or seal["funding_data_sha256"] != FUNDING_DATA_SHA256
        or seal["funding_manifest_sha256"] != FUNDING_MANIFEST_SHA256
    ):
        raise ValueError("GNRC premarket seal ancestry changed")
    if (
        seal["market_values_inspected_before_seal"] is not False
        or seal["funding_values_inspected_before_seal"] is not False
        or seal["post_2023_outcomes_inspected_before_seal"] is not False
    ):
        raise ValueError("GNRC premarket seal records premature outcome access")
    prereg.parse_utc(str(seal["sealed_at"]))
    return seal


def validate_source_support_report() -> dict[str, Any]:
    if sha256_file(SOURCE_SUPPORT_REPORT) != SOURCE_SUPPORT_REPORT_SHA256:
        raise ValueError("GNRC source-support report hash changed")
    report = _load_json(SOURCE_SUPPORT_REPORT)
    unhashed = dict(report)
    manifest_hash = unhashed.pop("manifest_hash", None)
    if (
        manifest_hash != SOURCE_SUPPORT_MANIFEST_HASH
        or manifest_hash != source_support.canonical_hash(unhashed)
    ):
        raise ValueError("GNRC source-support report internal hash changed")
    expected_ids = tuple(prereg.FAMILY_VARIANT_IDS)
    if (
        report.get("decision") != "advance_to_market"
        or report.get("family_support", {}).get("family_advances") is not True
        or set(report.get("variant_support", {})) != set(expected_ids)
    ):
        raise ValueError("GNRC source-support family did not advance unchanged")
    passing = tuple(
        variant_id
        for variant_id in expected_ids
        if report["variant_support"][variant_id].get("passes") is True
    )
    family = report["family_support"]
    if (
        tuple(family.get("passing_variant_ids", ())) != passing
        or family.get("passing_variant_count") != len(passing)
        or len(passing) != 17
        or not all(family.get("checks", {}).values())
    ):
        raise ValueError("GNRC source-support passing set changed")
    if report.get("outcome_boundary") != {
        "btc_market_rows_read": 0,
        "economic_metrics_computed": False,
        "funding_rows_read": 0,
        "future_return_rows_read": 0,
        "outcomes_opened": False,
        "post_2023_news_rows_read": 0,
        "return_or_pnl_fields_read": 0,
    }:
        raise ValueError("GNRC source-support report opened an outcome")
    return report


def validate_market_manifests() -> dict[str, Any]:
    frozen = {
        MARKET_DATA: MARKET_DATA_SHA256,
        MARKET_MANIFEST: MARKET_MANIFEST_SHA256,
        FUNDING_DATA: FUNDING_DATA_SHA256,
        FUNDING_MANIFEST: FUNDING_MANIFEST_SHA256,
    }
    for path, expected_hash in frozen.items():
        if sha256_file(path) != expected_hash:
            raise ValueError(f"GNRC frozen market input changed: {path}")

    market = _load_json(MARKET_MANIFEST)
    market_config = market.get("config", {})
    market_protocol = market.get("protocol", {})
    if (
        market_config.get("symbol") != "BTCUSDT"
        or market_config.get("interval") != "5m"
        or market_config.get("start") != "2020-01-01"
        or market_config.get("end") != "2024-01-01"
        or market.get("combined_output") != str(MARKET_DATA)
        or market.get("combined_sha256") != MARKET_DATA_SHA256
        or market.get("rows") != 1_461 * 24 * 12
        or market.get("first_date") != "2020-01-01 00:00:00"
        or market.get("last_date") != "2023-12-31 23:55:00"
        or tuple(market.get("columns", ())) != EXPECTED_MARKET_COLUMNS
        or market_protocol.get("source")
        != "official Binance USD-M daily kline archives"
        or market_protocol.get("archive_checksums_verified") is not True
        or market_protocol.get("end_is_exclusive") is not True
        or market_protocol.get("outcomes_opened") is not False
    ):
        raise ValueError("GNRC market manifest contract changed")

    funding = _load_json(FUNDING_MANIFEST)
    funding_core = {
        key: value
        for key, value in funding.items()
        if key not in {"manifest_hash", "created_at"}
    }
    data = funding.get("data", {})
    mapping = funding.get("mapping", {})
    if (
        funding.get("manifest_hash") != canonical_hash(funding_core)
        or funding.get("protocol_version")
        != "btc_um_funding_settlement_marks_2020_2023_v1"
        or funding.get("outcomes_opened") is not False
        or funding.get("strategy_outcomes_calculated") != []
        or data.get("path") != str(FUNDING_DATA)
        or data.get("sha256") != FUNDING_DATA_SHA256
        or data.get("rows") != 1_461 * 3
        or tuple(data.get("columns", ())) != EXPECTED_FUNDING_COLUMNS
        or mapping.get("funding_time") != "exact returned fundingTime retained"
        or mapping.get("mark")
        != "open of floor(fundingTime, 8h) official mark-price kline"
        or mapping.get("maximum_allowed_timestamp_offset_ms") != 60_000
    ):
        raise ValueError("GNRC funding manifest contract changed")
    return {"market": market, "funding": funding}


def _parse_market_timestamp(value: str) -> datetime:
    parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    return parsed.replace(tzinfo=prereg.UTC)


def load_market_bars(
    path: str | Path = MARKET_DATA,
    *,
    start: str = MARKET_START,
    end_exclusive: str = MARKET_END_EXCLUSIVE,
) -> tuple[Any, ...]:
    start_time = prereg.parse_utc(start)
    end_time = prereg.parse_utc(end_exclusive)
    rows: list[Any] = []
    last_physical_timestamp: datetime | None = None
    try:
        with gzip.open(
            repository_path(path), "rt", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.reader(handle)
            header = tuple(next(reader))
            if header != EXPECTED_MARKET_COLUMNS or header[0] != "date":
                raise ValueError("GNRC market columns changed")
            positions = {column: header.index(column) for column in header}
            for fields in reader:
                if len(fields) != len(header):
                    raise ValueError("GNRC market row shape changed")
                timestamp = _parse_market_timestamp(fields[0])
                last_physical_timestamp = timestamp
                if timestamp < start_time:
                    continue
                if timestamp >= end_time:
                    break
                rows.append(
                    prereg.MarketBar(
                        open_time=timestamp,
                        open=float(fields[positions["open"]]),
                        high=float(fields[positions["high"]]),
                        low=float(fields[positions["low"]]),
                        close=float(fields[positions["close"]]),
                    )
                )
    except (OSError, UnicodeDecodeError, csv.Error, StopIteration) as error:
        raise ValueError("GNRC market artifact is unreadable") from error
    expected = prereg._complete_bar_grid(start_time, end_time)
    if tuple(bar.open_time for bar in rows) != expected:
        raise ValueError("GNRC market bars do not equal the frozen 5m grid")
    if path == MARKET_DATA and last_physical_timestamp != expected[-1]:
        raise ValueError("GNRC market artifact terminal timestamp changed")
    for bar in rows:
        values = (bar.open, bar.high, bar.low, bar.close)
        if (
            not all(math.isfinite(value) and value > 0.0 for value in values)
            or bar.low > min(bar.open, bar.close)
            or bar.high < max(bar.open, bar.close)
            or bar.low > bar.high
        ):
            raise ValueError("GNRC market OHLC is invalid")
    return tuple(rows)


def load_funding_marks(
    path: str | Path = FUNDING_DATA,
    *,
    start: str = MARKET_START,
    end_exclusive: str = MARKET_END_EXCLUSIVE,
) -> tuple[Any, ...]:
    start_time = prereg.parse_utc(start)
    end_time = prereg.parse_utc(end_exclusive)
    start_ms = int(start_time.timestamp() * 1_000)
    end_ms = int(end_time.timestamp() * 1_000)
    rows: list[Any] = []
    last_physical_mark: datetime | None = None
    try:
        with gzip.open(
            repository_path(path), "rt", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.reader(handle)
            header = tuple(next(reader))
            if header != EXPECTED_FUNDING_COLUMNS or header[0] != "funding_time_ms":
                raise ValueError("GNRC funding columns changed")
            positions = {column: header.index(column) for column in header}
            for fields in reader:
                if len(fields) != len(header):
                    raise ValueError("GNRC funding row shape changed")
                funding_ms = int(fields[positions["funding_time_ms"]])
                mark_ms = int(fields[positions["mark_open_time_ms"]])
                mark_time = prereg.parse_utc(fields[positions["mark_open_time_utc"]])
                last_physical_mark = mark_time
                if mark_ms < start_ms:
                    continue
                if mark_ms >= end_ms:
                    break
                funding_time = prereg.parse_utc(
                    fields[positions["funding_time_utc"]]
                )
                offset = int(fields[positions["funding_time_offset_ms"]])
                if (
                    fields[positions["symbol"]] != "BTCUSDT"
                    or fields[positions["mark_source"]]
                    != "binance_8h_mark_price_kline_open"
                    or mark_ms != funding_ms // (8 * 60 * 60 * 1_000) * (
                        8 * 60 * 60 * 1_000
                    )
                    or int(funding_time.timestamp() * 1_000) != funding_ms
                    or int(mark_time.timestamp() * 1_000) != mark_ms
                    or offset != funding_ms - mark_ms
                    or not 0 <= offset <= 60_000
                ):
                    raise ValueError("GNRC funding timestamp mapping changed")
                rows.append(
                    prereg.FundingMark(
                        timestamp=mark_time,
                        mark_price=float(fields[positions["settlement_mark_price"]]),
                        funding_rate=float(fields[positions["funding_rate"]]),
                    )
                )
    except (OSError, UnicodeDecodeError, csv.Error, StopIteration) as error:
        raise ValueError("GNRC funding artifact is unreadable") from error
    expected = prereg._complete_funding_grid(start_time, end_time)
    if tuple(mark.timestamp for mark in rows) != expected:
        raise ValueError("GNRC funding marks do not equal the frozen 8h grid")
    if path == FUNDING_DATA and last_physical_mark != expected[-1]:
        raise ValueError("GNRC funding artifact terminal timestamp changed")
    for mark in rows:
        if (
            not math.isfinite(mark.mark_price)
            or mark.mark_price <= 0.0
            or not math.isfinite(mark.funding_rate)
        ):
            raise ValueError("GNRC funding value is invalid")
    return tuple(rows)


def build_schedules(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    row_by_date = {
        date.fromisoformat(str(row["date"])): index for index, row in enumerate(rows)
    }
    if len(row_by_date) != len(rows):
        raise ValueError("GNRC economic source dates are duplicated")
    score_cache: dict[tuple[int, int, date], dict[str, Any]] = {}

    def decision_for(
        source_date: date, score: str, fast_days: int, slow_days: int
    ) -> Any:
        cache_key = (fast_days, slow_days, source_date)
        if cache_key not in score_cache:
            index = row_by_date.get(source_date)
            if index is None or index + 1 < slow_days:
                raise ValueError("GNRC economic source history is incomplete")
            score_cache[cache_key] = prereg.compute_score_state(
                rows[index - slow_days + 1 : index + 1], fast_days, slow_days
            )
        state = score_cache[cache_key]
        score_state = state[score]
        return prereg.ScoreDecision(
            source_date=source_date,
            available_at=state["available_at"],
            long_score=score_state["long_score"],
            short_score=score_state["short_score"],
            evidence_ok=state["evidence_ok"],
        )

    schedules: dict[str, dict[str, Any]] = {}
    bounds = source_support._split_bounds()
    for variant in prereg.variants():
        variant_schedules: dict[str, Any] = {}
        for split_name, (split_start, split_end) in bounds.items():
            source_dates = prereg.expected_split_source_dates(
                split_start=split_start,
                split_end_exclusive=split_end,
                hold_days=variant["hold_days"],
            )
            decisions = [
                decision_for(
                    source_date,
                    variant["score"],
                    variant["fast_days"],
                    variant["slow_days"],
                )
                for source_date in source_dates
            ]
            variant_schedules[split_name] = prereg.schedule_events(
                decisions,
                threshold=variant["threshold"],
                hold_days=variant["hold_days"],
                split_start=split_start,
                split_end_exclusive=split_end,
            )
        schedules[str(variant["variant_id"])] = variant_schedules
    return schedules


def validate_schedules_against_source_support(
    schedules: Mapping[str, Mapping[str, Any]], report: Mapping[str, Any]
) -> None:
    if tuple(schedules) != tuple(prereg.FAMILY_VARIANT_IDS):
        raise ValueError("GNRC economic schedule family changed")
    for variant_id, split_schedules in schedules.items():
        frozen = report["variant_support"][variant_id]
        for split_name in ("train", "selection"):
            observed = prereg.support_rates(split_schedules[split_name])
            if observed != frozen[split_name]:
                raise ValueError(
                    f"GNRC economic schedule differs from source support: "
                    f"{variant_id}/{split_name}"
                )


def _validate_execution_inputs(
    schedule: Any,
    bars: Sequence[Any],
    funding_marks: Sequence[Any],
    *,
    split_start: datetime,
    split_end_exclusive: datetime,
    side_cost_bps: float,
) -> None:
    expected_bars = prereg._complete_bar_grid(split_start, split_end_exclusive)
    if tuple(bar.open_time for bar in bars) != expected_bars:
        raise ValueError("GNRC economic bars are incomplete")
    expected_funding = prereg._complete_funding_grid(
        split_start, split_end_exclusive
    )
    if tuple(mark.timestamp for mark in funding_marks) != expected_funding:
        raise ValueError("GNRC economic funding grid is incomplete")
    if side_cost_bps not in {
        prereg.ECONOMIC_PROTOCOL["base_cost"]["entry_bps"],
        prereg.ECONOMIC_PROTOCOL["stress_cost"]["entry_bps"],
    }:
        raise ValueError("GNRC economic side cost changed")
    events = schedule.admitted_events
    if tuple(events) != tuple(sorted(events, key=lambda event: event.entry_time)):
        raise ValueError("GNRC economic events are not causally sorted")

    def is_bar_open(timestamp: datetime) -> bool:
        offset = timestamp - split_start
        return (
            split_start <= timestamp < split_end_exclusive
            and offset % prereg.BAR_INTERVAL == timedelta(0)
        )

    for index, event in enumerate(events):
        expected_decision = datetime.combine(
            event.source_date, datetime.min.time(), tzinfo=prereg.UTC
        ) + timedelta(hours=48, minutes=15)
        if (
            event.side not in {-1, 1}
            or event.decision_time != expected_decision
            or event.entry_time != expected_decision + timedelta(minutes=10)
            or event.exit_time - event.entry_time
            not in {timedelta(days=days) for days in prereg.HOLD_DAYS}
            or not is_bar_open(event.entry_time)
            or not is_bar_open(event.exit_time)
            or (index and event.entry_time <= events[index - 1].exit_time)
        ):
            raise ValueError("GNRC economic event timing changed")


def simulate_market_path(
    schedule: Any,
    bars: Sequence[Any],
    funding_marks: Sequence[Any],
    *,
    split_start: datetime,
    split_end_exclusive: datetime,
    side_cost_bps: float,
) -> tuple[dict[str, float | int], np.ndarray]:
    _validate_execution_inputs(
        schedule,
        bars,
        funding_marks,
        split_start=split_start,
        split_end_exclusive=split_end_exclusive,
        side_cost_bps=side_cost_bps,
    )
    events = schedule.admitted_events
    entries = {event.entry_time: event for event in events}
    exits = {event.exit_time: event for event in events}
    funding_by_time = {mark.timestamp: mark for mark in funding_marks}
    cash = float(prereg.ECONOMIC_PROTOCOL["initial_equity"])
    units = 0.0
    active: Any | None = None
    peak = cash
    strict_mdd = 0.0
    daily_end_equity: list[float] = []

    def append_equity(price: float) -> float:
        nonlocal peak, strict_mdd
        equity = cash + units * price
        if not math.isfinite(equity) or equity <= 0.0:
            raise ValueError("GNRC economic strict path reached nonpositive equity")
        peak = max(peak, equity)
        strict_mdd = max(strict_mdd, (peak - equity) / peak)
        return equity

    for bar in bars:
        append_equity(bar.open)
        funding = funding_by_time.get(bar.open_time)
        if (
            funding is not None
            and active is not None
            and active.entry_time < funding.timestamp <= active.exit_time
        ):
            cash += prereg.funding_cash_change(
                units, funding.mark_price, funding.funding_rate
            )
            append_equity(bar.open)
        exiting = exits.get(bar.open_time)
        if exiting is not None:
            if active != exiting:
                raise ValueError("GNRC economic exit does not match active event")
            cash += units * bar.open
            cash -= prereg.execution_cost(units, bar.open, side_cost_bps)
            units = 0.0
            active = None
            append_equity(bar.open)
        entering = entries.get(bar.open_time)
        if entering is not None:
            if active is not None:
                raise ValueError("GNRC economic entry overlaps active event")
            equity_before_cost = cash
            units = entering.side * equity_before_cost / bar.open
            cash -= units * bar.open
            cash -= prereg.execution_cost(units, bar.open, side_cost_bps)
            active = entering
            append_equity(bar.open)
        if active is None:
            close_equity = append_equity(bar.close)
        else:
            close_equity = 0.0
            for price in prereg.strict_held_bar_prices(
                active.side, bar.high, bar.low, bar.close
            ):
                close_equity = append_equity(price)
        if (bar.open_time + prereg.BAR_INTERVAL).date() != bar.open_time.date():
            daily_end_equity.append(close_equity)
    if active is not None or units != 0.0:
        raise ValueError("GNRC economic position remained open at split boundary")
    calendar_days = (split_end_exclusive - split_start).days
    if (
        split_end_exclusive - split_start != timedelta(days=calendar_days)
        or len(daily_end_equity) != calendar_days
    ):
        raise ValueError("GNRC economic daily return grid changed")
    ending_equity = daily_end_equity[-1]
    absolute_return = ending_equity - 1.0
    cagr = ending_equity ** (365.2425 / calendar_days) - 1.0
    metrics: dict[str, float | int] = {
        "absolute_return": absolute_return,
        "cagr": cagr,
        "strict_mdd": strict_mdd,
        "cagr_to_strict_mdd": (
            cagr / strict_mdd
            if strict_mdd > 0.0
            else math.inf
            if cagr > 0.0
            else 0.0
        ),
        "full_calendar_days": float(calendar_days),
        "trade_count": len(events),
    }
    endpoints = np.asarray([1.0, *daily_end_equity], dtype=np.float64)
    daily_log_returns = np.diff(np.log(endpoints))
    if (
        len(daily_log_returns) != calendar_days
        or not np.isfinite(daily_log_returns).all()
        or not math.isclose(
            float(daily_log_returns.sum()), math.log(ending_equity), abs_tol=1e-12
        )
    ):
        raise ValueError("GNRC economic daily returns are inconsistent")
    return metrics, daily_log_returns


def qualifier_checks(
    base: Mapping[str, float | int],
    stress: Mapping[str, float | int],
    *,
    split: str,
) -> dict[str, bool]:
    if split == "train":
        gates = prereg.ECONOMIC_PROTOCOL["train_qualifiers"]
        return {
            "absolute_return_positive": base["absolute_return"] > 0.0,
            "minimum_cagr_to_strict_mdd": (
                base["cagr_to_strict_mdd"]
                >= gates["minimum_cagr_to_strict_mdd"]
            ),
            "maximum_strict_mdd": (
                base["strict_mdd"] * 100.0
                <= gates["maximum_strict_mdd_percent"]
            ),
            "minimum_trades": base["trade_count"] >= gates["minimum_trades"],
            "stress_absolute_return_positive": stress["absolute_return"] > 0.0,
        }
    if split == "selection":
        gates = prereg.ECONOMIC_PROTOCOL["selection_qualifiers"]
        return {
            "absolute_return_positive": base["absolute_return"] > 0.0,
            "minimum_cagr_to_strict_mdd": (
                base["cagr_to_strict_mdd"]
                >= gates["minimum_cagr_to_strict_mdd"]
            ),
            "maximum_strict_mdd": (
                base["strict_mdd"] * 100.0
                <= gates["maximum_strict_mdd_percent"]
            ),
            "minimum_trades": base["trade_count"] >= gates["minimum_trades"],
        }
    raise ValueError("GNRC economic qualifier split changed")


def _studentized_mean(values: np.ndarray) -> float:
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or len(sample) < 2 or not np.isfinite(sample).all():
        raise ValueError("GNRC Romano-Wolf input is invalid")
    if float(np.ptp(sample)) == 0.0:
        return 0.0
    standard_deviation = float(sample.std(ddof=1))
    return math.sqrt(len(sample)) * float(sample.mean()) / standard_deviation


def romano_wolf_stepdown(
    daily_returns: Mapping[str, np.ndarray],
    train_eligible_ids: Sequence[str],
    *,
    draws: int = 100_000,
    block_days: int = 7,
    seed: int = 20_260_720,
    batch_draws: int = BOOTSTRAP_BATCH_DRAWS,
) -> dict[str, Any]:
    family_ids = tuple(prereg.FAMILY_VARIANT_IDS)
    if tuple(daily_returns) != family_ids:
        raise ValueError("GNRC Romano-Wolf family order changed")
    matrix = np.vstack(
        [np.asarray(daily_returns[variant_id], dtype=np.float64) for variant_id in family_ids]
    )
    if (
        matrix.ndim != 2
        or matrix.shape[1] != 365
        or not np.isfinite(matrix).all()
        or draws < 1
        or block_days < 1
        or batch_draws < 1
    ):
        raise ValueError("GNRC Romano-Wolf configuration or returns changed")
    train_eligible = set(train_eligible_ids)
    if not train_eligible <= set(family_ids):
        raise ValueError("GNRC Romano-Wolf eligibility escaped the family")
    observed = {
        variant_id: _studentized_mean(matrix[index])
        for index, variant_id in enumerate(family_ids)
    }
    variance_positive = {
        variant_id: float(np.ptp(matrix[index])) > 0.0
        for index, variant_id in enumerate(family_ids)
    }
    eligible = tuple(
        variant_id
        for variant_id in family_ids
        if variant_id in train_eligible and variance_positive[variant_id]
    )
    order = tuple(sorted(eligible, key=lambda item: (-observed[item], item)))
    adjusted = {variant_id: 1.0 for variant_id in family_ids}
    raw_stepdown = {variant_id: 1.0 for variant_id in family_ids}
    if order:
        centered = matrix - matrix.mean(axis=1, keepdims=True)
        family_index = {variant_id: index for index, variant_id in enumerate(family_ids)}
        ordered_indices = [family_index[variant_id] for variant_id in order]
        exceedances = np.zeros(len(order), dtype=np.int64)
        generator = np.random.default_rng(seed)
        blocks_per_draw = math.ceil(matrix.shape[1] / block_days)
        block_offsets = np.arange(block_days, dtype=np.int64)
        completed = 0
        while completed < draws:
            current = min(batch_draws, draws - completed)
            starts = generator.integers(
                0,
                matrix.shape[1],
                size=(current, blocks_per_draw),
                dtype=np.int64,
            )
            indices = (
                starts[:, :, None] + block_offsets[None, None, :]
            ) % matrix.shape[1]
            indices = indices.reshape(current, -1)[:, : matrix.shape[1]]
            bootstrap_t = np.empty((current, len(order)), dtype=np.float64)
            for column, row_index in enumerate(ordered_indices):
                samples = centered[row_index][indices]
                deviations = samples.std(axis=1, ddof=1)
                statistics = np.zeros(current, dtype=np.float64)
                np.divide(
                    math.sqrt(matrix.shape[1]) * samples.mean(axis=1),
                    deviations,
                    out=statistics,
                    where=deviations > 0.0,
                )
                bootstrap_t[:, column] = statistics
            suffix_max = np.maximum.accumulate(
                bootstrap_t[:, ::-1], axis=1
            )[:, ::-1]
            tie_group_start = 0
            for column, variant_id in enumerate(order):
                if (
                    column > 0
                    and observed[variant_id] != observed[order[column - 1]]
                ):
                    tie_group_start = column
                exceedances[column] += int(
                    np.count_nonzero(
                        suffix_max[:, tie_group_start] >= observed[variant_id]
                    )
                )
            completed += current
        monotone = 0.0
        for column, variant_id in enumerate(order):
            raw = (int(exceedances[column]) + 1.0) / (draws + 1.0)
            monotone = max(monotone, raw)
            raw_stepdown[variant_id] = raw
            adjusted[variant_id] = monotone
    return {
        "method": "one-sided Romano-Wolf step-down max-t circular block bootstrap",
        "draws": draws,
        "block_days": block_days,
        "seed": seed,
        "batch_draws": batch_draws,
        "daily_observations": int(matrix.shape[1]),
        "synchronized_indices": True,
        "equal_observed_t_removed_as_one_group": True,
        "ordered_tested_variant_ids": list(order),
        "observed_t": observed,
        "variance_positive": variance_positive,
        "raw_stepdown_p": raw_stepdown,
        "adjusted_p": adjusted,
    }


def _slice_grid(
    rows: Sequence[Any], start: datetime, end_exclusive: datetime, field: str
) -> tuple[Any, ...]:
    selected = tuple(
        row for row in rows if start <= getattr(row, field) < end_exclusive
    )
    return selected


def select_champion(
    variant_results: Mapping[str, Mapping[str, Any]],
    candidate_ids: Sequence[str],
) -> tuple[str | None, str | None]:
    if not candidate_ids:
        return None, None
    if not set(candidate_ids) <= set(prereg.FAMILY_VARIANT_IDS):
        raise ValueError("GNRC champion candidates escaped the frozen family")
    champion_id = min(
        candidate_ids,
        key=lambda variant_id: (
            -float(
                variant_results[variant_id]["selection"]["base_2bps_per_side"][
                    "cagr_to_strict_mdd"
                ]
            ),
            float(
                variant_results[variant_id]["selection"]["base_2bps_per_side"][
                    "strict_mdd"
                ]
            ),
            variant_id,
        ),
    )
    return champion_id, str(variant_results[champion_id]["policy_hash"])


def _source_unsupported_split(schedule: Any) -> dict[str, Any]:
    return {
        "schedule": prereg.support_rates(schedule),
        "market_outcome_opened": False,
        "base_2bps_per_side": None,
        "stress_4bps_per_side": None,
        "qualifier_checks": None,
        "qualifies": False,
        "daily_log_return_count": 0,
        "daily_log_return_hash": None,
    }


def evaluate_supported_variant(
    variant_id: str,
    policy: Mapping[str, Any],
    variant_schedules: Mapping[str, Any],
    source_report: Mapping[str, Any],
    bars_by_split: Mapping[str, Sequence[Any]],
    funding_by_split: Mapping[str, Sequence[Any]],
    bounds: Mapping[str, tuple[datetime, datetime]],
) -> tuple[dict[str, Any], np.ndarray]:
    support_pass = bool(source_report["variant_support"][variant_id]["passes"])
    policy_result: dict[str, Any] = {
        "policy": dict(policy),
        "policy_hash": canonical_hash(policy),
        "source_support_pass": support_pass,
    }
    if not support_pass:
        policy_result.update(
            {
                "outcome_status": "not_opened_source_unsupported",
                "train": _source_unsupported_split(variant_schedules["train"]),
                "selection": _source_unsupported_split(
                    variant_schedules["selection"]
                ),
            }
        )
        return policy_result, np.zeros(365, dtype=np.float64)

    split_results: dict[str, Any] = {}
    selection_daily_returns: np.ndarray | None = None
    for split_name in ("train", "selection"):
        split_start, split_end = bounds[split_name]
        base, daily_returns = simulate_market_path(
            variant_schedules[split_name],
            bars_by_split[split_name],
            funding_by_split[split_name],
            split_start=split_start,
            split_end_exclusive=split_end,
            side_cost_bps=2.0,
        )
        stress, _ = simulate_market_path(
            variant_schedules[split_name],
            bars_by_split[split_name],
            funding_by_split[split_name],
            split_start=split_start,
            split_end_exclusive=split_end,
            side_cost_bps=4.0,
        )
        checks = qualifier_checks(base, stress, split=split_name)
        split_results[split_name] = {
            "schedule": prereg.support_rates(variant_schedules[split_name]),
            "market_outcome_opened": True,
            "base_2bps_per_side": base,
            "stress_4bps_per_side": stress,
            "qualifier_checks": checks,
            "qualifies": all(checks.values()),
            "daily_log_return_count": int(len(daily_returns)),
            "daily_log_return_hash": canonical_hash(daily_returns.tolist()),
        }
        if split_name == "selection":
            selection_daily_returns = daily_returns
    if selection_daily_returns is None:
        raise RuntimeError("GNRC supported variant lacks selection daily returns")
    policy_result.update({"outcome_status": "opened_pre2024", **split_results})
    return policy_result, selection_daily_returns


def build_report() -> dict[str, Any]:
    premarket_seal = validate_premarket_access_seal()
    source_report = validate_source_support_report()
    manifests = validate_market_manifests()
    source_rows = source_support.load_daily_rows()
    schedules = build_schedules(source_rows)
    validate_schedules_against_source_support(schedules, source_report)
    all_bars = load_market_bars()
    all_funding = load_funding_marks()
    if len(all_bars) != EXPECTED_MARKET_ROWS or len(all_funding) != EXPECTED_FUNDING_ROWS:
        raise ValueError("GNRC pre-2024 market row count changed")

    bounds = source_support._split_bounds()
    bars_by_split = {
        split: _slice_grid(all_bars, start, end, "open_time")
        for split, (start, end) in bounds.items()
    }
    funding_by_split = {
        split: _slice_grid(all_funding, start, end, "timestamp")
        for split, (start, end) in bounds.items()
    }
    variant_policies = {
        str(variant["variant_id"]): variant for variant in prereg.variants()
    }
    variant_results: dict[str, dict[str, Any]] = {}
    selection_daily_returns: dict[str, np.ndarray] = {}
    train_eligible_ids: list[str] = []
    for variant_id in prereg.FAMILY_VARIANT_IDS:
        result, daily_returns = evaluate_supported_variant(
            variant_id,
            variant_policies[variant_id],
            schedules[variant_id],
            source_report,
            bars_by_split,
            funding_by_split,
            bounds,
        )
        variant_results[variant_id] = result
        selection_daily_returns[variant_id] = daily_returns
        if result["train"]["qualifies"]:
            train_eligible_ids.append(variant_id)

    familywise = romano_wolf_stepdown(
        selection_daily_returns,
        train_eligible_ids,
        draws=prereg.ECONOMIC_PROTOCOL["familywise_test"]["draws"],
        block_days=prereg.ECONOMIC_PROTOCOL["familywise_test"]["block_days"],
        seed=prereg.ECONOMIC_PROTOCOL["familywise_test"]["seed"],
    )
    adjusted_p_limit = prereg.ECONOMIC_PROTOCOL["familywise_test"][
        "adjusted_p_maximum"
    ]
    candidate_ids: list[str] = []
    for variant_id in prereg.FAMILY_VARIANT_IDS:
        result = variant_results[variant_id]
        p_value = familywise["adjusted_p"][variant_id]
        result["familywise"] = {
            "observed_t": familywise["observed_t"][variant_id],
            "variance_positive": familywise["variance_positive"][variant_id],
            "raw_stepdown_p": familywise["raw_stepdown_p"][variant_id],
            "adjusted_p": p_value,
            "passes_adjusted_p": p_value <= adjusted_p_limit,
        }
        result["selection_candidate"] = bool(
            result["train"]["qualifies"]
            and result["selection"]["qualifies"]
            and p_value <= adjusted_p_limit
        )
        if result["selection_candidate"]:
            candidate_ids.append(variant_id)

    champion_id, champion_policy_hash = select_champion(
        variant_results, candidate_ids
    )

    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "as_of_date": AS_OF_DATE,
        "family": prereg.FAMILY,
        "decision": (
            "advance_to_sealed_oos" if champion_id else "retire_without_repair"
        ),
        "champion_variant_id": champion_id,
        "champion_policy_hash": champion_policy_hash,
        "selection_candidate_ids": candidate_ids,
        "source_support": {
            "path": str(SOURCE_SUPPORT_REPORT),
            "sha256": SOURCE_SUPPORT_REPORT_SHA256,
            "manifest_hash": SOURCE_SUPPORT_MANIFEST_HASH,
            "passing_variant_ids": source_report["family_support"][
                "passing_variant_ids"
            ],
        },
        "premarket_access_seal": {
            "path": str(PREMARKET_ACCESS_SEAL),
            "sha256": sha256_file(PREMARKET_ACCESS_SEAL),
            "sealed_at": premarket_seal["sealed_at"],
            "evaluator_source_sha256": premarket_seal[
                "evaluator_source_sha256"
            ],
            "protocol_document_sha256": premarket_seal[
                "protocol_document_sha256"
            ],
            "test_source_sha256": premarket_seal["test_source_sha256"],
        },
        "market_sources": {
            "market_data": str(MARKET_DATA),
            "market_data_sha256": MARKET_DATA_SHA256,
            "market_manifest": str(MARKET_MANIFEST),
            "market_manifest_sha256": MARKET_MANIFEST_SHA256,
            "market_rows_read": len(all_bars),
            "funding_data": str(FUNDING_DATA),
            "funding_data_sha256": FUNDING_DATA_SHA256,
            "funding_manifest": str(FUNDING_MANIFEST),
            "funding_manifest_sha256": FUNDING_MANIFEST_SHA256,
            "funding_rows_read": len(all_funding),
            "provenance_outcomes_opened": {
                "market": manifests["market"]["protocol"]["outcomes_opened"],
                "funding": manifests["funding"]["outcomes_opened"],
            },
        },
        "economic_protocol": prereg.ECONOMIC_PROTOCOL,
        "economic_protocol_hash": canonical_hash(prereg.ECONOMIC_PROTOCOL),
        "implementation_conventions": {
            "daily_return_endpoint": (
                "UTC calendar-day final 5m close equity; first day starts at "
                "split initial equity"
            ),
            "funding_clock": (
                "validate exact Binance fundingTime and use its audited floor-to-8h "
                "mark_open_time as the preregistered complete settlement grid"
            ),
            "romano_wolf": (
                "standard descending-observed-t step-down; maximum bootstrap t over "
                "each remaining hypothesis set; exact observed-t ties leave the set "
                "together; plus-one p-value correction and monotonicity enforcement"
            ),
        },
        "familywise_test": familywise,
        "variant_results": variant_results,
        "outcome_boundary": {
            "pre2024_market_rows_read": len(all_bars),
            "pre2024_funding_rows_read": len(all_funding),
            "post_2023_market_rows_read": 0,
            "post_2023_funding_rows_read": 0,
            "post_2023_news_rows_read": 0,
            "oos_opened": False,
        },
        "failure_action": "retire_without_sign_threshold_window_hold_or_gate_repair",
    }
    report["manifest_hash"] = canonical_hash(report)
    return report


def write_once(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    destination = repository_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"GNRC economic report is write-once: {destination}")
    report = build_report()
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"GNRC economic report is write-once: {destination}"
        ) from error
    return report
