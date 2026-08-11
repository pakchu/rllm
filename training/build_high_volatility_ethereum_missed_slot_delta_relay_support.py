"""Build outcome-blind source support for frozen HVEMSD-24."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_high_volatility_ethereum_missed_slot_delta_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "c5ba831750484ef6eeb010560851f08abec92b4e41dd3506d7f9be0b66766f5e"
BUILDER = Path("training/build_high_volatility_ethereum_missed_slot_delta_relay_support.py")
ROOT = Path("data/high_volatility_ethereum_missed_slot_delta_relay_sources_2022_2026")
RAW = ROOT / "ethereum_boundary_headers.json"
PANEL = ROOT / "ethereum_daily_missed_slots.csv.gz"
BTC_SOURCE = ROOT / "btc_1m_ts_open_close.csv.gz"
FEATURES = ROOT / "hvemds_preentry_features.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_ethereum_missed_slot_delta_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_ethereum_missed_slot_delta_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_ethereum_missed_slot_delta_relay_support_2026-08-12.json")

BOUNDARY_START = pd.Timestamp("2022-12-30T00:00:00Z")
BOUNDARY_END = pd.Timestamp("2026-08-01T00:00:00Z")
BTC_START = pd.Timestamp("2022-12-30T00:20:00Z")
BTC_END = pd.Timestamp("2026-08-01T00:21:00Z")
MERGE_BLOCK = 15_537_394
SLOTS_PER_DAY = 7_200
CONFIRMATION_SLOTS = 64
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00Z"), pd.Timestamp("2024-01-01T00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00Z"), pd.Timestamp("2025-01-01T00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00Z"), pd.Timestamp("2026-01-01T00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00Z"), pd.Timestamp("2026-08-01T00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_btc_volatility_gate",
    "no_missed_change_tail",
    "one_day_stale_change",
    "missed_level_change",
    "direction_flip",
    "same_clock_forced_long",
)
COLUMNS = (
    "candidate",
    "control",
    "split",
    "source_day",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
    "produced_blocks",
    "missed_slots",
    "missed_change",
    "missed_change_rank",
    "btc_variation",
    "btc_variation_rank",
)
QUERY = (
    "SELECT ts,open,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' "
    "AND ts>=:start AND ts<:end ORDER BY ts"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: Any, trailing_lf: bool = False) -> bytes:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if trailing_lf else b"")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class Header:
    number: int
    block_hash: str
    parent_hash: str
    timestamp: int

    def record(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "hash": self.block_hash,
            "parentHash": self.parent_hash,
            "timestamp": self.timestamp,
        }


class RpcClient:
    def __init__(self, url: str, timeout: float = 45.0, retries: int = 4):
        self.url = url
        self.timeout = timeout
        self.retries = retries

    def call(self, method: str, params: list[Any]) -> Any:
        body = canonical_bytes({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                request = Request(
                    self.url,
                    data=body,
                    headers={"Content-Type": "application/json", "User-Agent": "rllm-hvemsd/1.0"},
                    method="POST",
                )
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read())
                if not isinstance(payload, dict) or payload.get("error") is not None or "result" not in payload:
                    raise RuntimeError(f"HVEMSD malformed RPC response for {method}")
                return payload["result"]
            except Exception as exc:  # pragma: no cover - network recovery
                last = exc
                if attempt + 1 < self.retries:
                    time.sleep(0.25 * (2**attempt))
        raise RuntimeError(f"HVEMSD RPC failed at {self.url} for {method}") from last

    def latest_number(self) -> int:
        return parse_quantity(self.call("eth_blockNumber", []), "latest block number")

    def header(self, number: int) -> Header:
        return parse_header(self.call("eth_getBlockByNumber", [hex(number), False]))


def parse_quantity(value: Any, label: str) -> int:
    if not isinstance(value, str) or not value.startswith("0x"):
        raise RuntimeError(f"HVEMSD invalid {label}")
    try:
        parsed = int(value, 16)
    except ValueError as exc:
        raise RuntimeError(f"HVEMSD invalid {label}") from exc
    if parsed < 0:
        raise RuntimeError(f"HVEMSD negative {label}")
    return parsed


def parse_header(value: Any) -> Header:
    if not isinstance(value, dict):
        raise RuntimeError("HVEMSD block response is not an object")
    required = ("number", "hash", "parentHash", "timestamp")
    if any(key not in value for key in required):
        raise RuntimeError("HVEMSD block response lacks required fields")
    block_hash = value["hash"]
    parent_hash = value["parentHash"]
    if not isinstance(block_hash, str) or len(block_hash) != 66 or not block_hash.startswith("0x"):
        raise RuntimeError("HVEMSD invalid block hash")
    if not isinstance(parent_hash, str) or len(parent_hash) != 66 or not parent_hash.startswith("0x"):
        raise RuntimeError("HVEMSD invalid parent hash")
    return Header(
        number=parse_quantity(value["number"], "block number"),
        block_hash=block_hash.lower(),
        parent_hash=parent_hash.lower(),
        timestamp=parse_quantity(value["timestamp"], "block timestamp"),
    )


def first_block_at_or_after(
    getter: Callable[[int], Header], target_timestamp: int, low: int, high: int
) -> Header:
    left = getter(low)
    right = getter(high)
    if left.timestamp >= target_timestamp or right.timestamp < target_timestamp:
        raise RuntimeError("HVEMSD search bracket does not contain target")
    while high - low > 1:
        middle = (low + high) // 2
        header = getter(middle)
        if header.timestamp < target_timestamp:
            low = middle
        else:
            high = middle
    answer = getter(high)
    previous = getter(high - 1)
    if answer.timestamp < target_timestamp or previous.timestamp >= target_timestamp:
        raise RuntimeError("HVEMSD first-block search invariant failed")
    if answer.parent_hash != previous.block_hash:
        raise RuntimeError("HVEMSD boundary parent relation failed")
    return answer


def collect_headers(clients: tuple[RpcClient, RpcClient]) -> list[dict[str, Any]]:
    primary, witness = clients
    latest = min(primary.latest_number(), witness.latest_number())
    cache: dict[int, Header] = {}

    def primary_header(number: int) -> Header:
        if number not in cache:
            cache[number] = primary.header(number)
        return cache[number]

    targets = pd.date_range(BOUNDARY_START, BOUNDARY_END, freq="1D")
    records: list[dict[str, Any]] = []
    previous_number = MERGE_BLOCK
    for position, target in enumerate(targets):
        target_seconds = int(target.timestamp())
        low = MERGE_BLOCK if position == 0 else previous_number
        high = latest if position == 0 else min(latest, previous_number + 7_300)
        boundary = first_block_at_or_after(primary_header, target_seconds, low, high)
        anchor_target = target_seconds + 86_400 + CONFIRMATION_SLOTS * 12
        anchor = first_block_at_or_after(
            primary_header,
            anchor_target,
            boundary.number,
            min(latest, boundary.number + 7_400),
        )
        if anchor.timestamp > target_seconds + 86_400 + 20 * 60:
            raise RuntimeError("HVEMSD confirmation anchor is unavailable by decision time")
        for expected in (boundary, anchor):
            observed = witness.header(expected.number)
            if observed != expected:
                raise RuntimeError("HVEMSD dual-RPC header disagreement")
        records.append(
            {
                "target_day": target.strftime("%Y-%m-%d"),
                "boundary": boundary.record(),
                "confirmation_anchor": anchor.record(),
            }
        )
        previous_number = boundary.number
    return records


def normalize_header_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        day = pd.Timestamp(record["target_day"], tz="UTC")
        boundary = parse_header(
            {
                "number": hex(record["boundary"]["number"]),
                "hash": record["boundary"]["hash"],
                "parentHash": record["boundary"]["parentHash"],
                "timestamp": hex(record["boundary"]["timestamp"]),
            }
        )
        anchor = parse_header(
            {
                "number": hex(record["confirmation_anchor"]["number"]),
                "hash": record["confirmation_anchor"]["hash"],
                "parentHash": record["confirmation_anchor"]["parentHash"],
                "timestamp": hex(record["confirmation_anchor"]["timestamp"]),
            }
        )
        if boundary.timestamp < int(day.timestamp()):
            raise RuntimeError("HVEMSD boundary precedes target day")
        if anchor.timestamp < int(day.timestamp()) + 86_400 + CONFIRMATION_SLOTS * 12:
            raise RuntimeError("HVEMSD anchor precedes confirmation target")
        rows.append(
            {
                "source_day": day,
                "boundary_block_number": boundary.number,
                "boundary_block_hash": boundary.block_hash,
                "boundary_timestamp": pd.to_datetime(boundary.timestamp, unit="s", utc=True),
                "confirmation_block_number": anchor.number,
                "confirmation_block_hash": anchor.block_hash,
                "confirmation_timestamp": pd.to_datetime(anchor.timestamp, unit="s", utc=True),
            }
        )
    frame = pd.DataFrame(rows).sort_values("source_day").reset_index(drop=True)
    expected = pd.date_range(BOUNDARY_START, BOUNDARY_END, freq="1D")
    if len(frame) != len(expected) or not frame.source_day.equals(pd.Series(expected, name="source_day")):
        raise RuntimeError("HVEMSD boundary day grid is incomplete")
    produced = frame.boundary_block_number.shift(-1) - frame.boundary_block_number
    frame["produced_blocks"] = produced.astype("Int64")
    frame["missed_slots"] = (SLOTS_PER_DAY - produced).astype("Int64")
    valid = frame.missed_slots.dropna()
    if not valid.between(0, SLOTS_PER_DAY).all():
        raise RuntimeError("HVEMSD daily missed-slot count is invalid")
    return frame


def prior_rank(values: pd.Series, lookback: int, minimum: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = np.asarray(history[-lookback:], dtype=float)
        if np.isfinite(current) and len(prior) >= minimum:
            result.at[index] = ((prior < current).sum() + 0.5 * (prior == current).sum()) / len(prior)
        if np.isfinite(current):
            history.append(float(current))
    return result


def normalize_btc(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.columns.tolist() != ["ts", "open", "close"]:
        raise RuntimeError("HVEMSD BTC schema drift")
    frame = raw.copy()
    frame.ts = pd.to_datetime(frame.ts, utc=True, errors="raise")
    frame = frame.sort_values("ts").reset_index(drop=True)
    expected = pd.date_range(BTC_START, BTC_END, freq="1min", inclusive="left")
    if len(frame) != len(expected) or not frame.ts.equals(pd.Series(expected, name="ts")):
        raise RuntimeError("HVEMSD BTC source is not the exact requested 1m grid")
    for column in ("open", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[["open", "close"]].to_numpy(float)).all():
        raise RuntimeError("HVEMSD BTC source is nonfinite")
    if not frame[["open", "close"]].gt(0).all(axis=None):
        raise RuntimeError("HVEMSD BTC source is nonpositive")
    return frame


def load_btc(env_file: str = ENV_FILE) -> pd.DataFrame:
    from sqlalchemy import create_engine, text

    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(env_file)
    engine = create_engine(postgres_url_from_env(env_file), connect_args={"connect_timeout": 10})
    try:
        raw = pd.read_sql_query(
            text(QUERY),
            engine,
            params={"start": BTC_START.to_pydatetime(), "end": BTC_END.to_pydatetime()},
        )
    finally:
        engine.dispose()
    return normalize_btc(raw)


def build_features(panel: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    frame = panel.loc[panel.produced_blocks.notna(), ["source_day", "produced_blocks", "missed_slots"]].copy()
    frame["produced_blocks"] = frame.produced_blocks.astype(int)
    frame["missed_slots"] = frame.missed_slots.astype(int)
    frame["missed_change"] = frame.missed_slots.diff()
    frame["missed_change_rank"] = prior_rank(frame.missed_change.abs(), 365, 180)
    frame["decision_time"] = frame.source_day + pd.Timedelta(days=1, minutes=20)
    indexed = bars.set_index("ts")
    variations = []
    for decision in frame.decision_time:
        window = indexed.loc[
            (indexed.index >= decision - pd.Timedelta(hours=24)) & (indexed.index < decision)
        ]
        variations.append(
            float(np.sqrt(np.square(np.log(window.close.to_numpy(float) / window.open.to_numpy(float))).sum()))
            if len(window) == 1440
            else np.nan
        )
    frame["btc_variation"] = variations
    frame["state_valid"] = (
        frame.missed_change.notna()
        & frame.missed_change.ne(0)
        & np.isfinite(frame[["missed_change", "missed_change_rank", "btc_variation"]]).all(axis=1)
    )
    frame["btc_variation_rank"] = prior_rank(frame.btc_variation.where(frame.state_valid), 270, 180)
    return frame.reset_index(drop=True)


def signal(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = frame.shift(1) if control == "one_day_stale_change" else frame
    score = used.missed_change.copy()
    valid = used.state_valid.eq(True) & score.ne(0)
    tail = pd.Series(True, index=frame.index) if control == "no_missed_change_tail" else used.missed_change_rank.ge(0.70)
    volatile = pd.Series(True, index=frame.index) if control == "no_btc_volatility_gate" else frame.btc_variation_rank.ge(0.65)
    active = valid & tail & volatile
    if control == "missed_level_change":
        score = used.missed_slots.diff()
        active = used.missed_slots.notna() & score.ne(0) & used.missed_change_rank.ge(0.70) & volatile
    side = -np.sign(score).fillna(0).astype(int)
    if control == "direction_flip":
        side = -side
    elif control == "same_clock_forced_long":
        side = pd.Series(1, index=frame.index)
    active &= side.ne(0)
    return active.fillna(False), side


def build_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = signal(frame, control)
    rows = []
    next_allowed: pd.Timestamp | None = None
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=24)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        next_allowed = exit_time
        used = frame.loc[index - 1] if control == "one_day_stale_change" else frame.loc[index]
        rows.append(
            {
                "candidate": "HVEMSD-24",
                "control": control,
                "split": split,
                "source_day": used.source_day,
                "decision_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(side.at[index]),
                "produced_blocks": int(used.produced_blocks),
                "missed_slots": int(used.missed_slots),
                "missed_change": float(used.missed_change),
                "missed_change_rank": float(used.missed_change_rank),
                "btc_variation": float(frame.at[index, "btc_variation"]),
                "btc_variation_rank": float(frame.at[index, "btc_variation_rank"]),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    sample = clock.loc[clock.split.eq(split)]
    if sample.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(sample.side.eq(1).sum())
    shorts = int(sample.side.eq(-1).sum())
    months = pd.to_datetime(sample.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(sample),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(sample),
        "max_month_share": int(months.max()) / len(sample),
    }


def run(env_file: str = ENV_FILE) -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVEMSD preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    clients = tuple(RpcClient(url) for url in prereg.RPC_HOSTS)
    records = collect_headers(clients)  # type: ignore[arg-type]
    panel = normalize_header_records(records)
    bars = load_btc(env_file)
    features = build_features(panel, bars)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}

    ROOT.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    RAW.write_bytes(canonical_bytes(records, trailing_lf=True))
    _write_gzip_csv(panel, PANEL)
    _write_gzip_csv(bars, BTC_SOURCE)
    _write_gzip_csv(features, FEATURES)
    _write_gzip_csv(primary, CLOCK)
    for name, clock in controls.items():
        _write_gzip_csv(clock, CONTROL_DIR / f"{name}.csv.gz")

    source_core = {
        "protocol_version": "hvemsd_24_sources_v1",
        "preregistration_sha256": PREREG_SHA,
        "rpc_hosts": list(prereg.RPC_HOSTS),
        "raw": {"path": str(RAW), "sha256": sha(RAW), "rows": len(records)},
        "panel": {"path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel)},
        "btc": {"path": str(BTC_SOURCE), "sha256": sha(BTC_SOURCE), "query": QUERY, "rows": len(bars)},
        "features": {"path": str(FEATURES), "sha256": sha(FEATURES), "rows": len(features)},
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "outcomes_opened": False,
        "no_imputation": True,
    }
    manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    support = {name: stats(primary, name) for name in SPLITS}
    checks = {
        key: value
        for name, summary in support.items()
        for key, value in (
            (f"{name}_minimum_events", summary["events"] >= MINIMUM[name]),
            (f"{name}_side_balance", summary["minority_side_share"] >= 0.20),
            (f"{name}_month_concentration", summary["max_month_share"] <= 0.45),
        )
    }
    passed = all(checks.values())
    core = {
        "protocol_version": "hvemsd_24_source_support_v1",
        "policy_id": "HVEMSD-24",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST), "manifest_hash": manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(clock),
                "promotion_authorized": False,
            }
            for name, clock in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=ENV_FILE)
    args = parser.parse_args()
    output = run(args.env_file)
    print(json.dumps({"passed": output["support_passed"], "support": output["support"]}, indent=2))
