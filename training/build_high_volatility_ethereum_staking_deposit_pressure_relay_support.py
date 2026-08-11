"""Materialize outcome-blind source support for frozen HVESDP-24."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import build_ethereum_stablecoin_issuance_redemption as eth
from training import preregister_high_volatility_ethereum_staking_deposit_pressure_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_ethereum_staking_deposit_pressure_relay_support.py")
PREREG_SHA = "3779ce7d9ebf120c92ed979e9d44747b9926c828967413a2cd646a39820dbeef"
ETH_HELPER_SHA = "66936f283ecf8060c32d60767af504475329dfa556fb554596c2e27229f6c952"
SOURCE_DIR = Path("data/high_volatility_ethereum_staking_deposit_pressure_relay_sources_2023_2026")
RAW_EVENTS = SOURCE_DIR / "deposit_events_2022_2026.json.gz"
DAILY_PANEL = SOURCE_DIR / "daily_deposit_counts.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "preentry_states.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_ethereum_staking_deposit_pressure_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_ethereum_staking_deposit_pressure_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_ethereum_staking_deposit_pressure_relay_support_2026-08-12.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_btc_volatility_gate",
    "pressure_direction_flip",
    "one_day_stale_pressure",
    "raw_day_over_day_pressure",
    "same_clock_forced_long",
)
CLOCK_COLUMNS = (
    "candidate",
    "control",
    "split",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
    "daily_count",
    "pressure_change",
    "btc_realized_variation",
    "btc_variation_rank",
)
ABI_LENGTHS = (48, 32, 8, 96, 8)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    text = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        date_format="%Y-%m-%dT%H:%M:%SZ",
    )
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
        handle.write(text.encode())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())


def write_gzip_json(value: Any, path: Path) -> None:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
        handle.write(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())


def strict_prior_midrank(
    values: pd.Series, lookback: int = 270, minimum: int = 180
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if np.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior, dtype=float)
            output.at[index] = (
                np.count_nonzero(array < current)
                + 0.5 * np.count_nonzero(array == current)
            ) / len(array)
        if np.isfinite(current):
            history.append(float(current))
    return output


def decode_deposit_event_data(value: Any) -> tuple[bytes, ...]:
    if not isinstance(value, str) or not value.startswith("0x"):
        raise ValueError("DepositEvent data must be hexadecimal")
    try:
        payload = bytes.fromhex(value[2:])
    except ValueError as error:
        raise ValueError("DepositEvent data is not hexadecimal") from error
    head_size = 32 * len(ABI_LENGTHS)
    if len(payload) < head_size or len(payload) % 32:
        raise ValueError("DepositEvent ABI length is invalid")
    offsets = [int.from_bytes(payload[i * 32 : (i + 1) * 32], "big") for i in range(5)]
    expected_offset = head_size
    fields: list[bytes] = []
    for offset, expected_length in zip(offsets, ABI_LENGTHS, strict=True):
        if offset != expected_offset or offset + 32 > len(payload):
            raise ValueError("DepositEvent ABI offset is noncanonical")
        length = int.from_bytes(payload[offset : offset + 32], "big")
        if length != expected_length:
            raise ValueError("DepositEvent ABI field length differs from frozen schema")
        start = offset + 32
        padded_length = ((length + 31) // 32) * 32
        end = start + padded_length
        if end > len(payload) or any(payload[start + length : end]):
            raise ValueError("DepositEvent ABI padding is invalid")
        fields.append(payload[start : start + length])
        expected_offset = end
    if expected_offset != len(payload):
        raise ValueError("DepositEvent ABI has trailing bytes")
    return tuple(fields)


def normalize_log(raw: dict[str, Any], start_block: int, end_block: int) -> dict[str, Any]:
    address = eth._address(raw.get("address"))
    topics = raw.get("topics")
    if address != prereg.DEPOSIT_CONTRACT.lower():
        raise ValueError("DepositEvent address differs from frozen contract")
    if not isinstance(topics, list) or len(topics) != 1:
        raise ValueError("DepositEvent topic count differs from frozen ABI")
    topic0 = eth._hash(topics[0], "event topic")
    if topic0 != prereg.DEPOSIT_EVENT_TOPIC:
        raise ValueError("DepositEvent topic differs from frozen signature")
    if raw.get("removed") is not False:
        raise RuntimeError("DepositEvent is removed or has ambiguous reorg state")
    block_number = eth._quantity(raw.get("blockNumber"), "event block number")
    if not start_block <= block_number <= end_block:
        raise RuntimeError("Ethereum RPC returned an event outside the requested range")
    fields = decode_deposit_event_data(raw.get("data"))
    return {
        "block_number": block_number,
        "block_hash": eth._hash(raw.get("blockHash"), "event block hash"),
        "transaction_hash": eth._hash(raw.get("transactionHash"), "transaction hash"),
        "transaction_index": eth._quantity(raw.get("transactionIndex"), "transaction index"),
        "log_index": eth._quantity(raw.get("logIndex"), "log index"),
        "data_sha256": hashlib.sha256(bytes.fromhex(str(raw["data"])[2:])).hexdigest(),
        "deposit_index_little_endian": int.from_bytes(fields[4], "little"),
    }


def fetch_logs(rpc: eth.Rpc, start_block: int, end_block: int, max_block_range: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for first in range(start_block, end_block + 1, max_block_range):
        last = min(first + max_block_range - 1, end_block)
        result = rpc.call(
            "eth_getLogs",
            [{
                "fromBlock": hex(first),
                "toBlock": hex(last),
                "address": prereg.DEPOSIT_CONTRACT,
                "topics": [prereg.DEPOSIT_EVENT_TOPIC],
            }],
        )
        if not isinstance(result, list):
            raise RuntimeError("eth_getLogs result is not a list")
        rows.extend(result)
    return rows


def collect_log_source(rpc: eth.Rpc) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chain_id = eth._quantity(rpc.call("eth_chainId", []), "chain id")
    if chain_id != 1:
        raise RuntimeError(f"expected Ethereum mainnet chain id 1, got {chain_id}")
    source = prereg.build()["source_plan"]["ethereum_execution_logs"]
    start = int(pd.Timestamp(source["source_day_start"], tz="UTC").timestamp())
    end = int(pd.Timestamp(source["source_day_end_exclusive"], tz="UTC").timestamp())
    start_header = eth.find_first_block_at_or_after(rpc, start)
    end_header = eth.find_first_block_at_or_after(rpc, end)
    last_source_block = end_header.number - 1
    code_hashes: dict[str, str] = {}
    for label, block in (("start", start_header.number), ("end", last_source_block)):
        code = rpc.call("eth_getCode", [prereg.DEPOSIT_CONTRACT, hex(block)])
        if not isinstance(code, str) or code in {"0x", "0x0"}:
            raise RuntimeError(f"Ethereum deposit contract has no code at {label} source boundary")
        code_hashes[label] = hashlib.sha256(bytes.fromhex(code[2:])).hexdigest()
    raw = fetch_logs(rpc, start_header.number, last_source_block, source["maximum_log_query_block_span"])
    normalized = [normalize_log(row, start_header.number, last_source_block) for row in raw]
    normalized.sort(key=lambda row: (row["block_number"], row["transaction_index"], row["log_index"]))
    identities = [(row["block_hash"], row["transaction_hash"], row["log_index"]) for row in normalized]
    if len(identities) != len(set(identities)):
        raise RuntimeError("Ethereum source contains a duplicate canonical log identity")
    if not normalized:
        raise RuntimeError("Ethereum deposit source returned no events")
    indices = [row["deposit_index_little_endian"] for row in normalized]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise RuntimeError("DepositEvent indices are not unique and increasing")
    audit = {
        "chain_id": chain_id,
        "start_boundary": {"block_number": start_header.number, "block_hash": start_header.block_hash},
        "end_boundary_exclusive": {"block_number": end_header.number, "block_hash": end_header.block_hash},
        "last_source_block": last_source_block,
        "contract_code_sha256": code_hashes,
        "event_rows": len(normalized),
        "canonical_log_hash": canonical_hash(normalized),
    }
    return normalized, audit


def materialize_events(
    rpc: eth.Rpc, normalized: Sequence[dict[str, Any]], last_source_block: int
) -> list[dict[str, Any]]:
    numbers = [row["block_number"] for row in normalized]
    numbers.extend(row["block_number"] + 64 for row in normalized)
    headers = eth.fetch_headers(rpc, numbers, batch_size=100)
    headers.update(eth.fetch_headers(rpc, [last_source_block + 64], batch_size=100))
    finalized = eth.get_header(rpc, "finalized")
    if finalized.number < last_source_block + 64:
        raise RuntimeError("Ethereum finalized head does not cover the complete source interval plus 64 blocks")
    rows: list[dict[str, Any]] = []
    for row in normalized:
        event_header = headers[row["block_number"]]
        confirmation_header = headers[row["block_number"] + 64]
        if event_header.block_hash != row["block_hash"]:
            raise RuntimeError("event block hash differs from canonical header")
        if confirmation_header.timestamp <= event_header.timestamp:
            raise RuntimeError("confirmation block does not follow event block")
        source_day = pd.Timestamp(event_header.timestamp, unit="s", tz="UTC").floor("D")
        decision = source_day + pd.Timedelta(days=1, hours=12)
        available = pd.Timestamp(confirmation_header.timestamp, unit="s", tz="UTC")
        if available > decision:
            raise RuntimeError("DepositEvent was not confirmed by its frozen decision time")
        rows.append({**row, "block_timestamp": eth._format_timestamp(event_header.timestamp), "available_at": eth._format_timestamp(confirmation_header.timestamp), "source_day": source_day.strftime("%Y-%m-%d")})
    return rows


def load_events() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = prereg.build()["source_plan"]["ethereum_execution_logs"]
    clients = [
        eth.JsonRpcClient(source["primary_rpc"], timeout_sec=60, max_retries=6),
        eth.JsonRpcClient(source["verification_rpc"], timeout_sec=60, max_retries=6),
    ]
    first, first_audit = collect_log_source(clients[0])
    second, second_audit = collect_log_source(clients[1])
    if first != second:
        raise RuntimeError("independent Ethereum RPC replays disagree")
    boundary_keys = ("start_boundary", "end_boundary_exclusive", "last_source_block")
    if any(first_audit[key] != second_audit[key] for key in boundary_keys):
        raise RuntimeError("independent Ethereum RPC source boundaries disagree")
    if first_audit["contract_code_sha256"] != second_audit["contract_code_sha256"]:
        raise RuntimeError("independent Ethereum RPC contract bytecode differs")
    rows = materialize_events(clients[0], first, first_audit["last_source_block"])
    write_gzip_json(rows, RAW_EVENTS)
    return rows, {
        "dual_replay_equal": True,
        "transport_bindings": {
            "primary": source["primary_rpc"],
            "verification": source["verification_rpc"],
        },
        "primary_audit": first_audit,
        "verification_audit": second_audit,
        "materialized_event_hash": canonical_hash(rows),
    }


def build_daily_panel(events: Sequence[dict[str, Any]]) -> pd.DataFrame:
    start = pd.Timestamp("2022-01-01T00:00:00Z")
    end = pd.Timestamp("2026-07-30T00:00:00Z")
    counts = pd.Series([row["source_day"] for row in events]).value_counts()
    frame = pd.DataFrame({"source_day": pd.date_range(start, end, inclusive="left", freq="D")})
    frame["daily_count"] = frame.source_day.dt.strftime("%Y-%m-%d").map(counts).fillna(0).astype(int)
    frame["pressure_change"] = frame.daily_count - frame.daily_count.shift(7)
    frame["raw_day_over_day_change"] = frame.daily_count - frame.daily_count.shift(1)
    frame["result_side"] = np.sign(frame.pressure_change).fillna(0).astype(int)
    frame["raw_day_over_day_side"] = np.sign(frame.raw_day_over_day_change).fillna(0).astype(int)
    frame["decision_time"] = frame.source_day + pd.Timedelta(days=1, hours=12)
    return frame.iloc[7:].reset_index(drop=True)


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def variation_query(decisions: pd.Series) -> str:
    literals = ",".join(f"('{pd.Timestamp(item).isoformat()}'::timestamptz)" for item in decisions)
    return f"""WITH decisions(decision_time) AS (VALUES {literals})
SELECT d.decision_time, count(*) source_rows, count(DISTINCT b.ts) distinct_timestamps,
min(b.ts) first_ts, max(b.ts) last_ts, bool_and(b.open>0 AND b.close>0) positive_prices,
sqrt(sum(power(ln(b.close/b.open),2))) realized_variation
FROM decisions d JOIN bars_binance b ON b.symbol='BTCUSDT' AND b.interval='1m'
AND b.ts>=d.decision_time-interval '24 hours' AND b.ts<d.decision_time
GROUP BY d.decision_time ORDER BY d.decision_time"""


def load_variation(groups: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    from sqlalchemy import text

    query = variation_query(groups.decision_time)
    engine = postgres_engine()
    try:
        frame = pd.read_sql_query(text(query), engine)
    finally:
        engine.dispose()
    frame.decision_time = pd.to_datetime(frame.decision_time, utc=True, errors="raise")
    expected = pd.to_datetime(groups.decision_time, utc=True).reset_index(drop=True)
    if len(frame) != len(expected) or not frame.decision_time.equals(expected.rename("decision_time")):
        raise RuntimeError("HVESDP BTC decision grid incomplete")
    valid = frame.source_rows.eq(1440) & frame.distinct_timestamps.eq(1440) & frame.positive_prices.eq(True)
    valid &= pd.to_datetime(frame.first_ts, utc=True).eq(frame.decision_time - pd.Timedelta(days=1))
    valid &= pd.to_datetime(frame.last_ts, utc=True).eq(frame.decision_time - pd.Timedelta(minutes=1))
    frame.realized_variation = pd.to_numeric(frame.realized_variation, errors="coerce")
    valid &= np.isfinite(frame.realized_variation) & frame.realized_variation.gt(0)
    if not valid.all():
        raise RuntimeError("HVESDP invalid BTC variation source")
    return frame, query


def build_features(groups: pd.DataFrame, variation: pd.DataFrame) -> pd.DataFrame:
    frame = groups.merge(variation[["decision_time", "realized_variation"]], on="decision_time", how="left", validate="one_to_one")
    frame.rename(columns={"realized_variation": "btc_realized_variation"}, inplace=True)
    frame["btc_variation_rank"] = strict_prior_midrank(frame.btc_realized_variation)
    return frame


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    side = features.result_side.copy()
    if control == "one_day_stale_pressure":
        side = side.shift(1, fill_value=0)
    if control == "pressure_direction_flip":
        side = -side
    if control == "raw_day_over_day_pressure":
        side = features.raw_day_over_day_side.copy()
    eligible = side.ne(0) & features.btc_variation_rank.ge(0.65)
    if control == "no_btc_volatility_gate":
        eligible = side.ne(0)
    if control == "same_clock_forced_long":
        side = side.where(~eligible, 1)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in features.index[eligible]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=24)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        next_allowed = exit_time
        source_index = index - 1 if control == "one_day_stale_pressure" else index
        rows.append({"candidate": "HVESDP-24", "control": control, "split": split, "decision_time": decision, "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]), "daily_count": float(features.at[source_index, "daily_count"]), "pressure_change": float(features.at[source_index, "pressure_change"]), "btc_realized_variation": float(features.at[index, "btc_realized_variation"]), "btc_variation_rank": float(features.at[index, "btc_variation_rank"])})
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock.split.eq(split)].copy()
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    entries = pd.to_datetime(subset.entry_time, utc=True)
    longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(entries.dt.strftime("%Y-%m").value_counts().max()) / len(subset)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVESDP preregistration hash drift")
    helper = Path("training/build_ethereum_stablecoin_issuance_redemption.py")
    if sha(helper) != ETH_HELPER_SHA:
        raise RuntimeError("HVESDP Ethereum RPC helper hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    events, source_metadata = load_events()
    groups = build_daily_panel(events)
    variation, query = load_variation(groups)
    features = build_features(groups, variation)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    write_gzip_csv(groups, DAILY_PANEL)
    write_gzip_csv(features, FEATURE_PANEL)
    write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {"protocol_version": "hvesdp_24_sources_v1", "ethereum": source_metadata, "source_counts": {"deposit_events": len(events), "source_days": len(groups)}, "btc_query": query, "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)}, "ethereum_rpc_helper": {"path": str(helper), "sha256": ETH_HELPER_SHA}, "outputs": {"raw_events": {"path": str(RAW_EVENTS), "sha256": sha(RAW_EVENTS), "rows": len(events)}, "daily_panel": {"path": str(DAILY_PANEL), "sha256": sha(DAILY_PANEL), "rows": len(groups)}, "features": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(features)}}, "candidate_outcomes_opened": False, "no_imputation": True}
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support_values = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support_values.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {"protocol_version": "hvesdp_24_source_support_v1", "policy_id": "HVESDP-24", "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]}, "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()}, "support": support_values, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
