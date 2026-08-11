"""Materialize outcome-blind source support for frozen HVUWLS-24."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import build_ethereum_stablecoin_issuance_redemption as eth
from training import preregister_high_volatility_uniswap_wbtc_liquidity_shock_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
FROZEN_COMMIT = "c40a57c79c37a903e63482305ebd67b10a4abefc"
BUILDER = Path("training/build_high_volatility_uniswap_wbtc_liquidity_shock_relay_support.py")
PREREG_SOURCE = Path("training/preregister_high_volatility_uniswap_wbtc_liquidity_shock_relay.py")
PREREG_SOURCE_SHA256 = "b342449f28512c79b0204070ef70d1b91b2aff1e11d47386cc86bcda3883ff5c"
PREREG_ARTIFACT_SHA256 = "3761bc07e309e4d9f156cba286c385a5a016df6fcbd99ac4d21bd1c3b85fc9ea"
ETH_HELPER = Path("training/build_ethereum_stablecoin_issuance_redemption.py")
ETH_HELPER_SHA256 = "66936f283ecf8060c32d60767af504475329dfa556fb554596c2e27229f6c952"

SOURCE_DIR = Path("data/high_volatility_uniswap_wbtc_liquidity_shock_relay_sources_2023_2026")
PANEL = SOURCE_DIR / "daily_source_panel.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_uniswap_wbtc_liquidity_shock_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_uniswap_wbtc_liquidity_shock_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_uniswap_wbtc_liquidity_shock_relay_support_2026-08-12.json")

SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_btc_volatility_gate",
    "liquidity_shock_direction_flip",
    "one_day_stale_liquidity_shock",
    "daily_net_wbtc_flow",
    "same_clock_forced_long",
)
CLOCK_COLUMNS = (
    "candidate",
    "control",
    "split",
    "source_day",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
    "amount0_raw",
    "abs_wbtc_amount",
    "shock_magnitude_rank",
    "btc_realized_variation",
    "btc_variation_rank",
    "block_number",
    "transaction_index",
    "log_index",
)

GET_POOL_SELECTOR = "1698ee82"
TOKEN0_SELECTOR = "0dfe1681"
TOKEN1_SELECTOR = "d21220a7"
FEE_SELECTOR = "ddca3f43"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
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


def verify_frozen_preregistration() -> dict[str, Any]:
    if sha256(PREREG_SOURCE) != PREREG_SOURCE_SHA256:
        raise RuntimeError("HVUWLS preregistration source hash drift")
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_ARTIFACT_SHA256:
        raise RuntimeError("HVUWLS preregistration artifact hash drift")
    if sha256(ETH_HELPER) != ETH_HELPER_SHA256:
        raise RuntimeError("HVUWLS Ethereum RPC helper hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    return registration


def _abi_word(value: str, field: str) -> bytes:
    if not isinstance(value, str) or not value.startswith("0x") or len(value) != 66:
        raise RuntimeError(f"{field} is not one ABI word")
    try:
        return bytes.fromhex(value[2:])
    except ValueError as error:
        raise RuntimeError(f"{field} is not hexadecimal") from error


def _call(rpc: eth.Rpc, to: str, data: str, field: str) -> bytes:
    return _abi_word(rpc.call("eth_call", [{"to": to, "data": "0x" + data}, "latest"]), field)


def _decode_address_word(word: bytes, field: str) -> str:
    if len(word) != 32 or any(word[:12]):
        raise RuntimeError(f"{field} has noncanonical address encoding")
    return "0x" + word[12:].hex()


def _address_argument(address: str) -> str:
    normalized = eth._address(address)
    return "0" * 24 + normalized[2:]


def verify_pool_identity(rpc: eth.Rpc, role: str) -> dict[str, Any]:
    chain_id = eth._quantity(rpc.call("eth_chainId", []), f"{role} chain id")
    if chain_id != 1:
        raise RuntimeError(f"{role} RPC is not Ethereum mainnet")
    calldata = (
        GET_POOL_SELECTOR
        + _address_argument(prereg.WBTC)
        + _address_argument(prereg.USDC)
        + f"{3000:064x}"
    )
    pool = _decode_address_word(
        _call(rpc, prereg.FACTORY, calldata, f"{role} factory getPool"),
        f"{role} factory getPool",
    )
    token0 = _decode_address_word(
        _call(rpc, prereg.POOL, TOKEN0_SELECTOR, f"{role} token0"), f"{role} token0"
    )
    token1 = _decode_address_word(
        _call(rpc, prereg.POOL, TOKEN1_SELECTOR, f"{role} token1"), f"{role} token1"
    )
    fee_word = _call(rpc, prereg.POOL, FEE_SELECTOR, f"{role} fee")
    fee = int.from_bytes(fee_word, "big")
    expected = {
        "pool": prereg.POOL.lower(),
        "token0": prereg.WBTC.lower(),
        "token1": prereg.USDC.lower(),
        "fee": 3000,
    }
    actual = {"pool": pool, "token0": token0, "token1": token1, "fee": fee}
    if actual != expected:
        raise RuntimeError(f"{role} Uniswap factory/pool identity differs from frozen policy")
    return {
        "role": role,
        "chain_id": chain_id,
        "factory": prereg.FACTORY.lower(),
        **actual,
        "get_pool_calldata_sha256": hashlib.sha256(bytes.fromhex(calldata)).hexdigest(),
    }


def _hex_bytes(value: Any, length: int, field: str) -> bytes:
    if not isinstance(value, str) or not value.startswith("0x"):
        raise ValueError(f"{field} must be hexadecimal")
    try:
        raw = bytes.fromhex(value[2:])
    except ValueError as error:
        raise ValueError(f"{field} must be hexadecimal") from error
    if len(raw) != length:
        raise ValueError(f"{field} must contain exactly {length} bytes")
    return raw


def _signed_word(word: bytes, bits: int, field: str) -> int:
    raw = int.from_bytes(word, "big")
    mask = (1 << bits) - 1
    low = raw & mask
    value = low - (1 << bits) if low & (1 << (bits - 1)) else low
    if raw != value % (1 << 256):
        raise ValueError(f"{field} has noncanonical signed ABI extension")
    return value


def normalize_swap_log(raw: dict[str, Any], start_block: int, end_block_exclusive: int) -> dict[str, Any]:
    address = eth._address(raw.get("address"))
    if address != prereg.POOL.lower():
        raise ValueError("Swap address differs from frozen pool")
    topics = raw.get("topics")
    if not isinstance(topics, list) or len(topics) != 3:
        raise ValueError("Swap must have exactly three topics")
    topic_bytes = [_hex_bytes(topic, 32, f"Swap topic {index}") for index, topic in enumerate(topics)]
    normalized_topics = ["0x" + item.hex() for item in topic_bytes]
    if normalized_topics[0] != prereg.SWAP_TOPIC0:
        raise ValueError("Swap topic0 differs from frozen signature")
    sender = _decode_address_word(topic_bytes[1], "Swap sender topic")
    recipient = _decode_address_word(topic_bytes[2], "Swap recipient topic")
    data = _hex_bytes(raw.get("data"), 160, "Swap data")
    words = [data[index : index + 32] for index in range(0, len(data), 32)]
    amount0 = _signed_word(words[0], 256, "amount0")
    amount1 = _signed_word(words[1], 256, "amount1")
    if amount0 == 0 or amount1 == 0 or (amount0 > 0) == (amount1 > 0):
        raise ValueError("Swap amounts must be opposite-sign and nonzero")
    sqrt_price_x96 = int.from_bytes(words[2], "big")
    liquidity = int.from_bytes(words[3], "big")
    if sqrt_price_x96 >= 1 << 160:
        raise ValueError("sqrtPriceX96 exceeds uint160")
    if liquidity >= 1 << 128:
        raise ValueError("liquidity exceeds uint128")
    tick = _signed_word(words[4], 24, "tick")
    if raw.get("removed") is not False:
        raise RuntimeError("Swap log is removed or has ambiguous reorg state")
    block_number = eth._quantity(raw.get("blockNumber"), "Swap block number")
    if not start_block <= block_number < end_block_exclusive:
        raise RuntimeError("Ethereum RPC returned a Swap outside the requested range")
    block_hash = eth._hash(raw.get("blockHash"), "Swap block hash")
    transaction_hash = eth._hash(raw.get("transactionHash"), "Swap transaction hash")
    transaction_index = eth._quantity(raw.get("transactionIndex"), "Swap transaction index")
    log_index = eth._quantity(raw.get("logIndex"), "Swap log index")
    return {
        "block_number": block_number,
        "block_hash": block_hash,
        "transaction_hash": transaction_hash,
        "transaction_index": transaction_index,
        "log_index": log_index,
        "sender": sender,
        "recipient": recipient,
        "topics": normalized_topics,
        "topics_sha256": canonical_hash(normalized_topics),
        "data": "0x" + data.hex(),
        "data_sha256": hashlib.sha256(data).hexdigest(),
        "amount0_raw": amount0,
        "amount1_raw": amount1,
        "sqrt_price_x96": sqrt_price_x96,
        "liquidity": liquidity,
        "tick": tick,
    }


def fetch_logs(
    rpc: eth.Rpc, start_block: int, end_block_exclusive: int, max_block_span: int = 2000
) -> list[dict[str, Any]]:
    if max_block_span <= 0 or max_block_span > 2000:
        raise ValueError("log query block span must be in [1, 2000]")
    rows: list[dict[str, Any]] = []
    for first in range(start_block, end_block_exclusive, max_block_span):
        last = min(first + max_block_span, end_block_exclusive) - 1
        result = rpc.call(
            "eth_getLogs",
            [{
                "fromBlock": hex(first),
                "toBlock": hex(last),
                "address": prereg.POOL,
                "topics": [prereg.SWAP_TOPIC0],
            }],
        )
        if not isinstance(result, list):
            raise RuntimeError("eth_getLogs result is not a list")
        rows.extend(result)
    return rows


def _header_record(header: eth.BlockHeader) -> dict[str, Any]:
    return {
        "number": header.number,
        "hash": header.block_hash,
        "parent_hash": header.parent_hash,
        "timestamp": header.timestamp,
    }


def _verify_descendants(rpc: eth.Rpc, boundary: eth.BlockHeader, count: int = 64) -> dict[str, Any]:
    headers = eth.fetch_headers(rpc, range(boundary.number, boundary.number + count + 1), batch_size=100)
    first = headers.get(boundary.number)
    if first != boundary:
        raise RuntimeError("canonical boundary header changed during descendant proof")
    previous = first
    for number in range(boundary.number + 1, boundary.number + count + 1):
        current = headers.get(number)
        if current is None or previous is None or current.parent_hash != previous.block_hash:
            raise RuntimeError("boundary does not have the required canonical descendants")
        previous = current
    assert previous is not None
    return {"descendants": count, "tip": _header_record(previous)}


def find_first_blocks_at_or_after(
    rpc: eth.Rpc, timestamps: Sequence[int], batch_size: int = 100
) -> list[eth.BlockHeader]:
    """Resolve many exact timestamp boundaries with batched binary-search rounds."""
    targets = [int(value) for value in timestamps]
    if not targets:
        return []
    if targets != sorted(targets) or len(targets) != len(set(targets)):
        raise ValueError("Ethereum boundary timestamps must be unique and increasing")
    latest = eth.get_header(rpc, "latest")
    if targets[-1] > latest.timestamp:
        raise RuntimeError("source boundary is later than the Ethereum head")
    low = [0] * len(targets)
    high = [latest.number] * len(targets)
    while any(left < right for left, right in zip(low, high, strict=True)):
        middles = {
            (left + right) // 2
            for left, right in zip(low, high, strict=True)
            if left < right
        }
        headers = eth.fetch_headers(rpc, sorted(middles), batch_size=batch_size)
        for index, target in enumerate(targets):
            if low[index] >= high[index]:
                continue
            middle = (low[index] + high[index]) // 2
            if headers[middle].timestamp < target:
                low[index] = middle + 1
            else:
                high[index] = middle
    numbers = high
    boundaries = eth.fetch_headers(rpc, numbers, batch_size=batch_size)
    previous_numbers = [number - 1 for number in numbers if number]
    previous = eth.fetch_headers(rpc, previous_numbers, batch_size=batch_size)
    output: list[eth.BlockHeader] = []
    for target, number in zip(targets, numbers, strict=True):
        boundary = boundaries[number]
        if boundary.timestamp < target:
            raise RuntimeError("failed to locate Ethereum time boundary")
        if number and previous[number - 1].timestamp >= target:
            raise RuntimeError("Ethereum time boundary is not the first matching block")
        output.append(boundary)
    return output


def collect_host_source(
    rpc: eth.Rpc,
    role: str,
    identity: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    identity = identity or verify_pool_identity(rpc, role)
    source = prereg.build()["source_plan"]["ethereum_uniswap_v3_logs"]
    days = pd.date_range(source["source_day_start"], source["source_day_end_exclusive"], freq="D", inclusive="left", tz="UTC")
    boundary_times = days.append(pd.DatetimeIndex([pd.Timestamp(source["source_day_end_exclusive"], tz="UTC")]))
    boundaries = find_first_blocks_at_or_after(
        rpc, [int(item.timestamp()) for item in boundary_times]
    )
    if any(left.number >= right.number for left, right in zip(boundaries, boundaries[1:])):
        raise RuntimeError("UTC day boundary blocks are not strictly increasing")
    boundary_rows: list[dict[str, Any]] = []
    for day, start, end in zip(days, boundaries, boundaries[1:], strict=True):
        if not (start.timestamp >= int(day.timestamp()) and end.timestamp >= int((day + pd.Timedelta(days=1)).timestamp())):
            raise RuntimeError("UTC day boundary timestamp is incomplete")
        proof = _verify_descendants(rpc, end, prereg.build()["policy"]["confirmation_blocks"])
        boundary_rows.append({
            "source_day": day.strftime("%Y-%m-%d"),
            "start": _header_record(start),
            "end_exclusive": _header_record(end),
            "finality": proof,
        })
    start_block = boundaries[0].number
    end_block = boundaries[-1].number
    raw = fetch_logs(rpc, start_block, end_block, source["maximum_log_query_block_span"])
    logs = [normalize_swap_log(item, start_block, end_block) for item in raw]
    logs.sort(key=lambda row: (row["block_number"], row["transaction_index"], row["log_index"]))
    identities = [(row["block_hash"], row["transaction_hash"], row["log_index"]) for row in logs]
    if len(identities) != len(set(identities)):
        raise RuntimeError("Ethereum source contains duplicate canonical Swap identities")
    canonical_headers = eth.fetch_headers(rpc, sorted({row["block_number"] for row in logs}), batch_size=100)
    for row in logs:
        header = canonical_headers.get(row["block_number"])
        if header is None or header.block_hash != row["block_hash"]:
            raise RuntimeError("Swap block hash differs from canonical block header")
    audit = {
        "role": role,
        "pool_identity": identity,
        "day_rows": len(boundary_rows),
        "zero_log_days_bound_by_headers": True,
        "boundary_hash": canonical_hash(boundary_rows),
        "swap_rows": len(logs),
        "swap_replay_hash": canonical_hash(logs),
        "maximum_log_query_block_span": source["maximum_log_query_block_span"],
    }
    return logs, boundary_rows, audit


def load_chain_source() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = prereg.build()["source_plan"]["ethereum_uniswap_v3_logs"]
    clients = {
        "primary": eth.JsonRpcClient(source["primary_rpc"], timeout_sec=60, max_retries=6),
        "verification": eth.JsonRpcClient(source["verification_rpc"], timeout_sec=60, max_retries=6),
    }
    identities = {
        role: verify_pool_identity(client, role) for role, client in clients.items()
    }
    replay = {
        role: collect_host_source(client, role, identities[role])
        for role, client in clients.items()
    }
    primary_logs, primary_boundaries, primary_audit = replay["primary"]
    verify_logs, verify_boundaries, verify_audit = replay["verification"]
    if primary_logs != verify_logs:
        raise RuntimeError("two-host normalized Swap replay hash/content mismatch")
    if primary_boundaries != verify_boundaries:
        raise RuntimeError("two-host canonical UTC boundary/finality replay mismatch")
    return primary_logs, primary_boundaries, {
        "dual_host_exact_replay": True,
        "role_to_host": {"primary": source["primary_rpc"], "verification": source["verification_rpc"]},
        "primary": primary_audit,
        "verification": verify_audit,
        "exact_replay_hash": canonical_hash({"boundaries": primary_boundaries, "logs": primary_logs}),
    }


def strict_prior_midrank(values: pd.Series, lookback: int = 270, minimum: int = 180) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if np.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior, dtype=float)
            output.at[index] = (
                np.count_nonzero(array < current) + 0.5 * np.count_nonzero(array == current)
            ) / len(array)
        if np.isfinite(current):
            history.append(float(current))
    return output


def build_daily_panel(boundaries: Sequence[dict[str, Any]], logs: Sequence[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for day in boundaries:
        first = int(day["start"]["number"])
        last = int(day["end_exclusive"]["number"])
        selected = [row for row in logs if first <= int(row["block_number"]) < last]
        selected.sort(
            key=lambda row: (
                -abs(int(row["amount0_raw"])),
                int(row["block_number"]),
                int(row["transaction_index"]),
                int(row["log_index"]),
            )
        )
        largest = selected[0] if selected else None
        source_day = pd.Timestamp(day["source_day"], tz="UTC")
        row: dict[str, Any] = {
            "source_day": source_day,
            "decision_time": source_day + pd.Timedelta(days=1, hours=12),
            "source_valid": True,
            "start_block": first,
            "start_block_hash": day["start"]["hash"],
            "end_block_exclusive": last,
            "end_block_hash": day["end_exclusive"]["hash"],
            "confirmation_tip_block": day["finality"]["tip"]["number"],
            "confirmation_tip_hash": day["finality"]["tip"]["hash"],
            "swap_count": len(selected),
            "daily_net_amount0_raw": sum(int(item["amount0_raw"]) for item in selected),
        }
        for field in (
            "amount0_raw", "amount1_raw", "block_number", "block_hash", "transaction_hash",
            "transaction_index", "log_index", "topics_sha256", "data_sha256",
        ):
            row[field] = largest[field] if largest is not None else np.nan
        row["abs_wbtc_amount"] = abs(int(largest["amount0_raw"])) / 1e8 if largest else np.nan
        row["result_side"] = -int(math.copysign(1, int(largest["amount0_raw"]))) if largest else 0
        row["daily_net_side"] = -int(math.copysign(1, row["daily_net_amount0_raw"])) if row["daily_net_amount0_raw"] else 0
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("source_day").reset_index(drop=True)
    frame["shock_magnitude_rank"] = strict_prior_midrank(frame.abs_wbtc_amount.where(frame.source_valid))
    return frame


BTC_QUERY = """
SELECT ts, open, close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_btc_bars(decisions: pd.Series) -> pd.DataFrame:
    from sqlalchemy import text

    times = pd.to_datetime(decisions, utc=True, errors="raise")
    engine = postgres_engine()
    try:
        with engine.connect() as connection:
            connection.execute(text("SET TRANSACTION READ ONLY"))
            bars = pd.read_sql_query(
                text(BTC_QUERY),
                connection,
                params={
                    "start": (times.min() - pd.Timedelta(days=1)).to_pydatetime(),
                    "end": times.max().to_pydatetime(),
                },
            )
    finally:
        engine.dispose()
    if list(bars.columns) != ["ts", "open", "close"]:
        raise RuntimeError("HVUWLS BTC query returned columns outside frozen ts/open/close schema")
    return bars


def calculate_variation(decisions: pd.Series, bars: pd.DataFrame) -> pd.DataFrame:
    market = bars.copy()
    if list(market.columns) != ["ts", "open", "close"]:
        raise ValueError("BTC bars must contain only ts/open/close")
    market["ts"] = pd.to_datetime(market.ts, utc=True, errors="raise")
    if market.ts.duplicated().any():
        raise RuntimeError("HVUWLS duplicate BTC minute timestamp")
    market["open"] = pd.to_numeric(market.open, errors="coerce")
    market["close"] = pd.to_numeric(market.close, errors="coerce")
    market = market.set_index("ts").sort_index()
    rows: list[dict[str, Any]] = []
    for decision in pd.to_datetime(decisions, utc=True, errors="raise"):
        expected = pd.date_range(decision - pd.Timedelta(days=1), decision, freq="1min", inclusive="left")
        window = market.reindex(expected)
        valid = (
            len(window) == 1440
            and window.notna().all(axis=1).all()
            and np.isfinite(window[["open", "close"]]).all(axis=1).all()
            and window[["open", "close"]].gt(0).all(axis=1).all()
        )
        if not valid:
            raise RuntimeError(f"HVUWLS invalid BTC 1m window at {decision.isoformat()}")
        variation = float(np.sqrt(np.square(np.log(window.close / window.open)).sum()))
        if not math.isfinite(variation) or variation <= 0:
            raise RuntimeError(f"HVUWLS invalid BTC variation at {decision.isoformat()}")
        rows.append({"decision_time": decision, "btc_realized_variation": variation})
    return pd.DataFrame(rows)


def build_features(daily: pd.DataFrame, variation: pd.DataFrame) -> pd.DataFrame:
    frame = daily.merge(variation, on="decision_time", how="left", validate="one_to_one")
    if frame.btc_realized_variation.isna().any():
        raise RuntimeError("HVUWLS BTC decision grid incomplete")
    frame["btc_variation_rank"] = strict_prior_midrank(
        frame.btc_realized_variation.where(frame.source_valid)
    )
    return frame


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    frame = features.copy().sort_values("decision_time").reset_index(drop=True)
    amount0 = frame.amount0_raw.copy()
    magnitude = frame.abs_wbtc_amount.copy()
    shock_rank = frame.shock_magnitude_rank.copy()
    side = frame.result_side.copy()
    block_number = frame.block_number.copy()
    transaction_index = frame.transaction_index.copy()
    log_index = frame.log_index.copy()
    if control == "one_day_stale_liquidity_shock":
        amount0 = amount0.shift(1)
        magnitude = magnitude.shift(1)
        shock_rank = shock_rank.shift(1)
        side = side.shift(1, fill_value=0)
        block_number = block_number.shift(1)
        transaction_index = transaction_index.shift(1)
        log_index = log_index.shift(1)
    elif control == "daily_net_wbtc_flow":
        side = frame.daily_net_side.copy()
    eligible = frame.source_valid & side.ne(0) & shock_rank.ge(0.80) & frame.btc_variation_rank.ge(0.65)
    if control == "no_btc_volatility_gate":
        eligible = frame.source_valid & side.ne(0) & shock_rank.ge(0.80)
    if control == "liquidity_shock_direction_flip":
        side = -side
    elif control == "same_clock_forced_long":
        side = side.where(~eligible, 1)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in frame.index[eligible]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=24)
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        reserved_until = exit_time
        rows.append({
            "candidate": "HVUWLS-24",
            "control": control,
            "split": split,
            "source_day": pd.Timestamp(frame.at[index, "source_day"]),
            "decision_time": decision,
            "entry_time": entry,
            "exit_time": exit_time,
            "side": int(side.at[index]),
            "amount0_raw": int(amount0.at[index]),
            "abs_wbtc_amount": float(magnitude.at[index]),
            "shock_magnitude_rank": float(shock_rank.at[index]),
            "btc_realized_variation": float(frame.at[index, "btc_realized_variation"]),
            "btc_variation_rank": float(frame.at[index, "btc_variation_rank"]),
            "block_number": int(block_number.at[index]),
            "transaction_index": int(transaction_index.at[index]),
            "log_index": int(log_index.at[index]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = clock[clock.split.eq(split)]
    if subset.empty:
        return {
            "events": 0, "longs": 0, "shorts": 0,
            "minority_side_share": 0.0, "max_month_share": 0.0,
        }
    entries = pd.to_datetime(subset.entry_time, utc=True)
    longs = int(subset.side.eq(1).sum())
    shorts = int(subset.side.eq(-1).sum())
    return {
        "events": len(subset),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(entries.dt.strftime("%Y-%m").value_counts().max()) / len(subset),
    }


def run() -> dict[str, Any]:
    registration = verify_frozen_preregistration()
    logs, boundaries, chain_audit = load_chain_source()
    daily = build_daily_panel(boundaries, logs)
    bars = load_btc_bars(daily.decision_time)
    variation = calculate_variation(daily.decision_time, bars)
    panel = build_features(daily, variation)
    primary = build_clock(panel)
    controls = {name: build_clock(panel, name) for name in CONTROLS}

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    write_gzip_csv(panel, PANEL)
    write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")

    source_core = {
        "protocol_version": "hvuwls_24_source_materialization_v1",
        "frozen_commit": FROZEN_COMMIT,
        "preregistration": {
            "source_path": str(PREREG_SOURCE),
            "source_sha256": PREREG_SOURCE_SHA256,
            "artifact_path": str(prereg.DEFAULT_OUTPUT),
            "artifact_sha256": PREREG_ARTIFACT_SHA256,
            "manifest_hash": registration["manifest_hash"],
        },
        "builder": {"path": str(BUILDER), "sha256": sha256(BUILDER)},
        "ethereum_rpc_helper": {"path": str(ETH_HELPER), "sha256": ETH_HELPER_SHA256},
        "ethereum": chain_audit,
        "normalized_swap_rows": len(logs),
        "normalized_swap_hash": canonical_hash(logs),
        "canonical_day_rows": len(boundaries),
        "canonical_day_hash": canonical_hash(boundaries),
        "btc_query": BTC_QUERY,
        "btc_columns": ["ts", "open", "close"],
        "btc_rows": len(bars),
        "panel": {"path": str(PANEL), "sha256": sha256(PANEL), "rows": len(panel)},
        "no_imputation": True,
        "candidate_outcomes_opened": False,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(
        json.dumps(source_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )

    support = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "hvuwls_24_source_support_v1",
        "policy_id": "HVUWLS-24",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_ARTIFACT_SHA256,
            "source_sha256": PREREG_SOURCE_SHA256,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST),
            "sha256": sha256(SOURCE_MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "completed_preentry_sources_opened": True,
        "postentry_outcomes_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "diagnostics": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(frame),
                "promotion_authorized": False,
            }
            for name, frame in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_novelty": passed,
        "advance_to_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
