"""Freeze the pre-2024 chain-activity comparator schedule without outcomes.

This exporter reconstructs only the already-frozen decision/entry/exit/side
clock.  It reads market timestamps and closes solely because the frozen signal
definition contains 24-hour and 72-hour price-return signs.  It never loads
funding, calls the trade simulator, or computes trade returns, PnL, equity,
CAGR, or MDD.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, cast

import numpy as np
import pandas as pd


PROTOCOL_VERSION = "chain_activity_comparator_clock_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILDER = Path("training/freeze_chain_activity_comparator_clock.py")
CHAIN_IMPLEMENTATION = Path("training/search_chain_activity_impulse_momentum_alpha.py")
CHAIN_IMPLEMENTATION_SHA256 = (
    "83b8b07ec13eee7b650a030e00da1dda0500c0a4486b6a161da84f777703ab20"
)
CHAIN_MANIFEST = Path(
    "results/chain_activity_impulse_momentum_pre2024_manifest_2026-07-16.json"
)
CHAIN_MANIFEST_SHA256 = (
    "8aabc323d4098fec0a10963fc02465af4ce2e272107257b7c7f62d0aca9fb70e"
)
MARKET = Path("data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz")
MARKET_SHA256 = "cd5e7073248d00c4846a898be23bcd40361aad7245275e48d9905c9b167c6cc3"
NETWORK = Path("data/coinmetrics_btc_network_daily_2020_2023.csv.gz")
NETWORK_SHA256 = "97ab2ca9d0c347d85221b51734f98072763370072ca51f1c40e3214191159b42"
DEFAULT_CLOCK = Path(
    "results/chain_activity_impulse_momentum_pre2024_comparator_clock_2026-07-21.csv.gz"
)
DEFAULT_MANIFEST = Path(
    "results/chain_activity_impulse_momentum_pre2024_comparator_clock_manifest_2026-07-21.json"
)

MARKET_HEADER = (
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
    "tic",
    "day",
)
NETWORK_HEADER = (
    "observation_date",
    "available_at",
    "AdrActCnt",
    "TxCnt",
    "TxTfrCnt",
)
CLOCK_COLUMNS = (
    "window",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
)
WINDOWS = ("fit_2021", "fit_2022", "select_2023")
SELECTION_END = "2024-01-01"
POLICY_WINDOWS = {
    "fit_2021": ("2021-03-01", "2022-01-01"),
    "fit_2022": ("2022-01-01", "2023-01-01"),
    "select_2023": ("2023-01-01", SELECTION_END),
}
EXPECTED_SCHEDULE_HASHES = {
    "fit_2021": "cf19964f7e1ee900871a0af75aa52b4fd34daf37a17f7436bfac3ac296595995",
    "fit_2022": "aa2266685e04a64cf5e3e8aa3a3edd01a210457b700fd3b9c4832fe74248b32c",
    "select_2023": "1012d9b9b2d8ec4e71a1180044ae3fff8a8576bf577f879348a52376fc115211",
}


@dataclass(frozen=True)
class Config:
    market: str = str(MARKET)
    network: str = str(NETWORK)
    output_clock: str = str(DEFAULT_CLOCK)
    output_manifest: str = str(DEFAULT_MANIFEST)


@dataclass(frozen=True)
class ClockEngine:
    market: pd.DataFrame
    dates: pd.Series


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
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


def _require_file(path: str | Path, expected_sha: str, label: str) -> Path:
    target = _repository_path(path)
    if target.is_symlink():
        raise RuntimeError(f"{label} is a symlink")
    if not target.is_file():
        raise RuntimeError(f"{label} is missing")
    if sha256_file(target) != expected_sha:
        raise RuntimeError(f"{label} SHA drift")
    return target


def _gzip_header(path: Path) -> tuple[str, ...]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return tuple(next(csv.reader(handle)))


def _validate_provenance() -> None:
    # The monolithic research module is provenance only and is never imported.
    _require_file(
        CHAIN_IMPLEMENTATION,
        CHAIN_IMPLEMENTATION_SHA256,
        "chain signal implementation",
    )


def _load_policy(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("chain manifest schema drift")
    expected_keys = {
        "phase",
        "selection_end",
        "policy",
        "policy_hash",
        "source_prefix_hashes",
        "schedule_hashes",
        "execution_contract",
        "selection_stats_hash",
        "tested_cells",
        "manifest_hash",
        "created_at",
    }
    if set(manifest) != expected_keys:
        raise RuntimeError("chain manifest top-level schema drift")
    policy = manifest["policy"]
    if (
        not isinstance(policy, dict)
        or canonical_hash(policy) != manifest["policy_hash"]
    ):
        raise RuntimeError("chain policy hash drift")
    if manifest["schedule_hashes"] != {
        **EXPECTED_SCHEDULE_HASHES,
        "select_2023_h1": "c345fe0d18113477a0dce5f38b8610b6b63f683fc1ed7ea6ab7fb274648699cd",
        "select_2023_h2": "c1a730b49588411aa3e1cf1204a991f69b3f7ffdbccf9624761f2dc0dd6946e1",
    }:
        raise RuntimeError("chain schedule-hash contract drift")
    return cast(dict[str, Any], policy), manifest


def _load_market(path: Path, *, cutoff: str) -> pd.DataFrame:
    if _gzip_header(path) != MARKET_HEADER:
        raise RuntimeError("chain market header drift")
    cutoff_timestamp = cast(pd.Timestamp, pd.Timestamp(cutoff, tz="UTC"))
    records: list[tuple[str, str]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MARKET_HEADER:
            raise RuntimeError("chain market header drift")
        for row in reader:
            raw_timestamp = pd.Timestamp(row["date"])
            if bool(pd.isna(raw_timestamp)):
                raise RuntimeError("chain market timestamp is missing")
            timestamp = cast(pd.Timestamp, raw_timestamp)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            if timestamp >= cutoff_timestamp:
                break
            records.append((timestamp.isoformat(), row["close"]))
    frame = pd.DataFrame.from_records(records, columns=("date", "close"))
    frame["date"] = pd.to_datetime(
        frame["date"], utc=True, errors="raise"
    ).dt.tz_convert(None)
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    frame = cast(
        pd.DataFrame,
        frame.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True),
    )
    intervals = frame["date"].diff().dropna()
    if frame.empty or not bool(intervals.eq(pd.Timedelta(minutes=5)).all()):
        raise RuntimeError("chain market is not a complete pre-2024 five-minute grid")
    if not bool(np.isfinite(frame["close"].to_numpy(float)).all()):
        raise RuntimeError("chain market close is not finite")
    return frame


def _load_network(path: Path, *, cutoff: str) -> pd.DataFrame:
    if _gzip_header(path) != NETWORK_HEADER:
        raise RuntimeError("chain network header drift")
    frame = cast(pd.DataFrame, pd.read_csv(path))
    frame["observation_date"] = pd.to_datetime(
        frame["observation_date"], utc=True, errors="raise"
    ).dt.tz_convert(None)
    frame = cast(
        pd.DataFrame,
        frame.loc[frame["observation_date"] < pd.Timestamp(cutoff)]
        .sort_values("observation_date")
        .drop_duplicates("observation_date", keep="last")
        .reset_index(drop=True),
    )
    frame["available_at"] = pd.to_datetime(
        frame["available_at"], utc=True, errors="raise"
    ).dt.tz_convert(None)
    if bool(
        (frame["available_at"] < frame["observation_date"] + pd.Timedelta(days=1)).any()
    ):
        raise RuntimeError("network day was available before the UTC day completed")
    numeric = ("AdrActCnt", "TxCnt", "TxTfrCnt")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if bool((frame.loc[:, numeric] <= 0.0).any().any()):
        raise RuntimeError("chain network contains a non-positive core metric")
    return frame


def _rolling_z(values: pd.Series, window: int = 180) -> pd.Series:
    minimum = window // 2
    mean = values.rolling(window, min_periods=minimum).mean()
    std = values.rolling(window, min_periods=minimum).std(ddof=0).replace(0.0, np.nan)
    return (values - mean) / std


def _build_daily_features(network: pd.DataFrame, engine: ClockEngine) -> pd.DataFrame:
    frame = network[["observation_date", "available_at"]].copy()
    addresses = cast(pd.Series, np.log(cast(pd.Series, network["AdrActCnt"])))
    transactions = cast(pd.Series, np.log(cast(pd.Series, network["TxCnt"])))
    transfers = cast(pd.Series, np.log(cast(pd.Series, network["TxTfrCnt"])))
    activity_1d = (addresses.diff() + transactions.diff() + transfers.diff()) / 3.0
    activity_7d = (addresses.diff(7) + transactions.diff(7) + transfers.diff(7)) / 3.0
    frame["activity_shock_1d"] = _rolling_z(activity_1d)
    frame["activity_shock_7d"] = _rolling_z(activity_7d)

    market_dates = engine.dates.to_numpy(dtype="datetime64[ns]")
    anchors = np.searchsorted(
        market_dates,
        cast(pd.Series, frame["available_at"]).to_numpy(dtype="datetime64[ns]"),
        side="left",
    )
    valid = anchors < len(market_dates)
    frame = frame.loc[valid].copy().reset_index(drop=True)
    frame["anchor"] = anchors[valid].astype(np.int64)
    anchor = cast(pd.Series, frame["anchor"]).to_numpy(dtype=np.int64)
    frame["anchor_date"] = engine.dates.iloc[anchor].to_numpy()
    if bool((frame["anchor_date"] < frame["available_at"]).any()):
        raise RuntimeError("network observation mapped before provider completion")
    close = cast(
        pd.Series,
        pd.to_numeric(cast(pd.Series, engine.market["close"]), errors="raise"),
    ).to_numpy(dtype=np.float64)
    for hours in (24, 72):
        previous = anchor - hours * 12
        values = np.full(len(frame), np.nan)
        usable = previous >= 0
        values[usable] = np.log(close[anchor[usable]] / close[previous[usable]])
        frame[f"price_ret_{hours}h"] = values
    return frame.replace([np.inf, -np.inf], np.nan)


def _event_onset(active: np.ndarray) -> np.ndarray:
    active = np.asarray(active, dtype=bool)
    return active & ~np.r_[False, active[:-1]]


def _policy_masks(
    frame: pd.DataFrame,
    *,
    event: str,
    threshold: float,
    rule: str,
) -> tuple[np.ndarray, np.ndarray]:
    event_value = cast(
        pd.Series,
        pd.to_numeric(cast(pd.Series, frame[event]), errors="coerce"),
    ).to_numpy(dtype=np.float64)
    active = _event_onset(np.isfinite(event_value) & (event_value >= threshold))
    ret24 = cast(pd.Series, frame["price_ret_24h"]).to_numpy(dtype=np.float64)
    ret72 = cast(pd.Series, frame["price_ret_72h"]).to_numpy(dtype=np.float64)
    finite = np.isfinite(ret24) & np.isfinite(ret72)
    zeros = np.zeros(len(frame), dtype=bool)
    if rule == "absorption_long":
        long_active, short_active = (
            active & finite & (ret24 < 0.0) & (ret72 <= 0.0),
            zeros,
        )
    elif rule == "confirmation_long":
        long_active, short_active = (
            active & finite & (ret24 > 0.0) & (ret72 >= 0.0),
            zeros,
        )
    elif rule == "exhaustion_short":
        long_active, short_active = (
            zeros,
            active & finite & (ret24 > 0.0) & (ret72 >= 0.0),
        )
    elif rule == "failure_short":
        long_active, short_active = (
            zeros,
            active & finite & (ret24 < 0.0) & (ret72 <= 0.0),
        )
    elif rule == "momentum":
        long_active, short_active = (
            active & finite & (ret24 > 0.0),
            active & finite & (ret24 < 0.0),
        )
    elif rule == "reversal":
        long_active, short_active = (
            active & finite & (ret24 < 0.0),
            active & finite & (ret24 > 0.0),
        )
    else:
        raise RuntimeError(f"unknown frozen chain rule: {rule}")
    return long_active, short_active


def _schedule_hash(records: list[list[Any]]) -> str:
    payload = json.dumps(records, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _iso_utc(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if bool(pd.isna(timestamp)):
        raise RuntimeError("chain comparator timestamp is missing")
    timestamp = cast(pd.Timestamp, timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def _schedule_rows(
    dates: pd.Series,
    anchors: np.ndarray,
    long_active: np.ndarray,
    short_active: np.ndarray,
    *,
    hold_bars: int,
    windows: dict[str, tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if np.any(long_active & short_active):
        raise RuntimeError("chain policy emitted conflicting sides")
    if not (len(anchors) == len(long_active) == len(short_active)):
        raise RuntimeError("chain policy arrays have inconsistent lengths")
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for window in WINDOWS:
        start, end = windows[window]
        period = (
            (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))
        ).to_numpy(bool)
        records: list[list[Any]] = []
        next_allowed = 0
        for index in np.flatnonzero(long_active | short_active):
            signal_position = int(anchors[index])
            if signal_position < next_allowed or not period[signal_position]:
                continue
            entry_position = signal_position + 1
            exit_position = entry_position + int(hold_bars)
            if exit_position >= len(dates) or not period[exit_position]:
                continue
            side = 1 if bool(long_active[index]) else -1
            entry_date = str(dates.iloc[entry_position])
            records.append(
                [
                    signal_position,
                    entry_position,
                    exit_position,
                    side,
                    entry_date,
                ]
            )
            rows.append(
                {
                    "window": window,
                    "decision_time": _iso_utc(dates.iloc[signal_position]),
                    "entry_time": _iso_utc(dates.iloc[entry_position]),
                    "exit_time": _iso_utc(dates.iloc[exit_position]),
                    "side": side,
                }
            )
            next_allowed = exit_position + 1
        hashes[window] = _schedule_hash(records)
    return rows, hashes


def _validate_clock(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(rows, columns=CLOCK_COLUMNS)
    if frame.empty or tuple(frame.columns) != CLOCK_COLUMNS:
        raise RuntimeError("chain comparator clock is empty or malformed")
    if bool(frame.duplicated(["window", "decision_time", "entry_time"]).any()):
        raise RuntimeError("chain comparator clock contains duplicate rows")
    if not bool(frame["side"].astype(int).isin((-1, 1)).all()):
        raise RuntimeError("chain comparator clock contains an invalid side")
    decision = pd.to_datetime(frame["decision_time"], utc=True)
    entry = pd.to_datetime(frame["entry_time"], utc=True)
    exit_time = pd.to_datetime(frame["exit_time"], utc=True)
    if not bool(decision.lt(entry).all() and entry.lt(exit_time).all()):
        raise RuntimeError("chain comparator interval is invalid")
    return frame.sort_values(
        ["window", "decision_time", "entry_time"], kind="mergesort"
    ).reset_index(drop=True)


def _gzip_csv_bytes(frame: pd.DataFrame) -> bytes:
    raw = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
        handle.write(raw)
    return buffer.getvalue()


def _atomic_write(path: Path, payload: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
        path.chmod(mode)
    finally:
        temporary.unlink(missing_ok=True)


def build_clock(cfg: Config | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    frozen_cfg = Config() if cfg is None else cfg
    chain_manifest = _require_file(
        CHAIN_MANIFEST, CHAIN_MANIFEST_SHA256, "chain manifest"
    )
    market_path = _require_file(frozen_cfg.market, MARKET_SHA256, "chain market")
    network_path = _require_file(frozen_cfg.network, NETWORK_SHA256, "chain network")
    _validate_provenance()
    policy, source_manifest = _load_policy(chain_manifest)

    cutoff = str(source_manifest["selection_end"])
    if cutoff != SELECTION_END:
        raise RuntimeError("chain selection cutoff drift")
    market = _load_market(market_path, cutoff=cutoff)
    network = _load_network(network_path, cutoff=cutoff)
    engine = ClockEngine(market=market, dates=cast(pd.Series, market["date"]))
    features = _build_daily_features(network, engine)
    long_active, short_active = _policy_masks(
        features,
        event=policy["event"],
        threshold=float(policy["threshold"]),
        rule=policy["rule"],
    )
    rows, schedule_hashes = _schedule_rows(
        engine.dates,
        features["anchor"].to_numpy(int),
        long_active,
        short_active,
        hold_bars=int(policy["hold_days"]) * 288,
        windows=POLICY_WINDOWS,
    )
    if schedule_hashes != EXPECTED_SCHEDULE_HASHES:
        raise RuntimeError(
            "chain comparator replay does not match frozen schedule hashes"
        )
    frame = _validate_clock(rows)
    clock_bytes = _gzip_csv_bytes(frame)
    counts = {
        str(name): int(value)
        for name, value in frame["window"].value_counts().sort_index().items()
    }
    manifest_core = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": "chain_activity_impulse_momentum",
        "config": asdict(frozen_cfg),
        "builder": {"path": str(BUILDER), "sha256": sha256_file(BUILDER)},
        "inputs": {
            "chain_manifest": {
                "path": str(CHAIN_MANIFEST),
                "sha256": sha256_file(CHAIN_MANIFEST),
            },
            "chain_implementation": {
                "path": str(CHAIN_IMPLEMENTATION),
                "sha256": sha256_file(CHAIN_IMPLEMENTATION),
            },
            "market": {
                "path": frozen_cfg.market,
                "sha256": sha256_file(market_path),
                "columns_read": ["date", "close"],
            },
            "network": {
                "path": frozen_cfg.network,
                "sha256": sha256_file(network_path),
                "columns_read": list(NETWORK_HEADER),
            },
        },
        "frozen_policy_hash": source_manifest["policy_hash"],
        "frozen_source_prefix_hashes": source_manifest["source_prefix_hashes"],
        "schedule_hashes": schedule_hashes,
        "clock": {
            "path": frozen_cfg.output_clock,
            "sha256": hashlib.sha256(clock_bytes).hexdigest(),
            "rows": int(len(frame)),
            "counts": counts,
            "columns": list(CLOCK_COLUMNS),
        },
        "outcome_boundary": {
            "funding_rows_loaded": 0,
            "high_low_open_columns_loaded": 0,
            "trade_simulator_called": False,
            "trade_return_or_pnl_computed": False,
            "equity_cagr_mdd_computed": False,
            "post_2023_rows_loaded": 0,
            "signal_price_return_features_derived": [
                "price_ret_24h",
                "price_ret_72h",
            ],
        },
        "next_action": "consume only this clock in the CDLTR comparator normalizer",
    }
    manifest = {**manifest_core, "manifest_hash": canonical_hash(manifest_core)}
    return frame, manifest


def write_clock(cfg: Config | None = None) -> dict[str, Any]:
    frozen_cfg = Config() if cfg is None else cfg
    clock_path = _repository_path(frozen_cfg.output_clock)
    manifest_path = _repository_path(frozen_cfg.output_manifest)
    if clock_path.exists() or clock_path.is_symlink():
        raise FileExistsError("chain comparator clock is immutable")
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError("chain comparator manifest is immutable")
    frame, manifest = build_clock(frozen_cfg)
    clock_bytes = _gzip_csv_bytes(frame)
    if hashlib.sha256(clock_bytes).hexdigest() != manifest["clock"]["sha256"]:
        raise RuntimeError("chain comparator clock hash drift before write")
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _atomic_write(clock_path, clock_bytes, mode=0o444)
    try:
        _atomic_write(manifest_path, manifest_bytes, mode=0o444)
    except Exception:
        clock_path.chmod(0o644)
        clock_path.unlink(missing_ok=True)
        raise
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", default=str(MARKET))
    parser.add_argument("--network", default=str(NETWORK))
    parser.add_argument("--output-clock", default=str(DEFAULT_CLOCK))
    parser.add_argument("--output-manifest", default=str(DEFAULT_MANIFEST))
    return parser.parse_args()


def main() -> None:
    manifest = write_clock(Config(**vars(parse_args())))
    print(
        json.dumps(
            {
                "clock": manifest["clock"],
                "schedule_hashes": manifest["schedule_hashes"],
                "outcomes_computed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
