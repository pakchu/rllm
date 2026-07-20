"""Strict sequential pre-2024 evaluator for frozen BATE-288.

The evaluator has three irreversible stages: evaluator freeze, 2021-2022
train, and 2023 selection.  The train stage physically skips pre-2021 values
and stops before parsing any 2023 market or funding value.  The selection
stage remains sealed until the write-once train artifact exactly replays and
passes every preregistered gate.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training import build_block_arrival_throughput_elasticity_support as support_builder


SUPPORT_COMMIT = "a6212c7ea1c961acf2657c91566aba7fc5ce6187"
PREREGISTRATION = Path(
    "docs/block-arrival-throughput-elasticity-bate288-support-"
    "preregistration-2026-07-20.md"
)
PREREGISTRATION_SHA256 = (
    "bdb8a9b7e602ca671fb496045a59741a5dd77fea7ed11a66c374e5fd6b5893cd"
)
SUPPORT_SOURCE = Path("training/build_block_arrival_throughput_elasticity_support.py")
SUPPORT_SOURCE_SHA256 = (
    "91d32f887d5e34a5f1292bbdaa287520db0116a4c828323f09b827b2e9e4edfa"
)
SUPPORT_DOCUMENT = Path(
    "docs/block-arrival-throughput-elasticity-bate288-support-pass-2026-07-20.md"
)
SUPPORT_DOCUMENT_SHA256 = (
    "1005f512ded09434060b27ed4ce51bb02876e6e2f9d01e5aeb8ec160753f2a52"
)
SUPPORT_RESULT = Path(
    "results/block_arrival_throughput_elasticity_support_2026-07-20.json"
)
SUPPORT_RESULT_SHA256 = (
    "42598a24853b1d66f2e91a259b2a23e5939a1d0a640abafb4e087e3f209caefc"
)
PRIMARY_CLOCK = Path(
    "results/block_arrival_throughput_elasticity_clock_2026-07-20.csv"
)
PRIMARY_CLOCK_SHA256 = (
    "cd4fbd01c104bd969ca1c12a53b8da82dd0e9376990e233c286ff009a5115c02"
)
BLOCK_SOURCE = Path("data/bitcoin_block_summaries_2020_2023.csv.gz")
BLOCK_SOURCE_SHA256 = (
    "1f8d1c0153717f2c18d8fa6f09428c780a850b2aecc8fb42bc497e16e68e1833"
)
BLOCK_SOURCE_MANIFEST = Path(
    "results/bitcoin_block_summaries_source_manifest_2026-07-20.json"
)
BLOCK_SOURCE_MANIFEST_SHA256 = (
    "9b1c3a81d607632267fe4c87857b2e80d381d3bab90fca7bb0b7df0061775983"
)
MARKET_DATA = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
MARKET_DATA_SHA256 = (
    "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
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
EVALUATION_SOURCE = Path(
    "training/evaluate_block_arrival_throughput_elasticity_pre2024.py"
)
FREEZE_SOURCE = Path("training/freeze_block_arrival_throughput_elasticity_evaluator.py")
EVALUATION_FREEZE = Path(
    "results/block_arrival_throughput_elasticity_evaluator_freeze_2026-07-20.json"
)
TRAIN_OUTPUT = Path(
    "results/block_arrival_throughput_elasticity_train_2021_2022_2026-07-20.json"
)
SELECTION_OUTPUT = Path(
    "results/block_arrival_throughput_elasticity_selection_2023_2026-07-20.json"
)

TRAIN_START = pd.Timestamp("2021-01-01")
TRAIN_END = pd.Timestamp("2023-01-01")
SELECTION_END = pd.Timestamp("2024-01-01")
WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2021-01-01", "2023-01-01"),
    "train_2021": ("2021-01-01", "2022-01-01"),
    "train_2022": ("2022-01-01", "2023-01-01"),
    "selection_2023": ("2023-01-01", "2024-01-01"),
    "selection_2023_h1": ("2023-01-01", "2023-07-01"),
    "selection_2023_h2": ("2023-07-01", "2024-01-01"),
}
STAGE_WINDOWS: dict[str, tuple[str, tuple[str, ...]]] = {
    "train": ("train", ("train_2021", "train_2022")),
    "selection": (
        "selection_2023",
        ("selection_2023_h1", "selection_2023_h2"),
    ),
}
POLICY_NAMES = (
    "primary",
    "direction_flip",
    "weight_only",
    "tx_only",
    "denominator_free",
    "stale_24h",
    "random_clock",
    "one_bar_delayed_entry",
)
MECHANISM_REJECTION_CONTROLS = (
    "weight_only",
    "tx_only",
    "denominator_free",
)
CONTROL_SEMANTICS = {
    "primary": "frozen concordant throughput onset clock and frozen side",
    "direction_flip": "primary clock with every execution side multiplied by -1",
    "weight_only": (
        "weight-z one-channel HIGH/LOW onsets, then frozen confirmation and scheduler"
    ),
    "tx_only": (
        "transaction-z one-channel HIGH/LOW onsets, then frozen confirmation and scheduler"
    ),
    "denominator_free": (
        "concordant robust-z of log six-block mean weight and transaction count, "
        "without elapsed-time denominator"
    ),
    "stale_24h": (
        "primary execution clock and side delayed by exactly 288 five-minute bars"
    ),
    "random_clock": (
        "year-stratified deterministic random composition of all available five-minute "
        "slack around non-overlapping 288-bar holds, with primary year/side counts"
    ),
    "one_bar_delayed_entry": (
        "primary execution clock and side delayed by one five-minute bar"
    ),
}


@dataclass(frozen=True)
class EvaluationConfig:
    leverage: float = 1.0
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_720
    random_clock_seed: int = 73_100_288
    stale_bars: int = 288
    additional_delay_bars: int = 1
    minimum_mean_gross_underlying_bp: float = 30.0
    minimum_cagr_to_strict_mdd: float = 3.0
    maximum_strict_mdd_pct: float = 15.0
    maximum_weekly_cluster_p: float = 0.10


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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


def seal_result(core: dict[str, Any]) -> dict[str, Any]:
    return {**core, "result_hash": canonical_hash(core)}


def validate_result_hash(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "result_hash"}
    if canonical_hash(core) != payload.get("result_hash"):
        raise ValueError("BATE-288 result hash mismatch")


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _parse_primary_clock(path: Path) -> pd.DataFrame:
    clock = pd.read_csv(path)
    required = {
        "policy_id",
        "side",
        "state",
        "packet_end_height",
        "confirmation_height",
        "entry_time",
        "exit_time",
    }
    missing = required.difference(clock.columns)
    if missing:
        raise ValueError(f"BATE-288 primary clock lacks columns: {sorted(missing)}")
    for column in ("entry_time", "exit_time"):
        clock[column] = pd.to_datetime(clock[column], utc=True, errors="raise").dt.tz_convert(None)
    return clock


def _normalize_clock(clock: pd.DataFrame, *, name: str) -> pd.DataFrame:
    normalized = clock.copy()
    normalized = normalized.rename(columns={"entry_time": "entry_date", "exit_time": "exit_date"})
    for column in ("entry_date", "exit_date"):
        normalized[column] = pd.to_datetime(
            normalized[column], utc=True, errors="raise"
        ).dt.tz_convert(None)
    normalized["control"] = name
    if "state" not in normalized:
        normalized["state"] = np.where(normalized["side"].astype(int) > 0, "HIGH", "LOW")
    keep = ["control", "side", "state", "entry_date", "exit_date"]
    for optional in ("packet_end_height", "confirmation_height"):
        if optional in normalized:
            keep.append(optional)
    normalized = normalized[keep]
    normalized = normalized.loc[
        normalized["entry_date"].ge(TRAIN_START)
        & normalized["entry_date"].lt(SELECTION_END)
    ]
    return normalized.sort_values("entry_date").reset_index(drop=True)


def _clock_hash(clock: pd.DataFrame) -> str:
    columns = ["control", "side", "state", "entry_date", "exit_date"]
    rows = [
        {
            "control": str(row.control),
            "side": int(row.side),
            "state": str(row.state),
            "entry_date": pd.Timestamp(row.entry_date).isoformat(),
            "exit_date": pd.Timestamp(row.exit_date).isoformat(),
        }
        for row in clock[columns].itertuples(index=False)
    ]
    return canonical_hash(rows)


def _validate_clock(
    clock: pd.DataFrame,
    *,
    name: str,
    policy: support_builder.Policy,
) -> None:
    if clock.empty:
        raise ValueError(f"BATE-288 {name} clock is empty")
    if not clock["entry_date"].is_monotonic_increasing:
        raise ValueError(f"BATE-288 {name} clock is not sorted")
    if clock["entry_date"].duplicated().any():
        raise ValueError(f"BATE-288 {name} clock has duplicate entries")
    if not clock["side"].isin([-1, 1]).all():
        raise ValueError(f"BATE-288 {name} clock has an invalid side")
    if not clock["state"].isin(["HIGH", "LOW"]).all():
        raise ValueError(f"BATE-288 {name} clock has an invalid state label")
    hold = pd.Timedelta(seconds=policy.hold_bars * policy.bar_seconds)
    if not (clock["exit_date"] - clock["entry_date"]).eq(hold).all():
        raise ValueError(f"BATE-288 {name} hold changed")
    if clock["entry_date"].min() < TRAIN_START or clock["entry_date"].max() >= SELECTION_END:
        raise ValueError(f"BATE-288 {name} clock crossed the frozen entry interval")
    epoch_seconds = clock["entry_date"].astype("int64") // 1_000_000_000
    if not np.equal(epoch_seconds % policy.bar_seconds, 0).all():
        raise ValueError(f"BATE-288 {name} entries are not five-minute aligned")
    if len(clock) > 1:
        current = clock["entry_date"].iloc[1:].reset_index(drop=True)
        previous_exit = clock["exit_date"].iloc[:-1].reset_index(drop=True)
        if not current.ge(previous_exit).all():
            raise ValueError(f"BATE-288 {name} clock overlaps")


def _state_clock(
    features: pd.DataFrame,
    state: np.ndarray,
    state_valid: np.ndarray,
    *,
    name: str,
    policy: support_builder.Policy,
) -> pd.DataFrame:
    controlled = features.copy()
    controlled["state"] = np.asarray(state, dtype=np.int8)
    controlled["state_valid"] = np.asarray(state_valid, dtype=bool)
    controlled["onset"] = support_builder._state_onsets(
        controlled["state"].to_numpy(np.int8),
        controlled["state_valid"].to_numpy(bool),
    )
    return _normalize_clock(support_builder.schedule_clock(controlled, policy), name=name)


def _component_state(values: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    state = np.zeros(len(values), dtype=np.int8)
    state[valid & (values >= threshold)] = 1
    state[valid & (values <= -threshold)] = -1
    return state, valid


def _denominator_free_state(
    blocks: pd.DataFrame,
    features: pd.DataFrame,
    policy: support_builder.Policy,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = blocks.sort_values("height").reset_index(drop=True)
    packet_index = np.arange(
        policy.packet_blocks,
        len(ordered) - policy.confirmation_blocks,
        dtype=np.int64,
    )
    if len(packet_index) != len(features):
        raise ValueError("BATE-288 denominator-free packet alignment changed")
    if not np.array_equal(
        ordered["height"].to_numpy(np.int64)[packet_index],
        features["packet_end_height"].to_numpy(np.int64),
    ):
        raise ValueError("BATE-288 denominator-free heights do not align")
    weight = ordered["weight"].to_numpy(float)
    tx_count = ordered["tx_count"].to_numpy(float)
    weight_cumulative = np.concatenate(([0.0], np.cumsum(weight)))
    tx_cumulative = np.concatenate(([0.0], np.cumsum(tx_count)))
    start = packet_index - policy.packet_blocks + 1
    weight_mean = (
        weight_cumulative[packet_index + 1] - weight_cumulative[start]
    ) / policy.packet_blocks
    tx_mean = (
        tx_cumulative[packet_index + 1] - tx_cumulative[start]
    ) / policy.packet_blocks
    weight_z = support_builder._strict_prior_robust_z(
        np.log(weight_mean),
        reference=policy.reference_packets,
        consistency_scale=policy.mad_consistency_scale,
    )
    tx_z = support_builder._strict_prior_robust_z(
        np.log(tx_mean),
        reference=policy.reference_packets,
        consistency_scale=policy.mad_consistency_scale,
    )
    valid = np.isfinite(weight_z) & np.isfinite(tx_z)
    state = np.zeros(len(features), dtype=np.int8)
    state[valid & (weight_z >= policy.z_threshold) & (tx_z >= policy.z_threshold)] = 1
    state[valid & (weight_z <= -policy.z_threshold) & (tx_z <= -policy.z_threshold)] = -1
    return state, valid


def _shift_clock(
    clock: pd.DataFrame,
    *,
    name: str,
    bars: int,
    policy: support_builder.Policy,
) -> pd.DataFrame:
    shifted = clock.copy()
    delta = pd.Timedelta(seconds=bars * policy.bar_seconds)
    shifted["entry_date"] += delta
    shifted["exit_date"] += delta
    shifted["control"] = name
    shifted = shifted.loc[shifted["entry_date"].lt(SELECTION_END)]
    return shifted.sort_values("entry_date").reset_index(drop=True)


def _year_stratified_random_clock(
    primary: pd.DataFrame,
    *,
    cfg: EvaluationConfig,
    policy: support_builder.Policy,
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_clock_seed)
    hold = pd.Timedelta(seconds=policy.hold_bars * policy.bar_seconds)
    rows: list[dict[str, Any]] = []
    for year in (2021, 2022, 2023):
        year_primary = primary.loc[primary["entry_date"].dt.year.eq(year)]
        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year + 1, month=1, day=1)
        bar = pd.Timedelta(seconds=policy.bar_seconds)
        available_bars = int((end - start) / bar)
        occupied_bars = len(year_primary) * policy.hold_bars
        if occupied_bars > available_bars:
            raise ValueError(f"BATE-288 random clock lacks {year} support")
        slack_bars = available_bars - occupied_bars
        slack = rng.multinomial(
            slack_bars,
            np.full(len(year_primary) + 1, 1.0 / (len(year_primary) + 1)),
        )
        sides = year_primary["side"].to_numpy(np.int8).copy()
        rng.shuffle(sides)
        entry = start + int(slack[0]) * bar
        for index, side in enumerate(sides):
            rows.append(
                {
                    "control": "random_clock",
                    "side": int(side),
                    "state": "HIGH" if side > 0 else "LOW",
                    "entry_date": entry,
                    "exit_date": entry + hold,
                }
            )
            entry += hold + int(slack[index + 1]) * bar
        if entry != end:
            raise ValueError(f"BATE-288 random clock did not consume {year} lattice")
    return pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)


def build_control_clocks(
    primary_clock: pd.DataFrame,
    features: pd.DataFrame,
    blocks: pd.DataFrame,
    *,
    cfg: EvaluationConfig,
    policy: support_builder.Policy,
) -> dict[str, pd.DataFrame]:
    primary = _normalize_clock(primary_clock, name="primary")
    controls: dict[str, pd.DataFrame] = {"primary": primary}

    direction_flip = primary.copy()
    direction_flip["side"] *= -1
    direction_flip["control"] = "direction_flip"
    controls["direction_flip"] = direction_flip

    weight_state, weight_valid = _component_state(
        features["weight_z"].to_numpy(float), policy.z_threshold
    )
    controls["weight_only"] = _state_clock(
        features,
        weight_state,
        weight_valid,
        name="weight_only",
        policy=policy,
    )
    tx_state, tx_valid = _component_state(
        features["tx_z"].to_numpy(float), policy.z_threshold
    )
    controls["tx_only"] = _state_clock(
        features,
        tx_state,
        tx_valid,
        name="tx_only",
        policy=policy,
    )
    denominator_state, denominator_valid = _denominator_free_state(
        blocks, features, policy
    )
    controls["denominator_free"] = _state_clock(
        features,
        denominator_state,
        denominator_valid,
        name="denominator_free",
        policy=policy,
    )
    controls["stale_24h"] = _shift_clock(
        primary,
        name="stale_24h",
        bars=cfg.stale_bars,
        policy=policy,
    )
    controls["random_clock"] = _year_stratified_random_clock(
        primary, cfg=cfg, policy=policy
    )
    controls["one_bar_delayed_entry"] = _shift_clock(
        primary,
        name="one_bar_delayed_entry",
        bars=cfg.additional_delay_bars,
        policy=policy,
    )

    if tuple(controls) != POLICY_NAMES:
        raise ValueError("BATE-288 control order changed")
    for name, clock in controls.items():
        _validate_clock(clock, name=name, policy=policy)
    primary_year_side = (
        primary.assign(year=primary["entry_date"].dt.year)
        .groupby(["year", "side"])
        .size()
        .sort_index()
    )
    random_year_side = (
        controls["random_clock"]
        .assign(year=controls["random_clock"]["entry_date"].dt.year)
        .groupby(["year", "side"])
        .size()
        .sort_index()
    )
    if not primary_year_side.equals(random_year_side):
        raise ValueError("BATE-288 random clock changed year/side counts")
    return controls


def verify_support_and_control_clocks(
    cfg: EvaluationConfig | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frozen_cfg = EvaluationConfig() if cfg is None else cfg
    if frozen_cfg != EvaluationConfig():
        raise ValueError("BATE-288 evaluation parameters are frozen")
    frozen_files = (
        (PREREGISTRATION, PREREGISTRATION_SHA256),
        (SUPPORT_SOURCE, SUPPORT_SOURCE_SHA256),
        (SUPPORT_DOCUMENT, SUPPORT_DOCUMENT_SHA256),
        (SUPPORT_RESULT, SUPPORT_RESULT_SHA256),
        (PRIMARY_CLOCK, PRIMARY_CLOCK_SHA256),
        (BLOCK_SOURCE, BLOCK_SOURCE_SHA256),
        (BLOCK_SOURCE_MANIFEST, BLOCK_SOURCE_MANIFEST_SHA256),
    )
    for path, expected in frozen_files:
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen BATE-288 dependency changed: {path}")
    support = _read_json(SUPPORT_RESULT)
    support_core = {key: value for key, value in support.items() if key != "result_hash"}
    if support_builder.canonical_hash(support_core) != support.get("result_hash"):
        raise ValueError("BATE-288 support result hash mismatch")
    if support.get("outcomes_opened") is not False:
        raise ValueError("BATE-288 support artifact opened outcomes")
    if support.get("policy") != asdict(support_builder.Policy()):
        raise ValueError("BATE-288 support policy changed")
    if support.get("support_gate", {}).get("passed") is not True:
        raise ValueError("BATE-288 source/support gate did not pass")
    source = support.get("source", {})
    if (
        source.get("market_or_funding_rows_loaded") != 0
        or source.get("return_or_pnl_fields_loaded") != 0
        or source.get("post_2023_source_rows_loaded") != 0
    ):
        raise ValueError("BATE-288 support crossed its outcome boundary")
    if support.get("clock", {}).get("sha256") != PRIMARY_CLOCK_SHA256:
        raise ValueError("BATE-288 support clock hash changed")

    policy = support_builder.Policy()
    primary = _parse_primary_clock(PRIMARY_CLOCK)
    if not primary["policy_id"].eq(policy.policy_id).all():
        raise ValueError("BATE-288 primary policy id changed")
    expected_side = primary["state"].map({"HIGH": 1, "LOW": -1})
    if expected_side.isna().any() or not primary["side"].eq(expected_side.astype(int)).all():
        raise ValueError("BATE-288 primary side/state mapping changed")
    blocks, _ = support_builder.load_source(
        str(BLOCK_SOURCE), str(BLOCK_SOURCE_MANIFEST)
    )
    features = support_builder.build_features(blocks, policy)
    controls = build_control_clocks(
        primary,
        features,
        blocks,
        cfg=frozen_cfg,
        policy=policy,
    )
    if len(controls["primary"]) != support["clock"]["rows"]:
        raise ValueError("BATE-288 primary clock count changed")
    return controls, support


def _scan_market_timestamps(path: Path) -> dict[str, Any]:
    counts = {"train": 0, "selection": 0}
    total = 0
    first: str | None = None
    last: str | None = None
    sealed_boundary_seen = False
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        if not header or header[0] != "date":
            raise ValueError("BATE-288 market date must be the first physical column")
        for line in handle:
            date_text = line.split(",", 1)[0]
            if len(date_text) < 10 or date_text[4] != "-" or date_text[7] != "-":
                raise ValueError("BATE-288 market timestamp text is invalid")
            if last is not None and date_text <= last:
                raise ValueError("BATE-288 market timestamps are not strictly increasing")
            first = date_text if first is None else first
            last = date_text
            total += 1
            if date_text >= "2024-01-01":
                sealed_boundary_seen = True
                break
            if "2021-01-01" <= date_text < "2023-01-01":
                counts["train"] += 1
            elif "2023-01-01" <= date_text < "2024-01-01":
                counts["selection"] += 1
    if not sealed_boundary_seen:
        raise ValueError("BATE-288 market timestamps did not reach sealed 2024")
    return {
        "timestamp_column": "date",
        "timestamp_rows_scanned": total,
        "value_rows_parsed": 0,
        "first_timestamp": first,
        "last_timestamp": last,
        "sealed_2024_boundary_seen": sealed_boundary_seen,
        "window_value_row_counts": counts,
    }


def _scan_funding_timestamps(path: Path) -> dict[str, Any]:
    counts = {"train": 0, "selection": 0}
    total = 0
    first: int | None = None
    last: int | None = None
    train_start_ms = int(TRAIN_START.timestamp() * 1_000)
    train_end_ms = int(TRAIN_END.timestamp() * 1_000)
    selection_end_ms = int(SELECTION_END.timestamp() * 1_000)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        if not header or header[0] != "funding_time_ms":
            raise ValueError("BATE-288 funding timestamp must be the first physical column")
        for line in handle:
            timestamp_ms = int(line.split(",", 1)[0])
            if last is not None and timestamp_ms <= last:
                raise ValueError("BATE-288 funding timestamps are not strictly increasing")
            first = timestamp_ms if first is None else first
            last = timestamp_ms
            total += 1
            if train_start_ms <= timestamp_ms < train_end_ms:
                counts["train"] += 1
            elif train_end_ms <= timestamp_ms < selection_end_ms:
                counts["selection"] += 1
    return {
        "timestamp_column": "funding_time_ms",
        "timestamp_rows_scanned": total,
        "value_rows_parsed": 0,
        "first_timestamp_ms": first,
        "last_timestamp_ms": last,
        "window_value_row_counts": counts,
    }


def scan_outcome_boundaries() -> dict[str, Any]:
    if sha256_file(MARKET_DATA) != MARKET_DATA_SHA256:
        raise ValueError("BATE-288 market data differs from its frozen hash")
    if sha256_file(FUNDING_DATA) != FUNDING_DATA_SHA256:
        raise ValueError("BATE-288 funding data differs from its frozen hash")
    if sha256_file(FUNDING_MANIFEST) != FUNDING_MANIFEST_SHA256:
        raise ValueError("BATE-288 funding manifest differs from its frozen hash")
    manifest = _validate_funding_manifest()
    funding = _scan_funding_timestamps(FUNDING_DATA)
    if funding["timestamp_rows_scanned"] != manifest["data"]["rows"]:
        raise ValueError("BATE-288 funding timestamp row count differs from manifest")
    if funding["last_timestamp_ms"] != manifest["data"]["last_funding_time_ms"]:
        raise ValueError("BATE-288 funding terminal timestamp differs from manifest")
    selection_end = pd.Timestamp(manifest["selection_end_exclusive"])
    if selection_end != SELECTION_END or manifest["config"].get("interval") != "8h":
        raise ValueError("BATE-288 funding terminal interval contract changed")
    expected_terminal = int(
        (SELECTION_END - pd.Timedelta(hours=8)).timestamp() * 1_000
    )
    if funding["last_timestamp_ms"] != expected_terminal:
        raise ValueError("BATE-288 funding source lacks audited final 2023 settlement")
    return {"market": _scan_market_timestamps(MARKET_DATA), "funding": funding}


def verify_evaluation_freeze(
    cfg: EvaluationConfig | None = None,
) -> dict[str, Any]:
    frozen_cfg = EvaluationConfig() if cfg is None else cfg
    if frozen_cfg != EvaluationConfig():
        raise ValueError("BATE-288 evaluation parameters are frozen")
    if not EVALUATION_FREEZE.is_file():
        raise ValueError("BATE-288 evaluator freeze is missing")
    payload = _read_json(EVALUATION_FREEZE)
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise ValueError("BATE-288 evaluator freeze manifest hash mismatch")
    if payload.get("outcomes_opened") is not False or payload.get("opened_windows") != []:
        raise ValueError("BATE-288 evaluator was not frozen before outcomes")
    if payload.get("evaluation_source") != str(EVALUATION_SOURCE):
        raise ValueError("BATE-288 evaluator source path changed")
    if payload.get("evaluation_source_sha256") != sha256_file(EVALUATION_SOURCE):
        raise ValueError("BATE-288 evaluator differs from its pre-outcome freeze")
    if payload.get("freeze_source") != str(FREEZE_SOURCE):
        raise ValueError("BATE-288 freeze-tool source path changed")
    if payload.get("freeze_source_sha256") != sha256_file(FREEZE_SOURCE):
        raise ValueError("BATE-288 freeze tool differs from its pre-outcome freeze")
    if payload.get("support_commit") != SUPPORT_COMMIT:
        raise ValueError("BATE-288 support commit changed")
    if payload.get("sealed_windows") != [
        "train_2021_2022",
        "selection_2023",
        "2024",
        "2025",
        "2026_ytd",
    ]:
        raise ValueError("BATE-288 evaluator sealed windows changed")
    if payload.get("mutable_parameters") != []:
        raise ValueError("BATE-288 evaluator freeze permits mutable parameters")
    if payload.get("market_value_rows_parsed_during_freeze") != 0:
        raise ValueError("BATE-288 evaluator freeze parsed market values")
    if payload.get("funding_value_rows_parsed_during_freeze") != 0:
        raise ValueError("BATE-288 evaluator freeze parsed funding values")
    if payload.get("execution_simulation_run_during_freeze") is not False:
        raise ValueError("BATE-288 evaluator freeze simulated execution")
    if payload.get("evaluation_config") != asdict(frozen_cfg):
        raise ValueError("BATE-288 evaluator configuration changed")
    if payload.get("policy_names") != list(POLICY_NAMES):
        raise ValueError("BATE-288 evaluator control set changed")
    if payload.get("control_semantics") != CONTROL_SEMANTICS:
        raise ValueError("BATE-288 evaluator control semantics changed")
    controls, _ = verify_support_and_control_clocks(frozen_cfg)
    actual_hashes = {name: _clock_hash(clock) for name, clock in controls.items()}
    actual_counts = {name: int(len(clock)) for name, clock in controls.items()}
    if payload.get("control_clock_hashes") != actual_hashes:
        raise ValueError("BATE-288 control clock differs from pre-outcome freeze")
    if payload.get("control_clock_counts") != actual_counts:
        raise ValueError("BATE-288 control clock counts differ from freeze")
    if payload.get("outcome_boundaries") != scan_outcome_boundaries():
        raise ValueError("BATE-288 outcome timestamp boundaries changed")
    return payload


def _parse_market_window(path: Path, *, start: str, end: str) -> pd.DataFrame:
    rows: list[tuple[str, float, float, float, float]] = []
    end_boundary_seen = False
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        required = ["date", "open", "high", "low", "close"]
        positions = {column: header.index(column) for column in required}
        if positions["date"] != 0:
            raise ValueError("BATE-288 market date must be the first physical column")
        for line in handle:
            date_text = line.split(",", 1)[0]
            if date_text < start:
                continue
            if date_text >= end:
                end_boundary_seen = True
                break
            fields = next(csv.reader([line]))
            rows.append(
                (
                    date_text,
                    float(fields[positions["open"]]),
                    float(fields[positions["high"]]),
                    float(fields[positions["low"]]),
                    float(fields[positions["close"]]),
                )
            )
    if not end_boundary_seen:
        raise ValueError(f"BATE-288 market source did not reach sealed boundary {end}")
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])


def _validate_funding_manifest() -> dict[str, Any]:
    if sha256_file(FUNDING_MANIFEST) != FUNDING_MANIFEST_SHA256:
        raise ValueError("BATE-288 funding manifest differs from its frozen hash")
    manifest = _read_json(FUNDING_MANIFEST)
    if manifest.get("outcomes_opened") is not False:
        raise ValueError("BATE-288 funding source lacks unopened-source provenance")
    if manifest.get("strategy_outcomes_calculated") != []:
        raise ValueError("BATE-288 funding source calculated a strategy outcome")
    if manifest.get("data", {}).get("sha256") != FUNDING_DATA_SHA256:
        raise ValueError("BATE-288 funding manifest data hash differs")
    if manifest.get("quality", {}).get("events") != manifest.get("data", {}).get("rows"):
        raise ValueError("BATE-288 funding event count differs from manifest")
    maximum_error = manifest.get("quality", {}).get(
        "maximum_proxy_funding_cash_error_bp_notional", float("inf")
    )
    allowed_error = manifest.get("mapping", {}).get(
        "maximum_allowed_proxy_funding_cash_error_bp_notional", -1.0
    )
    if maximum_error > allowed_error:
        raise ValueError("BATE-288 funding-mark proxy error exceeds frozen limit")
    return manifest


def _parse_funding_window(
    path: Path,
    *,
    start: str,
    end: str,
    audited_eof_last_timestamp_ms: int | None = None,
) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start).timestamp() * 1_000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1_000)
    rows: list[tuple[int, str, str, str, str, str, str]] = []
    end_boundary_seen = False
    last_timestamp_ms: int | None = None
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        required = [
            "funding_time_ms",
            "funding_time_utc",
            "symbol",
            "funding_rate",
            "settlement_mark_price",
            "funding_time_offset_ms",
            "mark_source",
        ]
        positions = {column: header.index(column) for column in required}
        if positions["funding_time_ms"] != 0:
            raise ValueError("BATE-288 funding timestamp must be first")
        for line in handle:
            timestamp_ms = int(line.split(",", 1)[0])
            last_timestamp_ms = timestamp_ms
            if timestamp_ms < start_ms:
                continue
            if timestamp_ms >= end_ms:
                end_boundary_seen = True
                break
            fields = next(csv.reader([line]))
            rows.append(
                (
                    timestamp_ms,
                    fields[positions["funding_time_utc"]],
                    fields[positions["symbol"]],
                    fields[positions["funding_rate"]],
                    fields[positions["settlement_mark_price"]],
                    fields[positions["funding_time_offset_ms"]],
                    fields[positions["mark_source"]],
                )
            )
    if not end_boundary_seen and last_timestamp_ms != audited_eof_last_timestamp_ms:
        raise ValueError(
            f"BATE-288 funding source did not reach a physical or audited boundary {end}"
        )
    return pd.DataFrame(
        rows,
        columns=[
            "funding_time_ms",
            "funding_time_utc",
            "symbol",
            "funding_rate",
            "settlement_mark_price",
            "funding_time_offset_ms",
            "mark_source",
        ],
    )


def load_market_window(
    stage: str, freeze: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    full_window, _ = STAGE_WINDOWS[stage]
    start, end = WINDOWS[full_window]
    if sha256_file(MARKET_DATA) != MARKET_DATA_SHA256:
        raise ValueError("BATE-288 market data differs from its frozen hash")
    market = _parse_market_window(MARKET_DATA, start=start, end=end)
    expected_rows = freeze["outcome_boundaries"]["market"][
        "window_value_row_counts"
    ][stage]
    if len(market) != expected_rows:
        raise ValueError("BATE-288 market window row count differs from freeze")
    market["date"] = pd.to_datetime(market["date"], errors="raise")
    if (
        market.empty
        or market["date"].min() < pd.Timestamp(start)
        or market["date"].max() >= pd.Timestamp(end)
    ):
        raise ValueError("BATE-288 market interval is invalid")
    if market["date"].duplicated().any() or not market["date"].is_monotonic_increasing:
        raise ValueError("BATE-288 market timestamps are invalid")
    prices = market[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError("BATE-288 market contains invalid prices")
    opens, highs, lows, closes = (
        market[column].to_numpy(float) for column in ("open", "high", "low", "close")
    )
    if (
        (highs < np.maximum(opens, closes)).any()
        or (lows > np.minimum(opens, closes)).any()
        or (highs < lows).any()
    ):
        raise ValueError("BATE-288 market violates OHLC invariants")
    return market, {
        "sha256": MARKET_DATA_SHA256,
        "rows": int(len(market)),
        "columns_parsed": ["date", "open", "high", "low", "close"],
        "physical_value_window": f"{start} <= date < {end}",
        "values_before_start_parsed": 0,
        "values_at_or_after_end_parsed": 0,
        "first_date": str(market["date"].iloc[0]),
        "last_date": str(market["date"].iloc[-1]),
    }


def load_funding_window(
    stage: str, freeze: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = _validate_funding_manifest()
    full_window, _ = STAGE_WINDOWS[stage]
    start, end = WINDOWS[full_window]
    if sha256_file(FUNDING_DATA) != FUNDING_DATA_SHA256:
        raise ValueError("BATE-288 funding data differs from its frozen hash")
    funding = _parse_funding_window(
        FUNDING_DATA,
        start=start,
        end=end,
        audited_eof_last_timestamp_ms=(
            int(manifest["data"]["last_funding_time_ms"])
            if end == "2024-01-01"
            else None
        ),
    )
    expected_rows = freeze["outcome_boundaries"]["funding"][
        "window_value_row_counts"
    ][stage]
    if len(funding) != expected_rows:
        raise ValueError("BATE-288 funding window row count differs from freeze")
    funding["funding_time_ms"] = pd.to_numeric(
        funding["funding_time_ms"], errors="raise"
    ).astype(np.int64)
    utc = pd.to_datetime(
        funding["funding_time_utc"], utc=True, errors="raise"
    ).dt.tz_convert(None)
    epoch = pd.to_datetime(
        funding["funding_time_ms"], unit="ms", utc=True, errors="raise"
    ).dt.tz_convert(None)
    if not utc.equals(epoch):
        raise ValueError("BATE-288 funding timestamps disagree")
    if funding["funding_time_ms"].duplicated().any() or not funding[
        "funding_time_ms"
    ].is_monotonic_increasing:
        raise ValueError("BATE-288 funding timestamps are invalid")
    if not funding["symbol"].eq("BTCUSDT").all():
        raise ValueError("BATE-288 funding contains another symbol")
    if not funding["mark_source"].eq("binance_8h_mark_price_kline_open").all():
        raise ValueError("BATE-288 funding uses another mark-price source")
    offsets = pd.to_numeric(
        funding["funding_time_offset_ms"], errors="raise"
    ).to_numpy(np.int64)
    maximum_offset = manifest["mapping"]["maximum_allowed_timestamp_offset_ms"]
    if (offsets < 0).any() or (offsets > maximum_offset).any():
        raise ValueError("BATE-288 funding timestamps exceed mark tolerance")
    rates = pd.to_numeric(funding["funding_rate"], errors="raise").to_numpy(float)
    marks = pd.to_numeric(
        funding["settlement_mark_price"], errors="raise"
    ).to_numpy(float)
    if not np.isfinite(rates).all() or not np.isfinite(marks).all() or (marks <= 0.0).any():
        raise ValueError("BATE-288 funding values are invalid")
    normalized = pd.DataFrame(
        {
            "funding_time_ms": funding["funding_time_ms"].to_numpy(np.int64),
            "funding_time": utc,
            "funding_rate": rates,
            "settlement_mark_price": marks,
        }
    )
    if not normalized.empty and (
        normalized["funding_time"].min() < pd.Timestamp(start)
        or normalized["funding_time"].max() >= pd.Timestamp(end)
    ):
        raise ValueError("BATE-288 funding interval is invalid")
    return normalized, {
        "manifest_sha256": FUNDING_MANIFEST_SHA256,
        "data_sha256": FUNDING_DATA_SHA256,
        "rows": int(len(normalized)),
        "physical_value_window": f"{start} <= funding_time < {end}",
        "values_before_start_parsed": 0,
        "values_at_or_after_end_parsed": 0,
        "first_funding_time": (
            str(normalized["funding_time"].iloc[0]) if len(normalized) else None
        ),
        "last_funding_time": (
            str(normalized["funding_time"].iloc[-1]) if len(normalized) else None
        ),
    }


def _slice_schedule(schedule: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    inside = (
        schedule["entry_date"].ge(start_timestamp)
        & schedule["entry_date"].lt(end_timestamp)
        & schedule["exit_date"].gt(start_timestamp)
        & schedule["exit_date"].lt(end_timestamp)
    )
    return schedule.loc[inside].reset_index(drop=True)


def attach_market_positions(schedule: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    attached = schedule.copy()
    positions = pd.Series(np.arange(len(market), dtype=np.int64), index=market["date"])
    for label in ("entry", "exit"):
        mapped = attached[f"{label}_date"].map(positions)
        if mapped.isna().any():
            missing = attached.loc[mapped.isna(), f"{label}_date"].head().tolist()
            raise ValueError(f"BATE-288 {label} timestamps missing from market: {missing}")
        attached[f"{label}_position"] = mapped.astype(np.int64)
    if len(attached) and not (attached["entry_position"] < attached["exit_position"]).all():
        raise ValueError("BATE-288 market positions violate the frozen clock")
    return attached


def weekly_cluster_sign_flip(
    values: Iterable[float],
    entry_dates: Iterable[pd.Timestamp | str],
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    returns = np.asarray(list(values), dtype=float)
    dates = pd.to_datetime(list(entry_dates))
    if len(returns) == 0 or len(returns) != len(dates):
        return {
            "p_value_one_sided": 1.0,
            "observed_mean_return": 0.0,
            "cluster_count": 0,
            "method": "empty",
            "permutations": 0,
            "seed": int(seed),
        }
    frame = pd.DataFrame({"week": dates.to_period("W-SUN"), "return": returns})
    clusters = frame.groupby("week", sort=True)["return"].sum().to_numpy(float)
    observed = float(np.sum(clusters) / len(returns))
    if len(clusters) <= 18:
        outcomes = np.fromiter(
            (
                np.dot(signs, clusters) / len(returns)
                for signs in product((-1.0, 1.0), repeat=len(clusters))
            ),
            dtype=float,
        )
        p_value = float(np.mean(outcomes >= observed - 1e-15))
        method = "exact"
        completed = int(len(outcomes))
    else:
        rng = np.random.default_rng(seed)
        exceedances = 0
        completed = 0
        while completed < permutations:
            batch = min(10_000, permutations - completed)
            signs = rng.integers(0, 2, size=(batch, len(clusters)), dtype=np.int8)
            randomized = (signs.astype(float) * 2.0 - 1.0).dot(clusters) / len(returns)
            exceedances += int(np.count_nonzero(randomized >= observed - 1e-15))
            completed += batch
        p_value = float((1 + exceedances) / (permutations + 1))
        method = "monte_carlo"
    return {
        "p_value_one_sided": p_value,
        "observed_mean_return": observed,
        "cluster_count": int(len(clusters)),
        "method": method,
        "permutations": completed,
        "seed": int(seed),
    }


def _trade_statistics(values: list[float]) -> dict[str, Any]:
    count = len(values)
    if not count:
        return {
            "n_trades": 0,
            "mean_trade_return_pct": 0.0,
            "std_trade_return_pct": 0.0,
            "t_stat_like": 0.0,
            "ci95_mean_trade_return_pct": [0.0, 0.0],
        }
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if count > 1 else 0.0
    standard_error = std / math.sqrt(count)
    return {
        "n_trades": count,
        "mean_trade_return_pct": mean * 100.0,
        "std_trade_return_pct": std * 100.0,
        "t_stat_like": mean / standard_error if standard_error > 0.0 else 0.0,
        "ci95_mean_trade_return_pct": [
            (mean - 1.96 * standard_error) * 100.0,
            (mean + 1.96 * standard_error) * 100.0,
        ],
    }


def simulate_schedule(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    start: str,
    end: str,
    cost_notional_per_side: float,
    cfg: EvaluationConfig,
    compute_cluster: bool,
) -> dict[str, Any]:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if start_timestamp >= end_timestamp or cfg.leverage <= 0.0:
        raise ValueError("BATE-288 simulation parameters are invalid")
    per_side_cost = cost_notional_per_side * cfg.leverage
    if not 0.0 <= per_side_cost < 1.0:
        raise ValueError("BATE-288 per-side cost is invalid")
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    dates = market["date"]
    market_ms = (dates.astype("int64") // 1_000_000).to_numpy(np.int64)
    funding_times = funding["funding_time_ms"].to_numpy(np.int64)
    funding_rates = funding["funding_rate"].to_numpy(float)
    funding_marks = funding["settlement_mark_price"].to_numpy(float)

    equity = 1.0
    peak = 1.0
    strict_mdd = 0.0
    previous_exit = -1
    trade_returns: list[float] = []
    gross_returns: list[float] = []
    entry_dates: list[pd.Timestamp] = []
    sides: list[int] = []
    settlement_count = 0
    trades_with_funding = 0
    funding_cash_sum = 0.0
    entry_equity_sum = 0.0

    for row in schedule.itertuples(index=False):
        entry_position = int(row.entry_position)
        exit_position = int(row.exit_position)
        side = int(row.side)
        if side not in (-1, 1):
            raise ValueError("BATE-288 side must be long or short")
        if not 0 <= entry_position < exit_position < len(market):
            raise ValueError("BATE-288 scheduled positions are invalid")
        if entry_position < previous_exit:
            raise ValueError("BATE-288 schedules overlap")
        entry_time = pd.Timestamp(row.entry_date)
        exit_time = pd.Timestamp(row.exit_date)
        if entry_time != dates.iloc[entry_position] or exit_time != dates.iloc[exit_position]:
            raise ValueError("BATE-288 schedule timestamp differs from market")
        if not (
            start_timestamp <= entry_time < end_timestamp
            and start_timestamp < exit_time < end_timestamp
        ):
            raise ValueError("BATE-288 trade crosses a simulation split")

        entry_price = float(opens[entry_position])
        exit_price = float(opens[exit_position])
        held_high = float(np.max(highs[entry_position:exit_position]))
        held_low = float(np.min(lows[entry_position:exit_position]))
        if min(entry_price, exit_price, held_high, held_low) <= 0.0:
            raise ValueError("BATE-288 scheduled trade has an invalid price")

        entry_ms = int(market_ms[entry_position])
        exit_ms = int(market_ms[exit_position])
        left = int(np.searchsorted(funding_times, entry_ms, side="left"))
        right = int(np.searchsorted(funding_times, exit_ms, side="left"))
        rates = funding_rates[left:right]
        marks = funding_marks[left:right]
        funding_contributions = -cfg.leverage * side * rates * (marks / entry_price)
        funding_return = float(np.sum(funding_contributions, dtype=float))
        funding_credit = float(np.sum(np.maximum(funding_contributions, 0.0), dtype=float))
        funding_debit = float(np.sum(np.minimum(funding_contributions, 0.0), dtype=float))

        entry_equity = equity
        entry_equity_sum += entry_equity
        funding_cash_sum += entry_equity * funding_return
        peak = max(peak, equity)
        equity = entry_equity * (1.0 - per_side_cost)
        strict_mdd = max(strict_mdd, 1.0 - max(0.0, equity) / peak)

        favorable_price = held_high if side > 0 else held_low
        adverse_price = held_low if side > 0 else held_high
        favorable_equity = entry_equity * max(
            0.0,
            1.0
            - per_side_cost
            + cfg.leverage * side * (favorable_price / entry_price - 1.0)
            + funding_credit,
        )
        intratrade_peak = max(peak, favorable_equity)
        adverse_equity = entry_equity * max(
            0.0,
            1.0
            - per_side_cost
            + cfg.leverage * side * (adverse_price / entry_price - 1.0)
            + funding_debit
            - per_side_cost * (adverse_price / entry_price),
        )
        strict_mdd = max(strict_mdd, 1.0 - max(0.0, adverse_equity) / intratrade_peak)
        peak = intratrade_peak

        gross_return = side * (exit_price / entry_price - 1.0)
        equity = entry_equity * max(
            0.0,
            1.0
            - per_side_cost
            + cfg.leverage * gross_return
            + funding_return
            - per_side_cost * (exit_price / entry_price),
        )
        strict_mdd = max(strict_mdd, 1.0 - max(0.0, equity) / peak)
        peak = max(peak, equity)

        trade_returns.append(equity / entry_equity - 1.0 if entry_equity > 0.0 else -1.0)
        gross_returns.append(gross_return)
        entry_dates.append(entry_time)
        sides.append(side)
        settlement_count += int(len(rates))
        trades_with_funding += int(len(rates) > 0)
        previous_exit = exit_position

    years = (end_timestamp - start_timestamp).total_seconds() / (365.25 * 86_400.0)
    absolute_return = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0.0 else -100.0
    strict_mdd_pct = strict_mdd * 100.0
    if strict_mdd_pct > 1e-12:
        ratio = float(cagr / strict_mdd_pct)
        zero_mdd_ratio_cap_applied = False
    elif cagr > 0.0:
        ratio = 1.0e12
        zero_mdd_ratio_cap_applied = True
    else:
        ratio = 0.0
        zero_mdd_ratio_cap_applied = False
    cluster = (
        weekly_cluster_sign_flip(
            trade_returns,
            entry_dates,
            permutations=cfg.cluster_permutations,
            seed=cfg.cluster_seed,
        )
        if compute_cluster
        else None
    )
    return {
        "absolute_return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(strict_mdd_pct),
        "cagr_to_strict_mdd": ratio,
        "zero_mdd_ratio_cap_applied": zero_mdd_ratio_cap_applied,
        "trade_count": int(len(sides)),
        "long_count": int(sum(side > 0 for side in sides)),
        "short_count": int(sum(side < 0 for side in sides)),
        "wall_clock_years": float(years),
        "mean_gross_underlying_move_bp": (
            float(np.mean(gross_returns) * 10_000.0) if gross_returns else 0.0
        ),
        "funding_settlement_count": int(settlement_count),
        "trades_with_funding": int(trades_with_funding),
        "total_funding_cash_pct_of_entry_equity_sum": float(
            100.0 * funding_cash_sum / entry_equity_sum
            if entry_equity_sum > 0.0
            else 0.0
        ),
        "execution_cost_notional_per_side_bp": float(cost_notional_per_side * 10_000.0),
        "trade_statistics": _trade_statistics(trade_returns),
        "weekly_cluster_sign_flip": cluster,
    }


def _evaluate_policy_stage(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    clock: pd.DataFrame,
    *,
    stage: str,
    cfg: EvaluationConfig,
    compute_cluster: bool,
) -> dict[str, Any]:
    full_name, split_names = STAGE_WINDOWS[stage]
    full_start, full_end = WINDOWS[full_name]
    full_clock = _slice_schedule(clock, start=full_start, end=full_end)
    attached = attach_market_positions(full_clock, market)
    base = simulate_schedule(
        market,
        funding,
        attached,
        start=full_start,
        end=full_end,
        cost_notional_per_side=cfg.base_cost_notional_per_side,
        cfg=cfg,
        compute_cluster=compute_cluster,
    )
    stress = simulate_schedule(
        market,
        funding,
        attached,
        start=full_start,
        end=full_end,
        cost_notional_per_side=cfg.stress_cost_notional_per_side,
        cfg=cfg,
        compute_cluster=False,
    )
    splits: dict[str, Any] = {}
    for name in split_names:
        start, end = WINDOWS[name]
        split = _slice_schedule(attached, start=start, end=end)
        splits[name] = simulate_schedule(
            market,
            funding,
            split,
            start=start,
            end=end,
            cost_notional_per_side=cfg.base_cost_notional_per_side,
            cfg=cfg,
            compute_cluster=False,
        )
    sides: dict[str, Any] = {}
    for label, side in (("HIGH_long", 1), ("LOW_short", -1)):
        side_clock = attached.loc[attached["side"].eq(side)].reset_index(drop=True)
        sides[label] = simulate_schedule(
            market,
            funding,
            side_clock,
            start=full_start,
            end=full_end,
            cost_notional_per_side=cfg.base_cost_notional_per_side,
            cfg=cfg,
            compute_cluster=False,
        )
    return {
        "support_calendar_event_count": int(
            len(clock.loc[
                clock["entry_date"].ge(pd.Timestamp(full_start))
                & clock["entry_date"].lt(pd.Timestamp(full_end))
            ])
        ),
        "fully_contained_trade_count": int(len(attached)),
        "boundary_crossing_events_excluded": int(
            len(clock.loc[
                clock["entry_date"].ge(pd.Timestamp(full_start))
                & clock["entry_date"].lt(pd.Timestamp(full_end))
            ])
            - len(attached)
        ),
        "base_6bp": base,
        "stress_10bp": stress,
        "splits_base_6bp": splits,
        "side_contributions_base_6bp": sides,
    }


def stage_gate_failures(
    policy_result: dict[str, Any],
    *,
    stage: str,
    cfg: EvaluationConfig,
) -> list[str]:
    base = policy_result["base_6bp"]
    stress = policy_result["stress_10bp"]
    failures: list[str] = []
    if base["absolute_return_pct"] <= 0.0:
        failures.append(f"{stage}: non-positive absolute return")
    if base["cagr_to_strict_mdd"] < cfg.minimum_cagr_to_strict_mdd:
        failures.append(f"{stage}: CAGR/strict-MDD below 3")
    if base["strict_mdd_pct"] > cfg.maximum_strict_mdd_pct:
        failures.append(f"{stage}: strict MDD above 15%")
    if stress["absolute_return_pct"] <= 0.0:
        failures.append(f"{stage}: 10bp stress non-positive")
    if base["mean_gross_underlying_move_bp"] < cfg.minimum_mean_gross_underlying_bp:
        failures.append(f"{stage}: mean gross edge below 30 bp")
    cluster = base["weekly_cluster_sign_flip"]
    if cluster is None or cluster["p_value_one_sided"] > cfg.maximum_weekly_cluster_p:
        failures.append(f"{stage}: weekly-cluster p-value above 0.10")
    for label, metrics in policy_result["side_contributions_base_6bp"].items():
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{stage}: {label} contribution non-positive")
    for name, metrics in policy_result["splits_base_6bp"].items():
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
    return failures


def qualification(
    policy_results: dict[str, dict[str, Any]],
    *,
    stage: str,
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    primary_failures = stage_gate_failures(
        policy_results["primary"], stage=stage, cfg=cfg
    )
    delayed = policy_results["one_bar_delayed_entry"]["base_6bp"]
    delayed_failures = (
        []
        if delayed["absolute_return_pct"] > 0.0
        else [f"{stage}: one-bar-delayed entry non-positive absolute return"]
    )
    mechanism_failures = {
        name: stage_gate_failures(policy_results[name], stage=stage, cfg=cfg)
        for name in MECHANISM_REJECTION_CONTROLS
    }
    passing_nulls = [name for name, failures in mechanism_failures.items() if not failures]
    failures = [*primary_failures, *delayed_failures]
    failures.extend(
        f"mechanism-null control independently passed every gate: {name}"
        for name in passing_nulls
    )
    return {
        "qualifies": not failures,
        "stage": stage,
        "scope": "frozen pre-orthogonality performance and mechanism gates",
        "final_promotion_allowed": False,
        "failures": failures,
        "primary_performance_gate_failures": primary_failures,
        "delayed_entry_gate_failures": delayed_failures,
        "mechanism_control_gate_failures": mechanism_failures,
        "passing_mechanism_controls": passing_nulls,
        "direction_stale_and_random_controls_are_diagnostic_only": True,
    }


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
            "mean_gross_underlying_move_bp",
            "trade_count",
        )
    }


def _compute_stage_report(
    cfg: EvaluationConfig,
    freeze: dict[str, Any],
    controls: dict[str, pd.DataFrame],
    *,
    stage: str,
    created_at: str,
    train_parent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    market, market_source = load_market_window(stage, freeze)
    funding, funding_source = load_funding_window(stage, freeze)
    cluster_controls = {"primary", *MECHANISM_REJECTION_CONTROLS}
    policy_results = {
        name: _evaluate_policy_stage(
            market,
            funding,
            clock,
            stage=stage,
            cfg=cfg,
            compute_cluster=name in cluster_controls,
        )
        for name, clock in controls.items()
    }
    verdict = qualification(policy_results, stage=stage, cfg=cfg)
    if stage == "train":
        opened_windows = ["train_2021_2022"]
        decision = "open_selection_2023" if verdict["qualifies"] else "reject_before_selection"
        selection_opened = False
    else:
        opened_windows = ["train_2021_2022", "selection_2023"]
        decision = (
            "test_frozen_orthogonality_and_marginal_portfolio_contribution"
            if verdict["qualifies"]
            else "reject_before_orthogonality"
        )
        selection_opened = True
    core: dict[str, Any] = {
        "schema_version": 1,
        "created_at": created_at,
        "protocol": {
            "name": "BATE-288 strict sequential pre-2024 evaluation",
            "stage": stage,
            "opened_windows": opened_windows,
            "selection_2023_opened": selection_opened,
            "forward_windows_opened": False,
            "full_calendar_cagr_including_idle_cash": True,
            "next_open_execution": True,
            "funding_interval": "entry_time <= funding_time < exit_time",
            "funding_notional": (
                "fixed entry quantity times exact frozen realized funding rate "
                "and frozen settlement mark price"
            ),
            "strict_mdd": (
                "global/pre-entry HWM; entry cost; favorable then adverse held OHLC; "
                "funding credits raise HWM and debits lower adverse envelope; "
                "hypothetical adverse liquidation cost; exit cost"
            ),
            "controls_cannot_replace_primary": True,
            "absolute_return_always_reported": True,
        },
        "evaluation_config": asdict(cfg),
        "evaluation_freeze_sha256": sha256_file(EVALUATION_FREEZE),
        "evaluation_freeze_manifest_hash": freeze["manifest_hash"],
        "control_clock_hashes": {name: _clock_hash(clock) for name, clock in controls.items()},
        "control_clock_counts": {name: int(len(clock)) for name, clock in controls.items()},
        "source": {
            "support_result_sha256": SUPPORT_RESULT_SHA256,
            "primary_clock_sha256": PRIMARY_CLOCK_SHA256,
            "market": market_source,
            "funding": funding_source,
        },
        "policies": policy_results,
        "qualification": verdict,
        "decision": decision,
    }
    if train_parent is not None:
        core["train_parent"] = train_parent
        core["protocol"]["train_replayed_before_selection"] = True
    return seal_result(core)


def _write_result_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_result_hash(payload)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def evaluate_train(cfg: EvaluationConfig | None = None) -> dict[str, Any]:
    frozen_cfg = EvaluationConfig() if cfg is None else cfg
    freeze = verify_evaluation_freeze(frozen_cfg)
    if TRAIN_OUTPUT.exists():
        raise FileExistsError("BATE-288 train result is write-once")
    if SELECTION_OUTPUT.exists():
        raise ValueError("BATE-288 selection result exists before train evaluation")
    controls, _ = verify_support_and_control_clocks(frozen_cfg)
    report = _compute_stage_report(
        frozen_cfg,
        freeze,
        controls,
        stage="train",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_result_once(TRAIN_OUTPUT, report)
    return report


def _verify_passing_train_result(
    cfg: EvaluationConfig,
    freeze: dict[str, Any],
    controls: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    if not TRAIN_OUTPUT.is_file():
        raise PermissionError("2023 selection remains sealed: train artifact is missing")
    train = _read_json(TRAIN_OUTPUT)
    validate_result_hash(train)
    if train.get("schema_version") != 1:
        raise ValueError("BATE-288 train result schema changed")
    if train.get("evaluation_config") != asdict(cfg):
        raise ValueError("BATE-288 train config differs from evaluator freeze")
    if train.get("evaluation_freeze_sha256") != sha256_file(EVALUATION_FREEZE):
        raise ValueError("BATE-288 train belongs to another evaluator freeze")
    protocol = train.get("protocol", {})
    if protocol.get("opened_windows") != ["train_2021_2022"]:
        raise ValueError("BATE-288 train opened an unexpected window")
    if protocol.get("selection_2023_opened") is not False:
        raise ValueError("BATE-288 train already opened selection outcomes")
    if train.get("control_clock_hashes") != freeze["control_clock_hashes"]:
        raise ValueError("BATE-288 train used another control clock")
    verdict = train.get("qualification", {})
    if verdict.get("qualifies") is not True or verdict.get("failures") != []:
        raise PermissionError("2023 selection remains sealed because train failed")
    if train.get("decision") != "open_selection_2023":
        raise PermissionError("BATE-288 train did not authorize 2023 selection")
    created_at = train.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("BATE-288 train lacks a creation timestamp")
    replay = _compute_stage_report(
        cfg,
        freeze,
        controls,
        stage="train",
        created_at=created_at,
    )
    if replay != train:
        raise ValueError("BATE-288 train artifact does not exactly replay")
    return train


def evaluate_selection(cfg: EvaluationConfig | None = None) -> dict[str, Any]:
    frozen_cfg = EvaluationConfig() if cfg is None else cfg
    freeze = verify_evaluation_freeze(frozen_cfg)
    if SELECTION_OUTPUT.exists():
        raise FileExistsError("BATE-288 selection result is write-once")
    controls, _ = verify_support_and_control_clocks(frozen_cfg)
    train = _verify_passing_train_result(frozen_cfg, freeze, controls)
    train_parent = {
        "path": str(TRAIN_OUTPUT),
        "sha256": sha256_file(TRAIN_OUTPUT),
        "result_hash": train["result_hash"],
    }
    report = _compute_stage_report(
        frozen_cfg,
        freeze,
        controls,
        stage="selection",
        created_at=datetime.now(timezone.utc).isoformat(),
        train_parent=train_parent,
    )
    _write_result_once(SELECTION_OUTPUT, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "selection"), required=True)
    args = parser.parse_args()
    result = evaluate_train() if args.stage == "train" else evaluate_selection()
    full_name, _ = STAGE_WINDOWS[args.stage]
    print(
        json.dumps(
            {
                "stage": args.stage,
                "qualification": result["qualification"],
                "decision": result["decision"],
                "primary": _headline(result["policies"]["primary"]["base_6bp"]),
                "primary_stress_10bp": _headline(
                    result["policies"]["primary"]["stress_10bp"]
                ),
                "primary_splits": {
                    name: _headline(metrics)
                    for name, metrics in result["policies"]["primary"][
                        "splits_base_6bp"
                    ].items()
                },
                "primary_sides": {
                    name: _headline(metrics)
                    for name, metrics in result["policies"]["primary"][
                        "side_contributions_base_6bp"
                    ].items()
                },
                "controls": {
                    name: _headline(policy["base_6bp"])
                    for name, policy in result["policies"].items()
                    if name != "primary"
                },
                "full_window": full_name,
                "output": str(TRAIN_OUTPUT if args.stage == "train" else SELECTION_OUTPUT),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
