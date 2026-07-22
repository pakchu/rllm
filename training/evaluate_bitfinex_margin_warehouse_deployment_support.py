"""Evaluate the frozen outcome-blind BFMWD-144 source-support gate.

This stage may read only the hash-bound Bitfinex funding-stat source and the
six pre-frozen comparator clocks.  It must not open BTC prices, returns,
funding paid on positions, labels, PnL, or any row at or after 2024-01-01.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from training import preregister_bitfinex_margin_warehouse_deployment as prereg


PROTOCOL_VERSION = "bitfinex_margin_warehouse_deployment_support_v1"
AS_OF_DATE = "2026-07-20"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = Path(
    "results/bitfinex_margin_warehouse_deployment_preregistration_2026-07-20.json"
)
PREREGISTRATION_SHA256 = (
    "6e478bac6becb58d282867f4ee612d9d13e803d01985474477d6e3073cd49e58"
)
COMPARATOR_FREEZE = Path(
    "results/bitfinex_margin_warehouse_deployment_comparator_freeze_2026-07-20.json"
)
COMPARATOR_FREEZE_SHA256 = (
    "37ee403d33b5361c752b84ef94d46a05d991d82e1b0a77338b41a6c49e8410de"
)
SOURCE_MANIFEST = Path(
    "results/bitfinex_margin_funding_stats_source_manifest_2026-07-20.json"
)
SOURCE_MANIFEST_SHA256 = (
    "9d7c13d56983d7d33fec1c17e24f1794baca64fcfc666599b798d5d5b49cf9b9"
)
SOURCE_DATA = Path("data/bitfinex_margin_funding_stats_2020_2023.csv.gz")
SOURCE_DATA_SHA256 = (
    "71635b9f3a38efa7422a6fcf616859e6a41636bbb79ff0f85e160ef395b0d53c"
)
RAW_SOURCE = Path("data/bitfinex_margin_funding_stats_raw_2020_2023.jsonl.gz")
RAW_SOURCE_SHA256 = (
    "2f5ca2b344806be5bbfa63090fb79a86259d722e03c4f136cd316eb5787f8adb"
)
TRANSPORT_AMENDMENT = Path(
    "results/bitfinex_margin_funding_stats_transport_v2_amendment_2026-07-20.json"
)
TRANSPORT_AMENDMENT_SHA256 = (
    "1fc2d1b35242e7a1bd8232b3b0dfe65d479d0f8e2c4240c523efea1937dd00e9"
)
SQFD_PREFIX_TRANSPORT_FREEZE = Path(
    "results/bfmwd_sqfd_2023_comparator_prefix_transport_freeze_2026-07-20.json"
)
SQFD_PREFIX_TRANSPORT_FREEZE_SHA256 = (
    "c90a2370a76ba81a33b6b9c4102a0be27dbc08c89151d5905aee688403576913"
)
SQFD_PREFIX_MANIFEST = Path(
    "results/bfmwd_sqfd_2023_comparator_prefix_manifest_2026-07-20.json"
)
SQFD_PREFIX_MANIFEST_SHA256 = (
    "09c86f119e24a3379e8d35abf563b81c669e286c2b84e71a5868798c95e3e521"
)
SQFD_PREFIX = Path("data/bfmwd_sqfd_primary_clocks_2023_prefix.csv.gz")
SQFD_PREFIX_SHA256 = (
    "0afc8f0cce62e4276e3a6c0cfc66a0c91a868904236f7857445b88eb84db935a"
)
SOURCE_ACCESS_SEAL = Path(
    "results/bitfinex_margin_warehouse_deployment_source_access_seal_2026-07-20.json"
)
EVALUATOR_SOURCE = Path(
    "training/evaluate_bitfinex_margin_warehouse_deployment_support.py"
)
PROTOCOL_DOCUMENT = Path(
    "docs/bitfinex-margin-warehouse-deployment-support-protocol-2026-07-20.md"
)
DEFAULT_CLOCKS = Path(
    "data/bitfinex_margin_warehouse_deployment_clocks_2021_2023.csv.gz"
)
DEFAULT_RESULT = Path(
    "results/bitfinex_margin_warehouse_deployment_support_2026-07-20.json"
)

CONTROL_ORDER = ("primary", *prereg.SOURCE_ONLY_CONTROLS)
SPLITS = {
    "train": (
        pd.Timestamp(prereg.FROZEN_POLICY.train_start),
        pd.Timestamp(prereg.FROZEN_POLICY.train_end_exclusive),
    ),
    "selection": (
        pd.Timestamp(prereg.FROZEN_POLICY.selection_start),
        pd.Timestamp(prereg.FROZEN_POLICY.selection_end_exclusive),
    ),
}
CLOCK_COLUMNS = (
    "candidate",
    "variant_id",
    "control",
    "split",
    "symbol",
    "side",
    "observation_time",
    "source_available_at",
    "decision_available_at",
    "entry_time",
    "exit_time",
)
SAFE_COMPARATOR_COLUMNS = {"entry_time", "control"}
EXPECTED_SOURCE_COLUMNS = (
    "symbol",
    "observation_time",
    "available_at",
    "timestamp_ms",
    "frr",
    "average_period_days",
    "funding_amount",
    "funding_amount_used",
    "funding_below_threshold",
)


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def _timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp is pd.NaT:
        raise ValueError("BFMWD timestamp must not be NaT")
    return cast(pd.Timestamp, timestamp)


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
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(repository_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"BFMWD JSON input must be an object: {path}")
    return payload


def _verify_internal_hash(payload: Mapping[str, Any], label: str) -> None:
    manifest_hash = payload.get("manifest_hash")
    unhashed = dict(payload)
    unhashed.pop("manifest_hash", None)
    if manifest_hash != canonical_hash(unhashed):
        raise ValueError(f"BFMWD {label} internal hash mismatch")


def validate_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("BFMWD preregistration artifact hash mismatch")
    payload = _load_json(PREREGISTRATION)
    _verify_internal_hash(payload, "preregistration")
    if payload.get("candidate_family") != prereg.CANDIDATE_FAMILY:
        raise ValueError("BFMWD candidate family changed")
    expected_policy = prereg.asdict(prereg.FROZEN_POLICY)
    expected_policy["symbols"] = list(prereg.FROZEN_POLICY.symbols)
    if payload.get("policy", {}).get("config") != expected_policy:
        raise ValueError("BFMWD frozen policy changed")
    if payload.get("policy", {}).get("variants") != [
        prereg.asdict(variant) for variant in prereg.VARIANTS
    ]:
        raise ValueError("BFMWD frozen variants changed")
    if payload.get("support_gates") != prereg.SUPPORT_GATES:
        raise ValueError("BFMWD support gates changed")
    if tuple(payload.get("policy", {}).get("source_only_controls", ())) != (
        prereg.SOURCE_ONLY_CONTROLS
    ):
        raise ValueError("BFMWD source-only controls changed")
    files = payload.get("files", {})
    if files.get("preregistration_source") != str(prereg.PREREGISTRATION_SOURCE):
        raise ValueError("BFMWD preregistration source path changed")
    if files.get("preregistration_source_sha256") != sha256_file(
        prereg.PREREGISTRATION_SOURCE
    ):
        raise ValueError("BFMWD preregistration source hash changed")
    if files.get("preregistration_document") != str(prereg.PREREGISTRATION_DOCUMENT):
        raise ValueError("BFMWD preregistration document path changed")
    if files.get("preregistration_document_sha256") != sha256_file(
        prereg.PREREGISTRATION_DOCUMENT
    ):
        raise ValueError("BFMWD preregistration document hash changed")
    source_contract = payload.get("source_contract", {})
    if source_contract.get("builder") != str(prereg.SOURCE_BUILDER):
        raise ValueError("BFMWD source-builder path changed")
    if source_contract.get("builder_sha256") != sha256_file(prereg.SOURCE_BUILDER):
        raise ValueError("BFMWD source-builder hash changed")
    boundary = payload.get("outcome_boundary", {})
    if boundary.get("outcomes_opened") is not False:
        raise ValueError("BFMWD preregistration opened outcomes")
    return payload


def validate_comparator_freeze() -> dict[str, Any]:
    if sha256_file(COMPARATOR_FREEZE) != COMPARATOR_FREEZE_SHA256:
        raise ValueError("BFMWD comparator-freeze artifact hash mismatch")
    payload = _load_json(COMPARATOR_FREEZE)
    if payload.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("BFMWD comparator freeze lost preregistration binding")
    if payload.get("candidate_family") != prereg.CANDIDATE_FAMILY:
        raise ValueError("BFMWD comparator freeze candidate changed")
    boundary = payload.get("outcome_boundary", {})
    if boundary.get("outcomes_opened") is not False:
        raise ValueError("BFMWD comparator freeze opened outcomes")
    comparators = payload.get("comparators")
    if not isinstance(comparators, list) or len(comparators) != 6:
        raise ValueError("BFMWD comparator registry must contain six members")
    return payload


def validate_source_access_seal() -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate every frozen byte before a numeric source row can be parsed."""
    validate_preregistration()
    validate_comparator_freeze()
    seal = _load_json(SOURCE_ACCESS_SEAL)
    _verify_internal_hash(seal, "source-access seal")
    if seal.get("protocol_version") != (
        "bitfinex_margin_warehouse_deployment_source_access_seal_v1"
    ):
        raise ValueError("BFMWD source-access seal protocol changed")
    if seal.get("feature_values_inspected_before_seal") is not False:
        raise ValueError("BFMWD source values were opened before the seal")
    if seal.get("market_outcomes_opened_before_seal") is not False:
        raise ValueError("BFMWD market outcomes were opened before the seal")

    expected = {
        "preregistration": PREREGISTRATION,
        "comparator_freeze": COMPARATOR_FREEZE,
        "transport_amendment": TRANSPORT_AMENDMENT,
        "source_manifest": SOURCE_MANIFEST,
        "canonical_source": SOURCE_DATA,
        "raw_source": RAW_SOURCE,
        "sqfd_prefix_transport_freeze": SQFD_PREFIX_TRANSPORT_FREEZE,
        "sqfd_prefix_manifest": SQFD_PREFIX_MANIFEST,
        "sqfd_prefix": SQFD_PREFIX,
        "evaluator_source": EVALUATOR_SOURCE,
        "protocol_document": PROTOCOL_DOCUMENT,
    }
    bindings = seal.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != set(expected):
        raise ValueError("BFMWD source-access seal binding set changed")
    for name, path in expected.items():
        binding = bindings[name]
        if not isinstance(binding, dict) or binding.get("path") != str(path):
            raise ValueError(f"BFMWD source-access path changed: {name}")
        if binding.get("sha256") != sha256_file(path):
            raise ValueError(f"BFMWD source-access hash changed: {name}")

    frozen_hashes = {
        TRANSPORT_AMENDMENT: TRANSPORT_AMENDMENT_SHA256,
        SOURCE_MANIFEST: SOURCE_MANIFEST_SHA256,
        SOURCE_DATA: SOURCE_DATA_SHA256,
        RAW_SOURCE: RAW_SOURCE_SHA256,
        SQFD_PREFIX_TRANSPORT_FREEZE: SQFD_PREFIX_TRANSPORT_FREEZE_SHA256,
        SQFD_PREFIX_MANIFEST: SQFD_PREFIX_MANIFEST_SHA256,
        SQFD_PREFIX: SQFD_PREFIX_SHA256,
    }
    for path, expected_hash in frozen_hashes.items():
        if sha256_file(path) != expected_hash:
            raise ValueError(f"BFMWD frozen source input hash changed: {path}")

    amendment = _load_json(TRANSPORT_AMENDMENT)
    _verify_internal_hash(amendment, "transport amendment")
    if amendment.get("outcome_boundary", {}).get("outcomes_opened") is not False:
        raise ValueError("BFMWD transport amendment opened outcomes")

    manifest = _load_json(SOURCE_MANIFEST)
    if manifest.get("protocol_version") != "bitfinex_margin_funding_stats_source_v2":
        raise ValueError("Bitfinex source manifest is not transport v2")
    if manifest.get("transport_amendment", {}).get("sha256") != (
        TRANSPORT_AMENDMENT_SHA256
    ):
        raise ValueError("Bitfinex source manifest lost transport amendment")
    contract = manifest.get("source_contract", {})
    if contract.get("outcomes_opened") is not False:
        raise ValueError("Bitfinex source manifest opened outcomes")
    if contract.get("market_or_pnl_columns_loaded") is not False:
        raise ValueError("Bitfinex source manifest loaded market/PnL columns")
    if contract.get("post_2023_rows_requested") is not False:
        raise ValueError("Bitfinex source manifest requested post-2023 rows")
    files = manifest.get("files", {})
    if files.get("canonical", {}).get("path") != str(SOURCE_DATA):
        raise ValueError("Bitfinex canonical source path changed")
    if files.get("canonical", {}).get("sha256") != sha256_file(SOURCE_DATA):
        raise ValueError("Bitfinex canonical source hash changed")
    if files.get("raw", {}).get("path") != str(RAW_SOURCE):
        raise ValueError("Bitfinex raw source path changed")
    if files.get("raw", {}).get("sha256") != sha256_file(RAW_SOURCE):
        raise ValueError("Bitfinex raw source hash changed")

    prefix_manifest = _load_json(SQFD_PREFIX_MANIFEST)
    _verify_internal_hash(prefix_manifest, "SQFD prefix manifest")
    if prefix_manifest.get("output", {}).get("sha256") != SQFD_PREFIX_SHA256:
        raise ValueError("SQFD comparator prefix output hash changed")
    if prefix_manifest.get("filter") != {
        "control": "primary",
        "end_exclusive": "2024-01-01T00:00:00+00:00",
        "start_inclusive": "2023-01-01T00:00:00+00:00",
    }:
        raise ValueError("SQFD comparator prefix filter changed")
    if prefix_manifest.get("outcome_boundary", {}).get("outcomes_opened") is not False:
        raise ValueError("SQFD comparator prefix opened outcomes")
    return seal, manifest


def validate_source_header(
    columns: tuple[str, ...], manifest_columns: list[str]
) -> None:
    if columns != EXPECTED_SOURCE_COLUMNS:
        raise ValueError("BFMWD canonical source header is not exactly allowlisted")
    if manifest_columns != list(EXPECTED_SOURCE_COLUMNS):
        raise ValueError("BFMWD source-manifest column contract changed")


def load_source() -> tuple[pd.DataFrame, dict[str, Any]]:
    _, manifest = validate_source_access_seal()
    header = cast(
        pd.DataFrame, pd.read_csv(repository_path(SOURCE_DATA), nrows=0)
    )
    validate_source_header(
        tuple(str(column) for column in header.columns),
        list(manifest.get("files", {}).get("canonical", {}).get("columns", [])),
    )
    columns = {
        "symbol",
        "observation_time",
        "available_at",
        "average_period_days",
        "funding_amount",
        "funding_amount_used",
    }
    frame = cast(
        pd.DataFrame,
        pd.read_csv(
        repository_path(SOURCE_DATA),
        usecols=lambda column: column in columns,
        parse_dates=["observation_time", "available_at"],
        ),
    )
    if frame.empty or set(frame["symbol"]) != set(prereg.FROZEN_POLICY.symbols):
        raise ValueError("BFMWD source is empty or missing a symbol")
    for column in ("observation_time", "available_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    frame = frame.sort_values(["symbol", "observation_time"], kind="mergesort")
    frame = frame.reset_index(drop=True)
    if frame.duplicated(["symbol", "observation_time"]).any():
        raise ValueError("BFMWD source contains duplicate symbol observations")
    start = pd.Timestamp(prereg.FROZEN_POLICY.warmup_start)
    end = pd.Timestamp(prereg.FROZEN_POLICY.selection_end_exclusive)
    if frame["observation_time"].min() < start or frame["observation_time"].max() >= end:
        raise ValueError("BFMWD source escaped the frozen pre-2024 interval")
    regular_poll = frame["observation_time"].dt.floor("h") + pd.Timedelta(minutes=15)
    observed_bar = frame["observation_time"].dt.ceil("5min")
    expected_availability = cast(
        pd.Series,
        pd.concat([regular_poll, observed_bar], axis=1).max(axis=1),
    )
    if not frame["available_at"].equals(expected_availability):
        raise ValueError("BFMWD source availability clock changed")
    numeric = frame[
        ["average_period_days", "funding_amount", "funding_amount_used"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("BFMWD source contains non-finite required fields")
    if (frame["funding_amount"] <= 0).any():
        raise ValueError("BFMWD total funding must be positive")
    if (frame["funding_amount_used"] < 0).any():
        raise ValueError("BFMWD used funding must be non-negative")
    if (frame["funding_amount_used"] > frame["funding_amount"] + 1e-9).any():
        raise ValueError("BFMWD used funding exceeds total funding")
    expected_rows = int(manifest["files"]["canonical"]["rows"])
    if len(frame) != expected_rows:
        raise ValueError("BFMWD canonical source row count changed")
    return frame, manifest


def strict_prior_robust_zscore(
    values: pd.Series,
    *,
    window: int,
    minimum: int,
    mad_scale: float,
    block_rows: int = 256,
) -> pd.Series:
    """Exact median/MAD using only the preceding fixed hourly-row window."""
    if not (1 <= minimum <= window) or block_rows < 1 or mad_scale <= 0:
        raise ValueError("invalid BFMWD robust rolling configuration")
    raw = values.to_numpy(dtype=float, copy=True)
    padded = np.concatenate([np.full(window, np.nan), raw])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)[: len(raw)]
    output = np.full(len(raw), np.nan)
    for start in range(0, len(raw), block_rows):
        stop = min(len(raw), start + block_rows)
        block = windows[start:stop]
        counts = np.isfinite(block).sum(axis=1)
        eligible = (counts >= minimum) & np.isfinite(raw[start:stop])
        if not eligible.any():
            continue
        selected = block[eligible]
        centers = np.nanmedian(selected, axis=1)
        deviations = np.abs(selected - centers[:, None])
        mad = np.nanmedian(deviations, axis=1)
        scale = mad_scale * mad
        current = raw[start:stop][eligible]
        valid = np.isfinite(scale) & (scale > 1e-12)
        z = np.full(len(scale), np.nan)
        z[valid] = (current[valid] - centers[valid]) / scale[valid]
        block_output = output[start:stop]
        block_output[eligible] = z
        output[start:stop] = block_output
    return pd.Series(output, index=values.index, name=values.name)


def strict_prior_median(
    values: pd.Series, *, window: int, minimum: int
) -> pd.Series:
    return cast(
        pd.Series,
        values.shift(1).rolling(window=window, min_periods=minimum).median(),
    )


def exact_lag(values: pd.Series, times: pd.Series, *, hours: int) -> pd.Series:
    timestamps = pd.DatetimeIndex(pd.to_datetime(times, utc=True, errors="raise"))
    if timestamps.has_duplicates:
        raise ValueError("BFMWD exact-lag timestamps contain duplicates")
    lookup = pd.Series(values.to_numpy(dtype=float), index=timestamps)
    prior = lookup.reindex(timestamps - pd.Timedelta(hours=hours))
    return pd.Series(prior.to_numpy(dtype=float), index=values.index, name=values.name)


def exact_hourly_onset(
    trigger: pd.Series,
    times: pd.Series,
    *,
    source_complete: pd.Series | None = None,
) -> pd.Series:
    timestamps = pd.DatetimeIndex(pd.to_datetime(times, utc=True, errors="raise"))
    if len(trigger) != len(timestamps) or timestamps.has_duplicates:
        raise ValueError("BFMWD onset clock is misaligned or duplicated")
    current = trigger.fillna(False).astype(bool)
    complete = (
        pd.Series(True, index=current.index)
        if source_complete is None
        else source_complete.fillna(False).astype(bool)
    )
    lookup = pd.Series(current.to_numpy(dtype=bool), index=timestamps)
    prior = lookup.reindex(timestamps - pd.Timedelta(hours=1))
    complete_lookup = pd.Series(complete.to_numpy(dtype=bool), index=timestamps)
    prior_complete = complete_lookup.reindex(timestamps - pd.Timedelta(hours=1))
    return current & complete & pd.Series(
        (prior.eq(False) & prior_complete.eq(True)).to_numpy(dtype=bool),
        index=current.index,
    )


def normalize_symbol_hourly_grid(source: pd.DataFrame) -> pd.DataFrame:
    if source["symbol"].nunique() != 1:
        raise ValueError("BFMWD hourly normalizer requires one symbol")
    symbol = str(source["symbol"].iloc[0])
    frame = source.sort_values("observation_time", kind="mergesort").reset_index(
        drop=True
    )
    frame["source_hour"] = frame["observation_time"].dt.floor("h")
    # When the provider emitted two snapshots in one hour, the latest official
    # snapshot is the complete hour state available to the fixed live poll.
    frame = cast(
        pd.DataFrame,
        frame.drop_duplicates("source_hour", keep="last").set_index("source_hour"),
    )
    start = _timestamp(prereg.FROZEN_POLICY.warmup_start)
    end = _timestamp(prereg.FROZEN_POLICY.selection_end_exclusive)
    grid = pd.date_range(start=start, end=end, freq="h", inclusive="left")
    frame = cast(pd.DataFrame, frame.reindex(grid))
    frame.index.name = "source_hour"
    frame = frame.reset_index()
    frame["symbol"] = symbol
    return frame


def build_symbol_features(
    source: pd.DataFrame,
    variant: prereg.Variant,
    *,
    block_rows: int = 256,
) -> pd.DataFrame:
    frame = normalize_symbol_hourly_grid(source)
    times = cast(pd.Series, frame["source_hour"])
    total = cast(
        pd.Series, pd.to_numeric(_series(frame, "funding_amount"), errors="raise")
    ).astype(float)
    used = cast(
        pd.Series, pd.to_numeric(_series(frame, "funding_amount_used"), errors="raise")
    ).astype(float)
    unused = cast(pd.Series, total - used)
    utilization = cast(pd.Series, used / total).clip(
        prereg.FROZEN_POLICY.utilization_clip,
        1.0 - prereg.FROZEN_POLICY.utilization_clip,
    )
    d = variant.deployment_hours
    w = variant.warehouse_hours
    unused_d = exact_lag(unused, times, hours=d)
    unused_dw = exact_lag(unused, times, hours=d + w)
    used_d = exact_lag(used, times, hours=d)
    utilization_d = exact_lag(cast(pd.Series, utilization), times, hours=d)

    raw_features = {
        "warehouse_charge": np.log1p(unused_d) - np.log1p(unused_dw),
        "used_deployment": np.log1p(used) - np.log1p(used_d),
        "unused_draw": np.log1p(unused_d) - np.log1p(unused),
        "utilization_deployment": (
            np.log(utilization / (1.0 - utilization))
            - np.log(utilization_d / (1.0 - utilization_d))
        ),
    }
    output = cast(
        pd.DataFrame,
        frame.loc[
            :,
            [
                "symbol",
                "source_hour",
                "observation_time",
                "available_at",
                "average_period_days",
            ],
        ].copy(),
    )
    for name, values in raw_features.items():
        output[name] = cast(pd.Series, values)
        output[f"{name}_z"] = strict_prior_robust_zscore(
            cast(pd.Series, output[name]),
            window=prereg.FROZEN_POLICY.history_hours,
            minimum=prereg.FROZEN_POLICY.minimum_history_hours,
            mad_scale=prereg.FROZEN_POLICY.robust_mad_scale,
            block_rows=block_rows,
        )
    output["tenor_prior_median"] = strict_prior_median(
        cast(
            pd.Series,
            pd.to_numeric(_series(output, "average_period_days"), errors="raise"),
        ),
        window=prereg.FROZEN_POLICY.history_hours,
        minimum=prereg.FROZEN_POLICY.minimum_history_hours,
    )
    z_columns = [f"{name}_z" for name in raw_features]
    z_floor = variant.robust_z_threshold
    tenor = _series(output, "average_period_days") >= _series(
        output, "tenor_prior_median"
    )
    z_frame = cast(pd.DataFrame, output[z_columns])
    output["primary"] = z_frame.ge(z_floor).all(axis=1) & tenor
    output["no_warehouse_charge_prerequisite"] = (
        cast(
            pd.DataFrame,
            output[
                [column for column in z_columns if column != "warehouse_charge_z"]
            ],
        )
        .ge(z_floor)
        .all(axis=1)
        & tenor
    )
    output["no_unused_draw_confirmation"] = (
        cast(
            pd.DataFrame,
            output[[column for column in z_columns if column != "unused_draw_z"]],
        )
        .ge(z_floor)
        .all(axis=1)
        & tenor
    )
    output["no_tenor_confirmation"] = z_frame.ge(z_floor).all(axis=1)
    output["stale_24h_source"] = output["primary"]
    return cast(pd.DataFrame, output)


def _empty_clocks() -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(CLOCK_COLUMNS))


def _schedule_split(
    candidates: pd.DataFrame,
    *,
    split: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    eligible = cast(
        pd.DataFrame,
        candidates[
            candidates["entry_time"].ge(start) & candidates["exit_time"].lt(end)
        ].copy(),
    )
    eligible = cast(
        pd.DataFrame,
        cast(Any, eligible).sort_values(
            by=["entry_time", "observation_time", "symbol"], kind="mergesort"
        ),
    )
    rows: list[dict[str, Any]] = []
    prior_exit: pd.Timestamp | None = None
    for row in eligible.to_dict("records"):
        entry = _timestamp(row["entry_time"])
        exit_time = _timestamp(row["exit_time"])
        if prior_exit is not None and entry < prior_exit:
            continue
        row["split"] = split
        rows.append(row)
        prior_exit = exit_time
    if not rows:
        return _empty_clocks()
    return pd.DataFrame(rows, columns=pd.Index(CLOCK_COLUMNS))


def build_variant_clocks(
    source: pd.DataFrame,
    variant: prereg.Variant,
    *,
    block_rows: int = 256,
) -> pd.DataFrame:
    symbol_features = {
        symbol: build_symbol_features(
            cast(pd.DataFrame, source[source["symbol"].eq(symbol)].copy()),
            variant,
            block_rows=block_rows,
        )
        for symbol in prereg.FROZEN_POLICY.symbols
    }
    side_by_symbol = {
        "fUSD": prereg.FROZEN_POLICY.usd_side,
        "fBTC": prereg.FROZEN_POLICY.btc_side,
    }
    all_clocks: list[pd.DataFrame] = []
    hold = pd.Timedelta(
        minutes=variant.hold_bars * prereg.FROZEN_POLICY.bar_minutes
    )
    entry_delay = pd.Timedelta(minutes=prereg.FROZEN_POLICY.entry_delay_minutes)
    for control in CONTROL_ORDER:
        onsets: list[pd.DataFrame] = []
        for symbol, features in symbol_features.items():
            trigger = cast(pd.Series, features[control]).fillna(False).astype(bool)
            onset = exact_hourly_onset(
                trigger,
                cast(pd.Series, features["source_hour"]),
                source_complete=cast(
                    pd.Series, features["observation_time"].notna()
                ),
            )
            selected = features.loc[
                onset, ["source_hour", "observation_time", "available_at"]
            ].copy()
            if selected.empty:
                continue
            selected["candidate"] = prereg.CANDIDATE_FAMILY
            selected["variant_id"] = variant.variant_id
            selected["control"] = control
            selected["symbol"] = symbol
            selected["side"] = side_by_symbol[symbol]
            selected = selected.rename(columns={"available_at": "source_available_at"})
            stale_delay = (
                pd.Timedelta(hours=24)
                if control == "stale_24h_source"
                else pd.Timedelta(0)
            )
            selected["decision_available_at"] = (
                selected["source_available_at"] + stale_delay
            )
            selected["entry_time"] = selected["decision_available_at"] + entry_delay
            selected["exit_time"] = selected["entry_time"] + hold
            onsets.append(selected)
        if not onsets:
            continue
        candidates = pd.concat(onsets, ignore_index=True)
        # The two symbols represent opposing hypotheses.  A shared source-hour
        # anchor is therefore an explicit abstention, never a latency tie-break.
        simultaneous = candidates.duplicated("source_hour", keep=False)
        candidates = cast(pd.DataFrame, candidates.loc[~simultaneous].copy())
        for split, (start, end) in SPLITS.items():
            scheduled = _schedule_split(
                candidates,
                split=split,
                start=_timestamp(start),
                end=_timestamp(end),
            )
            if not scheduled.empty:
                all_clocks.append(scheduled)
    if not all_clocks:
        return _empty_clocks()
    clocks = pd.concat(all_clocks, ignore_index=True)
    clocks = clocks.sort_values(
        ["variant_id", "control", "entry_time"], kind="mergesort"
    ).reset_index(drop=True)
    if tuple(clocks.columns) != CLOCK_COLUMNS:
        raise ValueError("BFMWD clock schema changed")
    return clocks


def build_all_clocks(source: pd.DataFrame, *, block_rows: int = 256) -> pd.DataFrame:
    parts = [
        build_variant_clocks(source, variant, block_rows=block_rows)
        for variant in prereg.VARIANTS
    ]
    nonempty = [part for part in parts if not part.empty]
    if not nonempty:
        return _empty_clocks()
    clocks = pd.concat(nonempty, ignore_index=True)
    return clocks.sort_values(
        ["variant_id", "control", "entry_time"], kind="mergesort"
    ).reset_index(drop=True)


def rolling_interval_share(entries: pd.DatetimeIndex, *, days: int) -> float:
    if len(entries) == 0:
        return 0.0
    values = np.sort(np.asarray(entries.asi8, dtype=np.int64))
    width = int(pd.Timedelta(days=days).value)
    left = np.searchsorted(values, values - width, side="left")
    counts = np.arange(1, len(values) + 1) - left
    return float(counts.max() / len(values))


def support_summary(events: pd.DataFrame) -> dict[str, Any]:
    if events.empty:
        return {
            "events": 0,
            "long": 0,
            "short": 0,
            "long_share": 0.0,
            "short_share": 0.0,
            "year_counts": {},
            "half_counts": {},
            "month_counts": {},
            "weekday_counts": {},
            "maximum_month_share": 0.0,
            "maximum_weekday_share": 0.0,
            "maximum_rolling_14day_share": 0.0,
            "first_entry": None,
            "last_exit": None,
        }
    entry_series = cast(
        pd.Series, pd.to_datetime(_series(events, "entry_time"), utc=True)
    )
    exit_series = cast(
        pd.Series, pd.to_datetime(_series(events, "exit_time"), utc=True)
    )
    entries = pd.DatetimeIndex(entry_series)
    sides = cast(
        pd.Series, pd.to_numeric(_series(events, "side"), errors="raise")
    ).astype(int)
    if not sides.isin((-1, 1)).all():
        raise ValueError("BFMWD clock contains a non-directional side")
    count = len(events)
    months = entry_series.dt.strftime("%Y-%m").value_counts().sort_index()
    weekdays = entry_series.dt.day_name().value_counts().sort_index()
    years = entry_series.dt.year.astype(str).value_counts().sort_index()
    half_labels = entry_series.dt.year.astype(str) + "-H" + cast(
        pd.Series, np.where(entry_series.dt.month <= 6, "1", "2")
    )
    halves = half_labels.value_counts().sort_index()
    long_count = int(sides.eq(1).sum())
    short_count = int(sides.eq(-1).sum())
    return {
        "events": int(count),
        "long": long_count,
        "short": short_count,
        "long_share": float(long_count / count),
        "short_share": float(short_count / count),
        "year_counts": {str(key): int(value) for key, value in years.items()},
        "half_counts": {str(key): int(value) for key, value in halves.items()},
        "month_counts": {str(key): int(value) for key, value in months.items()},
        "weekday_counts": {str(key): int(value) for key, value in weekdays.items()},
        "maximum_month_share": float(months.max() / count),
        "maximum_weekday_share": float(weekdays.max() / count),
        "maximum_rolling_14day_share": rolling_interval_share(entries, days=14),
        "first_entry": _timestamp(entry_series.min()).isoformat(),
        "last_exit": _timestamp(exit_series.max()).isoformat(),
    }


def nearest_share(
    left: pd.DatetimeIndex, right: pd.DatetimeIndex, *, hours: int
) -> float:
    if len(left) == 0 or len(right) == 0:
        return 0.0
    left_ns = np.asarray(left.asi8, dtype=np.int64)
    right_ns = np.sort(np.asarray(right.asi8, dtype=np.int64))
    insertion = np.searchsorted(right_ns, left_ns)
    threshold = int(pd.Timedelta(hours=hours).value)
    matched = np.zeros(len(left_ns), dtype=bool)
    for positions in (insertion - 1, insertion):
        valid = (positions >= 0) & (positions < len(right_ns))
        distance = np.full(len(left_ns), np.iinfo(np.int64).max, dtype=np.int64)
        distance[valid] = np.abs(left_ns[valid] - right_ns[positions[valid]])
        matched |= distance <= threshold
    return float(matched.mean())


def novelty_metrics(
    candidate: pd.DatetimeIndex,
    comparator: pd.DatetimeIndex,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    hours: int,
    minimum_candidate: int,
    minimum_comparator: int,
) -> dict[str, Any]:
    left = candidate[(candidate >= start) & (candidate < end)].unique().sort_values()
    right = comparator[(comparator >= start) & (comparator < end)].unique().sort_values()
    intersection = left.intersection(right)
    union = left.union(right)
    sufficient = len(left) >= minimum_candidate and len(right) >= minimum_comparator
    left_near = nearest_share(left, right, hours=hours)
    right_near = nearest_share(right, left, hours=hours)
    return {
        "coverage": [start.isoformat(), end.isoformat()],
        "candidate_events": int(len(left)),
        "comparator_events": int(len(right)),
        "sufficient_common_support": bool(sufficient),
        "exact_intersection": int(len(intersection)),
        "exact_jaccard": float(len(intersection) / len(union)) if len(union) else 0.0,
        "containment_hours": int(hours),
        "candidate_near_share": left_near,
        "comparator_near_share": right_near,
        "maximum_bidirectional_containment": max(left_near, right_near),
    }


def load_comparator_clocks(
    freeze: Mapping[str, Any],
) -> tuple[dict[str, pd.DatetimeIndex], int]:
    output: dict[str, pd.DatetimeIndex] = {}
    rows = 0
    for contract in freeze["comparators"]:
        name = str(contract["candidate"])
        if name == "SQFD-6":
            path = SQFD_PREFIX
            expected_hash = SQFD_PREFIX_SHA256
        else:
            path = Path(contract["path"])
            expected_hash = str(contract["sha256"])
        if sha256_file(path) != expected_hash:
            raise ValueError(f"BFMWD comparator clock hash mismatch: {name}")
        header = pd.read_csv(repository_path(path), nrows=0)
        entry_column = str(contract["entry_column"])
        required = {entry_column, "control"}
        if not required.issubset(header.columns):
            raise ValueError(f"BFMWD comparator schema changed: {name}")
        frame = cast(
            pd.DataFrame,
            pd.read_csv(
                repository_path(path),
                usecols=lambda column: column in required,
            ),
        )
        frame = cast(pd.DataFrame, frame[frame["control"].eq("primary")].copy())
        entries = pd.DatetimeIndex(
            pd.to_datetime(frame[entry_column], utc=True, errors="raise")
        ).unique().sort_values()
        start = pd.Timestamp(contract["start"])
        end = pd.Timestamp(contract["end_exclusive"])
        entries = entries[(entries >= start) & (entries < end)]
        output[name] = entries
        rows += len(frame)
    if len(output) != 6:
        raise ValueError("BFMWD comparator clock registry changed")
    return output, rows


def evaluate_novelty(
    primary: pd.DataFrame,
    comparators: Mapping[str, pd.DatetimeIndex],
    freeze: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool], list[str]]:
    rules = freeze["rules"]
    contracts = {str(item["candidate"]): item for item in freeze["comparators"]}
    candidate_entries = pd.DatetimeIndex(
        pd.to_datetime(primary["entry_time"], utc=True, errors="raise")
    )
    metrics: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    sufficient: list[str] = []
    for name, entries in comparators.items():
        contract = contracts[name]
        result = novelty_metrics(
            candidate_entries,
            entries,
            start=_timestamp(contract["start"]),
            end=_timestamp(contract["end_exclusive"]),
            hours=int(rules["containment_hours"]),
            minimum_candidate=int(rules["minimum_common_candidate_events"]),
            minimum_comparator=int(rules["minimum_common_comparator_events"]),
        )
        metrics[name] = result
        if result["sufficient_common_support"]:
            sufficient.append(name)
            checks[f"{name}_exact_jaccard"] = result["exact_jaccard"] <= float(
                rules["maximum_exact_entry_jaccard"]
            )
            checks[f"{name}_bidirectional_containment"] = result[
                "maximum_bidirectional_containment"
            ] <= float(rules["maximum_bidirectional_containment"])
    checks["minimum_sufficient_comparators"] = len(sufficient) >= int(
        freeze["minimum_sufficient_comparators"]
    )
    checks["required_domain_comparator"] = bool({"CPR", "CCIPA-48"} & set(sufficient))
    checks["required_external_comparator"] = bool(
        {"AMTR-48", "SQFD-6"} & set(sufficient)
    )
    failures = [name for name, passed in checks.items() if not passed]
    return metrics, checks, failures


def support_checks(
    train: Mapping[str, Any],
    selection: Mapping[str, Any],
    combined: Mapping[str, Any],
) -> tuple[dict[str, bool], list[str]]:
    gate = prereg.SUPPORT_GATES
    checks = {
        "minimum_train_events": train["events"] >= gate["minimum_train_events"],
        "minimum_selection_events": selection["events"]
        >= gate["minimum_selection_events"],
        "minimum_2021_events": train["year_counts"].get("2021", 0)
        >= gate["minimum_events_each_train_year"],
        "minimum_2022_events": train["year_counts"].get("2022", 0)
        >= gate["minimum_events_each_train_year"],
        "minimum_2023_h1_events": selection["half_counts"].get("2023-H1", 0)
        >= gate["minimum_events_each_selection_half"],
        "minimum_2023_h2_events": selection["half_counts"].get("2023-H2", 0)
        >= gate["minimum_events_each_selection_half"],
        "minimum_long_share": combined["long_share"] >= gate["minimum_side_share"],
        "maximum_long_share": combined["long_share"] <= gate["maximum_side_share"],
        "minimum_short_share": combined["short_share"] >= gate["minimum_side_share"],
        "maximum_short_share": combined["short_share"] <= gate["maximum_side_share"],
        "maximum_calendar_month_share": combined["maximum_month_share"]
        <= gate["maximum_calendar_month_share"],
        "maximum_weekday_share": combined["maximum_weekday_share"]
        <= gate["maximum_weekday_share"],
        "maximum_rolling_14day_share": combined["maximum_rolling_14day_share"]
        <= gate["maximum_rolling_14day_share"],
    }
    failures = [name for name, passed in checks.items() if not passed]
    return checks, failures


def _clock_bytes(frame: pd.DataFrame) -> bytes:
    text = frame.to_csv(index=False, lineterminator="\n")
    output = io.BytesIO()
    with gzip.GzipFile(
        fileobj=output,
        mode="wb",
        filename="",
        compresslevel=9,
        mtime=0,
    ) as stream:
        stream.write(text.encode("utf-8"))
    return output.getvalue()


def write_once_bytes(path: str | Path, payload: bytes) -> None:
    target = repository_path(path)
    if target.exists() and target.read_bytes() != payload:
        raise FileExistsError(f"refusing to overwrite frozen BFMWD artifact: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def write_once_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    write_once_bytes(path, encoded)


def evaluate_source_only(
    *,
    clocks_path: str | Path = DEFAULT_CLOCKS,
    result_path: str | Path = DEFAULT_RESULT,
    block_rows: int = 256,
) -> dict[str, Any]:
    preregistration = validate_preregistration()
    freeze = validate_comparator_freeze()
    source, source_manifest = load_source()
    clocks = build_all_clocks(source, block_rows=block_rows)
    comparators, comparator_rows = load_comparator_clocks(freeze)

    variants: dict[str, Any] = {}
    passing: list[str] = []
    for variant in prereg.VARIANTS:
        variant_clocks = cast(
            pd.DataFrame, clocks[clocks["variant_id"].eq(variant.variant_id)].copy()
        )
        primary = cast(
            pd.DataFrame,
            variant_clocks[variant_clocks["control"].eq("primary")].copy(),
        )
        train = support_summary(
            cast(pd.DataFrame, primary[primary["split"].eq("train")].copy())
        )
        selection = support_summary(
            cast(pd.DataFrame, primary[primary["split"].eq("selection")].copy())
        )
        combined = support_summary(primary)
        incidence_checks, incidence_failures = support_checks(
            train, selection, combined
        )
        novelty, novelty_checks, novelty_failures = evaluate_novelty(
            primary, comparators, freeze
        )
        control_support = {
            control: support_summary(
                cast(
                    pd.DataFrame,
                    variant_clocks[variant_clocks["control"].eq(control)].copy(),
                )
            )
            for control in CONTROL_ORDER
        }
        failures = [*incidence_failures, *novelty_failures]
        passed = not failures
        if passed:
            passing.append(variant.variant_id)
        variants[variant.variant_id] = {
            "support": {
                "train": train,
                "selection": selection,
                "combined": combined,
            },
            "control_support": control_support,
            "incidence_checks": incidence_checks,
            "novelty": novelty,
            "novelty_checks": novelty_checks,
            "failures": failures,
            "support_passed": passed,
        }

    clock_payload = _clock_bytes(clocks)
    write_once_bytes(clocks_path, clock_payload)
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "as_of_date": AS_OF_DATE,
        "candidate_family": prereg.CANDIDATE_FAMILY,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": sha256_file(PREREGISTRATION),
            "manifest_hash": preregistration["manifest_hash"],
        },
        "comparator_freeze": {
            "path": str(COMPARATOR_FREEZE),
            "sha256": sha256_file(COMPARATOR_FREEZE),
        },
        "source_access_seal": {
            "path": str(SOURCE_ACCESS_SEAL),
            "sha256": sha256_file(SOURCE_ACCESS_SEAL),
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST),
            "sha256": sha256_file(SOURCE_MANIFEST),
            "protocol_version": source_manifest["protocol_version"],
        },
        "sqfd_comparator_prefix": {
            "path": str(SQFD_PREFIX),
            "sha256": sha256_file(SQFD_PREFIX),
            "manifest_path": str(SQFD_PREFIX_MANIFEST),
            "manifest_sha256": sha256_file(SQFD_PREFIX_MANIFEST),
        },
        "outcome_boundary": {
            "outcomes_opened": False,
            "outcome_sources_opened": [],
            "btc_market_rows_read": 0,
            "funding_paid_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_rows_read": 0,
        },
        "source_rows_read": int(len(source)),
        "comparator_clock_rows_read": int(comparator_rows),
        "clock_artifact": {
            "path": str(clocks_path),
            "sha256": hashlib.sha256(clock_payload).hexdigest(),
            "rows": int(len(clocks)),
            "columns": list(CLOCK_COLUMNS),
        },
        "variants": variants,
        "passing_variants": passing,
        "family_support_passed": bool(passing),
        "next_action": (
            "freeze strict 2021-2022 economic evaluator"
            if passing
            else "retire BFMWD-144 before every BTC market outcome"
        ),
    }
    core["manifest_hash"] = canonical_hash(core)
    write_once_json(result_path, core)
    return core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clocks", default=str(DEFAULT_CLOCKS))
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    parser.add_argument("--block-rows", type=int, default=256)
    args = parser.parse_args()
    result = evaluate_source_only(
        clocks_path=args.clocks,
        result_path=args.result,
        block_rows=args.block_rows,
    )
    print(
        json.dumps(
            {
                "family_support_passed": result["family_support_passed"],
                "passing_variants": result["passing_variants"],
                "variants": result["variants"],
                "outcome_boundary": result["outcome_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
