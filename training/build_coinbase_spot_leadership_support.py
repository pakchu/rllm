"""Build causal feature and event-clock support without reading trade outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.export_wikimedia_attention_source import (
    deterministic_gzip_csv,
    sha256_file,
)
from training.preregister_coinbase_spot_leadership_alpha import (
    DEFAULT_OUTPUT as DEFAULT_PREREGISTRATION,
    Policy,
    canonical_hash,
    policy_grid,
    validate_manifest as validate_preregistration,
)


DEFAULT_SOURCE_MANIFEST = "results/coinbase_spot_leadership_source_manifest_2026-07-16.json"
DEFAULT_FEATURE_OUTPUT = "data/coinbase_leadership_features_2020_2022.csv.gz"
DEFAULT_CLOCK_OUTPUT = "data/coinbase_leadership_support_clocks_2020_2022.csv.gz"
DEFAULT_MANIFEST = "results/coinbase_spot_leadership_support_manifest_2026-07-16.json"
ROBUST_WINDOW = 30 * 24 * 12
ROBUST_MINIMUM = 14 * 24 * 12
PREMIUM_CENTER_WINDOW = 3 * 24 * 12
MISSING_QUARANTINE_BARS = 12
SHARE_CLIP = 1e-12


@dataclass(frozen=True)
class Config:
    source_manifest: str = DEFAULT_SOURCE_MANIFEST
    preregistration: str = DEFAULT_PREREGISTRATION
    feature_output: str = DEFAULT_FEATURE_OUTPUT
    clock_output: str = DEFAULT_CLOCK_OUTPUT
    manifest_output: str = DEFAULT_MANIFEST
    robust_block_rows: int = 256


def validate_hashed_manifest(path: str | Path, timestamp_key: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", timestamp_key}
    }
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError(f"manifest hash mismatch: {path}")
    return payload


def resolve_manifest_output(
    source_manifest: dict[str, Any], name: str
) -> Path:
    metadata = source_manifest["outputs"][name]
    path = Path(metadata["path"])
    if not path.exists() or sha256_file(path) != metadata["sha256"]:
        raise RuntimeError(f"frozen {name} source is missing or hash-mismatched")
    return path


def lagged_exact_robust_zscore(
    values: pd.Series,
    *,
    window: int = ROBUST_WINDOW,
    minimum: int = ROBUST_MINIMUM,
    block_rows: int = 256,
) -> pd.Series:
    """Exact median/MAD over the finite strictly-prior rolling observations."""
    if not (1 <= minimum <= window) or block_rows < 1:
        raise ValueError("invalid robust rolling configuration")
    raw = values.to_numpy(dtype=float, copy=True)
    padded = np.concatenate([np.full(window, np.nan), raw])
    prior_windows = np.lib.stride_tricks.sliding_window_view(padded, window)[: len(raw)]
    output = np.full(len(raw), np.nan)
    for start in range(0, len(raw), block_rows):
        stop = min(len(raw), start + block_rows)
        block = prior_windows[start:stop]
        counts = np.isfinite(block).sum(axis=1)
        eligible = (counts >= minimum) & np.isfinite(raw[start:stop])
        if not eligible.any():
            continue
        selected = block[eligible]
        centers = np.nanmedian(selected, axis=1)
        deviations = np.abs(selected - centers[:, None])
        mad = np.nanmedian(deviations, axis=1)
        scale = 1.4826 * mad
        current = raw[start:stop][eligible]
        valid_scale = np.isfinite(scale) & (scale > 1e-12)
        z = np.full(len(scale), np.nan)
        z[valid_scale] = (current[valid_scale] - centers[valid_scale]) / scale[valid_scale]
        block_output = output[start:stop]
        block_output[eligible] = z
        output[start:stop] = block_output
    return pd.Series(output, index=values.index, name=values.name)


def missing_quarantine(source_complete: pd.Series) -> pd.Series:
    missing = ~source_complete.astype(bool)
    quarantined = missing.rolling(MISSING_QUARANTINE_BARS + 1, min_periods=1).max()
    return quarantined.astype(bool)


def load_feature_inputs(
    source_manifest: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coinbase = pd.read_csv(
        resolve_manifest_output(source_manifest, "coinbase"),
        usecols=["date", "close", "volume", "source_complete"],
        parse_dates=["date"],
    )
    binance = pd.read_csv(
        resolve_manifest_output(source_manifest, "binance"),
        usecols=["date", "close", "quote_asset_volume"],
        parse_dates=["date"],
    )
    if len(coinbase) != len(binance) or not coinbase["date"].equals(binance["date"]):
        raise RuntimeError("Coinbase/Binance frozen signal grids do not align")
    if coinbase["date"].max() >= pd.Timestamp("2023-01-01"):
        raise RuntimeError("support builder refuses to open the sealed 2023 holdout")
    return coinbase, binance


def build_features(
    coinbase: pd.DataFrame,
    binance: pd.DataFrame,
    *,
    block_rows: int,
) -> pd.DataFrame:
    complete = coinbase["source_complete"].eq(1)
    cb_close = pd.to_numeric(coinbase["close"], errors="coerce").where(complete)
    cb_volume = pd.to_numeric(coinbase["volume"], errors="coerce").where(complete)
    bn_close = pd.to_numeric(binance["close"], errors="coerce")
    bn_quote = pd.to_numeric(binance["quote_asset_volume"], errors="coerce")
    if (bn_close <= 0).any() or (bn_quote < 0).any():
        raise ValueError("invalid Binance signal inputs")

    cb_return = np.log(cb_close / cb_close.shift(1))
    bn_return = np.log(bn_close / bn_close.shift(1))
    relative_return = cb_return - bn_return
    cb_quote = cb_volume * cb_close
    denominator = cb_quote + bn_quote
    activity_share = (cb_quote / denominator).clip(SHARE_CLIP, 1.0 - SHARE_CLIP)
    logit_share = np.log(activity_share / (1.0 - activity_share))
    premium = np.log(cb_close / bn_close)
    premium_prior_center = premium.shift(1).rolling(
        PREMIUM_CENTER_WINDOW, min_periods=1
    ).median()
    premium_residual = premium - premium_prior_center
    premium_residual_change = premium_residual - premium_residual.shift(1)

    raw_features = {
        "ZR": relative_return,
        "ZP": premium_residual_change,
        "ZV": logit_share,
        "ZCB": cb_return,
        "ZBN": bn_return,
    }
    standardized: dict[str, pd.Series] = {}
    for token, values in raw_features.items():
        print(f"robust_z_start token={token} rows={len(values)}", flush=True)
        standardized[token] = lagged_exact_robust_zscore(
            values,
            block_rows=block_rows,
        )
        print(
            f"robust_z_done token={token} finite={int(standardized[token].notna().sum())}",
            flush=True,
        )

    quarantined = missing_quarantine(complete)
    frame = pd.DataFrame({"date": coinbase["date"]})
    frame["source_complete"] = complete.astype(np.int8)
    frame["source_quarantined"] = quarantined.astype(np.int8)
    for token, values in standardized.items():
        frame[token] = values
    return frame


def policy_mask(features: pd.DataFrame, policy: Policy) -> np.ndarray:
    side = float(policy.side)
    zr = features["ZR"].to_numpy(float)
    zp = features["ZP"].to_numpy(float)
    zv = features["ZV"].to_numpy(float)
    zcb = features["ZCB"].to_numpy(float)
    zbn = features["ZBN"].to_numpy(float)
    eligible = features["source_quarantined"].eq(0).to_numpy(bool)
    if policy.family == "relative_return_lead":
        rule = (side * zr >= 2.0) & (side * zcb >= 1.0) & (side * zbn < 1.5)
    elif policy.family == "premium_shock":
        rule = (side * zp >= 2.0) & (side * zcb >= 0.5)
    elif policy.family == "activity_confirmed_relative":
        rule = (zv >= 2.0) & (side * zcb >= 1.5) & (side * zr >= 1.0)
    elif policy.family == "activity_premium_confluence":
        rule = (zv >= 1.5) & (side * zp >= 1.5)
    elif policy.family == "return_premium_confluence":
        rule = (side * zr >= 1.5) & (side * zp >= 1.5)
    else:
        raise ValueError(f"unknown Coinbase policy family: {policy.family}")
    return eligible & rule


def schedule_nonoverlap(mask: np.ndarray, hold_bars: int) -> np.ndarray:
    selected: list[int] = []
    next_signal = 0
    maximum_signal = len(mask) - hold_bars - 2
    for index in np.flatnonzero(mask):
        if index < next_signal or index > maximum_signal:
            continue
        selected.append(int(index))
        next_signal = int(index) + hold_bars
    return np.asarray(selected, dtype=np.int64)


def build_policy_clocks(features: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for policy in policy_grid():
        indices = schedule_nonoverlap(policy_mask(features, policy), policy.hold_bars)
        for index in indices:
            records.append(
                {
                    "policy_id": policy.policy_id,
                    "family": policy.family,
                    "side": policy.side,
                    "hold_bars": policy.hold_bars,
                    "signal_date": features.at[index, "date"],
                    "signal_row": index,
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=["policy_id", "family", "side", "hold_bars", "signal_date", "signal_row"],
    ).sort_values(["policy_id", "signal_date"]).reset_index(drop=True)


def support_metrics(
    features: pd.DataFrame, family: str, hold_bars: int
) -> dict[str, Any]:
    policies = [
        policy
        for policy in policy_grid()
        if policy.family == family and policy.hold_bars == hold_bars
    ]
    if {policy.side for policy in policies} != {-1, 1}:
        raise RuntimeError("support group must contain a fixed long/short pair")
    side_by_index: dict[int, int] = {}
    for policy in policies:
        for index in np.flatnonzero(policy_mask(features, policy)):
            prior = side_by_index.get(int(index))
            if prior is not None and prior != policy.side:
                raise RuntimeError("long and short support masks overlap")
            side_by_index[int(index)] = policy.side
    raw_mask = np.zeros(len(features), dtype=bool)
    if side_by_index:
        raw_mask[np.fromiter(side_by_index, dtype=np.int64)] = True
    indices = schedule_nonoverlap(raw_mask, hold_bars)
    dates = features.loc[indices, "date"].reset_index(drop=True)
    sides = pd.Series([side_by_index[int(index)] for index in indices], dtype=int)
    total = int(len(indices))
    by_year = {str(year): int((dates.dt.year == year).sum()) for year in (2020, 2021, 2022)}
    by_side = {str(side): int((sides == side).sum()) for side in (-1, 1)}
    side_shares = {
        side: (count / total if total else 0.0) for side, count in by_side.items()
    }
    if total:
        monthly = dates.dt.to_period("M").value_counts()
        max_month_share = float(monthly.max() / total)
        max_month = str(monthly.idxmax())
    else:
        max_month_share = 1.0
        max_month = None
    gates = {
        "total_min_120": total >= 120,
        "each_year_min_25": min(by_year.values()) >= 25,
        "each_side_share_min_0_20": min(side_shares.values()) >= 0.20,
        "single_month_share_max_0_20": max_month_share <= 0.20,
    }
    return {
        "support_group": f"{family}_h{hold_bars}",
        "family": family,
        "hold_bars": hold_bars,
        "policy_ids": [policy.policy_id for policy in policies],
        "nonoverlap_events": total,
        "events_by_year": by_year,
        "events_by_side": by_side,
        "side_shares": side_shares,
        "maximum_month_share": max_month_share,
        "maximum_month": max_month,
        "gates": gates,
        "pass": all(gates.values()),
    }


def source_quality(features: pd.DataFrame) -> dict[str, Any]:
    unavailable = features["source_quarantined"].eq(1)
    dates = features["date"]
    month = dates.dt.to_period("M")
    monthly = unavailable.groupby(month).mean()
    global_fraction = float(unavailable.mean())
    maximum_month_fraction = float(monthly.max())
    gates = {
        "global_missing_or_quarantined_fraction_max_0_01": global_fraction <= 0.01,
        "monthly_missing_or_quarantined_fraction_max_0_03": maximum_month_fraction <= 0.03,
    }
    return {
        "source_missing_rows": int(features["source_complete"].eq(0).sum()),
        "missing_or_next_12_quarantined_rows": int(unavailable.sum()),
        "global_missing_or_quarantined_fraction": global_fraction,
        "maximum_monthly_missing_or_quarantined_fraction": maximum_month_fraction,
        "maximum_month": str(monthly.idxmax()),
        "gates": gates,
        "pass": all(gates.values()),
    }


def run(cfg: Config) -> dict[str, Any]:
    prereg = json.loads(Path(cfg.preregistration).read_text())
    validate_preregistration(prereg)
    source_manifest = validate_hashed_manifest(cfg.source_manifest, "retrieved_at")
    if source_manifest.get("forward_trade_outcomes_opened") is not False:
        raise RuntimeError("support phase requires unopened forward trade outcomes")
    audit = source_manifest.get("audit_amendment", {})
    if (
        audit.get("before_forward_trade_outcomes") is not True
        or audit.get("changed_source_values") is not False
        or "raw_inputs" not in source_manifest
        or "git_anchors" not in source_manifest
    ):
        raise RuntimeError("support phase requires the hardened source attestation")
    if source_manifest.get("end_exclusive") != "2023-01-01":
        raise RuntimeError("support source does not preserve the 2023 holdout")

    coinbase, binance = load_feature_inputs(source_manifest)
    features = build_features(coinbase, binance, block_rows=cfg.robust_block_rows)
    quality = source_quality(features)
    clocks = build_policy_clocks(features)
    groups = sorted({(policy.family, policy.hold_bars) for policy in policy_grid()})
    support = [support_metrics(features, family, hold) for family, hold in groups]
    passing_policy_ids = sorted(
        {policy_id for item in support if item["pass"] for policy_id in item["policy_ids"]}
    )
    if not quality["pass"]:
        passing_policy_ids = []

    deterministic_gzip_csv(features, cfg.feature_output)
    deterministic_gzip_csv(clocks, cfg.clock_output)
    core: dict[str, Any] = {
        "protocol_version": "coinbase_spot_leadership_support_v1",
        "phase": "support_only_before_forward_returns",
        "forward_trade_outcomes_opened": False,
        "source_columns_read": {
            "coinbase": ["date", "close", "volume", "source_complete"],
            "binance": ["date", "close", "quote_asset_volume"],
            "explicitly_not_read": [
                "Binance next open",
                "Binance future high/low",
                "funding realized after signal",
            ],
        },
        "implementation_contract": {
            "robust_window_bars": ROBUST_WINDOW,
            "robust_minimum_finite_prior_bars": ROBUST_MINIMUM,
            "median_mad": "exact finite observations in strictly-prior rolling window",
            "premium_prior_center_bars": PREMIUM_CENTER_WINDOW,
            "premium_prior_center_minimum": 1,
            "activity_share_clip": SHARE_CLIP,
            "missing_quarantine_bars_after_missing": MISSING_QUARANTINE_BARS,
            "nonoverlap": "next signal index >= prior signal index + hold_bars",
        },
        "hashes": {
            "preregistration_file_sha256": sha256_file(cfg.preregistration),
            "preregistration_manifest_hash": prereg["manifest_hash"],
            "source_manifest_file_sha256": sha256_file(cfg.source_manifest),
            "source_manifest_hash": source_manifest["manifest_hash"],
            "support_code_sha256": sha256_file(__file__),
        },
        "quality": quality,
        "support_groups": support,
        "passing_policy_ids": passing_policy_ids,
        "all_support_groups_pass": len(passing_policy_ids) == len(policy_grid()),
        "clock_counts_by_policy": {
            policy.policy_id: int((clocks["policy_id"] == policy.policy_id).sum())
            for policy in policy_grid()
        },
        "outputs": {
            "features": {
                "path": cfg.feature_output,
                "rows": int(len(features)),
                "bytes": Path(cfg.feature_output).stat().st_size,
                "sha256": sha256_file(cfg.feature_output),
            },
            "clocks": {
                "path": cfg.clock_output,
                "rows": int(len(clocks)),
                "bytes": Path(cfg.clock_output).stat().st_size,
                "sha256": sha256_file(cfg.clock_output),
            },
        },
        "selection_may_open_forward_returns": bool(passing_policy_ids),
        "sealed_holdout": ["2023-01-01", "2024-01-01"],
        "future_2024_plus_sealed": True,
    }
    payload = {
        **core,
        "manifest_hash": canonical_hash(core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(cfg.manifest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    for name in asdict(defaults):
        value = getattr(defaults, name)
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    return parser.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    print(
        json.dumps(
            {
                "manifest_hash": payload["manifest_hash"],
                "quality": payload["quality"],
                "support_groups": payload["support_groups"],
                "passing_policy_ids": payload["passing_policy_ids"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
