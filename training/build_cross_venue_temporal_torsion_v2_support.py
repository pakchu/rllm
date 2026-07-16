"""Freeze CVTT v2 causal features and event clocks before trade returns."""
from __future__ import annotations

import argparse
import json
import subprocess
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
from training.preregister_cross_venue_temporal_torsion_alpha_v2 import (
    DEFAULT_OUTPUT as DEFAULT_PREREGISTRATION,
    MINIMUM_CLEAN_CALENDAR_BARS,
    MINIMUM_PRIOR_ROUTE_EVENTS,
    Policy,
    canonical_hash,
    policy_grid,
    validate_manifest as validate_preregistration,
)


DEFAULT_SOURCE_MANIFEST = (
    "results/cross_venue_temporal_torsion_v2_source_manifest_2026-07-16.json"
)
DEFAULT_FEATURE_OUTPUT = (
    "data/cross_venue_temporal_torsion_v2_support_features_2020_2022.csv.gz"
)
DEFAULT_CLOCK_OUTPUT = (
    "data/cross_venue_temporal_torsion_v2_support_clocks_2020_2022.csv.gz"
)
DEFAULT_MANIFEST = (
    "results/cross_venue_temporal_torsion_v2_support_manifest_2026-07-16.json"
)
ROLLING_WINDOW_BARS = 8_640
ROUTE_QUANTILE = 0.95
EPISODE_LOOKBACK_BARS = 12
ENTRY_DELAY_BARS = 2


@dataclass(frozen=True)
class Config:
    source_manifest: str = DEFAULT_SOURCE_MANIFEST
    preregistration: str = DEFAULT_PREREGISTRATION
    feature_output: str = DEFAULT_FEATURE_OUTPUT
    clock_output: str = DEFAULT_CLOCK_OUTPUT
    manifest_output: str = DEFAULT_MANIFEST


def validate_hashed_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError(f"manifest hash mismatch: {path}")
    return payload


def resolve_manifest_output(source: dict[str, Any], name: str) -> Path:
    metadata = source["outputs"][name]
    path = Path(metadata["path"])
    if not path.exists() or sha256_file(path) != metadata["sha256"]:
        raise RuntimeError(f"frozen CVTT v2 {name} source is missing or mismatched")
    return path


def git_anchor(path: str | Path) -> dict[str, str]:
    candidate = Path(path)
    pathspec = str(candidate.resolve().relative_to(Path.cwd().resolve()))
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", pathspec],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"CVTT v2 support artifact is not clean: {pathspec}")
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", pathspec],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise RuntimeError(f"CVTT v2 support artifact is not committed: {pathspec}")
    return {"path": pathspec, "commit": commit, "sha256": sha256_file(candidate)}


def lagged_route_quantile(
    score: pd.Series,
    route_eligible: pd.Series,
    clean: pd.Series,
    *,
    window: int = ROLLING_WINDOW_BARS,
    minimum_clean: int = MINIMUM_CLEAN_CALENDAR_BARS,
    minimum_events: int = MINIMUM_PRIOR_ROUTE_EVENTS,
    quantile: float = ROUTE_QUANTILE,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if not (1 <= minimum_clean <= window) or not (1 <= minimum_events <= window):
        raise ValueError("invalid CVTT v2 rolling support configuration")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("invalid CVTT v2 route quantile")
    clean_prior_count = clean.astype(np.int8).shift(1, fill_value=0).rolling(
        window, min_periods=1
    ).sum()
    event_prior_count = route_eligible.astype(np.int8).shift(
        1, fill_value=0
    ).rolling(window, min_periods=1).sum()
    threshold = score.where(route_eligible).shift(1).rolling(
        window, min_periods=minimum_events
    ).quantile(quantile)
    threshold = threshold.where(clean_prior_count.ge(minimum_clean))
    return threshold, clean_prior_count, event_prior_count


def episode_start(active: pd.Series) -> pd.Series:
    prior = active.astype(bool).shift(1, fill_value=False).rolling(
        EPISODE_LOOKBACK_BARS, min_periods=1
    ).max().astype(bool)
    return active.astype(bool) & ~prior


def load_features(source: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "date",
        "feature_available_time_utc",
        "strategy_entry_earliest_time_utc",
        "source_valid_current",
        "source_quarantined",
        "source_available",
        "spot_flow_fraction",
        "um_flow_fraction",
        "spot_log_return_5m",
        "um_log_return_5m",
        "spot_flow_time_centroid",
        "um_flow_time_centroid",
        "spot_return_time_centroid",
        "um_return_time_centroid",
    ]
    frame = pd.read_csv(
        resolve_manifest_output(source, "features"),
        usecols=columns,
        parse_dates=[
            "date",
            "feature_available_time_utc",
            "strategy_entry_earliest_time_utc",
        ],
    )
    expected = pd.date_range("2020-01-01", "2023-01-01", freq="5min", inclusive="left")
    if not frame["date"].equals(pd.Series(expected)):
        raise RuntimeError("CVTT v2 support source is not the exact frozen grid")
    if not frame["feature_available_time_utc"].equals(
        frame["date"] + pd.Timedelta("5min")
    ):
        raise RuntimeError("CVTT v2 feature availability clock is invalid")
    if not frame["strategy_entry_earliest_time_utc"].equals(
        frame["date"] + pd.Timedelta("10min")
    ):
        raise RuntimeError("CVTT v2 strategy entry clock is invalid")
    return frame


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    clean = raw["source_available"].eq(1) & raw["source_quarantined"].eq(0)
    signal_columns = [
        "spot_flow_fraction",
        "um_flow_fraction",
        "spot_log_return_5m",
        "um_log_return_5m",
        "spot_flow_time_centroid",
        "um_flow_time_centroid",
        "spot_return_time_centroid",
        "um_return_time_centroid",
    ]
    values = raw[signal_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.loc[clean].to_numpy(float)).all():
        raise ValueError("clean CVTT v2 source contains a non-finite descriptor")
    if values.loc[~clean].notna().any().any():
        raise ValueError("quarantined CVTT v2 source retained signal descriptors")

    spot_delay = (
        values["spot_return_time_centroid"] - values["spot_flow_time_centroid"]
    )
    um_delay = values["um_return_time_centroid"] - values["um_flow_time_centroid"]
    spot_preload = spot_delay.clip(lower=0.0)
    spot_echo = (-spot_delay).clip(lower=0.0)
    um_preload = um_delay.clip(lower=0.0)
    um_echo = (-um_delay).clip(lower=0.0)
    spot_to_um_score = np.sqrt(spot_preload * um_echo)
    um_to_spot_score = np.sqrt(um_preload * spot_echo)
    spot_side = np.sign(values["spot_flow_fraction"]).fillna(0.0)
    um_side = np.sign(values["um_flow_fraction"]).fillna(0.0)
    spot_confirmed = (
        spot_side.ne(0)
        & (spot_side * values["um_flow_fraction"]).gt(0)
        & (spot_side * values["spot_log_return_5m"]).gt(0)
        & (spot_side * values["um_log_return_5m"]).gt(0)
    )
    um_confirmed = (
        um_side.ne(0)
        & (um_side * values["spot_flow_fraction"]).gt(0)
        & (um_side * values["um_log_return_5m"]).gt(0)
        & (um_side * values["spot_log_return_5m"]).gt(0)
    )
    spot_route_eligible = clean & spot_delay.gt(0) & um_delay.lt(0) & spot_confirmed
    um_route_eligible = clean & um_delay.gt(0) & spot_delay.lt(0) & um_confirmed

    spot_threshold, clean_prior_count, spot_prior_events = lagged_route_quantile(
        spot_to_um_score, spot_route_eligible, clean
    )
    um_threshold, clean_prior_count_2, um_prior_events = lagged_route_quantile(
        um_to_spot_score, um_route_eligible, clean
    )
    if not clean_prior_count.equals(clean_prior_count_2):
        raise RuntimeError("CVTT v2 clean support clocks disagree")
    spot_active = spot_route_eligible & spot_to_um_score.ge(spot_threshold)
    um_active = um_route_eligible & um_to_spot_score.ge(um_threshold)

    return pd.DataFrame(
        {
            "date": raw["date"],
            "feature_available_time_utc": raw["feature_available_time_utc"],
            "strategy_entry_earliest_time_utc": raw[
                "strategy_entry_earliest_time_utc"
            ],
            "source_valid_current": raw["source_valid_current"].astype(np.int8),
            "source_quarantined": raw["source_quarantined"].astype(np.int8),
            "source_available": clean.astype(np.int8),
            "spot_flow_to_return_delay": spot_delay,
            "um_flow_to_return_delay": um_delay,
            "spot_to_um_score": spot_to_um_score,
            "um_to_spot_score": um_to_spot_score,
            "spot_source_side": spot_side.astype(np.int8),
            "um_source_side": um_side.astype(np.int8),
            "clean_prior_30d_bars": clean_prior_count,
            "spot_to_um_prior_30d_events": spot_prior_events,
            "um_to_spot_prior_30d_events": um_prior_events,
            "spot_to_um_q95": spot_threshold,
            "um_to_spot_q95": um_threshold,
            "spot_to_um_eligible": spot_route_eligible.astype(np.int8),
            "um_to_spot_eligible": um_route_eligible.astype(np.int8),
            "spot_to_um_active": spot_active.astype(np.int8),
            "um_to_spot_active": um_active.astype(np.int8),
            "spot_to_um_start": episode_start(spot_active).astype(np.int8),
            "um_to_spot_start": episode_start(um_active).astype(np.int8),
        }
    )


def route_start_and_side(
    features: pd.DataFrame, policy: Policy
) -> tuple[np.ndarray, np.ndarray]:
    if policy.route == "spot_preload_um_echo":
        mask = features["spot_to_um_start"].eq(1).to_numpy(bool)
        side = features["spot_source_side"].to_numpy(np.int8)
    elif policy.route == "um_preload_spot_echo":
        mask = features["um_to_spot_start"].eq(1).to_numpy(bool)
        side = features["um_source_side"].to_numpy(np.int8)
    else:
        raise ValueError(f"unknown CVTT v2 route: {policy.route}")
    if np.any(mask & ~np.isin(side, (-1, 1))):
        raise RuntimeError("eligible CVTT v2 event has invalid side")
    return mask, side


def schedule_nonoverlap(mask: np.ndarray, hold_bars: int) -> np.ndarray:
    if hold_bars < 1:
        raise ValueError("hold_bars must be positive")
    selected: list[int] = []
    next_signal = 0
    maximum_signal = len(mask) - hold_bars - ENTRY_DELAY_BARS - 1
    for index in np.flatnonzero(mask):
        if index < next_signal or index > maximum_signal:
            continue
        selected.append(int(index))
        next_signal = int(index) + hold_bars
    return np.asarray(selected, dtype=np.int64)


def build_clocks(features: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for policy in policy_grid():
        mask, sides = route_start_and_side(features, policy)
        for index in schedule_nonoverlap(mask, policy.hold_bars):
            records.append(
                {
                    "policy_id": policy.policy_id,
                    "route": policy.route,
                    "side": int(sides[index]),
                    "hold_bars": policy.hold_bars,
                    "signal_date": features.at[index, "date"],
                    "signal_row": int(index),
                    "entry_date": features.at[index, "strategy_entry_earliest_time_utc"],
                }
            )
    columns = [
        "policy_id",
        "route",
        "side",
        "hold_bars",
        "signal_date",
        "signal_row",
        "entry_date",
    ]
    return (
        pd.DataFrame.from_records(records, columns=columns)
        .sort_values(["policy_id", "signal_date"])
        .reset_index(drop=True)
    )


def support_metrics(clocks: pd.DataFrame, policy: Policy) -> dict[str, Any]:
    selected = clocks.loc[clocks["policy_id"].eq(policy.policy_id)]
    dates = pd.to_datetime(selected["signal_date"])
    sides = selected["side"].astype(int)
    total = int(len(selected))
    by_year = {
        str(year): int((dates.dt.year == year).sum()) for year in (2020, 2021, 2022)
    }
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
        "total_min_600": total >= 600,
        "each_year_min_150": min(by_year.values()) >= 150,
        "each_side_share_min_0_35": min(side_shares.values()) >= 0.35,
        "single_month_share_max_0_10": max_month_share <= 0.10,
    }
    return {
        "policy_id": policy.policy_id,
        "route": policy.route,
        "hold_bars": policy.hold_bars,
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
    monthly = unavailable.groupby(features["date"].dt.to_period("M")).mean()
    global_fraction = float(unavailable.mean())
    maximum_month_fraction = float(monthly.max())
    gates = {
        "global_missing_or_quarantined_fraction_max_0_01": global_fraction <= 0.01,
        "monthly_missing_or_quarantined_fraction_max_0_05": (
            maximum_month_fraction <= 0.05
        ),
    }
    return {
        "raw_invalid_rows": int(features["source_valid_current"].eq(0).sum()),
        "quarantined_rows": int(unavailable.sum()),
        "global_fraction": global_fraction,
        "maximum_monthly_fraction": maximum_month_fraction,
        "maximum_month": str(monthly.idxmax()),
        "gates": gates,
        "pass": all(gates.values()),
    }


def run(cfg: Config) -> dict[str, Any]:
    prereg = json.loads(Path(cfg.preregistration).read_text())
    validate_preregistration(prereg)
    source = validate_hashed_manifest(cfg.source_manifest)
    if source.get("forward_trade_outcomes_opened") is not False:
        raise RuntimeError("CVTT v2 support requires unopened forward outcomes")
    if source.get("preregistration", {}).get("manifest_hash") != prereg.get(
        "manifest_hash"
    ):
        raise RuntimeError("CVTT v2 source and preregistration are not linked")

    raw = load_features(source)
    features = build_features(raw)
    quality = source_quality(features)
    clocks = build_clocks(features)
    support = [support_metrics(clocks, policy) for policy in policy_grid()]
    passing = sorted(item["policy_id"] for item in support if item["pass"])
    if not quality["pass"]:
        passing = []

    deterministic_gzip_csv(features, cfg.feature_output)
    deterministic_gzip_csv(clocks, cfg.clock_output)
    core: dict[str, Any] = {
        "protocol_version": "cross_venue_temporal_torsion_support_v2",
        "phase": "support_only_before_forward_returns",
        "forward_trade_outcomes_opened": False,
        "source_columns_read": list(raw.columns),
        "explicitly_not_read": [
            "market open/high/low/close",
            "future return",
            "realized funding after signal",
            "sealed 2023 features",
        ],
        "implementation_contract": {
            "rolling_window_bars": ROLLING_WINDOW_BARS,
            "minimum_clean_prior_calendar_bars": MINIMUM_CLEAN_CALENDAR_BARS,
            "minimum_prior_route_events": MINIMUM_PRIOR_ROUTE_EVENTS,
            "route_quantile": ROUTE_QUANTILE,
            "all_thresholds_shifted_bars": 1,
            "episode_prior_active_lookback_bars": EPISODE_LOOKBACK_BARS,
            "entry_delay_bars_from_bucket_open": ENTRY_DELAY_BARS,
            "nonoverlap": "next signal row >= prior signal row + hold_bars",
        },
        "hashes": {
            "preregistration_file_sha256": sha256_file(cfg.preregistration),
            "preregistration_manifest_hash": prereg["manifest_hash"],
            "source_manifest_file_sha256": sha256_file(cfg.source_manifest),
            "source_manifest_hash": source["manifest_hash"],
        },
        "git_anchors": {
            "preregistration": git_anchor(cfg.preregistration),
            "source_freezer": git_anchor(
                "training/export_cross_venue_temporal_torsion_prefix.py"
            ),
            "support_builder": git_anchor(__file__),
        },
        "quality": quality,
        "support_policies": support,
        "passing_policy_ids": passing,
        "all_support_policies_pass": len(passing) == len(policy_grid()),
        "clock_counts_by_policy": {
            policy.policy_id: int(clocks["policy_id"].eq(policy.policy_id).sum())
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
        "selection_may_open_forward_returns": bool(passing),
        "sealed_holdout": ["2023-01-01", "2024-01-01"],
    }
    payload = {
        **core,
        "manifest_hash": canonical_hash(core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(cfg.manifest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    for name, value in asdict(defaults).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    return parser.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    print(
        json.dumps(
            {
                "manifest_hash": payload["manifest_hash"],
                "quality": payload["quality"],
                "support_policies": payload["support_policies"],
                "passing_policy_ids": payload["passing_policy_ids"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
