"""Freeze causal TAAR features and event clocks before opening trade returns."""
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
from training.preregister_tail_arrival_absorption_alpha import (
    DEFAULT_OUTPUT as DEFAULT_PREREGISTRATION,
    Policy,
    canonical_hash,
    policy_grid,
    validate_manifest as validate_preregistration,
)


DEFAULT_SOURCE_MANIFEST = "results/tail_arrival_source_manifest_2026-07-16.json"
DEFAULT_FEATURE_OUTPUT = "data/tail_arrival_support_features_2020_2022.csv.gz"
DEFAULT_CLOCK_OUTPUT = "data/tail_arrival_support_clocks_2020_2022.csv.gz"
DEFAULT_MANIFEST = "results/tail_arrival_support_manifest_2026-07-16.json"
ROLLING_WINDOW_BARS = 30 * 24 * 12
ROLLING_MINIMUM_BARS = 7 * 24 * 12
QUARANTINE_FOLLOWING_BARS = 24
EPISODE_LOOKBACK_BARS = 12
ENTRY_DELAY_BARS = 2
MINIMUM_AGG_TRADE_COUNT = 64
SOURCE_GLOBAL_MAX = 0.02
SOURCE_MONTHLY_MAX = 0.05


@dataclass(frozen=True)
class Config:
    source_manifest: str = DEFAULT_SOURCE_MANIFEST
    preregistration: str = DEFAULT_PREREGISTRATION
    feature_output: str = DEFAULT_FEATURE_OUTPUT
    clock_output: str = DEFAULT_CLOCK_OUTPUT
    manifest_output: str = DEFAULT_MANIFEST


def validate_hashed_manifest(
    path: str | Path, *, timestamp_keys: tuple[str, ...] = ("created_at",)
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", *timestamp_keys}
    }
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError(f"manifest hash mismatch: {path}")
    return payload


def resolve_manifest_output(source_manifest: dict[str, Any], name: str) -> Path:
    metadata = source_manifest["outputs"][name]
    path = Path(metadata["path"])
    if not path.exists() or sha256_file(path) != metadata["sha256"]:
        raise RuntimeError(f"frozen {name} source is missing or hash-mismatched")
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
        raise RuntimeError(f"support artifact is not clean: {pathspec}")
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", pathspec],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise RuntimeError(f"support artifact is not committed: {pathspec}")
    return {"path": pathspec, "commit": commit, "sha256": sha256_file(candidate)}


def missing_quarantine(invalid: pd.Series) -> pd.Series:
    """Quarantine an invalid bar and the following 24 completed buckets."""
    return invalid.astype(bool).rolling(
        QUARANTINE_FOLLOWING_BARS + 1, min_periods=1
    ).max().astype(bool)


def lagged_quantile(
    values: pd.Series,
    eligible: pd.Series,
    quantile: float,
    *,
    window: int = ROLLING_WINDOW_BARS,
    minimum: int = ROLLING_MINIMUM_BARS,
) -> pd.Series:
    """Rolling quantile of finite eligible observations strictly before each bar."""
    if not (0.0 <= quantile <= 1.0) or not (1 <= minimum <= window):
        raise ValueError("invalid lagged quantile configuration")
    prior = values.where(eligible.astype(bool)).shift(1)
    return prior.rolling(window, min_periods=minimum).quantile(quantile)


def episode_start(active: pd.Series) -> pd.Series:
    """Current active bar with no active bar in the prior 12 completed buckets."""
    prior_active = active.astype(bool).shift(1, fill_value=False).rolling(
        EPISODE_LOOKBACK_BARS, min_periods=1
    ).max().astype(bool)
    return active.astype(bool) & ~prior_active


def load_feature_input(source_manifest: dict[str, Any]) -> pd.DataFrame:
    usecols = [
        "date",
        "source_complete",
        "source_gap_day",
        "agg_trade_count",
        "event_notional_mean",
        "event_notional_std",
        "event_notional_p50",
        "event_notional_p99",
        "event_notional_max",
        "interarrival_mean_ms",
        "interarrival_std_ms",
        "buy_sell_event_size_log_ratio",
        "micro_log_return",
    ]
    frame = pd.read_csv(
        resolve_manifest_output(source_manifest, "features"),
        usecols=usecols,
        parse_dates=["date"],
    )
    if frame.empty or frame["date"].max() >= pd.Timestamp("2023-01-01"):
        raise RuntimeError("support builder refuses the sealed 2023 holdout")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise RuntimeError("tail-arrival signal grid is not unique and chronological")
    expected = pd.date_range("2020-01-01", "2023-01-01", freq="5min", inclusive="left")
    if not frame["date"].equals(pd.Series(expected)):
        raise RuntimeError("tail-arrival support input is not the exact frozen grid")
    return frame


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "agg_trade_count",
        "event_notional_mean",
        "event_notional_std",
        "event_notional_p50",
        "event_notional_p99",
        "event_notional_max",
        "interarrival_mean_ms",
        "interarrival_std_ms",
        "buy_sell_event_size_log_ratio",
        "micro_log_return",
    ]
    numeric = raw[numeric_columns].apply(pd.to_numeric, errors="coerce")
    complete = raw["source_complete"].eq(1)
    gap_day = raw["source_gap_day"].eq(1)
    invalid_source = ~complete | gap_day
    quarantined = missing_quarantine(invalid_source)

    finite = pd.Series(np.isfinite(numeric.to_numpy()).all(axis=1), index=raw.index)
    coherent = (
        numeric["agg_trade_count"].ge(1)
        & numeric["event_notional_mean"].gt(0)
        & numeric["event_notional_std"].ge(0)
        & numeric["event_notional_p50"].gt(0)
        & numeric["event_notional_p99"].ge(numeric["event_notional_p50"])
        & numeric["event_notional_max"].ge(numeric["event_notional_p99"])
        & numeric["interarrival_mean_ms"].gt(0)
        & numeric["interarrival_std_ms"].ge(0)
    )
    feature_valid = complete & finite & coherent
    threshold_eligible = (
        feature_valid
        & ~quarantined
        & numeric["agg_trade_count"].ge(MINIMUM_AGG_TRADE_COUNT)
    )

    tail_span = np.log(
        numeric["event_notional_p99"] / numeric["event_notional_p50"]
    ) + 0.5 * np.log(
        numeric["event_notional_max"] / numeric["event_notional_p99"]
    )
    event_dispersion = np.log1p(
        numeric["event_notional_std"] / numeric["event_notional_mean"]
    )
    arrival_cv = np.log1p(
        numeric["interarrival_std_ms"] / numeric["interarrival_mean_ms"]
    )
    size_asymmetry = numeric["buy_sell_event_size_log_ratio"]
    packet_direction = np.sign(size_asymmetry).where(
        np.isfinite(size_asymmetry), 0.0
    ).astype(float)
    signed_response = packet_direction * numeric["micro_log_return"]
    abs_micro_return = numeric["micro_log_return"].abs()

    tail_q95 = lagged_quantile(tail_span, threshold_eligible, 0.95)
    dispersion_q75 = lagged_quantile(event_dispersion, threshold_eligible, 0.75)
    arrival_q75 = lagged_quantile(arrival_cv, threshold_eligible, 0.75)
    abs_size_asymmetry_q80 = lagged_quantile(
        size_asymmetry.abs(), threshold_eligible, 0.80
    )
    release_response_q75 = lagged_quantile(
        abs_micro_return, threshold_eligible, 0.75
    )

    thresholds_finite = pd.Series(
        np.isfinite(
            np.column_stack(
                [
                    tail_q95,
                    dispersion_q75,
                    arrival_q75,
                    abs_size_asymmetry_q80,
                    release_response_q75,
                ]
            )
        ).all(axis=1),
        index=raw.index,
    )
    common = (
        threshold_eligible
        & thresholds_finite
        & packet_direction.ne(0)
        & tail_span.ge(tail_q95)
        & event_dispersion.ge(dispersion_q75)
        & arrival_cv.ge(arrival_q75)
        & size_asymmetry.abs().ge(abs_size_asymmetry_q80)
    )
    absorption_active = common & signed_response.le(0)
    release_active = common & signed_response.ge(release_response_q75)

    frame = pd.DataFrame(
        {
            "date": raw["date"],
            "source_complete": complete.astype(np.int8),
            "source_gap_day": gap_day.astype(np.int8),
            "source_quarantined": quarantined.astype(np.int8),
            "feature_valid": feature_valid.astype(np.int8),
            "threshold_eligible": threshold_eligible.astype(np.int8),
            "agg_trade_count": numeric["agg_trade_count"],
            "tail_span": tail_span,
            "event_dispersion": event_dispersion,
            "arrival_cv": arrival_cv,
            "size_asymmetry": size_asymmetry,
            "packet_direction": packet_direction.astype(np.int8),
            "signed_response": signed_response,
            "tail_q95": tail_q95,
            "dispersion_q75": dispersion_q75,
            "arrival_q75": arrival_q75,
            "abs_size_asymmetry_q80": abs_size_asymmetry_q80,
            "release_response_q75": release_response_q75,
            "absorption_active": absorption_active.astype(np.int8),
            "absorption_start": episode_start(absorption_active).astype(np.int8),
            "release_active": release_active.astype(np.int8),
            "release_start": episode_start(release_active).astype(np.int8),
        }
    )
    invalid_derived = feature_valid & ~pd.Series(
        np.isfinite(
            frame[
                [
                    "tail_span",
                    "event_dispersion",
                    "arrival_cv",
                    "size_asymmetry",
                    "signed_response",
                ]
            ].to_numpy()
        ).all(axis=1),
        index=frame.index,
    )
    if invalid_derived.any():
        raise RuntimeError("finite coherent source produced a non-finite TAAR feature")
    return frame


def branch_start_and_side(
    features: pd.DataFrame, policy: Policy
) -> tuple[np.ndarray, np.ndarray]:
    direction = features["packet_direction"].to_numpy(np.int8)
    if policy.branch == "tail_absorption_fade":
        mask = features["absorption_start"].eq(1).to_numpy(bool)
        side = -direction
    elif policy.branch == "tail_release_follow":
        mask = features["release_start"].eq(1).to_numpy(bool)
        side = direction
    else:
        raise ValueError(f"unknown TAAR branch: {policy.branch}")
    if np.any(mask & ~np.isin(side, (-1, 1))):
        raise RuntimeError("eligible TAAR event has invalid side")
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


def build_policy_clocks(features: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for policy in policy_grid():
        mask, side = branch_start_and_side(features, policy)
        for index in schedule_nonoverlap(mask, policy.hold_bars):
            records.append(
                {
                    "policy_id": policy.policy_id,
                    "branch": policy.branch,
                    "side": int(side[index]),
                    "hold_bars": policy.hold_bars,
                    "signal_date": features.at[index, "date"],
                    "signal_row": int(index),
                }
            )
    columns = [
        "policy_id",
        "branch",
        "side",
        "hold_bars",
        "signal_date",
        "signal_row",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame.from_records(records, columns=columns)
        .sort_values(["policy_id", "signal_date"])
        .reset_index(drop=True)
    )


def support_metrics(clocks: pd.DataFrame, policy: Policy) -> dict[str, Any]:
    selected = clocks.loc[clocks["policy_id"].eq(policy.policy_id)].copy()
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
        "total_min_120": total >= 120,
        "each_year_min_25": min(by_year.values()) >= 25,
        "each_side_share_min_0_20": min(side_shares.values()) >= 0.20,
        "single_month_share_max_0_20": max_month_share <= 0.20,
    }
    return {
        "policy_id": policy.policy_id,
        "branch": policy.branch,
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
        "global_missing_or_quarantined_fraction_max_0_02": (
            global_fraction <= SOURCE_GLOBAL_MAX
        ),
        "monthly_missing_or_quarantined_fraction_max_0_05": (
            maximum_month_fraction <= SOURCE_MONTHLY_MAX
        ),
    }
    return {
        "source_missing_rows": int(features["source_complete"].eq(0).sum()),
        "source_gap_day_rows": int(features["source_gap_day"].eq(1).sum()),
        "missing_gap_or_following_24_quarantined_rows": int(unavailable.sum()),
        "global_missing_or_quarantined_fraction": global_fraction,
        "maximum_monthly_missing_or_quarantined_fraction": maximum_month_fraction,
        "maximum_month": str(monthly.idxmax()),
        "gates": gates,
        "pass": all(gates.values()),
    }


def run(cfg: Config) -> dict[str, Any]:
    prereg = json.loads(Path(cfg.preregistration).read_text())
    validate_preregistration(prereg)
    source_manifest = validate_hashed_manifest(cfg.source_manifest)
    if source_manifest.get("forward_trade_outcomes_opened") is not False:
        raise RuntimeError("support phase requires unopened forward trade outcomes")
    if source_manifest.get("end_exclusive") != "2023-01-01":
        raise RuntimeError("support source does not preserve the 2023 holdout")
    if source_manifest.get("preregistration", {}).get("manifest_hash") != prereg.get(
        "manifest_hash"
    ):
        raise RuntimeError("source and preregistration manifests are not linked")

    raw = load_feature_input(source_manifest)
    features = build_features(raw)
    quality = source_quality(features)
    clocks = build_policy_clocks(features)
    support = [support_metrics(clocks, policy) for policy in policy_grid()]
    passing_policy_ids = sorted(item["policy_id"] for item in support if item["pass"])
    if not quality["pass"]:
        passing_policy_ids = []

    deterministic_gzip_csv(features, cfg.feature_output)
    deterministic_gzip_csv(clocks, cfg.clock_output)
    core: dict[str, Any] = {
        "protocol_version": "tail_arrival_support_v1",
        "phase": "support_only_before_forward_returns",
        "forward_trade_outcomes_opened": False,
        "source_columns_read": {
            "features": list(raw.columns),
            "explicitly_not_read": [
                "market open/high/low/close",
                "future price return",
                "funding after signal",
                "sealed 2023 feature values",
            ],
        },
        "implementation_contract": {
            "rolling_window_bars": ROLLING_WINDOW_BARS,
            "rolling_minimum_finite_prior_bars": ROLLING_MINIMUM_BARS,
            "all_thresholds_shifted_bars": 1,
            "minimum_agg_trade_count": MINIMUM_AGG_TRADE_COUNT,
            "source_quarantine_current_plus_following_bars": (
                QUARANTINE_FOLLOWING_BARS
            ),
            "episode_prior_active_lookback_bars": EPISODE_LOOKBACK_BARS,
            "entry_delay_bars_reserved_for_tail_bound": ENTRY_DELAY_BARS,
            "nonoverlap": "next signal row >= prior signal row + hold_bars",
        },
        "hashes": {
            "preregistration_file_sha256": sha256_file(cfg.preregistration),
            "preregistration_manifest_hash": prereg["manifest_hash"],
            "source_manifest_file_sha256": sha256_file(cfg.source_manifest),
            "source_manifest_hash": source_manifest["manifest_hash"],
        },
        "git_anchors": {
            "preregistration": git_anchor(cfg.preregistration),
            "source_freezer": git_anchor("training/export_tail_arrival_selection_prefix.py"),
            "support_builder": git_anchor(__file__),
        },
        "quality": quality,
        "support_policies": support,
        "passing_policy_ids": passing_policy_ids,
        "all_support_policies_pass": len(passing_policy_ids) == len(policy_grid()),
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
