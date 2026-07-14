"""Outcome-blind support preregistration for CATCH-12.

CATCH uses completed-bar Spot-to-perpetual minute ordering only.  This module
may inspect causal feature incidence and clock overlap, but it contains no
post-entry return, future OHLC, CAGR, or MDD calculation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.build_binance_cross_venue_minute_leadership import OUTPUT_COLUMNS


SELECTION_END = pd.Timestamp("2024-01-01")
SUPPORT_QUANTILES = (0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975)
EXPECTED_FEATURE_SHA256 = (
    "00ab6a55fc7bfeb3012584db5bc97a7d7b98dd995491acfd3f865c6bd41f92cc"
)
EXPECTED_AUDIT_SHA256 = (
    "ffe0124ac9c5c0c3f1d1c284b672618cf910dc16cae36e65c1efe79710f039af"
)
CONTROL_SIDE_RULES = {
    "primary": "sign(spot_flow_fraction)",
    "direction_flip": "negative sign(spot_flow_fraction) on the primary clock",
    "venue_swap": "sign(um_flow_fraction)",
    "reverse_time": "sign(spot_flow_fraction)",
    "simultaneous_only": "sign(spot_flow_fraction)",
    "aggregate_only": "sign(spot_flow_fraction)",
    "basis_only": "sign(spot_flow_fraction)",
    "asymmetry_only": "sign(spot_flow_fraction)",
    "no_basis_lag": "sign(spot_flow_fraction)",
    "no_activity_order": "sign(spot_flow_fraction)",
    "stale_1h": "primary side stale by 12 completed bars",
    "stale_24h": "primary side stale by 288 completed bars",
    "signal_delay_1bar": "primary side delayed by one completed bar",
}
SCHEDULE_COLUMNS = (
    "signal_position",
    "entry_position",
    "exit_position",
    "signal_date",
    "entry_date",
    "exit_date",
    "side",
    "branch",
    "hold_bars",
)


@dataclass(frozen=True)
class Config:
    features: str = (
        "data/binance_cross_venue_minute_leadership_btc_2020_2023/"
        "BTCUSDT_cross_venue_minute_leadership_5m_2020-01_2023-12.csv.gz"
    )
    feature_manifest: str = (
        "data/binance_cross_venue_minute_leadership_btc_2020_2023/"
        "build_manifest.json"
    )
    source_audit: str = (
        "results/binance_cross_venue_minute_leadership_audit_2026-07-14.json"
    )
    cspr_clock: str = "results/cash_sponsored_perp_rejection_clock_2026-07-14.csv"
    cspr_clock_manifest: str = (
        "results/cash_sponsored_perp_rejection_clock_manifest_2026-07-14.json"
    )
    rift_clock: str = "results/refill_inference_flow_topology_clock_2026-07-14.csv"
    rift_clock_manifest: str = (
        "results/refill_inference_flow_topology_clock_manifest_2026-07-14.json"
    )
    output: str = "results/cash_auction_transfer_catchup_handoff_support_2026-07-14.json"
    baseline_bars: int = 8_640
    baseline_min_periods: int = 2_016
    post_gap_quarantine_bars: int = 24
    hold_bars: int = 12
    minimum_nonoverlap_total: int = 2_500
    minimum_nonoverlap_per_year: int = 500
    minimum_nonoverlap_per_2023_half: int = 200
    minimum_nonoverlap_per_2023_quarter: int = 100
    minimum_side_share: float = 0.35
    minimum_side_events_per_year: int = 150
    minimum_active_months: int = 42
    minimum_events_per_active_month: int = 20
    maximum_reverse_jaccard: float = 0.15
    maximum_reverse_containment: float = 0.25
    maximum_venue_swap_jaccard: float = 0.05
    maximum_venue_swap_containment: float = 0.10
    maximum_simultaneous_jaccard: float = 0.10
    maximum_simultaneous_containment: float = 0.10
    maximum_aggregate_jaccard: float = 0.10
    maximum_aggregate_containment: float = 0.15
    maximum_single_mechanism_jaccard: float = 0.15
    maximum_single_mechanism_containment: float = 0.20
    maximum_no_basis_retention: float = 0.65
    maximum_no_activity_jaccard: float = 0.40
    maximum_no_activity_containment: float = 0.40
    maximum_stale_jaccard: float = 0.05
    maximum_stale_containment: float = 0.10
    maximum_prior_clock_jaccard: float = 0.01
    maximum_prior_clock_containment: float = 0.01


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _quarantine_mask(source_available: pd.Series, bars: int) -> pd.Series:
    if bars < 0:
        raise ValueError("post-gap quarantine bars cannot be negative")
    return (
        (~source_available.astype(bool))
        .astype(np.int8)
        .rolling(bars + 1, min_periods=1)
        .max()
        .astype(bool)
    )


def load_causal_frame(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_path = Path(cfg.features)
    manifest_path = Path(cfg.feature_manifest)
    audit_path = Path(cfg.source_audit)
    manifest = _read_json(manifest_path)
    audit = _read_json(audit_path)
    feature_sha = _sha256(feature_path)
    audit_sha = _sha256(audit_path)
    if feature_sha != EXPECTED_FEATURE_SHA256:
        raise ValueError("CATCH feature source differs from its frozen SHA-256")
    if audit_sha != EXPECTED_AUDIT_SHA256:
        raise ValueError("CATCH source audit differs from its frozen SHA-256")
    if feature_sha != manifest.get("combined_sha256"):
        raise ValueError("cross-venue feature hash does not match manifest")
    if audit.get("passed") is not True or audit.get("failed_checks") != []:
        raise ValueError("cross-venue source audit did not pass cleanly")
    if (
        audit.get("manifest_diagnostics", {}).get("combined_sha256")
        != feature_sha
    ):
        raise ValueError("source audit and feature hash disagree")

    frame = pd.read_csv(
        feature_path,
        compression="gzip",
        parse_dates=["date", "feature_available_time_utc", "trade_earliest_time_utc"],
    )
    if tuple(frame.columns) != OUTPUT_COLUMNS:
        raise ValueError("cross-venue feature columns differ from frozen schema")
    if frame["date"].max() >= SELECTION_END:
        raise ValueError("CATCH support frame contains 2024+ rows")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError("CATCH timestamps are duplicate or unordered")
    expected = pd.date_range(
        frame["date"].min(), SELECTION_END, inclusive="left", freq="5min"
    )
    if not frame["date"].equals(pd.Series(expected, name="date")):
        raise ValueError("CATCH source must be a complete five-minute clock")
    if not frame["feature_available_time_utc"].equals(
        frame["date"] + pd.Timedelta("5min")
    ) or not frame["trade_earliest_time_utc"].equals(
        frame["feature_available_time_utc"]
    ):
        raise ValueError("CATCH source availability contract is inconsistent")
    available = (
        frame["source_complete"].astype(bool)
        & frame["cross_venue_feature_valid"].astype(bool)
        & frame["feature_invalid_reason"].eq("ok")
    )
    frame["quarantined"] = _quarantine_mask(
        available, cfg.post_gap_quarantine_bars
    )
    source = {
        "feature_sha256": feature_sha,
        "feature_manifest_sha256": _sha256(manifest_path),
        "source_audit_sha256": audit_sha,
        "source_available_rows": int(available.sum()),
        "quarantined_rows": int(frame["quarantined"].sum()),
        "first_date": str(frame["date"].min()),
        "last_date": str(frame["date"].max()),
    }
    return frame, source


def prior_quantile(
    values: pd.Series,
    clean: pd.Series,
    *,
    quantile: float,
    cfg: Config,
) -> pd.Series:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if not 1 <= cfg.baseline_min_periods <= cfg.baseline_bars:
        raise ValueError("baseline periods are invalid")
    return (
        pd.to_numeric(values, errors="coerce")
        .where(clean.astype(bool))
        .shift(1)
        .rolling(cfg.baseline_bars, min_periods=cfg.baseline_min_periods)
        .quantile(quantile)
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _geometric_mean(*components: pd.Series) -> pd.Series:
    if not components:
        raise ValueError("geometric mean requires at least one component")
    values = np.vstack(
        [component.clip(lower=0.0).fillna(0.0).to_numpy(float) for component in components]
    )
    return pd.Series(
        np.prod(values, axis=0) ** (1.0 / len(components)),
        index=components[0].index,
        dtype=float,
    )


def _components(frame: pd.DataFrame) -> dict[str, pd.Series]:
    values = {
        column: _numeric(frame, column)
        for column in (
            "spot_flow_fraction",
            "um_flow_fraction",
            "spot_flow_coherence",
            "um_flow_coherence",
            "spot_log_return_5m",
            "um_log_return_5m",
            "basis_change_bp",
            "um_minus_spot_activity_time_centroid",
            "spot_to_um_lagged_directional_alignment",
            "um_to_spot_lagged_directional_alignment",
            "lagged_directional_alignment_diff",
            "reverse_spot_to_um_lagged_directional_alignment",
            "reverse_lagged_directional_alignment_diff",
            "simultaneous_flow_sign_agreement",
            "simultaneous_return_sign_agreement",
            "flow_transfer_asymmetry",
            "return_leadership_asymmetry",
        )
    }
    values["spot_side"] = pd.Series(
        np.sign(values["spot_flow_fraction"]), index=frame.index, dtype=float
    )
    values["um_side"] = pd.Series(
        np.sign(values["um_flow_fraction"]), index=frame.index, dtype=float
    )
    values["spot_residual_basis_bp"] = (
        -values["spot_side"] * values["basis_change_bp"]
    )
    values["um_residual_basis_bp"] = values["um_side"] * values["basis_change_bp"]
    values["directed_handoff"] = pd.concat(
        [
            values["spot_to_um_lagged_directional_alignment"].clip(lower=0.0),
            values["lagged_directional_alignment_diff"].clip(lower=0.0),
        ],
        axis=1,
    ).min(axis=1)
    values["reverse_directed_handoff"] = pd.concat(
        [
            values[
                "reverse_spot_to_um_lagged_directional_alignment"
            ].clip(lower=0.0),
            values["reverse_lagged_directional_alignment_diff"].clip(lower=0.0),
        ],
        axis=1,
    ).min(axis=1)
    values["venue_swap_handoff"] = pd.concat(
        [
            values["um_to_spot_lagged_directional_alignment"].clip(lower=0.0),
            (-values["lagged_directional_alignment_diff"]).clip(lower=0.0),
        ],
        axis=1,
    ).min(axis=1)
    values["simultaneous_block"] = pd.concat(
        [
            values["simultaneous_flow_sign_agreement"].clip(lower=0.0),
            values["simultaneous_return_sign_agreement"].clip(lower=0.0),
        ],
        axis=1,
    ).min(axis=1)
    return values


def classify_events(
    frame: pd.DataFrame,
    cfg: Config,
    *,
    quantile: float,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.Series],
    dict[str, pd.Series],
    dict[str, pd.Series],
]:
    values = _components(frame)
    clean = ~frame["quarantined"].astype(bool)
    spot_side = values["spot_side"]
    um_side = values["um_side"]
    timing = values["um_minus_spot_activity_time_centroid"]
    cash_accepted = spot_side.ne(0.0) & (
        spot_side * values["spot_log_return_5m"]
    ).gt(0.0)
    basis_lag = values["spot_residual_basis_bp"].gt(0.0)
    common = clean & cash_accepted & basis_lag

    scores = {
        "primary": _geometric_mean(
            values["directed_handoff"],
            timing,
            values["spot_flow_coherence"],
        ),
        "reverse_time": _geometric_mean(
            values["reverse_directed_handoff"],
            timing,
            values["spot_flow_coherence"],
        ),
        "venue_swap": _geometric_mean(
            values["venue_swap_handoff"],
            -timing,
            values["um_flow_coherence"],
        ),
        "simultaneous_only": _geometric_mean(
            values["simultaneous_block"],
            values["spot_flow_coherence"],
            values["spot_flow_fraction"].abs(),
        ),
        "aggregate_only": _geometric_mean(
            values["spot_flow_coherence"], values["spot_flow_fraction"].abs()
        ),
        "basis_only": values["spot_residual_basis_bp"].clip(lower=0.0),
        "asymmetry_only": _geometric_mean(
            values["flow_transfer_asymmetry"],
            values["return_leadership_asymmetry"],
        ),
        "no_basis_lag": _geometric_mean(
            values["directed_handoff"], timing, values["spot_flow_coherence"]
        ),
        "no_activity_order": _geometric_mean(
            values["directed_handoff"], values["spot_flow_coherence"]
        ),
    }
    scores["stale_1h"] = scores["primary"].shift(12)
    scores["stale_24h"] = scores["primary"].shift(288)
    thresholds = {
        name: prior_quantile(score, clean, quantile=quantile, cfg=cfg)
        for name, score in scores.items()
    }
    primary_base = common & values["directed_handoff"].gt(0.0) & timing.gt(0.0)
    base_masks = {
        "primary": primary_base,
        "reverse_time": common
        & values["reverse_directed_handoff"].gt(0.0)
        & timing.gt(0.0),
        "venue_swap": (
            clean
            & um_side.ne(0.0)
            & (um_side * values["um_log_return_5m"]).gt(0.0)
            & values["um_residual_basis_bp"].gt(0.0)
            & values["venue_swap_handoff"].gt(0.0)
            & timing.lt(0.0)
        ),
        "simultaneous_only": common & values["simultaneous_block"].gt(0.0),
        "aggregate_only": common,
        "basis_only": clean & cash_accepted & basis_lag,
        "asymmetry_only": common
        & values["flow_transfer_asymmetry"].gt(0.0)
        & values["return_leadership_asymmetry"].gt(0.0),
        "no_basis_lag": clean
        & cash_accepted
        & values["directed_handoff"].gt(0.0)
        & timing.gt(0.0),
        "no_activity_order": common & values["directed_handoff"].gt(0.0),
        "stale_1h": primary_base.shift(12, fill_value=False) & clean,
        "stale_24h": primary_base.shift(288, fill_value=False) & clean,
    }
    scored_masks = {
        name: base_masks[name] & scores[name].ge(thresholds[name])
        for name in scores
    }
    primary = scored_masks["primary"].fillna(False)
    controls = {
        "primary": primary,
        "direction_flip": primary,
        **{
            name: mask.fillna(False)
            for name, mask in scored_masks.items()
            if name != "primary"
        },
        "signal_delay_1bar": primary.shift(1, fill_value=False) & clean,
    }
    control_sides = {
        "primary": spot_side,
        "direction_flip": -spot_side,
        "venue_swap": um_side,
        "reverse_time": spot_side,
        "simultaneous_only": spot_side,
        "aggregate_only": spot_side,
        "basis_only": spot_side,
        "asymmetry_only": spot_side,
        "no_basis_lag": spot_side,
        "no_activity_order": spot_side,
        "stale_1h": spot_side.shift(12).fillna(0.0),
        "stale_24h": spot_side.shift(288).fillna(0.0),
        "signal_delay_1bar": spot_side.shift(1).fillna(0.0),
    }
    signal = pd.DataFrame(
        {
            "date": frame["date"],
            "side": np.where(primary, spot_side, 0).astype(np.int8),
            "branch": np.where(primary, "catch12", "none"),
            "hold_bars": np.where(primary, cfg.hold_bars, 0).astype(np.int16),
            "score": scores["primary"],
            "threshold": thresholds["primary"],
            "directed_handoff": values["directed_handoff"],
            "quarantined": frame["quarantined"].astype(bool),
        }
    )
    diagnostics = {
        **{f"score_{name}": score for name, score in scores.items()},
        **{f"threshold_{name}": threshold for name, threshold in thresholds.items()},
        **{f"base_{name}": mask for name, mask in base_masks.items()},
    }
    return signal, controls, control_sides, diagnostics


def nonoverlapping_schedule(
    signal: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp = SELECTION_END,
) -> pd.DataFrame:
    """Reserve next-open fixed holds without consulting future source health."""
    start_time = frame["date"].min() if start is None else pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    if start_time >= end_time:
        raise ValueError("schedule start must precede end")
    period = frame["date"].ge(start_time) & frame["date"].lt(end_time)
    side = pd.to_numeric(signal["side"], errors="coerce").fillna(0).to_numpy(np.int8)
    hold = (
        pd.to_numeric(signal["hold_bars"], errors="coerce")
        .fillna(0)
        .to_numpy(np.int16)
    )
    branch = signal["branch"].astype(str).to_numpy()
    quarantined = frame["quarantined"].astype(bool).to_numpy()
    rows: list[dict[str, Any]] = []
    next_entry = 0
    for signal_position in np.flatnonzero(side):
        entry_position = int(signal_position + 1)
        exit_position = entry_position + int(hold[signal_position])
        if entry_position < next_entry or exit_position >= len(frame):
            continue
        if (
            not period.iloc[signal_position]
            or not period.iloc[entry_position]
            or not period.iloc[exit_position]
            or quarantined[signal_position]
        ):
            continue
        rows.append(
            {
                "signal_position": int(signal_position),
                "entry_position": entry_position,
                "exit_position": exit_position,
                "signal_date": str(frame["date"].iloc[signal_position]),
                "entry_date": str(frame["date"].iloc[entry_position]),
                "exit_date": str(frame["date"].iloc[exit_position]),
                "side": int(side[signal_position]),
                "branch": str(branch[signal_position]),
                "hold_bars": int(hold[signal_position]),
            }
        )
        next_entry = exit_position
    return pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)


def _control_signal(
    mask: pd.Series,
    side: pd.Series,
    cfg: Config,
    *,
    branch: str,
) -> pd.DataFrame:
    active = mask.fillna(False).astype(bool)
    return pd.DataFrame(
        {
            "side": np.where(active, side, 0).astype(np.int8),
            "branch": np.where(active, branch, "none"),
            "hold_bars": np.where(active, cfg.hold_bars, 0).astype(np.int16),
        }
    )


def _support(schedule: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    dates = (
        pd.to_datetime(schedule["entry_date"])
        if not schedule.empty
        else pd.Series([], dtype="datetime64[ns]")
    )
    by_year = {
        str(year): int(dates.dt.year.eq(year).sum())
        for year in (2020, 2021, 2022, 2023)
    }
    by_year_side = {
        str(year): {
            "long": int(
                (dates.dt.year.eq(year) & schedule["side"].gt(0).to_numpy()).sum()
            ),
            "short": int(
                (dates.dt.year.eq(year) & schedule["side"].lt(0).to_numpy()).sum()
            ),
        }
        for year in (2020, 2021, 2022, 2023)
    }
    h1 = int(((dates >= "2023-01-01") & (dates < "2023-07-01")).sum())
    h2 = int(((dates >= "2023-07-01") & (dates < "2024-01-01")).sum())
    quarters = {
        f"2023Q{quarter}": int(
            dates.dt.to_period("Q").eq(pd.Period(f"2023Q{quarter}")).sum()
        )
        for quarter in range(1, 5)
    }
    month_counts = dates.dt.to_period("M").value_counts()
    active_months = int((month_counts >= cfg.minimum_events_per_active_month).sum())
    total = int(len(schedule))
    long_share = float(schedule["side"].gt(0).mean()) if total else 0.0
    short_share = float(schedule["side"].lt(0).mean()) if total else 0.0
    passes = (
        total >= cfg.minimum_nonoverlap_total
        and all(count >= cfg.minimum_nonoverlap_per_year for count in by_year.values())
        and h1 >= cfg.minimum_nonoverlap_per_2023_half
        and h2 >= cfg.minimum_nonoverlap_per_2023_half
        and all(
            count >= cfg.minimum_nonoverlap_per_2023_quarter
            for count in quarters.values()
        )
        and min(long_share, short_share) >= cfg.minimum_side_share
        and all(
            min(counts.values()) >= cfg.minimum_side_events_per_year
            for counts in by_year_side.values()
        )
        and active_months >= cfg.minimum_active_months
    )
    return {
        "nonoverlap_total": total,
        "by_year": by_year,
        "by_year_side": by_year_side,
        "2023_h1": h1,
        "2023_h2": h2,
        "2023_quarters": quarters,
        "long_share": long_share,
        "short_share": short_share,
        "active_months": active_months,
        "minimum_events_per_active_month": cfg.minimum_events_per_active_month,
        "passes_count_support": bool(passes),
    }


def _overlap(left: pd.Series, right: pd.Series) -> dict[str, float | int]:
    left = left.fillna(False).astype(bool)
    right = right.fillna(False).astype(bool)
    intersection = int((left & right).sum())
    union = int((left | right).sum())
    primary = int(left.sum())
    return {
        "intersection": intersection,
        "union": union,
        "jaccard": float(intersection / union) if union else 0.0,
        "primary_containment": float(intersection / primary) if primary else 0.0,
    }


def _load_frozen_clock_mask(
    frame: pd.DataFrame,
    *,
    clock_path: str,
    manifest_path: str,
    label: str,
) -> tuple[pd.Series, dict[str, Any]]:
    clock_file = Path(clock_path)
    manifest_file = Path(manifest_path)
    manifest = _read_json(manifest_file)
    if manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError(f"{label} clock is not outcome-blind")
    if _sha256(clock_file) != manifest.get("clock", {}).get("sha256"):
        raise ValueError(f"{label} clock hash does not match manifest")
    clock = pd.read_csv(clock_file, usecols=["signal_date"])
    dates = pd.to_datetime(clock["signal_date"], errors="raise")
    if dates.max() >= SELECTION_END:
        raise ValueError(f"{label} clock contains 2024+ rows")
    mask = frame["date"].isin(pd.DatetimeIndex(dates))
    return mask, {
        "clock_sha256": _sha256(clock_file),
        "manifest_sha256": _sha256(manifest_file),
        "rows": int(len(clock)),
    }


def _schedule_mask(schedule: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
    if schedule.empty:
        return pd.Series(False, index=frame.index)
    return frame["date"].isin(pd.to_datetime(schedule["signal_date"]))


def run_support(cfg: Config) -> dict[str, Any]:
    frame, source = load_causal_frame(cfg)
    cspr_mask, cspr_source = _load_frozen_clock_mask(
        frame,
        clock_path=cfg.cspr_clock,
        manifest_path=cfg.cspr_clock_manifest,
        label="CSPR",
    )
    rift_mask, rift_source = _load_frozen_clock_mask(
        frame,
        clock_path=cfg.rift_clock,
        manifest_path=cfg.rift_clock_manifest,
        label="RIFT",
    )
    prior_union = cspr_mask | rift_mask
    rows: list[dict[str, Any]] = []
    for quantile in SUPPORT_QUANTILES:
        signal, controls, control_sides, _ = classify_events(
            frame, cfg, quantile=quantile
        )
        schedule = nonoverlapping_schedule(signal, frame)
        support = _support(schedule, cfg)
        primary = controls["primary"]
        raw_overlap = {
            name: _overlap(primary, mask)
            for name, mask in controls.items()
            if name not in {"primary", "direction_flip"}
        }
        no_basis_count = int(controls["no_basis_lag"].sum())
        no_basis_retention = float(int(primary.sum()) / max(1, no_basis_count))
        scheduled_primary = _schedule_mask(schedule, frame)
        control_schedules = {
            name: nonoverlapping_schedule(
                _control_signal(
                    mask,
                    control_sides[name],
                    cfg,
                    branch=f"control_{name}",
                ),
                frame,
            )
            for name, mask in controls.items()
            if name not in {"primary", "direction_flip"}
        }
        scheduled_no_basis_retention = float(
            len(schedule) / max(1, len(control_schedules["no_basis_lag"]))
        )
        scheduled_overlap = {
            name: _overlap(
                scheduled_primary, _schedule_mask(control_schedule, frame)
            )
            for name, control_schedule in control_schedules.items()
        }
        prior_overlap = {
            "cspr": _overlap(scheduled_primary, cspr_mask),
            "rift": _overlap(scheduled_primary, rift_mask),
            "cspr_or_rift": _overlap(scheduled_primary, prior_union),
        }
        def both_clocks_pass(name: str, jaccard: float, containment: float) -> bool:
            return all(
                overlap[name]["jaccard"] <= jaccard
                and overlap[name]["primary_containment"] <= containment
                for overlap in (raw_overlap, scheduled_overlap)
            )

        novelty = (
            both_clocks_pass(
                "reverse_time",
                cfg.maximum_reverse_jaccard,
                cfg.maximum_reverse_containment,
            )
            and both_clocks_pass(
                "venue_swap",
                cfg.maximum_venue_swap_jaccard,
                cfg.maximum_venue_swap_containment,
            )
            and both_clocks_pass(
                "simultaneous_only",
                cfg.maximum_simultaneous_jaccard,
                cfg.maximum_simultaneous_containment,
            )
            and both_clocks_pass(
                "aggregate_only",
                cfg.maximum_aggregate_jaccard,
                cfg.maximum_aggregate_containment,
            )
            and all(
                both_clocks_pass(
                    name,
                    cfg.maximum_single_mechanism_jaccard,
                    cfg.maximum_single_mechanism_containment,
                )
                for name in ("basis_only", "asymmetry_only")
            )
            and no_basis_retention <= cfg.maximum_no_basis_retention
            and scheduled_no_basis_retention <= cfg.maximum_no_basis_retention
            and both_clocks_pass(
                "no_activity_order",
                cfg.maximum_no_activity_jaccard,
                cfg.maximum_no_activity_containment,
            )
            and all(
                both_clocks_pass(
                    name,
                    cfg.maximum_stale_jaccard,
                    cfg.maximum_stale_containment,
                )
                for name in ("stale_1h", "stale_24h", "signal_delay_1bar")
            )
            and all(
                overlap["jaccard"] <= cfg.maximum_prior_clock_jaccard
                and overlap["primary_containment"]
                <= cfg.maximum_prior_clock_containment
                for overlap in prior_overlap.values()
            )
        )
        rows.append(
            {
                "quantile": quantile,
                "raw_primary": int(primary.sum()),
                "support": support,
                "control_raw_counts": {
                    name: int(mask.sum())
                    for name, mask in controls.items()
                    if name != "primary"
                },
                "control_overlap": raw_overlap,
                "control_scheduled_counts": {
                    name: int(len(control_schedule))
                    for name, control_schedule in control_schedules.items()
                },
                "control_scheduled_overlap": scheduled_overlap,
                "no_basis_retention": {
                    "raw": no_basis_retention,
                    "scheduled": scheduled_no_basis_retention,
                },
                "prior_clock_overlap": prior_overlap,
                "passes_novelty": bool(novelty),
                "passes_support": bool(
                    support["passes_count_support"] and novelty
                ),
            }
        )
    passing = [row for row in rows if row["passes_support"]]
    selected = max(passing, key=lambda row: row["quantile"]) if passing else None
    return {
        "protocol": {
            "name": "CATCH-12 — Cash Auction Transfer & Catch-up Handoff",
            "support_only": True,
            "outcomes_opened": False,
            "selection_end_exclusive": str(SELECTION_END),
            "economic_claim": (
                "completed-bar cash-flow discovery followed by an observable, "
                "not necessarily causal, perpetual catch-up handoff"
            ),
            "entry": "next USD-M 5m open after the completed signal bar",
            "hold": f"fixed {cfg.hold_bars} bars",
            "support_selection": (
                "highest tested strictly-prior score quantile passing every "
                "count, era, side-balance, and novelty floor"
            ),
            "control_side_rules": CONTROL_SIDE_RULES,
        },
        "config": asdict(cfg),
        "source": {
            **source,
            "cspr_clock": cspr_source,
            "rift_clock": rift_source,
        },
        "support_grid": rows,
        "support_decision": "pass" if selected else "reject_before_returns",
        "selected_quantile": selected["quantile"] if selected else None,
        "selected_support": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=Config.output)
    cfg = Config(output=parser.parse_args().output)
    result = run_support(cfg)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "outcomes_opened": result["protocol"]["outcomes_opened"],
                "support_decision": result["support_decision"],
                "selected_quantile": result["selected_quantile"],
                "support_grid": [
                    {
                        "quantile": row["quantile"],
                        "raw_primary": row["raw_primary"],
                        **row["support"],
                        "passes_novelty": row["passes_novelty"],
                        "passes_support": row["passes_support"],
                    }
                    for row in result["support_grid"]
                ],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
