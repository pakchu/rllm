"""Outcome-blind support preregistration for LURI-48.

LURI infers a three-hour USD-M inventory imbalance from completed bars, then
requires a completed USD-M-led release before next-open execution. This module
may inspect incidence and clock overlap, but it contains no future return,
post-entry OHLC, CAGR, MDD, or trade-profit calculation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_cash_auction_transfer_catchup_handoff as cross


SELECTION_END = pd.Timestamp("2024-01-01")
BASIS_QUANTILES = (0.25, 0.40, 0.55, 0.70)
CONTROL_SIDE_RULES = {
    "primary": "negative prior 36-bar USD-M inferred-inventory side",
    "direction_flip": "prior 36-bar USD-M inferred-inventory side",
    "no_inventory": "current USD-M flow side",
    "no_basis_history": "negative prior USD-M side",
    "no_cash_refusal": "negative prior USD-M side",
    "spot_confirmed": "negative prior USD-M side",
    "basis_only": "negative prior USD-M side",
    "reverse_time": "negative prior USD-M side",
    "simultaneous_only": "negative prior USD-M side",
    "spot_inventory_swap": "negative prior Spot inferred-inventory side",
    "stale_24h": "primary side stale by 288 completed bars",
    "signal_delay_1bar": "primary side delayed by one completed bar",
}
SCORE_BEARING_CONTROLS = (
    "no_inventory",
    "no_basis_history",
    "no_cash_refusal",
    "spot_confirmed",
    "basis_only",
    "reverse_time",
    "simultaneous_only",
    "spot_inventory_swap",
    "stale_24h",
)
SCHEDULE_COLUMNS = cross.SCHEDULE_COLUMNS


@dataclass(frozen=True)
class Config:
    output: str = (
        "results/leveraged_um_inventory_release_handoff_support_2026-07-14.json"
    )
    cspr_clock: str = "results/cash_sponsored_perp_rejection_clock_2026-07-14.csv"
    cspr_clock_manifest: str = (
        "results/cash_sponsored_perp_rejection_clock_manifest_2026-07-14.json"
    )
    rift_clock: str = "results/refill_inference_flow_topology_clock_2026-07-14.csv"
    rift_clock_manifest: str = (
        "results/refill_inference_flow_topology_clock_manifest_2026-07-14.json"
    )
    catch_clock: str = (
        "results/cash_auction_transfer_catchup_handoff_clock_2026-07-14.csv"
    )
    catch_clock_manifest: str = (
        "results/cash_auction_transfer_catchup_handoff_clock_manifest_2026-07-14.json"
    )
    formation_bars: int = 36
    hold_bars: int = 48
    baseline_bars: int = 8_640
    baseline_min_periods: int = 2_016
    minimum_nonoverlap_total: int = 420
    minimum_nonoverlap_per_year: int = 95
    minimum_nonoverlap_per_2023_half: int = 45
    minimum_nonoverlap_per_2023_quarter: int = 20
    minimum_side_share: float = 0.30
    minimum_side_events_per_year: int = 35
    minimum_active_months: int = 42
    minimum_events_per_active_month: int = 5
    maximum_no_inventory_retention: float = 0.10
    maximum_no_basis_retention: float = 0.55
    maximum_no_cash_retention: float = 0.20
    maximum_basis_only_retention: float = 0.30
    maximum_placebo_jaccard: float = 0.05
    maximum_placebo_containment: float = 0.10
    maximum_simultaneous_jaccard: float = 0.15
    maximum_simultaneous_raw_containment: float = 0.80
    maximum_simultaneous_scheduled_containment: float = 0.60
    maximum_stale_jaccard: float = 0.01
    maximum_stale_containment: float = 0.02
    maximum_prior_clock_jaccard: float = 0.01
    maximum_prior_clock_containment: float = 0.01
    maximum_catch_venue_swap_jaccard: float = 0.02
    maximum_catch_venue_swap_containment: float = 0.20


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_causal_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, source = cross.load_causal_frame(cross.Config())
    if frame["date"].max() >= SELECTION_END:
        raise ValueError("LURI support frame contains 2024+ rows")
    forbidden = {"open", "high", "low", "close", "future_return", "profit"}
    if forbidden.intersection(frame.columns):
        raise ValueError("LURI support frame unexpectedly contains outcome columns")
    return frame, source


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _history_clean(clean: pd.Series, bars: int) -> pd.Series:
    if bars < 1:
        raise ValueError("LURI formation bars must be positive")
    return (
        clean.shift(1, fill_value=False)
        .rolling(bars, min_periods=bars)
        .min()
        .fillna(0.0)
        .astype(bool)
    )


def _trailing_flow_fraction(frame: pd.DataFrame, venue: str, bars: int) -> pd.Series:
    signed = (
        _numeric(frame, f"{venue}_signed_quote_notional")
        .shift(1)
        .rolling(bars, min_periods=bars)
        .sum()
    )
    total = (
        _numeric(frame, f"{venue}_quote_notional")
        .shift(1)
        .rolling(bars, min_periods=bars)
        .sum()
    )
    return signed / total.where(total.gt(0.0))


def _trailing_return(frame: pd.DataFrame, venue: str, bars: int) -> pd.Series:
    return (
        _numeric(frame, f"{venue}_log_return_5m")
        .shift(1)
        .rolling(bars, min_periods=bars)
        .sum()
    )


def prior_quantile(
    values: pd.Series,
    clean: pd.Series,
    *,
    quantile: float,
    cfg: Config,
) -> pd.Series:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("LURI basis quantile must be in [0, 1]")
    if not 1 <= cfg.baseline_min_periods <= cfg.baseline_bars:
        raise ValueError("LURI baseline periods are invalid")
    return (
        pd.to_numeric(values, errors="coerce")
        .where(clean.astype(bool))
        .shift(1)
        .rolling(cfg.baseline_bars, min_periods=cfg.baseline_min_periods)
        .quantile(quantile)
    )


def _minimum_positive(left: pd.Series, right: pd.Series) -> pd.Series:
    return pd.concat([left.clip(lower=0.0), right.clip(lower=0.0)], axis=1).min(axis=1)


def classify_events(
    frame: pd.DataFrame,
    cfg: Config,
    *,
    basis_quantile: float,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.Series],
    dict[str, pd.Series],
    dict[str, pd.Series],
]:
    clean = ~frame["quarantined"].astype(bool)
    history_clean = _history_clean(clean, cfg.formation_bars)
    um_flow = _trailing_flow_fraction(frame, "um", cfg.formation_bars)
    spot_flow = _trailing_flow_fraction(frame, "spot", cfg.formation_bars)
    um_return = _trailing_return(frame, "um", cfg.formation_bars)
    spot_return = _trailing_return(frame, "spot", cfg.formation_bars)
    inventory_side = pd.Series(np.sign(um_flow), index=frame.index, dtype=float)
    release_side = -inventory_side
    spot_inventory_side = pd.Series(np.sign(spot_flow), index=frame.index, dtype=float)
    spot_release_side = -spot_inventory_side

    basis_delta = _numeric(frame, "close_basis_bp").shift(1) - _numeric(
        frame, "open_basis_bp"
    ).shift(cfg.formation_bars)
    basis_displacement = inventory_side * basis_delta
    spot_basis_displacement = -spot_inventory_side * basis_delta
    threshold_clean = clean & history_clean
    basis_threshold = prior_quantile(
        basis_displacement.where(basis_displacement.gt(0.0)),
        threshold_clean,
        quantile=basis_quantile,
        cfg=cfg,
    )
    spot_basis_threshold = prior_quantile(
        spot_basis_displacement.where(spot_basis_displacement.gt(0.0)),
        threshold_clean,
        quantile=basis_quantile,
        cfg=cfg,
    )

    forward_handoff = _minimum_positive(
        _numeric(frame, "um_to_spot_lagged_directional_alignment"),
        -_numeric(frame, "lagged_directional_alignment_diff"),
    )
    reverse_handoff = _minimum_positive(
        _numeric(frame, "reverse_um_to_spot_lagged_directional_alignment"),
        -_numeric(frame, "reverse_lagged_directional_alignment_diff"),
    )
    spot_forward_handoff = _minimum_positive(
        _numeric(frame, "spot_to_um_lagged_directional_alignment"),
        _numeric(frame, "lagged_directional_alignment_diff"),
    )
    spot_reverse_handoff = _minimum_positive(
        _numeric(frame, "reverse_spot_to_um_lagged_directional_alignment"),
        _numeric(frame, "reverse_lagged_directional_alignment_diff"),
    )
    um_first = -_numeric(frame, "um_minus_spot_activity_time_centroid")

    formation = (
        history_clean
        & inventory_side.ne(0.0)
        & basis_displacement.gt(0.0)
        & basis_displacement.ge(basis_threshold)
        & (-inventory_side * spot_flow).ge(0.0)
        & (-inventory_side * spot_return).ge(0.0)
    )
    release = (
        clean
        & (release_side * _numeric(frame, "um_flow_fraction")).gt(0.0)
        & (release_side * _numeric(frame, "um_log_return_5m")).gt(0.0)
        & (release_side * _numeric(frame, "basis_change_bp")).gt(0.0)
        & um_first.gt(0.0)
        & forward_handoff.gt(0.0)
        & forward_handoff.gt(reverse_handoff)
    )
    primary = (formation & release).fillna(False)

    current_um_side = pd.Series(
        np.sign(_numeric(frame, "um_flow_fraction")),
        index=frame.index,
        dtype=float,
    )
    no_inventory = (
        clean
        & current_um_side.ne(0.0)
        & (current_um_side * _numeric(frame, "um_log_return_5m")).gt(0.0)
        & (current_um_side * _numeric(frame, "basis_change_bp")).gt(0.0)
        & um_first.gt(0.0)
        & forward_handoff.gt(0.0)
        & forward_handoff.gt(reverse_handoff)
    )
    no_basis_history = (
        history_clean
        & inventory_side.ne(0.0)
        & (-inventory_side * spot_flow).ge(0.0)
        & (-inventory_side * spot_return).ge(0.0)
        & release
    )
    no_cash_refusal = (
        history_clean
        & inventory_side.ne(0.0)
        & basis_displacement.gt(0.0)
        & basis_displacement.ge(basis_threshold)
        & release
    )
    spot_confirmed = (
        history_clean
        & inventory_side.ne(0.0)
        & basis_displacement.gt(0.0)
        & basis_displacement.ge(basis_threshold)
        & (inventory_side * spot_flow).gt(0.0)
        & (inventory_side * spot_return).gt(0.0)
        & release
    )
    basis_only = (
        formation
        & clean
        & (release_side * _numeric(frame, "um_flow_fraction")).gt(0.0)
        & (release_side * _numeric(frame, "um_log_return_5m")).gt(0.0)
        & (release_side * _numeric(frame, "basis_change_bp")).gt(0.0)
    )
    reverse_time = (
        formation
        & clean
        & (release_side * _numeric(frame, "um_flow_fraction")).gt(0.0)
        & (release_side * _numeric(frame, "um_log_return_5m")).gt(0.0)
        & (release_side * _numeric(frame, "basis_change_bp")).gt(0.0)
        & um_first.gt(0.0)
        & reverse_handoff.gt(0.0)
        & reverse_handoff.gt(forward_handoff)
    )
    simultaneous_only = (
        formation
        & clean
        & (release_side * _numeric(frame, "um_flow_fraction")).gt(0.0)
        & (release_side * _numeric(frame, "um_log_return_5m")).gt(0.0)
        & (release_side * _numeric(frame, "basis_change_bp")).gt(0.0)
        & _numeric(frame, "simultaneous_flow_sign_agreement").gt(0.0)
        & _numeric(frame, "simultaneous_return_sign_agreement").gt(0.0)
    )

    spot_formation = (
        history_clean
        & spot_inventory_side.ne(0.0)
        & spot_basis_displacement.gt(0.0)
        & spot_basis_displacement.ge(spot_basis_threshold)
        & (-spot_inventory_side * um_flow).ge(0.0)
        & (-spot_inventory_side * um_return).ge(0.0)
    )
    spot_release = (
        clean
        & (spot_release_side * _numeric(frame, "spot_flow_fraction")).gt(0.0)
        & (spot_release_side * _numeric(frame, "spot_log_return_5m")).gt(0.0)
        & (-spot_release_side * _numeric(frame, "basis_change_bp")).gt(0.0)
        & (-um_first).gt(0.0)
        & spot_forward_handoff.gt(0.0)
        & spot_forward_handoff.gt(spot_reverse_handoff)
    )
    spot_inventory_swap = spot_formation & spot_release

    controls = {
        "primary": primary,
        "direction_flip": primary,
        "no_inventory": no_inventory.fillna(False),
        "no_basis_history": no_basis_history.fillna(False),
        "no_cash_refusal": no_cash_refusal.fillna(False),
        "spot_confirmed": spot_confirmed.fillna(False),
        "basis_only": basis_only.fillna(False),
        "reverse_time": reverse_time.fillna(False),
        "simultaneous_only": simultaneous_only.fillna(False),
        "spot_inventory_swap": spot_inventory_swap.fillna(False),
        "stale_24h": primary.shift(288, fill_value=False) & clean,
        "signal_delay_1bar": primary.shift(1, fill_value=False) & clean,
    }
    control_sides = {
        "primary": release_side,
        "direction_flip": inventory_side,
        "no_inventory": current_um_side,
        "no_basis_history": release_side,
        "no_cash_refusal": release_side,
        "spot_confirmed": release_side,
        "basis_only": release_side,
        "reverse_time": release_side,
        "simultaneous_only": release_side,
        "spot_inventory_swap": spot_release_side,
        "stale_24h": release_side.shift(288).fillna(0.0),
        "signal_delay_1bar": release_side.shift(1).fillna(0.0),
    }
    signal = pd.DataFrame(
        {
            "date": frame["date"],
            "side": np.where(primary, release_side, 0).astype(np.int8),
            "branch": np.where(primary, "luri48", "none"),
            "hold_bars": np.where(primary, cfg.hold_bars, 0).astype(np.int16),
            "basis_displacement": basis_displacement,
            "basis_threshold": basis_threshold,
            "forward_handoff": forward_handoff,
            "reverse_handoff": reverse_handoff,
            "quarantined": frame["quarantined"].astype(bool),
        }
    )
    diagnostics = {
        "history_clean": history_clean,
        "um_trailing_flow": um_flow,
        "spot_trailing_flow": spot_flow,
        "spot_trailing_return": spot_return,
        "basis_displacement": basis_displacement,
        "basis_threshold": basis_threshold,
        "forward_handoff": forward_handoff,
        "reverse_handoff": reverse_handoff,
        "formation": formation.fillna(False),
        "release": release.fillna(False),
    }
    return signal, controls, control_sides, diagnostics


def nonoverlapping_schedule(signal: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    return cross.nonoverlapping_schedule(signal, frame, end=SELECTION_END)


def _control_signal(
    mask: pd.Series,
    side: pd.Series,
    cfg: Config,
    *,
    branch: str,
) -> pd.DataFrame:
    active = mask.fillna(False).astype(bool)
    numeric_side = pd.to_numeric(side, errors="coerce").fillna(0.0)
    if not numeric_side.loc[active].isin((-1.0, 1.0)).all():
        raise ValueError(f"LURI control {branch} has invalid active side")
    return pd.DataFrame(
        {
            "side": np.where(active, numeric_side, 0).astype(np.int8),
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


def _retention(primary_count: int, control_count: int) -> float:
    return float(primary_count / max(1, control_count))


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
    protocol = manifest.get("protocol", {})
    if protocol.get("outcomes_opened") is not False:
        raise ValueError(f"{label} clock is not outcome-blind")
    if cross._sha256(clock_file) != manifest.get("clock", {}).get("sha256"):
        raise ValueError(f"{label} clock hash does not match manifest")
    clock = pd.read_csv(clock_file, usecols=["signal_date"])
    dates = pd.to_datetime(clock["signal_date"], errors="raise")
    if dates.max() >= SELECTION_END:
        raise ValueError(f"{label} clock contains 2024+ rows")
    return frame["date"].isin(pd.DatetimeIndex(dates)), {
        "clock_sha256": cross._sha256(clock_file),
        "manifest_sha256": cross._sha256(manifest_file),
        "rows": int(len(clock)),
    }


def _schedule_mask(schedule: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
    if schedule.empty:
        return pd.Series(False, index=frame.index)
    return frame["date"].isin(pd.to_datetime(schedule["signal_date"]))


def _catch_venue_swap_reference(
    frame: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    catch_cfg = cross.Config()
    _, controls, control_sides, _ = cross.classify_events(
        frame, catch_cfg, quantile=0.975
    )
    mask = controls["venue_swap"].fillna(False).astype(bool)
    signal = cross._control_signal(
        mask,
        control_sides["venue_swap"],
        catch_cfg,
        branch="control_venue_swap",
    )
    schedule = cross.nonoverlapping_schedule(signal, frame, end=SELECTION_END)
    return mask, schedule


def _both_overlap_pass(
    raw: dict[str, dict[str, float | int]],
    scheduled: dict[str, dict[str, float | int]],
    name: str,
    *,
    jaccard: float,
    raw_containment: float,
    scheduled_containment: float | None = None,
) -> bool:
    scheduled_limit = (
        raw_containment if scheduled_containment is None else scheduled_containment
    )
    return bool(
        raw[name]["jaccard"] <= jaccard
        and raw[name]["primary_containment"] <= raw_containment
        and scheduled[name]["jaccard"] <= jaccard
        and scheduled[name]["primary_containment"] <= scheduled_limit
    )


def run_support(cfg: Config) -> dict[str, Any]:
    frame, source = load_causal_frame()
    catch_venue_swap_raw, catch_venue_swap_schedule = _catch_venue_swap_reference(frame)
    catch_venue_swap_scheduled = _schedule_mask(catch_venue_swap_schedule, frame)
    prior_specs = {
        "cspr": (cfg.cspr_clock, cfg.cspr_clock_manifest),
        "rift": (cfg.rift_clock, cfg.rift_clock_manifest),
        "catch": (cfg.catch_clock, cfg.catch_clock_manifest),
    }
    prior_masks: dict[str, pd.Series] = {}
    prior_sources: dict[str, Any] = {}
    for label, (clock, manifest) in prior_specs.items():
        prior_masks[label], prior_sources[label] = _load_frozen_clock_mask(
            frame,
            clock_path=clock,
            manifest_path=manifest,
            label=label.upper(),
        )

    rows: list[dict[str, Any]] = []
    for quantile in BASIS_QUANTILES:
        signal, controls, control_sides, _ = classify_events(
            frame, cfg, basis_quantile=quantile
        )
        schedule = nonoverlapping_schedule(signal, frame)
        support = _support(schedule, cfg)
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
        primary = controls["primary"]
        scheduled_primary = _schedule_mask(schedule, frame)
        raw_overlap = {
            name: _overlap(primary, mask)
            for name, mask in controls.items()
            if name not in {"primary", "direction_flip"}
        }
        scheduled_overlap = {
            name: _overlap(scheduled_primary, _schedule_mask(control_schedule, frame))
            for name, control_schedule in control_schedules.items()
        }
        nested_controls = {
            "no_inventory": cfg.maximum_no_inventory_retention,
            "no_basis_history": cfg.maximum_no_basis_retention,
            "no_cash_refusal": cfg.maximum_no_cash_retention,
            "basis_only": cfg.maximum_basis_only_retention,
        }
        retention = {
            name: {
                "raw": _retention(int(primary.sum()), int(controls[name].sum())),
                "scheduled": _retention(len(schedule), len(control_schedules[name])),
                "maximum": maximum,
            }
            for name, maximum in nested_controls.items()
        }
        prior_overlap = {
            label: _overlap(scheduled_primary, mask)
            for label, mask in prior_masks.items()
        }
        catch_venue_swap_overlap = {
            "raw": _overlap(primary, catch_venue_swap_raw),
            "scheduled": _overlap(scheduled_primary, catch_venue_swap_scheduled),
        }
        novelty = (
            all(
                values["raw"] <= values["maximum"]
                and values["scheduled"] <= values["maximum"]
                for values in retention.values()
            )
            and all(
                _both_overlap_pass(
                    raw_overlap,
                    scheduled_overlap,
                    name,
                    jaccard=cfg.maximum_placebo_jaccard,
                    raw_containment=cfg.maximum_placebo_containment,
                )
                for name in (
                    "reverse_time",
                    "spot_confirmed",
                    "spot_inventory_swap",
                )
            )
            and _both_overlap_pass(
                raw_overlap,
                scheduled_overlap,
                "simultaneous_only",
                jaccard=cfg.maximum_simultaneous_jaccard,
                raw_containment=cfg.maximum_simultaneous_raw_containment,
                scheduled_containment=cfg.maximum_simultaneous_scheduled_containment,
            )
            and all(
                _both_overlap_pass(
                    raw_overlap,
                    scheduled_overlap,
                    name,
                    jaccard=cfg.maximum_stale_jaccard,
                    raw_containment=cfg.maximum_stale_containment,
                )
                for name in ("stale_24h", "signal_delay_1bar")
            )
            and all(
                values["jaccard"] <= cfg.maximum_prior_clock_jaccard
                and values["primary_containment"] <= cfg.maximum_prior_clock_containment
                for values in prior_overlap.values()
            )
            and all(
                values["jaccard"] <= cfg.maximum_catch_venue_swap_jaccard
                and values["primary_containment"]
                <= cfg.maximum_catch_venue_swap_containment
                for values in catch_venue_swap_overlap.values()
            )
        )
        rows.append(
            {
                "basis_quantile": quantile,
                "raw_primary": int(primary.sum()),
                "support": support,
                "control_raw_counts": {
                    name: int(mask.sum())
                    for name, mask in controls.items()
                    if name != "primary"
                },
                "control_scheduled_counts": {
                    name: int(len(control_schedule))
                    for name, control_schedule in control_schedules.items()
                },
                "control_overlap": raw_overlap,
                "control_scheduled_overlap": scheduled_overlap,
                "ablation_retention": retention,
                "prior_clock_overlap": prior_overlap,
                "catch_venue_swap_overlap": catch_venue_swap_overlap,
                "passes_novelty": bool(novelty),
                "passes_support": bool(support["passes_count_support"] and novelty),
            }
        )

    passing = [row for row in rows if row["passes_support"]]
    selected = max(passing, key=lambda row: row["basis_quantile"]) if passing else None
    return {
        "protocol": {
            "name": "LURI-48 — Leveraged USD-M Inventory Release Handoff",
            "support_only": True,
            "luri_outcomes_opened": False,
            "catch_outcomes_previously_opened": True,
            "selection_end_exclusive": str(SELECTION_END),
            "formation": f"{cfg.formation_bars} completed bars ending t-1",
            "entry": "next Binance USD-M 5m open after completed trigger bar",
            "hold": f"fixed {cfg.hold_bars} bars",
            "support_selection": (
                "highest frozen basis percentile passing every count, balance, "
                "ablation, temporal-placebo, and prior-clock novelty floor"
            ),
            "control_side_rules": CONTROL_SIDE_RULES,
            "score_bearing_controls": list(SCORE_BEARING_CONTROLS),
        },
        "config": asdict(cfg),
        "source": {
            **source,
            "prior_clocks": prior_sources,
            "catch_venue_swap": {
                "raw_rows": int(catch_venue_swap_raw.sum()),
                "scheduled_rows": int(len(catch_venue_swap_schedule)),
                "basis_quantile": 0.975,
            },
        },
        "support_grid": rows,
        "support_decision": "pass" if selected else "reject_before_returns",
        "selected_basis_quantile": (selected["basis_quantile"] if selected else None),
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
                "luri_outcomes_opened": result["protocol"]["luri_outcomes_opened"],
                "support_decision": result["support_decision"],
                "selected_basis_quantile": result["selected_basis_quantile"],
                "support_grid": [
                    {
                        "basis_quantile": row["basis_quantile"],
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
