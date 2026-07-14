"""Outcome-blind support preregistration for UMFR-36.

UMFR identifies completed five-minute bars where a late USD-M impulse stretches
basis farther than Spot validates, suggesting a forced derivatives-flow shock.
It trades the next-open reversion against the USD-M flow for a fixed 36 bars.
This module only inspects causal feature incidence and frozen-clock overlap; it
contains no post-entry return, future OHLC, CAGR, or MDD logic.
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
SUPPORT_QUANTILES = (0.50, 0.65, 0.75, 0.80, 0.85)
CONTROL_SIDE_RULES = {
    "primary": "negative sign(um_flow_fraction)",
    "direction_flip": "sign(um_flow_fraction) on the primary clock",
    "no_late_um": "negative sign(um_flow_fraction)",
    "no_ticket_surprise": "negative sign(um_flow_fraction)",
    "no_path_excess": "negative sign(um_flow_fraction)",
    "no_basis_stretch": "negative sign(um_flow_fraction)",
    "no_spot_underresponse": "negative sign(um_flow_fraction)",
    "early_um": "negative sign(um_flow_fraction)",
    "venue_swap": "negative sign(spot_flow_fraction)",
    "aggregate_only": "negative sign(um_flow_fraction)",
    "stale_1h": "primary side stale by 12 completed bars",
    "stale_24h": "primary side stale by 288 completed bars",
    "signal_delay_1bar": "primary side delayed by one completed bar",
}
SCORE_BEARING_CONTROLS = tuple(
    name
    for name in CONTROL_SIDE_RULES
    if name not in {"primary", "direction_flip", "signal_delay_1bar"}
)
SCHEDULE_COLUMNS = cross.SCHEDULE_COLUMNS


@dataclass(frozen=True)
class Config:
    output: str = "results/um_forced_flow_reversion_support_2026-07-14.json"
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
    luri_clock: str = (
        "results/leveraged_um_inventory_release_handoff_clock_2026-07-14.csv"
    )
    luri_clock_manifest: str = (
        "results/leveraged_um_inventory_release_handoff_clock_manifest_2026-07-14.json"
    )
    clasp_clock: str = (
        "results/cash_late_arrival_spillover_propagation_clock_2026-07-14.csv"
    )
    clasp_clock_manifest: str = (
        "results/cash_late_arrival_spillover_propagation_clock_manifest_2026-07-14.json"
    )
    ticket_baseline_bars: int = 8_640
    ticket_min_periods: int = 2_016
    event_baseline_bars: int = 17_280
    event_min_periods: int = 64
    hold_bars: int = 36
    minimum_nonoverlap_total: int = 900
    minimum_nonoverlap_per_year: int = 130
    minimum_nonoverlap_per_2023_half: int = 100
    minimum_nonoverlap_per_2023_quarter: int = 40
    minimum_side_share: float = 0.40
    minimum_side_events_per_year: int = 60
    minimum_active_months: int = 44
    minimum_events_per_active_month: int = 5
    maximum_temporal_jaccard: float = 0.03
    maximum_temporal_containment: float = 0.06
    maximum_stale_jaccard: float = 0.03
    maximum_stale_containment: float = 0.06
    maximum_prior_clock_jaccard: float = 0.05
    maximum_prior_clock_containment: float = 0.10


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_causal_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, source = cross.load_causal_frame(cross.Config())
    if frame["date"].max() >= SELECTION_END:
        raise ValueError("UMFR support frame contains 2024+ rows")
    forbidden = {"open", "high", "low", "close", "future_return", "profit"}
    if forbidden.intersection(frame.columns):
        raise ValueError("UMFR support frame unexpectedly contains outcome columns")
    return frame, source


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _minimum(*values: pd.Series) -> pd.Series:
    if not values:
        raise ValueError("UMFR minimum requires at least one series")
    return pd.concat(values, axis=1).min(axis=1, skipna=False)


def _geometric_mean(*components: pd.Series) -> pd.Series:
    if not components:
        raise ValueError("UMFR geometric mean requires at least one component")
    values = np.vstack(
        [
            component.clip(lower=0.0).fillna(0.0).to_numpy(float)
            for component in components
        ]
    )
    return pd.Series(
        np.prod(values, axis=0) ** (1.0 / len(components)),
        index=components[0].index,
        dtype=float,
    )


def strictly_prior_ticket_median(
    ticket_ratio: pd.Series, clean: pd.Series, cfg: Config
) -> pd.Series:
    if not 1 <= cfg.ticket_min_periods <= cfg.ticket_baseline_bars:
        raise ValueError("UMFR ticket baseline periods are invalid")
    return (
        pd.to_numeric(ticket_ratio, errors="coerce")
        .where(clean.astype(bool))
        .shift(1)
        .rolling(cfg.ticket_baseline_bars, min_periods=cfg.ticket_min_periods)
        .median()
    )


def prior_event_quantile(
    score: pd.Series,
    eligible: pd.Series,
    *,
    quantile: float,
    cfg: Config,
) -> pd.Series:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("UMFR support quantile must be in [0, 1]")
    if not 1 <= cfg.event_min_periods <= cfg.event_baseline_bars:
        raise ValueError("UMFR event baseline periods are invalid")
    return (
        pd.to_numeric(score, errors="coerce")
        .where(eligible.astype(bool))
        .shift(1)
        .rolling(cfg.event_baseline_bars, min_periods=cfg.event_min_periods)
        .quantile(quantile)
    )


def _components(frame: pd.DataFrame, cfg: Config) -> dict[str, pd.Series]:
    clean = ~frame["quarantined"].astype(bool)
    v = {
        column: _numeric(frame, column)
        for column in (
            "spot_quote_notional",
            "um_quote_notional",
            "spot_trade_count",
            "um_trade_count",
            "spot_flow_fraction",
            "um_flow_fraction",
            "um_flow_coherence",
            "spot_log_return_5m",
            "um_log_return_5m",
            "spot_abs_path_return_bp",
            "um_abs_path_return_bp",
            "spot_activity_time_centroid",
            "um_activity_time_centroid",
            "spot_flow_time_centroid",
            "um_flow_time_centroid",
            "spot_return_time_centroid",
            "um_return_time_centroid",
            "basis_change_bp",
        )
    }
    v["um_side"] = pd.Series(
        np.sign(v["um_flow_fraction"]), index=frame.index, dtype=float
    )
    v["spot_side"] = pd.Series(
        np.sign(v["spot_flow_fraction"]), index=frame.index, dtype=float
    )
    v["um_directed_return_bp"] = v["um_side"] * v["um_log_return_5m"] * 10_000.0
    v["spot_in_um_direction_bp"] = v["um_side"] * v["spot_log_return_5m"] * 10_000.0
    v["spot_directed_return_bp"] = v["spot_side"] * v["spot_log_return_5m"] * 10_000.0
    v["um_in_spot_direction_bp"] = v["spot_side"] * v["um_log_return_5m"] * 10_000.0
    v["um_basis_stretch_bp"] = v["um_side"] * v["basis_change_bp"]
    v["spot_basis_stretch_bp"] = -v["spot_side"] * v["basis_change_bp"]

    um_late_components = [
        v["um_activity_time_centroid"] - v["spot_activity_time_centroid"],
        v["um_flow_time_centroid"] - v["spot_flow_time_centroid"],
        v["um_return_time_centroid"] - v["spot_return_time_centroid"],
    ]
    v["um_late_pressure"] = _minimum(*um_late_components)
    v["spot_late_pressure"] = _minimum(
        *[-component for component in um_late_components]
    )

    um_path_excess = v["um_abs_path_return_bp"] - (
        v["um_log_return_5m"].abs() * 10_000.0
    )
    spot_path_excess = v["spot_abs_path_return_bp"] - (
        v["spot_log_return_5m"].abs() * 10_000.0
    )
    v["um_path_excess_advantage_bp"] = um_path_excess - spot_path_excess
    v["spot_path_excess_advantage_bp"] = spot_path_excess - um_path_excess

    spot_average_ticket = v["spot_quote_notional"] / v["spot_trade_count"].where(
        v["spot_trade_count"].gt(0.0)
    )
    um_average_ticket = v["um_quote_notional"] / v["um_trade_count"].where(
        v["um_trade_count"].gt(0.0)
    )
    ticket_ratio = np.log(um_average_ticket / spot_average_ticket).replace(
        [np.inf, -np.inf], np.nan
    )
    ticket_median = strictly_prior_ticket_median(ticket_ratio, clean, cfg)
    v["um_ticket_surprise"] = ticket_ratio - ticket_median
    v["spot_ticket_surprise"] = ticket_median - ticket_ratio
    v["clean"] = clean
    return v


def _score_components(
    *,
    late: pd.Series | None,
    basis: pd.Series | None,
    path: pd.Series | None,
    ticket: pd.Series | None,
    flow: pd.Series | None,
) -> pd.Series:
    components = []
    if late is not None:
        components.append(late)
    if basis is not None:
        components.append(np.log1p(basis.clip(lower=0.0)))
    if path is not None:
        components.append(np.log1p(path.clip(lower=0.0)))
    if ticket is not None:
        components.append(np.log1p(ticket.clip(lower=0.0)))
    if flow is not None:
        components.append(flow.abs())
    return _geometric_mean(*components)


def classify_events(
    frame: pd.DataFrame, cfg: Config, *, quantile: float
) -> tuple[
    pd.DataFrame, dict[str, pd.Series], dict[str, pd.Series], dict[str, pd.Series]
]:
    v = _components(frame, cfg)
    clean = v["clean"]
    um_common = clean & v["um_side"].ne(0.0) & v["um_directed_return_bp"].gt(0.0)
    spot_common = clean & v["spot_side"].ne(0.0) & v["spot_directed_return_bp"].gt(0.0)
    required = {
        "late": v["um_late_pressure"].gt(0.0),
        "ticket": v["um_ticket_surprise"].gt(0.0),
        "path": v["um_path_excess_advantage_bp"].gt(0.0),
        "basis": v["um_basis_stretch_bp"].gt(0.0),
        "underresponse": v["spot_in_um_direction_bp"].lt(v["um_directed_return_bp"]),
    }
    base_masks = {
        "primary": um_common & np.logical_and.reduce(list(required.values())),
        "no_late_um": um_common
        & required["ticket"]
        & required["path"]
        & required["basis"]
        & required["underresponse"],
        "no_ticket_surprise": um_common
        & required["late"]
        & required["path"]
        & required["basis"]
        & required["underresponse"],
        "no_path_excess": um_common
        & required["late"]
        & required["ticket"]
        & required["basis"]
        & required["underresponse"],
        "no_basis_stretch": um_common
        & required["late"]
        & required["ticket"]
        & required["path"]
        & required["underresponse"],
        "no_spot_underresponse": um_common
        & required["late"]
        & required["ticket"]
        & required["path"]
        & required["basis"],
        "early_um": um_common
        & v["spot_late_pressure"].gt(0.0)
        & required["ticket"]
        & required["path"]
        & required["basis"]
        & required["underresponse"],
        "venue_swap": spot_common
        & v["spot_late_pressure"].gt(0.0)
        & v["spot_ticket_surprise"].gt(0.0)
        & v["spot_path_excess_advantage_bp"].gt(0.0)
        & v["spot_basis_stretch_bp"].gt(0.0)
        & v["um_in_spot_direction_bp"].lt(v["spot_directed_return_bp"]),
        "aggregate_only": um_common & required["basis"] & required["underresponse"],
    }
    scores = {
        "primary": _score_components(
            late=v["um_late_pressure"],
            basis=v["um_basis_stretch_bp"],
            path=v["um_path_excess_advantage_bp"],
            ticket=v["um_ticket_surprise"],
            flow=v["um_flow_fraction"],
        ),
        "no_late_um": _score_components(
            late=None,
            basis=v["um_basis_stretch_bp"],
            path=v["um_path_excess_advantage_bp"],
            ticket=v["um_ticket_surprise"],
            flow=v["um_flow_fraction"],
        ),
        "no_ticket_surprise": _score_components(
            late=v["um_late_pressure"],
            basis=v["um_basis_stretch_bp"],
            path=v["um_path_excess_advantage_bp"],
            ticket=None,
            flow=v["um_flow_fraction"],
        ),
        "no_path_excess": _score_components(
            late=v["um_late_pressure"],
            basis=v["um_basis_stretch_bp"],
            path=None,
            ticket=v["um_ticket_surprise"],
            flow=v["um_flow_fraction"],
        ),
        "no_basis_stretch": _score_components(
            late=v["um_late_pressure"],
            basis=None,
            path=v["um_path_excess_advantage_bp"],
            ticket=v["um_ticket_surprise"],
            flow=v["um_flow_fraction"],
        ),
        "no_spot_underresponse": _score_components(
            late=v["um_late_pressure"],
            basis=v["um_basis_stretch_bp"],
            path=v["um_path_excess_advantage_bp"],
            ticket=v["um_ticket_surprise"],
            flow=v["um_flow_fraction"],
        ),
        "early_um": _score_components(
            late=v["spot_late_pressure"],
            basis=v["um_basis_stretch_bp"],
            path=v["um_path_excess_advantage_bp"],
            ticket=v["um_ticket_surprise"],
            flow=v["um_flow_fraction"],
        ),
        "venue_swap": _score_components(
            late=v["spot_late_pressure"],
            basis=v["spot_basis_stretch_bp"],
            path=v["spot_path_excess_advantage_bp"],
            ticket=v["spot_ticket_surprise"],
            flow=v["spot_flow_fraction"],
        ),
        "aggregate_only": _geometric_mean(
            v["um_flow_fraction"].abs(),
            v["um_flow_coherence"],
            np.log1p(v["um_basis_stretch_bp"].clip(lower=0.0)),
        ),
    }
    thresholds = {
        name: prior_event_quantile(score, base_masks[name], quantile=quantile, cfg=cfg)
        for name, score in scores.items()
    }
    selected = {
        name: (base_masks[name] & scores[name].ge(thresholds[name])).fillna(False)
        for name in scores
    }
    primary = selected["primary"]
    controls = {
        "primary": primary,
        "direction_flip": primary,
        **{name: mask for name, mask in selected.items() if name != "primary"},
        "stale_1h": primary.shift(12, fill_value=False) & clean,
        "stale_24h": primary.shift(288, fill_value=False) & clean,
        "signal_delay_1bar": primary.shift(1, fill_value=False) & clean,
    }
    control_sides = {
        "primary": -v["um_side"],
        "direction_flip": v["um_side"],
        **{
            name: (-v["spot_side"] if name == "venue_swap" else -v["um_side"])
            for name in selected
            if name != "primary"
        },
        "stale_1h": (-v["um_side"]).shift(12).fillna(0.0),
        "stale_24h": (-v["um_side"]).shift(288).fillna(0.0),
        "signal_delay_1bar": (-v["um_side"]).shift(1).fillna(0.0),
    }
    signal = pd.DataFrame(
        {
            "date": frame["date"],
            "side": np.where(primary, -v["um_side"], 0).astype(np.int8),
            "branch": np.where(primary, "umfr36", "none"),
            "hold_bars": np.where(primary, cfg.hold_bars, 0).astype(np.int16),
            "score": scores["primary"],
            "threshold": thresholds["primary"],
            "um_late_pressure": v["um_late_pressure"],
            "um_ticket_surprise": v["um_ticket_surprise"],
            "um_path_excess_advantage_bp": v["um_path_excess_advantage_bp"],
            "um_basis_stretch_bp": v["um_basis_stretch_bp"],
            "spot_underresponse_gap_bp": v["um_directed_return_bp"]
            - v["spot_in_um_direction_bp"],
            "quarantined": frame["quarantined"].astype(bool),
        }
    )
    diagnostics = {
        **{f"base_{name}": mask for name, mask in base_masks.items()},
        **{f"score_{name}": score for name, score in scores.items()},
        **{f"threshold_{name}": value for name, value in thresholds.items()},
    }
    return signal, controls, control_sides, diagnostics


def nonoverlapping_schedule(signal: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    return cross.nonoverlapping_schedule(signal, frame, end=SELECTION_END)


def _control_signal(
    mask: pd.Series, side: pd.Series, cfg: Config, *, branch: str
) -> pd.DataFrame:
    active = mask.fillna(False).astype(bool)
    numeric_side = pd.to_numeric(side, errors="coerce").fillna(0.0)
    if not numeric_side.loc[active].isin((-1.0, 1.0)).all():
        raise ValueError(f"UMFR control {branch} has invalid active side")
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


def _load_frozen_clock_mask(
    frame: pd.DataFrame, *, clock_path: str, manifest_path: str, label: str
) -> tuple[pd.Series, dict[str, Any]]:
    clock_file = Path(clock_path)
    manifest_file = Path(manifest_path)
    manifest = _read_json(manifest_file)
    protocol = manifest.get("protocol", {})
    opened = protocol.get("outcomes_opened")
    if opened is None:
        opened = protocol.get(f"{label.lower()}_outcomes_opened")
    if opened is not False:
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


def _passes_overlap(
    overlap: dict[str, float | int],
    *,
    maximum_jaccard: float,
    maximum_containment: float,
) -> bool:
    return bool(
        overlap["jaccard"] <= maximum_jaccard
        and overlap["primary_containment"] <= maximum_containment
    )


def run_support(cfg: Config) -> dict[str, Any]:
    frame, source = load_causal_frame()
    prior_specs = {
        "cspr": (cfg.cspr_clock, cfg.cspr_clock_manifest),
        "rift": (cfg.rift_clock, cfg.rift_clock_manifest),
        "catch": (cfg.catch_clock, cfg.catch_clock_manifest),
        "luri": (cfg.luri_clock, cfg.luri_clock_manifest),
        "clasp": (cfg.clasp_clock, cfg.clasp_clock_manifest),
    }
    prior_masks: dict[str, pd.Series] = {}
    prior_sources: dict[str, Any] = {}
    for label, (clock, manifest) in prior_specs.items():
        prior_masks[label], prior_sources[label] = _load_frozen_clock_mask(
            frame, clock_path=clock, manifest_path=manifest, label=label.upper()
        )

    rows: list[dict[str, Any]] = []
    for quantile in SUPPORT_QUANTILES:
        signal, controls, control_sides, diagnostics = classify_events(
            frame, cfg, quantile=quantile
        )
        schedule = nonoverlapping_schedule(signal, frame)
        support = _support(schedule, cfg)
        primary = controls["primary"]
        scheduled_primary = _schedule_mask(schedule, frame)
        control_schedules = {
            name: nonoverlapping_schedule(
                _control_signal(
                    mask, control_sides[name], cfg, branch=f"control_{name}"
                ),
                frame,
            )
            for name, mask in controls.items()
            if name not in {"primary", "direction_flip"}
        }
        raw_overlap = {
            name: _overlap(primary, mask)
            for name, mask in controls.items()
            if name not in {"primary", "direction_flip"}
        }
        scheduled_overlap = {
            name: _overlap(scheduled_primary, _schedule_mask(control_schedule, frame))
            for name, control_schedule in control_schedules.items()
        }
        prior_overlap = {
            label: _overlap(scheduled_primary, mask)
            for label, mask in prior_masks.items()
        }
        novelty = (
            all(
                _passes_overlap(
                    overlap[name],
                    maximum_jaccard=cfg.maximum_temporal_jaccard,
                    maximum_containment=cfg.maximum_temporal_containment,
                )
                for overlap in (raw_overlap, scheduled_overlap)
                for name in ("early_um", "venue_swap")
            )
            and all(
                _passes_overlap(
                    overlap[name],
                    maximum_jaccard=cfg.maximum_stale_jaccard,
                    maximum_containment=cfg.maximum_stale_containment,
                )
                for overlap in (raw_overlap, scheduled_overlap)
                for name in ("stale_1h", "stale_24h", "signal_delay_1bar")
            )
            and all(
                _passes_overlap(
                    value,
                    maximum_jaccard=cfg.maximum_prior_clock_jaccard,
                    maximum_containment=cfg.maximum_prior_clock_containment,
                )
                for value in prior_overlap.values()
            )
        )
        ablation_names = (
            "no_late_um",
            "no_ticket_surprise",
            "no_path_excess",
            "no_basis_stretch",
            "no_spot_underresponse",
            "aggregate_only",
        )
        rows.append(
            {
                "quantile": quantile,
                "eligible_primary": int(diagnostics["base_primary"].sum()),
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
                "ablation_retention_diagnostic_only": {
                    name: {
                        "raw": float(
                            int(primary.sum()) / max(1, int(controls[name].sum()))
                        ),
                        "scheduled": float(
                            len(schedule) / max(1, len(control_schedules[name]))
                        ),
                    }
                    for name in ablation_names
                },
                "prior_clock_overlap": prior_overlap,
                "passes_novelty": bool(novelty),
                "passes_support": bool(support["passes_count_support"] and novelty),
            }
        )

    passing = [row for row in rows if row["passes_support"]]
    selected = max(passing, key=lambda row: row["quantile"]) if passing else None
    return {
        "protocol": {
            "name": "UMFR-36 — USD-M Forced Flow Reversion",
            "support_only": True,
            "umfr_outcomes_opened": False,
            "selection_end_exclusive": str(SELECTION_END),
            "entry": "next Binance USD-M 5m open after completed signal bar",
            "hold": f"fixed {cfg.hold_bars} bars",
            "economic_claim": "a late noisy USD-M flow impulse stretches basis beyond Spot validation and reverts",
            "support_selection": "highest frozen causal event-score quantile passing every support and novelty floor",
            "control_side_rules": CONTROL_SIDE_RULES,
            "ablation_retention_is_gate": False,
        },
        "config": asdict(cfg),
        "source": {**source, "prior_clocks": prior_sources},
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
                "outcomes_opened": result["protocol"]["umfr_outcomes_opened"],
                "support_decision": result["support_decision"],
                "selected_quantile": result["selected_quantile"],
                "support_grid": [
                    {
                        "quantile": row["quantile"],
                        "eligible_primary": row["eligible_primary"],
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
