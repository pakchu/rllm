"""Outcome-blind support preregistration for PDF-10.

PDF-10 trades a divergence between displayed percentage-band depth and the
within-bar firmness of that depth.  It uses only completed calendar-2023
Binance bookDepth snapshots and contains no price, return, PnL, CAGR, or
drawdown calculation.
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

from training import preregister_cross_collateral_liquidity_hysteresis as cclh
from training import preregister_cross_collateral_liquidity_void_refill as clvr
from training.preregister_metaorder_fragmentation_impact_curvature import (
    nonoverlapping_schedule,
)


FROZEN_CREDIBILITY_MANIFEST = Path(
    "results/binance_cross_collateral_book_credibility_btc_2023_manifest.json"
)
FROZEN_CREDIBILITY_DATA = Path(
    "data/binance_cross_collateral_book_credibility_btc_2023/"
    "BTC_cross_collateral_book_credibility_5m_2023.csv.gz"
)
FROZEN_CREDIBILITY_MANIFEST_SHA256 = (
    "f530f472765c8cb56bf564efd346c734e2404e072b87e2ff8dc3b84e303c30f7"
)
FROZEN_CREDIBILITY_DATA_SHA256 = (
    "45026cc02620d9a0c67f250804f2a06705bf0e824f72257d6c2414f40ab7d429"
)
CCLH_EVENT_CLOCK_SHA256 = (
    "e90079d95b111f95ce64459c42d17e4286636a1a2854ed948e8ada497a13dfa7"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_cross_collateral_liquidity_credibility_fracture.py"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/cross-collateral-liquidity-credibility-fracture-"
    "preregistration-2026-07-14.md"
)
CREDIBILITY_BUILDER_SOURCE = Path(
    "training/build_binance_cross_collateral_book_credibility_2023.py"
)
STANDARDIZER_SOURCE = Path(
    "training/preregister_cross_collateral_liquidity_void_refill.py"
)
SCHEDULER_SOURCE = Path(
    "training/preregister_metaorder_fragmentation_impact_curvature.py"
)
CCLH_SOURCE = Path(
    "training/preregister_cross_collateral_liquidity_hysteresis.py"
)
CCLH_SUPPORT = Path(
    "results/cross_collateral_liquidity_hysteresis_support_2026-07-14.json"
)


@dataclass(frozen=True)
class Config:
    credibility_manifest: str = str(FROZEN_CREDIBILITY_MANIFEST)
    output: str = (
        "results/cross_collateral_liquidity_credibility_fracture_"
        "support_2026-07-14.json"
    )
    robust_baseline_bars: int = 8_640
    robust_min_periods: int = 2_016
    credibility_entry_z: float = 0.75
    display_entry_z: float = 1.00
    confirmation_bars: int = 2
    hold_bars: int = 2
    minimum_nonoverlap_total: int = 500
    minimum_nonoverlap_per_half: int = 180
    minimum_nonoverlap_per_quarter: int = 75
    minimum_side_share: float = 0.35
    maximum_quarter_share: float = 0.40
    short_overlap_tolerance_bars: int = 2
    maximum_short_overlap_jaccard: float = 0.15
    long_overlap_tolerance_bars: int = 12
    maximum_long_overlap_jaccard: float = 0.30
    maximum_geometry_correlation: float = 0.60


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_frozen_config(cfg: Config) -> None:
    expected = {
        "credibility_manifest": str(FROZEN_CREDIBILITY_MANIFEST),
        "robust_baseline_bars": 8_640,
        "robust_min_periods": 2_016,
        "credibility_entry_z": 0.75,
        "display_entry_z": 1.00,
        "confirmation_bars": 2,
        "hold_bars": 2,
        "minimum_nonoverlap_total": 500,
        "minimum_nonoverlap_per_half": 180,
        "minimum_nonoverlap_per_quarter": 75,
        "minimum_side_share": 0.35,
        "maximum_quarter_share": 0.40,
        "short_overlap_tolerance_bars": 2,
        "maximum_short_overlap_jaccard": 0.15,
        "long_overlap_tolerance_bars": 12,
        "maximum_long_overlap_jaccard": 0.30,
        "maximum_geometry_correlation": 0.60,
    }
    changed = {
        name: {"expected": value, "observed": getattr(cfg, name)}
        for name, value in expected.items()
        if getattr(cfg, name) != value
    }
    if changed:
        raise ValueError(f"PDF-10 v1 config is frozen: {changed}")


def _required_columns() -> list[str]:
    depth = [
        f"{venue}_depth_{side}{distance}"
        for venue in ("um", "cm")
        for side in ("m", "p")
        for distance in range(1, 6)
    ]
    credibility = [
        f"{venue}_{statistic}_{side}{distance}"
        for venue in ("um", "cm")
        for statistic in ("log_mad", "log_net", "log_step")
        for side in ("m", "p")
        for distance in range(1, 6)
    ]
    return depth + credibility


def load_credibility(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = Path(cfg.credibility_manifest)
    if _sha256(manifest_path) != FROZEN_CREDIBILITY_MANIFEST_SHA256:
        raise ValueError("credibility manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest.get("protocol", {})
    if protocol.get("outcomes_opened") is not False:
        raise ValueError("credibility manifest opened outcomes")
    if protocol.get("post_2023_rows_requested") is not False:
        raise ValueError("credibility manifest requested post-2023 rows")
    if protocol.get("base_depth_replayed_exactly") is not True:
        raise ValueError("credibility manifest did not replay base depth")

    item = manifest.get("file", {})
    path = Path(item.get("path", ""))
    if path != FROZEN_CREDIBILITY_DATA:
        raise ValueError("credibility manifest points to a non-frozen panel")
    if item.get("sha256") != FROZEN_CREDIBILITY_DATA_SHA256:
        raise ValueError("credibility manifest contains a non-frozen data hash")
    if not path.is_file() or _sha256(path) != FROZEN_CREDIBILITY_DATA_SHA256:
        raise ValueError("credibility data hash mismatch")
    frame = pd.read_csv(path, compression="gzip", parse_dates=["date"])
    clvr._validate_grid(
        frame,
        start="2023-01-01",
        end="2024-01-01",
        label="cross-collateral credibility",
    )
    if len(frame) != item.get("rows") or len(frame.columns) != item.get("columns"):
        raise ValueError("credibility dimensions differ from manifest")

    complete = frame["source_complete"]
    if complete.dtype != bool:
        complete = complete.astype("string").str.lower().map(
            {"true": True, "false": False}
        )
    if complete.isna().any():
        raise ValueError("source_complete contains an unknown value")
    frame["source_complete"] = complete.astype(bool)

    required = _required_columns()
    if not set(required).issubset(frame.columns):
        raise ValueError("credibility columns are incomplete")
    clean_values = frame.loc[frame["source_complete"], required]
    if not np.isfinite(clean_values.to_numpy(float)).all():
        raise ValueError("complete credibility row contains a non-finite value")
    nonnegative = [
        column
        for column in required
        if "log_mad" in column or "log_step" in column
    ]
    if (frame.loc[frame["source_complete"], nonnegative] < 0.0).any().any():
        raise ValueError("dispersion/activity must be non-negative")
    depth = [column for column in required if "_depth_" in column]
    if (frame.loc[frame["source_complete"], depth] <= 0.0).any().any():
        raise ValueError("complete displayed depth must be positive")

    frame["quarantined"] = False
    return frame, {
        "credibility_manifest_sha256": _sha256(manifest_path),
        "credibility_data_sha256": _sha256(path),
        "range_start": "2023-01-01 00:00:00",
        "range_end": "2023-12-31 23:55:00",
        "rows": int(len(frame)),
        "source_complete_rows": int(frame["source_complete"].sum()),
    }


def _lagged_z(values: pd.Series, clean: pd.Series, cfg: Config) -> pd.Series:
    return clvr.lagged_robust_zscore(
        values.where(clean),
        window=cfg.robust_baseline_bars,
        minimum=cfg.robust_min_periods,
    )


def _venue_features(
    frame: pd.DataFrame,
    venue: str,
    cfg: Config,
) -> pd.DataFrame:
    clean = frame["source_complete"].astype(bool)
    credibility_levels: list[pd.Series] = []
    display_levels: list[pd.Series] = []
    for distance in range(1, 6):
        standardized = {
            f"{statistic}_{side}": _lagged_z(
                frame[f"{venue}_{statistic}_{side}{distance}"],
                clean,
                cfg,
            )
            for statistic in ("log_mad", "log_net", "log_step")
            for side in ("m", "p")
        }
        net = standardized["log_net_m"] - standardized["log_net_p"]
        churn = 0.5 * (
            standardized["log_mad_m"] + standardized["log_step_m"]
        ) - 0.5 * (
            standardized["log_mad_p"] + standardized["log_step_p"]
        )
        credibility_levels.append(net - 0.5 * churn)

        display_raw = np.log(
            frame[f"{venue}_depth_m{distance}"]
            / frame[f"{venue}_depth_p{distance}"]
        )
        display_levels.append(_lagged_z(display_raw, clean, cfg))

    return pd.DataFrame(
        {
            "credibility": pd.concat(credibility_levels, axis=1).mean(
                axis=1, skipna=False
            ),
            "display": pd.concat(display_levels, axis=1).mean(
                axis=1, skipna=False
            ),
        },
        index=frame.index,
    )


def _fracture_state(
    um: pd.DataFrame,
    cm: pd.DataFrame,
    clean: pd.Series,
    cfg: Config,
) -> pd.DataFrame:
    if cfg.credibility_entry_z <= 0.0 or cfg.display_entry_z <= 0.0:
        raise ValueError("PDF-10 entry thresholds must be positive")
    if cfg.confirmation_bars != 2:
        raise ValueError("PDF-10 v1 freezes exactly two confirmation bars")

    credibility = 0.5 * (um["credibility"] + cm["credibility"])
    display = 0.5 * (um["display"] + cm["display"])
    values = pd.concat(
        [um, cm, credibility.rename("credibility"), display.rename("display")],
        axis=1,
    )
    observed = clean.astype(bool) & pd.Series(
        np.isfinite(values.to_numpy(float)).all(axis=1),
        index=values.index,
    )

    raw_state = pd.Series(0, index=um.index, dtype=np.int8)
    bullish = (
        observed
        & um["credibility"].gt(0.0)
        & cm["credibility"].gt(0.0)
        & credibility.ge(cfg.credibility_entry_z)
        & display.le(-cfg.display_entry_z)
    )
    bearish = (
        observed
        & um["credibility"].lt(0.0)
        & cm["credibility"].lt(0.0)
        & credibility.le(-cfg.credibility_entry_z)
        & display.ge(cfg.display_entry_z)
    )
    raw_state.loc[bullish] = 1
    raw_state.loc[bearish] = -1
    confirmed = raw_state.ne(0) & raw_state.eq(raw_state.shift(1))
    confirmed_state = raw_state.where(confirmed, 0).astype(np.int8)
    return pd.DataFrame(
        {
            "credibility": credibility,
            "display": display,
            "raw_state": raw_state,
            "confirmed_state": confirmed_state,
        },
        index=um.index,
    )


def build_signal(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    um = _venue_features(frame, "um", cfg)
    cm = _venue_features(frame, "cm", cfg)
    state = _fracture_state(um, cm, frame["source_complete"], cfg)
    side = state["confirmed_state"].astype(np.int8)
    branch = pd.Series("none", index=frame.index, dtype="string")
    branch.loc[side.gt(0)] = "bullish_display_firmness_divergence"
    branch.loc[side.lt(0)] = "bearish_display_firmness_divergence"
    return pd.DataFrame(
        {
            "date": frame["date"],
            "candidate": side.ne(0),
            "um_credibility": um["credibility"],
            "cm_credibility": cm["credibility"],
            "um_display": um["display"],
            "cm_display": cm["display"],
            **state.to_dict("series"),
            "side": side,
            "branch": branch,
            "hold_bars": np.where(side.ne(0), cfg.hold_bars, 0).astype(
                np.int16
            ),
        }
    )


def _quarterly_schedule(
    signal: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = pd.concat(
        [
            nonoverlapping_schedule(signal, frame, start=start, end=end)
            for start, end in (
                ("2023-01-01", "2023-04-01"),
                ("2023-04-01", "2023-07-01"),
                ("2023-07-01", "2023-10-01"),
                ("2023-10-01", "2024-01-01"),
            )
        ],
        ignore_index=True,
    )
    if output.empty:
        return pd.DataFrame(
            columns=[
                "signal_position",
                "entry_position",
                "exit_position",
                "signal_date",
                "entry_date",
                "exit_date",
                "side",
                "branch",
                "hold_bars",
            ]
        )
    return output


def _event_clock_sha256(schedule: pd.DataFrame) -> str:
    records = [
        {
            "signal_position": int(row.signal_position),
            "side": int(row.side),
        }
        for row in schedule[["signal_position", "side"]].itertuples(
            index=False
        )
    ]
    payload = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def support_summary(
    signal: pd.DataFrame,
    frame: pd.DataFrame,
    cfg: Config,
    *,
    schedule: pd.DataFrame | None = None,
) -> dict[str, Any]:
    del signal, frame
    if schedule is None:
        raise ValueError("PDF-10 support requires its frozen quarterly clock")
    signal_dates = pd.to_datetime(schedule["signal_date"])
    quarter = signal_dates.dt.quarter
    total = len(schedule)
    by_quarter = {
        f"q{number}": int(quarter.eq(number).sum()) for number in range(1, 5)
    }
    h1 = by_quarter["q1"] + by_quarter["q2"]
    h2 = by_quarter["q3"] + by_quarter["q4"]
    long_share = float(schedule["side"].gt(0).mean()) if total else 0.0
    short_share = float(schedule["side"].lt(0).mean()) if total else 0.0
    maximum_quarter_share = max(by_quarter.values()) / total if total else 1.0
    passes = (
        total >= cfg.minimum_nonoverlap_total
        and h1 >= cfg.minimum_nonoverlap_per_half
        and h2 >= cfg.minimum_nonoverlap_per_half
        and all(
            count >= cfg.minimum_nonoverlap_per_quarter
            for count in by_quarter.values()
        )
        and min(long_share, short_share) >= cfg.minimum_side_share
        and maximum_quarter_share <= cfg.maximum_quarter_share
    )
    return {
        "nonoverlap_total": int(total),
        "by_quarter": by_quarter,
        "h1": int(h1),
        "h2": int(h2),
        "long_share": long_share,
        "short_share": short_share,
        "maximum_observed_quarter_share": float(maximum_quarter_share),
        "passes_support": bool(passes),
    }


def _independence_summary(
    signal: pd.DataFrame,
    frame: pd.DataFrame,
    cfg: Config,
    *,
    schedule: pd.DataFrame,
) -> dict[str, Any]:
    cclh_signal = cclh.build_signal(frame, cclh.Config())
    prior_schedule = _quarterly_schedule(cclh_signal, frame)
    frozen_cclh = json.loads(CCLH_SUPPORT.read_text())
    replay = {
        "raw_candidate_count": int(cclh_signal["candidate"].sum()),
        "scheduled_side_counts": {
            "long": int(prior_schedule["side"].gt(0).sum()),
            "short": int(prior_schedule["side"].lt(0).sum()),
        },
        "support": cclh.support_summary(cclh_signal, frame, cclh.Config()),
    }
    for field in ("raw_candidate_count", "scheduled_side_counts", "support"):
        if replay[field] != frozen_cclh[field]:
            raise ValueError(f"CCLH {field} did not replay from credibility panel")
    event_clock_sha256 = _event_clock_sha256(prior_schedule)
    if event_clock_sha256 != CCLH_EVENT_CLOCK_SHA256:
        raise ValueError("CCLH event positions did not replay exactly")
    short_overlap = cclh.tolerant_event_jaccard(
        schedule["signal_position"].astype(int).tolist(),
        prior_schedule["signal_position"].astype(int).tolist(),
        tolerance_bars=cfg.short_overlap_tolerance_bars,
    )
    short_overlap["maximum_allowed_jaccard"] = (
        cfg.maximum_short_overlap_jaccard
    )
    short_overlap["passes"] = bool(
        short_overlap["jaccard"] <= cfg.maximum_short_overlap_jaccard
    )
    long_overlap = cclh.tolerant_event_jaccard(
        schedule["signal_position"].astype(int).tolist(),
        prior_schedule["signal_position"].astype(int).tolist(),
        tolerance_bars=cfg.long_overlap_tolerance_bars,
    )
    long_overlap["maximum_allowed_jaccard"] = cfg.maximum_long_overlap_jaccard
    long_overlap["passes"] = bool(
        long_overlap["jaccard"] <= cfg.maximum_long_overlap_jaccard
    )

    geometry = cclh.cross_collateral_geometry(frame)
    features = pd.DataFrame(
        {
            "pdf_credibility": signal["credibility"],
            "pdf_display": signal["display"],
            "cclh_cross_pressure": geometry["cross_pressure"],
            "cclh_cross_elasticity": geometry["cross_elasticity"],
        }
    ).replace([np.inf, -np.inf], np.nan)
    correlations = features.corr(method="spearman")
    raw_pairs = {
        f"{left}__{right}": float(correlations.loc[left, right])
        for left in ("pdf_credibility", "pdf_display")
        for right in ("cclh_cross_pressure", "cclh_cross_elasticity")
    }
    correlations_finite = all(np.isfinite(value) for value in raw_pairs.values())
    pairs = {
        name: value if np.isfinite(value) else None
        for name, value in raw_pairs.items()
    }
    maximum_absolute_correlation = (
        max(abs(value) for value in raw_pairs.values())
        if correlations_finite
        else None
    )
    correlation_passes = bool(
        maximum_absolute_correlation is not None
        and maximum_absolute_correlation <= cfg.maximum_geometry_correlation
    )
    return {
        "cclh_frozen_support_replayed_exactly": True,
        "cclh_event_clock_sha256": event_clock_sha256,
        "cclh_event_overlap_2_bars": short_overlap,
        "cclh_event_overlap_12_bars": long_overlap,
        "spearman_feature_correlations": pairs,
        "maximum_absolute_geometry_correlation": maximum_absolute_correlation,
        "maximum_allowed_geometry_correlation": cfg.maximum_geometry_correlation,
        "correlation_passes": correlation_passes,
        "passes_independence": bool(
            short_overlap["passes"]
            and long_overlap["passes"]
            and correlation_passes
        ),
    }


def run_support(cfg: Config) -> dict[str, Any]:
    _validate_frozen_config(cfg)
    frame, source = load_credibility(cfg)
    signal = build_signal(frame, cfg)
    schedule = _quarterly_schedule(signal, frame)
    support = support_summary(signal, frame, cfg, schedule=schedule)
    independence = _independence_summary(
        signal,
        frame,
        cfg,
        schedule=schedule,
    )
    passes_all = support["passes_support"] and independence["passes_independence"]
    return {
        "protocol": {
            "name": "PDF-10 — Phantom Display Fracture, 10-minute hold",
            "support_only": True,
            "outcomes_opened_for_pdf10": False,
            "support_rejected": not passes_all,
            "selection_end_exclusive": "2024-01-01 00:00:00",
            "event_clock": (
                "each completed 5m bar whose current and previous raw "
                "display/firmness divergence states have the same side"
            ),
            "signal_availability": (
                "current completed 5m credibility bar; enter next 5m open"
            ),
            "action_rule": "follow firmness, fade displayed percentage-band depth",
            "candidate_clock": (
                "every confirmed evidence bar; non-overlap scheduler may "
                "re-enter while a long divergence run persists"
            ),
            "holding_rule": "2 completed 5m bars; exit at scheduled open t+3",
            "support_parameters_searched": False,
            "price_or_return_loaded": False,
            "source_gap_policy": (
                "both confirmation bars must be source-complete; future "
                "credibility gaps do not cancel an entered trade"
            ),
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
        },
        "config": asdict(cfg),
        "frozen_artifacts": {
            "preregistration_source": str(PREREGISTRATION_SOURCE),
            "preregistration_source_sha256": _sha256(PREREGISTRATION_SOURCE),
            "preregistration_document": str(PREREGISTRATION_DOCUMENT),
            "preregistration_document_sha256": _sha256(
                PREREGISTRATION_DOCUMENT
            ),
            "credibility_builder_source": str(CREDIBILITY_BUILDER_SOURCE),
            "credibility_builder_source_sha256": _sha256(
                CREDIBILITY_BUILDER_SOURCE
            ),
            "standardizer_source": str(STANDARDIZER_SOURCE),
            "standardizer_source_sha256": _sha256(STANDARDIZER_SOURCE),
            "scheduler_source": str(SCHEDULER_SOURCE),
            "scheduler_source_sha256": _sha256(SCHEDULER_SOURCE),
            "cclh_source": str(CCLH_SOURCE),
            "cclh_source_sha256": _sha256(CCLH_SOURCE),
            "cclh_support": str(CCLH_SUPPORT),
            "cclh_support_sha256": _sha256(CCLH_SUPPORT),
            "credibility_manifest_sha256": _sha256(
                cfg.credibility_manifest
            ),
        },
        "source": source,
        "feature": {
            "raw_bullish_rows": int(signal["raw_state"].gt(0).sum()),
            "raw_bearish_rows": int(signal["raw_state"].lt(0).sum()),
            "confirmed_bullish_rows": int(
                signal["confirmed_state"].gt(0).sum()
            ),
            "confirmed_bearish_rows": int(
                signal["confirmed_state"].lt(0).sum()
            ),
            "standardization": (
                "each venue/statistic/side/level and each displayed-depth "
                "ratio uses a strictly lagged rolling median and recursive MAD"
            ),
        },
        "support_calibration": {
            "outcomes_opened_for_pdf10": False,
            "parameters_searched": False,
            "all_parameters_fixed": True,
            "further_support_repairs_allowed": False,
        },
        "raw_candidate_count": int(signal["candidate"].sum()),
        "scheduled_side_counts": {
            "long": int(schedule["side"].gt(0).sum()),
            "short": int(schedule["side"].lt(0).sum()),
        },
        "scheduled_branch_counts": {
            name: int(count)
            for name, count in schedule["branch"].value_counts().items()
        },
        "independence": independence,
        "support": support,
        "all_support_gates_pass": bool(passes_all),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=Config.output)
    args = parser.parse_args()
    result = run_support(Config(output=args.output))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "outcomes_opened_for_pdf10": False,
                "support_rejected": result["protocol"]["support_rejected"],
                "support": result["support"],
                "independence": result["independence"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
