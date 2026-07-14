"""Outcome-blind support preregistration for RCR-144.

RCR-144 measures the bid/ask-antisymmetric rotation of normalized visible
depth composition across non-overlapping radial shells.  It uses only the
frozen calendar-2023 shell panel and contains no market-price, return, PnL,
CAGR, or drawdown calculation.
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

from training import preregister_cross_collateral_liquidity_credibility_fracture as pdf
from training import preregister_cross_collateral_liquidity_hysteresis as cclh
from training import preregister_cross_collateral_liquidity_void_refill as clvr
from training import preregister_radial_liquidity_wavefront_cascade as rlwc


FROZEN_SHELL_MANIFEST = rlwc.FROZEN_SHELL_MANIFEST
FROZEN_SHELL_MANIFEST_SHA256 = rlwc.FROZEN_SHELL_MANIFEST_SHA256
FROZEN_SHELL_DATA = rlwc.FROZEN_SHELL_DATA
FROZEN_SHELL_DATA_SHA256 = rlwc.FROZEN_SHELL_DATA_SHA256
PDF_EVENT_CLOCK_SHA256 = rlwc.PDF_EVENT_CLOCK_SHA256
PREREGISTRATION_SOURCE = Path(
    "training/preregister_radial_composition_rotation.py"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/radial-composition-rotation-preregistration-2026-07-14.md"
)
SHELL_BUILDER_SOURCE = rlwc.SHELL_BUILDER_SOURCE
STANDARDIZER_SOURCE = rlwc.STANDARDIZER_SOURCE
SCHEDULER_SOURCE = rlwc.SCHEDULER_SOURCE
CCLH_SOURCE = rlwc.CCLH_SOURCE
CCLH_SUPPORT = rlwc.CCLH_SUPPORT
PDF_SOURCE = rlwc.PDF_SOURCE
PDF_SUPPORT = rlwc.PDF_SUPPORT
PDF_EVENT_CLOCK = rlwc.PDF_EVENT_CLOCK
RLWC_SOURCE = rlwc.PREREGISTRATION_SOURCE
RLWC_SUPPORT = Path(
    "results/radial_liquidity_wavefront_cascade_support_2026-07-14.json"
)


@dataclass(frozen=True)
class Config:
    shell_manifest: str = str(FROZEN_SHELL_MANIFEST)
    output: str = (
        "results/radial_composition_rotation_support_2026-07-14.json"
    )
    robust_baseline_bars: int = 8_640
    robust_min_periods: int = 2_016
    score_threshold: float = 2.00
    hold_bars: int = 144
    minimum_available_total: int = 90_000
    minimum_available_per_quarter: int = 15_000
    minimum_strong_per_quarter: int = 500
    minimum_strong_side_share: float = 0.35
    minimum_nonoverlap_total: int = 120
    minimum_nonoverlap_per_half: int = 45
    minimum_nonoverlap_per_quarter: int = 20
    minimum_side_share: float = 0.35
    maximum_quarter_share: float = 0.40
    overlap_tolerance_bars: int = 12
    maximum_prior_event_jaccard: float = 0.35
    maximum_prior_event_match_share: float = 0.35
    maximum_prior_feature_spearman: float = 0.60


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_frozen_config(cfg: Config) -> None:
    expected = {
        "shell_manifest": str(FROZEN_SHELL_MANIFEST),
        "robust_baseline_bars": 8_640,
        "robust_min_periods": 2_016,
        "score_threshold": 2.00,
        "hold_bars": 144,
        "minimum_available_total": 90_000,
        "minimum_available_per_quarter": 15_000,
        "minimum_strong_per_quarter": 500,
        "minimum_strong_side_share": 0.35,
        "minimum_nonoverlap_total": 120,
        "minimum_nonoverlap_per_half": 45,
        "minimum_nonoverlap_per_quarter": 20,
        "minimum_side_share": 0.35,
        "maximum_quarter_share": 0.40,
        "overlap_tolerance_bars": 12,
        "maximum_prior_event_jaccard": 0.35,
        "maximum_prior_event_match_share": 0.35,
        "maximum_prior_feature_spearman": 0.60,
    }
    changed = {
        name: {"expected": value, "observed": getattr(cfg, name)}
        for name, value in expected.items()
        if getattr(cfg, name) != value
    }
    if changed:
        raise ValueError(f"RCR-144 v1 config is frozen: {changed}")


def load_shells(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    if cfg.shell_manifest != str(FROZEN_SHELL_MANIFEST):
        raise ValueError("RCR-144 requires the frozen radial-shell manifest")
    return rlwc.load_shells(
        rlwc.Config(shell_manifest=cfg.shell_manifest)
    )


def _lagged_z(
    values: pd.Series,
    clean: pd.Series,
    cfg: Config,
) -> pd.Series:
    return clvr.lagged_robust_zscore(
        values.where(clean),
        window=cfg.robust_baseline_bars,
        minimum=cfg.robust_min_periods,
    )


def _radial_barycenter(
    frame: pd.DataFrame,
    venue: str,
    side: str,
) -> pd.Series:
    if venue not in ("um", "cm") or side not in ("m", "p"):
        raise ValueError("unknown RCR venue or side")
    shares = frame[
        [
            f"{venue}_shell_share_median_{side}{shell}"
            for shell in range(1, 6)
        ]
    ].to_numpy(float)
    total = shares.sum(axis=1)
    valid = (
        np.isfinite(shares).all(axis=1)
        & np.greater_equal(shares, 0.0).all(axis=1)
        & np.isfinite(total)
        & (total > 0.0)
    )
    values = np.full(len(frame), np.nan, dtype=float)
    weights = np.arange(1.0, 6.0)
    values[valid] = shares[valid].dot(weights) / total[valid]
    return pd.Series(values, index=frame.index)


def _venue_rotation(frame: pd.DataFrame, venue: str) -> pd.DataFrame:
    bid_barycenter = _radial_barycenter(frame, venue, "m")
    ask_barycenter = _radial_barycenter(frame, venue, "p")
    current_clean = frame["source_complete"].eq(True).fillna(False).astype(bool)
    pair_clean = current_clean & current_clean.shift(1, fill_value=False)
    bid_inward = bid_barycenter.shift(1) - bid_barycenter
    ask_inward = ask_barycenter.shift(1) - ask_barycenter
    polarization = (bid_inward - ask_inward) / np.sqrt(2.0)
    finite = pd.concat(
        [
            bid_barycenter,
            ask_barycenter,
            bid_barycenter.shift(1),
            ask_barycenter.shift(1),
            polarization,
        ],
        axis=1,
    ).notna().all(axis=1)
    return pd.DataFrame(
        {
            "bid_barycenter": bid_barycenter,
            "ask_barycenter": ask_barycenter,
            "bid_inward": bid_inward,
            "ask_inward": ask_inward,
            "polarization": polarization,
            "clean": pair_clean & finite,
        }
    )


def build_features(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    venue = {
        name: _venue_rotation(frame, name) for name in ("um", "cm")
    }
    output = pd.DataFrame({"date": frame["date"]})
    for name, values in venue.items():
        output[f"{name}_bid_barycenter"] = values["bid_barycenter"]
        output[f"{name}_ask_barycenter"] = values["ask_barycenter"]
        output[f"{name}_bid_inward"] = values["bid_inward"]
        output[f"{name}_ask_inward"] = values["ask_inward"]
        output[f"{name}_polarization"] = values["polarization"]
        output[f"{name}_polarization_z"] = _lagged_z(
            values["polarization"],
            values["clean"],
            cfg,
        )
        output[f"{name}_rotation_clean"] = values["clean"]
    output["score"] = (
        output["um_polarization_z"] + output["cm_polarization_z"]
    ) / np.sqrt(2.0)
    output["available"] = (
        output["um_rotation_clean"].astype(bool)
        & output["cm_rotation_clean"].astype(bool)
        & np.isfinite(output["score"])
    )
    return output


def build_signal(features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    score = features["score"].astype(float)
    strong = features["available"].astype(bool) & score.abs().ge(
        cfg.score_threshold
    )
    side = np.sign(score.where(strong, 0.0)).fillna(0.0).astype(np.int8)
    branch = pd.Series("none", index=features.index, dtype="string")
    branch.loc[side.gt(0)] = "bullish_radial_composition_rotation"
    branch.loc[side.lt(0)] = "bearish_radial_composition_rotation"
    return pd.DataFrame(
        {
            "date": features["date"],
            "candidate": side.ne(0),
            "side": side,
            "branch": branch,
            "hold_bars": np.where(side.ne(0), cfg.hold_bars, 0).astype(
                np.int16
            ),
        }
    )


def availability_summary(
    features: pd.DataFrame,
    signal: pd.DataFrame,
    cfg: Config,
) -> dict[str, Any]:
    quarter = pd.to_datetime(features["date"]).dt.quarter
    available = features["available"].astype(bool)
    strong = signal["candidate"].astype(bool)
    available_by_quarter = {
        f"q{number}": int((available & quarter.eq(number)).sum())
        for number in range(1, 5)
    }
    strong_by_quarter = {
        f"q{number}": int((strong & quarter.eq(number)).sum())
        for number in range(1, 5)
    }
    strong_total = int(strong.sum())
    long_share = (
        float(signal.loc[strong, "side"].gt(0).mean())
        if strong_total
        else 0.0
    )
    short_share = (
        float(signal.loc[strong, "side"].lt(0).mean())
        if strong_total
        else 0.0
    )
    passes = (
        int(available.sum()) >= cfg.minimum_available_total
        and all(
            value >= cfg.minimum_available_per_quarter
            for value in available_by_quarter.values()
        )
        and all(
            value >= cfg.minimum_strong_per_quarter
            for value in strong_by_quarter.values()
        )
        and min(long_share, short_share) >= cfg.minimum_strong_side_share
    )
    return {
        "available_total": int(available.sum()),
        "available_by_quarter": available_by_quarter,
        "strong_total": strong_total,
        "strong_by_quarter": strong_by_quarter,
        "strong_long_share": long_share,
        "strong_short_share": short_share,
        "passes_availability": bool(passes),
    }


def support_summary(
    schedule: pd.DataFrame,
    cfg: Config,
) -> dict[str, Any]:
    total = len(schedule)
    if total:
        quarter = pd.to_datetime(schedule["signal_date"]).dt.quarter
    else:
        quarter = pd.Series(dtype=np.int8)
    by_quarter = {
        f"q{number}": int(quarter.eq(number).sum())
        for number in range(1, 5)
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


def _prior_schedules(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cclh_signal = cclh.build_signal(frame, cclh.Config())
    cclh_schedule = pdf._quarterly_schedule(cclh_signal, frame)
    if pdf._event_clock_sha256(cclh_schedule) != pdf.CCLH_EVENT_CLOCK_SHA256:
        raise ValueError("CCLH event clock did not replay from shell panel")

    credibility, _ = pdf.load_credibility(pdf.Config())
    if not pd.Series(frame["date"].to_numpy()).equals(
        pd.Series(credibility["date"].to_numpy())
    ):
        raise ValueError("credibility and radial-shell clocks differ")
    pdf_signal = pdf.build_signal(credibility, pdf.Config())
    pdf_schedule = pdf._quarterly_schedule(pdf_signal, credibility)
    if rlwc._pdf_event_clock_sha256(pdf_schedule) != PDF_EVENT_CLOCK_SHA256:
        raise ValueError("PDF-10 event clock did not replay exactly")
    return cclh_signal, cclh_schedule, credibility, pdf_schedule


def _feature_comparators(
    frame: pd.DataFrame,
    credibility: pd.DataFrame,
) -> pd.DataFrame:
    geometry = cclh.cross_collateral_geometry(frame)
    pdf_um = pdf._venue_features(credibility, "um", pdf.Config())
    pdf_cm = pdf._venue_features(credibility, "cm", pdf.Config())
    return pd.DataFrame(
        {
            "cclh_cross_pressure": geometry["cross_pressure"],
            "cclh_cross_elasticity": geometry["cross_elasticity"],
            "pdf10_credibility": 0.5
            * (pdf_um["credibility"] + pdf_cm["credibility"]),
            "pdf10_display": 0.5
            * (pdf_um["display"] + pdf_cm["display"]),
        }
    )


def independence_summary(
    schedule: pd.DataFrame,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    cfg: Config,
) -> dict[str, Any]:
    _, cclh_schedule, credibility, pdf_schedule = _prior_schedules(frame)
    current_positions = schedule["signal_position"].astype(int).tolist()
    event_overlap: dict[str, Any] = {}
    for name, prior in (("cclh", cclh_schedule), ("pdf10", pdf_schedule)):
        overlap = cclh.tolerant_event_jaccard(
            current_positions,
            prior["signal_position"].astype(int).tolist(),
            tolerance_bars=cfg.overlap_tolerance_bars,
        )
        overlap["current_event_match_share"] = (
            overlap["matched_event_count"] / len(schedule)
            if len(schedule)
            else 1.0
        )
        overlap["maximum_allowed_jaccard"] = (
            cfg.maximum_prior_event_jaccard
        )
        overlap["maximum_allowed_current_event_match_share"] = (
            cfg.maximum_prior_event_match_share
        )
        overlap["passes"] = bool(
            overlap["jaccard"] <= cfg.maximum_prior_event_jaccard
            and overlap["current_event_match_share"]
            <= cfg.maximum_prior_event_match_share
        )
        event_overlap[name] = overlap

    comparators = _feature_comparators(frame, credibility)
    score = features["score"].astype(float)
    correlations: dict[str, float] = {}
    for name in comparators:
        paired = features["available"].astype(bool) & np.isfinite(
            comparators[name]
        )
        value = score.loc[paired].corr(
            comparators.loc[paired, name], method="spearman"
        )
        if not np.isfinite(value):
            raise ValueError(f"non-finite RCR independence correlation: {name}")
        correlations[name] = float(value)
    maximum_absolute = max(abs(value) for value in correlations.values())
    passes_feature = maximum_absolute <= cfg.maximum_prior_feature_spearman
    passes_event = all(item["passes"] for item in event_overlap.values())
    return {
        "event_overlap": event_overlap,
        "feature_spearman": correlations,
        "maximum_absolute_feature_spearman": float(maximum_absolute),
        "maximum_allowed_feature_spearman": (
            cfg.maximum_prior_feature_spearman
        ),
        "passes_event_independence": bool(passes_event),
        "passes_feature_independence": bool(passes_feature),
        "passes_independence": bool(passes_event and passes_feature),
    }


def run_support(cfg: Config) -> dict[str, Any]:
    _validate_frozen_config(cfg)
    frame, source = load_shells(cfg)
    features = build_features(frame, cfg)
    signal = build_signal(features, cfg)
    schedule = pdf._quarterly_schedule(signal, frame)
    availability = availability_summary(features, signal, cfg)
    support = support_summary(schedule, cfg)
    independence = independence_summary(schedule, frame, features, cfg)
    passes_all = (
        availability["passes_availability"]
        and support["passes_support"]
        and independence["passes_independence"]
    )
    available_score = features.loc[features["available"], "score"]
    score_quantiles = {
        str(quantile): float(value)
        for quantile, value in available_score.quantile(
            [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        ).items()
    }
    return {
        "protocol": {
            "name": "RCR-144 — Radial Composition Rotation",
            "support_only": True,
            "outcomes_opened_for_rcr144": False,
            "price_or_return_loaded": False,
            "support_rejected": not passes_all,
            "selection_end_exclusive": "2024-01-01 00:00:00",
            "signal_availability": (
                "current and prior 5m shell composition complete; enter "
                "next 5m open"
            ),
            "action_rule": (
                "bid-inward minus ask-inward composition rotation; exact "
                "bid/ask sign symmetry"
            ),
            "candidate_clock": (
                "every completed bar with absolute standardized cross-venue "
                "score at least 2.0"
            ),
            "holding_rule": "144 completed 5m bars; scheduled-open exit",
            "quarter_boundary_policy": (
                "four quarter-contained schedules concatenated; reset flat"
            ),
            "support_parameters_searched": False,
            "mechanisms_compared_outcome_blind": ["DS-RTP", "RCR-144"],
            "source_gap_policy": (
                "current and prior composition rows source-complete; future "
                "shell gaps do not cancel an entered trade"
            ),
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
        },
        "config": asdict(cfg),
        "frozen_artifacts": {
            "preregistration_source": str(PREREGISTRATION_SOURCE),
            "preregistration_source_sha256": _sha256(
                PREREGISTRATION_SOURCE
            ),
            "preregistration_document": str(PREREGISTRATION_DOCUMENT),
            "preregistration_document_sha256": _sha256(
                PREREGISTRATION_DOCUMENT
            ),
            "shell_builder_source": str(SHELL_BUILDER_SOURCE),
            "shell_builder_source_sha256": _sha256(SHELL_BUILDER_SOURCE),
            "standardizer_source": str(STANDARDIZER_SOURCE),
            "standardizer_source_sha256": _sha256(STANDARDIZER_SOURCE),
            "scheduler_source": str(SCHEDULER_SOURCE),
            "scheduler_source_sha256": _sha256(SCHEDULER_SOURCE),
            "cclh_source": str(CCLH_SOURCE),
            "cclh_source_sha256": _sha256(CCLH_SOURCE),
            "cclh_support": str(CCLH_SUPPORT),
            "cclh_support_sha256": _sha256(CCLH_SUPPORT),
            "pdf_source": str(PDF_SOURCE),
            "pdf_source_sha256": _sha256(PDF_SOURCE),
            "pdf_support": str(PDF_SUPPORT),
            "pdf_support_sha256": _sha256(PDF_SUPPORT),
            "pdf_event_clock": str(PDF_EVENT_CLOCK),
            "pdf_event_clock_sha256": _sha256(PDF_EVENT_CLOCK),
            "rlwc_source": str(RLWC_SOURCE),
            "rlwc_source_sha256": _sha256(RLWC_SOURCE),
            "rlwc_support": str(RLWC_SUPPORT),
            "rlwc_support_sha256": _sha256(RLWC_SUPPORT),
            "shell_manifest_sha256": _sha256(cfg.shell_manifest),
        },
        "source": source,
        "feature": {
            "score_quantiles": score_quantiles,
            "normalization": (
                "venue-side median shell shares renormalized to a radial "
                "barycenter; one-bar bid/ask rotation standardized with "
                "strictly lagged rolling median and recursive MAD"
            ),
        },
        "support_calibration": {
            "outcomes_opened_for_rcr144": False,
            "parameters_searched": False,
            "mechanism_selection_used_only_support_and_independence": True,
            "rejected_predecessor": (
                "DS-RTP max prior-feature Spearman 0.610049 exceeded 0.60"
            ),
            "all_parameters_fixed": True,
            "further_support_repairs_allowed": False,
        },
        "availability": availability,
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
                "outcomes_opened_for_rcr144": False,
                "support_rejected": result["protocol"]["support_rejected"],
                "availability": result["availability"],
                "support": result["support"],
                "independence": result["independence"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
