"""Outcome-blind support preregistration for RIFT-96.

RIFT infers persistent upward execution pressure from a two-completed-bar
sequence.  This module may inspect causal feature incidence and clock overlap,
but it contains no return, future OHLC, CAGR, or MDD calculation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_cash_sponsored_perp_rejection as cspr


SELECTION_END = pd.Timestamp("2024-01-01")
SUPPORT_QUANTILES = (0.80, 0.85, 0.90, 0.925, 0.95, 0.975)
CONTROL_ACTIONS = {
    "primary": 1,
    "direction_flip": -1,
    "same_bar_static": 1,
    "no_path_quality": 1,
    "no_derivatives_crowd": 1,
    "centroid_free_momentum": 1,
    "spot_only": 1,
    "stale_setup_1h": 1,
    "stale_setup_24h": 1,
    "signal_delay_1bar": 1,
    "simple_two_bar_momentum": 1,
}


@dataclass(frozen=True)
class Config:
    spot_features: str = (
        "data/binance_spot_kline_microstructure_btc_2020_2023/"
        "BTCUSDT_spot_kline_microstructure_5m_2020-01_2023-12.csv.gz"
    )
    spot_manifest: str = (
        "data/binance_spot_kline_microstructure_btc_2020_2023/build_manifest.json"
    )
    perp_features: str = (
        "data/binance_um_aggtrade_microstructure_btc_2020_2023/"
        "BTCUSDT_aggtrade_5m_2020-01-01_2023-12-31.csv.gz"
    )
    perp_manifest: str = (
        "data/binance_um_aggtrade_microstructure_btc_2020_2023/build_manifest.json"
    )
    market: str = (
        "data/binance_um_kline_reference_btc_2020_2023/"
        "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
    )
    market_manifest: str = (
        "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
    )
    output: str = "results/refill_inference_flow_topology_support_2026-07-14.json"
    baseline_bars: int = 8_640
    baseline_min_periods: int = 2_016
    post_gap_quarantine_bars: int = 24
    hold_bars: int = 96
    minimum_perp_agg_trade_count: int = 64
    minimum_nonoverlap_total: int = 300
    minimum_nonoverlap_per_year: int = 40
    minimum_nonoverlap_per_2023_half: int = 30
    maximum_same_bar_jaccard: float = 0.05
    maximum_no_path_jaccard: float = 0.40
    maximum_no_crowd_jaccard: float = 0.20
    maximum_centroid_free_jaccard: float = 0.75
    maximum_spot_only_jaccard: float = 0.20
    maximum_stale_jaccard: float = 0.10
    maximum_delay_jaccard: float = 0.05
    maximum_cspr_jaccard: float = 0.01


def load_causal_frame(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    return cspr.load_causal_frame(cfg)


def prior_quantile(
    values: pd.Series,
    clean: pd.Series,
    *,
    quantile: float,
    cfg: Config,
) -> pd.Series:
    return cspr.prior_quantile(
        values,
        clean,
        quantile=quantile,
        window=cfg.baseline_bars,
        min_periods=cfg.baseline_min_periods,
    )


def _components(frame: pd.DataFrame) -> dict[str, pd.Series]:
    mark = pd.to_numeric(
        frame["spot_close_vs_centroid_mid_bp"], errors="coerce"
    )
    spot_return = pd.to_numeric(frame["spot_micro_log_return"], errors="coerce")
    perp_return = pd.to_numeric(frame["micro_log_return"], errors="coerce")
    spot_flow = pd.to_numeric(
        frame["spot_signed_quote_notional"], errors="coerce"
    )
    perp_flow = pd.to_numeric(frame["signed_quote_notional"], errors="coerce")
    perp_event = pd.to_numeric(frame["signed_event_imbalance"], errors="coerce")
    price_efficiency = pd.to_numeric(
        frame["spot_minute_price_path_efficiency"], errors="coerce"
    ).clip(0.0, 1.0)
    flow_efficiency = pd.to_numeric(
        frame["spot_minute_flow_path_efficiency"], errors="coerce"
    ).clip(0.0, 1.0)
    alignment = (
        pd.to_numeric(
            frame["spot_minute_flow_price_alignment"], errors="coerce"
        ).clip(-1.0, 1.0)
        + 1.0
    ) / 2.0
    flip_persistence = 1.0 - pd.to_numeric(
        frame["spot_minute_flow_sign_flip_rate"], errors="coerce"
    ).clip(0.0, 1.0)
    event_hhi = pd.to_numeric(frame["event_notional_hhi"], errors="coerce").clip(
        lower=0.0
    )
    burst = (
        pd.to_numeric(frame["interarrival_burstiness"], errors="coerce").clip(
            -1.0, 1.0
        )
        + 1.0
    ) / 2.0
    path_quality = (
        price_efficiency * flow_efficiency * alignment * flip_persistence
    )
    crowd_structure = np.sqrt(event_hhi) * burst
    return {
        "mark": mark,
        "spot_return": spot_return,
        "perp_return": perp_return,
        "spot_flow": spot_flow,
        "perp_flow": perp_flow,
        "perp_event": perp_event,
        "price_efficiency": price_efficiency,
        "flow_efficiency": flow_efficiency,
        "alignment": alignment,
        "flip_persistence": flip_persistence,
        "event_hhi": event_hhi,
        "burst": burst,
        "path_quality": path_quality,
        "crowd_structure": crowd_structure,
    }


def _finite(values: dict[str, pd.Series]) -> pd.Series:
    return pd.concat(list(values.values()), axis=1).notna().all(axis=1)


def _scores(values: dict[str, pd.Series]) -> dict[str, pd.Series]:
    positive_mark = values["mark"].clip(lower=0.0)
    positive_return_bp = values["spot_return"].clip(lower=0.0) * 10_000.0
    return {
        "primary": (
            positive_mark
            * values["path_quality"]
            * values["crowd_structure"]
        ),
        "no_path": positive_mark * values["crowd_structure"],
        "no_crowd": positive_mark * values["path_quality"],
        "centroid_free": (
            positive_return_bp
            * values["path_quality"]
            * values["crowd_structure"]
        ),
        "spot_only": positive_mark * values["path_quality"],
    }


def classify_sequences(
    frame: pd.DataFrame,
    cfg: Config,
    *,
    quantile: float,
) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, pd.Series]]:
    values = _components(frame)
    scores = _scores(values)
    finite = _finite(values)
    no_path_finite = pd.concat(
        [
            values["mark"],
            values["spot_return"],
            values["perp_return"],
            values["spot_flow"],
            values["perp_flow"],
            values["perp_event"],
            values["event_hhi"],
            values["burst"],
            values["crowd_structure"],
        ],
        axis=1,
    ).notna().all(axis=1)
    no_crowd_finite = pd.concat(
        [
            values["mark"],
            values["spot_return"],
            values["perp_return"],
            values["spot_flow"],
            values["perp_flow"],
            values["perp_event"],
            values["price_efficiency"],
            values["flow_efficiency"],
            values["alignment"],
            values["flip_persistence"],
            values["path_quality"],
        ],
        axis=1,
    ).notna().all(axis=1)
    centroid_free_finite = pd.concat(
        [
            values["spot_return"],
            values["perp_return"],
            values["spot_flow"],
            values["perp_flow"],
            values["perp_event"],
            values["price_efficiency"],
            values["flow_efficiency"],
            values["alignment"],
            values["flip_persistence"],
            values["event_hhi"],
            values["burst"],
            values["path_quality"],
            values["crowd_structure"],
        ],
        axis=1,
    ).notna().all(axis=1)
    spot_finite = pd.concat(
        [
            values["mark"],
            values["spot_return"],
            values["spot_flow"],
            values["price_efficiency"],
            values["flow_efficiency"],
            values["alignment"],
            values["flip_persistence"],
            values["path_quality"],
        ],
        axis=1,
    ).notna().all(axis=1)
    perp_clean = ~frame["perp_quarantined"].astype(bool)
    spot_clean = ~frame["spot_quarantined"].astype(bool)
    joint_clean = ~frame["quarantined"].astype(bool)
    active = pd.to_numeric(frame["agg_trade_count"], errors="coerce").ge(
        cfg.minimum_perp_agg_trade_count
    )

    directional_pressure = (
        joint_clean
        & active
        & finite
        & values["mark"].gt(0.0)
        & values["spot_return"].gt(0.0)
        & values["perp_return"].gt(0.0)
        & values["spot_flow"].gt(0.0)
        & values["perp_flow"].gt(0.0)
        & values["perp_event"].ge(0.0)
    )
    centroid_free_pressure = (
        joint_clean
        & active
        & centroid_free_finite
        & values["spot_return"].gt(0.0)
        & values["perp_return"].gt(0.0)
        & values["spot_flow"].gt(0.0)
        & values["perp_flow"].gt(0.0)
        & values["perp_event"].ge(0.0)
    )
    spot_pressure = (
        spot_clean
        & spot_finite
        & values["mark"].gt(0.0)
        & values["spot_return"].gt(0.0)
        & values["spot_flow"].gt(0.0)
    )
    no_path_pressure = (
        joint_clean
        & active
        & no_path_finite
        & values["mark"].gt(0.0)
        & values["spot_return"].gt(0.0)
        & values["perp_return"].gt(0.0)
        & values["spot_flow"].gt(0.0)
        & values["perp_flow"].gt(0.0)
        & values["perp_event"].ge(0.0)
    )
    no_crowd_pressure = (
        joint_clean
        & active
        & no_crowd_finite
        & values["mark"].gt(0.0)
        & values["spot_return"].gt(0.0)
        & values["perp_return"].gt(0.0)
        & values["spot_flow"].gt(0.0)
        & values["perp_flow"].gt(0.0)
        & values["perp_event"].ge(0.0)
    )

    score_thresholds = {
        "primary": prior_quantile(
            scores["primary"], joint_clean, quantile=quantile, cfg=cfg
        ),
        "no_path": prior_quantile(
            scores["no_path"], joint_clean, quantile=quantile, cfg=cfg
        ),
        "no_crowd": prior_quantile(
            scores["no_crowd"], joint_clean, quantile=quantile, cfg=cfg
        ),
        "centroid_free": prior_quantile(
            scores["centroid_free"], joint_clean, quantile=quantile, cfg=cfg
        ),
        "spot_only": prior_quantile(
            scores["spot_only"], spot_clean, quantile=quantile, cfg=cfg
        ),
    }
    setup = directional_pressure & scores["primary"].ge(
        score_thresholds["primary"]
    )
    setups = {
        "primary": setup,
        "no_path": no_path_pressure
        & scores["no_path"].ge(score_thresholds["no_path"]),
        "no_crowd": no_crowd_pressure
        & scores["no_crowd"].ge(score_thresholds["no_crowd"]),
        "centroid_free": centroid_free_pressure
        & scores["centroid_free"].ge(score_thresholds["centroid_free"]),
        "spot_only": spot_pressure
        & scores["spot_only"].ge(score_thresholds["spot_only"]),
    }

    event_hhi_median = prior_quantile(
        values["event_hhi"], perp_clean, quantile=0.5, cfg=cfg
    )
    burst_median = prior_quantile(
        values["burst"], perp_clean, quantile=0.5, cfg=cfg
    )
    price_efficiency_median = prior_quantile(
        values["price_efficiency"], spot_clean, quantile=0.5, cfg=cfg
    )
    confirmation = (
        joint_clean
        & active
        & finite
        & values["mark"].gt(0.0)
        & values["spot_return"].ge(0.0)
        & values["perp_return"].ge(0.0)
        & values["event_hhi"].ge(event_hhi_median)
        & values["burst"].ge(burst_median)
        & values["price_efficiency"].ge(price_efficiency_median)
    )
    confirmation_no_path = (
        joint_clean
        & active
        & no_path_finite
        & values["mark"].gt(0.0)
        & values["spot_return"].ge(0.0)
        & values["perp_return"].ge(0.0)
        & values["event_hhi"].ge(event_hhi_median)
        & values["burst"].ge(burst_median)
    )
    confirmation_no_crowd = (
        joint_clean
        & active
        & no_crowd_finite
        & values["mark"].gt(0.0)
        & values["spot_return"].ge(0.0)
        & values["perp_return"].ge(0.0)
        & values["price_efficiency"].ge(price_efficiency_median)
    )
    confirmation_centroid_free = (
        joint_clean
        & active
        & centroid_free_finite
        & values["spot_return"].ge(0.0)
        & values["perp_return"].ge(0.0)
        & values["event_hhi"].ge(event_hhi_median)
        & values["burst"].ge(burst_median)
        & values["price_efficiency"].ge(price_efficiency_median)
    )
    confirmation_spot_only = (
        spot_clean
        & spot_finite
        & values["mark"].gt(0.0)
        & values["spot_return"].ge(0.0)
        & values["price_efficiency"].ge(price_efficiency_median)
    )

    primary = setups["primary"].shift(1, fill_value=False) & confirmation
    controls = {
        "primary": primary,
        "direction_flip": primary,
        "same_bar_static": setups["primary"],
        "no_path_quality": setups["no_path"].shift(
            1, fill_value=False
        )
        & confirmation_no_path,
        "no_derivatives_crowd": setups["no_crowd"].shift(
            1, fill_value=False
        )
        & confirmation_no_crowd,
        "centroid_free_momentum": setups["centroid_free"].shift(
            1, fill_value=False
        )
        & confirmation_centroid_free,
        "spot_only": setups["spot_only"].shift(1, fill_value=False)
        & confirmation_spot_only,
        "stale_setup_1h": setups["primary"].shift(12, fill_value=False)
        & confirmation,
        "stale_setup_24h": setups["primary"].shift(288, fill_value=False)
        & confirmation,
        "signal_delay_1bar": primary.shift(1, fill_value=False),
        "simple_two_bar_momentum": (
            joint_clean
            & active
            & values["spot_return"].notna()
            & values["perp_return"].notna()
            & values["spot_return"].gt(0.0)
            & values["perp_return"].gt(0.0)
        ).shift(1, fill_value=False)
        & joint_clean
        & active
        & values["spot_return"].notna()
        & values["perp_return"].notna()
        & values["spot_return"].gt(0.0)
        & values["perp_return"].gt(0.0),
    }
    signal = pd.DataFrame(
        {
            "date": frame["date"],
            "side": np.where(primary, 1, 0).astype(np.int8),
            "branch": np.where(primary, "rift96", "none"),
            "hold_bars": np.where(primary, cfg.hold_bars, 0).astype(np.int16),
            "setup_score": scores["primary"],
            "setup_threshold": score_thresholds["primary"],
            "setup_previous_bar": setups["primary"].shift(
                1, fill_value=False
            ),
            "confirmation": confirmation,
            "quarantined": frame["quarantined"].astype(bool),
        }
    )
    diagnostics = {
        **scores,
        **{f"threshold_{name}": value for name, value in score_thresholds.items()},
        "directional_pressure": directional_pressure,
        "confirmation": confirmation,
    }
    return signal, controls, diagnostics


def nonoverlapping_schedule(
    signal: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return cspr.nonoverlapping_schedule(signal, frame)


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
    h1 = int(
        ((dates >= "2023-01-01") & (dates < "2023-07-01")).sum()
    )
    h2 = int(
        ((dates >= "2023-07-01") & (dates < "2024-01-01")).sum()
    )
    total = int(len(schedule))
    passes = (
        total >= cfg.minimum_nonoverlap_total
        and all(
            count >= cfg.minimum_nonoverlap_per_year
            for count in by_year.values()
        )
        and h1 >= cfg.minimum_nonoverlap_per_2023_half
        and h2 >= cfg.minimum_nonoverlap_per_2023_half
        and (schedule["side"].eq(1).all() if total else False)
    )
    return {
        "nonoverlap_total": total,
        "by_year": by_year,
        "2023_h1": h1,
        "2023_h2": h2,
        "long_only": bool(schedule["side"].eq(1).all()) if total else False,
        "passes_count_support": bool(passes),
    }


def _jaccard(left: pd.Series, right: pd.Series) -> float:
    left = left.fillna(False).astype(bool)
    right = right.fillna(False).astype(bool)
    union = int((left | right).sum())
    return float((left & right).sum() / union) if union else 0.0


def run_support(cfg: Config) -> dict[str, Any]:
    frame, source = load_causal_frame(cfg)
    _, cspr_controls = cspr.classify_events(frame, cfg, quantile=0.5)
    rows: list[dict[str, Any]] = []
    for quantile in SUPPORT_QUANTILES:
        signal, controls, _ = classify_sequences(frame, cfg, quantile=quantile)
        schedule = nonoverlapping_schedule(signal, frame)
        support = _support(schedule, cfg)
        primary = controls["primary"]
        overlap = {
            name: _jaccard(primary, mask)
            for name, mask in controls.items()
            if name not in {"primary", "direction_flip"}
        }
        overlap["cspr_primary"] = _jaccard(primary, cspr_controls["primary"])
        novelty = (
            overlap["same_bar_static"] <= cfg.maximum_same_bar_jaccard
            and overlap["no_path_quality"] <= cfg.maximum_no_path_jaccard
            and overlap["no_derivatives_crowd"]
            <= cfg.maximum_no_crowd_jaccard
            and overlap["centroid_free_momentum"]
            <= cfg.maximum_centroid_free_jaccard
            and overlap["spot_only"] <= cfg.maximum_spot_only_jaccard
            and overlap["stale_setup_1h"] <= cfg.maximum_stale_jaccard
            and overlap["stale_setup_24h"] <= cfg.maximum_stale_jaccard
            and overlap["signal_delay_1bar"] <= cfg.maximum_delay_jaccard
            and overlap["cspr_primary"] <= cfg.maximum_cspr_jaccard
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
                "control_jaccard": overlap,
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
            "name": "RIFT-96 — Refill Inference from Flow Topology",
            "support_only": True,
            "outcomes_opened": False,
            "selection_end_exclusive": str(SELECTION_END),
            "economic_claim": (
                "persistent completed-bar execution pressure consistent with, "
                "but not direct observation of, weak offer replenishment"
            ),
            "entry": "next USD-M 5m open after setup plus one-bar confirmation",
            "hold": f"fixed {cfg.hold_bars} bars",
            "support_selection": (
                "highest tested setup-score quantile passing every frozen "
                "count and clock-novelty floor"
            ),
            "control_actions": CONTROL_ACTIONS,
        },
        "config": asdict(cfg),
        "source": source,
        "support_grid": rows,
        "support_decision": "pass" if selected else "reject_before_returns",
        "selected_quantile": selected["quantile"] if selected else None,
        "selected_support": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=Config.output)
    args = parser.parse_args()
    cfg = Config(output=args.output)
    result = run_support(cfg)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
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
