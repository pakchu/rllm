"""Support-only preregistration for CFCF.

CFCF observes the first completed premium-index hour after each synchronized
Binance/Bybit funding settlement.  It trades convergence only when lagged,
robust premium and realized-funding spreads agree.  This module contains no
return, PnL, CAGR, or drawdown calculation.
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

from training.preregister_metaorder_fragmentation_impact_curvature import (
    Config as MarketConfig,
    nonoverlapping_schedule,
)


SELECTION_END = pd.Timestamp("2024-01-01")
SUPPORT_CALIBRATION_GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 0.925, 0.95, 0.975)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_cross_venue_funding_consensus_fracture.py"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/cross-venue-funding-consensus-fracture-preregistration-2026-07-14.md"
)
SCHEDULER_SOURCE = Path(
    "training/preregister_metaorder_fragmentation_impact_curvature.py"
)


@dataclass(frozen=True)
class Config:
    binance_manifest: str = "results/binance_um_aux_btc_2021_2023_manifest.json"
    bybit_manifest: str = "results/bybit_linear_aux_btc_2021_2023_manifest.json"
    market_manifest: str = (
        "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
    )
    output: str = (
        "results/cross_venue_funding_consensus_fracture_support_2026-07-14.json"
    )
    crowding_quantile: float = 0.90
    premium_baseline_hours: int = 2_160
    premium_min_periods: int = 720
    funding_baseline_events: int = 270
    funding_min_periods: int = 90
    crowding_baseline_events: int = 540
    crowding_min_periods: int = 180
    hold_bars: int = 84
    minimum_nonoverlap_total: int = 200
    minimum_nonoverlap_per_year: int = 40
    minimum_nonoverlap_per_2023_half: int = 30
    minimum_side_share: float = 0.25
    minimum_branch_share: float = 0.25


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manifest_file(manifest: dict[str, Any], name: str) -> Path:
    item = manifest.get("files", {}).get(name, {})
    path = Path(item.get("path", ""))
    if not path.is_file() or _sha256(path) != item.get("sha256"):
        raise ValueError(f"cross-venue source hash mismatch: {name}")
    return path


def _validate_complete_grid(
    frame: pd.DataFrame,
    *,
    frequency: str,
    start: str,
    end: str,
    label: str,
) -> None:
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError(f"{label} timestamps are invalid")
    expected = pd.date_range(start, end, freq=frequency, inclusive="left")
    if not pd.DatetimeIndex(frame["date"]).equals(expected):
        raise ValueError(f"{label} is not a complete {frequency} grid")


def load_sources(
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    binance_manifest = json.loads(Path(cfg.binance_manifest).read_text())
    bybit_manifest = json.loads(Path(cfg.bybit_manifest).read_text())
    for label, manifest in (
        ("binance", binance_manifest),
        ("bybit", bybit_manifest),
    ):
        if manifest.get("protocol", {}).get("outcomes_opened") is not False:
            raise ValueError(f"{label} auxiliary manifest opened outcomes")

    binance_funding_path = _manifest_file(binance_manifest, "funding")
    binance_premium_path = _manifest_file(binance_manifest, "premium")
    bybit_funding_path = _manifest_file(bybit_manifest, "funding")
    bybit_premium_path = _manifest_file(bybit_manifest, "premium")

    premium_frames = []
    funding_frames = []
    for venue, premium_path, funding_path in (
        ("binance", binance_premium_path, binance_funding_path),
        ("bybit", bybit_premium_path, bybit_funding_path),
    ):
        premium = pd.read_csv(
            premium_path,
            compression="gzip",
            parse_dates=["date"],
        ).sort_values("date").reset_index(drop=True)
        funding = pd.read_csv(
            funding_path,
            compression="gzip",
            parse_dates=["date"],
        ).sort_values("date").reset_index(drop=True)
        _validate_complete_grid(
            premium,
            frequency="h",
            start="2021-01-01",
            end="2024-01-01",
            label=f"{venue} premium",
        )
        _validate_complete_grid(
            funding,
            frequency="8h",
            start="2021-01-01",
            end="2024-01-01",
            label=f"{venue} funding",
        )
        premium_frames.append(
            premium[["date", "close"]].rename(
                columns={"close": f"{venue}_premium_close"}
            )
        )
        funding_frames.append(
            funding[["date", "funding_rate"]].rename(
                columns={"funding_rate": f"{venue}_funding_rate"}
            )
        )

    premium_panel = premium_frames[0].merge(
        premium_frames[1], on="date", validate="one_to_one"
    )
    funding_panel = funding_frames[0].merge(
        funding_frames[1], on="date", validate="one_to_one"
    )

    market_manifest = json.loads(Path(cfg.market_manifest).read_text())
    if market_manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("market manifest opened outcomes")
    market_path = Path(MarketConfig().market)
    if _sha256(market_path) != market_manifest.get("combined_sha256"):
        raise ValueError("market hash differs from verified manifest")
    market = pd.read_csv(
        market_path,
        compression="gzip",
        parse_dates=["date"],
    )
    market = market.loc[
        market["date"].ge("2021-01-01") & market["date"].lt(SELECTION_END)
    ].reset_index(drop=True)
    _validate_complete_grid(
        market,
        frequency="5min",
        start="2021-01-01",
        end="2024-01-01",
        label="execution market",
    )
    market["quarantined"] = False

    metadata = {
        "binance_manifest_sha256": _sha256(cfg.binance_manifest),
        "bybit_manifest_sha256": _sha256(cfg.bybit_manifest),
        "market_manifest_sha256": _sha256(cfg.market_manifest),
        "market_sha256": _sha256(market_path),
        "range_start": "2021-01-01 00:00:00",
        "range_end": "2023-12-31 23:55:00",
    }
    return premium_panel, funding_panel, market, metadata


def lagged_robust_zscore(
    values: pd.Series,
    *,
    window: int,
    minimum: int,
) -> pd.Series:
    if not 1 <= minimum <= window:
        raise ValueError("robust baseline periods are invalid")
    prior = values.astype(float).shift(1)
    center = prior.rolling(window, min_periods=minimum).median()
    mad = (prior - center).abs().rolling(window, min_periods=minimum).median()
    return ((values.astype(float) - center) / (1.4826 * mad.replace(0.0, np.nan))).clip(
        -12.0, 12.0
    )


def build_settlement_features(
    premium: pd.DataFrame,
    funding: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    hourly = premium.copy()
    hourly["premium_spread"] = (
        hourly["bybit_premium_close"] - hourly["binance_premium_close"]
    )
    hourly["premium_z"] = lagged_robust_zscore(
        hourly["premium_spread"],
        window=cfg.premium_baseline_hours,
        minimum=cfg.premium_min_periods,
    )
    settlements = funding.merge(
        hourly[["date", "premium_spread", "premium_z"]],
        on="date",
        validate="one_to_one",
    )
    settlements["funding_spread"] = (
        settlements["bybit_funding_rate"]
        - settlements["binance_funding_rate"]
    )
    settlements["funding_z"] = lagged_robust_zscore(
        settlements["funding_spread"],
        window=cfg.funding_baseline_events,
        minimum=cfg.funding_min_periods,
    )
    same_direction = (
        np.sign(settlements["premium_z"])
        == np.sign(settlements["funding_z"])
    ) & settlements["premium_z"].ne(0.0) & settlements["funding_z"].ne(0.0)
    settlements["spread_agreement"] = same_direction
    settlements["crowding_score"] = (
        np.sign(settlements["premium_z"])
        * np.sqrt(
            settlements["premium_z"].abs()
            * settlements["funding_z"].abs()
        )
    )
    settlements["signal_date"] = settlements["date"] + pd.Timedelta(minutes=55)
    return settlements


def classify_settlements(
    settlements: pd.DataFrame,
    cfg: Config,
    *,
    crowding_quantile: float | None = None,
) -> pd.DataFrame:
    quantile = cfg.crowding_quantile if crowding_quantile is None else crowding_quantile
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("crowding quantile must be in [0, 1]")
    available = settlements[["premium_z", "funding_z", "crowding_score"]].notna().all(
        axis=1
    )
    baseline = (
        settlements["crowding_score"]
        .abs()
        .where(available)
        .shift(1)
        .rolling(
            cfg.crowding_baseline_events,
            min_periods=cfg.crowding_min_periods,
        )
        .quantile(quantile)
    )
    candidate = (
        available
        & settlements["spread_agreement"].astype(bool)
        & settlements["crowding_score"].abs().ge(baseline)
    )
    side = pd.Series(0, index=settlements.index, dtype=np.int8)
    side.loc[candidate] = -np.sign(
        settlements.loc[candidate, "crowding_score"]
    ).astype(np.int8)
    branch = pd.Series("none", index=settlements.index, dtype="string")
    branch.loc[candidate & settlements["crowding_score"].gt(0.0)] = "bybit_rich"
    branch.loc[candidate & settlements["crowding_score"].lt(0.0)] = "bybit_cheap"
    return pd.DataFrame(
        {
            "date": settlements["date"],
            "signal_date": settlements["signal_date"],
            "candidate": candidate,
            "premium_z": settlements["premium_z"],
            "funding_z": settlements["funding_z"],
            "crowding_score": settlements["crowding_score"],
            "crowding_baseline": baseline,
            "side": side,
            "branch": branch,
            "hold_bars": np.where(side.ne(0), cfg.hold_bars, 0).astype(np.int16),
        }
    )


def project_to_market(state: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    index = pd.Series(np.arange(len(market), dtype=np.int64), index=market["date"])
    signal_positions = index.loc[state["signal_date"]].to_numpy(np.int64)
    origin_positions = index.loc[state["date"]].to_numpy(np.int64)
    signal = pd.DataFrame(
        {
            "date": market["date"],
            "side": np.zeros(len(market), dtype=np.int8),
            "hold_bars": np.zeros(len(market), dtype=np.int16),
            "branch": pd.Series("none", index=market.index, dtype="string"),
            "origin_position": np.full(len(market), -1, dtype=np.int64),
        }
    )
    signal.loc[signal_positions, "side"] = state["side"].to_numpy(np.int8)
    signal.loc[signal_positions, "hold_bars"] = state["hold_bars"].to_numpy(np.int16)
    signal.loc[signal_positions, "branch"] = state["branch"].astype(str).to_numpy()
    signal.loc[signal_positions, "origin_position"] = origin_positions
    return signal


def nonoverlapping_cfcf_schedule(
    signal: pd.DataFrame,
    market: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    period = market["date"].ge(start) & market["date"].lt(end)
    origins = signal["origin_position"].to_numpy(np.int64)
    valid_origin = np.zeros(len(signal), dtype=bool)
    valid = origins >= 0
    valid_origin[valid] = period.to_numpy(bool)[origins[valid]]
    eligible = signal.copy()
    eligible.loc[~valid_origin, "side"] = 0
    schedule = nonoverlapping_schedule(eligible, market, start=start, end=end)
    if not schedule.empty:
        schedule["origin_position"] = [
            int(signal.loc[position, "origin_position"])
            for position in schedule["signal_position"]
        ]
    return schedule


def support_summary(
    signal: pd.DataFrame,
    market: pd.DataFrame,
    cfg: Config,
) -> dict[str, Any]:
    annual = {
        str(year): nonoverlapping_cfcf_schedule(
            signal,
            market,
            start=f"{year}-01-01",
            end=f"{year + 1}-01-01",
        )
        for year in range(2021, 2024)
    }
    schedule = pd.concat(annual.values(), ignore_index=True)
    total = len(schedule)
    h1 = len(
        nonoverlapping_cfcf_schedule(
            signal, market, start="2023-01-01", end="2023-07-01"
        )
    )
    h2 = len(
        nonoverlapping_cfcf_schedule(
            signal, market, start="2023-07-01", end="2024-01-01"
        )
    )
    by_year = {year: len(rows) for year, rows in annual.items()}
    long_share = float(schedule["side"].gt(0).mean()) if total else 0.0
    short_share = float(schedule["side"].lt(0).mean()) if total else 0.0
    rich_share = float(schedule["branch"].eq("bybit_rich").mean()) if total else 0.0
    cheap_share = float(schedule["branch"].eq("bybit_cheap").mean()) if total else 0.0
    passes = (
        total >= cfg.minimum_nonoverlap_total
        and all(value >= cfg.minimum_nonoverlap_per_year for value in by_year.values())
        and h1 >= cfg.minimum_nonoverlap_per_2023_half
        and h2 >= cfg.minimum_nonoverlap_per_2023_half
        and min(long_share, short_share) >= cfg.minimum_side_share
        and min(rich_share, cheap_share) >= cfg.minimum_branch_share
    )
    return {
        "nonoverlap_total": int(total),
        "by_year": by_year,
        "2023_h1": int(h1),
        "2023_h2": int(h2),
        "long_share": long_share,
        "short_share": short_share,
        "bybit_rich_share": rich_share,
        "bybit_cheap_share": cheap_share,
        "passes_support": bool(passes),
    }


def _selected_support_quantile(trials: list[dict[str, Any]]) -> float | None:
    passing = [
        float(trial["crowding_quantile"])
        for trial in trials
        if trial["passes_support"]
    ]
    return max(passing) if passing else None


def run_support(cfg: Config) -> dict[str, Any]:
    premium, funding, market, source = load_sources(cfg)
    settlements = build_settlement_features(premium, funding, cfg)
    trials: list[dict[str, Any]] = []
    selected_state: pd.DataFrame | None = None
    selected_signal: pd.DataFrame | None = None
    selected_support: dict[str, Any] | None = None
    for quantile in SUPPORT_CALIBRATION_GRID:
        state = classify_settlements(
            settlements,
            cfg,
            crowding_quantile=quantile,
        )
        signal = project_to_market(state, market)
        support = support_summary(signal, market, cfg)
        trials.append(
            {
                "crowding_quantile": quantile,
                "raw_candidate_count": int(state["candidate"].sum()),
                **support,
            }
        )
        if quantile == cfg.crowding_quantile:
            selected_state = state
            selected_signal = signal
            selected_support = support
    selected = _selected_support_quantile(trials)
    if selected != cfg.crowding_quantile:
        raise ValueError("configured CFCF quantile violates support stopping rule")
    if selected_state is None or selected_signal is None or selected_support is None:
        raise AssertionError("configured CFCF support was not evaluated")

    selected_schedule = pd.concat(
        [
            nonoverlapping_cfcf_schedule(
                selected_signal,
                market,
                start=f"{year}-01-01",
                end=f"{year + 1}-01-01",
            )
            for year in range(2021, 2024)
        ],
        ignore_index=True,
    )
    return {
        "protocol": {
            "name": "CFCF — Cross-Venue Funding Consensus Fracture",
            "support_only": True,
            "outcomes_opened_for_cfcf": False,
            "selection_end_exclusive": "2024-01-01 00:00:00",
            "event_clock": (
                "first completed premium hour after each synchronized 8h "
                "funding settlement"
            ),
            "signal_availability": (
                "the :55-labeled 5m slot is the completed premium hour's "
                "final slot; enter next 5m open"
            ),
            "direction": "fade the venue-consensus crowding spread toward convergence",
            "candidate_clock": (
                "fixed before any outcome; both branches share one "
                "settlement clock"
            ),
            "holding_rule": "84 completed 5m bars; exit at next funding boundary open",
            "source_gap_policy": (
                "every auxiliary and execution grid must be complete; any "
                "missing or duplicate timestamp fails closed, so no accepted "
                "row is quarantined"
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
            "scheduler_source": str(SCHEDULER_SOURCE),
            "scheduler_source_sha256": _sha256(SCHEDULER_SOURCE),
            "binance_manifest_sha256": _sha256(cfg.binance_manifest),
            "bybit_manifest_sha256": _sha256(cfg.bybit_manifest),
            "market_manifest_sha256": _sha256(cfg.market_manifest),
        },
        "source": source,
        "support_calibration": {
            "outcomes_opened_for_cfcf": False,
            "tested_crowding_quantiles": list(SUPPORT_CALIBRATION_GRID),
            "all_other_parameters_fixed": True,
            "stopping_rule": "highest tested quantile passing every frozen support floor",
            "selected_crowding_quantile": selected,
            "further_support_repairs_allowed": False,
            "trials": trials,
        },
        "raw_candidate_count": int(selected_state["candidate"].sum()),
        "scheduled_branch_counts": {
            name: int(value)
            for name, value in selected_schedule["branch"].value_counts().items()
        },
        "support": selected_support,
        "all_support_gates_pass": bool(selected_support["passes_support"]),
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
                "outcomes_opened_for_cfcf": False,
                "selected_crowding_quantile": result["support_calibration"][
                    "selected_crowding_quantile"
                ],
                "support": result["support"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
