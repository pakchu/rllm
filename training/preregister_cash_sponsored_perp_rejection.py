"""Support-only preregistration for cash-sponsored perpetual rejection (CSPR).

The module joins verified Binance Spot one-minute-kline microstructure with the
verified USD-M aggregate-trade topology.  It computes candidate incidence and
negative-control overlap only; no future return, OHLC path, or profitability
metric is available in this file.
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

from training import preregister_metaorder_fragmentation_impact_curvature as mfic


SELECTION_END = pd.Timestamp("2024-01-01")
SUPPORT_QUANTILES = (0.50, 0.60, 0.70, 0.80, 0.90)


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
    output: str = "results/cash_sponsored_perp_rejection_support_2026-07-14.json"
    baseline_bars: int = 8_640
    baseline_min_periods: int = 2_016
    post_gap_quarantine_bars: int = 24
    hold_bars: int = 12
    minimum_perp_agg_trade_count: int = 64
    minimum_nonoverlap_total: int = 300
    minimum_nonoverlap_per_year: int = 40
    minimum_nonoverlap_per_2023_half: int = 30
    minimum_side_share: float = 0.25
    maximum_centroid_ablation_retention: float = 0.80
    maximum_perp_event_ablation_retention: float = 0.80
    maximum_lag_placebo_jaccard: float = 0.25


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(path: Path, *, label: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError(f"{label} manifest does not preserve unopened outcomes")
    return manifest


def load_causal_frame(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    base_cfg = mfic.Config(
        features=cfg.perp_features,
        feature_manifest=cfg.perp_manifest,
        market=cfg.market,
        market_manifest=cfg.market_manifest,
        post_gap_quarantine_bars=cfg.post_gap_quarantine_bars,
    )
    frame, metadata = mfic.load_causal_frame(base_cfg)
    spot_path = Path(cfg.spot_features)
    spot_manifest_path = Path(cfg.spot_manifest)
    spot_manifest = _manifest(spot_manifest_path, label="spot")
    if _sha256(spot_path) != spot_manifest.get("combined_sha256"):
        raise ValueError("spot feature hash does not match manifest")
    spot = pd.read_csv(spot_path, compression="gzip", parse_dates=["date"])
    if spot["date"].max() >= SELECTION_END:
        raise ValueError("spot support frame contains 2024+ rows")
    if spot["date"].duplicated().any() or not spot["date"].is_monotonic_increasing:
        raise ValueError("spot timestamps are duplicate or unordered")
    spot = spot.rename(
        columns={column: f"spot_{column}" for column in spot.columns if column != "date"}
    )
    frame = frame.merge(spot, on="date", how="left", validate="one_to_one")
    frame["perp_quarantined"] = frame["quarantined"].astype(bool)
    spot_available = frame["spot_source_complete"].eq(True).fillna(False)
    spot_quarantined = mfic.quarantine_mask(
        spot_available,
        pd.Series(False, index=frame.index),
        cfg.post_gap_quarantine_bars,
    )
    frame["spot_available"] = spot_available
    frame["spot_quarantined"] = spot_quarantined
    frame["quarantined"] = frame["quarantined"].astype(bool) | spot_quarantined
    metadata.update(
        {
            "spot_feature_sha256": _sha256(spot_path),
            "spot_manifest_sha256": _sha256(spot_manifest_path),
            "spot_missing_or_incomplete_bars": int((~spot_available).sum()),
            "spot_quarantined_bars": int(spot_quarantined.sum()),
            "joint_quarantined_bars": int(frame["quarantined"].sum()),
        }
    )
    return frame, metadata


def prior_quantile(
    values: pd.Series,
    clean: pd.Series,
    *,
    quantile: float,
    window: int,
    min_periods: int,
) -> pd.Series:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if not 1 <= min_periods <= window:
        raise ValueError("baseline periods are invalid")
    return (
        pd.to_numeric(values, errors="coerce")
        .where(clean.astype(bool))
        .shift(1)
        .rolling(window, min_periods=min_periods)
        .quantile(quantile)
    )


def _directions(frame: pd.DataFrame) -> dict[str, pd.Series]:
    spot_flow = pd.to_numeric(frame["spot_signed_quote_notional"], errors="coerce")
    spot_return = pd.to_numeric(frame["spot_micro_log_return"], errors="coerce")
    perp_flow = pd.to_numeric(frame["signed_quote_notional"], errors="coerce")
    perp_event = pd.to_numeric(frame["signed_event_imbalance"], errors="coerce")
    perp_return = pd.to_numeric(frame["micro_log_return"], errors="coerce")
    return {
        "spot_flow": spot_flow,
        "spot_return": spot_return,
        "perp_flow": perp_flow,
        "perp_event": perp_event,
        "perp_return": perp_return,
        "side": pd.Series(np.sign(spot_flow), index=frame.index, dtype=float),
    }


def _centroid_confirmation(frame: pd.DataFrame, side: pd.Series) -> pd.Series:
    buyer = pd.to_numeric(frame["spot_buyer_execution_centroid"], errors="coerce")
    seller = pd.to_numeric(frame["spot_seller_execution_centroid"], errors="coerce")
    close = pd.to_numeric(frame["spot_close"], errors="coerce")
    long = side.gt(0.0) & buyer.lt(seller) & close.gt(pd.concat([buyer, seller], axis=1).max(axis=1))
    short = side.lt(0.0) & buyer.gt(seller) & close.lt(pd.concat([buyer, seller], axis=1).min(axis=1))
    return (long | short).fillna(False)


def classify_events(
    frame: pd.DataFrame,
    cfg: Config,
    *,
    quantile: float,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    values = _directions(frame)
    side = values["side"]
    perp_clean = ~frame["perp_quarantined"].astype(bool)
    spot_clean = ~frame["spot_quarantined"].astype(bool)
    clean = perp_clean & spot_clean & ~frame["quarantined"].astype(bool)
    spot_strength = pd.to_numeric(frame["spot_flow_coherence"], errors="coerce")
    perp_strength = pd.to_numeric(frame["flow_coherence"], errors="coerce")
    spot_threshold = prior_quantile(
        spot_strength,
        spot_clean,
        quantile=quantile,
        window=cfg.baseline_bars,
        min_periods=cfg.baseline_min_periods,
    )
    perp_threshold = prior_quantile(
        perp_strength,
        perp_clean,
        quantile=quantile,
        window=cfg.baseline_bars,
        min_periods=cfg.baseline_min_periods,
    )
    spot_finite = pd.concat(
        [
            values["spot_flow"],
            values["spot_return"],
            spot_strength,
            spot_threshold,
        ],
        axis=1,
    ).notna().all(axis=1)
    perp_finite = pd.concat(
        [
            values["perp_flow"],
            values["perp_event"],
            values["perp_return"],
            perp_strength,
            perp_threshold,
        ],
        axis=1,
    ).notna().all(axis=1)
    finite = spot_finite & perp_finite
    active_perp = pd.to_numeric(frame["agg_trade_count"], errors="coerce").ge(
        cfg.minimum_perp_agg_trade_count
    )
    strong_spot = spot_strength.ge(spot_threshold)
    strong_perp = perp_strength.ge(perp_threshold)
    spot_accepted = side.mul(values["spot_return"]).gt(0.0)
    perp_price_follows_cash = side.mul(values["perp_return"]).gt(0.0)
    perp_capital_rejected = side.mul(values["perp_flow"]).lt(0.0)
    perp_crowd_rejected = side.mul(values["perp_event"]).lt(0.0)
    centroid = _centroid_confirmation(frame, side)
    base = clean & finite & active_perp & side.ne(0.0) & strong_spot & strong_perp
    no_centroid = (
        base
        & spot_accepted
        & perp_price_follows_cash
        & perp_capital_rejected
        & perp_crowd_rejected
    )
    no_perp_event = (
        base
        & spot_accepted
        & perp_price_follows_cash
        & perp_capital_rejected
        & centroid
    )
    primary = no_centroid & centroid

    spot_only = spot_clean & spot_finite & side.ne(0.0) & strong_spot & spot_accepted & centroid
    perp_side = pd.Series(np.sign(values["perp_return"]), index=frame.index, dtype=float)
    perp_only = (
        perp_clean
        & perp_finite
        & active_perp
        & perp_side.ne(0.0)
        & strong_perp
        & perp_side.mul(values["perp_flow"]).lt(0.0)
        & perp_side.mul(values["perp_event"]).lt(0.0)
    )
    role_swap = (
        base
        & side.mul(values["spot_return"]).lt(0.0)
        & side.mul(values["perp_return"]).lt(0.0)
        & side.mul(values["perp_flow"]).gt(0.0)
        & side.mul(values["perp_event"]).gt(0.0)
        & centroid
    )

    controls: dict[str, pd.Series] = {
        "primary": primary,
        "direction_flip": primary,
        "signal_delay_1bar": primary.shift(1, fill_value=False),
        "no_centroid": no_centroid,
        "no_perp_event_confirmation": no_perp_event,
        "spot_only": spot_only,
        "perp_only": perp_only,
        "role_swap": role_swap,
    }
    for lag, name in ((12, "spot_lag_1h"), (288, "spot_lag_24h")):
        lag_side = side.shift(lag)
        lag_centroid = centroid.astype(bool).shift(lag, fill_value=False)
        lag_spot_ok = ~frame["spot_quarantined"].astype(bool).shift(
            lag, fill_value=True
        )
        lag_spot_finite = spot_finite.astype(bool).shift(lag, fill_value=False)
        lag_strong_spot = strong_spot.astype(bool).shift(lag, fill_value=False)
        controls[name] = (
            perp_clean
            & perp_finite
            & active_perp
            & strong_perp
            & lag_spot_ok
            & lag_spot_finite
            & lag_strong_spot
            & lag_side.ne(0.0)
            & lag_side.mul(values["spot_return"].shift(lag)).gt(0.0)
            & lag_side.mul(values["perp_return"]).gt(0.0)
            & lag_side.mul(values["perp_flow"]).lt(0.0)
            & lag_side.mul(values["perp_event"]).lt(0.0)
            & lag_centroid
        )

    signal = pd.DataFrame(
        {
            "date": frame["date"],
            "side": np.where(primary, side, 0).astype(np.int8),
            "branch": np.where(primary, "cash_sponsored_rejection", "none"),
            "hold_bars": np.where(primary, cfg.hold_bars, 0).astype(np.int16),
            "spot_strength": spot_strength,
            "spot_threshold": spot_threshold,
            "perp_strength": perp_strength,
            "perp_threshold": perp_threshold,
            "centroid_confirmation": centroid,
            "quarantined": frame["quarantined"].astype(bool),
        }
    )
    return signal, controls


def _schedule_from_mask(
    mask: pd.Series,
    side: pd.Series,
    frame: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    signal = pd.DataFrame(
        {
            "side": np.where(mask.fillna(False), side, 0).astype(np.int8),
            "branch": np.where(mask.fillna(False), "control", "none"),
            "hold_bars": np.where(mask.fillna(False), cfg.hold_bars, 0).astype(np.int16),
        }
    )
    return nonoverlapping_schedule(signal, frame)


def nonoverlapping_schedule(
    signal: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp = SELECTION_END,
) -> pd.DataFrame:
    """Reserve a fixed clock without using post-entry feature availability.

    CSPR needs Spot/perpetual features only on the completed signal bar. A data
    outage after entry cannot cancel an already entered fixed-hold trade in
    live operation, so future feature quarantine is deliberately ignored.
    """
    start_timestamp = frame["date"].min() if start is None else pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if start_timestamp >= end_timestamp:
        raise ValueError("schedule start must precede end")
    period = frame["date"].ge(start_timestamp) & frame["date"].lt(end_timestamp)
    side = pd.to_numeric(signal["side"], errors="coerce").fillna(0).to_numpy(np.int8)
    hold = pd.to_numeric(signal["hold_bars"], errors="coerce").fillna(0).to_numpy(np.int16)
    branch = signal["branch"].astype(str).to_numpy()
    quarantined = frame["quarantined"].astype(bool).to_numpy()
    rows: list[dict[str, Any]] = []
    next_entry = 0
    for signal_position in np.flatnonzero(side):
        entry_position = int(signal_position + 1)
        exit_position = entry_position + int(hold[signal_position])
        if entry_position < next_entry or exit_position >= len(frame):
            continue
        if not period.iloc[signal_position] or not period.iloc[entry_position] or not period.iloc[exit_position]:
            continue
        if quarantined[signal_position]:
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
    return pd.DataFrame(rows)


def _support(schedule: pd.DataFrame, frame: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    dates = pd.to_datetime(schedule["entry_date"]) if not schedule.empty else pd.Series([], dtype="datetime64[ns]")
    by_year = {str(year): int(dates.dt.year.eq(year).sum()) for year in (2020, 2021, 2022, 2023)}
    h1 = int(((dates >= pd.Timestamp("2023-01-01")) & (dates < pd.Timestamp("2023-07-01"))).sum())
    h2 = int(((dates >= pd.Timestamp("2023-07-01")) & (dates < pd.Timestamp("2024-01-01"))).sum())
    total = int(len(schedule))
    long_share = float(schedule["side"].gt(0).mean()) if total else 0.0
    short_share = float(schedule["side"].lt(0).mean()) if total else 0.0
    passes = (
        total >= cfg.minimum_nonoverlap_total
        and all(value >= cfg.minimum_nonoverlap_per_year for value in by_year.values())
        and h1 >= cfg.minimum_nonoverlap_per_2023_half
        and h2 >= cfg.minimum_nonoverlap_per_2023_half
        and min(long_share, short_share) >= cfg.minimum_side_share
    )
    return {
        "nonoverlap_total": total,
        "by_year": by_year,
        "2023_h1": h1,
        "2023_h2": h2,
        "long_share": long_share,
        "short_share": short_share,
        "passes_count_support": bool(passes),
    }


def _jaccard(left: pd.Series, right: pd.Series) -> float:
    left = left.fillna(False).astype(bool)
    right = right.fillna(False).astype(bool)
    union = int((left | right).sum())
    return float((left & right).sum() / union) if union else 0.0


def run_support(cfg: Config) -> dict[str, Any]:
    frame, source = load_causal_frame(cfg)
    side = _directions(frame)["side"]
    rows: list[dict[str, Any]] = []
    for quantile in SUPPORT_QUANTILES:
        signal, controls = classify_events(frame, cfg, quantile=quantile)
        schedule = nonoverlapping_schedule(signal, frame)
        support = _support(schedule, frame, cfg)
        raw_primary = controls["primary"]
        primary_count = int(raw_primary.sum())
        overlap = {name: _jaccard(raw_primary, mask) for name, mask in controls.items() if name != "primary"}
        retention = {
            name: float(primary_count / max(1, int(mask.sum())))
            for name, mask in controls.items()
            if name in {"no_centroid", "no_perp_event_confirmation"}
        }
        novelty_pass = (
            retention["no_centroid"] <= cfg.maximum_centroid_ablation_retention
            and retention["no_perp_event_confirmation"] <= cfg.maximum_perp_event_ablation_retention
            and overlap["spot_lag_1h"] <= cfg.maximum_lag_placebo_jaccard
            and overlap["spot_lag_24h"] <= cfg.maximum_lag_placebo_jaccard
        )
        rows.append(
            {
                "quantile": quantile,
                "raw_primary": primary_count,
                "support": support,
                "control_raw_counts": {name: int(mask.sum()) for name, mask in controls.items() if name != "primary"},
                "control_jaccard": overlap,
                "ablation_retention": retention,
                "passes_novelty": bool(novelty_pass),
                "passes_support": bool(support["passes_count_support"] and novelty_pass),
            }
        )
    passing = [row for row in rows if row["passes_support"]]
    selected = max(passing, key=lambda row: row["quantile"]) if passing else None
    result = {
        "protocol": {
            "name": "CSPR-12 — Cash-Sponsored Perpetual Rejection",
            "support_only": True,
            "outcomes_opened": False,
            "selection_end_exclusive": str(SELECTION_END),
            "signal_availability": "completed spot/perpetual 5m bars; future evaluator enters next USD-M open",
            "traded_instrument": "Binance USD-M BTCUSDT perpetual; spot determines cash direction only",
            "hold": f"fixed {cfg.hold_bars} bars",
            "support_selection": "highest tested strength quantile passing every frozen count and novelty floor",
            "control_action_semantics": {
                "direction_flip": "same primary clock with action equal to negative primary side",
                "signal_delay_1bar": "primary clock and action shifted by one completed 5m bar; entry remains next open",
            },
        },
        "config": asdict(cfg),
        "source": source,
        "support_grid": rows,
        "support_decision": "pass" if selected else "reject_before_returns",
        "selected_quantile": selected["quantile"] if selected else None,
        "selected_support": selected,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for field in Config.__dataclass_fields__.values():
        default = getattr(Config, field.name)
        parser.add_argument("--" + field.name.replace("_", "-"), type=type(default), default=default)
    cfg = Config(**vars(parser.parse_args()))
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
