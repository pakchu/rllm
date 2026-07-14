"""Support-only preregistration scaffold for the MFIC alpha.

This module deliberately contains no return calculation or backtest simulator.
It freezes causal feature formulas, source-gap quarantine, and candidate support
before any 2024+ outcome is opened.
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


SELECTION_END = pd.Timestamp("2024-01-01")


@dataclass(frozen=True)
class Candidate:
    name: str
    window_bars: int
    segment_bars: int
    continuation_hold_bars: int
    fade_hold_bars: int


CANDIDATES = (
    Candidate("mfic_fast", 12, 3, 3, 6),
    Candidate("mfic_slow", 24, 6, 6, 12),
)


@dataclass(frozen=True)
class Config:
    features: str = (
        "data/binance_um_aggtrade_microstructure_btc_2020_2023/"
        "BTCUSDT_aggtrade_5m_2020-01-01_2023-12-31.csv.gz"
    )
    feature_manifest: str = (
        "data/binance_um_aggtrade_microstructure_btc_2020_2023/"
        "build_manifest.json"
    )
    market: str = (
        "data/binance_um_kline_reference_btc_2020_2023/"
        "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
    )
    market_manifest: str = "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
    output: str = "results/metaorder_fragmentation_impact_curvature_support_2026-07-14.json"
    curvature_threshold: float = 0.002
    persistence_floor: float = 0.60
    coherence_floor: float = 0.20
    hidden_metaorder_quantile: float = 0.95
    hidden_metaorder_baseline_bars: int = 8_640
    hidden_metaorder_baseline_min_periods: int = 2_016
    minimum_agg_trade_count: int = 64
    post_gap_quarantine_bars: int = 24
    minimum_nonoverlap_total: int = 250
    minimum_nonoverlap_per_year: int = 40
    minimum_side_share: float = 0.25
    minimum_branch_share: float = 0.20


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_five_minute_grid(dates: pd.Series, *, label: str) -> None:
    if dates.empty:
        raise ValueError(f"{label} is empty")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta(minutes=5)).all():
        raise ValueError(f"{label} must be a complete 5-minute grid")


def _source_gap_days(manifest: dict[str, Any]) -> set[str]:
    archives = [
        archive
        for month in manifest.get("months", [])
        for archive in month.get("archives", [])
    ]
    gaps = {
        archive["date"]
        for archive in archives
        if int(archive["last_agg_trade_id"])
        - int(archive["first_agg_trade_id"])
        + 1
        - int(archive["agg_trade_rows"])
        > 0
    }
    for previous, current in zip(archives, archives[1:]):
        delta = int(current["first_agg_trade_id"]) - int(previous["last_agg_trade_id"]) - 1
        if delta > 0:
            gaps.add(previous["date"])
            gaps.add(current["date"])
        if delta < 0:
            raise ValueError("aggregate trade IDs overlap across source days")
    return gaps


def quarantine_mask(
    source_available: pd.Series,
    source_gap_day: pd.Series,
    post_gap_bars: int,
) -> pd.Series:
    if post_gap_bars < 0:
        raise ValueError("post-gap quarantine bars cannot be negative")
    base_invalid = (~source_available.astype(bool)) | source_gap_day.astype(bool)
    return (
        base_invalid.astype(np.int8)
        .rolling(post_gap_bars + 1, min_periods=1)
        .max()
        .astype(bool)
    )


def load_causal_frame(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_path = Path(cfg.features)
    market_path = Path(cfg.market)
    feature_manifest = json.loads(Path(cfg.feature_manifest).read_text())
    market_manifest = json.loads(Path(cfg.market_manifest).read_text())
    if feature_manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("feature manifest does not preserve the unopened-outcomes contract")
    if market_manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("market manifest does not preserve the unopened-outcomes contract")
    if _sha256(feature_path) != feature_manifest.get("combined_sha256"):
        raise ValueError("aggTrade feature hash does not match manifest")
    if _sha256(market_path) != market_manifest.get("combined_sha256"):
        raise ValueError("kline reference hash does not match manifest")

    market = pd.read_csv(market_path, compression="gzip", parse_dates=["date"])
    features = pd.read_csv(feature_path, compression="gzip", parse_dates=["date"])
    if market["date"].max() >= SELECTION_END or features["date"].max() >= SELECTION_END:
        raise ValueError("support frame contains 2024+ rows")
    if market["date"].duplicated().any() or not market["date"].is_monotonic_increasing:
        raise ValueError("market timestamps are invalid")
    _assert_five_minute_grid(market["date"], label="market")
    if features["date"].duplicated().any() or not features["date"].is_monotonic_increasing:
        raise ValueError("aggTrade timestamps are invalid")

    frame = market.merge(features, on="date", how="left", validate="one_to_one")
    _assert_five_minute_grid(frame["date"], label="merged frame")
    frame["source_available"] = frame["agg_trade_count"].notna()
    gap_days = _source_gap_days(feature_manifest)
    frame["source_gap_day"] = frame["date"].dt.strftime("%Y-%m-%d").isin(gap_days)
    frame["quarantined"] = quarantine_mask(
        frame["source_available"],
        frame["source_gap_day"],
        cfg.post_gap_quarantine_bars,
    )
    metadata = {
        "feature_sha256": _sha256(feature_path),
        "market_sha256": _sha256(market_path),
        "source_gap_days": sorted(gap_days),
        "missing_feature_bars": int((~frame["source_available"]).sum()),
        "quarantined_bars": int(frame["quarantined"].sum()),
        "first_date": str(frame["date"].min()),
        "last_date": str(frame["date"].max()),
    }
    return frame, metadata


def _selected(positive: pd.Series, negative: pd.Series, direction: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(direction.gt(0.0), positive, np.where(direction.lt(0.0), negative, np.nan)),
        index=direction.index,
        dtype=float,
    )


def compute_mfic(frame: pd.DataFrame, candidate: Candidate, cfg: Config) -> pd.DataFrame:
    window = candidate.window_bars
    segment = candidate.segment_bars
    if window != 4 * segment:
        raise ValueError("MFIC candidate requires window = 4 * segment")
    if not 0.0 <= cfg.hidden_metaorder_quantile <= 1.0:
        raise ValueError("hidden-metaorder quantile must be in [0, 1]")
    if not 1 <= cfg.hidden_metaorder_baseline_min_periods <= cfg.hidden_metaorder_baseline_bars:
        raise ValueError("hidden-metaorder baseline periods are invalid")

    flow = frame["signed_quote_notional"].fillna(0.0).astype(float)
    abs_flow = flow.abs()
    direction = np.sign(flow.rolling(window, min_periods=window).sum())
    direction = pd.Series(direction, index=frame.index, dtype=float)
    positive = flow.gt(0.0)
    negative = flow.lt(0.0)
    positive_weight = abs_flow.where(positive, 0.0)
    negative_weight = abs_flow.where(negative, 0.0)
    total_weight = abs_flow.rolling(window, min_periods=window).sum()
    positive_total = positive_weight.rolling(window, min_periods=window).sum()
    negative_total = negative_weight.rolling(window, min_periods=window).sum()
    same_total = _selected(positive_total, negative_total, direction)

    coherence = frame["flow_coherence"].fillna(0.0).astype(float)
    effective = frame["normalized_effective_event_count"].fillna(0.0).clip(lower=0.0)
    run_component = 0.5 * (
        1.0
        - frame["sign_flip_rate"].fillna(0.0).astype(float)
        + frame["max_same_sign_run_share"].fillna(0.0).astype(float)
    )

    def directional_weighted_mean(value: pd.Series) -> pd.Series:
        pos = (positive_weight * value).rolling(window, min_periods=window).sum()
        neg = (negative_weight * value).rolling(window, min_periods=window).sum()
        return _selected(pos, neg, direction).divide(same_total.replace(0.0, np.nan))

    persistence = same_total.divide(total_weight.replace(0.0, np.nan))
    directional_coherence = directional_weighted_mean(coherence)
    fragmentation = directional_weighted_mean(np.sqrt(effective))
    run_persistence = directional_weighted_mean(run_component)
    hidden_metaorder = (
        persistence * directional_coherence * fragmentation * run_persistence
    )

    # Square-root participation normalization follows the concave impact
    # literature; it avoids treating impact as linear in imbalance.
    impact_efficiency = frame["signed_price_response"].fillna(0.0).astype(float).divide(
        np.sqrt(coherence.clip(lower=0.01))
    )
    weighted_impact = abs_flow * impact_efficiency
    pos_impact = weighted_impact.where(positive, 0.0)
    neg_impact = weighted_impact.where(negative, 0.0)

    def segment_mean(weight: pd.Series, numerator: pd.Series, shift: int) -> pd.Series:
        denominator = weight.rolling(segment, min_periods=segment).sum().shift(shift)
        values = numerator.rolling(segment, min_periods=segment).sum().shift(shift)
        return values.divide(denominator.replace(0.0, np.nan))

    pos_recent = segment_mean(positive_weight, pos_impact, 0)
    neg_recent = segment_mean(negative_weight, neg_impact, 0)
    pos_prior = segment_mean(positive_weight, pos_impact, segment)
    neg_prior = segment_mean(negative_weight, neg_impact, segment)
    recent_impact = _selected(pos_recent, neg_recent, direction)
    prior_impact = _selected(pos_prior, neg_prior, direction)
    curvature = recent_impact - prior_impact

    pos_recent_count = positive.astype(int).rolling(segment, min_periods=segment).sum()
    neg_recent_count = negative.astype(int).rolling(segment, min_periods=segment).sum()
    pos_prior_count = pos_recent_count.shift(segment)
    neg_prior_count = neg_recent_count.shift(segment)
    recent_count = _selected(pos_recent_count, neg_recent_count, direction)
    prior_count = _selected(pos_prior_count, neg_prior_count, direction)

    close = frame["close"].astype(float)
    extension = direction * np.log(close / close.shift(window))
    unquarantined_window = (
        (~frame["quarantined"]).astype(int).rolling(window, min_periods=window).sum().eq(window)
    )
    # Fragmentation and run-length scales changed materially as Binance market
    # structure matured.  Compare their composite to a trailing, strictly
    # lagged baseline rather than selecting an era-specific absolute cutoff.
    hidden_metaorder_baseline = (
        hidden_metaorder.where(unquarantined_window)
        .shift(1)
        .rolling(
            cfg.hidden_metaorder_baseline_bars,
            min_periods=cfg.hidden_metaorder_baseline_min_periods,
        )
        .quantile(cfg.hidden_metaorder_quantile)
    )
    relative_hidden_metaorder_strength = hidden_metaorder.divide(
        hidden_metaorder_baseline.replace(0.0, np.nan)
    )
    active_window = (
        frame["agg_trade_count"]
        .fillna(0.0)
        .rolling(window, min_periods=window)
        .min()
        .ge(cfg.minimum_agg_trade_count)
    )
    mechanism = (
        unquarantined_window
        & active_window
        & direction.ne(0.0)
        & recent_count.ge(2.0)
        & prior_count.ge(2.0)
        & persistence.ge(cfg.persistence_floor)
        & directional_coherence.ge(cfg.coherence_floor)
        & hidden_metaorder.ge(hidden_metaorder_baseline)
    )
    continuation = (
        mechanism
        & curvature.ge(cfg.curvature_threshold)
        & recent_impact.gt(0.0)
    )
    fade = (
        mechanism
        & curvature.le(-cfg.curvature_threshold)
        & prior_impact.gt(0.0)
        & extension.gt(0.0)
    )
    side = pd.Series(0, index=frame.index, dtype=np.int8)
    side.loc[continuation] = direction.loc[continuation].astype(np.int8)
    side.loc[fade] = -direction.loc[fade].astype(np.int8)
    branch = pd.Series("none", index=frame.index, dtype="string")
    branch.loc[continuation] = "continuation"
    branch.loc[fade] = "fade"
    hold = pd.Series(0, index=frame.index, dtype=np.int16)
    hold.loc[continuation] = candidate.continuation_hold_bars
    hold.loc[fade] = candidate.fade_hold_bars

    return pd.DataFrame(
        {
            "date": frame["date"],
            "direction": direction,
            "persistence": persistence,
            "coherence": directional_coherence,
            "fragmentation": fragmentation,
            "run_persistence": run_persistence,
            "hidden_metaorder": hidden_metaorder,
            "hidden_metaorder_baseline": hidden_metaorder_baseline,
            "relative_hidden_metaorder_strength": relative_hidden_metaorder_strength,
            "recent_impact": recent_impact,
            "prior_impact": prior_impact,
            "curvature": curvature,
            "extension": extension,
            "mechanism": mechanism,
            "side": side,
            "branch": branch,
            "hold_bars": hold,
            "quarantined": frame["quarantined"],
        }
    )


def nonoverlapping_schedule(
    signal: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp = SELECTION_END,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    next_entry = 0
    start_timestamp = frame["date"].min() if start is None else pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if start_timestamp >= end_timestamp:
        raise ValueError("schedule start must be before end")
    period = (
        frame["date"].ge(start_timestamp) & frame["date"].lt(end_timestamp)
    ).to_numpy(bool)
    quarantined = frame["quarantined"].to_numpy(bool)
    side = signal["side"].to_numpy(np.int8)
    hold = signal["hold_bars"].to_numpy(np.int16)
    branch = signal["branch"].astype(str).to_numpy()
    for signal_position in np.flatnonzero(side):
        if not period[signal_position]:
            continue
        entry_position = int(signal_position + 1)
        hold_bars = int(hold[signal_position])
        exit_position = entry_position + hold_bars
        if entry_position < next_entry or exit_position >= len(frame):
            continue
        if not period[entry_position] or not period[exit_position]:
            continue
        # Fail closed even when callers provide a hand-built signal frame:
        # the signal bar, entry, holding path, and exit must all be clean.
        if quarantined[signal_position : exit_position + 1].any():
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
                "hold_bars": hold_bars,
            }
        )
        next_entry = exit_position
    return pd.DataFrame(rows)


def _support_summary(signal: pd.DataFrame, frame: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    schedule = nonoverlapping_schedule(signal, frame)
    if schedule.empty:
        return {
            "nonoverlap_total": 0,
            "by_year": {},
            "2023_h1": 0,
            "2023_h2": 0,
            "long_share": 0.0,
            "short_share": 0.0,
            "continuation_share": 0.0,
            "fade_share": 0.0,
            "passes_support": False,
        }
    required_years = (2020, 2021, 2022, 2023)
    by_year = {
        year: len(
            nonoverlapping_schedule(
                signal,
                frame,
                start=f"{year}-01-01",
                end=f"{year + 1}-01-01",
            )
        )
        for year in required_years
    }
    total = len(schedule)
    long_share = float(schedule["side"].gt(0).mean())
    short_share = float(schedule["side"].lt(0).mean())
    continuation_share = float(schedule["branch"].eq("continuation").mean())
    fade_share = float(schedule["branch"].eq("fade").mean())
    passes = (
        total >= cfg.minimum_nonoverlap_total
        and all(int(by_year.get(year, 0)) >= cfg.minimum_nonoverlap_per_year for year in required_years)
        and min(long_share, short_share) >= cfg.minimum_side_share
        and min(continuation_share, fade_share) >= cfg.minimum_branch_share
    )
    windows = {
        "2023_h1": len(
            nonoverlapping_schedule(
                signal, frame, start="2023-01-01", end="2023-07-01"
            )
        ),
        "2023_h2": len(
            nonoverlapping_schedule(
                signal, frame, start="2023-07-01", end="2024-01-01"
            )
        ),
    }
    return {
        "nonoverlap_total": int(total),
        "by_year": {str(key): int(value) for key, value in by_year.items()},
        **windows,
        "long_share": long_share,
        "short_share": short_share,
        "continuation_share": continuation_share,
        "fade_share": fade_share,
        "passes_support": bool(passes),
    }


def run_support(cfg: Config) -> dict[str, Any]:
    frame, source_metadata = load_causal_frame(cfg)
    candidates: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        signal = compute_mfic(frame, candidate, cfg)
        mechanism_values = signal.loc[
            signal["mechanism"],
            ["hidden_metaorder", "relative_hidden_metaorder_strength", "curvature"],
        ]
        candidates.append(
            {
                "candidate": asdict(candidate),
                "raw_signal_count": int(signal["side"].ne(0).sum()),
                "raw_continuation_count": int(signal["branch"].eq("continuation").sum()),
                "raw_fade_count": int(signal["branch"].eq("fade").sum()),
                "mechanism_count": int(signal["mechanism"].sum()),
                "mechanism_feature_quantiles": {
                    column: {
                        str(quantile): float(mechanism_values[column].quantile(quantile))
                        for quantile in (0.01, 0.10, 0.50, 0.90, 0.99)
                    }
                    for column in mechanism_values.columns
                },
                "support": _support_summary(signal, frame, cfg),
            }
        )
    result = {
        "protocol": {
            "name": "MFIC — Metaorder Fragmentation Impact Curvature",
            "support_only": True,
            "outcomes_opened": False,
            "selection_end_exclusive": str(SELECTION_END),
            "signal_availability": "completed 5m bar; any later evaluation enters at next open",
            "source_gap_policy": "full gap day, missing slot, and next 24 bars quarantined",
        },
        "config": asdict(cfg),
        "source": source_metadata,
        "candidates": candidates,
        "all_candidates_pass_support": all(
            candidate["support"]["passes_support"] for candidate in candidates
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for field in Config.__dataclass_fields__.values():
        argument = "--" + field.name.replace("_", "-")
        default = getattr(Config, field.name)
        parser.add_argument(argument, type=type(default), default=default)
    cfg = Config(**vars(parser.parse_args()))
    result = run_support(cfg)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "outcomes_opened": result["protocol"]["outcomes_opened"],
                "all_candidates_pass_support": result["all_candidates_pass_support"],
                "candidates": [
                    {
                        "name": item["candidate"]["name"],
                        **item["support"],
                    }
                    for item in result["candidates"]
                ],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
