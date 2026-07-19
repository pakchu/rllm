"""Outcome-blind support preregistration for RNCM-72.

RNCM-72 removes the strictly-prior linear response of cumulative average-quote
skew migration to the contemporaneous inner-band quote-center move.  Only the
remaining coherent radial migration can trigger an event.  This module never
loads OHLC, funding, returns, PnL, equity, or any post-2023 row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


POLICY_ID = "RNCM-72"
SOURCE_PANEL = Path(
    "data/binance_um_book_centroid_btcusdt_2023/"
    "BTCUSDT_um_book_centroid_skew_5m_2023.csv.gz"
)
SOURCE_MANIFEST = Path(
    "results/binance_um_book_centroid_btcusdt_2023_manifest.json"
)
SOURCE_PANEL_SHA256 = (
    "c4053ce27d28bebda4137349192b1a940360231469f63edc32bacabb2ce54131"
)
SOURCE_MANIFEST_SHA256 = (
    "d8237c4562d33c12eff162776f723cc5fc94649b69d26a6230e16fc38c52bba1"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_residual_notional_centroid_migration.py"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/residual-notional-centroid-migration-preregistration-2026-07-20.md"
)

COMPARATORS: dict[str, dict[str, str]] = {
    "CCBVFR-72": {
        "path": (
            "results/cross_collateral_book_validated_flow_rejection_"
            "event_clock_2026-07-18.json"
        ),
        "sha256": (
            "79b4838ae634efcff705e028a0ddff8b75d28d79180e3ac89f54b9cab7e5005f"
        ),
        "kind": "embedded_events",
    },
    "PDF-10": {
        "path": (
            "results/cross_collateral_liquidity_credibility_fracture_"
            "event_clock_2026-07-14.json"
        ),
        "sha256": (
            "ab8209308619b97880277b95fcc1a2f825b050a603e24b3e2125ddd5bfb226f8"
        ),
        "kind": "replay_pdf10",
    },
    "CRRC-72": {
        "path": (
            "results/cross_venue_radial_refill_compression_"
            "event_clock_2026-07-17.json"
        ),
        "sha256": (
            "09d2ca954c5c4d06b981575c6b0f0e4dc6b49d8a693da418f3f26e5cc454c835"
        ),
        "kind": "embedded_events",
    },
}

SKEW_COLUMNS = tuple(f"skew_{distance}_median" for distance in range(2, 6))
QUARTERS = {
    "q1": ("2023-01-01", "2023-04-01"),
    "q2": ("2023-04-01", "2023-07-01"),
    "q3": ("2023-07-01", "2023-10-01"),
    "q4": ("2023-10-01", "2024-01-01"),
}
EVENT_FIELDS = (
    "quarter",
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
    support_output: str = (
        "results/residual_notional_centroid_migration_support_2026-07-20.json"
    )
    event_clock_output: str = (
        "results/residual_notional_centroid_migration_event_clock_2026-07-20.json"
    )
    migration_bars: int = 6
    baseline_window_bars: int = 8_640
    baseline_minimum_bars: int = 4_032
    threshold_quantiles: tuple[float, ...] = (0.995, 0.99, 0.985, 0.975)
    quiet_center_quantile: float = 0.50
    minimum_residual_dominance: float = 0.25
    hold_bars: int = 72
    minimum_nonoverlap_total: int = 120
    minimum_nonoverlap_per_half: int = 45
    minimum_nonoverlap_per_quarter: int = 20
    minimum_side_share: float = 0.35
    maximum_quarter_share: float = 0.40
    overlap_tolerance_bars: int = 12
    maximum_comparator_jaccard: float = 0.35
    maximum_synthetic_false_events: int = 0


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_config(cfg: Config) -> None:
    expected = Config(
        support_output=cfg.support_output,
        event_clock_output=cfg.event_clock_output,
    )
    if cfg != expected:
        raise ValueError("RNCM-72 signal and support configuration is frozen")
    if tuple(sorted(cfg.threshold_quantiles, reverse=True)) != (
        cfg.threshold_quantiles
    ):
        raise ValueError("RNCM support quantiles must be strictest first")


def _parse_source_complete(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    parsed = values.astype("string").str.lower().map(
        {"true": True, "false": False}
    )
    if parsed.isna().any():
        raise ValueError("RNCM source_complete contains an unknown value")
    return parsed.astype(bool)


def load_source() -> tuple[pd.DataFrame, dict[str, Any]]:
    if sha256_file(SOURCE_PANEL) != SOURCE_PANEL_SHA256:
        raise ValueError("RNCM source panel hash mismatch")
    if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise ValueError("RNCM source manifest hash mismatch")
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    protocol = manifest.get("protocol", {})
    if protocol.get("outcomes_opened") is not False:
        raise ValueError("RNCM source manifest opened outcomes")
    if protocol.get("post_2023_rows_requested") is not False:
        raise ValueError("RNCM source manifest requested post-2023 rows")
    if protocol.get("external_market_ohlc_or_return_inputs_opened") is not False:
        raise ValueError("RNCM source manifest opened external market outcomes")
    if protocol.get("average_quote_price_levels_derived") is not True:
        raise ValueError("RNCM source lacks the quote-center control")
    item = manifest.get("file", {})
    if Path(item.get("path", "")) != SOURCE_PANEL:
        raise ValueError("RNCM source manifest points to another panel")
    if item.get("sha256") != SOURCE_PANEL_SHA256:
        raise ValueError("RNCM source manifest panel hash differs")

    columns = [
        "date",
        "center_quote_median",
        *SKEW_COLUMNS,
        "source_complete",
        "source_available_at",
    ]
    frame = cast(
        pd.DataFrame,
        pd.read_csv(
            str(SOURCE_PANEL),
            compression="gzip",
            usecols=columns,
            parse_dates=["date", "source_available_at"],
        ),
    )
    expected = pd.date_range(
        "2023-01-01", "2024-01-01", freq="5min", inclusive="left"
    )
    if not frame["date"].equals(pd.Series(expected, name="date")):
        raise ValueError("RNCM source is not the exact 2023 five-minute grid")
    if not frame["source_available_at"].eq(
        frame["date"] + pd.Timedelta(minutes=5)
    ).all():
        raise ValueError("RNCM source availability is not bar close")
    frame["source_complete"] = _parse_source_complete(
        frame["source_complete"]
    )
    feature_columns = ["center_quote_median", *SKEW_COLUMNS]
    complete = frame["source_complete"]
    values = frame.loc[complete, feature_columns].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("RNCM complete source row contains non-finite values")
    if not frame.loc[complete, "center_quote_median"].gt(0.0).all():
        raise ValueError("RNCM quote center must be positive")
    if frame.loc[~complete, feature_columns].notna().any().any():
        raise ValueError("RNCM incomplete source row carries feature values")
    if len(frame) != item.get("rows"):
        raise ValueError("RNCM source row count differs from manifest")
    return frame, {
        "panel_sha256": SOURCE_PANEL_SHA256,
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "rows": int(len(frame)),
        "source_complete_rows": int(complete.sum()),
        "range_start": str(frame["date"].iloc[0]),
        "range_end": str(frame["date"].iloc[-1]),
    }


def prior_quantile(
    values: pd.Series,
    *,
    quantile: float,
    window: int,
    minimum: int,
) -> pd.Series:
    """Rolling quantile whose newest admitted observation is t-1."""
    return cast(
        pd.Series,
        values.shift(1).rolling(window, min_periods=minimum).quantile(quantile),
    )


def prior_beta(
    response: pd.Series,
    driver: pd.Series,
    *,
    window: int,
    minimum: int,
) -> pd.Series:
    """Strictly-prior rolling OLS slope with an intercept."""
    response_prior = response.shift(1)
    driver_prior = driver.shift(1)
    rolling = driver_prior.rolling(window, min_periods=minimum)
    covariance = rolling.cov(response_prior, ddof=0)
    variance = rolling.var(ddof=0)
    return cast(pd.Series, (covariance / variance).where(variance.gt(1e-24)))


def migration_features(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    complete = cast(pd.Series, frame["source_complete"]).astype(bool)
    streak = complete.rolling(
        cfg.migration_bars + 1,
        min_periods=cfg.migration_bars + 1,
    ).sum().eq(cfg.migration_bars + 1)
    skew = frame[list(SKEW_COLUMNS)].where(complete)
    raw = skew - skew.shift(cfg.migration_bars)
    center = cast(pd.Series, frame["center_quote_median"]).where(complete)
    center_move = cast(
        pd.Series,
        np.log(center / center.shift(cfg.migration_bars)).where(streak),
    )
    raw = raw.where(streak)

    residual = pd.DataFrame(index=frame.index)
    beta = pd.DataFrame(index=frame.index)
    for column in SKEW_COLUMNS:
        beta[column] = prior_beta(
            raw[column],
            center_move,
            window=cfg.baseline_window_bars,
            minimum=cfg.baseline_minimum_bars,
        )
        residual[column] = raw[column] - beta[column] * center_move
    finite = pd.Series(
        np.isfinite(residual.to_numpy(float)).all(axis=1),
        index=frame.index,
    )
    residual = residual.where(finite)
    positive = residual.gt(0.0).all(axis=1)
    negative = residual.lt(0.0).all(axis=1)
    raw_magnitude = raw.abs().median(axis=1, skipna=False)
    residual_magnitude = residual.abs().median(axis=1, skipna=False)
    dominance = residual_magnitude / raw_magnitude.replace(0.0, np.nan)
    intensity = residual.median(axis=1, skipna=False)
    quiet_center_threshold = prior_quantile(
        center_move.abs(),
        quantile=cfg.quiet_center_quantile,
        window=cfg.baseline_window_bars,
        minimum=cfg.baseline_minimum_bars,
    )
    return pd.DataFrame(
        {
            "source_streak_complete": streak,
            "center_move_30m": center_move,
            "quiet_center_threshold": quiet_center_threshold,
            "raw_magnitude": raw_magnitude,
            "residual_magnitude": residual_magnitude,
            "residual_dominance": dominance,
            "coherent_positive": positive,
            "coherent_negative": negative,
            "intensity": intensity,
            **{
                f"beta_{column}": beta[column] for column in SKEW_COLUMNS
            },
            **{
                f"residual_{column}": residual[column]
                for column in SKEW_COLUMNS
            },
        }
    )


def build_signal(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    cfg: Config,
    *,
    quantile: float,
) -> pd.DataFrame:
    threshold = prior_quantile(
        cast(pd.Series, features["intensity"]).abs(),
        quantile=quantile,
        window=cfg.baseline_window_bars,
        minimum=cfg.baseline_minimum_bars,
    )
    coherent = features["coherent_positive"] | features["coherent_negative"]
    above = (
        coherent
        & features["residual_dominance"].ge(cfg.minimum_residual_dominance)
        & features["center_move_30m"].abs().le(
            features["quiet_center_threshold"]
        )
        & features["intensity"].abs().ge(threshold)
    )
    crossed_from_below = features["intensity"].abs().shift(1).lt(
        threshold.shift(1)
    )
    candidate = above & crossed_from_below
    side = pd.Series(0, index=frame.index, dtype=np.int8)
    side.loc[candidate & features["intensity"].gt(0.0)] = 1
    side.loc[candidate & features["intensity"].lt(0.0)] = -1
    branch = pd.Series("none", index=frame.index, dtype="string")
    branch.loc[side.gt(0)] = "outward_ask_residual_migration"
    branch.loc[side.lt(0)] = "outward_bid_residual_migration"
    return pd.DataFrame(
        {
            "date": frame["date"],
            "candidate": side.ne(0),
            "threshold": threshold,
            "side": side,
            "branch": branch,
            "hold_bars": np.where(side.ne(0), cfg.hold_bars, 0).astype(
                np.int16
            ),
        }
    )


def quarterly_nonoverlap_schedule(
    signal: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Next-open fixed-hold schedule reset at known UTC quarter boundaries."""
    rows: list[dict[str, Any]] = []
    sides = signal["side"].to_numpy(np.int8)
    holds = signal["hold_bars"].to_numpy(np.int16)
    branches = signal["branch"].astype(str).to_numpy()
    dates = frame["date"]
    for quarter, (start, end) in QUARTERS.items():
        period = dates.ge(start) & dates.lt(end)
        next_entry = 0
        for signal_position in np.flatnonzero(sides):
            if not bool(period.iloc[signal_position]):
                continue
            entry_position = int(signal_position + 1)
            hold_bars = int(holds[signal_position])
            exit_position = entry_position + hold_bars
            if entry_position < next_entry or exit_position >= len(frame):
                continue
            if not bool(period.iloc[entry_position]) or not bool(
                period.iloc[exit_position]
            ):
                continue
            rows.append(
                {
                    "quarter": quarter,
                    "signal_position": int(signal_position),
                    "entry_position": entry_position,
                    "exit_position": exit_position,
                    "signal_date": str(dates.iloc[signal_position]),
                    "entry_date": str(dates.iloc[entry_position]),
                    "exit_date": str(dates.iloc[exit_position]),
                    "side": int(sides[signal_position]),
                    "branch": str(branches[signal_position]),
                    "hold_bars": hold_bars,
                }
            )
            next_entry = exit_position
    return pd.DataFrame(rows)


def support_summary(schedule: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    by_quarter = {
        quarter: int(schedule["quarter"].eq(quarter).sum())
        if not schedule.empty
        else 0
        for quarter in QUARTERS
    }
    total = int(len(schedule))
    long_share = float(schedule["side"].gt(0).mean()) if total else 0.0
    short_share = float(schedule["side"].lt(0).mean()) if total else 0.0
    maximum_quarter_share = max(by_quarter.values()) / total if total else 1.0
    h1 = by_quarter["q1"] + by_quarter["q2"]
    h2 = by_quarter["q3"] + by_quarter["q4"]
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
        "nonoverlap_total": total,
        "by_quarter": by_quarter,
        "h1": int(h1),
        "h2": int(h2),
        "long_share": long_share,
        "short_share": short_share,
        "maximum_quarter_share": float(maximum_quarter_share),
        "passes_incidence": bool(passes),
    }


def select_strictest_passing(trials: list[dict[str, Any]]) -> dict[str, Any] | None:
    for trial in trials:
        if trial["support"]["passes_incidence"]:
            return trial
    return None


def tolerant_event_jaccard(
    first_positions: list[int],
    second_positions: list[int],
    *,
    tolerance_bars: int,
) -> dict[str, Any]:
    if tolerance_bars < 0:
        raise ValueError("RNCM overlap tolerance must be non-negative")
    first = sorted(int(value) for value in first_positions)
    second = sorted(int(value) for value in second_positions)
    first_cursor = second_cursor = matches = 0
    while first_cursor < len(first) and second_cursor < len(second):
        left = first[first_cursor]
        right = second[second_cursor]
        if abs(left - right) <= tolerance_bars:
            matches += 1
            first_cursor += 1
            second_cursor += 1
        elif left < right:
            first_cursor += 1
        else:
            second_cursor += 1
    union = len(first) + len(second) - matches
    return {
        "first_event_count": int(len(first)),
        "second_event_count": int(len(second)),
        "matched_event_count": int(matches),
        "tolerance_bars": int(tolerance_bars),
        "jaccard": float(matches / union) if union else 1.0,
    }


def _canonical_event_records(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    rows = cast(
        list[dict[str, Any]],
        schedule[list(EVENT_FIELDS)].to_dict(orient="records"),
    )
    return [
        {
            "quarter": str(row["quarter"]),
            "signal_position": int(row["signal_position"]),
            "entry_position": int(row["entry_position"]),
            "exit_position": int(row["exit_position"]),
            "signal_date": str(row["signal_date"]),
            "entry_date": str(row["entry_date"]),
            "exit_date": str(row["exit_date"]),
            "side": int(row["side"]),
            "branch": str(row["branch"]),
            "hold_bars": int(row["hold_bars"]),
        }
        for row in rows
    ]


def _full_clock_hash(schedule: pd.DataFrame) -> str:
    """PDF-10's historical six-field canonical clock hash."""
    pdf_fields = (
        "signal_position",
        "entry_position",
        "exit_position",
        "side",
        "branch",
        "hold_bars",
    )
    records = cast(
        list[dict[str, Any]],
        schedule[list(pdf_fields)].to_dict(orient="records"),
    )
    normalized = [
        {
            "signal_position": int(row["signal_position"]),
            "entry_position": int(row["entry_position"]),
            "exit_position": int(row["exit_position"]),
            "side": int(row["side"]),
            "branch": str(row["branch"]),
            "hold_bars": int(row["hold_bars"]),
        }
        for row in records
    ]
    return canonical_hash(normalized)


def rncm_event_clock_hash(
    schedule: pd.DataFrame,
    *,
    selected_quantile: float,
) -> str:
    return canonical_hash(
        {
            "policy_id": POLICY_ID,
            "selected_quantile": float(selected_quantile),
            "events": _canonical_event_records(schedule),
        }
    )


def _embedded_comparator_entries(path: Path) -> list[int]:
    payload = json.loads(path.read_text())
    if payload.get("selection_end_exclusive") != "2024-01-01 00:00:00":
        raise ValueError(f"comparator {path} is not sealed to 2023")
    outcome_flags = {
        key: payload[key]
        for key in ("outcomes_opened", "post_entry_outcomes_opened")
        if key in payload
    }
    if not outcome_flags or any(value is not False for value in outcome_flags.values()):
        raise ValueError(f"comparator {path} lacks an outcome-blind assertion")
    data_flags = {
        key: payload[key]
        for key in (
        "price_funding_return_or_equity_loaded",
        "entry_or_later_ohlc_loaded",
        )
        if key in payload
    }
    if not data_flags or any(value is not False for value in data_flags.values()):
        raise ValueError(f"comparator {path} lacks an outcome-data seal")
    events = payload.get("events")
    if not isinstance(events, list) or len(events) != payload.get("event_count"):
        raise ValueError(f"comparator {path} lacks canonical events")
    return [int(event["entry_position"]) for event in events]


def _pdf10_comparator_entries(path: Path) -> list[int]:
    from training import (
        preregister_cross_collateral_liquidity_credibility_fracture as pdf,
    )

    freeze = json.loads(path.read_text())
    if freeze.get("outcomes_opened_for_pdf10") is not False:
        raise ValueError("PDF-10 comparator opened outcomes")
    if freeze.get("price_or_return_loaded") is not False:
        raise ValueError("PDF-10 comparator loaded price or return")
    support_path = Path(
        "results/cross_collateral_liquidity_credibility_fracture_"
        "support_2026-07-14.json"
    )
    support_hash = (
        "9a3001db640ec8041d885645d33f11dd6075276685eb22f8ae3c618363d3099a"
    )
    if sha256_file(support_path) != support_hash:
        raise ValueError("PDF-10 support artifact hash mismatch")
    if freeze.get("preregistration_result_sha256") != support_hash:
        raise ValueError("PDF-10 freeze does not bind its support artifact")
    support = json.loads(support_path.read_text())
    support_protocol = support.get("protocol", {})
    if support_protocol.get("outcomes_opened_for_pdf10") is not False:
        raise ValueError("PDF-10 support opened outcomes")
    if support_protocol.get("selection_end_exclusive") != (
        "2024-01-01 00:00:00"
    ):
        raise ValueError("PDF-10 support is not sealed to 2023")
    cfg = pdf.Config()
    pdf._validate_frozen_config(cfg)
    frame, _ = pdf.load_credibility(cfg)
    signal = pdf.build_signal(frame, cfg)
    schedule = pdf._quarterly_schedule(signal, frame)
    if _full_clock_hash(schedule) != freeze.get("event_clock_sha256"):
        raise ValueError("PDF-10 comparator replay hash mismatch")
    if len(schedule) != freeze.get("event_count"):
        raise ValueError("PDF-10 comparator event count mismatch")
    return schedule["entry_position"].astype(int).tolist()


def comparator_entries(name: str) -> list[int]:
    spec = COMPARATORS[name]
    path = Path(spec["path"])
    if sha256_file(path) != spec["sha256"]:
        raise ValueError(f"{name} comparator artifact hash mismatch")
    if spec["kind"] == "embedded_events":
        return _embedded_comparator_entries(path)
    if spec["kind"] == "replay_pdf10":
        return _pdf10_comparator_entries(path)
    raise ValueError(f"unknown RNCM comparator kind: {spec['kind']}")


def synthetic_fixed_book_panel(
    rows: int = 105_120,
    *,
    scenario: str = "smooth_symmetric",
) -> pd.DataFrame:
    """Fixed absolute book seen through deterministic moving percentage bands."""
    position = np.arange(rows, dtype=float)
    smooth_anchor = 100.0 * (
        1.0
        + 0.0012 * np.sin(2.0 * np.pi * position / 288.0)
        + 0.0005 * np.sin(2.0 * np.pi * position / (288.0 * 7.0))
        + 0.0002 * np.sin(2.0 * np.pi * position / (288.0 * 29.0))
    )
    if scenario in {"smooth_symmetric", "missing_rows"}:
        anchor = smooth_anchor
        bid_best, ask_best = 99.8, 100.2
    elif scenario == "tick_rounded_anchor":
        anchor = np.round(smooth_anchor, 2)
        bid_best, ask_best = 99.8, 100.2
    elif scenario == "stepped_asymmetric":
        phase = (position % (288.0 * 5.0)) / (288.0 * 5.0)
        anchor = 100.0 + 0.15 * (2.0 * phase - 1.0)
        anchor = np.round(anchor, 2)
        bid_best, ask_best = 99.75, 100.25
    else:
        raise ValueError(f"unknown RNCM synthetic scenario: {scenario}")
    average: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for distance in range(1, 6):
        bid_lower = anchor * (1.0 - distance / 100.0)
        ask_upper = anchor * (1.0 + distance / 100.0)
        average[distance] = (
            0.5 * (bid_lower + bid_best),
            0.5 * (ask_best + ask_upper),
        )
    inner_bid, inner_ask = average[1]
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "center_quote_median": np.sqrt(inner_bid * inner_ask),
            "source_complete": True,
        }
    )
    for distance in range(2, 6):
        bid, ask = average[distance]
        frame[f"skew_{distance}_median"] = (
            np.log(ask / inner_ask) - np.log(inner_bid / bid)
        )
    if scenario == "missing_rows":
        missing = (position.astype(np.int64) % 1_009) < 3
        frame.loc[missing, "source_complete"] = False
        frame.loc[
            missing,
            ["center_quote_median", *SKEW_COLUMNS],
        ] = np.nan
    return frame


def synthetic_discrete_ladder_panel(rows: int = 105_120) -> pd.DataFrame:
    """Stationary asymmetric tick ladder sampled by moving percentage bands."""
    position = np.arange(rows, dtype=float)
    anchor = 100.0 * (
        1.0
        + 0.0011 * np.sin(2.0 * np.pi * position / 288.0)
        + 0.0004 * np.sin(2.0 * np.pi * position / (288.0 * 11.0))
    )
    anchor = np.round(anchor, 2)
    bid_price = np.arange(94.0, 99.8001, 0.002)
    ask_price = np.arange(100.2, 106.0001, 0.002)
    bid_quantity = 1.2 + 0.20 * np.sin(7.0 * bid_price) + 0.08 * np.cos(
        19.0 * bid_price
    )
    ask_quantity = 1.1 + 0.18 * np.cos(5.0 * ask_price) + 0.07 * np.sin(
        23.0 * ask_price
    )
    bid_depth_suffix = np.cumsum(bid_quantity[::-1])[::-1]
    bid_notional_suffix = np.cumsum((bid_quantity * bid_price)[::-1])[::-1]
    ask_depth_prefix = np.cumsum(ask_quantity)
    ask_notional_prefix = np.cumsum(ask_quantity * ask_price)
    average: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for distance in range(1, 6):
        bid_index = np.searchsorted(
            bid_price,
            anchor * (1.0 - distance / 100.0),
            side="left",
        )
        ask_index = np.searchsorted(
            ask_price,
            anchor * (1.0 + distance / 100.0),
            side="right",
        ) - 1
        bid = bid_notional_suffix[bid_index] / bid_depth_suffix[bid_index]
        ask = ask_notional_prefix[ask_index] / ask_depth_prefix[ask_index]
        average[distance] = (bid, ask)
    inner_bid, inner_ask = average[1]
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "center_quote_median": np.sqrt(inner_bid * inner_ask),
            "source_complete": True,
        }
    )
    for distance in range(2, 6):
        bid, ask = average[distance]
        frame[f"skew_{distance}_median"] = (
            np.log(ask / inner_ask) - np.log(inner_bid / bid)
        )
    return frame


def synthetic_null_suite() -> dict[str, pd.DataFrame]:
    return {
        name: synthetic_fixed_book_panel(scenario=name)
        for name in (
            "smooth_symmetric",
            "tick_rounded_anchor",
            "stepped_asymmetric",
            "missing_rows",
        )
    } | {"discrete_asymmetric_ladder": synthetic_discrete_ladder_panel()}


def synthetic_control(cfg: Config) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    passes = True
    for scenario, frame in synthetic_null_suite().items():
        features = migration_features(frame, cfg)
        trials: dict[str, Any] = {}
        for quantile in cfg.threshold_quantiles:
            signal = build_signal(frame, features, cfg, quantile=quantile)
            schedule = quarterly_nonoverlap_schedule(signal, frame)
            count = int(len(schedule))
            trials[str(quantile)] = {
                "raw_events": int(signal["candidate"].sum()),
                "nonoverlap_events": count,
            }
            passes &= count <= cfg.maximum_synthetic_false_events
        scenarios[scenario] = trials
    return {
        "mechanism": (
            "five fixed absolute-book nulls spanning smooth, tick-rounded, "
            "stepped, missing-row, asymmetric, and discrete-ladder geometry; "
            "only the reference anchor driving percentage bands moves"
        ),
        "maximum_allowed_nonoverlap_events_each_quantile": (
            cfg.maximum_synthetic_false_events
        ),
        "scenarios": scenarios,
        "passes": bool(passes),
    }


def protocol(cfg: Config) -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "support_only": True,
        "outcomes_opened": False,
        "external_ohlc_funding_return_or_equity_loaded": False,
        "selection_end_exclusive": "2024-01-01 00:00:00",
        "feature": (
            "30m migration of five-minute median cumulative depth-weighted "
            "average-quote skew at radial bands 2..5, after strictly-prior "
            "rolling OLS removal of the inner-band quote-center move"
        ),
        "signal": (
            "all four residual migrations share sign, residual magnitude is at "
            "least 25% of raw migration, the 30m quote-center move is no larger "
            "than its strictly-prior rolling median, and absolute median residual "
            "crosses a strictly-prior rolling threshold from below"
        ),
        "side": "positive residual migration long; negative short",
        "clock": (
            "source available at completed 5m bar; enter next 5m open; exit after "
            f"{cfg.hold_bars} completed 5m bars"
        ),
        "scheduler": (
            "one position; four quarter-contained schedules; non-overlap state "
            "resets at each known UTC quarter boundary"
        ),
        "source_gap_policy": (
            "signal requires source_complete for t-6..t; future source gaps never "
            "cancel a selected event"
        ),
        "selection": (
            "try quantiles strictest-first and select the first passing frozen "
            "incidence/balance/concentration gates; no fallback after novelty"
        ),
        "sealed_windows": ["test2024", "eval2025", "recent2026"],
    }


def run_support(cfg: Config) -> tuple[dict[str, Any], dict[str, Any] | None]:
    _validate_config(cfg)
    synthetic = synthetic_control(cfg)
    if not synthetic["passes"]:
        return (
            {
                "protocol": protocol(cfg),
                "config": asdict(cfg),
                "synthetic_control": synthetic,
                "source_loaded": False,
                "threshold_trials": [],
                "selected_quantile": None,
                "all_support_gates_pass": False,
                "rejection_reason": "moving-band synthetic control failed",
            },
            None,
        )

    frame, source = load_source()
    features = migration_features(frame, cfg)
    internal_trials: list[dict[str, Any]] = []
    public_trials: list[dict[str, Any]] = []
    for quantile in cfg.threshold_quantiles:
        signal = build_signal(frame, features, cfg, quantile=quantile)
        schedule = quarterly_nonoverlap_schedule(signal, frame)
        support = support_summary(schedule, cfg)
        internal_trials.append(
            {
                "quantile": quantile,
                "signal": signal,
                "schedule": schedule,
                "support": support,
            }
        )
        public_trials.append(
            {
                "quantile": quantile,
                "raw_event_count": int(signal["candidate"].sum()),
                "support": support,
            }
        )
        if support["passes_incidence"]:
            break
    selected = select_strictest_passing(internal_trials)
    if selected is None:
        result = {
            "protocol": protocol(cfg),
            "config": asdict(cfg),
            "synthetic_control": synthetic,
            "source_loaded": True,
            "source": source,
            "threshold_trials": public_trials,
            "selected_quantile": None,
            "all_support_gates_pass": False,
            "rejection_reason": "no frozen quantile passed incidence",
        }
        return result, None

    schedule = selected["schedule"]
    overlaps: dict[str, Any] = {}
    novelty_passes = True
    entries = schedule["entry_position"].astype(int).tolist()
    for name in COMPARATORS:
        overlap = tolerant_event_jaccard(
            entries,
            comparator_entries(name),
            tolerance_bars=cfg.overlap_tolerance_bars,
        )
        overlap["maximum_allowed_jaccard"] = cfg.maximum_comparator_jaccard
        overlap["passes"] = bool(
            overlap["jaccard"] <= cfg.maximum_comparator_jaccard
        )
        overlaps[name] = overlap
        novelty_passes &= overlap["passes"]
    all_pass = bool(selected["support"]["passes_incidence"] and novelty_passes)
    clock_hash = rncm_event_clock_hash(
        schedule,
        selected_quantile=float(selected["quantile"]),
    )
    result = {
        "protocol": protocol(cfg),
        "config": asdict(cfg),
        "frozen_artifacts": {
            "preregistration_source": str(PREREGISTRATION_SOURCE),
            "preregistration_source_sha256": sha256_file(
                PREREGISTRATION_SOURCE
            ),
            "preregistration_document": str(PREREGISTRATION_DOCUMENT),
            "preregistration_document_sha256": sha256_file(
                PREREGISTRATION_DOCUMENT
            ),
            "source_panel_sha256": SOURCE_PANEL_SHA256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "comparators": COMPARATORS,
        },
        "synthetic_control": synthetic,
        "source_loaded": True,
        "source": source,
        "threshold_trials": public_trials,
        "selected_quantile": float(selected["quantile"]),
        "selected_support": selected["support"],
        "selected_event_clock_sha256": clock_hash,
        "comparator_overlap": overlaps,
        "all_support_gates_pass": all_pass,
        "rejection_reason": None if all_pass else "comparator novelty failed",
    }
    if not all_pass:
        return result, None
    event_clock = {
        "protocol": "RNCM-72 canonical outcome-blind event-clock freeze",
        "outcomes_opened": False,
        "external_ohlc_funding_return_or_equity_loaded": False,
        "selection_end_exclusive": "2024-01-01 00:00:00",
        "selected_quantile": float(selected["quantile"]),
        "canonical_fields": list(EVENT_FIELDS),
        "event_count": int(len(schedule)),
        "quarter_counts": selected["support"]["by_quarter"],
        "side_counts": {
            "long": int(schedule["side"].gt(0).sum()),
            "short": int(schedule["side"].lt(0).sum()),
        },
        "event_clock_sha256": clock_hash,
        "events": _canonical_event_records(schedule),
    }
    return result, event_clock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-output", default=Config.support_output)
    parser.add_argument("--event-clock-output", default=Config.event_clock_output)
    args = parser.parse_args()
    cfg = Config(
        support_output=args.support_output,
        event_clock_output=args.event_clock_output,
    )
    result, event_clock = run_support(cfg)
    support_path = Path(cfg.support_output)
    support_path.parent.mkdir(parents=True, exist_ok=True)
    support_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    event_path: str | None = None
    if event_clock is not None:
        path = Path(cfg.event_clock_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                event_clock, ensure_ascii=False, indent=2, allow_nan=False
            )
            + "\n"
        )
        event_path = str(path)
    print(
        json.dumps(
            {
                "outcomes_opened": False,
                "all_support_gates_pass": result["all_support_gates_pass"],
                "selected_quantile": result.get("selected_quantile"),
                "support_output": str(support_path),
                "event_clock_output": event_path,
                "rejection_reason": result.get("rejection_reason"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
