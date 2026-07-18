"""Preregister an outcome-blind COIN-M calendar-curve compression alpha.

The support stage reads only contemporaneous front/next closes, contract volume,
contract identity, delivery clocks, and availability timestamps from the sealed
pre-2024 quarterly strip.  It never computes a post-entry return.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


SOURCE_CSV = Path(
    "data/binance_coinm_quarterly_strip_pre2024_v2/"
    "BTCUSD_front_next_quarterly_5m_20200701T0000_20231231T2350.csv.gz"
)
SOURCE_MANIFEST = Path(
    "data/binance_coinm_quarterly_strip_pre2024_v2/build_manifest.json"
)
EXPECTED_SOURCE_SHA256 = (
    "d2126e546fa890c3537610a59c0341cb8153c38861d42b59477b340280ced30b"
)
EXPECTED_MANIFEST_SHA256 = (
    "29a886f788776dcb3fd8b69b78798bf70ef5e092b54765437a63231c4ffb87af"
)

SOURCE_START = cast(pd.Timestamp, pd.Timestamp("2020-07-01 00:00:00"))
FIT_START = cast(pd.Timestamp, pd.Timestamp("2020-07-15 00:00:00"))
FIT_END = cast(pd.Timestamp, pd.Timestamp("2023-01-01 00:00:00"))
SELECTION_END = cast(pd.Timestamp, pd.Timestamp("2024-01-01 00:00:00"))

ROLLING_WINDOW_BARS = 14 * 24 * 12
ROLLING_MIN_PERIODS = 7 * 24 * 12
CURVE_Z_MIN = 2.0
MIN_RETRACE_BP = 2.0
MIN_RESIDUAL_BP = 15.0
VOLUME_QUANTILE = 0.25
MIN_FRONT_DTE_HOURS = 10.0 * 24.0
MAX_FRONT_DTE_HOURS = 75.0 * 24.0
DELIVERY_BUFFER_HOURS = 12.0

SOURCE_COLUMNS = (
    "signal_bar_open_utc",
    "feature_available_time_utc",
    "trade_earliest_time_utc",
    "front_symbol",
    "next_symbol",
    "front_delivery_utc",
    "next_delivery_utc",
    "front_close",
    "front_volume",
    "next_close",
    "next_volume",
    "feature_valid",
    "feature_invalid_reason",
    "front_hours_to_delivery",
    "next_hours_to_delivery",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    hold_bars: int
    total_gross: float
    minimum_fit_trades: int
    minimum_selection_trades: int
    minimum_selection_half_trades: int


CANDIDATE = Candidate(
    name="coinm_calendar_curve_compression_h12",
    hold_bars=12 * 12,
    total_gross=0.5,
    minimum_fit_trades=150,
    minimum_selection_trades=50,
    minimum_selection_half_trades=25,
)


@dataclass(frozen=True)
class Config:
    source_csv: str = str(SOURCE_CSV)
    source_manifest: str = str(SOURCE_MANIFEST)
    output: str = (
        "results/coinm_calendar_curve_compression_support_2026-07-19.json"
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_feature_valid(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalized.unique()) - {"true", "false"})
    if unexpected:
        raise ValueError(f"unexpected feature_valid values: {unexpected}")
    return normalized.eq("true")


def verify_source_seal(source_csv: str | Path, source_manifest: str | Path) -> None:
    if sha256_file(source_csv) != EXPECTED_SOURCE_SHA256:
        raise ValueError("COIN-M quarterly source SHA-256 mismatch")
    if sha256_file(source_manifest) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("COIN-M quarterly manifest SHA-256 mismatch")
    manifest = json.loads(Path(source_manifest).read_text())
    if manifest.get("output_sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("manifest does not bind the sealed quarterly source")
    protocol = manifest.get("protocol", {})
    if protocol.get("post2023_opened") is not False:
        raise ValueError("manifest does not preserve the pre-2024 seal")


def load_source(path: str | Path) -> pd.DataFrame:
    source = pd.read_csv(
        path,
        compression="infer",
        usecols=lambda column: str(column) in SOURCE_COLUMNS,
        parse_dates=[
            "signal_bar_open_utc",
            "feature_available_time_utc",
            "trade_earliest_time_utc",
            "front_delivery_utc",
            "next_delivery_utc",
        ],
    )
    signal_time = source["signal_bar_open_utc"]
    if source.empty or signal_time.duplicated().any():
        raise ValueError("quarterly strip is empty or has duplicate timestamps")
    if (
        signal_time.iloc[0] != SOURCE_START
        or signal_time.iloc[-1] != SELECTION_END - pd.Timedelta("10min")
    ):
        raise ValueError("quarterly strip is not the sealed pre-2024 interval")
    expected = pd.date_range(signal_time.iloc[0], signal_time.iloc[-1], freq="5min")
    if not signal_time.equals(pd.Series(expected, name="signal_bar_open_utc")):
        raise ValueError("quarterly strip is not a complete five-minute grid")
    expected_available = signal_time + pd.Timedelta("5min")
    if not source["feature_available_time_utc"].equals(expected_available):
        raise ValueError("feature availability is not the next five-minute boundary")
    if not source["trade_earliest_time_utc"].equals(expected_available):
        raise ValueError("trade clock precedes or differs from feature availability")
    if source["trade_earliest_time_utc"].max() >= SELECTION_END:
        raise ValueError("quarterly strip opens 2024 or later")

    numeric = [
        "front_close",
        "next_close",
        "front_volume",
        "next_volume",
        "front_hours_to_delivery",
        "next_hours_to_delivery",
    ]
    source[numeric] = source[numeric].apply(pd.to_numeric, errors="coerce")
    valid = parse_feature_valid(cast(pd.Series, source["feature_valid"]))
    source["feature_valid"] = valid
    reason_is_ok = source["feature_invalid_reason"].astype(str).eq("ok")
    if not valid.equals(reason_is_ok):
        raise ValueError("feature_valid disagrees with feature_invalid_reason")
    values = source.loc[valid, numeric].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("valid quarterly rows contain non-finite support inputs")
    if (source.loc[valid, ["front_close", "next_close"]] <= 0.0).any().any():
        raise ValueError("valid quarterly rows contain non-positive closes")
    if (source.loc[valid, ["front_volume", "next_volume"]] < 0.0).any().any():
        raise ValueError("valid quarterly rows contain negative volume")
    if source.loc[valid, ["front_symbol", "next_symbol"]].isna().any().any():
        raise ValueError("valid quarterly rows lack contract symbols")
    return source


def causal_rolling_state(
    values: pd.Series,
    pair_key: pd.Series,
    *,
    window: int = ROLLING_WINDOW_BARS,
    min_periods: int = ROLLING_MIN_PERIODS,
) -> pd.DataFrame:
    """Return pair-reset center, scale, and z using strictly preceding bars."""
    if window <= 0 or min_periods <= 0 or min_periods > window:
        raise ValueError("invalid rolling state window")
    values = cast(pd.Series, pd.to_numeric(values, errors="coerce"))
    output = pd.DataFrame(
        {"center": np.nan, "scale": np.nan, "z": np.nan}, index=values.index
    )
    grouped = pd.DataFrame({"value": values, "pair": pair_key}).groupby(
        "pair", sort=False, dropna=False
    )
    for _, group in grouped:
        history = group["value"].shift(1)
        rolling = history.rolling(window, min_periods=min_periods)
        center = rolling.median()
        scale = (rolling.quantile(0.75) - rolling.quantile(0.25)) / 1.349
        output.loc[group.index, "center"] = center
        output.loc[group.index, "scale"] = scale
        output.loc[group.index, "z"] = (
            (group["value"] - center) / scale.where(scale > 1e-12)
        )
    return output


def causal_prior_quantile(
    values: pd.Series,
    pair_key: pd.Series,
    quantile: float,
    *,
    window: int = ROLLING_WINDOW_BARS,
    min_periods: int = ROLLING_MIN_PERIODS,
) -> pd.Series:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    values = cast(pd.Series, pd.to_numeric(values, errors="coerce"))
    output = pd.Series(np.nan, index=values.index, dtype=float)
    grouped = pd.DataFrame({"value": values, "pair": pair_key}).groupby(
        "pair", sort=False, dropna=False
    )
    for _, group in grouped:
        output.loc[group.index] = (
            group["value"]
            .shift(1)
            .rolling(window, min_periods=min_periods)
            .quantile(quantile)
        )
    return output


def build_signal_state(source: pd.DataFrame) -> pd.DataFrame:
    valid = source["feature_valid"].astype(bool)
    pair = source["front_symbol"].astype(str) + "|" + source["next_symbol"].astype(str)
    curve = cast(
        pd.Series,
        np.log(source["next_close"] / source["front_close"]).where(valid),
    )
    rolling = causal_rolling_state(curve, pair)
    state = pd.DataFrame(index=source.index)
    state["pair"] = pair
    state["curve"] = curve
    state[["center", "scale", "z"]] = rolling
    state["source_valid"] = valid
    liquid = pd.Series(True, index=source.index)
    for leg in ("front", "next"):
        volume = cast(pd.Series, source[f"{leg}_volume"].where(valid))
        prior_q25 = causal_prior_quantile(volume, pair, VOLUME_QUANTILE)
        state[f"{leg}_prior_volume_q25"] = prior_q25
        state[f"{leg}_liquid"] = volume.ge(prior_q25)
        liquid &= state[f"{leg}_liquid"] & state[f"{leg}_liquid"].shift(
            1, fill_value=False
        )
    state["two_bar_liquid"] = liquid
    return state


def candidate_clock(
    source: pd.DataFrame, state: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    pair = state["pair"]
    same_three_bar_pair = pair.eq(pair.shift(1)) & pair.shift(1).eq(pair.shift(2))
    valid_three_bar_path = (
        state["source_valid"]
        & state["source_valid"].shift(1, fill_value=False)
        & state["source_valid"].shift(2, fill_value=False)
    )
    shock_sign = np.sign(state["curve"].shift(1) - state["center"].shift(1))
    current_sign = np.sign(state["curve"] - state["center"])
    crossed_into_shock = state["z"].shift(1).abs().ge(CURVE_Z_MIN) & state[
        "z"
    ].shift(2).abs().lt(CURVE_Z_MIN)
    retrace = (
        shock_sign * (state["curve"] - state["curve"].shift(1))
    ).le(-MIN_RETRACE_BP / 10_000.0)
    contracted = state["z"].abs().lt(state["z"].shift(1).abs())
    residual = (state["curve"] - state["center"]).abs().ge(
        MIN_RESIDUAL_BP / 10_000.0
    )
    dte = source["front_hours_to_delivery"]
    active = (
        same_three_bar_pair
        & valid_three_bar_path
        & shock_sign.ne(0.0)
        & crossed_into_shock
        & retrace
        & contracted
        & current_sign.eq(shock_sign)
        & residual
        & state["two_bar_liquid"]
        & dte.between(MIN_FRONT_DTE_HOURS, MAX_FRONT_DTE_HOURS)
    )
    active_array = active.fillna(False).to_numpy(bool)
    curve_side = np.where(active_array, -shock_sign, 0).astype(np.int8)
    return active_array, curve_side


def nonoverlapping_schedule(
    source: pd.DataFrame,
    state: pd.DataFrame,
    active: np.ndarray,
    curve_side: np.ndarray,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    start = cast(pd.Timestamp, pd.Timestamp(start))
    end = cast(pd.Timestamp, pd.Timestamp(end))
    if end <= start:
        raise ValueError("schedule end must follow start")
    rows: list[dict[str, Any]] = []
    next_allowed = start
    hold = CANDIDATE.hold_bars
    for confirmation_position in np.flatnonzero(np.asarray(active, dtype=bool)):
        entry_time = source.iloc[confirmation_position]["trade_earliest_time_utc"]
        exit_time = entry_time + pd.Timedelta(minutes=5 * hold)
        side = int(curve_side[confirmation_position])
        delivery_required = DELIVERY_BUFFER_HOURS + hold / 12.0 + 5.0 / 60.0
        if (
            entry_time < start
            or entry_time < next_allowed
            or exit_time >= end
            or side not in (-1, 1)
            or source.iloc[confirmation_position]["front_hours_to_delivery"]
            < delivery_required
            or source.iloc[confirmation_position]["next_hours_to_delivery"]
            < delivery_required
        ):
            continue
        rows.append(
            {
                "shock_bar_open": str(
                    source.iloc[confirmation_position - 1]["signal_bar_open_utc"]
                ),
                "confirmation_bar_open": str(
                    source.iloc[confirmation_position]["signal_bar_open_utc"]
                ),
                "feature_available": str(
                    source.iloc[confirmation_position]["feature_available_time_utc"]
                ),
                "entry_time": str(entry_time),
                "exit_time": str(exit_time),
                "front_symbol": str(
                    source.iloc[confirmation_position]["front_symbol"]
                ),
                "next_symbol": str(
                    source.iloc[confirmation_position]["next_symbol"]
                ),
                "front_side": -side,
                "next_side": side,
                "hold_bars": hold,
            }
        )
        next_allowed = exit_time
    return pd.DataFrame(rows)


def schedule_window(
    schedule: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Slice a parent non-overlap schedule without rescheduling subwindows."""
    if schedule.empty:
        return schedule.copy()
    entries = pd.to_datetime(schedule["entry_time"])
    return schedule.loc[entries.ge(start) & entries.lt(end)].reset_index(drop=True)


def period_support(schedule: pd.DataFrame) -> dict[str, Any]:
    if schedule.empty:
        return {
            "trades": 0,
            "long_next": 0,
            "short_next": 0,
            "long_next_share": None,
            "maximum_month_share": None,
            "maximum_pair_share": None,
            "month_counts": {},
            "pair_counts": {},
        }
    entries = pd.to_datetime(schedule["entry_time"])
    pairs = schedule["front_symbol"].astype(str) + "|" + schedule["next_symbol"].astype(str)
    month_counts = entries.dt.to_period("M").astype(str).value_counts().sort_index()
    pair_counts = pairs.value_counts().sort_index()
    total = len(schedule)
    long_next = int(schedule["next_side"].gt(0).sum())
    return {
        "trades": total,
        "long_next": long_next,
        "short_next": int(total - long_next),
        "long_next_share": long_next / total,
        "maximum_month_share": float(month_counts.max() / total),
        "maximum_pair_share": float(pair_counts.max() / total),
        "month_counts": {str(k): int(v) for k, v in month_counts.items()},
        "pair_counts": {str(k): int(v) for k, v in pair_counts.items()},
    }


def build_support(source: pd.DataFrame, state: pd.DataFrame) -> dict[str, Any]:
    active, side = candidate_clock(source, state)
    fit_schedule = nonoverlapping_schedule(
        source, state, active, side, start=FIT_START, end=FIT_END
    )
    selection_schedule = nonoverlapping_schedule(
        source, state, active, side, start=FIT_END, end=SELECTION_END
    )
    windows: dict[str, tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]] = {
        "fit": (fit_schedule, FIT_START, FIT_END),
        "fit_2020_partial": (
            fit_schedule,
            FIT_START,
            cast(pd.Timestamp, pd.Timestamp("2021-01-01")),
        ),
        "fit_2021": (
            fit_schedule,
            cast(pd.Timestamp, pd.Timestamp("2021-01-01")),
            cast(pd.Timestamp, pd.Timestamp("2022-01-01")),
        ),
        "fit_2022": (
            fit_schedule,
            cast(pd.Timestamp, pd.Timestamp("2022-01-01")),
            FIT_END,
        ),
        "select_2023": (selection_schedule, FIT_END, SELECTION_END),
        "select_2023_h1": (
            selection_schedule,
            FIT_END,
            cast(pd.Timestamp, pd.Timestamp("2023-07-01")),
        ),
        "select_2023_h2": (
            selection_schedule,
            cast(pd.Timestamp, pd.Timestamp("2023-07-01")),
            SELECTION_END,
        ),
    }
    summaries: dict[str, Any] = {}
    schedule_hashes: dict[str, str] = {}
    for name, (parent, start, end) in windows.items():
        sliced = schedule_window(parent, start, end)
        summaries[name] = period_support(sliced)
        schedule_hashes[name] = canonical_hash(sliced.to_dict(orient="records"))
    return {
        "raw_confirmation_events": int(np.asarray(active, dtype=bool).sum()),
        "windows": summaries,
        "schedule_hashes": schedule_hashes,
    }


def support_gates(support: dict[str, Any]) -> dict[str, bool]:
    windows = support["windows"]
    fit = windows["fit"]
    select = windows["select_2023"]
    def bounded(value: Any, lower: float, upper: float) -> bool:
        return value is not None and lower <= float(value) <= upper

    direction_ok = all(
        bounded(windows[name]["long_next_share"], 0.25, 0.75)
        for name in ("fit", "select_2023")
    )
    return {
        "fit_trades": fit["trades"] >= CANDIDATE.minimum_fit_trades,
        "fit_2020_partial_trades": windows["fit_2020_partial"]["trades"] >= 35,
        "fit_2021_trades": windows["fit_2021"]["trades"] >= 50,
        "fit_2022_trades": windows["fit_2022"]["trades"] >= 25,
        "selection_trades": select["trades"]
        >= CANDIDATE.minimum_selection_trades,
        "selection_halves": all(
            windows[name]["trades"] >= CANDIDATE.minimum_selection_half_trades
            for name in ("select_2023_h1", "select_2023_h2")
        ),
        "direction_balance": direction_ok,
        "fit_month_concentration": bounded(
            fit["maximum_month_share"], 0.0, 0.15
        ),
        "selection_month_concentration": bounded(
            select["maximum_month_share"], 0.0, 0.25
        ),
        "fit_pair_concentration": bounded(fit["maximum_pair_share"], 0.0, 0.25),
        "selection_pair_concentration": bounded(
            select["maximum_pair_share"], 0.0, 0.40
        ),
    }


def build_report(cfg: Config) -> dict[str, Any]:
    verify_source_seal(cfg.source_csv, cfg.source_manifest)
    source = load_source(cfg.source_csv)
    state = build_signal_state(source)
    support = build_support(source, state)
    gates = support_gates(support)
    report: dict[str, Any] = {
        "protocol": "COIN-M calendar curve compression support freeze v1",
        "candidate": asdict(CANDIDATE),
        "source": {
            "csv": cfg.source_csv,
            "csv_sha256": sha256_file(cfg.source_csv),
            "manifest": cfg.source_manifest,
            "manifest_sha256": sha256_file(cfg.source_manifest),
            "columns_loaded": list(SOURCE_COLUMNS),
            "outcome_columns_loaded": [],
            "rows": len(source),
            "valid_rows": int(source["feature_valid"].sum()),
            "first_signal_bar": str(source["signal_bar_open_utc"].iloc[0]),
            "last_signal_bar": str(source["signal_bar_open_utc"].iloc[-1]),
        },
        "feature_contract": {
            "curve": "log(next_close/front_close) on completed bars",
            "rolling_window_bars": ROLLING_WINDOW_BARS,
            "rolling_min_periods": ROLLING_MIN_PERIODS,
            "pair_reset": True,
            "strictly_prior_center_and_scale": True,
            "shock_crossing_z": CURVE_Z_MIN,
            "confirmation_retrace_bp": MIN_RETRACE_BP,
            "minimum_remaining_curve_dislocation_bp": MIN_RESIDUAL_BP,
            "both_legs_two_bar_volume_quantile": VOLUME_QUANTILE,
            "front_dte_hours": [MIN_FRONT_DTE_HOURS, MAX_FRONT_DTE_HOURS],
            "confirmation_bar_is_signal_bar": True,
            "entry": "next open after the completed confirmation bar",
            "side": "rich next: short next/long front; cheap next: long next/short front",
        },
        "execution_contract": {
            "delta_neutral_proxy": "equal USD face on opposite inverse legs",
            "total_account_gross": CANDIDATE.total_gross,
            "gross_per_leg": CANDIDATE.total_gross / 2.0,
            "fixed_hold_bars": CANDIDATE.hold_bars,
            "delivery_buffer_hours": DELIVERY_BUFFER_HOURS,
            "nonoverlap": True,
            "base_cost_per_leg_per_side_bp": 6.0,
            "stress_cost_per_leg_per_side_bp": 10.0,
            "funding": "none for delivery futures",
        },
        "selection_contract": {
            "fit": [str(FIT_START), str(FIT_END)],
            "selection": [str(FIT_END), str(SELECTION_END)],
            "sealed": ["2024", "2025", "2026_ytd"],
            "candidate_count": 1,
            "required_fit_and_selection_cagr_to_strict_mdd": 3.0,
            "maximum_fit_and_selection_strict_mdd_pct": 15.0,
            "weekly_cluster_signflip_p_value_max": 0.10,
            "minimum_mean_gross_curve_compression_bp": 12.0,
            "selection_halves_must_be_positive": True,
            "stress_cost_must_be_positive": True,
            "controls": ["direction_flip", "delay_1h", "delay_24h"],
        },
        "support": support,
        "support_gates": gates,
        "passes_support": bool(all(gates.values())),
        "post_entry_returns_computed": False,
        "outcomes_opened": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    hash_payload = {k: v for k, v in report.items() if k != "created_at"}
    report["manifest_hash"] = canonical_hash(hash_payload)
    return report


def write_exclusive(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", default=Config.source_csv)
    parser.add_argument("--source-manifest", default=Config.source_manifest)
    parser.add_argument("--output", default=Config.output)
    args = parser.parse_args()
    cfg = Config(
        source_csv=args.source_csv,
        source_manifest=args.source_manifest,
        output=args.output,
    )
    write_exclusive(cfg.output, build_report(cfg))


if __name__ == "__main__":
    main()
