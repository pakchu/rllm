"""Freeze outcome-blind support for contract-specific COIN-M roll events.

This module deliberately stops at event support.  It reads completed front and
next quarterly-contract bars, but never reads a price after the signal bar.  A
separate evaluator may open post-entry paths only after this event clock and
its thresholds have been committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_START = pd.Timestamp("2020-07-01")
FIT_START = pd.Timestamp("2020-07-15")
SELECTION_END = pd.Timestamp("2024-01-01")
ROBUST_WINDOW_BARS = 7 * 288
ROBUST_MIN_PERIODS = 1_613  # 80% of the prior seven-day window.
MAX_FRONT_DTE_HOURS = 45 * 24
DELIVERY_BUFFER_HOURS = 12.0
MIN_NEXT_SHARE = 0.10
MIN_FRONT_SHARE = 0.50
EXPECTED_SOURCE_SHA256 = "d107b6dee3f8d1012110db4744cb36d3e7e7fc36a1f93cc17f5ce4c92ab461f3"
EXPECTED_MANIFEST_SHA256 = "cdb1ea8f175b0edebf36373aa3231de0a9026413ee7bb3bf4ee602b5abe2db2e"


@dataclass(frozen=True)
class Candidate:
    name: str
    mechanism: str
    traded_leg: str
    hold_bars: int

    @property
    def hold_minutes(self) -> int:
        return self.hold_bars * 5


CANDIDATES = (
    Candidate(
        name="coinm_next_led_roll_migration_h60m",
        mechanism="next-led accepted flow continuation",
        traded_leg="next",
        hold_bars=12,
    ),
    Candidate(
        name="coinm_front_local_rejected_flow_h30m",
        mechanism="front-local rejected flow fade",
        traded_leg="front",
        hold_bars=6,
    ),
)


@dataclass(frozen=True)
class Config:
    input_csv: str = (
        "data/binance_coinm_quarterly_strip_pre2024/"
        "BTCUSD_front_next_quarterly_5m_20200701T0000_20231231T2350.csv.gz"
    )
    manifest_json: str = (
        "data/binance_coinm_quarterly_strip_pre2024/build_manifest.json"
    )
    output: str = "results/coinm_roll_migration_support_2026-07-19.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def implementation_path() -> str:
    source = Path(__file__).resolve()
    return str(source.relative_to(source.parents[1]))


def parse_feature_valid(values: pd.Series) -> pd.Series:
    """Parse only explicit boolean tokens; never rely on Python truthiness."""
    if values.isna().any():
        raise ValueError("feature_valid contains missing values")
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    tokens = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    unknown = sorted(set(tokens).difference(mapping))
    if unknown:
        raise ValueError(f"feature_valid contains invalid tokens: {unknown}")
    return tokens.map(mapping).astype(bool)


def verify_source_seal(
    input_csv: str | Path,
    manifest_json: str | Path,
    *,
    expected_source_sha256: str = EXPECTED_SOURCE_SHA256,
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
) -> dict[str, Any]:
    source_hash = sha256_file(input_csv)
    manifest_hash = sha256_file(manifest_json)
    if source_hash != expected_source_sha256:
        raise ValueError(
            f"quarterly source SHA mismatch: {source_hash} != {expected_source_sha256}"
        )
    if manifest_hash != expected_manifest_sha256:
        raise ValueError(
            f"quarterly manifest SHA mismatch: {manifest_hash} != {expected_manifest_sha256}"
        )
    with Path(manifest_json).open() as handle:
        manifest = json.load(handle)
    if manifest.get("output_sha256") != source_hash:
        raise ValueError("quarterly manifest does not bind the source output")
    if manifest.get("rows") != 368_351 or manifest.get("valid_rows") != 366_177:
        raise ValueError("quarterly manifest has unexpected row counts")
    if manifest.get("last_signal_bar") != "2023-12-31 23:50:00":
        raise ValueError("quarterly manifest is not sealed before 2024")
    return {
        "path": str(manifest_json),
        "sha256": manifest_hash,
        "output_sha256": source_hash,
    }


def load_source(path: str | Path) -> pd.DataFrame:
    required = {
        "signal_bar_open_utc",
        "feature_available_time_utc",
        "trade_earliest_time_utc",
        "front_symbol",
        "next_symbol",
        "front_open",
        "front_close",
        "front_volume",
        "front_taker_buy_volume",
        "next_open",
        "next_close",
        "next_volume",
        "next_taker_buy_volume",
        "feature_valid",
        "feature_invalid_reason",
        "front_hours_to_delivery",
        "next_hours_to_delivery",
    }
    source = pd.read_csv(
        path,
        compression="infer",
        usecols=lambda column: column in required,
        low_memory=False,
    )
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"quarterly strip missing columns: {missing}")
    for column in (
        "signal_bar_open_utc",
        "feature_available_time_utc",
        "trade_earliest_time_utc",
    ):
        source[column] = (
            pd.to_datetime(source[column], utc=True, errors="raise")
            .dt.tz_convert(None)
        )
    source = source.sort_values("signal_bar_open_utc").reset_index(drop=True)
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

    numeric_columns = [
        f"{leg}_{field}"
        for leg in ("front", "next")
        for field in ("open", "close", "volume", "taker_buy_volume")
    ] + ["front_hours_to_delivery", "next_hours_to_delivery"]
    source[numeric_columns] = source[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = parse_feature_valid(source["feature_valid"])
    source["feature_valid"] = valid
    reason_is_ok = source["feature_invalid_reason"].astype(str).eq("ok")
    if not valid.equals(reason_is_ok):
        raise ValueError("feature_valid disagrees with feature_invalid_reason")
    valid_numeric = source.loc[valid, numeric_columns].to_numpy(float)
    if not np.isfinite(valid_numeric).all():
        raise ValueError("valid quarterly rows contain non-finite signal inputs")
    for leg in ("front", "next"):
        if (source.loc[valid, [f"{leg}_open", f"{leg}_close"]] <= 0.0).any().any():
            raise ValueError("valid quarterly rows contain non-positive prices")
        if (source.loc[valid, [f"{leg}_volume", f"{leg}_taker_buy_volume"]] < 0.0).any().any():
            raise ValueError("valid quarterly rows contain negative contract flow")
        if (
            source.loc[valid, f"{leg}_taker_buy_volume"]
            > source.loc[valid, f"{leg}_volume"] + 1e-8
        ).any():
            raise ValueError("taker-buy contracts exceed total contracts")
    if source.loc[valid, ["front_symbol", "next_symbol"]].isna().any().any():
        raise ValueError("valid quarterly rows lack contract symbols")
    return source


def causal_robust_z(
    values: pd.Series,
    pair_key: pd.Series,
    *,
    window: int = ROBUST_WINDOW_BARS,
    min_periods: int = ROBUST_MIN_PERIODS,
) -> pd.Series:
    """Prior-only rolling median/IQR z-score, reset at every contract pair."""
    if window <= 0 or min_periods <= 0 or min_periods > window:
        raise ValueError("invalid robust window")
    values = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=values.index, dtype=float)
    grouped = pd.DataFrame({"value": values, "pair": pair_key}).groupby(
        "pair", sort=False, dropna=False
    )
    for _, group in grouped:
        history = group["value"].shift(1)
        rolling = history.rolling(window, min_periods=min_periods)
        median = rolling.median()
        q25 = rolling.quantile(0.25)
        q75 = rolling.quantile(0.75)
        scale = (q75 - q25) / 1.349
        zscore = (group["value"] - median) / scale.where(scale > 1e-12)
        output.loc[group.index] = zscore
    return output


def causal_prior_quantile(
    values: pd.Series,
    pair_key: pd.Series,
    quantile: float,
    *,
    window: int = ROBUST_WINDOW_BARS,
    min_periods: int = ROBUST_MIN_PERIODS,
) -> pd.Series:
    """Return a pair-reset rolling quantile of strictly preceding bars."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    values = pd.to_numeric(values, errors="coerce")
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
    """Build the two fixed event clocks from completed signal bars only."""
    state = pd.DataFrame(index=source.index)
    valid = source["feature_valid"].astype(bool)
    front_volume = source["front_volume"].where(valid & source["front_volume"].gt(0.0))
    next_volume = source["next_volume"].where(valid & source["next_volume"].gt(0.0))
    total_volume = front_volume + next_volume
    state["total_volume"] = total_volume
    state["front_share"] = front_volume / total_volume
    state["next_share"] = next_volume / total_volume
    for leg, volume in (("front", front_volume), ("next", next_volume)):
        imbalance = (2.0 * source[f"{leg}_taker_buy_volume"] - volume) / volume
        state[f"{leg}_pressure"] = imbalance * np.sqrt(volume)
        state[f"{leg}_bar_return"] = np.log(
            source[f"{leg}_close"] / source[f"{leg}_open"]
        ).where(valid)

    pair = source["front_symbol"].astype(str) + "|" + source["next_symbol"].astype(str)
    state["prior_q25_total_volume"] = causal_prior_quantile(
        state["total_volume"], pair, 0.25
    )
    state["z_front_share"] = causal_robust_z(state["front_share"], pair)
    state["z_next_share"] = causal_robust_z(state["next_share"], pair)
    state["z_abs_front_pressure"] = causal_robust_z(
        state["front_pressure"].abs(), pair
    )
    state["z_abs_next_pressure"] = causal_robust_z(
        state["next_pressure"].abs(), pair
    )
    state["front_direction"] = np.sign(state["front_pressure"]).fillna(0).astype(np.int8)
    state["next_direction"] = np.sign(state["next_pressure"]).fillna(0).astype(np.int8)
    state["source_valid"] = valid
    return state


def candidate_clock(
    source: pd.DataFrame,
    state: pd.DataFrame,
    candidate: Candidate,
) -> tuple[np.ndarray, np.ndarray]:
    if candidate not in CANDIDATES:
        raise ValueError("candidate is not preregistered")
    hold_hours = candidate.hold_minutes / 60.0
    delivery_safe = (
        source["front_hours_to_delivery"]
        >= DELIVERY_BUFFER_HOURS + hold_hours + 5.0 / 60.0
    ) & (
        source["next_hours_to_delivery"]
        >= DELIVERY_BUFFER_HOURS + hold_hours + 5.0 / 60.0
    )
    near_roll = source["front_hours_to_delivery"].le(MAX_FRONT_DTE_HOURS)
    common = (
        state["source_valid"]
        & delivery_safe
        & near_roll
        & state["total_volume"].ge(state["prior_q25_total_volume"])
    )

    if candidate.traded_leg == "next":
        direction = state["next_direction"]
        active = (
            common
            & state["next_share"].ge(MIN_NEXT_SHARE)
            & state["z_next_share"].ge(1.0)
            & state["z_abs_next_pressure"].ge(2.0)
            & direction.ne(0)
            & (direction * state["next_bar_return"]).ge(0.0005)
            & (direction * state["front_bar_return"]).ge(-0.0001)
        )
        side = direction
    else:
        direction = state["front_direction"]
        active = (
            common
            & state["front_share"].ge(MIN_FRONT_SHARE)
            & state["z_front_share"].ge(0.75)
            & state["z_abs_front_pressure"].ge(1.25)
            & state["z_abs_next_pressure"].le(0.75)
            & direction.ne(0)
            & (direction * state["front_bar_return"]).le(-0.0002)
            & (direction * state["next_bar_return"]).le(0.0)
        )
        side = -direction
    active_array = active.fillna(False).to_numpy(bool)
    return active_array, np.where(active_array, side, 0).astype(np.int8)


def nonoverlapping_schedule(
    source: pd.DataFrame,
    active: np.ndarray,
    side: np.ndarray,
    candidate: Candidate,
    *,
    start: pd.Timestamp = FIT_START,
    end: pd.Timestamp = SELECTION_END,
) -> pd.DataFrame:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if end <= start:
        raise ValueError("schedule end must follow start")
    rows: list[dict[str, Any]] = []
    next_allowed = start
    for position in np.flatnonzero(np.asarray(active, dtype=bool)):
        signal_time = source.iloc[position]["signal_bar_open_utc"]
        feature_available = source.iloc[position]["feature_available_time_utc"]
        entry_time = source.iloc[position]["trade_earliest_time_utc"]
        exit_time = entry_time + pd.Timedelta(minutes=candidate.hold_minutes)
        action = int(side[position])
        symbol = source.iloc[position][f"{candidate.traded_leg}_symbol"]
        if (
            feature_available != entry_time
            or entry_time < start
            or entry_time < next_allowed
            or exit_time >= end
            or action not in (-1, 1)
        ):
            continue
        rows.append(
            {
                "signal_bar_open": str(signal_time),
                "feature_available": str(feature_available),
                "entry_time": str(entry_time),
                "exit_time": str(exit_time),
                "side": action,
                "traded_leg": candidate.traded_leg,
                "symbol": str(symbol),
            }
        )
        next_allowed = exit_time
    return pd.DataFrame(
        rows,
        columns=[
            "signal_bar_open",
            "feature_available",
            "entry_time",
            "exit_time",
            "side",
            "traded_leg",
            "symbol",
        ],
    )


SUPPORT_WINDOWS = {
    "fit": ("2020-07-15", "2023-01-01"),
    "fit_2020h2": ("2020-07-15", "2021-01-01"),
    "fit_2021h1": ("2021-01-01", "2021-07-01"),
    "fit_2021h2": ("2021-07-01", "2022-01-01"),
    "fit_2022h1": ("2022-01-01", "2022-07-01"),
    "fit_2022h2": ("2022-07-01", "2023-01-01"),
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023h1": ("2023-01-01", "2023-07-01"),
    "select_2023h2": ("2023-07-01", "2024-01-01"),
}


def period_support(schedule: pd.DataFrame) -> dict[str, Any]:
    total = len(schedule)
    if not total:
        return {
            "total": 0,
            "longs": 0,
            "shorts": 0,
            "by_month": {},
            "max_month_fraction": 1.0,
            "symbols": {},
            "max_symbol_fraction": 1.0,
        }
    entries = pd.to_datetime(schedule["entry_time"])
    months = entries.dt.to_period("M").value_counts().sort_index()
    symbols = schedule["symbol"].value_counts().sort_index()
    return {
        "total": int(total),
        "longs": int(schedule["side"].gt(0).sum()),
        "shorts": int(schedule["side"].lt(0).sum()),
        "by_month": {str(key): int(value) for key, value in months.items()},
        "max_month_fraction": float(months.max() / total),
        "symbols": {str(key): int(value) for key, value in symbols.items()},
        "max_symbol_fraction": float(symbols.max() / total),
    }


def windowed_support_summary(
    source: pd.DataFrame,
    active: np.ndarray,
    side: np.ndarray,
    candidate: Candidate,
) -> tuple[dict[str, Any], dict[str, str]]:
    support: dict[str, Any] = {}
    schedule_hashes: dict[str, str] = {}
    for name, (start, end) in SUPPORT_WINDOWS.items():
        schedule = nonoverlapping_schedule(
            source,
            active,
            side,
            candidate,
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
        )
        support[name] = period_support(schedule)
        schedule_hashes[name] = canonical_hash(schedule.to_dict(orient="records"))
    return support, schedule_hashes


def support_gates(summary: dict[str, Any]) -> dict[str, bool]:
    fit = summary["fit"]
    select = summary["select_2023"]

    def each_side_at_least(row: dict[str, Any], fraction: float) -> bool:
        total = int(row["total"])
        return total > 0 and min(int(row["longs"]), int(row["shorts"])) / total >= fraction

    return {
        "fit_at_least_400": int(fit["total"]) >= 400,
        "select_2023_at_least_100": int(select["total"]) >= 100,
        "each_fit_half_at_least_50": min(
            int(summary[name]["total"])
            for name in (
                "fit_2020h2",
                "fit_2021h1",
                "fit_2021h2",
                "fit_2022h1",
                "fit_2022h2",
            )
        )
        >= 50,
        "each_2023_half_at_least_40": min(
            int(summary["select_2023h1"]["total"]),
            int(summary["select_2023h2"]["total"]),
        )
        >= 40,
        "fit_each_side_at_least_30pct": each_side_at_least(fit, 0.30),
        "select_each_side_at_least_30pct": each_side_at_least(select, 0.30),
        "fit_max_month_at_most_12pct": float(fit["max_month_fraction"]) <= 0.12,
        "select_max_month_at_most_20pct": float(select["max_month_fraction"]) <= 0.20,
        "fit_max_symbol_at_most_20pct": float(fit["max_symbol_fraction"]) <= 0.20,
        "select_max_symbol_at_most_35pct": float(select["max_symbol_fraction"]) <= 0.35,
    }


def build_report(cfg: Config) -> dict[str, Any]:
    manifest_seal = verify_source_seal(cfg.input_csv, cfg.manifest_json)
    source = load_source(cfg.input_csv)
    state = build_signal_state(source)
    candidates: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        active, side = candidate_clock(source, state, candidate)
        event_clock = [
            {
                "signal_bar_open": str(source.iloc[position]["signal_bar_open_utc"]),
                "side": int(side[position]),
                "symbol": str(source.iloc[position][f"{candidate.traded_leg}_symbol"]),
            }
            for position in np.flatnonzero(active)
        ]
        support, schedule_hashes = windowed_support_summary(
            source, active, side, candidate
        )
        gates = support_gates(support)
        candidates.append(
            {
                "candidate": asdict(candidate),
                "raw_events": int(active.sum()),
                "clock_hash": canonical_hash(event_clock),
                "schedule_hashes": schedule_hashes,
                "support": support,
                "gates": gates,
                "passes_support": bool(all(gates.values())),
            }
        )
    stable = {
        "protocol": {
            "outcomes_opened": False,
            "source_end_exclusive": str(SELECTION_END),
            "fit_start": str(FIT_START),
            "normalization": (
                "strictly-prior 7d rolling median/IQR, min 80%, reset by front/next pair"
            ),
            "volume_semantics": (
                "raw COIN-M contract counts; front/next share uses raw contracts, "
                "signed pressure is taker imbalance times sqrt(contracts)"
            ),
            "signal_clock": "completed five-minute bar; entry no earlier than next bar open",
            "liquidity_floor": "current combined contracts >= strictly-prior pair-local 25th percentile",
            "delivery_rule": (
                "front DTE <=45d and both legs remain >=12h from delivery after fixed exit"
            ),
            "candidate_count": len(CANDIDATES),
            "direction_repair_allowed": False,
            "threshold_repair_after_support_allowed": False,
            "threshold_repair_after_returns_allowed": False,
            "2024_plus_opened": False,
        },
        "thresholds": {
            "next_led": {
                "hold_minutes": 60,
                "next_share_min": MIN_NEXT_SHARE,
                "z_next_share_min": 1.0,
                "z_abs_next_pressure_min": 2.0,
                "directional_next_bar_return_min": 0.0005,
                "directional_front_bar_return_min": -0.0001,
            },
            "front_rejection": {
                "hold_minutes": 30,
                "front_share_min": MIN_FRONT_SHARE,
                "z_front_share_min": 0.75,
                "z_abs_front_pressure_min": 1.25,
                "z_abs_next_pressure_max": 0.75,
                "directional_front_bar_return_max": -0.0002,
                "directional_next_bar_return_max": 0.0,
            },
        },
        "source": {"path": cfg.input_csv, "sha256": sha256_file(cfg.input_csv)},
        "source_manifest": manifest_seal,
        "implementation": {
            "path": implementation_path(),
            "sha256": sha256_file(__file__),
        },
        "candidates": candidates,
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **stable,
        "support_freeze_hash": canonical_hash(stable),
    }


def write_exclusive(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=Config.input_csv)
    parser.add_argument("--manifest-json", default=Config.manifest_json)
    parser.add_argument("--output", default=Config.output)
    cfg = Config(**vars(parser.parse_args()))
    report = build_report(cfg)
    write_exclusive(cfg.output, report)
    print(
        json.dumps(
            {
                "candidates": len(report["candidates"]),
                "support_passes": sum(
                    item["passes_support"] for item in report["candidates"]
                ),
                "raw_events": {
                    item["candidate"]["name"]: item["raw_events"]
                    for item in report["candidates"]
                },
                "support_freeze_hash": report["support_freeze_hash"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
