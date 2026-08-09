"""Materialize source-only TPYSR-24 support clocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd

from training import preregister_treasury_parallel_yield_shock_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


PREREG_SHA = "95dd02f3b16368ff933f7108a9f314e0b21a6235d5a324bf567d502d20ca52b2"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
YEARS = (2023, 2024, 2025, 2026)
URL_TEMPLATE = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value={year}"
)
SOURCE_DIR = Path("data/treasury_parallel_yield_shock_relay_sources_2023_2026")
RAW_DIR = SOURCE_DIR / "official_xml"
OBSERVATIONS = SOURCE_DIR / "treasury_2y_10y_observations.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
PRICE = Path("data/options_oi_chase_exhaustion_sources_2023_2026/btc_completed_hour.csv.gz")
PRICE_SHA = "f075a882b80fc1d050aacd9abd417d4be6b6511c4307e39c98ef25f08822c496"
PRICE_MANIFEST = PRICE.parent / "manifest.json"
PRICE_MANIFEST_SHA = "3e350d16da72da7b60d9e91fbfb1ff4c2e13e5cb954b52b19ceaddf8c4f0e66d"
CLOCK = Path("data/treasury_parallel_yield_shock_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/treasury_parallel_yield_shock_relay_controls_2023_2026")
RESULT = Path("results/treasury_parallel_yield_shock_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "two_year_only",
    "ten_year_only",
    "no_factor_tail",
    "no_volatility_gate",
    "one_observation_stale_factor",
    "direction_flip",
)
COLUMNS = (
    "candidate", "control", "split", "source_day", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side", "yield_2y",
    "yield_10y", "delta_2y", "delta_10y", "z_2y", "z_10y",
    "parallel_factor", "absolute_factor_rank", "btc_realized_variation",
    "btc_realized_variation_rank",
)
NS = {
    "a": "http://www.w3.org/2005/Atom",
    "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
    "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def causal_z(values: pd.Series, lookback: int = 90, minimum: int = 60) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    output = np.full(len(numeric), np.nan)
    history: list[float] = []
    for index, current in enumerate(numeric):
        prior = np.asarray(history[-lookback:], dtype=float)
        if math.isfinite(current) and len(prior) >= minimum:
            std = float(np.std(prior, ddof=1))
            if std > 0:
                output[index] = (current - float(np.mean(prior))) / std
        if math.isfinite(current):
            history.append(float(current))
    return pd.Series(output, index=values.index)


def strict_prior_midrank(values: pd.Series, lookback: int = 90, minimum: int = 60) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    output = np.full(len(numeric), np.nan)
    history: list[float] = []
    for index, current in enumerate(numeric):
        prior = np.asarray(history[-lookback:], dtype=float)
        if math.isfinite(current) and len(prior) >= minimum:
            output[index] = (np.sum(prior < current) + 0.5 * np.sum(prior == current)) / len(prior)
        if math.isfinite(current):
            history.append(float(current))
    return pd.Series(output, index=values.index)


def parse_xml(payload: bytes) -> pd.DataFrame:
    root = ET.fromstring(payload)
    rows = []
    for properties in root.findall("a:entry/a:content/m:properties", NS):
        date = properties.findtext("d:NEW_DATE", namespaces=NS)
        yield_2y = properties.findtext("d:BC_2YEAR", namespaces=NS)
        yield_10y = properties.findtext("d:BC_10YEAR", namespaces=NS)
        rows.append({"source_day": date, "yield_2y": yield_2y, "yield_10y": yield_10y})
    frame = pd.DataFrame(rows, columns=("source_day", "yield_2y", "yield_10y"))
    frame["source_day"] = pd.to_datetime(frame.source_day, utc=True, errors="coerce")
    frame["yield_2y"] = pd.to_numeric(frame.yield_2y, errors="coerce")
    frame["yield_10y"] = pd.to_numeric(frame.yield_10y, errors="coerce")
    return frame


def download_official_sources() -> tuple[pd.DataFrame, dict[str, Any]]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    files = {}
    for year in YEARS:
        url = URL_TEMPLATE.format(year=year)
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        path = RAW_DIR / f"daily_treasury_yield_curve_{year}.xml"
        path.write_bytes(payload)
        frame = parse_xml(payload)
        frame["source_year"] = year
        frames.append(frame)
        files[str(year)] = {
            "url": url,
            "path": str(path),
            "sha256": sha(path),
            "bytes": len(payload),
            "parsed_rows": len(frame),
        }
    observations = pd.concat(frames, ignore_index=True)
    observations = observations[
        observations.source_day.ge(START) & observations.source_day.lt(END)
    ].copy()
    observations = observations.sort_values("source_day").reset_index(drop=True)
    if observations.source_day.isna().any() or observations.source_day.duplicated().any():
        raise RuntimeError("TPYSR Treasury dates are missing or duplicated")
    observations["source_valid"] = (
        np.isfinite(observations[["yield_2y", "yield_10y"]]).all(axis=1)
        & observations[["yield_2y", "yield_10y"]].gt(0).all(axis=1)
    )
    gap = observations.source_day.diff().dt.days
    observations["transition_valid"] = (
        observations.source_valid
        & observations.source_valid.shift(1, fill_value=False)
        & gap.between(1, 5, inclusive="both")
    )
    observations["delta_2y"] = observations.yield_2y.diff().where(observations.transition_valid)
    observations["delta_10y"] = observations.yield_10y.diff().where(observations.transition_valid)
    observations["z_2y"] = causal_z(observations.delta_2y)
    observations["z_10y"] = causal_z(observations.delta_10y)
    observations["same_sign"] = (
        observations.delta_2y.ne(0)
        & observations.delta_10y.ne(0)
        & np.sign(observations.delta_2y).eq(np.sign(observations.delta_10y))
    )
    observations["parallel_factor"] = observations[["z_2y", "z_10y"]].median(axis=1).where(
        observations.transition_valid & observations.same_sign
        & np.isfinite(observations[["z_2y", "z_10y"]]).all(axis=1)
    )
    observations["absolute_factor_rank"] = strict_prior_midrank(observations.parallel_factor.abs())
    observations["decision_time"] = observations.source_day + pd.Timedelta(days=1)
    _write_gzip_csv(observations, OBSERVATIONS)
    core = {
        "protocol_version": "tpysr_24_treasury_source_v1",
        "official_source": "US Treasury Daily Treasury Par Yield Curve Rates XML",
        "files": files,
        "window": [START.isoformat(), END.isoformat()],
        "fields": {"date": "NEW_DATE", "two_year": "BC_2YEAR", "ten_year": "BC_10YEAR"},
        "conservative_availability": "source_day + 1 calendar day at 00:00 UTC",
        "outcomes_opened": False,
        "candidate_incidence_opened": False,
        "no_imputation": True,
        "output": {"path": str(OBSERVATIONS), "sha256": sha(OBSERVATIONS), "rows": len(observations)},
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    SOURCE_MANIFEST.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return observations, manifest


def add_btc_features(frame: pd.DataFrame) -> pd.DataFrame:
    if sha(PRICE) != PRICE_SHA or sha(PRICE_MANIFEST) != PRICE_MANIFEST_SHA:
        raise RuntimeError("TPYSR completed BTC source drift")
    prices = pd.read_csv(PRICE, compression="gzip")
    prices["decision_time"] = pd.to_datetime(prices.decision_time, utc=True, format="mixed")
    prices["open"] = pd.to_numeric(prices.open, errors="coerce")
    prices["close"] = pd.to_numeric(prices.close, errors="coerce")
    prices["valid"] = (
        prices.source_valid.astype(str).str.lower().eq("true")
        & np.isfinite(prices[["open", "close"]]).all(axis=1)
        & prices[["open", "close"]].gt(0).all(axis=1)
    )
    prices = prices.sort_values("decision_time").reset_index(drop=True)
    prices["hour_return"] = np.log(prices.close / prices.open)
    consecutive = prices.decision_time.diff().eq(pd.Timedelta(hours=1))
    prices["btc_realized_variation"] = np.sqrt(
        prices.hour_return.pow(2).rolling(24, min_periods=24).sum()
    )
    prices["btc_valid"] = (
        prices.valid.rolling(24, min_periods=24).sum().eq(24)
        & consecutive.rolling(23, min_periods=23).sum().eq(23)
        & np.isfinite(prices.btc_realized_variation)
    )
    joined = frame.merge(
        prices[["decision_time", "btc_realized_variation", "btc_valid"]],
        on="decision_time", how="left", validate="one_to_one",
    )
    joined["btc_realized_variation_rank"] = strict_prior_midrank(
        joined.btc_realized_variation.where(joined.btc_valid)
    )
    joined["signal_valid"] = (
        joined.transition_valid & joined.same_sign & joined.btc_valid.fillna(False)
        & np.isfinite(joined[["z_2y", "z_10y", "parallel_factor", "absolute_factor_rank",
                              "btc_realized_variation", "btc_realized_variation_rank"]]).all(axis=1)
        & joined.parallel_factor.ne(0)
    )
    return joined


def conditions(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    factor = frame.parallel_factor
    rank = frame.absolute_factor_rank
    if control == "two_year_only":
        factor = frame.z_2y
    elif control == "ten_year_only":
        factor = frame.z_10y
    elif control == "one_observation_stale_factor":
        factor = factor.shift(1)
        rank = rank.shift(1)
    tail = pd.Series(True, index=frame.index) if control == "no_factor_tail" else rank.ge(0.70)
    volatility = (
        pd.Series(True, index=frame.index)
        if control == "no_volatility_gate" else frame.btc_realized_variation_rank.ge(0.65)
    )
    active = frame.signal_valid & np.isfinite(factor) & factor.ne(0) & tail & volatility
    side = -np.sign(factor)
    if control == "direction_flip":
        side = -side
    return active, side


def build_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = conditions(frame, control)
    rows = []
    next_allowed = None
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=24)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items()
                      if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        next_allowed = exit_time
        rows.append({
            "candidate": "TPYSR-24", "control": control, "split": split,
            "source_day": frame.at[index, "source_day"], "decision_time": decision,
            "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time,
            "side": int(side.at[index]), "yield_2y": float(frame.at[index, "yield_2y"]),
            "yield_10y": float(frame.at[index, "yield_10y"]),
            "delta_2y": float(frame.at[index, "delta_2y"]),
            "delta_10y": float(frame.at[index, "delta_10y"]),
            "z_2y": float(frame.at[index, "z_2y"]), "z_10y": float(frame.at[index, "z_10y"]),
            "parallel_factor": float(frame.at[index, "parallel_factor"]),
            "absolute_factor_rank": float(frame.at[index, "absolute_factor_rank"]),
            "btc_realized_variation": float(frame.at[index, "btc_realized_variation"]),
            "btc_realized_variation_rank": float(frame.at[index, "btc_realized_variation_rank"]),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def split_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0,
                "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    monthly = selected.entry_time.dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(selected), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(monthly.max()) / len(selected),
    }


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("TPYSR preregistration hash drift")
    observations, source_manifest = download_official_sources()
    featured = add_btc_features(observations)
    primary = build_clock(featured)
    controls = {name: build_clock(featured, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, clock in controls.items():
        _write_gzip_csv(clock, CONTROL_DIR / f"{name}.csv.gz")
    support = {split: split_stats(primary, split) for split in SPLITS}
    checks = {}
    for split, stats in support.items():
        checks[f"{split}_minimum_events"] = stats["events"] >= MINIMUM[split]
        checks[f"{split}_side_balance"] = stats["minority_side_share"] >= 0.20
        checks[f"{split}_month_concentration"] = stats["max_month_share"] <= 0.45
    preregistration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    passed = all(checks.values())
    core = {
        "protocol_version": "tpysr_24_source_support_v1", "policy_id": "TPYSR-24",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha(prereg.DEFAULT_OUTPUT),
                            "manifest_hash": preregistration["manifest_hash"]},
        "source_manifests": {
            "treasury": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST),
                         "manifest_hash": source_manifest["manifest_hash"]},
            "completed_btc": {"path": str(PRICE_MANIFEST), "sha256": sha(PRICE_MANIFEST)},
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"),
                            "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(clock),
                            "promotion_authorized": False} for name, clock in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
