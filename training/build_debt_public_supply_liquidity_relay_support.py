"""Materialize source-only DPSLR-24 support clocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_debt_public_supply_liquidity_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


PREREG_SHA = "e13b50964732cedde7280923e87ef4dcb7346e1c50705aaf56bff8de4b127ef9"
START = pd.Timestamp("2023-01-01T00:00:00Z")
SOURCE_END = pd.Timestamp("2026-07-25T00:00:00Z")
API_ENDPOINT = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/"
    "od/debt_to_penny"
)
SOURCE_DIR = Path("data/debt_public_supply_liquidity_relay_sources_2023_2026")
RAW_RESPONSE = SOURCE_DIR / "debt_to_the_penny_2023_2026.json"
OBSERVATIONS = SOURCE_DIR / "debt_public_supply_observations.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
PRICE = Path("data/options_oi_chase_exhaustion_sources_2023_2026/btc_completed_hour.csv.gz")
PRICE_SHA = "f075a882b80fc1d050aacd9abd417d4be6b6511c4307e39c98ef25f08822c496"
PRICE_MANIFEST = PRICE.parent / "manifest.json"
PRICE_MANIFEST_SHA = "3e350d16da72da7b60d9e91fbfb1ff4c2e13e5cb954b52b19ceaddf8c4f0e66d"
CLOCK = Path("data/debt_public_supply_liquidity_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/debt_public_supply_liquidity_relay_controls_2023_2026")
RESULT = Path("results/debt_public_supply_liquidity_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "total_public_debt_change",
    "intragovernmental_change",
    "no_magnitude_tail",
    "no_volatility_gate",
    "one_observation_stale_supply_change",
    "direction_flip",
)
COLUMNS = (
    "candidate", "control", "split", "source_day", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side", "debt_held_public_amt",
    "intragov_hold_amt", "tot_pub_debt_out_amt", "public_supply_change",
    "standardized_public_supply_change",
    "absolute_magnitude_rank", "btc_realized_variation",
    "btc_realized_variation_rank",
)
API_FIELDS = (
    "record_date", "debt_held_public_amt", "intragov_hold_amt",
    "tot_pub_debt_out_amt", "src_line_nbr",
)


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


def parse_response(payload: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    value = json.loads(payload)
    if not isinstance(value, dict) or set(value) != {"data", "meta", "links"}:
        raise RuntimeError("DPSLR Fiscal Data response envelope drift")
    rows = value["data"]
    meta = value["meta"]
    if not isinstance(rows, list) or not isinstance(meta, dict):
        raise RuntimeError("DPSLR Fiscal Data response types drift")
    if int(meta.get("total-count", -1)) != len(rows) or int(meta.get("total-pages", -1)) != 1:
        raise RuntimeError("DPSLR Fiscal Data response is incomplete")
    expected = set(API_FIELDS)
    decimals = ("debt_held_public_amt", "intragov_hold_amt", "tot_pub_debt_out_amt")
    parsed = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected or row["src_line_nbr"] != "1":
            raise RuntimeError("DPSLR Fiscal Data row schema drift")
        try:
            amounts = {name: Decimal(row[name]) for name in decimals}
        except (InvalidOperation, TypeError) as exc:
            raise RuntimeError("DPSLR Fiscal Data amount parse failed") from exc
        if any(amount <= 0 for amount in amounts.values()):
            raise RuntimeError("DPSLR Fiscal Data amount is not positive")
        if amounts["debt_held_public_amt"] + amounts["intragov_hold_amt"] != amounts["tot_pub_debt_out_amt"]:
            raise RuntimeError("DPSLR Fiscal Data debt identity failed")
        parsed.append({"source_day": row["record_date"], **{name: float(amounts[name]) for name in decimals}})
    frame = pd.DataFrame(parsed)
    frame["source_day"] = pd.to_datetime(frame.source_day, utc=True, errors="coerce")
    return frame, meta


def download_official_sources() -> tuple[pd.DataFrame, dict[str, Any]]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    query = urlencode({
        "filter": f"record_date:gte:{START.date()},record_date:lt:{SOURCE_END.date()}",
        "fields": ",".join(API_FIELDS), "page[size]": "10000", "sort": "record_date",
    })
    url = API_ENDPOINT + "?" + query
    request = Request(url, headers={"User-Agent": "rllm-dpslr-source/1.0"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    RAW_RESPONSE.write_bytes(payload)
    observations, metadata = parse_response(payload)
    observations = observations.sort_values("source_day").reset_index(drop=True)
    if observations.source_day.isna().any() or observations.source_day.duplicated().any():
        raise RuntimeError("DPSLR Fiscal Data dates are missing or duplicated")
    if observations.empty or observations.source_day.min() < START or observations.source_day.max() >= SOURCE_END:
        raise RuntimeError("DPSLR Fiscal Data source window drift")
    amount_columns = ["debt_held_public_amt", "intragov_hold_amt", "tot_pub_debt_out_amt"]
    observations["source_valid"] = (
        np.isfinite(observations[amount_columns]).all(axis=1)
        & observations[amount_columns].gt(0).all(axis=1)
    )
    gap = observations.source_day.diff().dt.days
    observations["transition_valid"] = (
        observations.source_valid
        & observations.source_valid.shift(1, fill_value=False)
        & gap.between(1, 5, inclusive="both")
    )
    observations["public_supply_change"] = np.log(
        observations.debt_held_public_amt / observations.debt_held_public_amt.shift(1)
    ).where(observations.transition_valid)
    observations["total_public_debt_change"] = np.log(
        observations.tot_pub_debt_out_amt / observations.tot_pub_debt_out_amt.shift(1)
    ).where(observations.transition_valid)
    observations["intragovernmental_change"] = np.log(
        observations.intragov_hold_amt / observations.intragov_hold_amt.shift(1)
    ).where(observations.transition_valid)
    observations["standardized_public_supply_change"] = causal_z(observations.public_supply_change)
    observations["standardized_total_public_debt_change"] = causal_z(observations.total_public_debt_change)
    observations["standardized_intragovernmental_change"] = causal_z(observations.intragovernmental_change)
    observations["absolute_magnitude_rank"] = strict_prior_midrank(
        observations.standardized_public_supply_change.abs()
    )
    observations["decision_time"] = observations.source_day + pd.Timedelta(days=7)
    _write_gzip_csv(observations, OBSERVATIONS)
    core = {
        "protocol_version": "dpslr_24_fiscal_data_source_v1",
        "official_source": "US Treasury Fiscal Data Debt to the Penny API",
        "request": {"url": url, "path": str(RAW_RESPONSE), "sha256": sha(RAW_RESPONSE),
                    "bytes": len(payload), "meta": metadata},
        "window": [START.isoformat(), SOURCE_END.isoformat()],
        "fields": list(API_FIELDS),
        "conservative_availability": "record_date + 7 calendar days at 00:00 UTC",
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
        raise RuntimeError("DPSLR completed BTC source drift")
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
        joined.transition_valid & joined.btc_valid.eq(True)
        & np.isfinite(joined[["standardized_public_supply_change", "absolute_magnitude_rank",
                              "btc_realized_variation", "btc_realized_variation_rank"]]).all(axis=1)
        & joined.standardized_public_supply_change.ne(0)
    )
    return joined


def conditions(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    factor = frame.standardized_public_supply_change
    rank = frame.absolute_magnitude_rank
    if control == "total_public_debt_change":
        factor = frame.standardized_total_public_debt_change
    elif control == "intragovernmental_change":
        factor = frame.standardized_intragovernmental_change
    elif control == "one_observation_stale_supply_change":
        factor = factor.shift(1)
        rank = rank.shift(1)
    tail = pd.Series(True, index=frame.index) if control == "no_magnitude_tail" else rank.ge(0.70)
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
            "candidate": "DPSLR-24", "control": control, "split": split,
            "source_day": frame.at[index, "source_day"], "decision_time": decision,
            "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time,
            "side": int(side.at[index]),
            "debt_held_public_amt": float(frame.at[index, "debt_held_public_amt"]),
            "intragov_hold_amt": float(frame.at[index, "intragov_hold_amt"]),
            "tot_pub_debt_out_amt": float(frame.at[index, "tot_pub_debt_out_amt"]),
            "public_supply_change": float(frame.at[index, "public_supply_change"]),
            "standardized_public_supply_change": float(
                frame.at[index, "standardized_public_supply_change"]
            ),
            "absolute_magnitude_rank": float(frame.at[index, "absolute_magnitude_rank"]),
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
        raise RuntimeError("DPSLR preregistration hash drift")
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
        "protocol_version": "dpslr_24_source_support_v1", "policy_id": "DPSLR-24",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha(prereg.DEFAULT_OUTPUT),
                            "manifest_hash": preregistration["manifest_hash"]},
        "source_manifests": {
            "fiscal_data": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST),
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
