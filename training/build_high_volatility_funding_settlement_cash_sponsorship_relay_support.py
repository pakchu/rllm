"""Build source-only HVFSCS-6 clocks before Gross9 or economics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_funding_settlement_cash_sponsorship_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "a90dbe0784cb4b6ef55de8e5348c22adeec97c31fc5c09e355b1ed63f9034fdf"
START = pd.Timestamp("2022-12-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR = Path("data/high_volatility_funding_settlement_cash_sponsorship_relay_sources_2022_2026")
PANEL = SOURCE_DIR / "settlement_cash_confirmation_panel.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_funding_settlement_cash_sponsorship_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_funding_settlement_cash_sponsorship_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_funding_settlement_cash_sponsorship_relay_support_2026-08-13.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_funding_tail", "no_variation_gate", "funding_and_price_only",
    "one_settlement_stale_funding", "direction_flip", "same_clock_forced_long",
)
COLUMNS = (
    "candidate", "control", "split", "settlement_time", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side", "funding_rate",
    "funding_rank", "pre_settlement_return", "pre_settlement_variation",
    "variation_rank", "spot_return", "spot_aggressive_quote_flow",
)
FUNDING_QUERY = """
SELECT funding_time,funding_rate
FROM funding_rates_binance
WHERE symbol='BTCUSDT' AND funding_time>=:start AND funding_time<:end
ORDER BY funding_time
"""
PERP_QUERY = """
SELECT ts,open,high,low,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='5m' AND ts>=:start AND ts<:end
ORDER BY ts
"""
SPOT_QUERY = """
SELECT ts,open,high,low,close,quote_asset_volume,taker_buy_quote
FROM bars_binance_spot
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = np.asarray(history[-270:], dtype=float)
        if math.isfinite(current) and len(prior) >= 252:
            result.at[index] = ((prior < current).sum() + 0.5 * (prior == current).sum()) / len(prior)
        if math.isfinite(current):
            history.append(float(current))
    return result


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def prepare(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    result["ts"] = pd.to_datetime(result["ts"], utc=True)
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.drop_duplicates("ts", keep=False).set_index("ts").sort_index()


def coherent(window: pd.DataFrame, expected_rows: int) -> pd.Series:
    finite = np.isfinite(window[["open", "high", "low", "close"]]).all(axis=1)
    positive = window[["open", "high", "low", "close"]].gt(0).all(axis=1)
    shape = (
        window.high.ge(window[["open", "close"]].max(axis=1))
        & window.low.le(window[["open", "close"]].min(axis=1))
        & window.high.ge(window.low)
    )
    return finite & positive & shape if len(window) == expected_rows else pd.Series(False, index=window.index)


def build_panel(funding: pd.DataFrame, perp: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    rates = funding.copy()
    rates["funding_time"] = pd.to_datetime(rates.funding_time, utc=True)
    rates["funding_rate"] = pd.to_numeric(rates.funding_rate, errors="coerce")
    rates = rates.sort_values("funding_time").drop_duplicates("funding_time", keep=False).reset_index(drop=True)
    perp = prepare(perp, ("open", "high", "low", "close"))
    spot = prepare(spot, ("open", "high", "low", "close", "quote_asset_volume", "taker_buy_quote"))
    rows: list[dict[str, Any]] = []
    for settlement, funding_rate in rates[["funding_time", "funding_rate"]].itertuples(index=False, name=None):
        pre_grid = pd.date_range(settlement - pd.Timedelta(hours=8), settlement, freq="5min", inclusive="left")
        cash_grid = pd.date_range(settlement, settlement + pd.Timedelta(hours=1), freq="1min", inclusive="left")
        pre, cash = perp.reindex(pre_grid), spot.reindex(cash_grid)
        pre_ok = len(pre) == 96 and bool(coherent(pre, 96).all())
        cash_valid = coherent(cash, 60)
        cash_volume = np.isfinite(cash[["quote_asset_volume", "taker_buy_quote"]]).all(axis=1) & cash[["quote_asset_volume", "taker_buy_quote"]].ge(0).all(axis=1) & cash.taker_buy_quote.le(cash.quote_asset_volume)
        cash_ok = len(cash) == 60 and bool((cash_valid & cash_volume).all()) and float(cash.quote_asset_volume.sum()) > 0
        if pre_ok:
            pre_return = float(np.log(pre.close.iloc[-1] / pre.open.iloc[0]))
            bar_returns = np.log(pre.close.to_numpy(float) / np.r_[pre.open.iloc[0], pre.close.to_numpy(float)[:-1]])
            variation = float(np.sqrt(np.square(bar_returns).sum()))
        else:
            pre_return = variation = np.nan
        if cash_ok:
            spot_return = float(np.log(cash.close.iloc[-1] / cash.open.iloc[0]))
            spot_flow = float(2.0 * cash.taker_buy_quote.sum() - cash.quote_asset_volume.sum())
        else:
            spot_return = spot_flow = np.nan
        source_valid = bool(
            np.isfinite([funding_rate, pre_return, variation, spot_return, spot_flow]).all()
            and funding_rate != 0 and pre_return != 0 and variation > 0 and spot_return != 0 and spot_flow != 0
        )
        rows.append({
            "settlement_time": settlement, "decision_time": settlement + pd.Timedelta(hours=1),
            "funding_rate": funding_rate, "pre_settlement_return": pre_return,
            "pre_settlement_variation": variation, "spot_return": spot_return,
            "spot_aggressive_quote_flow": spot_flow, "perp_source_rows": int(pre.notna().all(axis=1).sum()),
            "spot_source_rows": int(cash.notna().all(axis=1).sum()), "source_valid": source_valid,
        })
    panel = pd.DataFrame(rows)
    panel["funding_rank"] = rank(panel.funding_rate.abs().where(np.isfinite(panel.funding_rate)))
    panel["variation_rank"] = rank(panel.pre_settlement_variation.where(panel.source_valid))
    return panel


def materialize() -> dict[str, Any]:
    from sqlalchemy import text

    engine = postgres_engine()
    params = {"start": START.to_pydatetime(), "end": END.to_pydatetime()}
    with engine.connect() as connection:
        funding = pd.read_sql_query(text(FUNDING_QUERY), connection, params=params)
        perp = pd.read_sql_query(text(PERP_QUERY), connection, params={"start": (START - pd.Timedelta(hours=8)).to_pydatetime(), "end": END.to_pydatetime()})
        spot = pd.read_sql_query(text(SPOT_QUERY), connection, params=params)
    engine.dispose()
    panel = build_panel(funding, perp, spot)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(panel, PANEL)
    core = {
        "protocol_version": "hvfscs_6_sources_v1", "queries": {"funding": FUNDING_QUERY, "perpetual": PERP_QUERY, "spot": SPOT_QUERY},
        "tables": ["funding_rates_binance", "bars_binance", "bars_binance_spot"],
        "symbol": "BTCUSDT", "window": [START.isoformat(), END.isoformat()],
        "candidate_incidence_opened": False, "postentry_outcomes_opened": False, "gross9_rows_opened": False, "no_imputation": True,
        "output": {"path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel), "valid_rows": int(panel.source_valid.sum())},
    }
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}
    SOURCE_MANIFEST.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def features() -> pd.DataFrame:
    frame = pd.read_csv(PANEL, compression="gzip")
    for column in ("settlement_time", "decision_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame["source_valid"] = frame.source_valid.astype(str).str.lower().eq("true")
    for column in ("funding_rate", "funding_rank", "pre_settlement_return", "pre_settlement_variation", "variation_rank", "spot_return", "spot_aggressive_quote_flow"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def build_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    rate, funding_rank = frame.funding_rate, frame.funding_rank
    if control == "one_settlement_stale_funding":
        rate, funding_rank = rate.shift(1), funding_rank.shift(1)
    side = np.sign(rate).fillna(0).astype(int)
    funding_gate = np.isfinite(rate) & rate.ne(0)
    if control != "no_funding_tail":
        funding_gate &= funding_rank.ge(0.60)
    variation_gate = pd.Series(True, index=frame.index) if control == "no_variation_gate" else frame.variation_rank.ge(0.65)
    aligned = side.ne(0) & np.sign(frame.pre_settlement_return).eq(side)
    if control == "funding_and_price_only":
        cash = pd.Series(True, index=frame.index)
    else:
        cash = np.sign(frame.spot_return).eq(side) & np.sign(frame.spot_aggressive_quote_flow).eq(side)
    active = frame.source_valid & funding_gate & variation_gate & aligned & cash
    if control == "direction_flip":
        side = -side
    elif control == "same_clock_forced_long":
        side = pd.Series(1, index=frame.index)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in frame.index[active]:
        settlement = pd.Timestamp(frame.at[index, "settlement_time"])
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry, exit_time = decision + pd.Timedelta(minutes=5), decision + pd.Timedelta(hours=6, minutes=5)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        next_allowed = exit_time
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "settlement_time": settlement, "decision_time": decision, "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]),
            "funding_rate": float(rate.at[index]), "funding_rank": float(funding_rank.at[index]),
            "pre_settlement_return": float(frame.at[index, "pre_settlement_return"]),
            "pre_settlement_variation": float(frame.at[index, "pre_settlement_variation"]),
            "variation_rank": float(frame.at[index, "variation_rank"]), "spot_return": float(frame.at[index, "spot_return"]),
            "spot_aggressive_quote_flow": float(frame.at[index, "spot_aggressive_quote_flow"]),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(candidate: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = candidate[candidate.split.eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    months = subset.entry_time.dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(months.max()) / len(subset)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVFSCS preregistration drift")
    source_manifest = materialize()
    frame = features()
    primary = build_clock(frame)
    controls = {name: build_clock(frame, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, candidate in controls.items():
        _write_gzip_csv(candidate, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {key: value for name, values in support.items() for key, value in ((f"{name}_minimum_events", values["events"] >= MINIMUM[name]), (f"{name}_side_balance", values["minority_side_share"] >= 0.2), (f"{name}_month_concentration", values["max_month_share"] <= 0.45))}
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvfscs_6_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(candidate), "promotion_authorized": False} for name, candidate in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    output = run()
    print(json.dumps({"passed": output["support_passed"], "support": output["support"]}, indent=2))
