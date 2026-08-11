"""Build outcome-blind source support for frozen CAMOP2W-168."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_cftc_asset_manager_option_position_second_week_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market


PREREG_SHA = "8e9318ec654374cb1d4d3f7cc252eff2c0b0fc004f10d200983ab2df219434c4"
MARKET = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
MARKET_SHA = "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
RAW_DIR = Path("data/cftc_camop2w_raw")
SOURCE_DIR = Path("data/cftc_asset_manager_option_position_second_week_relay_sources_2021_2026")
STATES = SOURCE_DIR / "weekly_preentry_states.csv.gz"
MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/cftc_asset_manager_option_position_second_week_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/cftc_asset_manager_option_position_second_week_relay_controls_2023_2026")
RESULT = Path("results/cftc_asset_manager_option_position_second_week_relay_support_2026-08-11.json")
CONTRACT_CODE = "133741"
CONTRACT_NAME = "BITCOIN - CHICAGO MERCANTILE EXCHANGE"
YEARS = tuple(range(2021, 2027))
END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_variation_gate",
    "combined_net_without_futures_isolation",
    "one_report_stale_position_change",
    "direction_flip",
    "forced_long",
)
COLUMNS = (
    "candidate", "control", "split", "report_date", "conservative_availability",
    "decision_time", "feature_available_time", "entry_time", "exit_time", "side",
    "futures_asset_manager_net", "combined_asset_manager_net", "isolated_option_net",
    "position_change", "btc_variation", "btc_variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_prior_midrank(values: pd.Series, lookback: int = 270, minimum: int = 180) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = np.asarray(history[-lookback:], dtype=float)
        if np.isfinite(current) and len(prior) >= minimum:
            output.at[index] = ((prior < current).sum() + 0.5 * (prior == current).sum()) / len(prior)
        if np.isfinite(current):
            history.append(float(current))
    return output


def _annual_rows(kind: str, year: int) -> list[dict[str, str]]:
    path = RAW_DIR / f"{kind}_fin_txt_{year}.zip"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise RuntimeError(f"unexpected CFTC archive members: {path}")
        with archive.open(names[0]) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            return [
                row for row in reader
                if row["CFTC_Contract_Market_Code"].strip() == CONTRACT_CODE
                and row["Market_and_Exchange_Names"].strip() == CONTRACT_NAME
            ]


def load_cftc() -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    bindings: dict[str, str] = {}
    for kind in ("fut", "com"):
        rows = []
        for year in YEARS:
            path = RAW_DIR / f"{kind}_fin_txt_{year}.zip"
            bindings[str(path)] = sha(path)
            rows.extend(_annual_rows(kind, year))
        frame = pd.DataFrame(rows)
        frame["report_date"] = pd.to_datetime(frame["Report_Date_as_YYYY-MM-DD"], utc=True)
        frame["asset_manager_net"] = (
            pd.to_numeric(frame["Asset_Mgr_Positions_Long_All"], errors="coerce")
            - pd.to_numeric(frame["Asset_Mgr_Positions_Short_All"], errors="coerce")
        )
        if frame.report_date.duplicated().any():
            raise RuntimeError(f"duplicate CFTC {kind} report date")
        frames.append(frame[["report_date", "asset_manager_net"]].rename(columns={"asset_manager_net": f"{kind}_net"}))
    merged = frames[0].merge(frames[1], on="report_date", how="inner", validate="one_to_one").sort_values("report_date")
    if len(merged) != len(frames[0]) or len(merged) != len(frames[1]):
        raise RuntimeError("CFTC futures/combined report-date mismatch")
    return merged.reset_index(drop=True), {"archive_sha256": bindings, "rows": len(merged)}


def daily_variation_states(market: pd.DataFrame) -> pd.DataFrame:
    frame = market.copy()
    frame["date"] = pd.to_datetime(frame.date, utc=True)
    frame = frame.sort_values("date").set_index("date")
    expected = pd.date_range(frame.index.min(), END, freq="5min", inclusive="left")
    frame = frame.reindex(expected)
    op = pd.to_numeric(frame.open, errors="coerce")
    cl = pd.to_numeric(frame.close, errors="coerce")
    valid = op.gt(0) & cl.gt(0) & np.isfinite(op) & np.isfinite(cl)
    squared = np.log(cl / op).pow(2).where(valid)
    rolling = squared.rolling(288, min_periods=288)
    variation = np.sqrt(rolling.sum())
    complete = valid.rolling(288, min_periods=288).sum().eq(288)
    # At 00:05, the just-completed [00:00,00:05) bar is causally available.
    boundary = pd.date_range(expected.min().ceil("D") + pd.Timedelta(minutes=5), END, freq="1D", inclusive="left")
    values = variation.where(complete).reindex(boundary - pd.Timedelta(minutes=5)).to_numpy()
    states = pd.DataFrame({"feature_available_time": boundary, "btc_variation": values})
    states["btc_variation_rank"] = strict_prior_midrank(states.btc_variation)
    return states


def score_states(cftc: pd.DataFrame, variation: pd.DataFrame) -> pd.DataFrame:
    states = cftc.copy()
    states["isolated_option_net"] = states.com_net - states.fut_net
    consecutive = states.report_date.diff().eq(pd.Timedelta(days=7))
    states["position_change"] = (states.isolated_option_net - states.isolated_option_net.shift(1)).where(consecutive)
    states["combined_position_change"] = (states.com_net - states.com_net.shift(1)).where(consecutive)
    states["stale_position_change"] = (states.isolated_option_net.shift(1) - states.isolated_option_net.shift(2)).where(
        consecutive & consecutive.shift(1, fill_value=False)
    )
    states["conservative_availability"] = states.report_date + pd.Timedelta(days=7)
    states["decision_time"] = states.report_date + pd.Timedelta(days=14)
    states["feature_available_time"] = states.decision_time + pd.Timedelta(minutes=5)
    states = states.merge(variation, on="feature_available_time", how="left", validate="many_to_one")
    return states


def build_clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    if control == "combined_net_without_futures_isolation":
        change = states.combined_position_change
    elif control == "one_report_stale_position_change":
        change = states.stale_position_change
    else:
        change = states.position_change
    active = np.isfinite(change) & change.ne(0) & np.isfinite(states.btc_variation)
    if control != "no_variation_gate":
        active &= states.btc_variation_rank.ge(0.60)
    side = np.sign(change).fillna(0).astype(int)
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = pd.Series(1, index=states.index)
    rows: list[dict[str, Any]] = []
    for index in states.index[active]:
        item = states.loc[index]
        entry = pd.Timestamp(item.feature_available_time)
        exit_time = entry + pd.Timedelta(hours=168)
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "report_date": item.report_date, "conservative_availability": item.conservative_availability,
            "decision_time": item.decision_time, "feature_available_time": item.feature_available_time,
            "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]),
            "futures_asset_manager_net": float(item.fut_net), "combined_asset_manager_net": float(item.com_net),
            "isolated_option_net": float(item.isolated_option_net), "position_change": float(change.at[index]),
            "btc_variation": float(item.btc_variation), "btc_variation_rank": float(item.btc_variation_rank),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    rows = clock[clock.split.eq(split)]
    if rows.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(rows.side.eq(1).sum()), int(rows.side.eq(-1).sum())
    months = pd.to_datetime(rows.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(rows), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(rows), "max_month_share": int(months.max()) / len(rows)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA or sha(MARKET) != MARKET_SHA:
        raise RuntimeError("CAMOP2W frozen input drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    cftc, cftc_source = load_cftc()
    market, market_source = load_market()
    variation = daily_variation_states(market)
    states = score_states(cftc, variation)
    primary = build_clock(states)
    controls = {name: build_clock(states, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, STATES)
    _write_gzip_csv(primary, CLOCK)
    for name, value in controls.items():
        _write_gzip_csv(value, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "camop2w_168_sources_v1", "contract_code": CONTRACT_CODE,
        "contract_name": CONTRACT_NAME, "cftc": cftc_source,
        "market_binding": {"path": str(MARKET), "sha256": MARKET_SHA}, "market_source": market_source,
        "states": {"path": str(STATES), "sha256": sha(STATES), "rows": len(states)},
        "outcomes_opened": False, "gross9_rows_opened": False, "no_imputation": True,
    }
    manifest = {**source_core, "manifest_hash": prereg.canonical_hash(source_core)}
    MANIFEST.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {key: value for name, values in support.items() for key, value in (
        (f"{name}_minimum_events", values["events"] >= MINIMUM[name]),
        (f"{name}_side_balance", values["minority_side_share"] >= 0.20),
        (f"{name}_month_concentration", values["max_month_share"] <= 0.45),
    )}
    passed = all(checks.values())
    core = {
        "protocol_version": "camop2w_168_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST), "manifest_hash": manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(value), "promotion_authorized": False} for name, value in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
