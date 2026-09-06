"""Build source-only HVICLR-72 clocks before Gross9 or economics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_initial_claims_labor_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market

PREREG_SHA = "3ebeb23668b88aef2f5a1aeb06dcd2b8e77052a595738749ee9466207b94d2e1"
SOURCE_SHA = "92d6627594c2845180ed303e9f7490bab5d121755dec0d977f49fbb960d3a707"
SOURCE_MANIFEST = Path("data/initial_claims_alfred_causal_vintages_2020_2026_manifest.json")
SOURCE_MANIFEST_SHA = "340165b6970ecb635ccab256686b245e49b57179fe8b0050f6a25fe596a929eb"
HELPER = Path("training/build_scheduled_trend_concordance_relay_support.py")
HELPER_SHA = "8ca554d88506df277434f73e5eb8850426614a880110088eb91aaae3b23c154f"
END = pd.Timestamp("2026-08-01T00:00:00Z")
STATE = Path("data/high_volatility_initial_claims_labor_relay_sources_2023_2026/weekly_states.csv.gz")
CLOCK = Path("data/high_volatility_initial_claims_labor_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_initial_claims_labor_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_initial_claims_labor_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_variation_gate",
    "one_release_stale_claims",
    "current_unrevised_claims",
    "direction_flip",
    "same_clock_forced_long",
)
COLUMNS = (
    "candidate",
    "control",
    "split",
    "reference_date",
    "first_vintage_date",
    "revised_vintage_date",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "claims_change",
    "claims_first_change",
    "btc_variation",
    "btc_variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_prior_rank(values: pd.Series, maximum: int = 270, minimum: int = 180) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = np.asarray(history[-maximum:], dtype=float)
        if np.isfinite(current) and len(prior) >= minimum:
            output.at[index] = ((prior < current).sum() + 0.5 * (prior == current).sum()) / len(prior)
        if np.isfinite(current):
            history.append(float(current))
    return output


def score_states(source: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    states = source.copy()
    states["reference_date"] = pd.to_datetime(states.reference_date, utc=True)
    states["first_vintage_date"] = pd.to_datetime(states.first_vintage_date, utc=True)
    states["revised_vintage_date"] = pd.to_datetime(states.revised_vintage_date, utc=True)
    states["icsa_first"] = pd.to_numeric(states.icsa_first, errors="coerce")
    states["icsa_revised"] = pd.to_numeric(states.icsa_revised, errors="coerce")
    states = states.sort_values("reference_date").reset_index(drop=True)
    if states.reference_date.duplicated().any():
        raise RuntimeError("HVICLR duplicate reference date")
    indexed = states.set_index("reference_date")
    prior_dates = states.reference_date - pd.Timedelta(days=28)
    states["claims_change"] = states.icsa_revised - indexed.icsa_revised.reindex(prior_dates).to_numpy()
    states["claims_first_change"] = states.icsa_first - indexed.icsa_first.reindex(prior_dates).to_numpy()
    decision_dates = states.revised_vintage_date.dt.tz_convert(None).dt.normalize()
    states["decision_time"] = (
        decision_dates.dt.tz_localize("America/New_York")
        + pd.Timedelta(hours=8, minutes=35)
    ).dt.tz_convert("UTC")

    candles = market.copy()
    candles["date"] = pd.to_datetime(candles.date, utc=True)
    candles = candles.sort_values("date").set_index("date")
    close = pd.to_numeric(candles.close, errors="coerce")
    valid = np.isfinite(close) & close.gt(0)
    contiguous = candles.index.to_series().diff().eq(pd.Timedelta(minutes=5))
    squared_returns = np.log(close / close.shift(1)).pow(2)
    variation = np.sqrt(squared_returns.rolling(2016, min_periods=2016).sum())
    complete = valid.rolling(2017, min_periods=2017).sum().eq(2017) & contiguous.rolling(2016, min_periods=2016).sum().eq(2016)
    lookup = pd.DataFrame(
        {"decision_time": candles.index + pd.Timedelta(minutes=5), "btc_variation": variation.where(complete).to_numpy()}
    )
    states = states.merge(lookup, on="decision_time", how="left", validate="one_to_one")
    states["state_valid"] = (
        np.isfinite(states[["icsa_revised", "claims_change", "btc_variation"]]).all(axis=1)
        & states.claims_change.ne(0)
    )
    states["btc_variation_rank"] = strict_prior_rank(states.btc_variation.where(states.state_valid))
    return states


def build_clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    if control == "one_release_stale_claims":
        used = states.set_index("reference_date").reindex(states.reference_date - pd.Timedelta(days=7)).copy()
        used["reference_date"] = used.index
        used = used.reset_index(drop=True)
        used.index = states.index
    else:
        used = states
    change_column = "claims_first_change" if control == "current_unrevised_claims" else "claims_change"
    valid = np.isfinite(used[change_column]) & used[change_column].ne(0) & np.isfinite(states.btc_variation)
    variation_gate = pd.Series(True, index=states.index) if control == "no_variation_gate" else states.btc_variation_rank.ge(0.65)
    active = valid & variation_gate
    side = -np.sign(used[change_column]).fillna(0).astype(int)
    if control == "direction_flip":
        side = -side
    elif control == "same_clock_forced_long":
        side = pd.Series(1, index=states.index)
    rows = []
    for index in states.index[active]:
        decision = pd.Timestamp(states.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=72)
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        source = used.loc[index]
        rows.append(
            {
                "candidate": prereg.POLICY_ID,
                "control": control,
                "split": split,
                "reference_date": source.reference_date,
                "first_vintage_date": source.first_vintage_date,
                "revised_vintage_date": source.revised_vintage_date,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(side.at[index]),
                "claims_change": float(source[change_column]),
                "claims_first_change": float(source.claims_first_change),
                "btc_variation": float(states.at[index, "btc_variation"]),
                "btc_variation_rank": float(states.at[index, "btc_variation_rank"]),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(selected),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(months.max()) / len(selected),
    }


def run() -> dict[str, Any]:
    bindings = {
        prereg.DEFAULT_OUTPUT: PREREG_SHA,
        prereg.SOURCE: SOURCE_SHA,
        SOURCE_MANIFEST: SOURCE_MANIFEST_SHA,
        HELPER: HELPER_SHA,
        prereg.MARKET: prereg.MARKET_SHA,
    }
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise RuntimeError(f"HVICLR binding drift: {path}")
    source = pd.read_csv(prereg.SOURCE)
    market, market_source = load_market()
    states = score_states(source, market)
    primary = build_clock(states)
    controls = {name: build_clock(states, name) for name in CONTROLS}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, STATE)
    _write_gzip_csv(primary, CLOCK)
    for name, clock in controls.items():
        _write_gzip_csv(clock, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {
        key: value
        for name, values in support.items()
        for key, value in (
            (f"{name}_minimum_events", values["events"] >= MINIMUM[name]),
            (f"{name}_side_balance", values["minority_side_share"] >= 0.2),
            (f"{name}_month_concentration", values["max_month_share"] <= 0.45),
        )
    }
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hviclr_72_source_support_v1",
        "policy_id": prereg.POLICY_ID,
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "bindings": {str(path): expected for path, expected in bindings.items()},
        "market_source": market_source,
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "source_state": {"path": str(STATE), "sha256": sha(STATE), "rows": len(states)},
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(clock),
                "promotion_authorized": False,
            }
            for name, clock in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
