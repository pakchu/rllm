"""Build source-only HVMNSD-24 clocks before Gross9 or economics."""
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

from training import preregister_high_volatility_macro_news_sentiment_dispersion_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market


PREREG_SHA = "0f1d99f422b4b198d542773a6bdb198ae168df652b94b782f34f47c6419fbe5f"
HELPER = Path("training/build_scheduled_trend_concordance_relay_support.py")
HELPER_SHA = "8ca554d88506df277434f73e5eb8850426614a880110088eb91aaae3b23c154f"
END = pd.Timestamp("2026-08-01T00:00:00Z")
STATE_DIR = Path(
    "data/high_volatility_macro_news_sentiment_dispersion_relay_sources_2023_2026"
)
STATE = STATE_DIR / "daily_states.csv.gz"
CLOCK = Path(
    "data/high_volatility_macro_news_sentiment_dispersion_relay_clocks_2023_2026.csv.gz"
)
CONTROL_DIR = Path(
    "data/high_volatility_macro_news_sentiment_dispersion_relay_controls_2023_2026"
)
MANIFEST = STATE_DIR / "manifest.json"
RESULT = Path(
    "results/high_volatility_macro_news_sentiment_dispersion_relay_support_2026-08-13.json"
)
SPLITS = {
    "train": (
        pd.Timestamp("2023-07-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T00:00:00Z"),
    ),
    "test": (
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:00:00Z"),
    ),
    "eval": (
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T00:00:00Z"),
    ),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_btc_variation_gate",
    "one_week_stale_dispersion",
    "direction_flip",
    "same_clock_forced_long",
    "signed_tone_change",
)
COLUMNS = (
    "candidate",
    "control",
    "split",
    "source_day",
    "feature_source_day",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "current_dispersion",
    "prior_dispersion",
    "dispersion_change",
    "signed_tone_change",
    "btc_variation",
    "btc_variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = np.asarray(history[-270:], dtype=float)
        if np.isfinite(current) and len(prior) >= 180:
            result.at[index] = (
                (prior < current).sum() + 0.5 * (prior == current).sum()
            ) / len(prior)
        if np.isfinite(current):
            history.append(float(current))
    return result


def score_sentiment(sentiment: pd.DataFrame) -> pd.DataFrame:
    result = sentiment.copy()
    result["source_day"] = pd.to_datetime(result["date"], utc=True)
    result["news_sentiment"] = pd.to_numeric(
        result["news_sentiment"], errors="coerce"
    )
    result = result.sort_values("source_day").reset_index(drop=True)
    if result["source_day"].duplicated().any():
        raise RuntimeError("HVMNSD duplicate source day")
    lookup = result.set_index("source_day")["news_sentiment"]
    matrix = np.column_stack(
        [
            lookup.reindex(result["source_day"] - pd.Timedelta(days=lag)).to_numpy()
            for lag in range(14)
        ]
    )
    exact = np.isfinite(matrix).all(axis=1)
    current = np.std(matrix[:, :7], axis=1, ddof=0)
    prior = np.std(matrix[:, 7:], axis=1, ddof=0)
    result["current_dispersion"] = np.where(exact, current, np.nan)
    result["prior_dispersion"] = np.where(exact, prior, np.nan)
    positive = exact & (current > 0.0) & (prior > 0.0)
    change = np.full(len(result), np.nan, dtype=float)
    change[positive] = np.log(current[positive] / prior[positive])
    result["dispersion_change"] = change
    result["signed_tone_change"] = np.where(exact, matrix[:, 0] - matrix[:, 7], np.nan)
    result["dispersion_valid"] = positive & np.isfinite(change) & (change != 0.0)
    result["decision_time"] = result["source_day"] + pd.Timedelta(days=8)
    return result


def score_states(sentiment: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    result = score_sentiment(sentiment)
    prices = market.copy()
    prices["date"] = pd.to_datetime(prices["date"], utc=True)
    prices = prices.sort_values("date").set_index("date")
    close = pd.to_numeric(prices["close"], errors="coerce")
    valid_close = np.isfinite(close) & close.gt(0.0)
    exact_step = prices.index.to_series().diff().eq(pd.Timedelta(minutes=5))
    variation = np.sqrt(
        np.log(close / close.shift(1)).pow(2).rolling(288, min_periods=288).sum()
    )
    complete = (
        valid_close.rolling(289, min_periods=289).sum().eq(289)
        & exact_step.rolling(288, min_periods=288).sum().eq(288)
    )
    market_lookup = pd.DataFrame(
        {
            "decision_time": prices.index + pd.Timedelta(minutes=5),
            "btc_variation": variation.where(complete).to_numpy(),
        }
    )
    result = result.merge(
        market_lookup, on="decision_time", how="left", validate="one_to_one"
    )
    result["state_valid"] = result["dispersion_valid"].eq(True) & np.isfinite(
        result["btc_variation"]
    )
    result["btc_variation_rank"] = rank(
        result["btc_variation"].where(result["state_valid"])
    )
    return result


def build_clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    feature = states
    if control == "one_week_stale_dispersion":
        indexed = states.set_index("source_day")
        feature = indexed.reindex(states["source_day"] - pd.Timedelta(days=7)).copy()
        feature["source_day"] = feature.index
        feature = feature.reset_index(drop=True)
        feature.index = states.index
        dispersion_valid = feature["dispersion_valid"].eq(True)
    else:
        dispersion_valid = states["dispersion_valid"].eq(True)
    variation_valid = np.isfinite(states["btc_variation"])
    if control == "no_btc_variation_gate":
        variation_gate = pd.Series(True, index=states.index)
    else:
        variation_gate = states["btc_variation_rank"].ge(0.65)
    active = dispersion_valid & variation_valid & variation_gate
    side = -np.sign(feature["dispersion_change"]).fillna(0).astype(int)
    if control == "direction_flip":
        side = -side
    elif control == "same_clock_forced_long":
        side = pd.Series(1, index=states.index)
    elif control == "signed_tone_change":
        side = np.sign(feature["signed_tone_change"]).fillna(0).astype(int)
        active &= side.ne(0)
    rows: list[dict[str, Any]] = []
    for index in states.index[active]:
        decision = pd.Timestamp(states.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=24)
        split = next(
            (
                name
                for name, (start, end) in SPLITS.items()
                if entry >= start and exit_time <= end
            ),
            None,
        )
        if split is None:
            continue
        source = feature.loc[index]
        rows.append(
            {
                "candidate": prereg.POLICY_ID,
                "control": control,
                "split": split,
                "source_day": states.at[index, "source_day"],
                "feature_source_day": source["source_day"],
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(side.at[index]),
                "current_dispersion": float(source["current_dispersion"]),
                "prior_dispersion": float(source["prior_dispersion"]),
                "dispersion_change": float(source["dispersion_change"]),
                "signed_tone_change": float(source["signed_tone_change"]),
                "btc_variation": float(states.at[index, "btc_variation"]),
                "btc_variation_rank": float(states.at[index, "btc_variation_rank"]),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock["split"].eq(split)]
    if subset.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "minority_side_share": 0.0,
            "max_month_share": 0.0,
        }
    longs = int(subset["side"].eq(1).sum())
    shorts = int(subset["side"].eq(-1).sum())
    months = pd.to_datetime(subset["entry_time"], utc=True).dt.strftime("%Y-%m")
    return {
        "events": len(subset),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(months.value_counts().max()) / len(subset),
    }


def run() -> dict[str, Any]:
    bindings = {
        prereg.DEFAULT_OUTPUT: PREREG_SHA,
        prereg.SOURCE: prereg.SOURCE_SHA,
        prereg.DERIVED: prereg.DERIVED_SHA,
        prereg.SOURCE_MANIFEST: prereg.SOURCE_MANIFEST_SHA,
        HELPER: HELPER_SHA,
        prereg.MARKET: prereg.MARKET_SHA,
    }
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise RuntimeError(f"HVMNSD binding drift: {path}")
    sentiment = pd.read_csv(prereg.DERIVED)
    market, market_source = load_market()
    states = score_states(sentiment, market)
    primary = build_clock(states)
    controls = {name: build_clock(states, name) for name in CONTROLS}
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, STATE)
    _write_gzip_csv(primary, CLOCK)
    for name, clock in controls.items():
        _write_gzip_csv(clock, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {
        key: value
        for split, values in support.items()
        for key, value in (
            (f"{split}_minimum_events", values["events"] >= MINIMUM[split]),
            (f"{split}_side_balance", values["minority_side_share"] >= 0.2),
            (f"{split}_month_concentration", values["max_month_share"] <= 0.45),
        )
    }
    passed = all(checks.values())
    preregistration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    files = {
        "state": {"path": str(STATE), "sha256": sha(STATE), "rows": len(states)},
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
    }
    manifest_core = {
        "protocol_version": "hvmnsd_24_source_artifacts_v1",
        "policy_id": prereg.POLICY_ID,
        "files": files,
    }
    manifest = {**manifest_core, "manifest_hash": prereg.canonical_hash(manifest_core)}
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    core = {
        "protocol_version": "hvmnsd_24_source_support_v1",
        "policy_id": prereg.POLICY_ID,
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA,
            "manifest_hash": preregistration["manifest_hash"],
        },
        "bindings": {str(path): expected for path, expected in bindings.items()},
        "market_source": market_source,
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "source_artifacts": {**files, "manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST)}},
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}
    RESULT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    value = run()
    print(
        json.dumps(
            {"passed": value["support_passed"], "support": value["support"]},
            indent=2,
        )
    )
