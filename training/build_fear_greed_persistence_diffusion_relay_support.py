"""Materialize source-only FGPDR-24 support clocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_fear_greed_persistence_diffusion_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


PREREG_SHA = "1c1f43809a0b0ae3dea0e4384894cdb1808fa4ee2c9554d956bb0c6ee8cf5748"
SENTIMENT = Path(
    "data/fear_greed_extremity_reversal_sources_2023_2026/fear_greed_daily.csv.gz"
)
SENTIMENT_SHA = "a50769db6ca15b9cbb538b4f03fd71956a42a3ca418a7628d8ba0c63d0b8f1dd"
SENTIMENT_MANIFEST = SENTIMENT.parent / "manifest.json"
SENTIMENT_MANIFEST_SHA = (
    "afa0f674838270d1945b83278478d2111928f181de07ae8d020ed1c4bc406302"
)
PRICE = Path("data/options_oi_chase_exhaustion_sources_2023_2026/btc_completed_hour.csv.gz")
PRICE_SHA = "f075a882b80fc1d050aacd9abd417d4be6b6511c4307e39c98ef25f08822c496"
PRICE_MANIFEST = PRICE.parent / "manifest.json"
PRICE_MANIFEST_SHA = "3e350d16da72da7b60d9e91fbfb1ff4c2e13e5cb954b52b19ceaddf8c4f0e66d"

SOURCE_DIR = Path("data/fear_greed_persistence_diffusion_relay_sources_2023_2026")
FEATURES = SOURCE_DIR / "preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/fear_greed_persistence_diffusion_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/fear_greed_persistence_diffusion_relay_controls_2023_2026")
RESULT = Path("results/fear_greed_persistence_diffusion_relay_support_2026-08-09.json")

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
    "final": (
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-08-01T00:00:00Z"),
    ),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_volatility_gate",
    "no_persistence_magnitude_tail",
    "two_day_persistence",
    "one_day_stale_persistence",
    "direction_flip",
)
COLUMNS = (
    "candidate",
    "control",
    "split",
    "sentiment_date",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "fear_greed_value",
    "cumulative_change_3d",
    "persistence_magnitude_rank",
    "btc_realized_variation",
    "btc_variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def strict_prior_midrank(
    values: pd.Series, lookback: int = 180, minimum: int = 90
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            output.at[index] = (
                np.sum(array < current) + 0.5 * np.sum(array == current)
            ) / len(array)
        if math.isfinite(current):
            history.append(float(current))
    return output


def _sentiment_features() -> pd.DataFrame:
    source = pd.read_csv(SENTIMENT, compression="gzip")
    source["sentiment_date"] = pd.to_datetime(source.sentiment_date, utc=True)
    source["fear_greed_value"] = pd.to_numeric(
        source.fear_greed_value, errors="coerce"
    )
    source = source.sort_values("sentiment_date").reset_index(drop=True)
    consecutive = source.sentiment_date.diff().eq(pd.Timedelta(days=1))
    source["change_1d"] = (source.fear_greed_value - source.fear_greed_value.shift(1)).where(
        consecutive
    )
    exact_four_dates = consecutive & consecutive.shift(1, fill_value=False) & consecutive.shift(
        2, fill_value=False
    )
    signs = pd.concat(
        [
            np.sign(source.change_1d),
            np.sign(source.change_1d.shift(1)),
            np.sign(source.change_1d.shift(2)),
        ],
        axis=1,
    )
    source["persistent_3d"] = (
        exact_four_dates
        & signs.ne(0).all(axis=1)
        & signs.eq(signs.iloc[:, 0], axis=0).all(axis=1)
    )
    source["cumulative_change_3d"] = (
        source.fear_greed_value - source.fear_greed_value.shift(3)
    ).where(source.persistent_3d)
    source["persistence_magnitude_rank"] = strict_prior_midrank(
        source.cumulative_change_3d.abs()
    )

    exact_three_dates = consecutive & consecutive.shift(1, fill_value=False)
    signs_2d = pd.concat(
        [np.sign(source.change_1d), np.sign(source.change_1d.shift(1))], axis=1
    )
    source["persistent_2d"] = (
        exact_three_dates
        & signs_2d.ne(0).all(axis=1)
        & signs_2d.eq(signs_2d.iloc[:, 0], axis=0).all(axis=1)
    )
    source["cumulative_change_2d"] = (
        source.fear_greed_value - source.fear_greed_value.shift(2)
    ).where(source.persistent_2d)
    source["persistence_2d_rank"] = strict_prior_midrank(
        source.cumulative_change_2d.abs()
    )
    source["decision_time"] = source.sentiment_date + pd.Timedelta(days=1)
    return source


def _btc_features() -> pd.DataFrame:
    price = pd.read_csv(PRICE, compression="gzip")
    price["hour_start"] = pd.to_datetime(price.hour_start, utc=True)
    price["open"] = pd.to_numeric(price.open, errors="coerce")
    price["close"] = pd.to_numeric(price.close, errors="coerce")
    price["valid"] = (
        price.source_valid.astype(str).str.lower().eq("true")
        & np.isfinite(price[["open", "close"]]).all(axis=1)
        & price[["open", "close"]].gt(0).all(axis=1)
    )
    price["hour_return"] = np.log(price.close / price.open)
    price["sentiment_date"] = price.hour_start.dt.floor("D")
    grouped = price.groupby("sentiment_date", as_index=False).agg(
        hours=("hour_start", "size"),
        first_hour=("hour_start", "min"),
        last_hour=("hour_start", "max"),
        all_valid=("valid", "all"),
        squared_return_sum=("hour_return", lambda values: float(np.square(values).sum())),
    )
    grouped["btc_valid"] = (
        grouped.hours.eq(24)
        & grouped.first_hour.eq(grouped.sentiment_date)
        & grouped.last_hour.eq(grouped.sentiment_date + pd.Timedelta(hours=23))
        & grouped.all_valid
        & np.isfinite(grouped.squared_return_sum)
    )
    grouped["btc_realized_variation"] = np.sqrt(grouped.squared_return_sum)
    grouped["btc_variation_rank"] = strict_prior_midrank(
        grouped.btc_realized_variation.where(grouped.btc_valid)
    )
    grouped["decision_time"] = grouped.sentiment_date + pd.Timedelta(days=1)
    return grouped


def features() -> pd.DataFrame:
    if (
        sha(SENTIMENT) != SENTIMENT_SHA
        or sha(SENTIMENT_MANIFEST) != SENTIMENT_MANIFEST_SHA
        or sha(PRICE) != PRICE_SHA
        or sha(PRICE_MANIFEST) != PRICE_MANIFEST_SHA
    ):
        raise RuntimeError("FGPDR source drift")
    sentiment = _sentiment_features()
    btc = _btc_features()
    frame = sentiment.merge(
        btc[
            [
                "sentiment_date",
                "decision_time",
                "btc_valid",
                "btc_realized_variation",
                "btc_variation_rank",
            ]
        ],
        on=["sentiment_date", "decision_time"],
        how="inner",
        validate="one_to_one",
    )
    frame["signal_valid"] = (
        frame.persistent_3d
        & frame.btc_valid
        & np.isfinite(
            frame[
                [
                    "cumulative_change_3d",
                    "persistence_magnitude_rank",
                    "btc_realized_variation",
                    "btc_variation_rank",
                ]
            ]
        ).all(axis=1)
        & frame.cumulative_change_3d.ne(0)
    )
    return frame


def conditions(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    cumulative = frame.cumulative_change_3d
    magnitude_rank = frame.persistence_magnitude_rank
    persistent = frame.persistent_3d
    valid = frame.signal_valid
    if control == "two_day_persistence":
        cumulative = frame.cumulative_change_2d
        magnitude_rank = frame.persistence_2d_rank
        persistent = frame.persistent_2d
        valid = frame.btc_valid & np.isfinite(
            frame[["cumulative_change_2d", "persistence_2d_rank", "btc_variation_rank"]]
        ).all(axis=1)
    elif control == "one_day_stale_persistence":
        cumulative = cumulative.shift(1)
        magnitude_rank = magnitude_rank.shift(1)
        persistent = persistent.shift(1, fill_value=False)
        valid = valid.shift(1, fill_value=False) & frame.btc_valid
    tail = (
        pd.Series(True, index=frame.index)
        if control == "no_persistence_magnitude_tail"
        else magnitude_rank.ge(0.60)
    )
    volatile = (
        pd.Series(True, index=frame.index)
        if control == "no_volatility_gate"
        else frame.btc_variation_rank.ge(0.65)
    )
    active = (
        valid
        & persistent
        & np.isfinite(cumulative)
        & cumulative.ne(0)
        & tail
        & volatile
    )
    side = np.sign(cumulative)
    if control == "direction_flip":
        side = -side
    return active, side


def clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = conditions(frame, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=24)
        if next_allowed is not None and entry < next_allowed:
            continue
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
        next_allowed = exit_time
        rows.append(
            {
                "candidate": "FGPDR-24",
                "control": control,
                "split": split,
                "sentiment_date": frame.at[index, "sentiment_date"],
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(side.at[index]),
                "fear_greed_value": float(frame.at[index, "fear_greed_value"]),
                "cumulative_change_3d": float(frame.at[index, "cumulative_change_3d"]),
                "persistence_magnitude_rank": float(
                    frame.at[index, "persistence_magnitude_rank"]
                ),
                "btc_realized_variation": float(
                    frame.at[index, "btc_realized_variation"]
                ),
                "btc_variation_rank": float(frame.at[index, "btc_variation_rank"]),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(clock_frame: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = clock_frame[clock_frame.split.eq(split)]
    if subset.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "minority_side_share": 0.0,
            "max_month_share": 0.0,
        }
    longs = int(subset.side.eq(1).sum())
    shorts = int(subset.side.eq(-1).sum())
    months = subset.entry_time.dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(subset),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(months.max()) / len(subset),
    }


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("FGPDR preregistration hash drift")
    frame = features()
    primary = clock(frame)
    controls = {name: clock(frame, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(frame, FEATURES)
    _write_gzip_csv(primary, CLOCK)
    for name, control in controls.items():
        _write_gzip_csv(control, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "fgpdr_24_preentry_sources_v1",
        "sentiment": {
            "path": str(SENTIMENT),
            "sha256": SENTIMENT_SHA,
            "manifest_path": str(SENTIMENT_MANIFEST),
            "manifest_sha256": SENTIMENT_MANIFEST_SHA,
        },
        "completed_btc": {
            "path": str(PRICE),
            "sha256": PRICE_SHA,
            "manifest_path": str(PRICE_MANIFEST),
            "manifest_sha256": PRICE_MANIFEST_SHA,
        },
        "features": {
            "path": str(FEATURES),
            "sha256": sha(FEATURES),
            "rows": len(frame),
        },
        "candidate_incidence_opened": False,
        "postentry_outcomes_opened": False,
        "no_imputation": True,
    }
    source_manifest = {
        **source_core,
        "manifest_hash": canonical_hash(source_core),
    }
    SOURCE_MANIFEST.write_text(
        json.dumps(source_manifest, indent=2, allow_nan=False) + "\n"
    )
    support = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, split_stats in support.items():
        checks[f"{name}_minimum_events"] = split_stats["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = split_stats["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = split_stats["max_month_share"] <= 0.45
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    passed = all(checks.values())
    core = {
        "protocol_version": "fgpdr_24_source_support_v1",
        "policy_id": "FGPDR-24",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST),
            "sha256": sha(SOURCE_MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(control),
                "promotion_authorized": False,
            }
            for name, control in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(
        json.dumps(
            {"passed": result["support_passed"], "support": result["support"]},
            indent=2,
        )
    )
