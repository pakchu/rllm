"""Freeze outcome-blind clocks for COIN-M liquidation burst release (CLBR-24)."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from training.build_binance_aggtrade_microstructure import _write_gzip_csv


SOURCE = Path(
    "data/binance_coinm_liquidation_snapshot_btc_2023_2024/"
    "BTCUSD_PERP_liquidation_5m_2023-06-25_2024-10-14.csv.gz"
)
SOURCE_MANIFEST = Path(
    "results/binance_coinm_liquidation_snapshot_btc_2023_2024_manifest.json"
)
EXPECTED_SOURCE_SHA256 = (
    "a23b93d8567a589e9f045ae4a56393e493a8da2748c5a051804c9bdf9388ccc3"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "5d78686e7c40d69261f09bc77e27ff734f682abba4abb95c2291e8282380053e"
)
DEFAULT_CLOCKS = Path("data/coinm_liquidation_burst_release_clocks_2023_2024.csv.gz")
DEFAULT_RESULT = Path(
    "results/coinm_liquidation_burst_release_support_2026-07-19.json"
)
BAR_MINUTES = 5
ROLLING_DAYS = 14
ROLLING_BARS = ROLLING_DAYS * 24 * 12
MIN_POSITIVE_OBSERVATIONS = 200
BURST_QUANTILE = 0.975
MIN_EVENT_COUNT = 3
MIN_DOMINANCE = 0.8
RELEASE_RATIO = 0.25
MAX_COUNTERFLOW_RATIO = 0.10
HOLD_BARS = 24
STOP_BUFFER_BPS = 25.0
SPLITS = {
    "train": ("2023-06-25", "2023-10-15"),
    "test": ("2023-10-15", "2024-04-15"),
    "eval": ("2024-04-15", "2024-10-15"),
}
MIN_SUPPORT = {"train": 40, "test": 100, "eval": 100}
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Config:
    source: str = str(SOURCE)
    source_manifest: str = str(SOURCE_MANIFEST)
    clocks: str = str(DEFAULT_CLOCKS)
    result: str = str(DEFAULT_RESULT)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_source(cfg: Config) -> pd.DataFrame:
    source = Path(cfg.source)
    manifest_path = Path(cfg.source_manifest)
    if sha256_file(source) != EXPECTED_SOURCE_SHA256:
        raise ValueError("liquidation source hash mismatch")
    if sha256_file(manifest_path) != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("liquidation source manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["protocol"]["outcomes_opened"] is not False:
        raise ValueError("liquidation source is not outcome blind")
    if manifest["file"]["sha256"] != EXPECTED_SOURCE_SHA256:
        raise ValueError("liquidation source manifest points to another panel")

    frame = pd.read_csv(
        source,
        compression="gzip",
        parse_dates=["date", "feature_available_time"],
    )
    expected = pd.Series(
        pd.date_range("2023-06-25", "2024-10-15", freq="5min", inclusive="left"),
        name="date",
    )
    if not frame["date"].equals(expected):
        raise ValueError("liquidation source grid mismatch")
    delay = cast(pd.Series, frame["feature_available_time"].sub(frame["date"]))
    if not bool(delay.eq(pd.Timedelta(minutes=5, seconds=1)).all()):
        raise ValueError("liquidation source availability mismatch")
    return frame


def derive_release_state(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive one fixed source-only release rule using strictly prior thresholds."""

    output = frame.copy()
    valid = cast(pd.Series, output["source_valid"]).astype(bool)
    positive_notional = cast(pd.Series, output["total_liquidation_usd"]).where(
        valid & cast(pd.Series, output["event_count"]).gt(0)
    )
    threshold = (
        positive_notional.shift(1)
        .rolling(ROLLING_BARS, min_periods=MIN_POSITIVE_OBSERVATIONS)
        .quantile(BURST_QUANTILE)
    )
    output["prior_burst_threshold_usd"] = threshold
    imbalance = cast(pd.Series, output["liquidation_imbalance"])
    burst = (
        valid
        & cast(pd.Series, output["total_liquidation_usd"]).ge(threshold)
        & cast(pd.Series, output["event_count"]).ge(MIN_EVENT_COUNT)
        & imbalance.abs().ge(MIN_DOMINANCE)
    )
    previous_total = cast(pd.Series, output["total_liquidation_usd"]).shift(1)
    previous_imbalance = imbalance.shift(1)
    counterflow = pd.Series(0.0, index=output.index)
    prior_long_liquidation = previous_imbalance.lt(0.0)
    prior_short_liquidation = previous_imbalance.gt(0.0)
    counterflow.loc[prior_long_liquidation] = cast(
        pd.Series, output.loc[prior_long_liquidation, "short_liquidation_usd"]
    )
    counterflow.loc[prior_short_liquidation] = cast(
        pd.Series, output.loc[prior_short_liquidation, "long_liquidation_usd"]
    )
    release = (
        burst.shift(1, fill_value=False)
        & valid
        & cast(pd.Series, output["total_liquidation_usd"]).le(
            previous_total * RELEASE_RATIO
        )
        & counterflow.le(previous_total * MAX_COUNTERFLOW_RATIO)
    )
    output["is_burst"] = burst
    output["counterflow_usd"] = counterflow
    output["is_release"] = release
    return output


def _clock_row(frame: pd.DataFrame, index: int, split: str) -> dict[str, Any]:
    current = frame.loc[index]
    burst = frame.loc[index - 1]
    direction = 1 if float(burst["liquidation_imbalance"]) < 0.0 else -1
    feature_available = cast(pd.Timestamp, current["feature_available_time"])
    entry_time = cast(pd.Timestamp, current["date"]) + pd.Timedelta(
        minutes=2 * BAR_MINUTES
    )
    stop_anchor = float(
        burst[
            "min_snapshot_average_price"
            if direction > 0
            else "max_snapshot_average_price"
        ]
    )
    stop_price = stop_anchor * (
        1.0 - STOP_BUFFER_BPS / 10_000.0
        if direction > 0
        else 1.0 + STOP_BUFFER_BPS / 10_000.0
    )
    return {
        "candidate": "CLBR-24",
        "split": split,
        "burst_time": burst["date"],
        "release_time": current["date"],
        "feature_available_time": feature_available,
        "entry_time": entry_time,
        "planned_exit_time": entry_time
        + pd.Timedelta(minutes=BAR_MINUTES * HOLD_BARS),
        "direction": direction,
        "stop_anchor": stop_anchor,
        "stop_price": stop_price,
        "burst_total_liquidation_usd": float(burst["total_liquidation_usd"]),
        "burst_threshold_usd": float(burst["prior_burst_threshold_usd"]),
        "burst_imbalance": float(burst["liquidation_imbalance"]),
        "burst_event_count": int(burst["event_count"]),
        "burst_price_range_bps": float(burst["snapshot_price_range_bps"]),
        "burst_closing_location": float(burst["snapshot_price_closing_location"]),
        "release_total_liquidation_usd": float(current["total_liquidation_usd"]),
        "release_counterflow_usd": float(current["counterflow_usd"]),
    }


def build_clocks(frame: pd.DataFrame) -> pd.DataFrame:
    state = derive_release_state(frame)
    rows: list[dict[str, Any]] = []
    for split, (start_text, end_text) in SPLITS.items():
        start = cast(pd.Timestamp, pd.Timestamp(start_text))
        end = cast(pd.Timestamp, pd.Timestamp(end_text))
        release = cast(pd.Series, state["is_release"]).astype(bool)
        indices = state.index[release & state["date"].ge(start) & state["date"].lt(end)]
        next_entry_allowed = start
        for raw_index in indices:
            index = int(raw_index)
            row = _clock_row(state, index, split)
            entry = cast(pd.Timestamp, row["entry_time"])
            exit_time = cast(pd.Timestamp, row["planned_exit_time"])
            if entry < next_entry_allowed or exit_time > end:
                continue
            rows.append(row)
            next_entry_allowed = exit_time
    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError("CLBR-24 produced no source-only clocks")
    if output["entry_time"].duplicated().any():
        raise ValueError("CLBR-24 produced duplicate entries")
    if not output["entry_time"].is_monotonic_increasing:
        raise ValueError("CLBR-24 entries are not monotonic")
    return output


def build(cfg: Config) -> dict[str, Any]:
    source = load_source(cfg)
    clocks = build_clocks(source)
    clock_path = Path(cfg.clocks)
    _write_gzip_csv(clocks, clock_path)
    support: dict[str, Any] = {}
    for split in SPLITS:
        subset = clocks[clocks["split"].eq(split)]
        direction = cast(pd.Series, subset["direction"])
        support[split] = {
            "count": int(len(subset)),
            "long": int(direction.gt(0).sum()),
            "short": int(direction.lt(0).sum()),
            "minimum_required": MIN_SUPPORT[split],
            "passes": int(len(subset)) >= MIN_SUPPORT[split],
        }
    core = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "name": "COIN-M liquidation burst release CLBR-24",
            "outcomes_opened": False,
            "candidate_count": 1,
            "candidate_repair_after_train": False,
            "test_and_eval_clocks_opened_source_only": True,
            "market_prices_opened": False,
        },
        "config": asdict(cfg),
        "source": {
            "path": str(SOURCE),
            "sha256": EXPECTED_SOURCE_SHA256,
            "manifest": str(SOURCE_MANIFEST),
            "manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        },
        "parameters": {
            "rolling_days": ROLLING_DAYS,
            "rolling_bars": ROLLING_BARS,
            "minimum_positive_observations": MIN_POSITIVE_OBSERVATIONS,
            "burst_quantile": BURST_QUANTILE,
            "minimum_event_count": MIN_EVENT_COUNT,
            "minimum_absolute_imbalance": MIN_DOMINANCE,
            "release_ratio": RELEASE_RATIO,
            "maximum_counterflow_ratio": MAX_COUNTERFLOW_RATIO,
            "hold_bars": HOLD_BARS,
            "bar_minutes": BAR_MINUTES,
            "stop_buffer_bps": STOP_BUFFER_BPS,
        },
        "splits": SPLITS,
        "support": support,
        "clocks": {
            "path": str(clock_path),
            "sha256": sha256_file(clock_path),
            "rows": int(len(clocks)),
            "first_entry": str(clocks["entry_time"].min()),
            "last_exit": str(clocks["planned_exit_time"].max()),
        },
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    result_path = Path(cfg.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=Config.source)
    parser.add_argument("--source-manifest", default=Config.source_manifest)
    parser.add_argument("--clocks", default=Config.clocks)
    parser.add_argument("--result", default=Config.result)
    result = build(Config(**vars(parser.parse_args())))
    print(
        json.dumps(
            {
                "outcomes_opened": result["protocol"]["outcomes_opened"],
                "support": result["support"],
                "clocks": result["clocks"],
                "manifest_hash": result["manifest_hash"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
