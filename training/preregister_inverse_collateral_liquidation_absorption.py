"""Freeze source-only clocks for inverse-collateral liquidation absorption."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.build_binance_aggtrade_microstructure import _write_gzip_csv  # noqa: E402
from training.preregister_coinm_liquidation_burst_release import (  # noqa: E402
    Config as LiquidationConfig,
)
from training.preregister_coinm_liquidation_burst_release import (  # noqa: E402
    load_source as load_liquidation_source,
)


ACTIVITY_SOURCE = Path(
    "data/binance_um_activity_5m_2023_2024/"
    "BTCUSDT_5m_activity_2023-06-25_2024-10-15_exclusive.csv.gz"
)
ACTIVITY_MANIFEST = Path("results/binance_um_activity_5m_2023_2024_manifest.json")
EXPECTED_ACTIVITY_SHA256 = (
    "dde78b3b14ca1689abaacd00e9085a81f63429ee077de7e22d7e108ad4eb697e"
)
EXPECTED_ACTIVITY_MANIFEST_SHA256 = (
    "ee22fa7facc901b4bac383f753391c527eafadd71e521dcfd9187de6f2d4b493"
)
CLBR_CLOCKS = Path("data/coinm_liquidation_burst_release_clocks_2023_2024.csv.gz")
EXPECTED_CLBR_CLOCKS_SHA256 = (
    "df619a5ffc3b849d3c35fc7112641c33105ba76c81cbb7b8c7f3c975fd80bee0"
)
DEFAULT_CLOCKS = Path(
    "data/inverse_collateral_liquidation_absorption_clocks_2023_2024.csv.gz"
)
DEFAULT_RESULT = Path(
    "results/inverse_collateral_liquidation_absorption_support_2026-07-19.json"
)
BAR_MINUTES = 5
WAVE_BARS = 12
REFERENCE_DAYS = 14
REFERENCE_BARS = REFERENCE_DAYS * 24 * 12
MIN_POSITIVE_OBSERVATIONS = 200
WAVE_QUANTILE = 0.90
MIN_ABSOLUTE_LIQUIDATION_IMBALANCE = 0.60
HOLD_BARS = 12
SPLITS = {
    "train": ("2023-06-25", "2023-10-15"),
    "test": ("2023-10-15", "2024-04-15"),
    "eval": ("2024-04-15", "2024-10-15"),
}
MIN_SUPPORT = {"train": 25, "test": 90, "eval": 90}
MIN_SIDE_SUPPORT = {"train": 8, "test": 20, "eval": 20}
MAX_MONTH_SHARE = {"train": 0.35, "test": 0.25, "eval": 0.30}
MAX_CLBR_ENTRY_JACCARD = 0.10
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Config:
    activity_source: str = str(ACTIVITY_SOURCE)
    activity_manifest: str = str(ACTIVITY_MANIFEST)
    clocks: str = str(DEFAULT_CLOCKS)
    result: str = str(DEFAULT_RESULT)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_activity_source(cfg: Config) -> pd.DataFrame:
    source = Path(cfg.activity_source)
    manifest_path = Path(cfg.activity_manifest)
    if sha256_file(source) != EXPECTED_ACTIVITY_SHA256:
        raise ValueError("USD-M activity source hash mismatch")
    if sha256_file(manifest_path) != EXPECTED_ACTIVITY_MANIFEST_SHA256:
        raise ValueError("USD-M activity manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest.get("protocol", {})
    if protocol.get("outcomes_opened") is not False:
        raise ValueError("USD-M activity source opened outcomes")
    if protocol.get("activity_only") is not True:
        raise ValueError("USD-M source is not activity-only")
    if protocol.get("prices_retained") is not False:
        raise ValueError("USD-M activity source retained prices")
    if manifest.get("file", {}).get("sha256") != EXPECTED_ACTIVITY_SHA256:
        raise ValueError("USD-M manifest points to another source")

    frame = pd.read_csv(
        source,
        compression="gzip",
        parse_dates=["date", "feature_available_time"],
    )
    expected = pd.Series(
        pd.date_range("2023-06-25", "2024-10-15", freq="5min", inclusive="left"),
        name="date",
    )
    if not cast(pd.Series, frame["date"]).equals(expected):
        raise ValueError("USD-M activity source grid mismatch")
    delay = cast(pd.Series, frame["feature_available_time"]).sub(frame["date"])
    if not bool(delay.eq(pd.Timedelta(minutes=5, seconds=1)).all()):
        raise ValueError("USD-M activity availability mismatch")
    values = frame[
        [
            "quote_asset_volume",
            "taker_buy_quote",
            "taker_sell_quote",
            "taker_imbalance",
            "number_of_trades",
        ]
    ].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("USD-M activity source contains non-finite values")
    if bool(cast(pd.Series, frame["quote_asset_volume"]).lt(0.0).any()):
        raise ValueError("USD-M activity source contains negative volume")
    return frame


def load_sources(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    liquidation = load_liquidation_source(LiquidationConfig())
    activity = load_activity_source(cfg)
    if not cast(pd.Series, liquidation["date"]).equals(activity["date"]):
        raise ValueError("COIN-M liquidation and USD-M activity grids differ")
    return liquidation, activity


def derive_wave_state(
    liquidation: pd.DataFrame, activity: pd.DataFrame
) -> pd.DataFrame:
    """Create the frozen 60-minute source state with a strictly prior threshold."""

    if not cast(pd.Series, liquidation["date"]).equals(activity["date"]):
        raise ValueError("source grids differ")
    output = liquidation[
        [
            "date",
            "feature_available_time",
            "source_valid",
            "event_count",
            "total_liquidation_usd",
            "signed_liquidation_usd",
        ]
    ].copy()
    liquidation_available = cast(pd.Series, output["feature_available_time"]).copy()
    output = cast(pd.DataFrame, output.drop(columns=["feature_available_time"]))
    output["liquidation_available_time"] = liquidation_available
    output["activity_available_time"] = activity["feature_available_time"]
    output["quote_asset_volume"] = activity["quote_asset_volume"].to_numpy(float)
    output["signed_taker_quote"] = (
        activity["taker_buy_quote"].to_numpy(float)
        - activity["taker_sell_quote"].to_numpy(float)
    )
    output["feature_available_time"] = output[
        ["liquidation_available_time", "activity_available_time"]
    ].max(axis=1)

    valid = cast(pd.Series, output["source_valid"]).astype(bool)
    valid_count = cast(
        pd.Series,
        valid.astype(int).rolling(WAVE_BARS, min_periods=WAVE_BARS).sum(),
    )
    output["wave_source_valid"] = valid_count.eq(WAVE_BARS)
    output["wave_event_count"] = (
        cast(pd.Series, output["event_count"])
        .rolling(WAVE_BARS, min_periods=WAVE_BARS)
        .sum()
    )
    output["wave_total_liquidation_usd"] = (
        cast(pd.Series, output["total_liquidation_usd"])
        .rolling(WAVE_BARS, min_periods=WAVE_BARS)
        .sum()
    )
    output["wave_signed_liquidation_usd"] = (
        cast(pd.Series, output["signed_liquidation_usd"])
        .rolling(WAVE_BARS, min_periods=WAVE_BARS)
        .sum()
    )
    total = cast(pd.Series, output["wave_total_liquidation_usd"])
    output["wave_liquidation_imbalance"] = cast(
        pd.Series, output["wave_signed_liquidation_usd"]
    ).div(total.where(total.gt(0.0)))
    output["wave_quote_asset_volume"] = (
        cast(pd.Series, output["quote_asset_volume"])
        .rolling(WAVE_BARS, min_periods=WAVE_BARS)
        .sum()
    )
    output["wave_signed_taker_quote"] = (
        cast(pd.Series, output["signed_taker_quote"])
        .rolling(WAVE_BARS, min_periods=WAVE_BARS)
        .sum()
    )
    quote = cast(pd.Series, output["wave_quote_asset_volume"])
    output["wave_usdm_taker_imbalance"] = cast(
        pd.Series, output["wave_signed_taker_quote"]
    ).div(quote.where(quote.gt(0.0)))

    positive_wave = total.where(
        cast(pd.Series, output["wave_source_valid"]).astype(bool) & total.gt(0.0)
    )
    output["prior_wave_threshold_usd"] = (
        positive_wave.shift(1)
        .rolling(REFERENCE_BARS, min_periods=MIN_POSITIVE_OBSERVATIONS)
        .quantile(WAVE_QUANTILE)
    )
    liquidation_imbalance = cast(pd.Series, output["wave_liquidation_imbalance"])
    direction = pd.Series(
        np.where(liquidation_imbalance.lt(0.0), 1, -1),
        index=output.index,
        dtype="int64",
    )
    direction = direction.where(total.gt(0.0), 0)
    output["direction"] = direction
    output["absorption_alignment"] = direction.mul(
        cast(pd.Series, output["wave_usdm_taker_imbalance"])
    )
    output["is_candidate"] = (
        cast(pd.Series, output["wave_source_valid"]).astype(bool)
        & cast(pd.Series, output["wave_event_count"]).ge(1.0)
        & total.ge(output["prior_wave_threshold_usd"])
        & liquidation_imbalance.abs().ge(MIN_ABSOLUTE_LIQUIDATION_IMBALANCE)
        & cast(pd.Series, output["absorption_alignment"]).gt(0.0)
        & direction.ne(0)
    )
    return output


def _clock_row(state: pd.DataFrame, index: int, split: str) -> dict[str, Any]:
    current = state.loc[index]
    last_bar_open = cast(pd.Timestamp, current["date"])
    wave_completed = last_bar_open + pd.Timedelta(minutes=BAR_MINUTES)
    feature_available = cast(pd.Timestamp, current["feature_available_time"])
    entry_time = last_bar_open + pd.Timedelta(minutes=2 * BAR_MINUTES)
    if entry_time <= feature_available:
        raise ValueError("entry must follow source availability")
    return {
        "candidate": "ICLA-60",
        "split": split,
        "first_bar_open_time": last_bar_open
        - pd.Timedelta(minutes=BAR_MINUTES * (WAVE_BARS - 1)),
        "last_bar_open_time": last_bar_open,
        "wave_completed_time": wave_completed,
        "feature_available_time": feature_available,
        "entry_time": entry_time,
        "planned_exit_time": entry_time
        + pd.Timedelta(minutes=BAR_MINUTES * HOLD_BARS),
        "direction": int(current["direction"]),
        "wave_event_count": int(current["wave_event_count"]),
        "wave_total_liquidation_usd": float(
            current["wave_total_liquidation_usd"]
        ),
        "wave_threshold_usd": float(current["prior_wave_threshold_usd"]),
        "wave_liquidation_imbalance": float(
            current["wave_liquidation_imbalance"]
        ),
        "wave_quote_asset_volume": float(current["wave_quote_asset_volume"]),
        "wave_usdm_taker_imbalance": float(
            current["wave_usdm_taker_imbalance"]
        ),
        "absorption_alignment": float(current["absorption_alignment"]),
    }


def build_clocks(state: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidates = cast(pd.Series, state["is_candidate"]).astype(bool)
    for split, (start_text, end_text) in SPLITS.items():
        start = cast(pd.Timestamp, pd.Timestamp(start_text))
        end = cast(pd.Timestamp, pd.Timestamp(end_text))
        indices = state.index[
            candidates & state["date"].ge(start) & state["date"].lt(end)
        ]
        next_entry_allowed = start
        for raw_index in indices:
            row = _clock_row(state, int(raw_index), split)
            entry_time = cast(pd.Timestamp, row["entry_time"])
            exit_time = cast(pd.Timestamp, row["planned_exit_time"])
            if entry_time < next_entry_allowed or exit_time >= end:
                continue
            rows.append(row)
            next_entry_allowed = exit_time
    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError("ICLA-60 produced no source-only clocks")
    entries = cast(pd.Series, output["entry_time"])
    if entries.duplicated().any() or not entries.is_monotonic_increasing:
        raise ValueError("ICLA-60 entries are invalid")
    exits = cast(pd.Series, output["planned_exit_time"])
    if len(output) > 1 and not entries.iloc[1:].reset_index(drop=True).ge(
        exits.iloc[:-1].reset_index(drop=True)
    ).all():
        raise ValueError("ICLA-60 clocks overlap")
    return output


def _clock_overlap(clocks: pd.DataFrame) -> dict[str, Any]:
    if sha256_file(CLBR_CLOCKS) != EXPECTED_CLBR_CLOCKS_SHA256:
        raise ValueError("CLBR comparison clock hash mismatch")
    clbr = pd.read_csv(CLBR_CLOCKS, compression="gzip", parse_dates=["entry_time"])
    primary_entries = set(cast(pd.Series, clocks["entry_time"]).tolist())
    clbr_entries = set(cast(pd.Series, clbr["entry_time"]).tolist())
    intersection = primary_entries.intersection(clbr_entries)
    union = primary_entries.union(clbr_entries)
    return {
        "comparison": "CLBR-24 exact entry clock",
        "clbr_clock_sha256": EXPECTED_CLBR_CLOCKS_SHA256,
        "primary_entries": len(primary_entries),
        "clbr_entries": len(clbr_entries),
        "intersection_entries": len(intersection),
        "entry_jaccard": len(intersection) / len(union) if union else 0.0,
        "maximum_entry_jaccard_allowed": MAX_CLBR_ENTRY_JACCARD,
    }


def build(cfg: Config) -> dict[str, Any]:
    liquidation, activity = load_sources(cfg)
    state = derive_wave_state(liquidation, activity)
    clocks = build_clocks(state)
    clock_path = Path(cfg.clocks)
    _write_gzip_csv(clocks, clock_path)

    support: dict[str, Any] = {}
    for split in SPLITS:
        subset = cast(pd.DataFrame, clocks.loc[clocks["split"].eq(split)].copy())
        directions = cast(pd.Series, subset["direction"])
        month_share = (
            cast(pd.Series, subset["entry_time"])
            .dt.to_period("M")
            .value_counts(normalize=True)
        )
        maximum_month_share = float(month_share.max()) if len(month_share) else 1.0
        long_count = int(directions.gt(0).sum())
        short_count = int(directions.lt(0).sum())
        checks = {
            "minimum_total": len(subset) >= MIN_SUPPORT[split],
            "minimum_long": long_count >= MIN_SIDE_SUPPORT[split],
            "minimum_short": short_count >= MIN_SIDE_SUPPORT[split],
            "maximum_month_share": maximum_month_share <= MAX_MONTH_SHARE[split],
        }
        support[split] = {
            "count": int(len(subset)),
            "long": long_count,
            "short": short_count,
            "minimum_required": MIN_SUPPORT[split],
            "minimum_per_side": MIN_SIDE_SUPPORT[split],
            "maximum_month_share": maximum_month_share,
            "maximum_month_share_allowed": MAX_MONTH_SHARE[split],
            "checks": checks,
            "passes": bool(all(checks.values())),
        }

    clock_overlap = _clock_overlap(clocks)
    novelty_passes = (
        float(clock_overlap["entry_jaccard"]) <= MAX_CLBR_ENTRY_JACCARD
    )
    core: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "name": "inverse-collateral liquidation absorption ICLA-60",
            "outcomes_opened": False,
            "execution_prices_opened": False,
            "funding_opened": False,
            "return_labels_constructed": False,
            "candidate_count": 1,
            "candidate_repair_after_outcomes": False,
            "all_stage_clocks_opened_source_only": True,
            "later_stage_outcomes_opened": False,
        },
        "config": asdict(cfg),
        "sources": {
            "coinm_liquidation": {
                "path": str(LiquidationConfig.source),
                "source_manifest": str(LiquidationConfig.source_manifest),
            },
            "usdm_activity": {
                "path": cfg.activity_source,
                "sha256": EXPECTED_ACTIVITY_SHA256,
                "manifest": cfg.activity_manifest,
                "manifest_sha256": EXPECTED_ACTIVITY_MANIFEST_SHA256,
            },
        },
        "parameters": {
            "bar_minutes": BAR_MINUTES,
            "wave_bars": WAVE_BARS,
            "reference_days": REFERENCE_DAYS,
            "reference_bars": REFERENCE_BARS,
            "minimum_positive_observations": MIN_POSITIVE_OBSERVATIONS,
            "wave_quantile": WAVE_QUANTILE,
            "minimum_absolute_liquidation_imbalance": (
                MIN_ABSOLUTE_LIQUIDATION_IMBALANCE
            ),
            "absorption_alignment": "fade direction times USD-M taker imbalance > 0",
            "hold_bars": HOLD_BARS,
            "nonoverlap": True,
        },
        "splits": SPLITS,
        "support": support,
        "clock_overlap": {**clock_overlap, "passes": novelty_passes},
        "clocks": {
            "path": str(clock_path),
            "sha256": sha256_file(clock_path),
            "rows": int(len(clocks)),
            "first_entry": str(clocks["entry_time"].min()),
            "last_exit": str(clocks["planned_exit_time"].max()),
        },
    }
    core["support_passes"] = bool(
        all(item["passes"] for item in support.values()) and novelty_passes
    )
    result = {**core, "manifest_hash": canonical_hash(core)}
    result_path = Path(cfg.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity-source", default=Config.activity_source)
    parser.add_argument("--activity-manifest", default=Config.activity_manifest)
    parser.add_argument("--clocks", default=Config.clocks)
    parser.add_argument("--result", default=Config.result)
    result = build(Config(**vars(parser.parse_args())))
    print(
        json.dumps(
            {
                "outcomes_opened": result["protocol"]["outcomes_opened"],
                "support": result["support"],
                "clock_overlap": result["clock_overlap"],
                "clocks": result["clocks"],
                "manifest_hash": result["manifest_hash"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
