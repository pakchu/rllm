"""Freeze outcome-blind clocks for ETH-to-BTC liquidation relay EBLR-60/30."""

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


CANDIDATE = "EBLR-60/30"
BTC_SOURCE = Path(
    "data/binance_coinm_liquidation_snapshot_btc_2023_2024/"
    "BTCUSD_PERP_liquidation_5m_2023-06-25_2024-10-14.csv.gz"
)
BTC_MANIFEST = Path(
    "results/binance_coinm_liquidation_snapshot_btc_2023_2024_manifest.json"
)
ETH_SOURCE = Path(
    "data/binance_coinm_liquidation_snapshot_eth_2023_2024/"
    "ETHUSD_PERP_liquidation_5m_2023-06-25_2024-10-14.csv.gz"
)
ETH_MANIFEST = Path(
    "results/binance_coinm_liquidation_snapshot_eth_2023_2024_manifest.json"
)
EXPECTED_BTC_SOURCE_SHA256 = (
    "a23b93d8567a589e9f045ae4a56393e493a8da2748c5a051804c9bdf9388ccc3"
)
EXPECTED_BTC_MANIFEST_SHA256 = (
    "5d78686e7c40d69261f09bc77e27ff734f682abba4abb95c2291e8282380053e"
)
EXPECTED_ETH_SOURCE_SHA256 = (
    "8d17ab3d5f9592f5254fef2e649065233be1777b8976983b4af38c77a8cc5bff"
)
EXPECTED_ETH_MANIFEST_SHA256 = (
    "c515731a9029d1786c8650f5106923d4cfbe8c35ed7a947f5420a16154601f5d"
)
CLBR_CLOCKS = Path("data/coinm_liquidation_burst_release_clocks_2023_2024.csv.gz")
ICLA_CLOCKS = Path(
    "data/inverse_collateral_liquidation_absorption_clocks_2023_2024.csv.gz"
)
EXPECTED_CLBR_CLOCKS_SHA256 = (
    "df619a5ffc3b849d3c35fc7112641c33105ba76c81cbb7b8c7f3c975fd80bee0"
)
EXPECTED_ICLA_CLOCKS_SHA256 = (
    "a55c23a7a0c296b98bb7a8958f713548c4313c0c682f1693c8f8be80b70dd053"
)
DEFAULT_CLOCKS = Path("data/eth_btc_liquidation_relay_clocks_2023_2024.csv.gz")
DEFAULT_RESULT = Path("results/eth_btc_liquidation_relay_support_2026-07-19.json")

BAR_MINUTES = 5
WAVE_BARS = 12
LOOKBACK_DAYS = 28
LOOKBACK_BARS = LOOKBACK_DAYS * 24 * 12
MIN_POSITIVE_WINDOWS = 300
REFERENCE_QUANTILE = 0.95
ETH_SEVERITY_MIN = 1.0
ETH_MIN_EVENT_COUNT = 3
ETH_ABS_IMBALANCE_MIN = 0.70
BTC_QUIET_SEVERITY_MAX = 0.50
HOLD_BARS = 6
MAX_ENTRY_JACCARD = 0.10
SPLITS = {
    "train": ("2023-06-25", "2023-10-15"),
    "test": ("2023-10-15", "2024-04-15"),
    "eval": ("2024-04-15", "2024-10-15"),
}
SUPPORT_MIN_TOTAL = {"train": 20, "test": 50, "eval": 50}
SUPPORT_MIN_PER_SIDE = {"train": 6, "test": 12, "eval": 12}
SUPPORT_MAX_MONTH_SHARE = {"train": 0.40, "test": 0.30, "eval": 0.30}
SCHEMA_VERSION = 1

BTC_SIGNAL_COLUMNS = (
    "date",
    "feature_available_time",
    "source_valid",
    "event_count",
    "total_liquidation_usd",
    "signed_liquidation_usd",
)
ETH_SIGNAL_COLUMNS = (
    "date",
    "feature_available_time",
    "source_valid",
    "event_count",
    "total_liquidation_contracts",
    "signed_liquidation_contracts",
)


@dataclass(frozen=True)
class Config:
    btc_source_path: str = str(BTC_SOURCE)
    btc_manifest_path: str = str(BTC_MANIFEST)
    eth_source_path: str = str(ETH_SOURCE)
    eth_manifest_path: str = str(ETH_MANIFEST)
    expected_btc_source_sha256: str = EXPECTED_BTC_SOURCE_SHA256
    expected_btc_manifest_sha256: str = EXPECTED_BTC_MANIFEST_SHA256
    expected_eth_source_sha256: str = EXPECTED_ETH_SOURCE_SHA256
    expected_eth_manifest_sha256: str = EXPECTED_ETH_MANIFEST_SHA256
    clbr_clock_path: str = str(CLBR_CLOCKS)
    icla_clock_path: str = str(ICLA_CLOCKS)
    output_clock_path: str = str(DEFAULT_CLOCKS)
    output_result_path: str = str(DEFAULT_RESULT)


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


def _validate_manifest(
    path: Path,
    *,
    expected_hash: str,
    expected_source_hash: str,
    label: str,
) -> dict[str, Any]:
    if sha256_file(path) != expected_hash:
        raise ValueError(f"{label} manifest sha256 mismatch")
    manifest = json.loads(path.read_text())
    protocol = manifest.get("protocol", {})
    if protocol.get("outcomes_opened") is not False:
        raise ValueError(f"{label} source is not outcome blind")
    if protocol.get("source_only") is not True:
        raise ValueError(f"{label} manifest is not source-only")
    if manifest.get("file", {}).get("sha256") != expected_source_hash:
        raise ValueError(f"{label} source sha256 disagrees with manifest")
    return manifest


def _load_panel(
    path: Path,
    *,
    expected_hash: str,
    columns: tuple[str, ...],
    label: str,
) -> pd.DataFrame:
    if sha256_file(path) != expected_hash:
        raise ValueError(f"{label} source sha256 mismatch")
    frame = pd.read_csv(
        path,
        compression="gzip",
        parse_dates=["date", "feature_available_time"],
    )
    frame = frame.loc[:, list(columns)]
    expected = pd.Series(
        pd.date_range("2023-06-25", "2024-10-15", freq="5min", inclusive="left"),
        name="date",
    )
    if not cast(pd.Series, frame["date"]).equals(expected):
        raise ValueError(f"{label} source grid mismatch")
    delay = cast(pd.Series, frame["feature_available_time"]).sub(frame["date"])
    if not bool(delay.eq(pd.Timedelta(minutes=5, seconds=1)).all()):
        raise ValueError(f"{label} source availability mismatch")
    if frame["date"].duplicated().any():
        raise ValueError(f"{label} source contains duplicate timestamps")
    return frame


def load_sources(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only the frozen source fields used by the outcome-blind rule."""

    _validate_manifest(
        Path(cfg.btc_manifest_path),
        expected_hash=cfg.expected_btc_manifest_sha256,
        expected_source_hash=cfg.expected_btc_source_sha256,
        label="BTC",
    )
    _validate_manifest(
        Path(cfg.eth_manifest_path),
        expected_hash=cfg.expected_eth_manifest_sha256,
        expected_source_hash=cfg.expected_eth_source_sha256,
        label="ETH",
    )
    btc = _load_panel(
        Path(cfg.btc_source_path),
        expected_hash=cfg.expected_btc_source_sha256,
        columns=BTC_SIGNAL_COLUMNS,
        label="BTC",
    )
    eth = _load_panel(
        Path(cfg.eth_source_path),
        expected_hash=cfg.expected_eth_source_sha256,
        columns=ETH_SIGNAL_COLUMNS,
        label="ETH",
    )
    if not cast(pd.Series, btc["date"]).equals(eth["date"]):
        raise ValueError("BTC and ETH source grids differ")
    return btc, eth


def _rolling_wave(
    frame: pd.DataFrame,
    *,
    total_column: str,
    signed_column: str,
) -> pd.DataFrame:
    valid = cast(pd.Series, frame["source_valid"]).astype(bool)
    valid_count = valid.astype(int).rolling(
        WAVE_BARS, min_periods=WAVE_BARS
    ).sum()
    output = pd.DataFrame(index=frame.index)
    output["valid"] = cast(pd.Series, valid_count).eq(WAVE_BARS)
    output["event_count"] = cast(pd.Series, frame["event_count"]).rolling(
        WAVE_BARS, min_periods=WAVE_BARS
    ).sum()
    output["total"] = cast(pd.Series, frame[total_column]).rolling(
        WAVE_BARS, min_periods=WAVE_BARS
    ).sum()
    output["signed"] = cast(pd.Series, frame[signed_column]).rolling(
        WAVE_BARS, min_periods=WAVE_BARS
    ).sum()
    output["imbalance"] = cast(pd.Series, output["signed"]).div(
        cast(pd.Series, output["total"]).where(output["total"].gt(0.0))
    )
    positive = cast(pd.Series, output["total"]).where(
        cast(pd.Series, output["valid"]).astype(bool) & output["total"].gt(0.0)
    )
    prior_q95 = (
        positive.shift(1)
        .rolling(LOOKBACK_BARS, min_periods=MIN_POSITIVE_WINDOWS)
        .quantile(REFERENCE_QUANTILE)
    )
    full_reference_elapsed = pd.Series(
        np.arange(len(output)) >= LOOKBACK_BARS,
        index=output.index,
    )
    output["prior_q95"] = cast(pd.Series, prior_q95).where(
        full_reference_elapsed
    )
    output["severity"] = cast(pd.Series, output["total"]).div(
        cast(pd.Series, output["prior_q95"]).where(output["prior_q95"].gt(0.0))
    )
    return output


def derive_relay_state(btc: pd.DataFrame, eth: pd.DataFrame) -> pd.DataFrame:
    """Derive the single frozen relay rule using strictly-prior thresholds."""

    if not cast(pd.Series, btc["date"]).equals(eth["date"]):
        raise ValueError("BTC and ETH source grids differ")
    btc_wave = _rolling_wave(
        btc,
        total_column="total_liquidation_usd",
        signed_column="signed_liquidation_usd",
    )
    eth_wave = _rolling_wave(
        eth,
        total_column="total_liquidation_contracts",
        signed_column="signed_liquidation_contracts",
    )
    output = pd.DataFrame({"date": btc["date"]})
    output["btc_feature_available_time"] = btc["feature_available_time"]
    output["eth_feature_available_time"] = eth["feature_available_time"]
    output["feature_available_time"] = output[
        ["btc_feature_available_time", "eth_feature_available_time"]
    ].max(axis=1)
    output["wave_source_valid"] = cast(pd.Series, btc_wave["valid"]).astype(
        bool
    ) & cast(pd.Series, eth_wave["valid"]).astype(bool)
    output["eth_event_count_60m"] = eth_wave["event_count"]
    output["eth_wave_total"] = eth_wave["total"]
    output["eth_wave_signed"] = eth_wave["signed"]
    output["eth_wave_imbalance"] = eth_wave["imbalance"]
    output["eth_prior_q95"] = eth_wave["prior_q95"]
    output["eth_severity"] = eth_wave["severity"]
    output["btc_event_count_60m"] = btc_wave["event_count"]
    output["btc_wave_total"] = btc_wave["total"]
    output["btc_wave_signed"] = btc_wave["signed"]
    output["btc_wave_imbalance"] = btc_wave["imbalance"]
    output["btc_prior_q95"] = btc_wave["prior_q95"]
    output["btc_quiet_severity"] = btc_wave["severity"]

    eth_imbalance = cast(pd.Series, output["eth_wave_imbalance"])
    direction = pd.Series(
        np.where(eth_imbalance.ge(ETH_ABS_IMBALANCE_MIN), 1, -1),
        index=output.index,
        dtype="int64",
    )
    direction = direction.where(
        eth_imbalance.abs().ge(ETH_ABS_IMBALANCE_MIN), 0
    )
    output["direction"] = direction
    output["is_candidate"] = (
        cast(pd.Series, output["wave_source_valid"]).astype(bool)
        & cast(pd.Series, output["eth_event_count_60m"]).ge(ETH_MIN_EVENT_COUNT)
        & cast(pd.Series, output["eth_severity"]).ge(ETH_SEVERITY_MIN)
        & eth_imbalance.abs().ge(ETH_ABS_IMBALANCE_MIN)
        & cast(pd.Series, output["btc_quiet_severity"]).le(
            BTC_QUIET_SEVERITY_MAX
        )
        & direction.ne(0)
    )
    return output


def _clock_row(state: pd.DataFrame, index: int, split: str) -> dict[str, Any]:
    current = state.loc[index]
    last_bar_open = cast(pd.Timestamp, current["date"])
    wave_completed = last_bar_open + pd.Timedelta(minutes=BAR_MINUTES)
    feature_available = cast(pd.Timestamp, current["feature_available_time"])
    entry_time = last_bar_open + pd.Timedelta(minutes=2 * BAR_MINUTES)
    if feature_available < wave_completed or entry_time <= feature_available:
        raise ValueError("entry must follow completed source availability")
    return {
        "candidate": CANDIDATE,
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
        "eth_event_count_60m": int(current["eth_event_count_60m"]),
        "eth_wave_total": float(current["eth_wave_total"]),
        "eth_prior_q95": float(current["eth_prior_q95"]),
        "eth_severity": float(current["eth_severity"]),
        "eth_wave_imbalance": float(current["eth_wave_imbalance"]),
        "btc_wave_total": float(current["btc_wave_total"]),
        "btc_prior_q95": float(current["btc_prior_q95"]),
        "btc_quiet_severity": float(current["btc_quiet_severity"]),
    }


def build_clocks(state: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidate = cast(pd.Series, state["is_candidate"]).astype(bool)
    for split, (start_text, end_text) in SPLITS.items():
        start = cast(pd.Timestamp, pd.Timestamp(start_text))
        end = cast(pd.Timestamp, pd.Timestamp(end_text))
        indices = state.index[
            candidate & state["date"].ge(start) & state["date"].lt(end)
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
        raise ValueError(f"{CANDIDATE} produced no source-only clocks")
    entries = cast(pd.Series, output["entry_time"])
    if entries.duplicated().any() or not entries.is_monotonic_increasing:
        raise ValueError(f"{CANDIDATE} entries are invalid")
    exits = cast(pd.Series, output["planned_exit_time"])
    if len(output) > 1 and not entries.iloc[1:].reset_index(drop=True).ge(
        exits.iloc[:-1].reset_index(drop=True)
    ).all():
        raise ValueError(f"{CANDIDATE} clocks overlap")
    return output


def _support(clocks: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    stage_passes: list[bool] = []
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
            "minimum_total": len(subset) >= SUPPORT_MIN_TOTAL[split],
            "minimum_long": long_count >= SUPPORT_MIN_PER_SIDE[split],
            "minimum_short": short_count >= SUPPORT_MIN_PER_SIDE[split],
            "maximum_month_share": (
                maximum_month_share <= SUPPORT_MAX_MONTH_SHARE[split]
            ),
        }
        passes = bool(all(checks.values()))
        stage_passes.append(passes)
        result[split] = {
            "count": int(len(subset)),
            "long": long_count,
            "short": short_count,
            "minimum_required": SUPPORT_MIN_TOTAL[split],
            "minimum_per_side": SUPPORT_MIN_PER_SIDE[split],
            "maximum_month_share": maximum_month_share,
            "maximum_month_share_allowed": SUPPORT_MAX_MONTH_SHARE[split],
            "checks": checks,
            "passes": passes,
        }
    result["passes"] = bool(all(stage_passes))
    return result


def _entry_jaccard(
    primary: pd.DataFrame,
    comparison_path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    if comparison_path == CLBR_CLOCKS:
        expected = EXPECTED_CLBR_CLOCKS_SHA256
    elif comparison_path == ICLA_CLOCKS:
        expected = EXPECTED_ICLA_CLOCKS_SHA256
    else:
        expected = None
    actual_hash = sha256_file(comparison_path)
    if expected is not None and actual_hash != expected:
        raise ValueError(f"{label} comparison clock hash mismatch")
    comparison = pd.read_csv(
        comparison_path, compression="gzip", parse_dates=["entry_time"]
    )
    primary_entries = set(cast(pd.Series, primary["entry_time"]).tolist())
    comparison_entries = set(
        cast(pd.Series, comparison["entry_time"]).tolist()
    )
    intersection = primary_entries.intersection(comparison_entries)
    union = primary_entries.union(comparison_entries)
    return {
        "path": str(comparison_path),
        "sha256": actual_hash,
        "primary_entries": len(primary_entries),
        "comparison_entries": len(comparison_entries),
        "intersection_entries": len(intersection),
        "entry_jaccard": len(intersection) / len(union) if union else 0.0,
    }


def _clock_overlap(clocks: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    clbr = _entry_jaccard(clocks, Path(cfg.clbr_clock_path), label="CLBR")
    icla = _entry_jaccard(clocks, Path(cfg.icla_clock_path), label="ICLA")
    passes = all(
        float(item["entry_jaccard"]) <= MAX_ENTRY_JACCARD
        for item in (clbr, icla)
    )
    return {
        "CLBR": clbr,
        "ICLA": icla,
        "maximum_entry_jaccard_allowed": MAX_ENTRY_JACCARD,
        "passes": bool(passes),
    }


def build(cfg: Config) -> dict[str, Any]:
    btc, eth = load_sources(cfg)
    state = derive_relay_state(btc, eth)
    clocks = build_clocks(state)
    clock_path = Path(cfg.output_clock_path)
    _write_gzip_csv(clocks, clock_path)
    support = _support(clocks)
    overlap = _clock_overlap(clocks, cfg)
    support_passes = bool(support["passes"] and overlap["passes"])
    core: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "candidate": CANDIDATE,
            "name": "ETH-to-BTC inverse-collateral liquidation relay",
            "candidate_count": 1,
            "outcomes_opened": False,
            "market_prices_opened": False,
            "funding_opened": False,
            "return_labels_constructed": False,
            "all_stage_clocks_opened_source_only": True,
            "candidate_repair_after_outcomes": False,
            "later_stage_outcomes_opened": False,
        },
        "config": asdict(cfg),
        "sources": {
            "btc": {
                "path": cfg.btc_source_path,
                "sha256": cfg.expected_btc_source_sha256,
                "manifest": cfg.btc_manifest_path,
                "manifest_sha256": cfg.expected_btc_manifest_sha256,
                "role": "within-BTC quietness gate only",
            },
            "eth": {
                "path": cfg.eth_source_path,
                "sha256": cfg.expected_eth_source_sha256,
                "manifest": cfg.eth_manifest_path,
                "manifest_sha256": cfg.expected_eth_manifest_sha256,
                "role": "sole directional trigger",
            },
        },
        "parameters": {
            "bar_minutes": BAR_MINUTES,
            "wave_bars": WAVE_BARS,
            "lookback_days": LOOKBACK_DAYS,
            "lookback_bars": LOOKBACK_BARS,
            "minimum_positive_windows": MIN_POSITIVE_WINDOWS,
            "reference_quantile": REFERENCE_QUANTILE,
            "eth_severity_minimum": ETH_SEVERITY_MIN,
            "eth_minimum_event_count": ETH_MIN_EVENT_COUNT,
            "eth_minimum_absolute_imbalance": ETH_ABS_IMBALANCE_MIN,
            "btc_quiet_severity_maximum": BTC_QUIET_SEVERITY_MAX,
            "direction": "follow ETH forced-flow sign",
            "hold_bars": HOLD_BARS,
            "nonoverlap": True,
            "raw_cross_symbol_scale_comparison": False,
        },
        "splits": SPLITS,
        "support": support,
        "clock_overlap": overlap,
        "support_passes": support_passes,
        "clocks": {
            "path": str(clock_path),
            "sha256": sha256_file(clock_path),
            "rows": int(len(clocks)),
            "first_entry": str(clocks["entry_time"].min()),
            "last_exit": str(clocks["planned_exit_time"].max()),
        },
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    result_path = Path(cfg.output_result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--btc-source-path", default=Config.btc_source_path)
    parser.add_argument("--btc-manifest-path", default=Config.btc_manifest_path)
    parser.add_argument("--eth-source-path", default=Config.eth_source_path)
    parser.add_argument("--eth-manifest-path", default=Config.eth_manifest_path)
    parser.add_argument("--clbr-clock-path", default=Config.clbr_clock_path)
    parser.add_argument("--icla-clock-path", default=Config.icla_clock_path)
    parser.add_argument("--output-clock-path", default=Config.output_clock_path)
    parser.add_argument("--output-result-path", default=Config.output_result_path)
    cfg = Config(**vars(parser.parse_args()))
    result = build(cfg)
    print(
        json.dumps(
            {
                "outcomes_opened": result["protocol"]["outcomes_opened"],
                "support": result["support"],
                "clock_overlap": result["clock_overlap"],
                "support_passes": result["support_passes"],
                "clocks": result["clocks"],
                "manifest_hash": result["manifest_hash"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
