"""Freeze CCBS-12 support and entry-clock orthogonality without opening its PnL."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import training.portfolio_opt_all_discovered_alpha_gross10 as legacy_all
import training.portfolio_opt_combined_rex_new_alpha as legacy_base
import training.portfolio_opt_new_alpha_pool as new_alpha
from training.preregister_cross_collateral_basis_snapback import (
    SOURCE_MANIFEST_CONTENT_HASH,
    SOURCE_MANIFEST_FILE_SHA256,
    SOURCE_PANEL_FILE_SHA256,
    canonical_hash,
)


PREREGISTRATION = "results/cross_collateral_basis_snapback_preregistration_2026-07-17.json"
PREREGISTRATION_HASH = "33b6d8e4dec120b9cf1177e4fa37695f9a0e485d8dd29872e9d403fd72eacc25"
SOURCE_MANIFEST = "results/binance_cross_collateral_quarterly_curve_2021_2023_manifest.json"
SOURCE_PANEL = (
    "data/binance_cross_collateral_quarterly_curve_2021_2023/"
    "BTCUSDT_BTCUSD_CURRENT_QUARTER_5m_2021_2023.csv.gz"
)
OUTPUT = "results/cross_collateral_basis_snapback_support_2026-07-17.json"
DOCS_OUTPUT = "docs/cross-collateral-basis-snapback-support-2026-07-17.md"
ANCHOR_CLOCK_OUTPUT = "results/cross_collateral_basis_snapback_live_anchor_clock_2023.json"

SIGNAL_COLUMNS = (
    "open_time",
    "available_time",
    "um_close",
    "um_ohlc_valid",
    "cm_close",
    "cm_ohlc_valid",
    "source_complete",
    "delivery_time",
    "contract_segment",
)


@dataclass(frozen=True)
class Config:
    source_manifest: str = SOURCE_MANIFEST
    source_panel: str = SOURCE_PANEL
    preregistration: str = PREREGISTRATION
    output: str = OUTPUT
    docs_output: str = DOCS_OUTPUT
    anchor_clock_output: str = ANCHOR_CLOCK_OUTPUT
    lookback_bars: int = 4_032
    minimum_prior_bars: int = 3_226
    dislocation_floor: float = 0.002
    threshold_grid: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0)
    maximum_hold_bars: int = 144
    anchor_year: int = 2023
    cost_rate: float = 0.0006


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def verify_frozen_inputs(cfg: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    prereg = _load_json(cfg.preregistration)
    if prereg.get("protocol_hash") != PREREGISTRATION_HASH:
        raise ValueError("CCBS preregistration hash drifted")
    if canonical_hash(prereg.get("protocol")) != PREREGISTRATION_HASH:
        raise ValueError("CCBS preregistration body is not canonical-hash stable")

    manifest = _load_json(cfg.source_manifest)
    if file_sha256(cfg.source_manifest) != SOURCE_MANIFEST_FILE_SHA256:
        raise ValueError("quarterly source manifest file hash drifted")
    if manifest.get("manifest_hash") != SOURCE_MANIFEST_CONTENT_HASH:
        raise ValueError("quarterly source manifest content hash drifted")
    if file_sha256(cfg.source_panel) != SOURCE_PANEL_FILE_SHA256:
        raise ValueError("quarterly source panel file hash drifted")
    if manifest.get("file", {}).get("sha256") != SOURCE_PANEL_FILE_SHA256:
        raise ValueError("quarterly source panel is not bound by its manifest")
    return prereg, manifest


def load_signal_frame(path: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=list(SIGNAL_COLUMNS),
        parse_dates=["open_time", "available_time", "delivery_time"],
    )
    if tuple(frame.columns) != SIGNAL_COLUMNS:
        raise ValueError("signal panel columns drifted")
    if frame["open_time"].duplicated().any() or not frame["open_time"].is_monotonic_increasing:
        raise ValueError("signal clock is duplicated or unsorted")
    if not frame["available_time"].eq(frame["open_time"] + pd.Timedelta(minutes=5)).all():
        raise ValueError("signal availability is not one completed bar after open")
    return frame


def build_signal_features(
    frame: pd.DataFrame,
    *,
    lookback_bars: int,
    minimum_prior_bars: int,
) -> pd.DataFrame:
    if not 1 <= minimum_prior_bars <= lookback_bars:
        raise ValueError("invalid robust support window")
    clean = (
        frame["source_complete"].astype(bool)
        & frame["um_ohlc_valid"].astype(bool)
        & frame["cm_ohlc_valid"].astype(bool)
    )
    wedge = np.log(
        pd.to_numeric(frame["um_close"], errors="coerce")
        / pd.to_numeric(frame["cm_close"], errors="coerce")
    ).where(clean)
    parts: list[pd.DataFrame] = []
    for _, group in frame.assign(wedge=wedge).groupby("contract_segment", sort=False):
        current = group["wedge"].astype(float)
        prior = current.shift(1)
        center = prior.rolling(lookback_bars, min_periods=minimum_prior_bars).median()
        recursive_mad = (prior - center).abs().rolling(
            lookback_bars,
            min_periods=minimum_prior_bars,
        ).median()
        scale = 1.4826 * recursive_mad.replace(0.0, np.nan)
        output = group[list(SIGNAL_COLUMNS)].copy()
        output["wedge"] = current
        output["center"] = center
        output["recursive_mad"] = recursive_mad
        output["deviation"] = current - center
        output["zscore"] = (current - center) / scale
        parts.append(output)
    return pd.concat(parts).sort_index()


def candidate_events(
    features: pd.DataFrame,
    threshold: float,
    *,
    dislocation_floor: float,
    maximum_hold_bars: int,
) -> pd.DataFrame:
    active = (
        features["zscore"].abs().ge(float(threshold))
        & features["deviation"].abs().ge(float(dislocation_floor))
        & features["source_complete"].astype(bool)
    )
    grouped = features["contract_segment"]
    previous_active = active.groupby(grouped).shift(1, fill_value=False)
    previous_sign = np.sign(features["zscore"]).groupby(grouped).shift(1)
    onset = active & (
        ~previous_active | np.sign(features["zscore"]).ne(previous_sign)
    )
    candidates = features.loc[onset].copy()
    available_open_times = set(features["open_time"])
    reservation_end: pd.Timestamp | None = None
    kept: list[int] = []
    for index, row in candidates.iterrows():
        entry_time = row["open_time"] + pd.Timedelta(minutes=10)
        maximum_exit_time = entry_time + pd.Timedelta(minutes=5 * maximum_hold_bars)
        if maximum_exit_time >= row["delivery_time"]:
            continue
        if entry_time not in available_open_times or maximum_exit_time not in available_open_times:
            continue
        if reservation_end is not None and entry_time < reservation_end:
            continue
        kept.append(index)
        reservation_end = maximum_exit_time

    events = features.loc[kept, [
        "open_time",
        "available_time",
        "delivery_time",
        "contract_segment",
        "wedge",
        "center",
        "recursive_mad",
        "deviation",
        "zscore",
    ]].copy()
    events["entry_time"] = events["open_time"] + pd.Timedelta(minutes=10)
    events["maximum_exit_time"] = events["entry_time"] + pd.Timedelta(
        minutes=5 * maximum_hold_bars
    )
    events["rich_leg"] = np.where(events["zscore"] > 0.0, "um", "cm")
    return events.reset_index(drop=True)


def _bucket_counts(events: pd.DataFrame) -> dict[str, Any]:
    entry = events["entry_time"]
    half = entry.dt.year.astype(str) + "H" + np.where(entry.dt.month <= 6, "1", "2")
    quarter = entry.dt.year.astype(str) + "Q" + (((entry.dt.month - 1) // 3) + 1).astype(str)
    signs = np.sign(events["zscore"])
    return {
        "events": int(len(events)),
        "by_year": {str(k): int(v) for k, v in entry.dt.year.value_counts().sort_index().items()},
        "by_half": {str(k): int(v) for k, v in half.value_counts().sort_index().items()},
        "by_quarter": {
            str(k): int(v) for k, v in quarter.value_counts().sort_index().items()
        },
        "by_sign": {
            "um_rich": int((signs > 0).sum()),
            "cm_rich": int((signs < 0).sum()),
        },
    }


def support_passes(counts: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if counts["events"] < 130:
        failures.append("pre_2023_total_events_below_130")
    for year in ("2021", "2022"):
        if counts["by_year"].get(year, 0) < 50:
            failures.append(f"{year}_events_below_50")
    for half in ("2021H1", "2021H2", "2022H1", "2022H2"):
        if counts["by_half"].get(half, 0) < 25:
            failures.append(f"{half}_events_below_25")
    for quarter in (
        "2021Q1",
        "2021Q2",
        "2021Q3",
        "2021Q4",
        "2022Q1",
        "2022Q2",
        "2022Q3",
        "2022Q4",
    ):
        if counts["by_quarter"].get(quarter, 0) < 6:
            failures.append(f"{quarter}_events_below_6")
    if counts["events"]:
        for name, count in counts["by_sign"].items():
            if count / counts["events"] < 0.30:
                failures.append(f"{name}_share_below_30pct")
    else:
        failures.append("sign_share_undefined")
    return not failures, failures


def select_threshold(
    features: pd.DataFrame,
    cfg: Config,
) -> tuple[float | None, list[dict[str, Any]], dict[float, pd.DataFrame]]:
    all_events: dict[float, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for threshold in cfg.threshold_grid:
        events = candidate_events(
            features,
            threshold,
            dislocation_floor=cfg.dislocation_floor,
            maximum_hold_bars=cfg.maximum_hold_bars,
        )
        all_events[threshold] = events
        pre_2023 = events.loc[events["entry_time"] < pd.Timestamp("2023-01-01", tz="UTC")]
        counts = _bucket_counts(pre_2023)
        passed, failures = support_passes(counts)
        rows.append(
            {
                "threshold": threshold,
                "selection_period_counts": counts,
                "support_passed": passed,
                "failures": failures,
                "2023_diagnostic_counts_not_used_for_selection": _bucket_counts(
                    events.loc[
                        events["entry_time"].between(
                            pd.Timestamp("2023-01-01", tz="UTC"),
                            pd.Timestamp("2024-01-01", tz="UTC"),
                            inclusive="left",
                        )
                    ]
                ),
            }
        )
    passing = [float(row["threshold"]) for row in rows if row["support_passed"]]
    return (max(passing) if passing else None), rows, all_events


def _frozen_anchor_input_hashes() -> dict[str, dict[str, str]]:
    return {
        "live_anchor_config": {
            "path": "configs/live/portfolio_gross385_trainmdd40_2026-07-12.json",
            "sha256": "86f255ca3967245b8b0676b00025b955d7f33668ab1ef9d813623191b4ecd1e7",
        },
        "oi_sleeve_config": {
            "path": "configs/live/oi_upbit_ratio288_low_candidate.json",
            "sha256": "659239373e1f51fc2df9615f5387686fd9252a56e1c366b45421bf39d3d6223f",
        },
        "funding_sleeve_config": {
            "path": "configs/live/new_long_minimal_funding_premium_candidate.json",
            "sha256": "f0848c5fea1fcc7823ed15b6e4b865a8dc2731c2d2bfd2ba21b0f92c534f0f03",
        },
        "rex_sleeve_config": {
            "path": "configs/live/rex_veto_7_candidate.json",
            "sha256": "36df47c4737eb99f4ca5e2b257d9bd2fbf130df9d731b9ac02fcfe5192acd4db",
        },
        "rex_veto_scan": {
            "path": "results/rex_failure_veto_alpha_scan_2026-07-12.json",
            "sha256": "e84f580c8a2dff0c35b11d2a3ff2c1db916f1c6a31225e471d4e3698f11f71cc",
        },
        "anchor_builder": {
            "path": "training/portfolio_opt_added_alpha_update.py",
            "sha256": "d98b79db1053190087ed274d0b37f91961f55c2128b64c3066970a957d313db9",
        },
        "combined_event_builder": {
            "path": "training/portfolio_opt_combined_rex_new_alpha.py",
            "sha256": "6c973c50f089b1f9eff9fb3c160ed7be1063609a3db18e1d0861889b7f631439",
        },
        "legacy_event_builder": {
            "path": "training/evaluate_volume_wave_portfolio_combo.py",
            "sha256": "3e47f15ef583e6f59d43ab2a6cfcedf8969b35e18bfef1c3b608c02f88a1bcb5",
        },
        "rex_event_builder": {
            "path": "training/portfolio_opt_all_discovered_alpha_gross10.py",
            "sha256": "9bee1d61e399aa6a8435e0ec5d27823670e763c63765ebc41eaae471cb33b74d",
        },
        "new_alpha_builder": {
            "path": "training/portfolio_opt_new_alpha_pool.py",
            "sha256": "88434d2aefafd763a6b09d2ad78b4b77dd493e4102b696e80cc53e67437e91fa",
        },
        "market": {
            "path": "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz",
            "sha256": "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c",
        },
        "funding": {
            "path": "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz",
            "sha256": "4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7",
        },
        "premium": {
            "path": "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz",
            "sha256": "b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7",
        },
        "open_interest_cache": {
            "path": "/tmp/btcusdt_open_interest_5m_2020_2026.csv",
            "sha256": "e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31",
        },
        "oi_base_config": {
            "path": "configs/live/oi_divergence_sma24_highfreq_h30_s6_candidate.json",
            "sha256": "30f2f4abf397805550b9a393c1168c920d6d10f9220e795567b43d043a0d5406",
        },
        "upbit_prefix": {
            "path": (
                "/home/pakchu/workspace/wave_trading/data/"
                "2020-01-01_2025-12-15_4bd081fc54811fccdee66850692c435e.csv.gz"
            ),
            "sha256": "7c377c402b4c1c3db3dafb5e15cd06e93f6e9c2c08d154ed88dd47e91f86eb35",
        },
        "rex_reasoning_clock": {
            "path": "data/rex_event_reasoning_policy_sft_20260712.jsonl",
            "sha256": "2f5f477ed7ffd6063bd25b1fdbcb6cbaa804685be43b4522b7105dfba1b75d48",
        },
    }


def verify_anchor_inputs() -> dict[str, dict[str, str]]:
    inputs = _frozen_anchor_input_hashes()
    for name, record in inputs.items():
        path = Path(record["path"])
        if not path.exists():
            raise FileNotFoundError(f"anchor input is missing: {name}={path}")
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"anchor input hash drifted: {name}")
    return inputs


def _individual_rex_clock(
    market: pd.DataFrame,
    masks: dict[str, np.ndarray],
    *,
    cost_rate: float,
) -> list[dict[str, Any]]:
    report = legacy_all.load_json(legacy_all.SCAN_FILES["rex_veto"])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bucket in ("top", "tte_top"):
        for row in report.get(bucket, [])[:50]:
            key = json.dumps(row.get("gates", []), sort_keys=True)
            if key not in seen:
                seen.add(key)
                rows.append(row)
    if len(rows) <= 7:
        raise RuntimeError("frozen REX veto row 7 is unavailable")
    gate_row = rows[7]
    source = [
        json.loads(line)
        for line in Path("data/rex_event_reasoning_policy_sft_20260712.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    features = legacy_all._build_light_rex_features(market)
    source_dates = pd.to_datetime(market["date"])
    dates = pd.to_datetime(market["date"], utc=True)
    output: list[dict[str, Any]] = []
    for split, split_mask in masks.items():
        next_allowed = 0
        for row in source:
            position = int(row.get("signal_pos", -1))
            if (
                position < 0
                or position >= len(market)
                or not split_mask[position]
                or position < next_allowed
            ):
                continue
            if pd.Timestamp(row["date"]) != pd.Timestamp(source_dates.iloc[position]):
                raise RuntimeError("REX source clock drifted from shared market")
            side = str((row.get("base_event") or {}).get("base_side", "")).lower()
            if side not in {"long", "short"} or not legacy_all._rex_row_matches(
                gate_row.get("gates", []), features, row
            ):
                continue
            exit_position = position + 145
            if exit_position >= len(market) or not split_mask[exit_position]:
                continue
            path = new_alpha._event_path(
                market,
                position,
                side=side,
                hold=144,
                cost_rate=cost_rate,
                entry_delay=1,
                leverage=0.5,
            )
            if path is None:
                continue
            output.append(
                {
                    "sleeve": "cand_rex_veto_7",
                    "split": split,
                    "signal_time": dates.iloc[position],
                    "entry_time": dates.iloc[position + 1],
                    "side": side,
                }
            )
            next_allowed = exit_position + 1
    return output


def build_anchor_clock(year: int, cfg: Config) -> tuple[dict[str, Any], pd.DataFrame]:
    inputs = verify_anchor_inputs()
    legacy_cfg = legacy_all.Config(
        random_samples=0,
        candidate_rex_top_n=50,
        train_mdd_cap=40.0,
        oos_mdd_cap=20.0,
        gross_cap=10.0,
        min_nonzero_weight=0.25,
        weight_step=0.05,
        cost_rate=cfg.cost_rate,
    )
    market, _, masks, _, events, _ = legacy_base.build_combined_events(legacy_cfg)
    dates = pd.to_datetime(market["date"], utc=True)
    entries: list[dict[str, Any]] = []
    for event in events:
        if event.get("sleeve") not in {
            "oi_upbit_ratio288_low",
            "new_long_minimal_funding_premium",
        }:
            continue
        position = int(event["signal_pos"])
        if position + 1 >= len(market):
            raise RuntimeError("anchor signal has no delayed entry row")
        entries.append(
            {
                "sleeve": str(event["sleeve"]),
                "split": str(event["split"]),
                "signal_time": dates.iloc[position],
                "entry_time": dates.iloc[position + 1],
                "side": str(event.get("side", "")),
            }
        )
    entries.extend(_individual_rex_clock(market, masks, cost_rate=cfg.cost_rate))
    clock = pd.DataFrame(entries)
    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
    clock = clock.loc[clock["entry_time"].between(start, end, inclusive="left")].copy()
    clock = clock.sort_values(["entry_time", "sleeve"]).reset_index(drop=True)
    rows = [
        {
            **row,
            "signal_time": pd.Timestamp(row["signal_time"]).isoformat(),
            "entry_time": pd.Timestamp(row["entry_time"]).isoformat(),
        }
        for row in clock.to_dict("records")
    ]
    body = {
        "year": year,
        "live_anchor_weights": {
            "oi_upbit_ratio288_low": 0.65,
            "new_long_minimal_funding_premium": 1.75,
            "cand_rex_veto_7": 1.45,
        },
        "input_hashes": inputs,
        "events": rows,
        "counts_by_sleeve": {
            str(k): int(v) for k, v in clock["sleeve"].value_counts().sort_index().items()
        },
        "unique_entry_times": int(clock["entry_time"].nunique()),
        "unique_entry_days": int(clock["entry_time"].dt.normalize().nunique()),
    }
    payload = {**body, "content_hash": canonical_hash(body)}
    return payload, clock


def clock_overlap(ccbs: pd.DataFrame, anchor: pd.DataFrame) -> dict[str, Any]:
    ccbs_times = set(pd.to_datetime(ccbs["entry_time"], utc=True))
    anchor_times = set(pd.to_datetime(anchor["entry_time"], utc=True))
    ccbs_days = {timestamp.normalize() for timestamp in ccbs_times}
    anchor_days = {timestamp.normalize() for timestamp in anchor_times}
    exact = ccbs_times & anchor_times
    day_intersection = ccbs_days & anchor_days
    day_union = ccbs_days | anchor_days
    return {
        "ccbs_entries": len(ccbs_times),
        "anchor_entries": len(anchor_times),
        "exact_5m_intersections": len(exact),
        "exact_5m_overlap_share_of_ccbs": len(exact) / len(ccbs_times) if ccbs_times else 1.0,
        "ccbs_entry_days": len(ccbs_days),
        "anchor_entry_days": len(anchor_days),
        "entry_day_intersections": len(day_intersection),
        "entry_day_union": len(day_union),
        "entry_day_jaccard": len(day_intersection) / len(day_union) if day_union else 1.0,
        "exact_intersection_times": [timestamp.isoformat() for timestamp in sorted(exact)],
    }


def _event_records(events: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in events.to_dict("records"):
        output.append(
            {
                **row,
                "open_time": pd.Timestamp(row["open_time"]).isoformat(),
                "available_time": pd.Timestamp(row["available_time"]).isoformat(),
                "entry_time": pd.Timestamp(row["entry_time"]).isoformat(),
                "maximum_exit_time": pd.Timestamp(row["maximum_exit_time"]).isoformat(),
                "delivery_time": pd.Timestamp(row["delivery_time"]).isoformat(),
                "contract_segment": str(row["contract_segment"]),
            }
        )
    return output


def markdown(report: dict[str, Any]) -> str:
    selected = report["selected_threshold"]
    support = next(row for row in report["threshold_support"] if row["threshold"] == selected)
    pre = support["selection_period_counts"]
    diag = support["2023_diagnostic_counts_not_used_for_selection"]
    overlap = report["2023_entry_clock_orthogonality"]
    return f"""# CCBS-12 support and entry-clock orthogonality — 2026-07-17

## Outcome boundary

This unit loaded only completed close-based CCBS signal columns. It did **not**
open CCBS entry/exit prices, held OHLC paths, returns, PnL, CAGR, or MDD. Known
live-anchor paths were used only to reconstruct the already-live entry clock;
they could veto but could not select or change the CCBS threshold.

## Support-only threshold

The largest 2021-2022-supported threshold is **z={selected:.1f}**. It produced
{pre['events']} pre-2023 events ({pre['by_year']}) and passed every frozen
year/half/quarter/sign floor. Its 2023 count is {diag['events']} and is reported
only as a feature-support diagnostic; 2023 was already declared development,
not pristine OOS.

## 2023 live-anchor clock overlap

- CCBS entries: {overlap['ccbs_entries']}; live-anchor unique entries: {overlap['anchor_entries']};
- exact 5m intersections: {overlap['exact_5m_intersections']} ({overlap['exact_5m_overlap_share_of_ccbs']:.3%} of CCBS);
- entry-day Jaccard: {overlap['entry_day_jaccard']:.4f};
- frozen limits: exact overlap <= 10%, day Jaccard <= 0.20.

Disposition: **{report['disposition']}**. PnL may open only when this support
unit passes. Daily-PnL/BTC/portfolio orthogonality remains a later outcome gate,
and live promotion remains blocked by the omitted COIN-M collateral ledger.

Report content hash: `{report['content_hash']}`
"""


def run(cfg: Config) -> dict[str, Any]:
    prereg, source_manifest = verify_frozen_inputs(cfg)
    frame = load_signal_frame(cfg.source_panel)
    features = build_signal_features(
        frame,
        lookback_bars=cfg.lookback_bars,
        minimum_prior_bars=cfg.minimum_prior_bars,
    )
    selected, threshold_rows, all_events = select_threshold(features, cfg)
    if selected is None:
        report = {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "protocol_hash": PREREGISTRATION_HASH,
            "source_manifest_content_hash": source_manifest["manifest_hash"],
            "loaded_ccbs_columns": list(SIGNAL_COLUMNS),
            "forbidden_ccbs_columns_loaded": [],
            "threshold_support": threshold_rows,
            "selected_threshold": None,
            "disposition": "REJECT_SUPPORT",
        }
        body = {key: value for key, value in report.items() if key != "as_of"}
        report["content_hash"] = canonical_hash(body)
        Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        return report

    selected_events = all_events[selected]
    year_start = pd.Timestamp(year=cfg.anchor_year, month=1, day=1, tz="UTC")
    year_end = pd.Timestamp(year=cfg.anchor_year + 1, month=1, day=1, tz="UTC")
    development_events = selected_events.loc[
        selected_events["entry_time"].between(year_start, year_end, inclusive="left")
    ].copy()
    anchor_payload, anchor_clock = build_anchor_clock(cfg.anchor_year, cfg)
    Path(cfg.anchor_clock_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.anchor_clock_output).write_text(
        json.dumps(anchor_payload, indent=2, ensure_ascii=False) + "\n"
    )
    overlap = clock_overlap(development_events, anchor_clock)
    overlap_passed = (
        overlap["exact_5m_overlap_share_of_ccbs"] <= 0.10
        and overlap["entry_day_jaccard"] <= 0.20
    )
    body = {
        "protocol_hash": prereg["protocol_hash"],
        "support_builder_sha256": file_sha256(__file__),
        "source_manifest_content_hash": source_manifest["manifest_hash"],
        "source_panel_sha256": file_sha256(cfg.source_panel),
        "loaded_ccbs_columns": list(SIGNAL_COLUMNS),
        "forbidden_ccbs_columns_loaded": [],
        "feature_rows": int(len(features)),
        "finite_zscore_rows": int(features["zscore"].notna().sum()),
        "first_finite_zscore_time": (
            features.loc[features["zscore"].notna(), "open_time"].min().isoformat()
        ),
        "threshold_selection_period": ["2021-01-01", "2023-01-01"],
        "threshold_support": threshold_rows,
        "selected_threshold": selected,
        "selected_events_2021_2023": _event_records(selected_events),
        "anchor_clock": {
            "path": cfg.anchor_clock_output,
            "file_sha256": file_sha256(cfg.anchor_clock_output),
            "content_hash": anchor_payload["content_hash"],
        },
        "2023_entry_clock_orthogonality": overlap,
        "entry_clock_orthogonality_passed": overlap_passed,
        "disposition": "PASS_SUPPORT_OPEN_2023_PNL" if overlap_passed else "REJECT_CLOCK_OVERLAP",
    }
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        **body,
        "content_hash": canonical_hash(body),
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    Path(cfg.docs_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.docs_output).write_text(markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--docs-output", default=DOCS_OUTPUT)
    parser.add_argument("--anchor-clock-output", default=ANCHOR_CLOCK_OUTPUT)
    args = parser.parse_args()
    report = run(
        Config(
            output=args.output,
            docs_output=args.docs_output,
            anchor_clock_output=args.anchor_clock_output,
        )
    )
    print(
        json.dumps(
            {
                "selected_threshold": report.get("selected_threshold"),
                "disposition": report["disposition"],
                "content_hash": report["content_hash"],
                "entry_clock": report.get("2023_entry_clock_orthogonality"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
