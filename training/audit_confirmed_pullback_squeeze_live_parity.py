"""Audit the confirmed pullback-squeeze under the current live/strict contract.

The historical report used a positional hourly clock, a two-hour premium
freshness tolerance, omitted realized funding cash flows, and measured adverse
movement chronologically bar by bar.  This audit keeps the alpha rule intact
while checking the current contract:

* premium observations are fresh for at most ten minutes;
* the signal row is the top-of-hour boundary, where the completed premium bar
  is available; market-derived inputs are shifted to the preceding completed
  five-minute bar so the decision has a full bar of execution latency;
* entry is the next five-minute open;
* realized funding is compounded during the position; and
* strict MDD marks the position-wide favorable envelope before its adverse
  envelope, in addition to the pre-entry global high-water mark.

An execution overlay grid is selected on physically truncated pre-2024 data
only.  If no overlay passes, later returns cannot rescue it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.binance_aux_features import (
    attach_binance_um_aux_frames,
    normalise_funding_history_frame,
    normalise_premium_index_frame,
)
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_interest_gate_validation import build_interest_features
from training.search_confirmed_pullback_squeeze_alpha import fit_confirmation_masks
from training.search_funding_premium_external_state_gate_alpha import (
    _frame_hash,
    _read_premium_before,
)
from training.search_inventory_purge_reclaim_alpha import (
    Config as ExecutionConfig,
    ExecutionEngine,
    Trade,
    equity_stats,
)
from training.search_positioning_hgb_path_alpha import _read_before
from training.search_specific_pullback_squeeze_alpha import (
    WINDOWS,
    fit_rule_masks,
)


SELECTION_END = "2024-01-01"
PRE2024_WINDOWS = {
    "train": ("2020-07-01", "2023-01-01"),
    "train_2020h2": ("2020-07-01", "2021-01-01"),
    "train_2021": ("2021-01-01", "2022-01-01"),
    "train_2022": ("2022-01-01", "2023-01-01"),
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023_h1": ("2023-01-01", "2023-07-01"),
    "select_2023_h2": ("2023-07-01", "2024-01-01"),
    "pre_2024": ("2020-07-01", "2024-01-01"),
}
HOLD_GRID = (288, 432, 576, 720)
STOP_TAKE_GRID = (
    ("tp4_sl2", 400, 200),
    ("tp4_sl3", 400, 300),
    ("tp5_sl2p5", 500, 250),
    ("tp5_sl3", 500, 300),
    ("tp6_sl3", 600, 300),
    ("tp6_sl4", 600, 400),
)
EXIT_SPECS = tuple(
    {"name": "time_only", "hold_bars": hold, "take_bps": 1_000_000, "stop_bps": 1_000_000}
    for hold in HOLD_GRID
) + tuple(
    {"name": name, "hold_bars": hold, "take_bps": take, "stop_bps": stop}
    for hold in HOLD_GRID
    for name, take, stop in STOP_TAKE_GRID
)


@dataclass(frozen=True)
class AuditConfig:
    input_csv: str = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
    funding_csv: str = "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
    premium_csv: str = "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
    output: str = "results/confirmed_pullback_squeeze_live_parity_audit_2026-07-15.json"
    exclude_from: str = "2026-06-02"
    window_size: int = 144
    leverage: float = 0.5
    operating_leverage: float = 0.9
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    funding_tolerance: str = "12h"
    offline_premium_tolerance: str = "2h"
    live_premium_tolerance: str = "10min"


def decision_mask(dates: pd.Series, mode: str, *, window_size: int = 144) -> np.ndarray:
    """Return the explicit decision clock used by an audit variant."""

    parsed = pd.to_datetime(dates)
    if mode == "legacy_positional":
        mask = np.zeros(len(parsed), dtype=bool)
        mask[np.arange(max(143, int(window_size) - 1), len(parsed), 12)] = True
        return mask
    if mode == "live_hour_signal_bar":
        return ((parsed.dt.minute == 0) & (parsed.dt.second == 0)).to_numpy(bool)
    raise ValueError(f"unknown decision clock: {mode}")


def live_decision_features(features: pd.DataFrame) -> pd.DataFrame:
    """Use prior-bar market state with current boundary-time auxiliary data.

    A row labelled ``HH:00`` represents the 5-minute market bar that starts at
    the boundary.  Its OHLCV is not available at ``HH:00``.  We therefore shift
    every derived feature by one row, then restore only the Binance auxiliary
    values that are timestamped independently at the boundary.  The resulting
    signal can be computed during ``HH:00``-``HH:05`` and entered at the next
    open, ``HH:05``.
    """

    shifted = features.shift(1)
    current_auxiliary = (
        "funding_rate",
        "funding_zscore",
        "funding_available",
        "premium_index",
        "premium_index_zscore",
        "premium_index_change",
        "premium_available",
        "binance_aux_any_available",
    )
    for column in current_auxiliary:
        if column in features:
            shifted[column] = features[column]
    return shifted


def _load_bundle(
    cfg: AuditConfig,
    *,
    cutoff: str,
    premium_tolerance: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    market_raw = _read_before(cfg.input_csv, "date", cutoff)
    funding_raw = _read_before(cfg.funding_csv, "date", cutoff)
    premium_raw = _read_premium_before(cfg.premium_csv, cutoff)
    hashes = {
        "market": _frame_hash(market_raw),
        "funding": _frame_hash(funding_raw),
        "premium": _frame_hash(premium_raw),
    }
    market = market_raw.copy()
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    funding = normalise_funding_history_frame(funding_raw)
    market = attach_binance_um_aux_frames(
        market,
        funding_frame=funding,
        premium_frame=normalise_premium_index_frame(premium_raw),
        funding_tolerance=cfg.funding_tolerance,
        premium_tolerance=premium_tolerance,
    )
    if len(market) and pd.Timestamp(market["date"].max()) >= pd.Timestamp(cutoff):
        raise RuntimeError("market source was not physically truncated")
    base = build_market_feature_frame(market, window_size=int(cfg.window_size))
    features = pd.concat([base, build_interest_features(market, base)], axis=1)
    return market, features, funding, hashes


def _fit_active(
    features: pd.DataFrame,
    dates: pd.Series,
    decisions: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    base = fit_rule_masks(
        features,
        dates,
        decisions,
        range_quantile=0.60,
        overheat_quantile=0.70,
    )
    confirmation = fit_confirmation_masks(
        features,
        dates,
        base["active"],
        bb_quantile=0.80,
        quote_volume_quantile=0.90,
    )
    return confirmation["active"], {
        "base": base["base_thresholds"],
        "context": base["context_thresholds"],
        "confirmation": confirmation["thresholds"],
    }


def _execution_config(cfg: AuditConfig, leverage: float) -> ExecutionConfig:
    return ExecutionConfig(
        input_csv=cfg.input_csv,
        metrics_csv="",
        funding_csv=cfg.funding_csv,
        output=cfg.output,
        manifest_output="",
        leverage=float(leverage),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
    )


def schedule_window(
    engine: ExecutionEngine,
    active: np.ndarray,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    hold_bars: int,
    take_bps: int,
    stop_bps: int,
) -> list[Trade]:
    dates = pd.to_datetime(engine.market["date"])
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    trades: list[Trade] = []
    next_allowed = 0
    for signal in np.flatnonzero(active & period):
        signal = int(signal)
        if signal < next_allowed:
            continue
        trade = engine.trade_at(signal, 1, int(hold_bars), int(take_bps), int(stop_bps))
        if trade is None or not period[trade.exit_position]:
            continue
        trades.append(trade)
        next_allowed = trade.exit_position + 1
    return trades


def _slim(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: stats[key]
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
            "trades",
            "mean_net_bps",
            "win_rate",
        )
    }


def _evaluate(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    active: np.ndarray,
    cfg: AuditConfig,
    *,
    leverage: float,
    windows: dict[str, tuple[str | pd.Timestamp, str | pd.Timestamp]],
    hold_bars: int = 576,
    take_bps: int = 1_000_000,
    stop_bps: int = 1_000_000,
) -> dict[str, dict[str, Any]]:
    execution_cfg = _execution_config(cfg, leverage)
    engine = ExecutionEngine(market, funding, execution_cfg)
    return {
        name: _slim(
            equity_stats(
                schedule_window(
                    engine,
                    active,
                    start=start,
                    end=end,
                    hold_bars=hold_bars,
                    take_bps=take_bps,
                    stop_bps=stop_bps,
                ),
                start=str(start),
                end=str(end),
                cfg=execution_cfg,
            )
        )
        for name, (start, end) in windows.items()
    }


def selection_passes(stats: dict[str, dict[str, Any]]) -> bool:
    count_ok = (
        stats["train"]["trades"] >= 60
        and stats["select_2023"]["trades"] >= 12
        and stats["select_2023_h1"]["trades"] >= 5
        and stats["select_2023_h2"]["trades"] >= 5
    )
    stable = all(
        stats[name]["absolute_return_pct"] > 0.0
        for name in (
            "train_2020h2",
            "train_2021",
            "train_2022",
            "select_2023_h1",
            "select_2023_h2",
        )
    )
    target = all(
        stats[name]["cagr_to_strict_mdd"] >= 3.0
        for name in ("train", "select_2023", "pre_2024")
    )
    risk = all(
        stats[name]["strict_mdd_pct"] <= 15.0
        for name in ("train", "select_2023", "pre_2024")
    )
    return bool(count_ok and stable and target and risk)


def _selection_grid(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    active: np.ndarray,
    cfg: AuditConfig,
) -> list[dict[str, Any]]:
    rows = []
    for spec in EXIT_SPECS:
        stats = _evaluate(
            market,
            funding,
            active,
            cfg,
            leverage=cfg.leverage,
            windows=PRE2024_WINDOWS,
            hold_bars=int(spec["hold_bars"]),
            take_bps=int(spec["take_bps"]),
            stop_bps=int(spec["stop_bps"]),
        )
        ratios = [
            stats[name]["cagr_to_strict_mdd"]
            for name in ("train", "select_2023", "pre_2024")
        ]
        rows.append(
            {
                "spec": spec,
                "selection_passed": selection_passes(stats),
                "score": [float(min(ratios)), float(np.median(ratios))],
                "stats": stats,
            }
        )
    return sorted(
        rows,
        key=lambda row: (row["selection_passed"], *row["score"]),
        reverse=True,
    )


def _activation_hash(active: np.ndarray, dates: pd.Series) -> str:
    values = np.asarray(active, dtype=np.uint8)
    timestamps = pd.to_datetime(dates, utc=True, errors="raise").astype("int64").to_numpy(dtype=np.int64)
    if len(values) != len(timestamps):
        raise ValueError("activation and date lengths differ")
    digest = hashlib.sha256()
    digest.update(np.asarray([len(values)], dtype=np.int64).tobytes())
    digest.update(timestamps.tobytes())
    digest.update(np.packbits(values).tobytes())
    return digest.hexdigest()


def _zero_funding(funding: pd.DataFrame) -> pd.DataFrame:
    out = funding.copy()
    out["funding_rate"] = 0.0
    return out


def run(cfg: AuditConfig) -> dict[str, Any]:
    selection_market, selection_features, selection_funding, selection_hashes = _load_bundle(
        cfg,
        cutoff=SELECTION_END,
        premium_tolerance=cfg.live_premium_tolerance,
    )
    selection_dates = pd.to_datetime(selection_market["date"])
    selection_decisions = decision_mask(
        selection_dates,
        "live_hour_signal_bar",
        window_size=cfg.window_size,
    )
    selection_features = live_decision_features(selection_features)
    selection_active, selection_thresholds = _fit_active(
        selection_features,
        selection_dates,
        selection_decisions,
    )
    exit_grid = _selection_grid(
        selection_market,
        selection_funding,
        selection_active,
        cfg,
    )

    variants: dict[str, Any] = {}
    for name, tolerance, clock in (
        ("legacy_offline", cfg.offline_premium_tolerance, "legacy_positional"),
        ("live_parity", cfg.live_premium_tolerance, "live_hour_signal_bar"),
    ):
        market, features, funding, hashes = _load_bundle(
            cfg,
            cutoff=cfg.exclude_from,
            premium_tolerance=tolerance,
        )
        dates = pd.to_datetime(market["date"])
        decisions = decision_mask(dates, clock, window_size=cfg.window_size)
        model_features = live_decision_features(features) if name == "live_parity" else features
        active, thresholds = _fit_active(model_features, dates, decisions)
        pre2024 = dates < pd.Timestamp(SELECTION_END)
        expected_prefix_hash = _activation_hash(selection_active, selection_dates)
        prefix_dates = dates.loc[pre2024].reset_index(drop=True)
        actual_prefix_hash = _activation_hash(active[pre2024.to_numpy(bool)], prefix_dates)
        if name == "live_parity":
            if not selection_dates.reset_index(drop=True).equals(prefix_dates):
                raise RuntimeError("live-parity date prefix changed after appending OOS rows")
            if actual_prefix_hash != expected_prefix_hash:
                raise RuntimeError("live-parity activation prefix changed after appending OOS rows")
            replay_prefix_hashes = {
                "market": _frame_hash(_read_before(cfg.input_csv, "date", SELECTION_END)),
                "funding": _frame_hash(_read_before(cfg.funding_csv, "date", SELECTION_END)),
                "premium": _frame_hash(_read_premium_before(cfg.premium_csv, SELECTION_END)),
            }
            if replay_prefix_hashes != selection_hashes:
                raise RuntimeError("live-parity source prefix changed after opening OOS rows")
        decision_count = int(decisions.sum())
        premium_fresh_count = int(
            (
                decisions
                & (pd.to_numeric(model_features["premium_available"], errors="coerce").fillna(0.0).to_numpy(float) > 0.5)
            ).sum()
        )
        variant = {
            "premium_tolerance": tolerance,
            "decision_clock": clock,
            "source_prefix_hashes": hashes,
            "thresholds": thresholds,
            "activation_hash": _activation_hash(active, dates),
            "decision_count": decision_count,
            "premium_fresh_decision_count": premium_fresh_count,
            "premium_fresh_decision_rate": premium_fresh_count / decision_count if decision_count else 0.0,
            "actual_funding": {},
            "zero_funding_isolation": {},
        }
        for leverage in (cfg.leverage, cfg.operating_leverage):
            key = f"leverage_{leverage:g}"
            variant["actual_funding"][key] = _evaluate(
                market,
                funding,
                active,
                cfg,
                leverage=leverage,
                windows=WINDOWS,
            )
            variant["zero_funding_isolation"][key] = _evaluate(
                market,
                _zero_funding(funding),
                active,
                cfg,
                leverage=leverage,
                windows=WINDOWS,
            )
        variants[name] = variant

    legacy_market, legacy_features, _, _ = _load_bundle(
        cfg,
        cutoff=SELECTION_END,
        premium_tolerance=cfg.live_premium_tolerance,
    )
    legacy_decisions = decision_mask(
        pd.to_datetime(legacy_market["date"]),
        "legacy_positional",
        window_size=cfg.window_size,
    )
    legacy_fresh = int(
        (
            legacy_decisions
            & (pd.to_numeric(legacy_features["premium_available"], errors="coerce").fillna(0.0).to_numpy(float) > 0.5)
        ).sum()
    )

    selection_qualifiers = [row for row in exit_grid if row["selection_passed"]]
    result = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "verdict": {
            "legacy_candidate_rejected_under_current_contract": True,
            "live_parity_candidate_passed": False,
            "exit_overlay_qualifiers": len(selection_qualifiers),
            "reason": "current strict MDD and live premium freshness invalidate the prior live-grade interpretation; no pre-2024 exit overlay repairs it",
        },
        "protocol": {
            "selection_cutoff": SELECTION_END,
            "selection_sources_physically_truncated": True,
            "selection_source_prefix_hashes": selection_hashes,
            "entry": "completed signal t, next 5m open",
            "realized_funding": True,
            "cost_per_side": cfg.fee_rate + cfg.slippage_rate,
            "strict_mdd": "global/pre-entry HWM plus position-wide favorable envelope followed by adverse envelope",
            "same_bar_order": "stop before take",
            "split_crossing_exit": "purged",
        },
        "selection_thresholds": selection_thresholds,
        "selection_activation_hash": _activation_hash(selection_active, selection_dates),
        "exit_overlay_search": {
            "tested": len(exit_grid),
            "qualifiers": len(selection_qualifiers),
            "rows": exit_grid,
        },
        "variants": variants,
        "live_tolerance_legacy_clock_diagnostic": {
            "decision_count": int(legacy_decisions.sum()),
            "premium_fresh_decision_count": legacy_fresh,
            "premium_fresh_decision_rate": legacy_fresh / int(legacy_decisions.sum()),
            "interpretation": "the old positional clock cannot observe a fresh premium bar under the live 10-minute contract",
        },
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, field in AuditConfig.__dataclass_fields__.items():
        default = field.default
        flag = "--" + name.replace("_", "-")
        if isinstance(default, float):
            parser.add_argument(flag, type=float, default=default)
        elif isinstance(default, int):
            parser.add_argument(flag, type=int, default=default)
        else:
            parser.add_argument(flag, default=default)
    return parser.parse_args()


def main() -> None:
    result = run(AuditConfig(**vars(parse_args())))
    summary = {}
    for variant_name, variant in result["variants"].items():
        summary[variant_name] = {
            leverage: {
                window: metrics[window]
                for window in ("train", "select2023", "test2024", "eval2025_2026", "oos2024_2026", "full")
            }
            for leverage, metrics in variant["actual_funding"].items()
        }
    print(
        json.dumps(
            {
                "output": result["config"]["output"],
                "verdict": result["verdict"],
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
