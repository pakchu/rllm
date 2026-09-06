"""Fresh standalone audit for the frozen legacy dollar-rally short candidate.

This is intentionally not a search.  It replays only ``top[0]`` from
``results/short_base_alpha_scan_fast2_2026-07-08.json`` with the repository's
current G9 fixed-unit ledger/funding accounting and writes the result under
``research/legacy_dollar_rally_short``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from preprocessing import market_features as market_feature_helpers
from preprocessing.market_features import build_market_feature_frame
from training import evaluate_g9_macro_historical as g9hist
from training import g9_joint_net_ledger as ledger
from training import search_independent_short_candidates as ind_short
from training import search_meaningful_alpha_combinations as base

OUT = base.ROOT / "research/legacy_dollar_rally_short"
LEGACY_SCAN = base.ROOT / "results/short_base_alpha_scan_fast2_2026-07-08.json"
WINDOWS: dict[str, tuple[str, str]] = {
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
    "full2026H1": ("2026-01-01", "2026-07-01"),
}
COSTS = (0.0006, 0.0010)
WINDOW_SIZE = 144
ENTRY_DELAY_BARS = 1

DESIGN: dict[str, Any] = {
    "version": 1,
    "selection": "audit only frozen top[0] from legacy short_base_alpha_scan_fast2_2026-07-08; no tuning, no replacement",
    "candidate": {
        "name": "legacy_dollar_rally_short_top0",
        "side": "SHORT",
        "gates": [
            {"feature": "dxy_momentum", "op": "ge", "threshold": 0.0021818982809893497},
            {"feature": "htf_1d_return_4", "op": "ge", "threshold": 0.016096783732847175},
        ],
        "hold_bars": 144,
        "hold_hours": 12,
        "stride_bars": 12,
        "take_profit": None,
        "stop_loss": None,
    },
    "clock": {
        "feature_rows": "5m market feature frame, features at signal bar depend only on rows <= signal bar",
        "entry": "signal bar + 1 five-minute bar open",
        "global_phase": "np.arange(window_size-1, n-hold-entry_delay, stride) over full merged market",
        "non_overlap": "search_independent_short_candidates.schedule; next entry must be after prior exit",
    },
    "accounting": "search_independent_short_candidates.exact over g9_joint_net_ledger; fixed 1x short target on entry, realized funding from blocks(), 6bp/10bp per side",
    "feature_source": "preprocessing.market_features.build_market_feature_frame(window_size=144) on evaluate_g9_macro_historical.merged_market()",
    "source_sensitivity": ["original", "require_dxy_available_at_signal", "delay_dxy_gate_1h", "delay_dxy_gate_24h"],
    "live_enabled": False,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    return value


def legacy_top0(path: Path = LEGACY_SCAN) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    top = payload.get("top", [])
    if not top:
        raise ValueError(f"legacy scan has no top rows: {path}")
    row = top[0]
    expected = DESIGN["candidate"]
    if int(row.get("hold")) != int(expected["hold_bars"]):
        raise ValueError("legacy top[0] hold no longer matches frozen audit")
    if int(row.get("stride")) != int(expected["stride_bars"]):
        raise ValueError("legacy top[0] stride no longer matches frozen audit")
    gates = row.get("gates", [])
    frozen = {(g["feature"], g["op"], float(g["thr"])) for g in gates}
    required = {("dxy_momentum", "ge", 0.0021818982809893497), ("htf_1d_return_4", "ge", 0.016096783732847175)}
    if frozen != required:
        raise ValueError("legacy top[0] gates no longer match frozen audit")
    return row


def register() -> dict[str, Any]:
    legacy = legacy_top0()
    payload = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": DESIGN,
        "hashes": {
            "self": base.sha(__file__),
            "legacy_scan": base.sha(LEGACY_SCAN),
            "evaluate_g9_macro_historical": base.sha(g9hist.__file__),
            "g9_joint_net_ledger": base.sha(ledger.__file__),
            "search_independent_short_candidates": base.sha(ind_short.__file__),
            "preprocessing_market_features": base.sha(market_feature_helpers.__file__),
            "market_cache": base.sha(g9hist.MARKET),
            "funding_cache": base.sha(g9hist.FUND),
            "spot_cache": base.sha(g9hist.SPOT),
            "raw_enriched_cache": base.sha(g9hist.RAW),
        },
        "legacy_prior": {
            "as_of": json.loads(LEGACY_SCAN.read_text()).get("as_of"),
            "assumptions": json.loads(LEGACY_SCAN.read_text()).get("assumptions"),
            "top0": legacy,
            "historical_exposure_warning": "The old record selected/ranked with OOS inspected and used older cost/accounting; these stats are prior exposed evidence, not fresh OOS.",
        },
    }
    base.write_json(OUT / "design.json", _jsonable(payload))
    return payload


def _gate_active(features: pd.DataFrame, *, dxy_shift: int = 0) -> np.ndarray:
    dxy = features["dxy_momentum"].shift(int(dxy_shift)) if dxy_shift else features["dxy_momentum"]
    htf = features["htf_1d_return_4"]
    active = (
        (dxy.to_numpy(float) >= 0.0021818982809893497)
        & (htf.to_numpy(float) >= 0.016096783732847175)
        & np.isfinite(dxy.to_numpy(float))
        & np.isfinite(htf.to_numpy(float))
    )
    return active


def _global_signal_positions(n: int, *, hold_bars: int = 144, stride_bars: int = 12) -> np.ndarray:
    # Matches the visible legacy scanner convention in nonrex_short_bear_tp_refine:
    # signal at ``window_size-1`` then every global stride; entry is next 5m open.
    stop = max(0, int(n) - int(hold_bars) - ENTRY_DELAY_BARS)
    return np.arange(WINDOW_SIZE - 1, stop, int(stride_bars), dtype=np.int64)


def _window_entries(
    dates: pd.DatetimeIndex,
    features: pd.DataFrame,
    *,
    start: str,
    end: str,
    variant: str,
    hold_bars: int = 144,
    stride_bars: int = 12,
) -> tuple[np.ndarray, dict[str, Any]]:
    signals = _global_signal_positions(len(dates), hold_bars=hold_bars, stride_bars=stride_bars)
    active = _gate_active(features)
    dxy_available = features["dxy_available"].to_numpy(float) > 0.5 if "dxy_available" in features else np.zeros(len(features), bool)
    if variant == "original":
        variant_active = active
    elif variant == "require_dxy_available_at_signal":
        variant_active = active & dxy_available
    elif variant == "delay_dxy_gate_1h":
        variant_active = _gate_active(features, dxy_shift=12)
    elif variant == "delay_dxy_gate_24h":
        variant_active = _gate_active(features, dxy_shift=288)
    else:
        raise ValueError(f"unknown variant {variant}")
    entry = signals + ENTRY_DELAY_BARS
    mask = (dates[entry] >= pd.Timestamp(start)) & (dates[entry] < pd.Timestamp(end)) & variant_active[signals]
    selected_signals = signals[mask]
    selected_entries = entry[mask]
    availability = {
        "candidate_signal_count_before_schedule": int(len(selected_entries)),
        "dxy_available_at_signal_count": int(dxy_available[selected_signals].sum()) if len(selected_signals) else 0,
        "dxy_unavailable_at_signal_count": int((~dxy_available[selected_signals]).sum()) if len(selected_signals) else 0,
        "dxy_unavailable_at_signal_share": float((~dxy_available[selected_signals]).mean()) if len(selected_signals) else 0.0,
    }
    return selected_entries, availability


def _trades_to_rows(trades: dict[str, np.ndarray], dates: pd.DatetimeIndex) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for j in range(len(trades["entry"])):
        rows.append(
            {
                "entry_index": int(trades["entry"][j]),
                "entry_date": str(dates[int(trades["entry"][j])]),
                "exit_index": int(trades["exit"][j]),
                "exit_date": str(dates[min(int(trades["exit"][j]), len(dates) - 1)]),
                "barrier": bool(trades["barrier"][j]),
                "exit_price": float(trades["exit_price"][j]),
                "gross_factor_before_cost": float(trades["gross_factor"][j]),
                "exit_ratio": float(trades["exit_ratio"][j]),
            }
        )
    return rows


def replay_window(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    features: pd.DataFrame,
    window: str,
    variant: str = "original",
) -> dict[str, Any]:
    start, end = WINDOWS[window]
    ids, d = g9hist.blocks(market, funding, start, end)
    full_dates = pd.DatetimeIndex(market["date"])
    local_dates = pd.DatetimeIndex(d["date"])
    entries, availability = _window_entries(full_dates, features, start=start, end=end, variant=variant)
    local_entries = np.searchsorted(local_dates.to_numpy(), full_dates[entries].to_numpy())
    if len(local_entries) and not np.array_equal(local_dates[local_entries].to_numpy(), full_dates[entries].to_numpy()):
        raise ValueError("entry dates failed to map onto local window grid")
    potential = ind_short.potential_trades(d, local_entries, hold=12, tp=None, sl=None)
    trades = ind_short.schedule(potential, np.ones(len(local_entries), dtype=bool))
    metrics = {str(cost): ind_short.exact(d, trades, cost) for cost in COSTS}
    return {
        "window": window,
        "variant": variant,
        "period": {"start": start, "end_exclusive": end, "bars": int(len(local_dates))},
        "availability": availability,
        "metrics": metrics,
        "trade_count_after_schedule": int(len(trades["entry"])),
        "trades": _trades_to_rows(trades, local_dates),
    }


def run() -> dict[str, Any]:
    registration = register()
    market, funding, source_receipt = g9hist.merged_market()
    features = build_market_feature_frame(market, window_size=WINDOW_SIZE)
    reports: dict[str, Any] = {}
    trades_export: dict[str, Any] = {}
    variants = ["original", "require_dxy_available_at_signal", "delay_dxy_gate_1h", "delay_dxy_gate_24h"]
    for window in WINDOWS:
        reports[window] = {}
        trades_export[window] = {}
        for variant in variants:
            row = replay_window(market, funding, features, window, variant)
            trades_export[window][variant] = row.pop("trades")
            reports[window][variant] = row
            if variant == "original":
                base6 = row["metrics"]["0.0006"]
                print(window, {k: base6[k] for k in ["return_pct", "mdd_pct", "trades", "win_rate", "funding_pct_initial"]}, flush=True)
    result = {
        "registration": registration,
        "source_receipt": source_receipt,
        "input_range": {"start": str(market.date.iloc[0]), "end": str(market.date.iloc[-1]), "rows": int(len(market))},
        "reports": reports,
        "live_enabled": False,
        "limitations": [
            "No parameter tuning, no alternative candidate selection, and no live config/commit changes.",
            "Legacy top[0] and all requested replay windows were historically exposed before this audit; results are fresh accounting, not pristine OOS.",
            "Original replay preserves the legacy DXY feature behavior even when source availability flags indicate stale/unavailable FX rows; conservative variants quantify sensitivity.",
            "Intrabar path ordering is bounded by g9_joint_net_ledger/search_independent_short_candidates conventions, not tick-order reconstruction.",
        ],
    }
    base.write_json(OUT / "report.json", _jsonable(result))
    base.write_json(OUT / "trades.json", _jsonable(trades_export))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="write design registration only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.freeze:
        print(json.dumps(_jsonable(register()), indent=2, ensure_ascii=False))
    else:
        report = run()
        compact = {
            "output": str(OUT / "report.json"),
            "design": str(OUT / "design.json"),
            "original_base_6bp": {
                window: report["reports"][window]["original"]["metrics"]["0.0006"] for window in WINDOWS
            },
        }
        print(json.dumps(_jsonable(compact), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
