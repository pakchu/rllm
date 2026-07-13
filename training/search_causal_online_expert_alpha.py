"""Search a causal online expert-selector alpha over six fixed sleeves.

The expert universe and execution rules are fixed.  A bounded selector grid is
evaluated with only completed counterfactual expert trades.  Selector policies
are ranked on 2023 while every source is physically truncated before 2024; the
frozen Top-10 manifest is then replayed unchanged on 2024-2026.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from preprocessing.binance_aux_features import (
    attach_binance_um_aux_frames,
    normalise_funding_history_frame,
    normalise_premium_index_frame,
)
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_interest_gate_validation import build_interest_features
from training.portfolio_opt_new_alpha_pool import ALPHAS, LONG_COMPONENTS
from training.search_funding_premium_external_state_gate_alpha import (
    _file_sha256,
    _frame_hash,
    _manifest_core_hash,
    _read_premium_before,
    _validate_manifest,
)
from training.search_positioning_hgb_path_alpha import _read_before


SELECTION_END = "2024-01-01"
STRIDE_BARS = 12
WINDOWS = {
    "fit": ("2020-06-01", "2023-01-01"),
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023_h1": ("2023-01-01", "2023-07-01"),
    "select_2023_h2": ("2023-07-01", "2024-01-01"),
    "test_2024": ("2024-01-01", "2025-01-01"),
    "eval_2025": ("2025-01-01", "2026-01-01"),
    "holdout_2026": ("2026-01-01", "2026-06-02"),
    "oos_2024_2026": ("2024-01-01", "2026-06-02"),
}
QUARTER_WINDOWS = {
    "2024Q1": ("2024-01-01", "2024-04-01"),
    "2024Q2": ("2024-04-01", "2024-07-01"),
    "2024Q3": ("2024-07-01", "2024-10-01"),
    "2024Q4": ("2024-10-01", "2025-01-01"),
    "2025Q1": ("2025-01-01", "2025-04-01"),
    "2025Q2": ("2025-04-01", "2025-07-01"),
    "2025Q3": ("2025-07-01", "2025-10-01"),
    "2025Q4": ("2025-10-01", "2026-01-01"),
    "2026Q1": ("2026-01-01", "2026-04-01"),
    "2026Q2_to_Jun02": ("2026-04-01", "2026-06-02"),
}


@dataclass(frozen=True)
class OnlineExpertConfig:
    input_csv: str
    funding_csv: str
    premium_csv: str
    output: str
    manifest_output: str
    docs_output: str
    exclude_from: str = "2026-06-02"
    window_size: int = 144
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    stress_fee_rate: float = 0.0009
    min_history: int = 5
    top_n: int = 10
    refresh_manifest: bool = False


def _window_mask(dates: pd.Series, name: str, windows: dict[str, tuple[str, str]] = WINDOWS) -> np.ndarray:
    start, end = windows[name]
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)


def _source_hashes(cfg: OnlineExpertConfig) -> dict[str, str]:
    return {
        str(Path(path)): _file_sha256(path)
        for path in (cfg.input_csv, cfg.funding_csv, cfg.premium_csv)
    }


def _load_bundle(
    cfg: OnlineExpertConfig,
    *,
    cutoff: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    market_raw = _read_before(cfg.input_csv, "date", cutoff)
    funding_raw = _read_before(cfg.funding_csv, "date", cutoff)
    premium_raw = _read_premium_before(cfg.premium_csv, cutoff)
    prefix_hashes = {
        "market": _frame_hash(market_raw),
        "funding": _frame_hash(funding_raw),
        "premium": _frame_hash(premium_raw),
    }
    market = market_raw.copy()
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    market = attach_binance_um_aux_frames(
        market,
        funding_frame=normalise_funding_history_frame(funding_raw),
        premium_frame=normalise_premium_index_frame(premium_raw),
        funding_tolerance="12h",
        premium_tolerance="2h",
    )
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(cutoff):
        raise RuntimeError("market source was not physically truncated")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise ValueError("online expert search requires a complete 5-minute grid")
    base = build_market_feature_frame(market, window_size=int(cfg.window_size))
    features = pd.concat([base, build_interest_features(market, base)], axis=1)
    return market, features, prefix_hashes


def _mask_conditions(features: pd.DataFrame, conditions: Iterable[tuple[str, str, float]]) -> np.ndarray:
    active = np.ones(len(features), dtype=bool)
    for column, operator, threshold in conditions:
        values = pd.to_numeric(features[column], errors="coerce").to_numpy(float)
        comparison = values <= float(threshold) if operator == "le" else values >= float(threshold)
        active &= np.isfinite(values) & comparison
    return active


def _expert_active(features: pd.DataFrame, name: str) -> np.ndarray:
    spec = ALPHAS[name]
    if "components" in spec:
        masks = [_mask_conditions(features, LONG_COMPONENTS[component]) for component in spec["components"]]
        return np.logical_or.reduce(masks)
    if spec["kind"] == "fx_stress":
        return _mask_conditions(features, (("htf_3d_return_1", "le", -0.0325294973), ("usdkrw_zscore", "ge", 1.3870063775)))
    if spec["kind"] == "premium_panic":
        return _mask_conditions(features, (("htf_3d_range_pos", "le", -0.5114186851), ("premium_index_zscore", "le", -1.47209312)))
    if spec["kind"] == "premium_kimchi_union":
        premium = _mask_conditions(features, (("htf_3d_range_pos", "le", -0.5114186851), ("premium_index_zscore", "le", -1.47209312)))
        kimchi = _mask_conditions(features, (("htf_3d_return_1", "le", -0.0303196833), ("kimchi_premium_change", "le", -0.0046123752)))
        return premium | kimchi
    raise KeyError(name)


def _make_event(
    market: pd.DataFrame,
    signal_pos: int,
    *,
    expert: str,
    cost_rate: float,
    entry_delay: int,
    leverage: float,
) -> dict[str, Any] | None:
    spec = ALPHAS[expert]
    hold = int(spec["hold"])
    entry = int(signal_pos) + int(entry_delay)
    max_exit = entry + hold
    if entry >= len(market) - 1 or max_exit >= len(market):
        return None
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    entry_open = float(opens[entry])
    if not np.isfinite(entry_open) or entry_open <= 0.0:
        return None
    ret = np.zeros(hold + 1, dtype=np.float64)
    adv = np.zeros(hold + 1, dtype=np.float64)
    ret[0] -= float(cost_rate) * float(leverage)
    # Strict adverse path starts after paying the entry-side cost.
    adv[0] -= float(cost_rate) * float(leverage)
    exit_pos = max_exit
    side = str(spec["side"])
    tp = spec.get("tp")
    sl = spec.get("sl")
    for j in range(entry, max_exit):
        offset = j - entry
        open_j = float(opens[j])
        if not np.isfinite(open_j) or open_j <= 0.0:
            continue
        if side == "long":
            if sl is not None and float(lows[j]) <= entry_open * (1.0 - float(sl)):
                realized = (entry_open * (1.0 - float(sl)) - open_j) / open_j
                ret[offset] += float(leverage) * realized
                adv[offset] += min(0.0, float(leverage) * realized)
                exit_pos = j
                break
            if tp is not None and float(highs[j]) >= entry_open * (1.0 + float(tp)):
                adverse = (float(lows[j]) - open_j) / open_j
                realized = (entry_open * (1.0 + float(tp)) - open_j) / open_j
                adv[offset] += min(0.0, float(leverage) * adverse)
                ret[offset] += float(leverage) * realized
                exit_pos = j
                break
            adverse = (float(lows[j]) - open_j) / open_j
            close_return = (float(opens[j + 1]) - open_j) / open_j
        else:
            if sl is not None and float(highs[j]) >= entry_open * (1.0 + float(sl)):
                realized = (open_j - entry_open * (1.0 + float(sl))) / open_j
                ret[offset] += float(leverage) * realized
                adv[offset] += min(0.0, float(leverage) * realized)
                exit_pos = j
                break
            if tp is not None and float(lows[j]) <= entry_open * (1.0 - float(tp)):
                adverse = (open_j - float(highs[j])) / open_j
                realized = (open_j - entry_open * (1.0 - float(tp))) / open_j
                adv[offset] += min(0.0, float(leverage) * adverse)
                ret[offset] += float(leverage) * realized
                exit_pos = j
                break
            adverse = (open_j - float(highs[j])) / open_j
            close_return = (open_j - float(opens[j + 1])) / open_j
        adv[offset] += min(0.0, float(leverage) * adverse)
        ret[offset] += float(leverage) * close_return
    used = exit_pos - entry + 1
    ret[used - 1] -= float(cost_rate) * float(leverage)
    ret = ret[:used]
    adv = adv[:used]
    realized_return = float(np.prod(np.maximum(0.0, 1.0 + ret)) - 1.0)
    max_adverse = float(max(0.0, -np.nanmin(adv))) if len(adv) else 0.0
    return {
        "expert": expert,
        "side": side,
        "signal_pos": int(signal_pos),
        "entry_pos": entry,
        "exit_pos": int(exit_pos),
        "ret": ret,
        "adv": adv,
        "realized_return": realized_return,
        "max_adverse": max_adverse,
    }


def _build_expert_events(
    market: pd.DataFrame,
    features: pd.DataFrame,
    cfg: OnlineExpertConfig,
    *,
    cost_rate: float,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    start = max(143, int(cfg.window_size) - 1)
    for expert in ALPHAS:
        active = _expert_active(features, expert)
        positions = np.arange(start, len(market), STRIDE_BARS, dtype=np.int64)
        next_allowed = 0
        for pos in positions[active[positions]]:
            if int(pos) < next_allowed:
                continue
            event = _make_event(
                market,
                int(pos),
                expert=expert,
                cost_rate=cost_rate,
                entry_delay=int(cfg.entry_delay_bars),
                leverage=float(cfg.leverage),
            )
            if event is None:
                continue
            events.append(event)
            next_allowed = int(event["exit_pos"])
    return sorted(events, key=lambda event: (event["signal_pos"], event["expert"]))


def _event_hash(events: Iterable[dict[str, Any]], *, before_pos: int | None = None) -> str:
    rows = []
    for event in events:
        if before_pos is not None and int(event["exit_pos"]) >= before_pos:
            continue
        rows.append(
            (
                event["expert"],
                event["side"],
                int(event["signal_pos"]),
                int(event["entry_pos"]),
                int(event["exit_pos"]),
                round(float(event["realized_return"]), 12),
                round(float(event["max_adverse"]), 12),
            )
        )
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _score(history: list[tuple[float, float]], *, lookback: int, method: str, mae_penalty: float) -> float:
    recent = history[-int(lookback):]
    if not recent:
        return float("-inf")
    returns = np.asarray([row[0] for row in recent], dtype=float)
    adverse = np.asarray([row[1] for row in recent], dtype=float)
    scale = max(1e-6, float(np.mean(np.abs(returns))))
    if method == "normalized_mean":
        utility = returns
    elif method == "adverse_utility":
        utility = returns - float(mae_penalty) * adverse
    else:
        raise KeyError(method)
    return float(np.mean(utility) / scale)


def _run_selector(
    events: list[dict[str, Any]],
    spec: dict[str, Any],
    *,
    min_history: int,
) -> list[dict[str, Any]]:
    by_signal: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        by_signal.setdefault(int(event["signal_pos"]), []).append(event)
    histories: dict[str, list[tuple[float, float]]] = {name: [] for name in ALPHAS}
    by_exit = sorted(events, key=lambda event: (event["exit_pos"], event["expert"]))
    release_index = 0
    next_allowed = 0
    accepted: list[dict[str, Any]] = []
    for signal_pos in sorted(by_signal):
        while release_index < len(by_exit) and int(by_exit[release_index]["exit_pos"]) <= signal_pos:
            matured = by_exit[release_index]
            histories[matured["expert"]].append((float(matured["realized_return"]), float(matured["max_adverse"])))
            release_index += 1
        scores = {
            expert: _score(
                history,
                lookback=int(spec["lookback"]),
                method=str(spec["method"]),
                mae_penalty=float(spec["mae_penalty"]),
            )
            for expert, history in histories.items()
            if len(history) >= int(min_history)
        }
        ranked = sorted(scores, key=lambda expert: (-scores[expert], expert))
        allowed = set(ranked[: int(spec["top_k"])])
        eligible = [
            event
            for event in by_signal[signal_pos]
            if event["expert"] in allowed and scores[event["expert"]] >= float(spec["threshold"])
        ]
        if signal_pos < next_allowed or not eligible:
            continue
        selected = min(eligible, key=lambda event: (-scores[event["expert"]], event["expert"]))
        accepted.append(selected)
        next_allowed = int(selected["exit_pos"])
    return accepted


def _global_nonoverlap(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic no-learning union comparator over the same event universe."""
    accepted: list[dict[str, Any]] = []
    next_allowed = 0
    for event in sorted(events, key=lambda row: (row["signal_pos"], row["expert"])):
        if int(event["signal_pos"]) < next_allowed:
            continue
        accepted.append(event)
        next_allowed = int(event["exit_pos"])
    return accepted


def _normal_p_value(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    array = np.asarray(values, dtype=float)
    std = float(array.std(ddof=1))
    if not np.isfinite(std) or std <= 1e-12:
        return 0.0 if abs(float(array.mean())) > 1e-12 else 1.0
    z = abs(float(array.mean())) / (std / math.sqrt(len(array)))
    return float(math.erfc(z / math.sqrt(2.0)))


def _metric(
    events: list[dict[str, Any]],
    dates: pd.Series,
    start: str,
    end: str,
) -> dict[str, Any]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    window = ((dates >= start_ts) & (dates < end_ts)).to_numpy(bool)
    positions = np.flatnonzero(window)
    if not len(positions):
        raise ValueError(f"empty metric window {start}..{end}")
    first, last = int(positions[0]), int(positions[-1]) + 1
    selected = [
        event for event in events
        if int(event["signal_pos"]) >= first and int(event["exit_pos"]) < last
    ]
    ret = np.zeros(last - first, dtype=np.float64)
    adv = np.zeros(last - first, dtype=np.float64)
    for event in selected:
        offset = int(event["entry_pos"]) - first
        length = len(event["ret"])
        ret[offset: offset + length] += event["ret"]
        adv[offset: offset + length] += event["adv"]
    eq = peak = 1.0
    max_drawdown = 0.0
    for bar_return, bar_adverse in zip(ret, adv):
        adverse_equity = eq * max(0.0, 1.0 + float(bar_adverse))
        max_drawdown = max(max_drawdown, 1.0 - adverse_equity / max(peak, 1e-12))
        eq *= max(0.0, 1.0 + float(bar_return))
        peak = max(peak, eq)
        max_drawdown = max(max_drawdown, 1.0 - eq / max(peak, 1e-12))
    years = (end_ts - start_ts).total_seconds() / (365.25 * 24 * 3600)
    return_pct = (eq - 1.0) * 100.0
    cagr_pct = ((eq ** (1.0 / years) - 1.0) * 100.0) if eq > 0.0 else -100.0
    mdd_pct = max_drawdown * 100.0
    realized = [float(event["realized_return"]) for event in selected]
    long_count = sum(event["side"] == "long" for event in selected)
    return {
        "return_pct": float(return_pct),
        "cagr_pct": float(cagr_pct),
        "strict_mdd_pct": float(mdd_pct),
        "ratio": float(cagr_pct / mdd_pct) if mdd_pct > 1e-12 else 0.0,
        "trades": len(selected),
        "long_trades": int(long_count),
        "short_trades": int(len(selected) - long_count),
        "win_rate": float(np.mean(np.asarray(realized) > 0.0)) if realized else 0.0,
        "p_value_mean_return_approx": _normal_p_value(realized),
        "years": float(years),
    }


def _stats(events: list[dict[str, Any]], dates: pd.Series, names: Iterable[str]) -> dict[str, Any]:
    return {name: _metric(events, dates, *WINDOWS[name]) for name in names}


def _policy_specs() -> list[dict[str, Any]]:
    specs = []
    for method in ("normalized_mean", "adverse_utility"):
        penalties = (0.0,) if method == "normalized_mean" else (0.5, 1.0)
        for mae_penalty in penalties:
            for lookback in (10, 20, 40):
                for top_k in (1, 2, 3):
                    for threshold in (0.0, 0.10):
                        specs.append(
                            {
                                "method": method,
                                "mae_penalty": mae_penalty,
                                "lookback": lookback,
                                "top_k": top_k,
                                "threshold": threshold,
                            }
                        )
    return specs


def _path_hash(events: list[dict[str, Any]], dates: pd.Series, window: str) -> str:
    mask = _window_mask(dates, window)
    positions = np.flatnonzero(mask)
    first, last = int(positions[0]), int(positions[-1]) + 1
    path = [
        (event["expert"], int(event["signal_pos"]), int(event["exit_pos"]))
        for event in events
        if int(event["signal_pos"]) >= first and int(event["exit_pos"]) < last
    ]
    return hashlib.sha256(json.dumps(path, separators=(",", ":")).encode()).hexdigest()


def _selection_score(stats: dict[str, Any]) -> float:
    full = stats["select_2023"]
    h1, h2 = stats["select_2023_h1"], stats["select_2023_h2"]
    if full["return_pct"] <= 0.0 or full["trades"] < 8 or min(h1["trades"], h2["trades"]) < 2:
        return -1e12
    stability = min(h1["ratio"], h2["ratio"])
    return float(full["ratio"] + 0.25 * stability + 0.02 * min(full["trades"], 50))


def _select_manifest(cfg: OnlineExpertConfig) -> dict[str, Any]:
    market, features, prefix_hashes = _load_bundle(cfg, cutoff=SELECTION_END)
    dates = pd.to_datetime(market["date"])
    events = _build_expert_events(market, features, cfg, cost_rate=float(cfg.fee_rate + cfg.slippage_rate))
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for spec in _policy_specs():
        accepted = _run_selector(events, spec, min_history=int(cfg.min_history))
        stats = _stats(accepted, dates, ("fit", "select_2023", "select_2023_h1", "select_2023_h2"))
        score = _selection_score(stats)
        if score <= -1e11:
            continue
        path_hash = _path_hash(accepted, dates, "select_2023")
        if path_hash in seen_paths:
            continue
        seen_paths.add(path_hash)
        rows.append({"spec": spec, "selection_score": score, "selection_stats": stats, "selection_path_hash": path_hash})
    rows.sort(key=lambda row: (-row["selection_score"], json.dumps(row["spec"], sort_keys=True)))
    selected = rows[: int(cfg.top_n)]
    core = {
        "protocol": {
            "expert_universe": list(ALPHAS),
            "online_update": "release every counterfactual expert reward only when its actual exit_pos <= current signal_pos",
            "history": "expert-local rolling completed-event utility; no current or future event outcome",
            "selection": {name: WINDOWS[name] for name in ("select_2023", "select_2023_h1", "select_2023_h2")},
            "all_sources_physically_excluded_before_manifest": True,
            "later_metrics_included": False,
            "search_cap": f"{len(_policy_specs())} fixed policies",
            "entry": "next 5m open",
            "execution": "one global non-overlapping position; selected expert's fixed hold/TP/SL",
            "cost": "6bp/side base; 10bp/side stress; 0.5x leverage",
            "annualization": "full calendar window including idle time",
            "mdd": "strict bar-level adverse OHLC excursion plus realized equity high-water",
            "status_ceiling": "shadow research; no retrospective live promotion",
            "expert_provenance_warning": "the six sleeve definitions were committed on 2026-07-10 after this programme had inspected 2024-2026; selector OOS is mechanically frozen but not a fresh-data claim",
        },
        "source_prefix_hashes": prefix_hashes,
        "expert_event_hash": _event_hash(events),
        "search_space": {"raw_specs": len(_policy_specs()), "eligible_unique_paths": len(rows), "top_n": int(cfg.top_n)},
        "selected": selected,
    }
    manifest = {"as_of": datetime.now(timezone.utc).isoformat(), "sha256": _manifest_core_hash(core), **core}
    path = Path(cfg.manifest_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    return manifest


def _replay(cfg: OnlineExpertConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    _validate_manifest(manifest)
    prefix_market, prefix_features, prefix_hashes = _load_bundle(cfg, cutoff=SELECTION_END)
    if prefix_hashes != manifest["source_prefix_hashes"]:
        raise RuntimeError("pre-2024 source prefixes changed after manifest freeze")
    prefix_events = _build_expert_events(
        prefix_market,
        prefix_features,
        cfg,
        cost_rate=float(cfg.fee_rate + cfg.slippage_rate),
    )
    if _event_hash(prefix_events) != manifest["expert_event_hash"]:
        raise RuntimeError("pre-2024 expert event reconstruction drift")
    market, features, _ = _load_bundle(cfg, cutoff=cfg.exclude_from)
    dates = pd.to_datetime(market["date"])
    events = _build_expert_events(market, features, cfg, cost_rate=float(cfg.fee_rate + cfg.slippage_rate))
    cutoff_pos = int(np.searchsorted(dates.to_numpy(), np.datetime64(SELECTION_END)))
    if _event_hash(events, before_pos=cutoff_pos) != manifest["expert_event_hash"]:
        raise RuntimeError("full replay pre-2024 expert event prefix drift")
    stress_cfg = replace(cfg, fee_rate=cfg.stress_fee_rate, slippage_rate=0.0001)
    stress_events = _build_expert_events(
        market,
        features,
        stress_cfg,
        cost_rate=float(stress_cfg.fee_rate + stress_cfg.slippage_rate),
    )
    baseline_windows = ("test_2024", "eval_2025", "holdout_2026", "oos_2024_2026")
    baselines = {
        "fixed_experts": {
            expert: _stats([event for event in events if event["expert"] == expert], dates, baseline_windows)
            for expert in ALPHAS
        },
        "deterministic_all_expert_union": _stats(_global_nonoverlap(events), dates, baseline_windows),
    }
    selected_rows = []
    union_combined = baselines["deterministic_all_expert_union"]["oos_2024_2026"]
    for rank, frozen in enumerate(manifest["selected"], start=1):
        accepted = _run_selector(events, frozen["spec"], min_history=int(cfg.min_history))
        selection_stats = _stats(accepted, dates, ("fit", "select_2023", "select_2023_h1", "select_2023_h2"))
        if selection_stats != frozen["selection_stats"]:
            raise RuntimeError(f"selection metric drift at rank {rank}")
        if _path_hash(accepted, dates, "select_2023") != frozen["selection_path_hash"]:
            raise RuntimeError(f"selection path drift at rank {rank}")
        stats = _stats(accepted, dates, WINDOWS)
        stress_accepted = _run_selector(stress_events, frozen["spec"], min_history=int(cfg.min_history))
        stress = _stats(stress_accepted, dates, ("test_2024", "eval_2025", "holdout_2026", "oos_2024_2026"))
        quarterly = {
            name: _metric(accepted, dates, start, end)
            for name, (start, end) in QUARTER_WINDOWS.items()
        }
        quarter_summary = {
            "positive_return_quarters": sum(row["return_pct"] > 0.0 for row in quarterly.values()),
            "negative_return_quarters": sum(row["return_pct"] < 0.0 for row in quarterly.values()),
            "flat_quarters": sum(row["trades"] == 0 for row in quarterly.values()),
            "total_quarters": len(quarterly),
        }
        test, evaluation = stats["test_2024"], stats["eval_2025"]
        holdout, combined = stats["holdout_2026"], stats["oos_2024_2026"]
        passes_standalone_gate = (
            test["return_pct"] > 0.0 and test["ratio"] >= 2.5 and test["trades"] >= 20
            and evaluation["return_pct"] > 0.0 and evaluation["ratio"] >= 2.5 and evaluation["trades"] >= 20
            and holdout["return_pct"] > 0.0 and holdout["trades"] >= 12
            and combined["return_pct"] > 0.0
        )
        adds_value_vs_union = (
            combined["ratio"] > union_combined["ratio"]
            and combined["return_pct"] > union_combined["return_pct"]
        )
        passes_alpha_pool = passes_standalone_gate and adds_value_vs_union
        bonferroni = min(1.0, combined["p_value_mean_return_approx"] * max(1, len(manifest["selected"])))
        strong_shadow = (
            passes_alpha_pool
            and min(test["ratio"], evaluation["ratio"], holdout["ratio"], combined["ratio"]) >= 3.0
            and quarter_summary["positive_return_quarters"] >= 7
            and quarter_summary["negative_return_quarters"] <= 1
            and bonferroni <= 0.05
            and min(stress[name]["ratio"] for name in ("test_2024", "eval_2025", "holdout_2026")) >= 2.5
        )
        selected_rows.append(
            {
                "manifest_rank": rank,
                **frozen,
                "stats": stats,
                "stress_10bp_each_side": stress,
                "quarterly_stats": quarterly,
                "quarterly_summary": quarter_summary,
                "top_n_bonferroni_p_value": float(bonferroni),
                "passes_standalone_gate": bool(passes_standalone_gate),
                "adds_value_vs_deterministic_union": bool(adds_value_vs_union),
                "passes_alpha_pool": bool(passes_alpha_pool),
                "passes_strong_shadow": bool(strong_shadow),
                "passes_live_grade": False,
            }
        )
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "manifest": cfg.manifest_output,
        "manifest_sha256": manifest["sha256"],
        "protocol": manifest["protocol"],
        "source_file_hashes_after_manifest_freeze": _source_hashes(cfg),
        "expert_event_counts": {name: sum(event["expert"] == name for event in events) for name in ALPHAS},
        "baselines": baselines,
        "selected": selected_rows,
        "standalone_gate_qualifiers": [row for row in selected_rows if row["passes_standalone_gate"]],
        "alpha_pool_qualifiers": [row for row in selected_rows if row["passes_alpha_pool"]],
        "strong_shadow": [row for row in selected_rows if row["passes_strong_shadow"]],
        "live_grade": [],
    }


def _fmt(row: dict[str, Any]) -> str:
    return f"{row['return_pct']:.2f}/{row['cagr_pct']:.2f}/{row['strict_mdd_pct']:.2f}/{row['ratio']:.2f}/{row['trades']}"


def _write_doc(cfg: OnlineExpertConfig, report: dict[str, Any]) -> None:
    lines = [
        "# Causal online expert alpha search (2026-07-13)",
        "",
        "This experiment replaces static future-selected gates with a selector that learns only from expert trades whose exits are already observable.",
        "",
        "Metric format: `absolute return / CAGR / strict MDD / CAGR-MDD / trades`.",
        "",
        "## Frozen Top-10 replay",
        "",
        "| rank | selector | 2024 test | 2025 eval | 2026 holdout | combined | pool | strong |",
        "|---:|---|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in report["selected"]:
        stats = row["stats"]
        lines.append(
            f"| {row['manifest_rank']} | `{row['spec']}` | {_fmt(stats['test_2024'])} | {_fmt(stats['eval_2025'])} | "
            f"{_fmt(stats['holdout_2026'])} | {_fmt(stats['oos_2024_2026'])} | "
            f"{'yes' if row['passes_alpha_pool'] else 'no'} | {'yes' if row['passes_strong_shadow'] else 'no'} |"
        )
    lines += [
        "",
        "## Fixed-policy comparators",
        "",
        "| policy | 2024 test | 2025 eval | 2026 holdout | combined |",
        "|---|---:|---:|---:|---:|",
    ]
    comparators = dict(report["baselines"]["fixed_experts"])
    comparators["deterministic_all_expert_union"] = report["baselines"]["deterministic_all_expert_union"]
    for name, stats in comparators.items():
        lines.append(
            f"| `{name}` | {_fmt(stats['test_2024'])} | {_fmt(stats['eval_2025'])} | "
            f"{_fmt(stats['holdout_2026'])} | {_fmt(stats['oos_2024_2026'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        f"- Standalone metric qualifiers: {len(report['standalone_gate_qualifiers'])}; incremental alpha-pool qualifiers: {len(report['alpha_pool_qualifiers'])}; strong-shadow qualifiers: {len(report['strong_shadow'])}; live-grade: 0.",
        "- The deterministic no-learning union is the required marginal-value comparator. No selector beat it on both combined return and CAGR/MDD, so this online selection usage is rejected as a new alpha.",
        "- The six experts are fixed templates. The selector may adapt only after a counterfactual expert trade has fully exited.",
        "- 2024+ rows did not influence selector hyperparameters, Top-10 ranking, or path de-duplication in this run.",
        "- Important provenance limit: the six sleeve definitions were committed on 2026-07-10 after this research programme had repeatedly inspected 2024-2026. These are mechanically frozen historical replays, not fresh-data OOS claims.",
        "- The rank-8 strong row was discovered only after replay as one member of the frozen Top-10; its reported p-value is Bonferroni-adjusted for all ten members.",
        "- Even a passing row remains shadow research because this research programme has repeatedly inspected the same later windows.",
        "",
        "## Reproduction",
        "",
        "```bash",
        f"python -m training.search_causal_online_expert_alpha --input-csv {cfg.input_csv} --funding-csv {cfg.funding_csv} --premium-csv {cfg.premium_csv} --manifest-output {cfg.manifest_output} --output {cfg.output} --docs-output {cfg.docs_output}",
        "```",
    ]
    path = Path(cfg.docs_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run(cfg: OnlineExpertConfig) -> dict[str, Any]:
    manifest_path = Path(cfg.manifest_output)
    if manifest_path.exists() and not cfg.refresh_manifest:
        manifest = json.loads(manifest_path.read_text())
        _validate_manifest(manifest)
    else:
        manifest = _select_manifest(cfg)
    report = _replay(cfg, manifest)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    _write_doc(cfg, report)
    return report


def parse_args() -> OnlineExpertConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--funding-csv", required=True)
    parser.add_argument("--premium-csv", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--docs-output", required=True)
    parser.add_argument("--exclude-from", default=OnlineExpertConfig.exclude_from)
    parser.add_argument("--top-n", type=int, default=OnlineExpertConfig.top_n)
    parser.add_argument("--min-history", type=int, default=OnlineExpertConfig.min_history)
    parser.add_argument("--refresh-manifest", action="store_true")
    return OnlineExpertConfig(**vars(parser.parse_args()))


def main() -> None:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "output": report["config"]["output"],
                "manifest": report["manifest"],
                "alpha_pool_qualifiers": len(report["alpha_pool_qualifiers"]),
                "standalone_gate_qualifiers": len(report["standalone_gate_qualifiers"]),
                "strong_shadow": len(report["strong_shadow"]),
                "top": report["selected"][:3],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
