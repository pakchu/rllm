"""Search causal spot-perpetual residual transition alphas.

The signal uses the residual between the directly observed Binance perpetual-
spot basis and the Binance premium index after a trailing, shifted regression.
It tests contraction and first expansion-onset transitions. Policies are
selected with all sources physically truncated before 2024 and replayed on
2024-2026 without modification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training.search_causal_online_expert_alpha import (
    ALPHAS,
    OnlineExpertConfig,
    _build_expert_events,
    _global_nonoverlap,
    _load_bundle as _load_expert_bundle,
    _metric,
)
from training.search_funding_premium_external_state_gate_alpha import (
    _file_sha256,
    _frame_hash,
    _manifest_core_hash,
    _validate_manifest,
)
from training.search_positioning_hgb_path_alpha import _feature_hash, _read_before


SELECTION_END = "2024-01-01"
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
ROLLING_WINDOWS = (2016, 8640)
Z_ENTRIES = (2.0, 3.0)
CONTRACTION_DELTAS = (0.0, 0.25)
MODES = ("basis_reversion", "lead_residual", "flow_fade", "spot_absorption")
DIRECTIONS = ("contra", "continuation")
PHASES = ("contraction", "expansion_onset")
MAX_HOLDS = (48, 96, 144)


@dataclass(frozen=True)
class SpotPerpConfig:
    input_csv: str
    spot_csv: str
    funding_csv: str
    premium_csv: str
    output: str
    manifest_output: str
    docs_output: str
    exclude_from: str = "2026-06-02"
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    stress_fee_rate: float = 0.0009
    exit_abs_z: float = 0.5
    top_n: int = 10
    top_per_mode: int = 3
    refresh_manifest: bool = False


def _window_mask(dates: pd.Series, name: str) -> np.ndarray:
    start, end = WINDOWS[name]
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)


def _source_hashes(cfg: SpotPerpConfig) -> dict[str, str]:
    return {
        str(Path(path)): _file_sha256(path)
        for path in (cfg.input_csv, cfg.spot_csv, cfg.funding_csv, cfg.premium_csv)
    }


def _load_market_spot(
    cfg: SpotPerpConfig,
    *,
    cutoff: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    market_raw = _read_before(cfg.input_csv, "date", cutoff)
    spot_raw = _read_before(cfg.spot_csv, "date", cutoff)
    prefix_hashes = {"market": _frame_hash(market_raw), "spot": _frame_hash(spot_raw)}
    market = market_raw.copy()
    spot = spot_raw.copy()
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    spot["date"] = pd.to_datetime(spot["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last")
    spot = spot.sort_values("date").drop_duplicates("date", keep="last")
    columns = [
        "date",
        "spot_close",
        "spot_volume",
        "spot_rows",
        "premium_index_1m_close",
        "premium_rows",
    ]
    market = market.merge(spot[columns], on="date", how="left", validate="one_to_one").reset_index(drop=True)
    market["spot_available"] = (
        pd.to_numeric(market["spot_rows"], errors="coerce").eq(5)
        & pd.to_numeric(market["premium_rows"], errors="coerce").eq(5)
        & pd.to_numeric(market["spot_close"], errors="coerce").gt(0.0)
        & pd.to_numeric(market["premium_index_1m_close"], errors="coerce").notna()
    )
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(cutoff):
        raise RuntimeError("market/spot bundle was not physically truncated")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise ValueError("spot-perp search requires a complete futures 5-minute grid")
    return market, prefix_hashes


def _prior_z(values: pd.Series, window: int) -> pd.Series:
    prior = pd.to_numeric(values, errors="coerce").shift(1)
    minimum = max(288, window // 2)
    mean = prior.rolling(window, min_periods=minimum).mean()
    std = prior.rolling(window, min_periods=minimum).std(ddof=0).replace(0.0, np.nan)
    return (pd.to_numeric(values, errors="coerce") - mean) / std


def _rolling_residual(y: pd.Series, x: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Current residual against a regression fitted strictly through t-1."""
    y = pd.to_numeric(y, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    prior_x, prior_y = x.shift(1), y.shift(1)
    minimum = max(288, window // 2)
    mean_x = prior_x.rolling(window, min_periods=minimum).mean()
    mean_y = prior_y.rolling(window, min_periods=minimum).mean()
    covariance = prior_x.rolling(window, min_periods=minimum).cov(prior_y)
    variance = prior_x.rolling(window, min_periods=minimum).var(ddof=1).replace(0.0, np.nan)
    beta = (covariance / variance).clip(-5.0, 5.0)
    residual = (y - mean_y) - beta * (x - mean_x)
    return residual, beta


def build_features(market: pd.DataFrame) -> pd.DataFrame:
    available = market["spot_available"].to_numpy(bool)
    perp_close = pd.to_numeric(market["close"], errors="coerce")
    spot_close = pd.to_numeric(market["spot_close"], errors="coerce")
    premium = pd.to_numeric(market["premium_index_1m_close"], errors="coerce")
    basis = np.log(perp_close / spot_close).where(available)
    perp_return = np.log(perp_close).diff()
    spot_return = np.log(spot_close).diff()
    quote_volume = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    taker_quote = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    flow = (2.0 * taker_quote / quote_volume.replace(0.0, np.nan) - 1.0).where(available)
    perp_volume = pd.to_numeric(market["volume"], errors="coerce")
    spot_volume = pd.to_numeric(market["spot_volume"], errors="coerce")
    spot_share = (spot_volume / (spot_volume + perp_volume).replace(0.0, np.nan)).where(available)
    out: dict[str, pd.Series] = {
        "spot_perp_available": pd.Series(available, index=market.index, dtype=float),
        "direct_basis": basis,
        "premium_index_5m_close": premium.where(available),
    }
    for window in ROLLING_WINDOWS:
        spread, basis_beta = _rolling_residual(basis, premium, window)
        lead_residual, return_beta = _rolling_residual(perp_return, spot_return, window)
        out[f"spa_basis_beta_{window}"] = basis_beta
        out[f"spa_return_beta_{window}"] = return_beta
        out[f"spa_residual_{window}"] = spread
        out[f"spa_residual_z_{window}"] = _prior_z(spread, window)
        out[f"spa_lead_residual_z_{window}"] = _prior_z(lead_residual, window)
        out[f"spa_flow_z_{window}"] = _prior_z(flow, window)
        out[f"spa_spot_share_z_{window}"] = _prior_z(spot_share, window)
    frame = pd.DataFrame(out, index=market.index).replace([np.inf, -np.inf], np.nan)
    frame.loc[~available, :] = np.nan
    frame.loc[:, "spot_perp_available"] = available.astype(float)
    return frame.astype(np.float32)


def _policy_specs() -> list[dict[str, Any]]:
    return [
        {
            "window": window,
            "z_entry": z_entry,
            "contraction_delta": contraction,
            "mode": mode,
            "direction": direction,
            "phase": phase,
            "max_hold": hold,
        }
        for window in ROLLING_WINDOWS
        for z_entry in Z_ENTRIES
        for contraction in CONTRACTION_DELTAS
        for mode in MODES
        for direction in DIRECTIONS
        for phase in PHASES
        for hold in MAX_HOLDS
    ]


def _signals(features: pd.DataFrame, spec: dict[str, Any], *, flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    window = int(spec["window"])
    z = pd.to_numeric(features[f"spa_residual_z_{window}"], errors="coerce").to_numpy(float)
    lead = pd.to_numeric(features[f"spa_lead_residual_z_{window}"], errors="coerce").to_numpy(float)
    flow = pd.to_numeric(features[f"spa_flow_z_{window}"], errors="coerce").to_numpy(float)
    spot_share = pd.to_numeric(features[f"spa_spot_share_z_{window}"], errors="coerce").to_numpy(float)
    previous = np.r_[np.nan, z[:-1]]
    phase = str(spec["phase"])
    sign = np.sign(previous) if phase == "contraction" else np.sign(z)
    finite = np.isfinite(z) & np.isfinite(previous)
    if phase == "contraction":
        active = (
            finite
            & (np.abs(previous) >= float(spec["z_entry"]))
            & (np.sign(z) == sign)
            & (np.abs(z) <= np.abs(previous) - float(spec["contraction_delta"]))
        )
    elif phase == "expansion_onset":
        active = (
            finite
            & (np.abs(previous) < float(spec["z_entry"]))
            & (np.abs(z) >= float(spec["z_entry"]) + float(spec["contraction_delta"]))
        )
    else:
        raise KeyError(phase)
    mode = str(spec["mode"])
    if mode in {"lead_residual", "spot_absorption"}:
        lead_aligned = sign * lead
        active &= np.isfinite(lead) & ((lead_aligned <= 0.0) if phase == "contraction" else (lead_aligned >= 0.0))
    if mode == "flow_fade":
        flow_aligned = sign * flow
        active &= np.isfinite(flow) & ((flow_aligned <= 0.0) if phase == "contraction" else (flow_aligned >= 0.0))
    if mode == "spot_absorption":
        active &= np.isfinite(spot_share) & ((spot_share >= 0.0) if phase == "contraction" else (spot_share <= 0.0))
    side = np.where(sign < 0.0, 1, -1).astype(np.int8)
    if str(spec["direction"]) == "continuation":
        side = -side
    if flip:
        side = -side
    side[~active] = 0
    return active, side


def _make_event(
    market: pd.DataFrame,
    residual_z: np.ndarray,
    signal_pos: int,
    side: int,
    residual_sign: int,
    *,
    max_hold: int,
    dynamic_exit: bool,
    exit_abs_z: float,
    cost_rate: float,
    leverage: float,
    name: str,
) -> dict[str, Any] | None:
    entry = int(signal_pos) + 1
    max_exit = entry + int(max_hold)
    if entry >= len(market) - 1 or max_exit >= len(market):
        return None
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    ret = np.zeros(int(max_hold) + 1, dtype=np.float64)
    adv = np.zeros(int(max_hold) + 1, dtype=np.float64)
    cost = float(cost_rate) * float(leverage)
    ret[0] -= cost
    adv[0] -= cost
    exit_pos = max_exit
    initial_sign = int(residual_sign)
    for j in range(entry, max_exit):
        offset = j - entry
        open_j = float(opens[j])
        if not np.isfinite(open_j) or open_j <= 0.0:
            continue
        if side > 0:
            adverse = (float(lows[j]) - open_j) / open_j
            close_return = (float(opens[j + 1]) - open_j) / open_j
        else:
            adverse = (open_j - float(highs[j])) / open_j
            close_return = (open_j - float(opens[j + 1])) / open_j
        adv[offset] += min(0.0, float(leverage) * adverse)
        ret[offset] += float(leverage) * close_return
        current_z = float(residual_z[j])
        if dynamic_exit and np.isfinite(current_z) and (abs(current_z) <= float(exit_abs_z) or np.sign(current_z) != initial_sign):
            exit_pos = j + 1
            break
    used = exit_pos - entry + 1
    ret[used - 1] -= cost
    ret = ret[:used]
    adv = adv[:used]
    realized = float(np.prod(np.maximum(0.0, 1.0 + ret)) - 1.0)
    return {
        "expert": name,
        "side": "long" if side > 0 else "short",
        "signal_pos": int(signal_pos),
        "entry_pos": int(entry),
        "exit_pos": int(exit_pos),
        "ret": ret,
        "adv": adv,
        "realized_return": realized,
        "max_adverse": float(max(0.0, -np.nanmin(adv))),
    }


def _build_events(
    market: pd.DataFrame,
    features: pd.DataFrame,
    spec: dict[str, Any],
    cfg: SpotPerpConfig,
    *,
    cost_rate: float,
    flip: bool = False,
) -> list[dict[str, Any]]:
    active, sides = _signals(features, spec, flip=flip)
    window = int(spec["window"])
    residual_z = pd.to_numeric(features[f"spa_residual_z_{window}"], errors="coerce").to_numpy(float)
    positions = np.flatnonzero(active)
    events: list[dict[str, Any]] = []
    next_allowed = 0
    name = "spot_perp_absorption_flip" if flip else "spot_perp_absorption"
    for pos in positions:
        if int(pos) < next_allowed:
            continue
        event = _make_event(
            market,
            residual_z,
            int(pos),
            int(sides[pos]),
            int(np.sign(residual_z[pos])),
            max_hold=int(spec["max_hold"]),
            dynamic_exit=str(spec["phase"]) == "contraction",
            exit_abs_z=float(cfg.exit_abs_z),
            cost_rate=cost_rate,
            leverage=float(cfg.leverage),
            name=name,
        )
        if event is None:
            continue
        events.append(event)
        next_allowed = int(event["exit_pos"])
    return events


def _stats(events: list[dict[str, Any]], dates: pd.Series, names: Iterable[str]) -> dict[str, Any]:
    return {name: _metric(events, dates, *WINDOWS[name]) for name in names}


def _path_hash(events: list[dict[str, Any]], dates: pd.Series, name: str) -> str:
    mask = _window_mask(dates, name)
    positions = np.flatnonzero(mask)
    first, last = int(positions[0]), int(positions[-1]) + 1
    rows = [
        (event["side"], int(event["signal_pos"]), int(event["exit_pos"]))
        for event in events
        if int(event["signal_pos"]) >= first and int(event["exit_pos"]) < last
    ]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _selection_score(stats: dict[str, Any]) -> float:
    fit = stats["fit"]
    full = stats["select_2023"]
    h1, h2 = stats["select_2023_h1"], stats["select_2023_h2"]
    if (
        fit["return_pct"] <= 0.0 or fit["trades"] < 50
        or full["return_pct"] <= 0.0 or full["ratio"] < 1.0 or full["trades"] < 20
        or min(h1["return_pct"], h2["return_pct"]) <= 0.0
        or min(h1["trades"], h2["trades"]) < 8
    ):
        return -1e12
    return float(min(full["ratio"], h1["ratio"], h2["ratio"]) + 0.1 * fit["ratio"] + 0.01 * min(full["trades"], 100))


def _select_top(rows: list[dict[str, Any]], *, top_n: int, top_per_mode: int) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: (-row["selection_score"], json.dumps(row["spec"], sort_keys=True)))
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in rows:
        mode = str(row["spec"]["mode"])
        if counts.get(mode, 0) >= int(top_per_mode):
            continue
        selected.append(row)
        counts[mode] = counts.get(mode, 0) + 1
        if len(selected) >= int(top_n):
            break
    return selected


def _select_manifest(cfg: SpotPerpConfig) -> dict[str, Any]:
    market, prefix_hashes = _load_market_spot(cfg, cutoff=SELECTION_END)
    dates = pd.to_datetime(market["date"])
    features = build_features(market)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in _policy_specs():
        events = _build_events(
            market,
            features,
            spec,
            cfg,
            cost_rate=float(cfg.fee_rate + cfg.slippage_rate),
        )
        stats = _stats(events, dates, ("fit", "select_2023", "select_2023_h1", "select_2023_h2"))
        score = _selection_score(stats)
        if score <= -1e11:
            continue
        path_hash = _path_hash(events, dates, "select_2023")
        if path_hash in seen:
            continue
        seen.add(path_hash)
        rows.append({"spec": spec, "selection_score": score, "selection_stats": stats, "selection_path_hash": path_hash})
    selected = _select_top(rows, top_n=cfg.top_n, top_per_mode=cfg.top_per_mode)
    core = {
        "protocol": {
            "hypothesis": "trade fixed contra/continuation directions after direct perp-spot basis residual contraction or first expansion onset",
            "orthogonalization": "direct perp-spot log basis residualized against completed Binance premium-index close with trailing shifted OLS",
            "normalization": "current values compared with rolling mean/std fitted strictly through t-1",
            "selection": {name: WINDOWS[name] for name in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")},
            "all_market_and_spot_rows_physically_excluded_before_manifest": True,
            "later_metrics_included": False,
            "search_cap": f"{len(_policy_specs())} predeclared transition policies",
            "preflight_revision": "the initial 96-policy contra-only and then 192-policy symmetric contraction passes produced zero eligible paths and were uniformly negative; expansion-onset with fixed hold was added before any 2024+ replay",
            "entry": "next 5m open after completed futures and spot bar",
            "exit": f"contraction: next open after residual |z| <= {cfg.exit_abs_z} or sign cross; expansion onset: fixed max hold",
            "cost": "6bp/side base, 10bp/side stress, 0.5x",
            "mdd": "strict entry cost plus intrabar adverse OHLC and realized high-water",
            "marginal_rule": "must improve deterministic six-sleeve union on both combined absolute return and CAGR/MDD",
            "status_ceiling": "shadow research because later calendar windows are not fresh to the programme",
        },
        "source_prefix_hashes": prefix_hashes,
        "feature_hash": _feature_hash(features, dates),
        "search_space": {"raw_specs": len(_policy_specs()), "eligible_unique_paths": len(rows), "top_n": int(cfg.top_n)},
        "selected": selected,
    }
    manifest = {"as_of": datetime.now(timezone.utc).isoformat(), "sha256": _manifest_core_hash(core), **core}
    path = Path(cfg.manifest_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    return manifest


def _expert_config(cfg: SpotPerpConfig) -> OnlineExpertConfig:
    return OnlineExpertConfig(
        input_csv=cfg.input_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        output="",
        manifest_output="",
        docs_output="",
        exclude_from=cfg.exclude_from,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        stress_fee_rate=cfg.stress_fee_rate,
    )


def _merge_with_priority(base: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tagged = [(event, 0) for event in base] + [(event, 1) for event in candidate]
    accepted: list[dict[str, Any]] = []
    next_allowed = 0
    for event, priority in sorted(tagged, key=lambda row: (row[0]["signal_pos"], row[1], row[0]["expert"])):
        del priority
        if int(event["signal_pos"]) < next_allowed:
            continue
        accepted.append(event)
        next_allowed = int(event["exit_pos"])
    return accepted


def _jaccard(candidate: list[dict[str, Any]], base: list[dict[str, Any]], dates: pd.Series) -> float:
    mask = _window_mask(dates, "oos_2024_2026")
    positions = np.flatnonzero(mask)
    first, last = int(positions[0]), int(positions[-1]) + 1
    a = {int(event["signal_pos"]) for event in candidate if first <= int(event["signal_pos"]) < last}
    b = {int(event["signal_pos"]) for event in base if first <= int(event["signal_pos"]) < last}
    union = a | b
    return float(len(a & b) / len(union)) if union else 0.0


def _replay(cfg: SpotPerpConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    _validate_manifest(manifest)
    prefix_market, prefix_hashes = _load_market_spot(cfg, cutoff=SELECTION_END)
    prefix_dates = pd.to_datetime(prefix_market["date"])
    prefix_features = build_features(prefix_market)
    if prefix_hashes != manifest["source_prefix_hashes"]:
        raise RuntimeError("pre-2024 spot/perp source prefixes changed")
    if _feature_hash(prefix_features, prefix_dates) != manifest["feature_hash"]:
        raise RuntimeError("pre-2024 spot/perp feature reconstruction drift")
    market, _ = _load_market_spot(cfg, cutoff=cfg.exclude_from)
    dates = pd.to_datetime(market["date"])
    features = build_features(market)
    prefix = dates < pd.Timestamp(SELECTION_END)
    if _feature_hash(features.loc[prefix].reset_index(drop=True), dates.loc[prefix].reset_index(drop=True)) != manifest["feature_hash"]:
        raise RuntimeError("full replay spot/perp feature prefix drift")

    expert_cfg = _expert_config(cfg)
    expert_market, expert_features, _ = _load_expert_bundle(expert_cfg, cutoff=cfg.exclude_from)
    if not pd.to_datetime(expert_market["date"]).equals(dates):
        raise RuntimeError("spot/perp and expert baseline market grids differ")
    base_events = _build_expert_events(
        expert_market,
        expert_features,
        expert_cfg,
        cost_rate=float(cfg.fee_rate + cfg.slippage_rate),
    )
    base_by_expert = {name: [event for event in base_events if event["expert"] == name] for name in ALPHAS}
    base_union = _global_nonoverlap(base_events)
    eval_windows = ("test_2024", "eval_2025", "holdout_2026", "oos_2024_2026")
    baseline_stats = _stats(base_union, dates, eval_windows)

    stress_cfg = replace(cfg, fee_rate=cfg.stress_fee_rate, slippage_rate=0.0001)
    stress_expert_cfg = _expert_config(stress_cfg)
    stress_base_events = _build_expert_events(
        expert_market,
        expert_features,
        stress_expert_cfg,
        cost_rate=float(stress_cfg.fee_rate + stress_cfg.slippage_rate),
    )
    stress_base_union = _global_nonoverlap(stress_base_events)
    rows = []
    for rank, frozen in enumerate(manifest["selected"], start=1):
        events = _build_events(
            market,
            features,
            frozen["spec"],
            cfg,
            cost_rate=float(cfg.fee_rate + cfg.slippage_rate),
        )
        selection_stats = _stats(events, dates, ("fit", "select_2023", "select_2023_h1", "select_2023_h2"))
        if selection_stats != frozen["selection_stats"] or _path_hash(events, dates, "select_2023") != frozen["selection_path_hash"]:
            raise RuntimeError(f"pre-2024 policy replay drift at rank {rank}")
        stats = _stats(events, dates, WINDOWS)
        flipped = _build_events(
            market,
            features,
            frozen["spec"],
            cfg,
            cost_rate=float(cfg.fee_rate + cfg.slippage_rate),
            flip=True,
        )
        stress_events = _build_events(
            market,
            features,
            frozen["spec"],
            stress_cfg,
            cost_rate=float(stress_cfg.fee_rate + stress_cfg.slippage_rate),
        )
        combined = _merge_with_priority(base_union, events)
        stress_combined = _merge_with_priority(stress_base_union, stress_events)
        combined_stats = _stats(combined, dates, eval_windows)
        stress_stats = _stats(stress_events, dates, eval_windows)
        stress_combined_stats = _stats(stress_combined, dates, eval_windows)
        flipped_stats = _stats(flipped, dates, eval_windows)
        quarterly = {name: _metric(events, dates, start, end) for name, (start, end) in QUARTER_WINDOWS.items()}
        quarter_summary = {
            "positive_return_quarters": sum(row["return_pct"] > 0.0 for row in quarterly.values()),
            "negative_return_quarters": sum(row["return_pct"] < 0.0 for row in quarterly.values()),
            "flat_quarters": sum(row["trades"] == 0 for row in quarterly.values()),
            "total_quarters": len(quarterly),
        }
        jaccards = {name: _jaccard(events, source, dates) for name, source in base_by_expert.items()}
        test, evaluation = stats["test_2024"], stats["eval_2025"]
        holdout, combined_candidate = stats["holdout_2026"], stats["oos_2024_2026"]
        standalone = (
            test["return_pct"] > 0.0 and test["ratio"] >= 3.0 and test["trades"] >= 20
            and evaluation["return_pct"] > 0.0 and evaluation["ratio"] >= 3.0 and evaluation["trades"] >= 20
            and holdout["return_pct"] > 0.0 and holdout["trades"] >= 12
            and combined_candidate["return_pct"] > 0.0
        )
        base_all = baseline_stats["oos_2024_2026"]
        merged_all = combined_stats["oos_2024_2026"]
        marginal = merged_all["return_pct"] > base_all["return_pct"] and merged_all["ratio"] > base_all["ratio"]
        stress_ok = min(stress_stats[name]["ratio"] for name in ("test_2024", "eval_2025", "holdout_2026")) >= 2.5
        bonferroni = min(1.0, combined_candidate["p_value_mean_return_approx"] * max(1, len(manifest["selected"])))
        qualifies = standalone and marginal and stress_ok and max(jaccards.values(), default=0.0) <= 0.25 and bonferroni <= 0.05
        rows.append(
            {
                "manifest_rank": rank,
                **frozen,
                "stats": stats,
                "direction_flipped": flipped_stats,
                "stress_10bp_each_side": stress_stats,
                "combined_with_six_sleeve_union": combined_stats,
                "stress_combined_with_six_sleeve_union": stress_combined_stats,
                "quarterly_stats": quarterly,
                "quarterly_summary": quarter_summary,
                "signal_jaccard_vs_fixed_experts": jaccards,
                "top_n_bonferroni_p_value": float(bonferroni),
                "passes_standalone_gate": bool(standalone),
                "adds_value_vs_six_sleeve_union": bool(marginal),
                "passes_cost_stress": bool(stress_ok),
                "passes_alpha_pool": bool(qualifies),
                "passes_live_grade": False,
            }
        )
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "manifest": cfg.manifest_output,
        "manifest_sha256": manifest["sha256"],
        "protocol": manifest["protocol"],
        "protocol_scope_correction": "Frozen manifest wording says absorption, but documented preflight added expansion_onset before OOS; contraction had zero eligible paths and every selected replay row is expansion_onset.",
        "source_file_hashes_after_manifest_freeze": _source_hashes(cfg),
        "feature_correlation_audit": {
            "direct_basis_vs_premium_index_fit_pearson": float(features.loc[_window_mask(dates, "fit"), "direct_basis"].corr(features.loc[_window_mask(dates, "fit"), "premium_index_5m_close"])),
            "residual_z2016_vs_premium_index_fit_spearman": float(features.loc[_window_mask(dates, "fit"), "spa_residual_z_2016"].corr(features.loc[_window_mask(dates, "fit"), "premium_index_5m_close"], method="spearman")),
        },
        "selected_phase_scope": sorted({str(row["spec"]["phase"]) for row in rows}),
        "six_sleeve_union_baseline": baseline_stats,
        "selected": rows,
        "alpha_pool_qualifiers": [row for row in rows if row["passes_alpha_pool"]],
        "live_grade": [],
    }


def _fmt(row: dict[str, Any]) -> str:
    return f"{row['return_pct']:.2f}/{row['cagr_pct']:.2f}/{row['strict_mdd_pct']:.2f}/{row['ratio']:.2f}/{row['trades']}"


def _write_doc(cfg: SpotPerpConfig, report: dict[str, Any]) -> None:
    lines = [
        "# Spot–perpetual residual transition alpha search (2026-07-13)",
        "",
        "Metric format: `absolute return / CAGR / strict MDD / CAGR-MDD / trades`.",
        "",
        "## Frozen Top-10 replay",
        "",
        "| rank | policy | 2024 | 2025 | 2026 | combined | +union combined | alpha |",
        "|---:|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["selected"]:
        stats = row["stats"]
        merged = row["combined_with_six_sleeve_union"]["oos_2024_2026"]
        lines.append(
            f"| {row['manifest_rank']} | `{row['spec']}` | {_fmt(stats['test_2024'])} | {_fmt(stats['eval_2025'])} | "
            f"{_fmt(stats['holdout_2026'])} | {_fmt(stats['oos_2024_2026'])} | {_fmt(merged)} | "
            f"{'yes' if row['passes_alpha_pool'] else 'no'} |"
        )
    baseline = report["six_sleeve_union_baseline"]
    lines += [
        "",
        "## Required comparator",
        "",
        f"Deterministic six-sleeve union combined: `{_fmt(baseline['oos_2024_2026'])}`.",
        "",
        "## Interpretation",
        "",
        f"- Alpha-pool qualifiers: {len(report['alpha_pool_qualifiers'])}; live-grade: 0 by protocol.",
        f"- Final frozen replay scope: `{report['selected_phase_scope']}`. Contraction policies produced zero eligible pre-2024 paths; all replayed Top candidates are expansion-onset policies.",
        "- Direct perp-spot basis is explicitly residualized against the completed premium index before event construction; current values never enter rolling fit statistics.",
        "- A standalone pass is insufficient: the seventh stream must improve the existing union on both absolute return and CAGR/MDD.",
        "- The rejection therefore applies to fixed contraction mappings at preflight and fixed expansion-onset mappings in OOS; it does not prove that every nonlinear use of the continuous residual is useless.",
        "- Any future pass remains shadow-only because 2024-2026 are not fresh calendar windows for the broader programme.",
        "",
        "## Reproduction",
        "",
        "```bash",
        f"python -m training.search_spot_perp_absorption_alpha --input-csv {cfg.input_csv} --spot-csv {cfg.spot_csv} --funding-csv {cfg.funding_csv} --premium-csv {cfg.premium_csv} --manifest-output {cfg.manifest_output} --output {cfg.output} --docs-output {cfg.docs_output}",
        "```",
    ]
    path = Path(cfg.docs_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run(cfg: SpotPerpConfig) -> dict[str, Any]:
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


def parse_args() -> SpotPerpConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--spot-csv", required=True)
    parser.add_argument("--funding-csv", required=True)
    parser.add_argument("--premium-csv", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--docs-output", required=True)
    parser.add_argument("--exclude-from", default=SpotPerpConfig.exclude_from)
    parser.add_argument("--refresh-manifest", action="store_true")
    return SpotPerpConfig(**vars(parser.parse_args()))


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"manifest": report["manifest"], "qualifiers": len(report["alpha_pool_qualifiers"]), "top": report["selected"][:3]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
