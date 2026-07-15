"""Freeze a weak-state exit router for the live-parity pullback squeeze.

The base entry remains the corrected confirmed pullback-squeeze.  Three causal
weak representations do not vote on direction.  They assign each long entry
to an economic state and route only its exit:

* support: keep the structural 48-hour horizon;
* neutral: shorten the inventory clock to 24 hours; and
* adverse absorption: retain the 48-hour cap but lock a 4% take.

Wasserstein flow-response strain and frozen causal-cone rupture are computed
on the completed ``:55`` bar and shifted to the ``:00`` decision boundary.
The dual intrinsic clock supplies a source-specific continuation/absorption
state.  Every threshold is fitted only on pre-2023 base events.  OOS can be
opened only from a committed pre-2024 manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.audit_confirmed_pullback_squeeze_live_parity import (
    AuditConfig,
    PRE2024_WINDOWS,
    _activation_hash,
    _execution_config,
    _fit_active,
    _load_bundle,
    decision_mask,
    live_decision_features,
    selection_passes,
)
from training.search_dual_intrinsic_clock_alpha import (
    build_clock_features,
    build_paths,
    directional_change_events,
    impact_state,
)
from training.search_frozen_causal_cone_rupture_alpha import build_cone_state
from training.search_funding_premium_external_state_gate_alpha import _frame_hash
from training.search_inventory_purge_reclaim_alpha import (
    ExecutionEngine,
    Trade,
    _schedule_hash,
    equity_stats,
)
from training.search_wasserstein_flow_response_strain_alpha import (
    build_response_inputs,
    build_transport_state,
)


SELECTION_END = "2024-01-01"
FIT_START = "2020-07-01"
FIT_END = "2023-01-01"
NO_BARRIER_BPS = 1_000_000
FUTURE_WINDOWS: dict[str, tuple[str, str]] = {
    "test_2024": ("2024-01-01", "2025-01-01"),
    "eval_2025": ("2025-01-01", "2026-01-01"),
    "holdout_2026": ("2026-01-01", "2026-06-02"),
    "oos_2024_2026": ("2024-01-01", "2026-06-02"),
}
OOS_GATE_WINDOWS = ("test_2024", "eval_2025", "oos_2024_2026")

WIDTH_GRID = (1.0, 1.5)
CLOCK_GRID = (144, 288)
DOMINANCE_GRID = (1.5, 2.0)
QUANTILE_GRID = (0.50, 0.70, 0.80)
STATE_MODE_GRID = ("consensus", "one_plus_nonoppose", "two_of_three")
PRIORITY_GRID = ("adverse", "support")
NEUTRAL_ACTION_GRID = ("time24", "time36", "time48")
ADVERSE_ACTION_GRID = ("skip", "time12", "tp4_48")

ACTIONS: dict[str, tuple[int, int, int] | None] = {
    "skip": None,
    "time12": (144, NO_BARRIER_BPS, NO_BARRIER_BPS),
    "time24": (288, NO_BARRIER_BPS, NO_BARRIER_BPS),
    "time36": (432, NO_BARRIER_BPS, NO_BARRIER_BPS),
    "time48": (576, NO_BARRIER_BPS, NO_BARRIER_BPS),
    "tp4_48": (576, 400, NO_BARRIER_BPS),
}

FROZEN_SPEC: dict[str, Any] = {
    "width": 1.5,
    "clock": 144,
    "dominance": 2.0,
    "quantile": 0.70,
    "mode": "one_plus_nonoppose",
    "priority": "adverse",
    "support": "time48",
    "neutral": "time24",
    "adverse": "tp4_48",
}

FROZEN_CONFIG_KEYS = (
    "input_csv",
    "funding_csv",
    "premium_csv",
    "exclude_from",
    "window_size",
    "leverage",
    "fee_rate",
    "slippage_rate",
    "funding_tolerance",
    "live_premium_tolerance",
)


@dataclass(frozen=True)
class Config(AuditConfig):
    output: str = "results/pullback_weak_state_exit_router_selection_2026-07-15.json"
    manifest_output: str = "results/pullback_weak_state_exit_router_manifest_2026-07-15.json"
    docs_output: str = "docs/pullback-weak-state-exit-router-selection-2026-07-15.md"
    leverage: float = 0.50
    open_oos: bool = False


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolved_path(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def _frozen_execution_config(cfg: Config) -> dict[str, Any]:
    values = asdict(cfg)
    frozen = {key: values[key] for key in FROZEN_CONFIG_KEYS}
    for key in ("input_csv", "funding_csv", "premium_csv"):
        frozen[key] = _resolved_path(str(frozen[key]))
    return frozen


def _spec_hash() -> str:
    encoded = json.dumps(FROZEN_SPEC, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _implementation_hash() -> str:
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for helper in (
        _load_bundle,
        _fit_active,
        live_decision_features,
        build_response_inputs,
        build_transport_state,
        build_cone_state,
        build_paths,
        directional_change_events,
        build_clock_features,
        impact_state,
        ExecutionEngine.trade_at,
        equity_stats,
    ):
        digest.update(inspect.getsource(helper).encode())
    return digest.hexdigest()


def shift_completed_to_boundary(values: np.ndarray | pd.Series, fill: float | int) -> np.ndarray:
    """Move a completed ``:55`` observation to the next ``:00`` source row."""

    array = np.asarray(values)
    dtype = np.result_type(array.dtype, np.asarray(fill).dtype)
    shifted = np.full(array.shape, fill, dtype=dtype)
    shifted[1:] = array[:-1]
    return shifted


def _fit_abs_threshold(
    values: np.ndarray,
    dates: pd.Series,
    base_active: np.ndarray,
    quantile: float,
) -> float:
    fit = (
        np.asarray(base_active, dtype=bool)
        & (dates >= pd.Timestamp(FIT_START)).to_numpy(bool)
        & (dates < pd.Timestamp(FIT_END)).to_numpy(bool)
    )
    reference = np.abs(np.asarray(values, dtype=float)[fit])
    reference = reference[np.isfinite(reference)]
    if len(reference) < 40:
        raise ValueError(f"insufficient base-event weak state: {len(reference)}")
    return float(np.quantile(reference, quantile))


def _dual_state(
    paths: dict[str, np.ndarray],
    price_events: np.ndarray,
    flow_events: np.ndarray,
    dates: pd.Series,
    *,
    clock: int,
    dominance: float,
) -> tuple[np.ndarray, np.ndarray]:
    features = build_clock_features(
        paths,
        price_events,
        flow_events,
        dates,
        clock_window=int(clock),
    )
    state = impact_state(features, float(dominance))
    price_move = features["price_displacement_z"].to_numpy(float)
    flow_move = features["flow_displacement_z"].to_numpy(float)
    side = np.nan_to_num(
        np.where(state == 1, -np.sign(flow_move), np.where(state == 2, np.sign(price_move), 0.0))
    ).astype(np.int8)
    return (
        shift_completed_to_boundary(state, 0).astype(np.int8),
        shift_completed_to_boundary(side, 0).astype(np.int8),
    )


def build_weak_state_bank(
    market: pd.DataFrame,
    dates: pd.Series,
    base_active: np.ndarray,
) -> dict[str, Any]:
    response_inputs, flow_tail = build_response_inputs(market, dates)
    transport = build_transport_state(response_inputs, lookback=288, flow_tail=flow_tail)
    wscore = shift_completed_to_boundary(transport["score"].to_numpy(float), np.nan)

    cone = build_cone_state(market, dates, horizon=2016)
    cone_score = shift_completed_to_boundary(cone["score"].to_numpy(float), np.nan)
    cone_side = np.nan_to_num(
        shift_completed_to_boundary(cone["side"].to_numpy(float), np.nan)
    ).astype(np.int8)

    thresholds = {
        str(quantile): {
            "wasserstein_abs": _fit_abs_threshold(wscore, dates, base_active, quantile),
            "cone_abs": _fit_abs_threshold(cone_score, dates, base_active, quantile),
        }
        for quantile in QUANTILE_GRID
    }

    paths = build_paths(market)
    price_events = {
        width: directional_change_events(
            paths["log_price"], paths["price_scale"], width=float(width)
        )
        for width in WIDTH_GRID
    }
    flow_events = {
        width: directional_change_events(
            paths["flow_path"], paths["flow_scale"], width=float(width)
        )
        for width in WIDTH_GRID
    }
    dual = {
        (width, clock, dominance): _dual_state(
            paths,
            price_events[width],
            flow_events[width],
            dates,
            clock=clock,
            dominance=dominance,
        )
        for width, clock, dominance in itertools.product(
            WIDTH_GRID, CLOCK_GRID, DOMINANCE_GRID
        )
    }
    return {
        "wasserstein_score": np.asarray(wscore, dtype=float),
        "cone_score": np.asarray(cone_score, dtype=float),
        "cone_side": np.asarray(cone_side, dtype=np.int8),
        "dual": dual,
        "thresholds": thresholds,
        "flow_tail": float(flow_tail),
    }


def state_labels(
    wasserstein_score: np.ndarray,
    cone_score: np.ndarray,
    cone_side: np.ndarray,
    dual_state: np.ndarray,
    dual_side: np.ndarray,
    *,
    wasserstein_threshold: float,
    cone_threshold: float,
    mode: str,
    priority: str,
) -> np.ndarray:
    """Return ``1`` support, ``0`` neutral, or ``-1`` adverse absorption."""

    wscore = np.asarray(wasserstein_score, dtype=float)
    cscore = np.asarray(cone_score, dtype=float)
    cside = np.asarray(cone_side, dtype=np.int8)
    dstate = np.asarray(dual_state, dtype=np.int8)
    dside = np.asarray(dual_side, dtype=np.int8)
    if not (wscore.shape == cscore.shape == cside.shape == dstate.shape == dside.shape):
        raise ValueError("weak-state arrays must align")
    if priority not in PRIORITY_GRID:
        raise KeyError(priority)

    wside = np.nan_to_num(np.sign(wscore)).astype(np.int8)
    wstrong = np.isfinite(wscore) & (np.abs(wscore) >= float(wasserstein_threshold))
    cstrong = np.isfinite(cscore) & (cscore >= float(cone_threshold))
    wpos, wneg = wstrong & (wside > 0), wstrong & (wside < 0)
    cpos, cneg = cstrong & (cside > 0), cstrong & (cside < 0)
    dual_support = (dstate == 2) & (dside > 0)
    dual_adverse = (dstate == 1) & (dside < 0)

    if mode == "consensus":
        support = (wpos & cpos) | dual_support
        adverse = (wneg & cneg) | dual_adverse
    elif mode == "one_plus_nonoppose":
        support = (wpos & (cside >= 0)) | (cpos & (wside >= 0)) | dual_support
        adverse = (wneg & (cside <= 0)) | (cneg & (wside <= 0)) | dual_adverse
    elif mode == "two_of_three":
        positive = wpos.astype(np.int8) + cpos.astype(np.int8) + dual_support.astype(np.int8)
        negative = wneg.astype(np.int8) + cneg.astype(np.int8) + dual_adverse.astype(np.int8)
        support, adverse = positive >= 2, negative >= 2
    else:
        raise KeyError(mode)

    labels = np.zeros(len(wscore), dtype=np.int8)
    if priority == "adverse":
        labels[support] = 1
        labels[adverse] = -1
    else:
        labels[adverse] = -1
        labels[support] = 1
    return labels


def candidate_specs() -> list[dict[str, Any]]:
    return [
        {
            "width": float(width),
            "clock": int(clock),
            "dominance": float(dominance),
            "quantile": float(quantile),
            "mode": mode,
            "priority": priority,
            "support": "time48",
            "neutral": neutral,
            "adverse": adverse,
        }
        for width, clock, dominance, quantile, mode, priority, neutral, adverse in itertools.product(
            WIDTH_GRID,
            CLOCK_GRID,
            DOMINANCE_GRID,
            QUANTILE_GRID,
            STATE_MODE_GRID,
            PRIORITY_GRID,
            NEUTRAL_ACTION_GRID,
            ADVERSE_ACTION_GRID,
        )
    ]


def labels_for_spec(bank: dict[str, Any], spec: dict[str, Any]) -> np.ndarray:
    threshold = bank["thresholds"][str(float(spec["quantile"]))]
    dual_state, dual_side = bank["dual"][(
        float(spec["width"]),
        int(spec["clock"]),
        float(spec["dominance"]),
    )]
    return state_labels(
        bank["wasserstein_score"],
        bank["cone_score"],
        bank["cone_side"],
        dual_state,
        dual_side,
        wasserstein_threshold=threshold["wasserstein_abs"],
        cone_threshold=threshold["cone_abs"],
        mode=str(spec["mode"]),
        priority=str(spec["priority"]),
    )


def schedule_window(
    engine: ExecutionEngine,
    active: np.ndarray,
    labels: np.ndarray,
    spec: dict[str, Any],
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    trade_cache: dict[tuple[int, str], Trade | None] | None = None,
) -> list[Trade]:
    dates = pd.to_datetime(engine.market["date"])
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    cache = {} if trade_cache is None else trade_cache
    trades: list[Trade] = []
    next_allowed = 0
    for signal in np.flatnonzero(np.asarray(active, dtype=bool) & period):
        signal = int(signal)
        if signal < next_allowed:
            continue
        label = int(labels[signal])
        action = str(spec["support"] if label > 0 else spec["adverse"] if label < 0 else spec["neutral"])
        action_spec = ACTIONS[action]
        if action_spec is None:
            continue
        key = (signal, action)
        if key not in cache:
            cache[key] = engine.trade_at(signal, 1, *action_spec)
        trade = cache[key]
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


def schedules_and_stats(
    engine: ExecutionEngine,
    cfg: Config,
    active: np.ndarray,
    labels: np.ndarray,
    spec: dict[str, Any],
    *,
    windows: dict[str, tuple[str, str]],
    trade_cache: dict[tuple[int, str], Trade | None] | None = None,
) -> tuple[dict[str, list[Trade]], dict[str, dict[str, Any]]]:
    cache = {} if trade_cache is None else trade_cache
    schedules = {
        name: schedule_window(
            engine,
            active,
            labels,
            spec,
            start=start,
            end=end,
            trade_cache=cache,
        )
        for name, (start, end) in windows.items()
    }
    execution_cfg = _execution_config(cfg, cfg.leverage)
    stats = {
        name: _slim(equity_stats(schedules[name], start=start, end=end, cfg=execution_cfg))
        for name, (start, end) in windows.items()
    }
    return schedules, stats


def oos_passes(stats: dict[str, dict[str, Any]]) -> bool:
    return bool(
        all(stats[name]["absolute_return_pct"] > 0.0 for name in OOS_GATE_WINDOWS)
        and all(stats[name]["cagr_to_strict_mdd"] >= 3.0 for name in OOS_GATE_WINDOWS)
        and all(stats[name]["strict_mdd_pct"] <= 15.0 for name in OOS_GATE_WINDOWS)
        and stats["test_2024"]["trades"] >= 12
        and stats["eval_2025"]["trades"] >= 12
        and stats["oos_2024_2026"]["trades"] >= 30
    )


def _selection_rank(stats: dict[str, dict[str, Any]], spec: dict[str, Any]) -> list[Any]:
    support = (
        stats["train"]["trades"] >= 60
        and stats["select_2023"]["trades"] >= 12
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 5
    )
    stable = sum(
        stats[name]["absolute_return_pct"] > 0.0
        for name in ("train_2020h2", "train_2021", "train_2022", "select_2023_h1", "select_2023_h2")
    )
    ratios = [
        stats[name]["cagr_to_strict_mdd"]
        for name in ("train", "select_2023", "pre_2024")
    ]
    return [
        bool(selection_passes(stats)),
        bool(support),
        int(stable),
        float(min(ratios)),
        float(np.median(ratios)),
        int(stats["pre_2024"]["trades"]),
        float(spec["dominance"]),
        float(spec["width"]),
        -int(spec["clock"]),
    ]


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _weak_frame_hash(bank: dict[str, Any], labels: np.ndarray, spec: dict[str, Any]) -> str:
    dual_state, dual_side = bank["dual"][(
        float(spec["width"]), int(spec["clock"]), float(spec["dominance"])
    )]
    frame = pd.DataFrame(
        {
            "wasserstein_score": bank["wasserstein_score"],
            "cone_score": bank["cone_score"],
            "cone_side": bank["cone_side"],
            "dual_state": dual_state,
            "dual_side": dual_side,
            "label": labels,
        }
    )
    return _frame_hash(frame)


def _grid_hash(rows: Iterable[dict[str, Any]]) -> str:
    encoded = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


FROZEN_MANIFEST_PAYLOAD_KEYS = (
    "phase",
    "oos_opened",
    "selection_end",
    "spec",
    "spec_hash",
    "implementation_hash",
    "frozen_execution_config",
    "source_prefix_hashes",
    "base_activation_hash",
    "weak_feature_hash",
    "state_label_hash",
    "weak_thresholds",
    "flow_tail",
    "selection_passed",
    "selection_stats",
    "selection_schedule_hashes",
    "selection_grid_hash",
    "selection_grid_cells",
)
FROZEN_MANIFEST_KEYS = frozenset((*FROZEN_MANIFEST_PAYLOAD_KEYS, "freeze_hash"))


def _freeze_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in FROZEN_MANIFEST_PAYLOAD_KEYS}


def _freeze_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _freeze_payload(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_manifest(cfg: Config, manifest: dict[str, Any]) -> None:
    actual_keys = frozenset(manifest)
    if actual_keys != FROZEN_MANIFEST_KEYS:
        missing = sorted(FROZEN_MANIFEST_KEYS.difference(actual_keys))
        extra = sorted(actual_keys.difference(FROZEN_MANIFEST_KEYS))
        raise RuntimeError(f"manifest key contract changed: missing={missing}, extra={extra}")
    if manifest.get("oos_opened") is not False or manifest.get("phase") != "pre_oos_frozen":
        raise RuntimeError("manifest is not an immutable pre-OOS freeze")
    if manifest.get("selection_passed") is not True:
        raise RuntimeError("manifest failed selection")
    if manifest.get("spec") != FROZEN_SPEC or manifest.get("spec_hash") != _spec_hash():
        raise RuntimeError("manifest strategy specification changed")
    if manifest.get("implementation_hash") != _implementation_hash():
        raise RuntimeError("manifest implementation changed")
    if manifest.get("frozen_execution_config") != _frozen_execution_config(cfg):
        raise RuntimeError("runtime execution configuration changed")
    if manifest.get("freeze_hash") != _freeze_hash(manifest):
        raise RuntimeError("manifest freeze hash mismatch")


def _write_manifest_once(path: Path, payload: dict[str, Any], cfg: Config) -> dict[str, Any]:
    _validate_manifest(cfg, payload)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        _validate_manifest(cfg, existing)
        if existing["freeze_hash"] != payload["freeze_hash"]:
            raise RuntimeError("refusing to replace a different frozen manifest")
        return existing
    _atomic_write_json(path, payload)
    return payload


def _oos_sidecar_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(manifest_path.stem + ".oos-opened.json")


def _mark_oos_opened(manifest_path: Path, manifest: dict[str, Any], output: str) -> dict[str, Any]:
    sidecar_path = _oos_sidecar_path(manifest_path)
    expected = {
        "phase": "oos_opening",
        "freeze_hash": manifest["freeze_hash"],
        "oos_output": output,
    }
    if sidecar_path.exists():
        existing = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in expected.items()):
            raise RuntimeError("incompatible OOS-open sidecar")
        return existing
    payload = {**expected, "opened_at": datetime.now(timezone.utc).isoformat()}
    _atomic_write_json(sidecar_path, payload)
    return payload


def _load_context(cfg: Config, cutoff: str) -> dict[str, Any]:
    market, features, funding, source_hashes = _load_bundle(
        cfg,
        cutoff=cutoff,
        premium_tolerance=cfg.live_premium_tolerance,
    )
    dates = pd.to_datetime(market["date"])
    live_features = live_decision_features(features)
    decisions = decision_mask(dates, "live_hour_signal_bar", window_size=cfg.window_size)
    active, base_thresholds = _fit_active(live_features, dates, decisions)
    bank = build_weak_state_bank(market, dates, active)
    return {
        "market": market,
        "funding": funding,
        "dates": dates,
        "active": active,
        "base_thresholds": base_thresholds,
        "bank": bank,
        "source_hashes": source_hashes,
    }


def _selection_docs(payload: dict[str, Any]) -> str:
    stats = payload["selection_stats"]
    lines = [
        "# Pullback weak-state exit router — pre-OOS selection",
        "",
        "## Verdict",
        "",
        "**PRE_OOS_CANDIDATE_FROZEN.** The exact state router passed the strict pre-2024 contract.",
        "2024+ rows were not opened.",
        "",
        "## Frozen state/action rule",
        "",
        "- Base event: corrected live-parity confirmed pullback squeeze.",
        "- Support: 48-hour time exit.",
        "- Neutral: 24-hour time exit.",
        "- Adverse absorption: 48-hour cap with 4% take, no stop.",
        "- Weak states: completed-bar Wasserstein flow-response strain, causal-cone rupture, and dual price/flow clock.",
        "- Entry: next 5-minute open; 6bp/notional/side plus realized funding.",
        "- Strict MDD: global/pre-entry HWM plus favorable-before-adverse position envelope.",
        "",
        "## Frozen performance",
        "",
        "| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in PRE2024_WINDOWS:
        value = stats[name]
        lines.append(
            f"| {name} | {value['absolute_return_pct']:+.2f}% | {value['cagr_pct']:+.2f}% | "
            f"{value['strict_mdd_pct']:.2f}% | {value['cagr_to_strict_mdd']:.2f} | {value['trades']} |"
        )
    lines += [
        "",
        "## Search accounting",
        "",
        f"- Structured cells: {payload['selection_grid_cells']:,}",
        f"- Strict qualifiers: {payload['selection_qualifiers']}",
        "- Two top cells had identical executed schedules; deterministic tie-breaking chose the higher 2.0 clock-dominance threshold.",
        f"- Frozen manifest hash: `{payload['freeze_hash']}`",
        "",
        "The programme has inspected related pullback families on later years. The exact router is frozen before its own OOS replay, but that replay is contamination-aware rather than a pristine programme-level holdout.",
    ]
    return "\n".join(lines) + "\n"


def _oos_docs(payload: dict[str, Any]) -> str:
    stats = payload["oos_stats"]
    verdict = "ACCEPTED_OOS" if payload["oos_passed"] else "REJECTED_OOS"
    lines = [
        "# Pullback weak-state exit router — frozen OOS replay",
        "",
        f"## Verdict: **{verdict}**",
        "",
        "| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in FUTURE_WINDOWS:
        value = stats[name]
        lines.append(
            f"| {name} | {value['absolute_return_pct']:+.2f}% | {value['cagr_pct']:+.2f}% | "
            f"{value['strict_mdd_pct']:.2f}% | {value['cagr_to_strict_mdd']:.2f} | {value['trades']} |"
        )
    lines += [
        "",
        f"Manifest `{payload['manifest_freeze_hash']}` was reconstructed on the pre-2024 prefix before the OOS-open sidecar was written.",
        "No OOS threshold, action, hold, or sizing was retuned.",
    ]
    return "\n".join(lines) + "\n"


def _selection(cfg: Config) -> dict[str, Any]:
    context = _load_context(cfg, SELECTION_END)
    engine = ExecutionEngine(context["market"], context["funding"], _execution_config(cfg, cfg.leverage))
    cache: dict[tuple[int, str], Trade | None] = {}
    rows: list[dict[str, Any]] = []
    for spec in candidate_specs():
        labels = labels_for_spec(context["bank"], spec)
        schedules, stats = schedules_and_stats(
            engine,
            cfg,
            context["active"],
            labels,
            spec,
            windows=PRE2024_WINDOWS,
            trade_cache=cache,
        )
        rows.append(
            {
                "spec": spec,
                "state_counts": {
                    "support": int((context["active"] & (labels > 0)).sum()),
                    "neutral": int((context["active"] & (labels == 0)).sum()),
                    "adverse": int((context["active"] & (labels < 0)).sum()),
                },
                "selection_passed": bool(selection_passes(stats)),
                "rank": _selection_rank(stats, spec),
                "stats": stats,
                "schedule_hashes": {
                    name: _schedule_hash(schedules[name]) for name in PRE2024_WINDOWS
                },
            }
        )
    rows.sort(key=lambda row: tuple(row["rank"]), reverse=True)
    champion = rows[0]
    if champion["spec"] != FROZEN_SPEC:
        raise RuntimeError(f"frozen champion changed: {champion['spec']}")
    if not champion["selection_passed"]:
        raise RuntimeError("frozen champion no longer passes selection")

    labels = labels_for_spec(context["bank"], FROZEN_SPEC)
    grid_hash = _grid_hash(rows)
    manifest: dict[str, Any] = {
        "phase": "pre_oos_frozen",
        "oos_opened": False,
        "selection_end": SELECTION_END,
        "spec": FROZEN_SPEC,
        "spec_hash": _spec_hash(),
        "implementation_hash": _implementation_hash(),
        "frozen_execution_config": _frozen_execution_config(cfg),
        "source_prefix_hashes": context["source_hashes"],
        "base_activation_hash": _activation_hash(context["active"], context["dates"]),
        "weak_feature_hash": _weak_frame_hash(context["bank"], labels, FROZEN_SPEC),
        "state_label_hash": _array_hash(labels),
        "weak_thresholds": context["bank"]["thresholds"],
        "flow_tail": context["bank"]["flow_tail"],
        "selection_passed": True,
        "selection_stats": champion["stats"],
        "selection_schedule_hashes": champion["schedule_hashes"],
        "selection_grid_hash": grid_hash,
        "selection_grid_cells": len(rows),
    }
    manifest["freeze_hash"] = _freeze_hash(manifest)
    _write_manifest_once(Path(cfg.manifest_output), manifest, cfg)

    payload = {
        **manifest,
        "base_thresholds": context["base_thresholds"],
        "selection_qualifiers": int(sum(row["selection_passed"] for row in rows)),
        "selection_grid": rows,
    }
    _atomic_write_json(Path(cfg.output), payload)
    docs_path = Path(cfg.docs_output)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(_selection_docs(payload), encoding="utf-8")
    return payload


def _assert_reconstructed(manifest: dict[str, Any], context: dict[str, Any], cfg: Config) -> None:
    if context["source_hashes"] != manifest["source_prefix_hashes"]:
        raise RuntimeError("pre-2024 source prefix changed")
    if _activation_hash(context["active"], context["dates"]) != manifest["base_activation_hash"]:
        raise RuntimeError("base activation prefix changed")
    if context["bank"]["thresholds"] != manifest["weak_thresholds"]:
        raise RuntimeError("weak thresholds changed")
    if context["bank"]["flow_tail"] != manifest["flow_tail"]:
        raise RuntimeError("flow-tail threshold changed")
    labels = labels_for_spec(context["bank"], FROZEN_SPEC)
    if _array_hash(labels) != manifest["state_label_hash"]:
        raise RuntimeError("state labels changed")
    if _weak_frame_hash(context["bank"], labels, FROZEN_SPEC) != manifest["weak_feature_hash"]:
        raise RuntimeError("weak feature prefix changed")
    engine = ExecutionEngine(context["market"], context["funding"], _execution_config(cfg, cfg.leverage))
    schedules, stats = schedules_and_stats(
        engine,
        cfg,
        context["active"],
        labels,
        FROZEN_SPEC,
        windows=PRE2024_WINDOWS,
    )
    hashes = {name: _schedule_hash(schedules[name]) for name in PRE2024_WINDOWS}
    if hashes != manifest["selection_schedule_hashes"] or stats != manifest["selection_stats"]:
        raise RuntimeError("selection replay changed")


def _oos(cfg: Config) -> dict[str, Any]:
    manifest_path = Path(cfg.manifest_output)
    if not manifest_path.exists():
        raise RuntimeError("pre-OOS manifest is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(cfg, manifest)

    prefix = _load_context(cfg, SELECTION_END)
    _assert_reconstructed(manifest, prefix, cfg)
    sidecar = _mark_oos_opened(manifest_path, manifest, cfg.output)

    context = _load_context(cfg, cfg.exclude_from)
    if context["bank"]["thresholds"] != manifest["weak_thresholds"]:
        raise RuntimeError("full replay did not reproduce frozen weak thresholds")
    labels = labels_for_spec(context["bank"], FROZEN_SPEC)
    engine = ExecutionEngine(context["market"], context["funding"], _execution_config(cfg, cfg.leverage))
    schedules, stats = schedules_and_stats(
        engine,
        cfg,
        context["active"],
        labels,
        FROZEN_SPEC,
        windows=FUTURE_WINDOWS,
    )
    payload = {
        "phase": "oos_replay",
        "oos_opened": True,
        "opened_at": sidecar["opened_at"],
        "manifest_freeze_hash": manifest["freeze_hash"],
        "spec": FROZEN_SPEC,
        "oos_passed": bool(oos_passes(stats)),
        "oos_stats": stats,
        "oos_schedule_hashes": {name: _schedule_hash(schedules[name]) for name in FUTURE_WINDOWS},
    }
    _atomic_write_json(Path(cfg.output), payload)
    docs_path = Path(cfg.docs_output)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(_oos_docs(payload), encoding="utf-8")
    return payload


def run(cfg: Config) -> dict[str, Any]:
    return _oos(cfg) if cfg.open_oos else _selection(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=Config.input_csv)
    parser.add_argument("--funding-csv", default=Config.funding_csv)
    parser.add_argument("--premium-csv", default=Config.premium_csv)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--manifest-output", default=Config.manifest_output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    parser.add_argument("--exclude-from", default=Config.exclude_from)
    parser.add_argument("--open-oos", action="store_true")
    args = parser.parse_args()
    payload = run(Config(**vars(args)))
    key = "oos_stats" if args.open_oos else "selection_stats"
    print(json.dumps({"passed": payload.get("oos_passed", payload.get("selection_passed")), key: payload[key]}, indent=2))


if __name__ == "__main__":
    main()
