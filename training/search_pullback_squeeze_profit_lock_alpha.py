"""Freeze and evaluate a causal pullback-squeeze profit-lock candidate.

The corrected live-parity audit found that the confirmed pullback-squeeze's
largest strict drawdown could be created by a profitable trade: the strict
metric conservatively places the complete-position favorable envelope before
the adverse envelope.  This experiment therefore changes the exit, not the
entry signal.  It searches only fixed-hold, take-profit-only exits on physically
truncated pre-2024 data.  No hard stop is allowed.

Selection and future evaluation are deliberately separate.  ``--open-oos``
requires an existing pre-2024 manifest and evaluates only its frozen champion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.audit_confirmed_pullback_squeeze_live_parity import (
    AuditConfig,
    _activation_hash,
    _evaluate,
    _fit_active,
    _load_bundle,
    decision_mask,
    live_decision_features,
)


SELECTION_END = "2024-01-01"
HOLD_GRID = (288, 432, 576, 720)
TAKE_PROFIT_GRID_BPS = (200, 300, 400, 500, 600, 800, 1_000, 1_200, 1_500, 2_000, 1_000_000)
NO_STOP_BPS = 1_000_000
EXPECTED_FROZEN_MANIFEST_HASH = "ca64871a079b5e7fe558417d6870bed63e0e39a9392137bb9392a9f8825c6325"
MANIFEST_PAYLOAD_FIELDS = (
    "selection_end",
    "source_hashes",
    "activation_hash",
    "thresholds",
    "spec",
    "selection_score",
    "passes_pre_oos_gate",
    "frozen_protocol",
)

SELECTION_WINDOWS: dict[str, tuple[str | pd.Timestamp, str | pd.Timestamp]] = {
    "train": ("2020-07-01", "2023-01-01"),
    "train_2020h2": ("2020-07-01", "2021-01-01"),
    "train_2021": ("2021-01-01", "2022-01-01"),
    "train_2022": ("2022-01-01", "2023-01-01"),
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023_h1": ("2023-01-01", "2023-07-01"),
    "select_2023_h2": ("2023-07-01", "2024-01-01"),
    "pre_2024": ("2020-07-01", "2024-01-01"),
}

FUTURE_WINDOWS: dict[str, tuple[str | pd.Timestamp, str | pd.Timestamp]] = {
    "test_2024": ("2024-01-01", "2025-01-01"),
    "eval_2025": ("2025-01-01", "2026-01-01"),
    "holdout_2026": ("2026-01-01", "2026-06-02"),
    "oos_2024_2026": ("2024-01-01", "2026-06-02"),
}


@dataclass(frozen=True)
class Config(AuditConfig):
    output: str = "results/pullback_squeeze_profit_lock_selection_2026-07-15.json"
    manifest_output: str = "results/pullback_squeeze_profit_lock_manifest_2026-07-15.json"
    open_oos: bool = False


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


def _slim_all(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: _slim(value) for name, value in stats.items()}


def _spec_name(hold_bars: int, take_bps: int) -> str:
    if int(take_bps) >= NO_STOP_BPS:
        return f"hold_{int(hold_bars)}_time_only"
    return f"hold_{int(hold_bars)}_tp_{int(take_bps)}bps_no_stop"


def _selection_score(stats: dict[str, dict[str, Any]]) -> tuple[float, ...]:
    stability = ("train_2020h2", "train_2021", "train_2022", "select_2023_h1", "select_2023_h2")
    primary = ("train", "select_2023", "pre_2024")
    positive_segments = sum(stats[name]["absolute_return_pct"] > 0.0 for name in stability)
    ratios = [float(stats[name]["cagr_to_strict_mdd"]) for name in primary]
    return (
        float(positive_segments),
        float(min(ratios)),
        float(np.median(ratios)),
        float(stats["pre_2024"]["cagr_to_strict_mdd"]),
        float(stats["pre_2024"]["trades"]),
    )


def _passes_pre_oos_gate(stats: dict[str, dict[str, Any]]) -> bool:
    stable = all(
        stats[name]["absolute_return_pct"] > 0.0
        for name in ("train_2020h2", "train_2021", "train_2022", "select_2023_h1", "select_2023_h2")
    )
    support = (
        stats["train"]["trades"] >= 60
        and stats["select_2023"]["trades"] >= 12
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 5
    )
    target = (
        stats["train"]["cagr_to_strict_mdd"] >= 3.0
        and stats["select_2023"]["cagr_to_strict_mdd"] >= 3.0
        and stats["pre_2024"]["cagr_to_strict_mdd"] >= 2.5
    )
    risk = max(stats[name]["strict_mdd_pct"] for name in ("train", "select_2023", "pre_2024")) <= 15.0
    return bool(stable and support and target and risk)


def _shift_signal(active: np.ndarray, bars: int) -> np.ndarray:
    delay = max(0, int(bars))
    values = np.asarray(active, dtype=bool)
    if delay == 0:
        return values.copy()
    shifted = np.zeros_like(values)
    shifted[delay:] = values[:-delay]
    return shifted


def _selection_grid(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    active: np.ndarray,
    cfg: Config,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hold_bars in HOLD_GRID:
        for take_bps in TAKE_PROFIT_GRID_BPS:
            stats = _evaluate(
                market,
                funding,
                active,
                cfg,
                leverage=cfg.leverage,
                windows=SELECTION_WINDOWS,
                hold_bars=hold_bars,
                take_bps=take_bps,
                stop_bps=NO_STOP_BPS,
            )
            rows.append(
                {
                    "spec": {
                        "name": _spec_name(hold_bars, take_bps),
                        "hold_bars": int(hold_bars),
                        "hold_hours": float(hold_bars * 5.0 / 60.0),
                        "take_bps": int(take_bps),
                        "stop_bps": NO_STOP_BPS,
                    },
                    "selection_score": list(_selection_score(stats)),
                    "passes_pre_oos_gate": _passes_pre_oos_gate(stats),
                    "stats": _slim_all(stats),
                }
            )
    return sorted(
        rows,
        key=lambda row: (row["passes_pre_oos_gate"], *row["selection_score"]),
        reverse=True,
    )


def _manifest_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frozen_protocol(cfg: Config) -> dict[str, Any]:
    return {
        "window_size": int(cfg.window_size),
        "leverage": float(cfg.leverage),
        "fee_rate": float(cfg.fee_rate),
        "slippage_rate": float(cfg.slippage_rate),
        "funding_tolerance": str(cfg.funding_tolerance),
        "live_premium_tolerance": str(cfg.live_premium_tolerance),
        "future_cutoff": str(cfg.exclude_from),
        "decision_clock": "live_hour_signal_bar",
        "market_feature_lag_bars": 1,
        "entry_delay_bars": 1,
        "realized_funding": True,
        "strict_mdd": "global_hwm_plus_position_favorable_then_adverse_envelope",
    }


def _validate_manifest(
    manifest: dict[str, Any],
    cfg: Config,
    *,
    expected_hash: str = EXPECTED_FROZEN_MANIFEST_HASH,
) -> None:
    if manifest.get("phase") != "pre_oos_frozen" or not manifest.get("passes_pre_oos_gate"):
        raise RuntimeError("manifest is not an admitted pre-OOS candidate")
    try:
        frozen_payload = {key: manifest[key] for key in MANIFEST_PAYLOAD_FIELDS}
    except KeyError as exc:
        raise RuntimeError(f"manifest is missing frozen field: {exc.args[0]}") from exc
    actual_hash = str(manifest.get("manifest_hash", ""))
    if _manifest_hash(frozen_payload) != actual_hash:
        raise RuntimeError("manifest hash does not match frozen payload")
    if actual_hash != str(expected_hash):
        raise RuntimeError("manifest hash differs from the externally pinned OOS hash")
    if manifest["frozen_protocol"] != _frozen_protocol(cfg):
        raise RuntimeError("runtime protocol differs from frozen pre-OOS protocol")
    spec = manifest["spec"]
    if int(spec.get("stop_bps", -1)) != NO_STOP_BPS:
        raise RuntimeError("frozen spec is not a take-profit-only/no-stop candidate")
    if int(spec.get("hold_bars", -1)) not in HOLD_GRID:
        raise RuntimeError("frozen hold is outside the inspected selection grid")
    if int(spec.get("take_bps", -1)) not in TAKE_PROFIT_GRID_BPS:
        raise RuntimeError("frozen take-profit is outside the inspected selection grid")
    if spec.get("name") != _spec_name(int(spec["hold_bars"]), int(spec["take_bps"])):
        raise RuntimeError("frozen spec name does not match its execution parameters")


def _write_manifest_once(path: Path, manifest: dict[str, Any], cfg: Config) -> dict[str, Any]:
    _validate_manifest(manifest, cfg)
    if path.exists():
        existing = json.loads(path.read_text())
        _validate_manifest(existing, cfg)
        if existing != manifest:
            # ``created_at`` is metadata and is deliberately not part of the
            # frozen payload. Preserve the first declaration when only it differs.
            comparable_existing = {key: value for key, value in existing.items() if key != "created_at"}
            comparable_new = {key: value for key, value in manifest.items() if key != "created_at"}
            if comparable_existing != comparable_new:
                raise RuntimeError("refusing to overwrite an existing frozen manifest")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _selection_run(cfg: Config) -> dict[str, Any]:
    market, features, funding, source_hashes = _load_bundle(
        cfg,
        cutoff=SELECTION_END,
        premium_tolerance=cfg.live_premium_tolerance,
    )
    dates = pd.to_datetime(market["date"])
    decisions = decision_mask(dates, "live_hour_signal_bar", window_size=cfg.window_size)
    model_features = live_decision_features(features)
    active, thresholds = _fit_active(model_features, dates, decisions)
    rows = _selection_grid(market, funding, active, cfg)
    champion = rows[0]

    cost_10bp_cfg = replace(cfg, fee_rate=0.0005, slippage_rate=0.0005)
    stress = {
        "cost_10bp_side": _slim_all(
            _evaluate(
                market,
                funding,
                active,
                cost_10bp_cfg,
                leverage=cfg.leverage,
                windows=SELECTION_WINDOWS,
                hold_bars=champion["spec"]["hold_bars"],
                take_bps=champion["spec"]["take_bps"],
                stop_bps=NO_STOP_BPS,
            )
        ),
        "entry_lag_3_bars": _slim_all(
            _evaluate(
                market,
                funding,
                _shift_signal(active, 2),
                cfg,
                leverage=cfg.leverage,
                windows=SELECTION_WINDOWS,
                hold_bars=champion["spec"]["hold_bars"],
                take_bps=champion["spec"]["take_bps"],
                stop_bps=NO_STOP_BPS,
            )
        ),
    }
    frozen_payload = {
        "selection_end": SELECTION_END,
        "source_hashes": source_hashes,
        "activation_hash": _activation_hash(active, dates),
        "thresholds": thresholds,
        "spec": champion["spec"],
        "selection_score": champion["selection_score"],
        "passes_pre_oos_gate": champion["passes_pre_oos_gate"],
        "frozen_protocol": _frozen_protocol(cfg),
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "pre_oos_frozen",
        "manifest_hash": _manifest_hash(frozen_payload),
        **frozen_payload,
    }
    manifest_path = Path(cfg.manifest_output)
    manifest = _write_manifest_once(manifest_path, manifest, cfg)

    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "verdict": "PRE_OOS_CANDIDATE_FROZEN" if champion["passes_pre_oos_gate"] else "REJECTED_PRE_OOS",
        "config": asdict(cfg),
        "protocol": {
            "selection_source_cutoff": SELECTION_END,
            "future_opened": False,
            "market_clock": (
                "HH:00 decision; market-derived features through HH:00 shifted one 5m bar; boundary aux current"
            ),
            "entry": "next 5m open (HH:05)",
            "cost_per_side": cfg.fee_rate + cfg.slippage_rate,
            "realized_funding": True,
            "strict_mdd": "global/pre-entry HWM plus position-wide favorable envelope before adverse envelope",
            "same_bar_exit_order": "stop before take; selected family has no finite stop",
            "tested_exit_family": "fixed hold x take-profit-only; no hard stop",
            "tested_candidates": len(rows),
            "inspected_take_profit_bps": list(TAKE_PROFIT_GRID_BPS),
        },
        "input": {
            "rows": int(len(market)),
            "start": str(dates.min()),
            "end": str(dates.max()),
            "hourly_decisions": int(decisions.sum()),
            "active_signals": int(active.sum()),
        },
        "manifest": str(manifest_path),
        "manifest_hash": manifest["manifest_hash"],
        "thresholds": thresholds,
        "champion": champion,
        "stress": stress,
        "top": rows[:12],
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _passes_oos_gate(stats: dict[str, dict[str, Any]]) -> bool:
    return bool(
        all(stats[name]["absolute_return_pct"] > 0.0 for name in ("test_2024", "eval_2025"))
        and all(stats[name]["cagr_to_strict_mdd"] >= 3.0 for name in ("test_2024", "eval_2025"))
        and all(stats[name]["strict_mdd_pct"] <= 15.0 for name in ("test_2024", "eval_2025", "oos_2024_2026"))
        and all(stats[name]["trades"] >= 12 for name in ("test_2024", "eval_2025"))
        and stats["oos_2024_2026"]["cagr_to_strict_mdd"] >= 3.0
    )


def _oos_run(cfg: Config) -> dict[str, Any]:
    manifest_path = Path(cfg.manifest_output)
    if not manifest_path.exists():
        raise FileNotFoundError("pre-OOS manifest is required before --open-oos")
    manifest = json.loads(manifest_path.read_text())
    _validate_manifest(manifest, cfg)

    selection_market, selection_features, _, selection_hashes = _load_bundle(
        cfg,
        cutoff=SELECTION_END,
        premium_tolerance=cfg.live_premium_tolerance,
    )
    selection_dates = pd.to_datetime(selection_market["date"])
    selection_decisions = decision_mask(selection_dates, "live_hour_signal_bar", window_size=cfg.window_size)
    selection_active, selection_thresholds = _fit_active(
        live_decision_features(selection_features),
        selection_dates,
        selection_decisions,
    )
    if selection_hashes != manifest["source_hashes"]:
        raise RuntimeError("selection source hashes changed after manifest freeze")
    if selection_thresholds != manifest["thresholds"]:
        raise RuntimeError("selection thresholds changed after manifest freeze")
    if _activation_hash(selection_active, selection_dates) != manifest["activation_hash"]:
        raise RuntimeError("selection activation changed after manifest freeze")

    market, features, funding, _ = _load_bundle(
        cfg,
        cutoff=cfg.exclude_from,
        premium_tolerance=cfg.live_premium_tolerance,
    )
    dates = pd.to_datetime(market["date"])
    decisions = decision_mask(dates, "live_hour_signal_bar", window_size=cfg.window_size)
    active, thresholds = _fit_active(live_decision_features(features), dates, decisions)
    if not dates.iloc[: len(selection_dates)].reset_index(drop=True).equals(selection_dates.reset_index(drop=True)):
        raise RuntimeError("pre-2024 timestamp prefix changed after OOS was admitted")
    if not np.array_equal(active[: len(selection_active)], selection_active):
        raise RuntimeError("pre-2024 activation prefix changed after OOS was admitted")
    if thresholds != manifest["thresholds"]:
        raise RuntimeError("full-history threshold fit differs from frozen pre-2024 fit")

    spec = manifest["spec"]
    stop_bps = int(spec["stop_bps"])
    stats = _evaluate(
        market,
        funding,
        active,
        cfg,
        leverage=cfg.leverage,
        windows=FUTURE_WINDOWS,
        hold_bars=int(spec["hold_bars"]),
        take_bps=int(spec["take_bps"]),
        stop_bps=stop_bps,
    )
    cost_10bp_cfg = replace(cfg, fee_rate=0.0005, slippage_rate=0.0005)
    cost_stress = _evaluate(
        market,
        funding,
        active,
        cost_10bp_cfg,
        leverage=cfg.leverage,
        windows=FUTURE_WINDOWS,
        hold_bars=int(spec["hold_bars"]),
        take_bps=int(spec["take_bps"]),
        stop_bps=stop_bps,
    )
    admitted = _passes_oos_gate(stats)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "verdict": "ALPHA_QUALIFIED" if admitted else "REJECTED_OOS",
        "config": asdict(cfg),
        "manifest": str(manifest_path),
        "manifest_hash": manifest["manifest_hash"],
        "frozen_spec": spec,
        "protocol": {
            "only_frozen_candidate_evaluated": True,
            "future_windows": FUTURE_WINDOWS,
            "cost_per_side": cfg.fee_rate + cfg.slippage_rate,
            "realized_funding": True,
            "strict_mdd": "global/pre-entry HWM plus position-wide favorable envelope before adverse envelope",
        },
        "passes_oos_gate": admitted,
        "stats": _slim_all(stats),
        "cost_10bp_side": _slim_all(cost_stress),
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def run(cfg: Config) -> dict[str, Any]:
    return _oos_run(cfg) if cfg.open_oos else _selection_run(cfg)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=Config.input_csv)
    parser.add_argument("--funding-csv", default=Config.funding_csv)
    parser.add_argument("--premium-csv", default=Config.premium_csv)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--manifest-output", default=Config.manifest_output)
    parser.add_argument("--exclude-from", default=Config.exclude_from)
    parser.add_argument("--window-size", type=int, default=Config.window_size)
    parser.add_argument("--leverage", type=float, default=Config.leverage)
    parser.add_argument("--fee-rate", type=float, default=Config.fee_rate)
    parser.add_argument("--slippage-rate", type=float, default=Config.slippage_rate)
    parser.add_argument("--funding-tolerance", default=Config.funding_tolerance)
    parser.add_argument("--live-premium-tolerance", default=Config.live_premium_tolerance)
    parser.add_argument("--open-oos", action="store_true")
    return Config(**vars(parser.parse_args()))


def main() -> None:
    report = run(parse_args())
    summary = report.get("stats", report.get("champion", {}).get("stats", {}))
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "manifest_hash": report["manifest_hash"],
                "spec": report.get("frozen_spec", report.get("champion", {}).get("spec")),
                "stats": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
